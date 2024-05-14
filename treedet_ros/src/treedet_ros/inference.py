import cv2
import numpy as np
import onnxruntime
import os
import ros_numpy
import rospy
import rospkg
import threading
import tf2_ros
import tf2_sensor_msgs
import time

from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from typing import Union
# from visualization_msgs.msg import MarkerArray

from treedet_ros.cutting_data import get_cutting_data
from treedet_ros.src.treedet_ros.sort_tracker import Sort
from treedet_ros.rviz import point_markers, np_to_markers, view_trackers

RATE_LIMIT = 5.0  # process incoming images at given frequency

br = CvBridge()

detection_pub = rospy.Publisher("/tree_det/felling_cut", PointCloud2, queue_size=10)
# marker_pub = rospy.Publisher("/tree_det/markers", MarkerArray, queue_size=10)


def preprocess_rgb(img: np.ndarray, input_size: tuple, swap=(2, 0, 1)):
    assert len(img.shape) == 3

    # initialize padded image with "neutral" color 114
    padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)

    # place the resized image in top left corner of padded_img
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    # change from (height, width, channels) to (channels, height, width)
    padded_img = padded_img.transpose(swap)

    # ensure mem is contig. for performance reasons
    return np.ascontiguousarray(padded_img, dtype=np.float32), r


def np_to_pcd2(
    XYZ: np.ndarray,
) -> PointCloud2:
    header = rospy.Header(
        frame_id="zed2i_left_camera_optical_frame", stamp=rospy.Time.now()
    )
    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
    ]
    points = [(xyz[0], xyz[1], xyz[2]) for xyz in XYZ]

    return point_cloud2.create_cloud(header, fields, points)


def get_detections(raw_dets: list, rescale_ratio: float):
    """
    Get filtered detections in scaling of original rgb and depth image
    """
    # filter uncertain bad detections
    raw_dets = raw_dets[raw_dets[:, 4] >= 0.9]

    # rescale bbox and kpts w.r.t original image
    raw_dets[:, :4] /= rescale_ratio
    raw_dets[:, 6::3] /= rescale_ratio
    raw_dets[:, 7::3] /= rescale_ratio

    # kpts in format "kpC", "kpL", "kpL", "ax1", "ax2" and each kpt gets (x, y, conf)
    return raw_dets[:, :4], raw_dets[:, 4], raw_dets[:, 6:]


def pc2_to_np(pcl: PointCloud2):
    pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(pcl)
    xyz_array = np.vstack((pc_array["x"], pc_array["y"], pc_array["z"])).transpose()
    return xyz_array


class PointCloudTransformer:
    def __init__(self):
        # Initialize the tf2 buffer and listener once
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

    def tf(self, cloud_in: PointCloud2, from_frame: str, to_frame: str) -> PointCloud2:
        # Wait for the transform to be available
        try:
            transform = self.tf_buffer.lookup_transform(
                to_frame, from_frame, rospy.Time(0), rospy.Duration(1.0)
            )

            return tf2_sensor_msgs.do_transform_cloud(cloud_in, transform)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logerr("Error transforming point cloud: %s" % str(e))
            return None





class TreeDetector:
    def __init__(self):
        package_path = rospkg.RosPack().get_path("treedet_ros")
        model_path = os.path.join(package_path, "model.onnx")
        self.session = onnxruntime.InferenceSession(model_path)
        print("Loaded Model")

        self.data_buffer = ([], [])  # RGB, Depth

        self.tree_index = {}

        self.tree_tracker = Sort(
            max_age=2,
            min_hits=3,
        )

        self.lock = threading.Lock() # lock prevents simulataneous R/W to the buffer

        self.rgb_subscriber = rospy.Subscriber(
            "/zed2i/zed_node/rgb/image_rect_color/compressed",
            CompressedImage,
            self.rgb_callback,
        )

        # TODO: should set to self_filtered topic once available from RSL
        self.lidar_subscriber = rospy.Subscriber(
            "/hesai/pandar",
            PointCloud2,
            self.lidar_callback,
        )

        self.pcl_transformer: PointCloudTransformer = PointCloudTransformer()

        # Timer to process messages at a desired frequency (e.g., 1 Hz)
        self.timer = rospy.Timer(rospy.Duration(1 / RATE_LIMIT), self.timer_callback)

    def rgb_callback(self, comp_image: CompressedImage) -> None:
        with self.lock:
            self.data_buffer[0].append(comp_image)

    def lidar_callback(self, pcd: PointCloud2) -> None:
        with self.lock:
            self.data_buffer[1].append(pcd)

    def timer_callback(self, event) -> None:
        with self.lock:
            if self.data_buffer[0] and self.data_buffer[1]:
                # Process the last message received
                rgb_msg: CompressedImage = self.data_buffer[0][-1]
                rgb_img: np.ndarray = br.compressed_imgmsg_to_cv2(rgb_msg)
                rgb_img = rgb_img[:, :, :3]  # cut out the alpha channel (bgra8 -> bgr8)

                lidar_pcl: PointCloud2 = self.data_buffer[1][-1]

                lidar_pcl = self.pcl_transformer.tf(
                    lidar_pcl, "PandarQT", "zed2i_left_camera_optical_frame"
                )

                if lidar_pcl:
                    lidar_pcl: np.ndarray = pc2_to_np, view_trackers(lidar_pcl)
                    self.process(rgb_img, lidar_pcl)

                self.data_buffer = ([], [])

    def find_overlapping_bbox_if_exists(bbox: np.ndarray, bboxes: np.ndarray, iou_thresh: float = 0.3) -> Union[int, None]:



    def update_tree_index(
        self, cut_xyzs: np.ndarray, cut_boxes: np.ndarray, tracking_ids: np.ndarray
    ) -> None:
        """Update the tree index with new cutting data"""

        # atm only will do based on tracking ID. later on will also associate based on 3D bbox if necessary
        # like if the tree bboxes overlap (2D birds eye view) then consider trees same
        tree_data = np.hstack((cut_xyzs, cut_boxes))
        for data, tracking_id in zip(tree_data, tracking_ids):
            id = tracking_id[0]
            if id in self.tree_index:
                self.tree_index[id].append(data)
            else:
                self.tree_index[id] = [data]

    def fetch_trees(self):
        """Read the tree index and prepare an average of values"""
        for tree_data in self.tree_index.values():
            if len(tree_data):
                tree_data = np.vstack(tree_data)
                tree_data = np.mean(tree_data, axis=0)

    def process(self, rgb_img: np.ndarray, pcl: np.ndarray):
        # pass image through model
        start = time.perf_counter()
        img, ratio = preprocess_rgb(rgb_img, (384, 672))
        ort_inputs = {self.session.get_inputs()[0].name: img[None, :, :, :]}
        output = self.session.run(None, ort_inputs)

        bboxes, confs, kpts = get_detections(output[0], ratio)
        print(f"get_detections:  \t{round((time.perf_counter() - start) * 1000, 1)} ms")

        # pass the detections through object tracker across frames. Tells us which trees are visible.
        # trackers: [:, bbox x1 y1 x2 y2, tracking_id]
        trackers, det_to_id_map = self.tree_tracker.update(dets=bboxes)

        # ensure kpts in same order as the tracked bboxes
        tracked_kpts = np.zeros((trackers.shape[0], kpts.shape[1]))
        for t_kpts, tracker in zip(tracked_kpts, trackers):
            tracking_id = tracker[4]
            det_id = det_to_id_map[tracking_id]
            t_kpts[:] = kpts[det_id, :]

        # view_trackers(trackers, tracked_kpts, rgb_img)

        # extract cutting info from the tracked trees
        start = time.perf_counter()
        cut_xyzs, cut_boxes, tracking_ids = get_cutting_data(
            trackers[:, :4], tracked_kpts, trackers[:, 4], pcl, fit_cylinder=False
        )

        assert (
            cut_xyzs.shape[0] == cut_boxes.shape[0]
            and cut_boxes.shape[0] == tracking_ids.shape[0]
        )

        self.update_tree_index(
            cut_xyz=cut_xyzs, cut_box=cut_boxes, tracking_ids=tracking_ids
        )

        print(f"get_cutting_data:\t{round((time.perf_counter() - start) * 1000, 1)} ms")

        return
        # transform the cutting point coordinates into map frame
        out_pcd = np_to_pcd2(cut_xyz)
        out_pcd = self.pcl_transformer.tf(
            out_pcd, "zed2i_left_camera_optical_frame", "map"
        )
        detection_pub.publish(out_pcd)
        cut_xyz_map_np = pc2_to_np(out_pcd)

        # # these markers look wonky in z-axis because pivot point of cylinder marker is at center. tree hence appear to go underground
        # marker_pub.publish(np_to_markers(pc2_to_np(out_pcd), cut_diam, "map"))


def main():
    rospy.init_node("treedet_inference", anonymous=True)
    rospy.set_param("/use_sim_time", True)

    TreeDetector()
    rospy.spin()
