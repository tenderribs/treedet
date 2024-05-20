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

from visualization_msgs.msg import MarkerArray

from treedet_ros.cutting_data import get_cutting_data
from treedet_ros.sort_tracker import Sort

# from treedet_ros.bbox import tree_data_to_bbox
from treedet_ros.rviz import np_to_markers

RATE_LIMIT = 5.0  # process incoming images at given frequency

br = CvBridge()

detection_pub = rospy.Publisher("/tree_det/felling_cut", PointCloud2, queue_size=10)
marker_pub = rospy.Publisher("/tree_det/markers", MarkerArray, queue_size=10)


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


def np_to_pcd2(XYZ: np.ndarray, frame: str) -> PointCloud2:
    header = rospy.Header(frame_id=frame, stamp=rospy.Time.now())
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
        self.frame_count = 0
        self.max_age = 2

        self.tree_tracker = Sort(
            max_age=self.max_age,
        )

        self.lock = threading.Lock()  # lock prevents simulataneous R/W to the buffer

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
                    lidar_pcl: np.ndarray = pc2_to_np(lidar_pcl)
                    self.process(rgb_img, lidar_pcl)

                self.data_buffer = ([], [])

    def update_tree_index(
        self, cut_xyzs: np.ndarray, cut_boxes: np.ndarray, tracking_ids: list
    ) -> None:
        """Update the tree index with new cutting data"""
        assert cut_xyzs.shape[0] == cut_boxes.shape[0]

        # remove old trees
        to_del = []
        self.frame_count += 1
        for tracking_id, data in self.tree_index.items():
            if self.frame_count - self.tree_index[tracking_id][-1][6] > RATE_LIMIT:
                to_del.append(tracking_id)

        for tracking_id in to_del:
            del self.tree_index[tracking_id]

        # add the new trees
        if cut_xyzs.shape[0] > 0:
            # remember the time that we found the tree
            frame_tag = np.full((cut_xyzs.shape[0], 1), self.frame_count)
            new_tree_data = np.hstack((cut_xyzs, cut_boxes, frame_tag))

            for new_d, tracking_id in zip(new_tree_data, tracking_ids):
                if tracking_id in self.tree_index:
                    self.tree_index[tracking_id].append(new_d)
                else:
                    self.tree_index[tracking_id] = [new_d]

    def fetch_tree_data(self):
        """Read the tree index and prepare an average of values"""
        # ret_tracking_ids = []
        ret_tree_data = []

        for t_id, tree_data in self.tree_index.items():
            # compute the mean of existing values
            tree_data = np.vstack(tree_data)
            ret_tree_data.append(np.mean(tree_data, axis=0))
            # ret_tracking_ids.append(t_id)

        if len(ret_tree_data) > 0:
            return np.vstack(ret_tree_data)  # , ret_tracking_ids
        return np.empty((0, 7))  # , []

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

        # extract cutting info from the tracked trees
        start = time.perf_counter()
        cut_xyzs, cut_boxes, tracking_ids = get_cutting_data(
            trackers[:, :4], tracked_kpts, trackers[:, 4], pcl, fit_cylinder=False
        )

        assert cut_xyzs.shape[0] == cut_boxes.shape[0] and cut_boxes.shape[0] == len(
            tracking_ids
        )

        map_cut_pcd = self.pcl_transformer.tf(
            np_to_pcd2(cut_xyzs, "zed2i_left_camera_optical_frame"),
            "zed2i_left_camera_optical_frame",
            "map",
        )

        self.update_tree_index(
            cut_xyzs=pc2_to_np(map_cut_pcd),
            cut_boxes=cut_boxes,
            tracking_ids=tracking_ids,
        )

        map_tree_data = self.fetch_tree_data()

        print(
            f"extract_cutting_data:\t{round((time.perf_counter() - start) * 1000, 1)} ms"
        )

        # transform the cutting point coordinates in map frame
        # detection_pub.publish(np_to_pcd2(XYZ=map_tree_data[:, :3], frame="map"))
        marker_pub.publish(
            np_to_markers(map_tree_data[:, :3], map_tree_data[:, 3:6], "map")
        )


def main():
    rospy.init_node("treedet_inference", anonymous=True)

    TreeDetector()
    rospy.spin()
