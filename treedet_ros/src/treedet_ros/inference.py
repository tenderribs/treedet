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

from harveri_msgs.msg import HarveriDetectedTrees, HarveriDetectedTree
from treedet_ros.cutting_data import get_cutting_data
from treedet_ros.sort_tracker import Sort

RATE_LIMIT = 5.0
MAX_TREE_LATERAL_ERR: float = 0.4
DETECTION_CONF_THRESH: float = 0.95
TRACKER_MAX_AGE: int = 1
DET_RETENTION_S: int = 1
FIT_CYLINDER: bool = True

br = CvBridge()

felling_cut_pub = rospy.Publisher("/treedet/felling_cut", PointCloud2, queue_size=10)
detection_pub = rospy.Publisher(
    "/treedet/detected_trees", HarveriDetectedTrees, queue_size=10
)


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


def np_to_hvri_det_trees(
    xyz: np.ndarray, dims: np.ndarray, frame: str = "map"
) -> HarveriDetectedTrees:
    assert xyz.shape[1] == 3 and dims.shape[1] == 3

    tree_list = HarveriDetectedTrees()
    tree_list.header.frame_id = frame

    for i, (_xyz, _dims) in enumerate(zip(xyz, dims)):
        msg = HarveriDetectedTree()
        msg.id = i

        msg.x = _xyz[0]
        msg.y = _xyz[1]
        msg.z = _xyz[2]
        msg.dim_x = _dims[0]
        msg.dim_y = _dims[2]
        msg.dim_z = _dims[2]
        tree_list.trees.append(msg)
    return tree_list


def get_detections(raw_dets: list, rescale_ratio: float):
    """
    Get filtered detections in scaling of original rgb and depth image
    """
    # filter uncertain bad detections
    raw_dets = raw_dets[raw_dets[:, 4] >= DETECTION_CONF_THRESH]

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
        self.max_age = TRACKER_MAX_AGE

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

    def find_existing(
        self, new_d: np.ndarray, existing_trees: np.ndarray, existing_t_ids: list
    ):
        if existing_trees.shape[0] == 0:
            return None

        distances = np.linalg.norm(new_d[:2] - existing_trees[:, :2], axis=1)
        min_distance_index = np.argmin(distances)
        if distances[min_distance_index] < MAX_TREE_LATERAL_ERR:
            return existing_t_ids[min_distance_index]

        return None

    def update_tree_index(
        self, cut_xyzs: np.ndarray, cut_boxes: np.ndarray, tracking_ids: list
    ) -> None:
        """Update the tree index with new cutting data"""
        assert cut_xyzs.shape[0] == cut_boxes.shape[0]

        self.frame_count += 1

        # remove old trees
        to_del = []
        for tracking_id, data in self.tree_index.items():
            # if the tree has not received any detections for longer than 4 seconds
            if (
                self.frame_count - self.tree_index[tracking_id][-1][6]
                > DET_RETENTION_S * RATE_LIMIT
            ):
                to_del.append(tracking_id)

        for tracking_id in to_del:
            del self.tree_index[tracking_id]

        # add new trees
        if cut_xyzs.shape[0] > 0:
            # remember the time that we found the tree
            frame_tag = np.full((cut_xyzs.shape[0], 1), self.frame_count)
            new_tree_data = np.hstack((cut_xyzs, cut_boxes, frame_tag))

            existing_trees, existing_t_ids = self.fetch_tree_data()

            for new_d, tracking_id in zip(new_tree_data, tracking_ids):
                # if tree is being tracked
                if tracking_id in self.tree_index:
                    self.tree_index[tracking_id].append(new_d)
                    continue

                # if unable to associate det. with existing tracker, try to see if close
                existing_t_id = self.find_existing(
                    new_d, existing_trees, existing_t_ids
                )

                # prefer to associate tree with existing ones to prevent duplicates
                if existing_t_id is not None:
                    tracking_id = existing_t_id
                else:
                    self.tree_index[tracking_id] = [new_d]

    def fetch_tree_data(self):
        """Read the tree index and prepare an average of values"""
        ret_tree_data = []

        for tree_data in self.tree_index.values():
            # compute the mean of existing values
            tree_data = np.vstack(tree_data)
            ret_tree_data.append(np.mean(tree_data, axis=0))

        if len(ret_tree_data) > 0:
            return np.vstack(ret_tree_data), list(self.tree_index.keys())
        return np.empty((0, 7)), []

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
            trackers[:, :4],
            tracked_kpts,
            trackers[:, 4],
            pcl,
            fit_cylinder=FIT_CYLINDER,
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

        tree_cutting_data, _ = self.fetch_tree_data()

        print(
            f"extract_cutting_data:\t{round((time.perf_counter() - start) * 1000, 1)} ms"
        )

        # transform the cutting point coordinates in map frame
        felling_cut_pub.publish(np_to_pcd2(XYZ=tree_cutting_data[:, :3], frame="map"))
        detection_pub.publish(
            np_to_hvri_det_trees(
                tree_cutting_data[:, :3], tree_cutting_data[:, 3:6], frame="map"
            )
        )


def main():
    rospy.init_node("treedet_inference", anonymous=True)

    TreeDetector()
    rospy.spin()
