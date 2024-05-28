"""For evaluation, determine which tree bboxes are even possible targets"""

import numpy as np
import pandas as pd
import time
import os
import rospy
import rospkg

from rostopic import get_topic_class, ROSTopicException
from sensor_msgs.msg import PointCloud2
from scipy.spatial import Delaunay

from treedet_ros.pcl import apply_hom_tf
from treedet_ros.inference import PointCloudTransformer, np_to_pcd2, pc2_to_np
from treedet_ros.cutting_data import uv2xyz


Z_MIN = 0.1
Z_MAX = 15

RECORD_INTERVAL = 0.2  # record every x seconds


def is_topic_published(topic_name: str) -> bool:
    topics = rospy.get_published_topics()
    return any(topic_name in topic[0] for topic in topics)


def record_frustums():
    bbox = np.array([0, 0, 640, 360])  # viewport as full screen bbox

    frustum = np.array(  # calculate the frustum points for each bbox corner in the camera space
        [
            uv2xyz(bbox[[0, 1]], Z_MIN),
            uv2xyz(bbox[[2, 1]], Z_MIN),
            uv2xyz(bbox[[0, 3]], Z_MIN),
            uv2xyz(bbox[[2, 3]], Z_MIN),
            uv2xyz(bbox[[0, 1]], Z_MAX),
            uv2xyz(bbox[[2, 1]], Z_MAX),
            uv2xyz(bbox[[0, 3]], Z_MAX),
            uv2xyz(bbox[[2, 3]], Z_MAX),
        ]
    )
    view_frustum_pc2 = np_to_pcd2(frustum, frame="zed2i_left_camera_optical_frame")

    transformer = PointCloudTransformer()

    print("run the rosbags pls...")

    while not is_topic_published("/tf") and not is_topic_published("/tf_static"):
        time.sleep(RECORD_INTERVAL)

    print("transforms are available, recording")

    view_frustums = []
    while is_topic_published("/tf") and is_topic_published("/tf_static"):
        # transform the frustum to the map frame
        _frustum: PointCloud2 = transformer.tf(
            view_frustum_pc2, "zed2i_left_camera_optical_frame", "map"
        )
        view_frustums.append(pc2_to_np(_frustum))

        time.sleep(RECORD_INTERVAL)

    if len(view_frustums) == 0:
        print("No view frustums recorded")
        return None

    print(f"recorded {len(view_frustums)} view_frustums")

    view_frustums = np.vstack(view_frustums)
    return view_frustums


def main():
    package_path = rospkg.RosPack().get_path("treedet_ros")
    base_path = os.path.join(package_path, "scripts")

    rospy.init_node("auto_select_trees")
    if os.path.isfile(base_path + "/view_frustums.npy"):
        view_frustums = np.load(base_path + "/view_frustums.npy")
    else:
        view_frustums = record_frustums()
        assert view_frustums is not None
        np.save("view_frustums", view_frustums)

    def mark_viewport_visbility(filename: str, _frustums):
        df = pd.read_csv(os.path.join(base_path, filename))

        # precompute the convex hulls of each frustum
        hulls = [Delaunay(points=_frustums[i : (8 + i), :]) for i in range(0, len(_frustums), 8)]

        # determine whether any target's centers are contained within the convex hulls
        visibility_mask = np.zeros(len(df), dtype=bool)
        centers = df[["pos_x", "pos_y", "pos_z"]].to_numpy()
        for hull in hulls:
            visibility_mask |= hull.find_simplex(centers) >= 0

        df.loc[visibility_mask, "visible"] = True
        df.to_csv(os.path.join(base_path, filename), index=False)

        print(f"marked {len(df[df['visible'] == True])} targets as visible")

    # transform the recorded frustums to the map_o3d frame
    view_frustums_o3d = apply_hom_tf(view_frustums, src="map", dest="map_o3d")

    mark_viewport_visbility("tree_targets.csv", view_frustums_o3d)


if __name__ == "__main__":
    main()
