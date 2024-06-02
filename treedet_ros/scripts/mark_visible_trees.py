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
# Z_MAX = 4.14  # selected based on harveri manual. maximum is 6m
Z_MAXES = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

RECORD_INTERVAL = 0.2  # record every x seconds


def is_topic_published(topic_name: str) -> bool:
    topics = rospy.get_published_topics()
    return any(topic_name in topic[0] for topic in topics)


def record_frustums():
    bbox = np.array([0, 0, 640, 360])  # viewport as full screen bbox

    cam_frustums = {
        Z_MAX: np_to_pcd2(
            np.array(
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
            ),
            frame="zed2i_left_camera_optical_frame",
        )
        for Z_MAX in Z_MAXES
    }

    map_frustums = {Z_MAX: [] for Z_MAX in Z_MAXES}

    transformer = PointCloudTransformer()

    print("run the rosbags pls...")

    while not is_topic_published("/tf") and not is_topic_published("/tf_static"):
        time.sleep(RECORD_INTERVAL)

    print("transforms are available, recording")

    while is_topic_published("/tf") and is_topic_published("/tf_static"):
        # transform the frustums from cam to the map frame
        for z, cam_frustum in cam_frustums.items():
            map_frustum: PointCloud2 = transformer.tf(
                cam_frustum, "zed2i_left_camera_optical_frame", "map"
            )
            map_frustums[z].append(pc2_to_np(map_frustum))

        time.sleep(RECORD_INTERVAL)

    for z, map_frustum_list in map_frustums.items():
        if len(map_frustum_list) == 0:
            print(f"No view frustums recorded for {z}")
            continue

        frustums = np.vstack(map_frustum_list)
        np.save(f"map_frustums_{z}", frustums)


def main():
    package_path = rospkg.RosPack().get_path("treedet_ros")
    base_path = os.path.join(package_path, "scripts")

    rospy.init_node("auto_select_trees")

    if "map_frustums" not in "".join(os.listdir(base_path)):
        record_frustums()
        return

    # find distances associated with the map frustum files
    zs = [
        int(file.split("map_frustums_", 1)[1][:-4])
        for file in os.listdir(base_path)
        if "map_frustums" in file
    ]

    zs = sorted(zs)

    # load the data and transform to map_o3d frame
    map_frustums = {z: np.load(base_path + f"/map_frustums_{z}.npy") for z in zs}
    map_o3d_frustums = {
        z: apply_hom_tf(m_frustums, src="map", dest="map_o3d")
        for z, m_frustums in map_frustums.items()
    }
    del map_frustums

    def mark_viewport_visbility(infile: str, outfile: str, _frustums):
        df = pd.read_csv(os.path.join(base_path, infile))

        # precompute the convex hulls of each frustum
        hulls = [Delaunay(points=_frustums[i : (8 + i), :]) for i in range(0, len(_frustums), 8)]

        # determine whether any target's centers are contained within the convex hulls
        visibility_mask = np.zeros(len(df), dtype=bool)
        centers = df[["pos_x", "pos_y", "pos_z"]].to_numpy()
        for hull in hulls:
            visibility_mask |= hull.find_simplex(centers) >= 0

        df["visible"] = False
        df.loc[visibility_mask, "visible"] = True
        df.to_csv(os.path.join(base_path, outfile), index=False)

        print(f"marked {len(df[df['visible'] == True])} targets as visible")

    # export CSV file of the visible files
    for z, map_o3d_frustum in map_o3d_frustums.items():
        mark_viewport_visbility("tree_targets.csv", f"map_o3d_targets_{z}.csv", map_o3d_frustum)


if __name__ == "__main__":
    main()
