import pandas as pd
import numpy as np
import rospy
import matplotlib.pyplot as plt


from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


def rpy_2_rot_matrix(yaw: float):
    yawMatrix = np.matrix(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ]
    )

    return yawMatrix


def create_marker(
    df,
    frame_id: str,
    color=(0, 1, 0),
):
    marker_array = MarkerArray()

    for idx, row in df.iterrows():
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "trees"
        marker.id = idx * 2
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        # Set the position
        marker.pose.position.x = row["pos_x"]
        marker.pose.position.y = row["pos_y"]
        marker.pose.position.z = row["pos_z"] + row["dim_z"] / 2

        # Set the orientation (quaternion, here just identity since it's a cylinder)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Set the scale of the cylinder
        marker.scale.x = row["dim_x"]  # diameter
        marker.scale.y = row["dim_x"]  # diameter
        marker.scale.z = row["dim_z"]  # height

        # Set the color (here just an example, RGBA)
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0
        marker_array.markers.append(marker)

    return marker_array


def do_eval(dets: np.ndarray, targets: np.ndarray):
    dets = dets[["pos_x", "pos_y", "pos_z"]].to_numpy()
    targets = targets[["pos_x", "pos_y", "pos_z"]].to_numpy()

    # transform map_o3d to map
    T = np.array(
        [
            [8.99265226e-01, -4.37336201e-01, 7.68766682e-03, -8.61728469e-01],
            [4.37391503e-01, 8.99232715e-01, -8.31845589e-03, -4.94360570e00],
            [-3.27503961e-03, 1.08430183e-02, 9.99935849e-01, 1.56276816e-01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )
    #
    targets = np.hstack((targets, np.ones((targets.shape[0], 1))))
    targets = (T @ targets.T).T
    targets = targets[:, :3]

    found = [False] * len(targets)

    min_distances = []
    for target in targets:
        target = np.full((dets.shape[0], 2), target[:2])
        distances = np.linalg.norm(target[:, :2] - dets[:, :2], axis=1)
        min_dist = np.min(distances)
        min_distances.append(min_dist)

    plt.scatter(dets[:, 1], dets[:, 0], c="g", label="Detections")
    plt.scatter(targets[:, 1], targets[:, 0], c="r", label="Targets")

    for i, (dist, targ) in enumerate(zip(min_distances, targets)):
        plt.text(targ[1], targ[0] - 2, f"{round(dist, 2)}m", fontsize=12, ha="center")

    plt.xlabel("Y [m]")
    plt.ylabel("X [m]")
    plt.title("Distance between tree target and closest detection")
    plt.gca().set_aspect("equal", adjustable="box")  # Set equal aspect ratio

    plt.legend()
    plt.show()


def main():

    dets = pd.read_csv("tree_detections.csv")
    targets = pd.read_csv("tree_targets.csv")
    targets = targets[targets["selected"] == True]

    # have to apply a static transformation before we do anything.
    do_eval(dets, targets)

    return
    rospy.init_node("view_detected_trees")

    pub1 = rospy.Publisher("/treedet_ros/viz_dets", MarkerArray, queue_size=10)
    pub2 = rospy.Publisher("/treedet_ros/viz_targs", MarkerArray, queue_size=10)

    rate = rospy.Rate(1)  # 1 Hz
    while not rospy.is_shutdown():
        # Load and publish markers
        pub1.publish(create_marker(dets, "map", (0, 1, 0)))
        pub2.publish(create_marker(targets, "map_o3d", (1, 0, 0)))

        rate.sleep()


if __name__ == "__main__":
    # rosrun pcl_ros pcd_to_pointcloud map_small.pcd 1
    main()
