import os
import pandas as pd
import numpy as np
import rospy
import matplotlib.pyplot as plt

from pcl import apply_hom_tf

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

# if distance between target and detection under this number, consider target "found"
FOUND_THRESHOLD = 1.0


def rotate(xyz: np.ndarray):
    rad = 7 / 6 * np.pi
    T = np.array(
        [
            [np.cos(rad), -np.sin(rad), 0],
            [np.sin(rad), np.cos(rad), 0],
            [0, 0, 1],
        ]
    )
    return (T @ xyz.T).T


def plot(
    dets: np.ndarray,
    targets: np.ndarray,
    min_distances: "list[float]",
    det_counts: "list[int]",
):
    # display the centerline of robot motion
    # view_frustums = np.load("view_frustums.npy")

    # diff = view_frustums[0::8, :] - view_frustums[1::8, :]
    # center_line = view_frustums[1::8, :] + diff / 2

    dets = rotate(dets)
    targets = rotate(targets)
    # center_line = rotate(center_line) view_frustums = rotate(view_frustums)

    dets = dets[dets[:, 1] <= 150]
    # center_line = center_line[center_line[:, 1] <= 150]

    plt.scatter(dets[:, 1], dets[:, 0], c="g", label="Detections")
    plt.scatter(targets[:, 1], targets[:, 0], c="r", label="Targets")
    # plt.scatter(center_line[::4, 1], center_line[::4, 0], c="b", label="Robot Odometry", s=0.3)

    for i, (dist, count, targ) in enumerate(zip(min_distances, det_counts, targets)):
        text: str = f"{round(dist, 2)}m"
        plt.text(targ[1], targ[0] - 2, text, fontsize=12, ha="center")
        circle = plt.Circle((targ[1], targ[0]), FOUND_THRESHOLD, color="b", fill=True, alpha=0.15)
        plt.gca().add_patch(circle)

    plt.xlabel("Y [m]")
    plt.ylabel("X [m]")
    plt.title("Target detection at x view frustum depth")
    plt.gca().set_aspect("equal", adjustable="box")  # Set equal aspect ratio

    plt.legend()
    plt.grid(True)
    plt.show()


def do_eval(dets: pd.DataFrame, targets: pd.DataFrame):
    # have to apply a static transformation before we do anything.
    dets = dets[["pos_x", "pos_y", "pos_z"]].to_numpy()
    targets = targets[["pos_x", "pos_y", "pos_z"]].to_numpy()

    targets = apply_hom_tf(targets, src="map_o3d", dest="map")

    min_distances = []
    det_counts = []
    for target in targets:
        target = np.full((dets.shape[0], 2), target[:2])
        distances = np.linalg.norm(target[:, :2] - dets[:, :2], axis=1)
        min_dist = np.min(distances)

        # save the closest distance
        min_distances.append(min_dist)

        # save number of detections that are under threshold
        det_counts.append(len([d for d in distances if d < FOUND_THRESHOLD]))

    # TP is number of targets that had a close enough detection
    tp = len([dist for dist in min_distances if dist < FOUND_THRESHOLD])
    fn = len(min_distances) - tp

    fp = dets.shape[0] - tp
    tn = 0  # cannot be calculated with available information.

    print(f"tp, fp:\t\t{tp}\t{fp}")
    print(f"fn, tn:\t\t{fn}\t{tn}")
    print()
    print(f"precision:\t{tp / (tp + fp)}")
    print(f"recall:\t\t{tp / (tp + fn)}")
    recall = tp / (tp + fn)
    print(f"accuracy:\t{(tp + tn) / (tp + fp + fn + tn)}")

    # plot(dets, targets, min_distances, det_counts)
    return recall * 100


def pub_markers(dets: pd.DataFrame, targets: pd.DataFrame):
    def create_marker(
        df,
        color=(0, 1, 0),
    ):
        marker_array = MarkerArray()

        for idx, row in df.iterrows():
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "trees"
            marker.id = idx * 2
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            # Set the position
            marker.pose.position.x = row["pos_x"]
            marker.pose.position.y = row["pos_y"]
            marker.pose.position.z = row["pos_z"]

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
            marker.color.a = 1
            marker_array.markers.append(marker)

        return marker_array

    rospy.init_node("eval_detected_trees")

    dets_pub = rospy.Publisher("/treedet_ros/viz_dets", MarkerArray, queue_size=10)
    targets_pub = rospy.Publisher("/treedet_ros/viz_targets", MarkerArray, queue_size=10)

    r = rospy.Rate(1)
    while not rospy.is_shutdown():
        dets_pub.publish(create_marker(dets, color=(0, 1, 0)))
        targets_pub.publish(create_marker(targets, color=(1, 0, 0)))
        r.sleep()


def main():
    dets = pd.read_csv("tree_detections.csv")

    target_distances = [
        int(file.split("map_o3d_targets_", 1)[1][:-4])
        for file in os.listdir()
        if "map_o3d_targets_" in file
    ]

    target_distances = sorted(target_distances)

    recalls = {dist: -1.0 for dist in target_distances}

    for dist in target_distances:
        t_df = pd.read_csv(f"map_o3d_targets_{dist}.csv")

        # only consider the visible targets from camera's perspective
        t_df = t_df[t_df["visible"] == True]

        recalls[dist] = do_eval(dets, t_df)

    plt.plot(list(recalls.keys()), list(recalls.values()), marker="o")
    plt.xlabel("Frustum Depth [m]")
    plt.ylabel("Targets detected [%]")
    plt.title("Target Detection Rate of Drive-By")
    plt.grid(True)
    plt.show()

    # dets_np = dets[["pos_x", "pos_y", "pos_z"]].to_numpy()
    # dets_np = apply_hom_tf(dets_np, "map", "map_o3d")
    # dets[["pos_x", "pos_y", "pos_z"]] = dets_np
    # pub_markers(dets, targets)

    return


if __name__ == "__main__":
    main()
