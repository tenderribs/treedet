import pandas as pd
import numpy as np
import rospy
import matplotlib.pyplot as plt

from treedet_ros.pcl import apply_hom_tf

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

# if distance between target and detection under this number, consider target "found"
FOUND_THRESHOLD = 1.0


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


def plot(
    dets: np.ndarray,
    targets: np.ndarray,
    min_distances: "list[float]",
    det_counts: "list[int]",
):
    plt.scatter(dets[:, 1], dets[:, 0], c="g", label="Detections")
    plt.scatter(targets[:, 1], targets[:, 0], c="r", label="Targets")

    for i, (dist, count, targ) in enumerate(zip(min_distances, det_counts, targets)):
        text: str = f"{round(dist, 2)}m, {count}" if count > 0 else f"{round(dist, 2)}m"
        plt.text(targ[1], targ[0] - 2, text, fontsize=12, ha="center")
        circle = plt.Circle(
            (targ[1], targ[0]), FOUND_THRESHOLD, color="b", fill=True, alpha=0.15
        )
        plt.gca().add_patch(circle)

    plt.xlabel("Y [m]")
    plt.ylabel("X [m]")
    plt.title("Distance between tree target and closest detection")
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
    print(f"accuracy:\t{(tp + tn) / (tp + fp + fn + tn)}")

    plot(dets, targets, min_distances, det_counts)


def main():
    dets = pd.read_csv("tree_detections.csv")
    targets = pd.read_csv("tree_targets.csv")

    # only consider the visible targets from camera's perspective
    targets = targets[targets["visible"] == True]

    do_eval(dets, targets)
    return


if __name__ == "__main__":
    main()
