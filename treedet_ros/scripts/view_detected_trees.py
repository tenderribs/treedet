import pandas as pd
import rospy

import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


def create_marker(row, marker_id):
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "trees"
    marker.id = marker_id
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
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0

    return marker


def create_pointcloud2(cloud):
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "map"

    points = []
    for point in cloud:
        points.append([point[0], point[1], point[2]])

    point_cloud_msg = pc2.create_cloud_xyz32(header, points)
    return point_cloud_msg


def main():
    rospy.init_node("view_detected_trees")

    # Load and publish markers
    dets = pd.read_csv("tree_detections.csv")
    marker_pub = rospy.Publisher(
        "visualization_marker_array", MarkerArray, queue_size=10
    )
    marker_array = MarkerArray()

    for idx, row in dets.iterrows():
        marker = create_marker(row, idx)
        marker_array.markers.append(marker)

    rate = rospy.Rate(1)  # 1 Hz
    while not rospy.is_shutdown():
        marker_pub.publish(marker_array)
        rate.sleep()


if __name__ == "__main__":
    # rosrun pcl_ros pcd_to_pointcloud map_small.pcd 1
    main()
