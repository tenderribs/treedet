import tf.transformations
import cv2

from geometry_msgs.msg import Quaternion

from visualization_msgs.msg import Marker, MarkerArray


def point_markers(XYZ, frame_id="zed2i_left_camera_optical_frame"):
    marker_array = MarkerArray()
    for i, point in enumerate(XYZ):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.type = Marker.SPHERE
        marker.id = i * 2
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = point[2]
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        quat = tf.transformations.quaternion_from_euler(0, 0, 0)
        marker.pose.orientation = Quaternion(*quat)
        marker_array.markers.append(marker)
    return marker_array


def np_to_markers(XYZ, dims, frame_id):
    marker_array = MarkerArray()

    for i, (point, dim) in enumerate(zip(XYZ, dims)):
        m = Marker()
        m.header.frame_id = frame_id
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.id = i * 2
        m.scale.x = dim[0]
        m.scale.y = dim[1]
        m.scale.z = dim[2]
        m.pose.position.x = point[0]
        m.pose.position.y = point[1]
        m.pose.position.z = point[2] + dim[2] / 2
        m.color.a = 1.0
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        quat = tf.transformations.quaternion_from_euler(0, 0, 0)
        m.pose.orientation = Quaternion(*quat)

        marker_array.markers.append(m)
    return marker_array


def view_trackers(tracked, tracked_kpts, rgb_img):
    for t, kpts in zip(tracked, tracked_kpts):
        p1 = (int(t[0]), int(t[1]))
        p2 = (int(t[2]), int(t[3]))

        color = ((t[4] * 75) % 255, (t[4] * 50) % 255, (t[4] * 150) % 255)
        cv2.rectangle(rgb_img, p1, p2, color, thickness=2)
        cv2.circle(
            rgb_img, (int(kpts[0]), int(kpts[1])), radius=4, color=color, thickness=-1
        )

    cv2.imshow("Image with Bounding Boxes", rgb_img)
    cv2.waitKey(2)  # Wait for a key press to close
