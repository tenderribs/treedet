import numpy as np
import os
import open3d as o3d
import rospy
import ros_numpy
import tf2_ros
import time
from sensor_msgs.msg import PointCloud2
import tf2_sensor_msgs

from treedet_ros.icp import icp


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


class PointCloudExtractor:
    def __init__(self):
        rospy.init_node("pointcloud_extractor", anonymous=True)
        self.sub = rospy.Subscriber("/hesai/pandar", PointCloud2, self.lidar_callback)
        self.transformer = PointCloudTransformer()
        self.pc2s = []
        self.ready = False

    def lidar_callback(self, pandar_pc2: PointCloud2):
        # only process first five pointclouds
        if len(self.pc2s) >= 5:
            self.sub.unregister()
            self.ready = True
            print("received 5 pointclouds")
        else:
            pc2_map = self.transformer.tf(pandar_pc2, "PandarQT", "map")
            self.pc2s.append(pc2_map)

    def pc2_to_np(self, pcl: PointCloud2):
        pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(pcl)
        xyz_array = np.vstack((pc_array["x"], pc_array["y"], pc_array["z"])).transpose()
        return xyz_array

    def get_agg_pointcloud(self):
        print(f"returning {len(self.pc2s)} aggregated map pointclouds")

        if len(self.pc2s):
            res = [self.pc2_to_np(pc2) for pc2 in self.pc2s]
            return np.vstack(res)
        return np.empty((0, 3))


def plot(xyz_map, xyz_map_o3d):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    def set_axes_equal(ax):
        """
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc.

        Input
        ax: a matplotlib axis, e.g., as output from plt.gca().
        """

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.scatter(xyz_map[:, 0], xyz_map[:, 1], xyz_map[:, 2], label="xyz_map")
    ax.scatter(
        xyz_map_o3d[:, 0], xyz_map_o3d[:, 1], xyz_map_o3d[:, 2], label="xyz_map_o3d"
    )
    set_axes_equal(ax)
    ax.legend()
    plt.show()


def crop_z(xyz: np.ndarray):
    xyz = xyz[xyz[:, 2] > -1.5]
    xyz = xyz[xyz[:, 2] < 5]
    return xyz


def clamp(xyz: np.ndarray, target_size: int):
    if target_size < xyz.shape[0]:
        indices = np.random.choice(xyz.shape[0], target_size, replace=False)
        return xyz[indices]

    assert (
        xyz.shape[0] == target_size
    ), f"trying to subsample to a target size {target_size} larger than xyz.size {xyz.size}"
    return xyz


def subsample(xyz: np.ndarray, factor: float = 0.1):
    target_size = int(xyz.shape[0] * factor)
    indices = np.random.choice(xyz.shape[0], target_size, replace=False)
    return xyz[indices]


if __name__ == "__main__":
    # fetch the first lidar scans of rosbag, within the map frame
    if not os.path.isfile("xyz_map.npy"):
        pce = PointCloudExtractor()
        while not pce.ready:
            time.sleep(0.1)
        xyz_map: np.ndarray = pce.get_agg_pointcloud()
        np.save("xyz_map", xyz_map)
    else:
        xyz_map = np.load("xyz_map.npy")

    pcd_map_o3d = o3d.io.read_point_cloud("/datasets/maps/map_small.pcd")
    xyz_map_o3d = np.asarray(pcd_map_o3d.points)

    # # remember the original values
    # xyz_map_og = subsample(xyz_map, 0.1)
    # xyz_map_o3d_og = subsample(xyz_map_o3d, 0.01)

    # crop height
    xyz_map, xyz_map_o3d = crop_z(xyz_map), crop_z(xyz_map_o3d)

    # subsample to ensure equal number of points
    target_rows = min(xyz_map.shape[0], xyz_map_o3d.shape[0])

    xyz_map, xyz_map_o3d = clamp(xyz_map, target_rows), clamp(xyz_map_o3d, target_rows)

    # create an initial guess for the transformation with dyaw, dx, dy = 0.43 rad, -1m, -5.3m):
    T_init = np.eye(4, 4)
    T_init[:2, :2] = np.array(
        [[np.cos(0.43), -np.sin(0.43)], [np.sin(0.43), np.cos(0.43)]]
    )
    T_init[:3, 3] = [-1, -5.3, 0]

    def apply_hom_matrix(mat: np.ndarray, data):
        data = np.hstack((data, np.ones((data.shape[0], 1))))
        data = (mat @ data.T).T
        return data[:, :3]

    # get tf from map_o3d to map
    T, _, iters = icp(xyz_map_o3d, xyz_map, init_pose=T_init)
    print(f"ran ICP in {iters} iters")

    print(f"init guess:\n{T_init}")
    print(f"final matrx:\n{T}")

    xyz_map_o3d = apply_hom_matrix(T, xyz_map_o3d)
    plot(xyz_map, xyz_map_o3d)
