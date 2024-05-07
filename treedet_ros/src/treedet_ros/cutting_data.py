import rospy
import numpy as np

from treedet_ros.icp import icp
from scipy.spatial import Delaunay


# taken from P matrix published by /zed2i/zed_node/depth/camera_info
P = np.array(
    [
        [486.89678955078125, 0.0, 325.613525390625, 0.0],
        [0.0, 486.89678955078125, 189.09512329101562, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
)

fx = P[0, 0]
fy = P[1, 1]
cx = P[0, 2]
cy = P[1, 2]

# depth limits of the frustum
Z_MIN = 0.1
Z_MAX = 20

# currently assume robot_pose is constant


def uv2xyz(uv: np.ndarray, Z: float):
    """
    Conv pixel coords to world coords in zed2i frame with pinhole camera model

    """
    X = (uv[0] - cx) * Z / fx
    Y = (uv[1] - cy) * Z / fy
    return [X, Y, Z]


# def uv2incl(ax1: np.ndarray, fc: np.ndarray) -> np.ndarray:
#     """Get counter-clockwise inclination in radians of tree w.r.t the "upwards" vertical axis"""
#     du = fc[:, 0] - ax1[:, 0]
#     dv = fc[:, 1] - ax1[:, 1]
#     return -(np.pi / 2 + np.arctan2(du, dv))


def ray_vec(kpt: np.ndarray):
    """
    Get unit direction vec
    https://support.stereolabs.com/hc/en-us/articles/4554115218711-How-can-I-convert-3D-world-coordinates-to-2D-image-coordinates-and-viceversa
    """
    # fix Z = 1m -> (( u - c_x) * Z) / (f_x) simplifies to (u - c_x) / (f_x)
    w = np.array([(kpt[0] - cx) / fx, (kpt[1] - cy) / fy, 1])
    return w / np.linalg.norm(w)


def ray(vec: np.ndarray):
    """
    Convert the ray vector into actual points along the vector with 10cm resolution
    """
    assert vec.shape[0] == 3
    meters = 30  # determine how far away we look into distance

    return np.array(
        [
            vec[0] * np.linspace(0, meters, meters * 10),
            vec[1] * np.linspace(0, meters, meters * 10),
            vec[2] * np.linspace(0, meters, meters * 10),
        ]
    ).T


def estimate_3d(pcl: np.ndarray, ray_vec: np.ndarray):
    """
    Projects ray vector into 3d space and then computes average of the closest points to the ray
    param ray_vec: should be normalized!
    """
    # assert np.linalg.norm(ray_vec) == 1

    pcl = np.array(  # sort points based on distance to ray vector
        sorted(
            list(pcl),
            key=lambda p: np.linalg.norm(np.cross(p, ray_vec)),
        )
    )

    # start off with inital guess of where the kpt is. 3 small number because far away tree pcls don't have many points
    closest = pcl[:3, :]
    return np.mean(closest, axis=0)  # return the centroid


def project_2dto3d(kpts: np.ndarray, pcl: np.ndarray):
    """
    Fit a cylinder to the lidar pointclouds of trees
    kpts: 2D image plane coordinates of tree keypoints
    pcl: lidar pointcloud
    P: camera projection matrix

    https://www.stereolabs.com/docs/positional-tracking/coordinate-frames
    """

    w_fc = ray_vec(kpts[0:2])
    w_l = ray_vec(kpts[3:5])
    w_r = ray_vec(kpts[6:8])
    w_ax2 = ray_vec(kpts[12:14])

    # the projection ray from the camera matrix:
    ray_fc = ray(w_fc)
    ray_l = ray(w_l)
    ray_r = ray(w_r)
    ray_ax2 = ray(w_ax2)

    # calculate the width of the tree based on initial estimate
    radius = (
        np.sqrt(np.sum((estimate_3d(pcl, ray_l) - estimate_3d(pcl, ray_r)) ** 2)) / 2
    )

    # calculate height
    height = np.sqrt(
        np.sum((estimate_3d(pcl, ray_fc) - estimate_3d(pcl, ray_ax2)) ** 2)
    )

    fc3d = estimate_3d(pcl, w_fc)
    return (
        height,
        radius,
        fc3d,
    )


def create_cylinder(radius=0.3, height=4, num_pts=50, part=0.3):
    """
    create a cylinder-like shape
    part: fully extruded cylinder -> part = 1. But ex. only want half-circle -> set part = 0.5
    """
    # Getting a good fit works best when you have roughly num_pts same as pcl point count
    z = radius * np.sin(np.linspace(0, 2 * part * np.pi, 4))
    x = radius * np.cos(np.linspace(0, 2 * part * np.pi, 4))
    heights = np.linspace(0, -height, int(num_pts / 4))

    # assemble the cylinder by stacking rings at each height
    cylinder = np.array([]).reshape(-1, 3)
    for h in heights:
        circle = np.column_stack([x, h * np.ones_like(x), z])
        cylinder = np.vstack([cylinder, circle])
    return cylinder


def do_fit_cylinder(kpts: np.ndarray, pcl: np.ndarray):
    height, radius, fc3d = project_2dto3d(kpts, pcl)

    # calculate the 3D rotation matrix to rotate (0, -1, 0) to to tree orientation
    init_vec = np.array([0, -1, 0])

    w_ax1 = ray_vec(kpts[9:11])

    incl_vec = estimate_3d(pcl, w_ax1) - fc3d
    incl_vec /= np.linalg.norm(incl_vec)

    rot_axis = np.cross(init_vec, incl_vec)
    rot_axis /= np.linalg.norm(rot_axis)

    cos_theta = np.dot(init_vec, incl_vec)
    theta = np.arccos(cos_theta)

    # Rodrigues' rotation formula to find the rotation matrix
    K = np.array(
        [
            [0, -rot_axis[2], rot_axis[1]],
            [rot_axis[2], 0, -rot_axis[0]],
            [-rot_axis[1], rot_axis[0], 0],
        ]
    )

    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

    cylinder = create_cylinder(radius=radius, height=height, num_pts=pcl.shape[0])

    # ensure that equal number of points in pcl and cylinder
    if cylinder.shape[0] > pcl.shape[0]:
        cylinder = cylinder[: pcl.shape[0], :]

    if pcl.shape[0] > cylinder.shape[0]:
        # only keep the points closest to the cutting keypoint
        pcl = pcl[: cylinder.shape[0], :]

    # find initial guess for T as spot of felling cut with rotation based on the camera projection matrix:
    init_pose = np.array(
        [
            [1, 0, 0, fc3d[0]],
            [0, 1, 0, fc3d[1]],
            [0, 0, 1, fc3d[2]],
            [0, 0, 0, 1],
        ]
    )
    init_pose[:3, :3] = R  # insert rotation

    T, _, iters = icp(cylinder, pcl, init_pose=init_pose, tolerance=1e-3)
    cylinder_tf = np.column_stack([cylinder, np.ones(cylinder.shape[0])])
    cylinder_tf = (T @ cylinder_tf.T).T

    rospy.logdebug(f"ran icp in {iters} iters")

    return height, radius, T


def get_cutting_data(
    bboxes: np.ndarray, kpts: np.ndarray, pcl: np.ndarray, fit_cylinder: bool = True
):
    """
    param fit_cylinder: optionally try to fit cylinder to the lidar kpts with icp algorithm. is slower tho
    """

    assert bboxes.shape[0] == kpts.shape[0]  # sanity checks
    assert bboxes.shape[1] == 4 and kpts.shape[1] == 15 and pcl.shape[1] == 3

    # # get the inclination of the cut in degrees
    # cut_uv = kpts[:, 0:2]
    # ax1_uv = kpts[:, 9:11]
    # incl_radians = uv2incl(ax1=ax1_uv, fc=cut_uv)

    cut_xyzs, dim_xyzs = [], []

    # fit cylinder to each bbox
    for bbox, kpts in zip(bboxes, kpts):  # , incl , incl_radians
        frustum = np.array(  # calculate the frustum points for each bbox corner
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

        # filter pointcloud for each frustum (only search pcl within)
        hull = Delaunay(points=frustum)
        inside = pcl[hull.find_simplex(pcl) >= 0]

        inside = inside[inside[:, 2] >= 2]  # reject too close points

        # fit a cylinder to the points inside
        if fit_cylinder:
            height, tree_radius, T_matrix = do_fit_cylinder(kpts=kpts, pcl=inside)
            cut_xyz = T_matrix[:3, 3]  # translation part of T matrix
        else:
            height, tree_radius, cut_xyz = project_2dto3d(kpts=kpts, pcl=inside)

        # felling cut 3d coords and 3d bounding box
        dim_xyz = np.array([2 * tree_radius, 2 * tree_radius, height])

        cut_xyzs.append(cut_xyz)
        dim_xyzs.append(dim_xyz)

    return np.vstack(cut_xyzs), np.vstack(dim_xyzs)
