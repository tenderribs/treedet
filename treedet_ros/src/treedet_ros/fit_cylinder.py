import numpy as np
from treedet_ros.icp import icp


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


def fit_cylinder(kpts: np.ndarray, pcl: np.ndarray, P: np.ndarray):
    """
    Fit a cylinder to the lidar pointclouds of trees
    kpts: 2D image plane coordinates of tree keypoints
    pcl: lidar pointcloud
    P: camera projection matrix

    returns

    https://www.stereolabs.com/docs/positional-tracking/coordinate-frames
    """
    assert P.shape[0] == 3 and P.shape[1] == 4, "Camera matrix is misshapen"

    fx = P[0, 0]
    fy = P[1, 1]
    cx = P[0, 2]
    cy = P[1, 2]

    def ray_vec(kpt: np.ndarray):
        w = np.array([(kpt[0] - cx) / fx, (kpt[1] - cy) / fy, 1])
        return w / np.linalg.norm(w)

    def ray(vec: np.ndarray):
        assert vec.shape[0] == 3
        meters = 20
        return np.array(
            [
                vec[0] * np.linspace(0, meters, meters * 10),
                vec[1] * np.linspace(0, meters, meters * 10),
                vec[2] * np.linspace(0, meters, meters * 10),
            ]
        ).T

    def estimate_3d(pcl, ray_vec):
        pcl = np.array(  # sort points based on distance to ray vector
            sorted(
                list(pcl),
                key=lambda p: np.linalg.norm(np.cross(p, ray_vec)),
            )
        )

        # start off with inital guess of where the kpt is:
        closest = pcl[:4, :]
        return np.mean(closest, axis=0)  # return the centroid

    w_fc = ray_vec(kpts[0:2])
    w_l = ray_vec(kpts[3:5])
    w_r = ray_vec(kpts[6:8])
    w_ax1 = ray_vec(kpts[9:11])
    w_ax2 = ray_vec(kpts[12:14])

    # the projection ray from the camera matrix:
    ray_fc = ray(w_fc)
    ray_l = ray(w_l)
    ray_r = ray(w_r)
    ray_ax2 = ray(w_ax2)

    pcl = pcl[pcl[:, 2] >= 2]  # reject too close points

    # calculate the width of the tree based on initial estimate
    radius = (
        np.sqrt(np.sum((estimate_3d(pcl, ray_l) - estimate_3d(pcl, ray_r)) ** 2)) / 2
    )

    # calculate height
    height = np.sqrt(
        np.sum((estimate_3d(pcl, ray_fc) - estimate_3d(pcl, ray_ax2)) ** 2)
    )

    # calculate the 3D rotation matrix to rotate (0, -1, 0) to to tree orientation
    init_vec = np.array([0, -1, 0])

    fc3d = estimate_3d(pcl, w_fc)

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
    init = np.array(
        [
            [1, 0, 0, fc3d[0]],
            [0, 1, 0, fc3d[1]],
            [0, 0, 1, fc3d[2]],
            [0, 0, 0, 1],
        ]
    )
    init[:3, :3] = R  # insert rotation
    init_guess = np.column_stack([cylinder, np.ones(cylinder.shape[0])])
    init_guess = (init @ init_guess.T).T

    T, _, iters = icp(cylinder, pcl, init_pose=None, tolerance=1e-4)
    cylinder_tf = np.column_stack([cylinder, np.ones(cylinder.shape[0])])
    cylinder_tf = (T @ cylinder_tf.T).T

    return height, radius, T
