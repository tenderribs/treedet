import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


from treedet_ros.icp import icp
from treedet_ros.pcl import pc2 as pcl, kp2 as kpts, set_axes_equal

# https://www.stereolabs.com/docs/positional-tracking/coordinate-frames

pcl = pcl[pcl[:, 2] >= 3]  # reject too close points
# pcl = pcl[pcl[:, 2] <= 6]  # reject too far points

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


def ray_vec(kpt: np.ndarray):
    w = np.array([(kpt[0] - cx) / fx, (kpt[1] - cy) / fy, 1])
    return w / np.linalg.norm(w)


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


def create_cylinder(radius=0.3, height=4, num_pts=50, part=0.3):
    """
    create a cylinder-like shape
    part: fully extruded cylinder -> part = 1. But ex. only want half-circle -> part = 0.5
    """
    print(f"radius: {radius}")
    print(f"height: {height}")
    print(f"num_pts: {num_pts}")

    rim_points = round(0.5 * np.sqrt(num_pts))
    print(f"rim_points: {rim_points}")

    phi = 2.0 * np.pi * part
    rot_offset = -0.5 * (phi + np.pi)  # add offset so that cylinder faces camera
    rim = np.linspace(0, phi, rim_points) + rot_offset
    z = radius * np.sin(rim)
    x = radius * np.cos(rim)

    heights = np.linspace(0, -height, int(num_pts / rim_points))

    # assemble the cylinder by stacking rings at each height
    cylinder = np.array([]).reshape(-1, 3)
    for h in heights:
        circle = np.column_stack([x, h * np.ones_like(x), z])
        cylinder = np.vstack([cylinder, circle])
    return cylinder


w_fc = ray_vec(kpts[0:2])
w_ax2 = ray_vec(kpts[12:14])
p_fc = estimate_3d(pcl, w_fc)
p_ax2 = estimate_3d(pcl, w_ax2)


def XY_from_uvZ(uv, Z):
    X = (uv[0] - cx) * Z / fx
    Y = (uv[1] - cy) * Z / fy
    return np.array([X, Y])


l_uv = kpts[3:5]
r_uv = kpts[6:8]


# calculate the left and right XY coordinates based on the felling cut depth
P_lp = XY_from_uvZ(l_uv, p_fc[2])
P_rp = XY_from_uvZ(r_uv, p_fc[2])

a = 0.5 * np.linalg.norm(P_lp - P_rp)  # half of separation distance
l1 = 0.5 * (np.linalg.norm(P_lp) + np.linalg.norm(P_rp))  # average of both points
d = np.linalg.norm(p_fc[:2])

radius = ((l1 + a) ** 2 - d**2) / (2 * d)
height = np.linalg.norm(p_ax2 - p_fc)

cylinder = create_cylinder(radius=radius, height=height, num_pts=pcl.shape[0], part=0.2)

# ensure that equal number of points in pcl and cylinder
if cylinder.shape[0] > pcl.shape[0]:
    cylinder = cylinder[: pcl.shape[0], :]

if pcl.shape[0] > cylinder.shape[0]:
    # only keep the points closest to the cutting keypoint
    pcl = pcl[: cylinder.shape[0], :]

# find initial guess for T as centroid of the pcl with rotation based on the camera projection matrix:
init = np.array(
    [
        [1, 0, 0, p_fc[0]],
        [0, 1, 0, p_fc[1]],  # move up a bit to prevent
        [0, 0, 1, p_fc[2]],
        [0, 0, 0, 1],
    ]
)

init_guess = np.column_stack([cylinder, np.ones(cylinder.shape[0])])
init_guess = (init @ init_guess.T).T


T, distances, iters = icp(cylinder, pcl, init_pose=init, tolerance=1e-2)
cylinder_tf = np.column_stack([cylinder, np.ones(cylinder.shape[0])])
cylinder_tf = (T @ cylinder_tf.T).T

coords = T[:3, 3]

pcl = pcl[pcl[:, 2] <= 6]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2], label="Filtered LiDAR pcl", alpha=0.1)
ax.scatter(
    cylinder_tf[:, 0], cylinder_tf[:, 1], cylinder_tf[:, 2], label="cylinder", alpha=0.1
)
ax.scatter(p_fc[0], p_fc[1], p_fc[2], label="p_fc")

set_axes_equal(ax)
ax.legend()
plt.show()
print(f"ran ICP in {iters} iters")
