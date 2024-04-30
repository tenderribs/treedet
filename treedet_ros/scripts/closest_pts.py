import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


from icp import icp
from pcl import pcl, set_axes_equal

# https://www.stereolabs.com/docs/positional-tracking/coordinate-frames


pcl = pcl[pcl[:, 2] >= 3]  # reject too close points
pcl = pcl[pcl[:, 2] <= 6]  # and the ones that are too far away (better axis scaling)

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

kpt = np.array([122.42979, 283.6861])
w = np.array([(kpt[0] - cx) / fx, (kpt[1] - cy) / fy, 1])


def create_shape():
    # create a cylinder-like shape
    radius = 0.2
    height = 1.5
    part = 0.33  # fully extruded cylinder -> part = 1. But ex. only want half-circle -> part = 0.5

    x = radius * np.cos(np.linspace(0, part * 2 * np.pi, 30))
    z = radius * np.sin(np.linspace(0, part * 2 * np.pi, 30))
    heights = np.linspace(0, -height, 30)

    # assemble the cylinder by stacking rings at each height
    cylinder = np.array([]).reshape(-1, 3)
    for h in heights:
        circle = np.column_stack([x, h * np.ones_like(x), z])
        cylinder = np.vstack([cylinder, circle])
    return cylinder


shape = create_shape()

# the projection ray from the camera matrix:
ray = np.array(
    [
        w[0] * np.linspace(0, 5, 100),
        w[1] * np.linspace(0, 5, 100),
        w[2] * np.linspace(0, 5, 100),
    ]
).T

print(f"ray.shape: {ray.shape}")
print(f"pcl.shape: {pcl.shape}")

# sort points based on distance to ray vector
pcl = np.array(
    sorted(
        list(pcl),
        key=lambda p: np.linalg.norm(np.cross(p, w)) / np.linalg.norm(w),
    )
)

closest = pcl[:10, :]
rest = pcl[10:, :]

guess = np.mean(closest, axis=0)
print(guess)
# get the distances:
# distances = np.array([np.linalg.norm(np.cross(p, w)) / np.linalg.norm(w) for p in rest])

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.scatter(ray[:, 0], ray[:, 1], ray[:, 2], label="ray")
ax.scatter(shape[:, 0], shape[:, 1], shape[:, 2], label="shape")
ax.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2])
# ax.scatter(rest[:, 0], rest[:, 1], rest[:, 2], c=distances, cmap="Greens_r")
ax.scatter(guess[0], guess[1], guess[2], color="#d62728", marker="x", s=400)
set_axes_equal(ax)

ax.legend()
plt.show()
