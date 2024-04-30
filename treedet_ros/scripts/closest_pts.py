import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

w = np.array([0.5, 0.5, 1.0])
w = w

# gen random points on XY plane at z=1
rng = np.random.default_rng()
points = rng.uniform([0, 0, 1], [1, 1, 1], size=(1000, 3))

# the projection ray from the camera matrix:
line = np.array(
    [
        w[0] * np.linspace(0, 2, 200),
        w[1] * np.linspace(0, 2, 200),
        w[2] * np.linspace(0, 2, 200),
    ]
).T

print(f"line.shape: {line.shape}")
print(f"points.shape: {points.shape}")

# sort points based on distance
points = np.array(
    sorted(
        list(points),
        key=lambda p: np.linalg.norm(np.cross(p, w)) / np.linalg.norm(w),
    )
)

closest = points[:10, :]
rest = points[10:, :]

# get the distances:
distances = np.array([np.linalg.norm(np.cross(p, w)) / np.linalg.norm(w) for p in rest])

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.scatter(line[:, 0], line[:, 1], line[:, 2])
ax.scatter(rest[:, 0], rest[:, 1], rest[:, 2], c=distances, cmap="Greens_r")
ax.scatter(closest[:, 0], closest[:, 1], closest[:, 2], color="#d62728")
plt.show()
