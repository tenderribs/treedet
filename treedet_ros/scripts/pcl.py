import numpy as np


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


def apply_hom_tf(xyz, src: str, dest: str):
    assert xyz.shape[0] and xyz.shape[1] == 3

    # calculated this matrix using scripts/map_to_map_o3d
    T = np.array(
        [
            [9.13790585e-01, -4.06126417e-01, -6.93543378e-03, -4.83355110e-01],
            [4.06185371e-01, 9.13676105e-01, 1.44713363e-02, -5.62073317e00],
            [4.59548175e-04, -1.60408426e-02, 9.99871232e-01, -2.87429848e-01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )
    # transform map_o3d to map
    if src == "map_o3d" and dest == "map":
        pass
    elif src == "map" and dest == "map_o3d":
        T = np.linalg.inv(T)
    else:
        assert False, "unsupported combo of src and dest"

    xyz = np.hstack((xyz, np.ones((xyz.shape[0], 1))))
    xyz = (T @ xyz.T).T
    return xyz[:, :3]
