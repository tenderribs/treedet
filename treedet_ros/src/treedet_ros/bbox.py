import numpy as np
from treedet_ros.cutting_data import xyz2uv


def tree_data_to_bbox(cam_cut_xyzs: np.ndarray, cut_boxes) -> np.ndarray:
    """
    return bbox coordinates of tree trunk IN THE CAMERA FRAME
    """
    assert cam_cut_xyzs.shape[1] == 3 and cut_boxes.shape[1] == 3

    print(f"cam_cut_xyzs:\n{cam_cut_xyzs}")
    print(f"cut_boxes:\n{cut_boxes}")

    # cylinder params: center, base, radius height
    xc = cam_cut_xyzs[:, 0]
    y0 = cam_cut_xyzs[:, 1]
    zc = cam_cut_xyzs[:, 2]
    r = cut_boxes[:, 0] / 2
    h = cut_boxes[:, 2]

    d = np.sqrt(xc**2 + zc**2)
    l1 = np.sqrt(d**2 - r**2)

    print(f"r:\t{r}")
    print(f"xc:\t{xc}")
    print(f"zc:\t{zc}")
    print(f"d:\t{d}")
    print(f"l1:\t{l1}")

    # xc == 0 and / or d == 0 imply the tree lies in origin, should be impossible
    # assert np.all(xc != 0, axis=0) and np.all(d != 0, axis=0)
    theta = np.arctan2(zc, xc)
    phi = np.arctan2(r, l1)

    x1 = l1 * np.cos(theta + phi)
    z1 = l1 * np.sin(theta + phi)
    x2 = l1 * np.cos(theta - phi)
    z2 = l1 * np.sin(theta - phi)

    print(f"x1:\t{x1}")
    print(f"z1:\t{z1}")
    print(f"x2:\t{x2}")
    print(f"z2:\t{z2}")

    u1, v1 = xyz2uv(x1, y0 - h, z1)  # top point
    u2, v2 = xyz2uv(x2, y0, z2)  # bottom right
    return np.vstack((u1, v1, u2, v2)).T
