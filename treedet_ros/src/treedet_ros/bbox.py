import numpy as np
from typing import Union


def tree_data_to_bbox(tree_data: Union[np.ndarray, None]) -> np.ndarray:
    """return bbox coordinates of tree trunk in the map frame"""
    assert tree_data.shape[1] == 6

    # cylinder params: center, radius height
    xc = tree_data[:, 0]
    yc = tree_data[:, 1]
    r = tree_data[:, 3]
    h = tree_data[:, 5]

    d = np.sqrt(xc**2 + yc**2)
    l = np.sqrt(d**2 - r**2)

    theta = np.arctan2(yc, xc)
    phi = np.arcsin(r, d)

    x1 = l * np.cos(theta + phi)
    y1 = l * np.sin(theta + phi)
    x2 = l * np.cos(theta - phi)
    y2 = l * np.sin(theta - phi)

    return


def find_overlapping_tree_id(
    ex_bboxes: Union[np.ndarray, None],
    new_bbox: np.ndarray,
    tracking_ids: np.ndarray,
    iou_thresh: float = 0.3,
) -> Union[int, None]:
    if (ex_bboxes is None) or (ex_bboxes.shape[0] == 0):
        return None
    print("in find_overlapping")
    new_bbox = new_bbox[0]
    x1, y1, x2, y2 = new_bbox
    bboxes_x1 = ex_bboxes[:, 0]
    bboxes_y1 = ex_bboxes[:, 1]
    bboxes_x2 = ex_bboxes[:, 2]
    bboxes_y2 = ex_bboxes[:, 3]
    print(bboxes_x1)
    # coordinates of the intersection
    print(f"x1.shape: {x1.shape}")
    print(f"bbxox1.shape: {bboxes_x1.shape}")
    i_x1 = np.max(x1, bboxes_x1)
    i_y1 = np.max(y1, bboxes_y1)
    i_x2 = np.min(x2, bboxes_x2)
    i_y2 = np.min(y2, bboxes_y2)
    inter_area = np.max(0, i_x2 - i_x1) * np.max(0, i_y2 - i_y1)

    single_bbox_area = (x2 - x1) * (y2 - y1)
    all_area = (bboxes_x2 - bboxes_x1) * (bboxes_y2 - bboxes_y1)

    union_area = single_bbox_area + all_area - inter_area
    iou = inter_area / union_area
    print(iou)
    assert False