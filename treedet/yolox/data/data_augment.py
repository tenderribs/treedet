#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import math
import random

import cv2
import numpy as np

from yolox.utils import xyxy2cxcywh


def augment_hsv(img, hgain=5, sgain=30, vgain=30):
    hsv_augs = np.random.uniform(-1, 1, 3) * [
        hgain,
        sgain,
        vgain,
    ]  # random gains
    hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

    cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def get_aug_params(value, center=0):
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
                          or single float values. Got {}".format(
                value
            )
        )


def get_affine_matrix(
    target_size,
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
    camera_matrix=None,
):
    twidth, theight = target_size

    # Rotation and Scale
    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # Translation
    translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(translate) * theight  # y translation (pixels)

    M[0, 2] += translation_x
    M[1, 2] += translation_y

    return M, scale, angle


def apply_affine_to_bboxes(targets, target_size, M, scale):
    num_gts = len(targets)

    # warp corner points
    twidth, theight = target_size
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4 * num_gts, 2
    )  # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine transform
    corner_points = corner_points.reshape(num_gts, 8)

    # create new boxes
    corner_xs = corner_points[:, 0::2]
    corner_ys = corner_points[:, 1::2]
    new_bboxes = (
        np.concatenate(
            (
                corner_xs.min(1),
                corner_ys.min(1),
                corner_xs.max(1),
                corner_ys.max(1),
            )
        )
        .reshape(4, num_gts)
        .T
    )

    # clip boxes
    new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
    new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)

    targets[:, :4] = new_bboxes

    return targets


def apply_affine_to_kpts(targets, target_size, M, scale, num_kpts=17):
    num_gts = len(targets)
    # warp corner points
    twidth, theight = target_size
    xy_kpts = np.ones((num_gts * num_kpts, 3))
    xy_kpts[:, :2] = targets[:, 5:].reshape(num_gts * num_kpts, 2)  # num_kpt is hardcoded to 17
    xy_kpts = xy_kpts @ M.T  # transform
    xy_kpts = xy_kpts[:, :2].reshape(num_gts, num_kpts * 2)  # perspective rescale or affine
    xy_kpts[targets[:, 5:] == 0] = 0
    x_kpts = xy_kpts[:, list(range(0, num_kpts * 2, 2))]
    y_kpts = xy_kpts[:, list(range(1, num_kpts * 2, 2))]

    x_kpts[np.logical_or.reduce((x_kpts < 0, x_kpts > twidth, y_kpts < 0, y_kpts > theight))] = 0
    y_kpts[np.logical_or.reduce((x_kpts < 0, x_kpts > twidth, y_kpts < 0, y_kpts > theight))] = 0
    xy_kpts[:, list(range(0, num_kpts * 2, 2))] = x_kpts
    xy_kpts[:, list(range(1, num_kpts * 2, 2))] = y_kpts

    targets[:, 5:] = xy_kpts

    return targets


def random_affine(
    img,
    targets=(),
    target_size=(640, 640),
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
    num_kpts=17,
):
    M, scale, angle = get_affine_matrix(
        (target_size[1], target_size[0]), degrees, translate, scales, shear
    )

    img = cv2.warpAffine(
        img,
        M,
        dsize=(target_size[1], target_size[0]),
        borderValue=(114, 114, 114),
    )

    # Transform label coordinates
    if len(targets) > 0:
        targets = apply_affine_to_bboxes(targets, (target_size[1], target_size[0]), M, scale)
        targets = apply_affine_to_kpts(targets, target_size, M, scale, num_kpts=num_kpts)

    return img, targets


def _mirror(image, boxes, prob=0.5, human_kpts=None, flip_index=None):
    _, width, _ = image.shape
    if random.random() < prob:
        # flip image vertical (invert cols order)
        image = image[:, ::-1]

        # invert bbox cx and w w.r.t width
        boxes[:, 0::2] = width - boxes[:, 2::-2]
        human_kpts[:, 0::2] = (width - human_kpts[:, 0::2]) * (human_kpts[:, 0::2] != 0)

        # semantically flip, i.e. left and right labelled need flipping
        human_kpts[:, 0::2] = human_kpts[:, 0::2][:, flip_index]
        human_kpts[:, 1::2] = human_kpts[:, 1::2][:, flip_index]

    return image, boxes, human_kpts


def preproc(img, input_size, swap=(2, 0, 1)):
    # initialize padded image with "neutral" color 114
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)

    # place the resized image in top left corner of padded_img
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    # change from (height, width, channels) to (channels, height, width)
    padded_img = padded_img.transpose(swap)

    # ensure mem is contig. for performance reasons
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class TrainTransform:
    def __init__(
        self,
        max_labels=50,
        flip_prob=0.5,
        hsv_prob=1.0,
        flip_index=None,
        num_kpts=5,
    ):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.flip_index = flip_index
        self.num_kpts = num_kpts
        self.target_size = 5 + 2 * self.num_kpts  # 5+ 2*17

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        human_kpts = targets[:, 5:].copy()

        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, self.target_size), dtype=np.float32)
            image, r_o = preproc(image, input_dim)
            return image, targets

        # create copies to manipulate
        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        human_kpts_o = targets_o[:, 5:]

        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        if random.random() < self.hsv_prob:
            augment_hsv(image)

        image_t, boxes, human_kpts = _mirror(
            image,
            boxes,
            self.flip_prob,
            human_kpts=human_kpts,
            flip_index=self.flip_index,
        )

        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_
        human_kpts *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]
        human_kpts_t = human_kpts[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o
            human_kpts_t = human_kpts_o
            human_kpts_t *= r_o

        labels_t = np.expand_dims(labels_t, 1)

        # ensure labels and targets are in correct dimensions for concatenation
        targets_t = np.hstack((labels_t, boxes_t, human_kpts_t))

        # concat labels, boxes, kpts
        padded_labels = np.zeros((self.max_labels, self.target_size))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[: self.max_labels]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels


class ValTransform:
    def __init__(self, swap=(2, 0, 1)):
        self.swap = swap

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.swap)
        return img, np.zeros((1, 5))
