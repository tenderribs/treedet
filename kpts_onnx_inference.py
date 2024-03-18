#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os

import cv2
import numpy as np
import time

import onnxruntime

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default='test_image.png',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='demo_output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.8,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="384,672",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save results to a text file.",
    )

    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()

    input_shape = tuple(map(int, args.input_shape.split(',')))
    origin_img = cv2.imread(args.image_path)
    img, ratio = preprocess(origin_img, input_shape)

    session = onnxruntime.InferenceSession(args.model)

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)

    # each row in dets contains the following in order:
    # reg_output (bbox), obj_output, cls_output, kpts_output
    dets = output[0]
    dets[:, :4] /= ratio # rescale the bbox
    dets[:, 6::3] /= ratio # rescale x of kpts
    dets[:, 7::3] /= ratio # rescale y of kpts

    sig = lambda x : 1/(1 + np.exp(-x))
    dets[:, 8::3] = sig(dets[:, 8::3]) # convert logit to prob

    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                         conf=args.score_thr, class_names=("tree"))

        for det in dets:
            if det[4] < args.score_thr: # only accept detections with objectness above the threshold
                continue

            # plot the x and y keypoints with sufficient confidence score
            for x, y, conf, label in zip(det[6::3], det[7::3], det[8::3], ["kpC", "kpL", "kpL", "ax1", "ax2"]):
                if (conf > args.score_thr):
                    print(f"{label}\t\tx: {x}\ty:{y}\tkptconf:\t{conf}")
                    cv2.circle(origin_img, (int(x), int(y)), radius=2, color=(255, 0, 0), thickness=-1)

    mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, args.image_path.split("/")[-1])
    cv2.imwrite(output_path, origin_img)