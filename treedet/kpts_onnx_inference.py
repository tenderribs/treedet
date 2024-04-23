#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os

import cv2
import time

import onnxruntime

from yolox.data.data_augment import preproc as preprocess
from yolox.utils import mkdir


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
        "--images_path",
        type=str,
        default="test_image.png",
        help="Path to your input images directory.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="demo_output",
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


def visualize(img, det):
    # plot the bounding box
    p1 = (int(det[0]), int(det[1]))
    p2 = (int(det[2]), int(det[3]))
    conf = int(round(det[4] * 100, 0))

    # Draw a filled rectangle on the overlay with some opacity
    overlay = img.copy()
    cv2.rectangle(overlay, p1, p2, (255, 251, 43), -1)  # -1 fills the rectangle

    # Alpha determines the transparency of the overlay: 0 is fully transparent, 1 is fully opaque
    alpha = 0.4

    # Blend the overlay with the original image
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Draw the bounding box edges on the original image
    cv2.rectangle(img, p1, p2, (255, 251, 43), 2)

    # draw confidence score
    cv2.putText(
        img=img,
        text=f"{conf}%",
        org=(int(det[15]) - 10, int(det[16]) + 20),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(59, 40, 34),
        thickness=2,
    )

    # plot the x and y keypoints with sufficient confidence score
    for x, y, conf, label in zip(
        det[6::3],
        det[7::3],
        det[8::3],
        ["kpC", "kpL", "kpL", "ax1", "ax2"],
    ):
        cv2.circle(
            img,
            (int(x), int(y)),
            radius=4,
            color=(52, 64, 235),
            thickness=-1,
        )

    return img


if __name__ == "__main__":
    args = make_parser().parse_args()
    input_shape = tuple(map(int, args.input_shape.split(",")))

    image_files = [
        f
        for f in os.listdir(args.images_path)
        if os.path.isfile(os.path.join(args.images_path, f))
    ]
    image_files = [
        f for f in image_files if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]  # Filter for image files

    session = onnxruntime.InferenceSession(args.model)

    total_inference = 0

    for image_file in image_files:
        origin_img = cv2.imread(os.path.join(args.images_path, image_file))
        img, ratio = preprocess(origin_img, input_shape)

        start = time.perf_counter()
        ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
        output = session.run(None, ort_inputs)
        total_inference += time.perf_counter() - start

        # each row in dets contains the following in order:
        # reg_output (bbox), obj_output, cls_output, kpts_output
        dets = output[0]

        dets = dets[dets[:, 4] >= args.score_thr]

        # rescale bbox and kpts
        dets[:, :4] /= ratio
        dets[:, 6::3] /= ratio
        dets[:, 7::3] /= ratio

        if dets is not None:
            for det in dets:
                origin_img = visualize(origin_img, det)

        mkdir(args.output_dir)
        output_path = os.path.join(args.output_dir, image_file)
        print(output_path)
        cv2.imwrite(output_path, origin_img)

    print(
        f"total_inference: {total_inference}, {len(image_files)} images, avg {total_inference / len(image_files)}"
    )
