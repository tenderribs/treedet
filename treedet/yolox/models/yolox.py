#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_kpts_head import YOLOXHeadKPTS
from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(
        self, mean_bgr, std_bgr, backbone=None, head=None, freeze_backbone=False
    ):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN(freeze_backbone=freeze_backbone)
        if head is None:
            head = YOLOXHead(80)

        # Convert mean and std to tensors and reshape for broadcasting as B,C,W,H
        self.mean_bgr = torch.tensor(mean_bgr).view(1, 3, 1, 1).float()
        self.std_bgr = torch.tensor(std_bgr).view(1, 3, 1, 1).float()

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        # Ensure normalization parameters are on the same device as input
        std_bgr = self.std_bgr.to(device=x.device, dtype=x.dtype)
        mean_bgr = self.mean_bgr.to(device=x.device, dtype=x.dtype)

        # Normalize input
        x = (x - mean_bgr) / std_bgr

        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            if isinstance(self.head, YOLOXHead):
                loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                    fpn_outs, targets, x
                )
                outputs = {
                    "total_loss": loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    "num_fg": num_fg,
                }
            if isinstance(self.head, YOLOXHeadKPTS):
                (
                    loss,
                    iou_loss,
                    conf_loss,
                    cls_loss,
                    l1_loss,
                    kpts_loss,
                    kpts_vis_loss,
                    loss_l1_kpts,
                    num_fg,
                ) = self.head(fpn_outs, targets, x)
                outputs = {
                    "total_loss": loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    "kpts_loss": kpts_loss,
                    "kpts_vis_loss": kpts_vis_loss,
                    "l1_loss_kpts": loss_l1_kpts,
                    "num_fg": num_fg,
                }

        else:
            outputs = self.head(fpn_outs)

        return outputs
