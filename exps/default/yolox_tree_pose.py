#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp
import torch
import torch.distributed as dist
import torch.nn as nn

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.device_type = 'cuda'

        # ---------------- model config ---------------- #
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.num_classes = 1 # just trees
        self.num_kpts = 5
        self.default_sigmas = False # refers the the sigmas used in OKS formula
        self.input_size = (384, 672)    # (height, width)
        self.act = "relu"
        # ---------------- dataloader config ---------------- #
        self.data_subdir = "SynthTree43k"
        self.train_ann = "trees_train.json"
        self.test_ann = "trees_test.json"
        self.val_ann = "trees_val.json"
        # --------------- transform config ----------------- #
        # self.mosaic_prob = 0.0
        # self.mixup_prob = 0.0
        # self.hsv_prob = 1.0
        # self.flip_prob = 0.0
        # self.degrees = 10.0
        # self.translate = 0.1
        # self.mosaic_scale = (0.9, 1.1)
        # self.mixup_scale = (1.0, 1.0)
        # self.shear = 0.0
        # self.perspective = 0.0
        # self.enable_mixup = False
        # self.shape_loss = False
        # --------------  training config --------------------- #
        self.max_epoch = 100
        # self.eval_interval = 10
        # self.print_interval = 25
        self.basic_lr_per_img = 0.02 / 64 # batch size 32
        # -----------------  testing config ------------------ #
        self.human_pose = True
        self.visualize = False #True
        self.od_weights = None
        self.test_size = self.input_size
    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHeadKPTS

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act, conv_focus=True, split_max_pool_kernel=True)
            head = YOLOXHeadKPTS(self.num_classes, self.width, in_channels=in_channels, act=self.act, default_sigmas=self.default_sigmas, num_kpts=self.num_kpts)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_data_loader(
        self, batch_size, is_distributed, no_aug=False, cache_img=False
    ):
        from yolox.data import (
            TREEKPTSDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import (
            wait_for_the_master,
            get_local_rank,
        )

        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            dataset = TREEKPTSDataset(
                data_dir=self.data_subdir,
                json_file=self.train_ann,
                num_kpts=self.num_kpts,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob,
                    num_kpts=self.num_kpts),
                cache=cache_img,
            )


        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
                flip_index=dataset.flip_index,
                num_kpts=self.num_kpts,
            ),
            num_kpts=self.num_kpts,
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import TREEKPTSDataset, ValTransform

        valdataset = TREEKPTSDataset(
            data_dir=self.data_subdir,
            json_file=self.val_ann if not testdev else self.test_ann,
            num_kpts=self.num_kpts,
            img_size=self.test_size,
            preproc=ValTransform(),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import TreeKptsEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        output_dir = os.path.join(self.output_dir, self.exp_name)
        evaluator = TreeKptsEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            num_classes=self.num_classes,
            num_kpts=self.num_kpts,
            default_sigmas=self.default_sigmas,
            device_type=self.device_type
        )
        return evaluator
