import os
import cv2
import torch
import argparse
import numpy as np


from torch.utils.data import DataLoader, Subset

from yolox.data.datasets import TREEKPTSDataset
from config import datasets

BATCH_SIZE = 32
SAMPLE_SIZE = 10000


class NormalizationDataset(TREEKPTSDataset):
    def __getitem__(self, index):
        """
        Overrides the __getitem__ method to return only the image for the given index.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, _ = self.preproc(img, target, self.input_dim)

        return img

def make_parser():
    parser = argparse.ArgumentParser("Find Dataset Normalization Coefs")
    parser.add_argument("--dataset", default=None, type=str, choices=list(datasets.keys()), required=True)
    return parser


def calculate_normalization_coefficients(ds_conf):
    # only use training annotation to prevent info leak
    dataset = NormalizationDataset(
                data_dir=ds_conf['data_subdir'],
                json_file=ds_conf['train_ann'],
                mean_bgr=None, # explicitly None for clarity, as we are trying to find these out :)
                std_bgr=None
            )

    # Randomly sample from the dataset if it has more than SAMPLE_SIZE images
    if len(dataset) > SAMPLE_SIZE:
        indices = np.random.choice(len(dataset), SAMPLE_SIZE, replace=False)
        dataset = Subset(dataset, indices)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    mean = 0
    std = 0
    nbatches = 0.

    for images in loader:
        print(f"{nbatches} / {len(dataset) / BATCH_SIZE}")
        # example images.size(): torch.Size([32, 720, 1280, 3])
        images = images.float()
        batch_size, height, width, num_channels = images.shape

        # for each channel get the mean and stddev for this batch
        mean += images.float().mean(axis=(0, 1, 2))
        std += images.float().std(axis=(0, 1, 2))

        nbatches += 1

    mean /= nbatches
    std /= nbatches
    return mean, std


if __name__ == "__main__":
    args = make_parser().parse_args()

    # Specify your dataset folder path
    ds_conf = datasets[args.dataset]
    mean, std = calculate_normalization_coefficients(ds_conf)
    print(f'Mean: {mean}')
    print(f'Std: {std}')
