import torch
import argparse
import os

from yolox.data import TREEKPTSDataset, ValTransform
from yolox.evaluators import KptsEvaluator
from yolox.exp import get_exp

from data_config import SUPPORTED_DATASETS, synth43k, cana100


def make_parser():
    parser = argparse.ArgumentParser("Evaluate kpts")
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--dataset", default=None, type=str, help="dataset for evaluation", required=True)
    parser.add_argument("-f", "--exp_file", type=str, default="synth43k", required=True)
    parser.add_argument("-c", "--ckpt", type=str, default="synth43k", required=True)
    parser.add_argument("--testset", action="store_true", required=True)
    return parser

def get_val_loader(exp, dataset, testset):
    valdataset = TREEKPTSDataset(
        data_dir=dataset["data_subdir"],
        json_file=dataset["test_ann"] if testset else dataset["val_ann"],
        num_kpts=exp.num_kpts,
        img_size=exp.test_size,
        preproc=ValTransform(),
    )

    sampler = torch.utils.data.SequentialSampler(valdataset)

    dataloader_kwargs = {
        "num_workers": 4,
        "batch_size": 32,
        "pin_memory": True,
        "sampler": sampler,
    }
    return torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)


if __name__ == '__main__':
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    ds =  cana100 if args.dataset == "cana100" else synth43k

    val_loader = get_val_loader(exp, ds, args.testset)

    evaluator = KptsEvaluator(
        dataloader=val_loader,
        img_size=exp.test_size,
        num_classes=exp.num_classes,
        num_kpts=exp.num_kpts,
        default_sigmas=exp.default_sigmas,
        device_type=exp.device_type
    )

    # load the model
    model = exp.get_model()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "model" in ckpt: ckpt = ckpt["model"]
    model.load_state_dict(ckpt)

    model.eval()
    model.to(exp.device_type)

    # run evaluation
    ap50_95, ap50, summary = evaluator.evaluate(
        model,
        distributed=False,
        half=False,
        test_size=exp.test_size
    )

    print(f"ap50_95: {ap50_95}, ap50: {ap50}\n\n{summary}")
