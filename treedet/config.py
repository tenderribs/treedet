datasets = {
    "synth43k": {
        "data_subdir": "SynthTree43k",
        "train_ann": "trees_train.json",
        "test_ann": "trees_test.json",
        "val_ann": "trees_val.json",
        "mean_bgr": [64.2190, 80.8418, 89.3639],
        "std_bgr": [55.3860, 56.5531, 60.9160],
    },
    "canawiki200": {
        "data_subdir": "CanaWikiTree200",
        "train_ann": "CanaWikiTree200_train.json",
        "test_ann": "CanaWikiTree200_val.json",  # use val for testing since only 200 images
        "val_ann": "CanaWikiTree200_val.json",
        "mean_bgr": [83.3623, 94.0783, 96.6455],
        "std_bgr": [53.8556, 54.2480, 54.6537],
    },
    "cana100": {
        "data_subdir": "CanaTree100",
        "train_ann": "cana100_train.json",
        "test_ann": "cana100_val.json",  # use val for testing since only 100 images
        "val_ann": "cana100_val.json",
        "mean_bgr": [94.9900, 89.0835, 95.2646],
        "std_bgr": [54.7747, 54.7887, 53.5127],
    },
    "wiki100": {
        "data_subdir": "WikiTree100",
        "train_ann": "wiki100_train.json",
        "test_ann": "wiki100_val.json",  # use val for testing since only 100 images
        "val_ann": "wiki100_val.json",
        "mean_bgr": [81.4670, 100.9762, 98.7759],
        "std_bgr": [53.8882, 53.1455, 53.0477],
    },
}

model_sizes = {  # depth, width
    "s": (0.33, 0.5),
    "m": (0.67, 0.75),
    "l": (1, 1),
}
