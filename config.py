datasets = {
    "synth43k": {
        "data_subdir": "SynthTree43k",
        "train_ann": "trees_train.json",
        "test_ann": "trees_test.json",
        "val_ann": "trees_val.json",
        "mean_bgr": [64.2190, 80.8418, 89.3639],
        "std_bgr": [55.3860, 56.5531, 60.9160]
    },
    "canawiki200": {
        "data_subdir": "CanaWikiTree200",
        "train_ann": "CanaWikiTree200_train.json",
        "test_ann": "CanaWikiTree200_val.json", # use val for testing since only 200 images
        "val_ann": "CanaWikiTree200_val.json",
        "mean_bgr": [83.3623, 94.0783, 96.6455],
        "std_bgr": [53.8556, 54.2480, 54.6537]
    }
}

model_sizes = { # depth, width
    "s": (0.33, 0.5),
    "m": (0.67, 0.75),
    "l": (1, 1),
}