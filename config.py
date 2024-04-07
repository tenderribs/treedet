datasets = {
    "synth43k": {
        "data_subdir": "SynthTree43k",
        "train_ann": "trees_train.json",
        "test_ann": "trees_test.json",
        "val_ann": "trees_val.json",
        "train_mean": [64.3229, 81.2194, 89.7864],
        "train_std": [57.0427, 58.6869, 63.5141]
    },
    "cana100": {
        "data_subdir": "CanaTree100",
        "train_ann": "trees_train.json",
        "test_ann": "trees_val.json", # use val for testing cuz only 100 images
        "val_ann": "trees_val.json",
        "train_mean": [63.9591, 80.7588, 89.1241],
        "train_std": [56.9384, 58.6090, 63.3794]
    }
}

model_sizes = { # depth, width
    "s": (0.33, 0.5),
    "m": (0.67, 0.75),
    "l": (1, 1),
}