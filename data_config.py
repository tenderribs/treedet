SUPPORTED_DATASETS = ["synth43k", "cana100"]

synth43k = {
    "data_subdir": "SynthTree43k",
    "train_ann": "trees_train.json",
    "test_ann": "trees_test.json",
    "val_ann": "trees_val.json",
}

cana100 = {
    "data_subdir": "CanaTree100",
    "train_ann": "trees_train.json",
    "test_ann": "trees_val.json", # use val for testing cuz only 500 images
    "val_ann": "trees_val.json",
}