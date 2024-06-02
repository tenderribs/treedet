# Treedet

The ROS related stuff is located in `treedet_ros` while the development code for the computer vision model is in `treedet`.

The computer vision subfolder is based on [edgeai-yolox](https://github.com/TexasInstruments/edgeai-yolox).

## Getting Started

Install the packages for running the rospy node running with

```sh
pip install onnxruntime numpy scikit-learn scikit-image filterpy
```

This project has a two configured VSCode Devcontainers in `.devcontainer` to facilitate setup for both ROS and ML model development.


## ROS

### Nodes

```
roslaunch treedet_ros main.launch
```

- When running the code on rosbags from field tests, it is important to use the `--clock` flag to indicate sim time, for example:
```sh
rosbag play -l --clock *.bag
```

## Datasets

To train the network, you need the synthetic and real datasets.

| Dataset     | Description                            | Link                                                                                                 |
| ----------- | -------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Synth43k    | 43k computer generated images of trees | [Github](https://github.com/norlab-ulaval/PercepTreeV1)                                              |
| CanaTree100 | 100 real images from Canadian forests  | [Github](https://github.com/norlab-ulaval/PercepTreeV1)                                              |
| WikiTree    | 100 real images from forests worldwide | [Google Drive](https://drive.google.com/drive/folders/1CBwYHaWVl0_Li1czkcmdCc6V6Mw0ee8D?usp=sharing) |

- Annotations for Synth43k and CanaTree100 are provided in the folders on S3 and Onedrive respectively. Wikitrees has annotations on Google Drive and also has annotations of both real datasets merged together. Note that you should still symlink the image files of CanaTree100 into the WikiTree100 for the full CanaWikiTree200 dataset.

- The best performing model is already located in the `treedet/treedet` folder in onnx format for your convenience. For the other model checkpoint files, check out this link: [Pretrained Model Checkpoint Files](https://drive.google.com/drive/folders/13LVyUGIS0vzHjzDNI97sHVZ7jDmVtHd1?usp=drive_link)

- Here is the expected example file system format for the datasets folder, starting from the `treedet/treedet` folder:

```
treedet/treedet
└── datasets
    ├── SynthTree43k
    │   ├── annotations
    │   │   ├── trees_train.json
    │   │   ├── trees_test.json
    │   │   └── trees_val.json
    │   └── *.png
    └── CanaWikiTree200
        ├── annotations
        │   ├── trees_train.json
        │   ├── trees_test.json
        │   └── trees_val.json
        └── *.png
```

There is a list of commands in [`./treedet/command.sh`](./treedet/commands.sh), which give examples for the syntax and things you can do in the ML container related to training and evaluation.