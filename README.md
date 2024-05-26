# Treedet
This repository is based on [edgeai-yolox](https://github.com/TexasInstruments/edgeai-yolox) and contains a  of the YOLOX repository for supporting additional tasks and embedded friendly ti_lite models.

The ROS related stuff is located in `treedet_ros` while the development code for the computer vision model is in `treedet`.

## ROS

### Nodes




```
roslaunch treedet_ros
```

- When running the code on rosbags from field tests, it is important to use the `--clock` flag to indicate sim time, for example:
```sh
rosbag play -l --clock *.bag
```
- The expectation is that you simlink only the `treedet_ros` folder into the catkin_ws. The `treedet` folder only contains files related to training the machine learning model.
- For inference, copy an onnx file of your choice from the pretrained models into the root directory of `treedet_ros` as `treedet_ros/model.onnx`.


## Datasets

To train the network, you need the synthetic and real datasets.

| Dataset     | Description                            | Link                                                                                                 |
| ----------- | -------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Synth43k    | 43k computer generated images of trees | [Github](https://github.com/norlab-ulaval/PercepTreeV1)                                              |
| CanaTree100 | 100 real images from Canadian forests  | [Github](https://github.com/norlab-ulaval/PercepTreeV1)                                              |
| WikiTree    | 100 real images from forests worldwide | [Google Drive](https://drive.google.com/drive/folders/1CBwYHaWVl0_Li1czkcmdCc6V6Mw0ee8D?usp=sharing) |

TODO: Add links to the pretrained models

Annotations for Synth43k and CanaTree100 are provided in the folders on S3 and Onedrive respectively. Wikitrees has annotations on Google Drive and also has annotations of both real datasets merged together. Note that you should still symlink the image files of CanaTree100 into the WikiTree100 for the full CanaWikiTree200 dataset.

Here is the expected file system format for the datasets folder, starting from root:

```
.
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

### Scripts

This project has a few partially configured VSCode Devcontainers to facilitate setup for ROS and ML model development. If you so wish, the Docker containers is in the `.devcontainer` folder for direct usage. Note that the ROS container does not contain all necessary packages. The ML container needs the following command to be run manually after build:

```sh
pip3 install --no-input -r requirements.txt && python3 setup.py develop
```

There is a list of commands in [`command.sh`](./commands.sh), which give examples for the syntax and things you can do in the ML container related to training and evaluation.