# Treedet
This repository is based on [edgeai-yolox](https://github.com/TexasInstruments/edgeai-yolox) and contains a  of the YOLOX repository for supporting additional tasks and embedded friendly ti_lite models.


## ROS Node

The ROS package wrapper is located in `treedet_ros` while the development code for the computer vision model is in `treedet`. When running the code on rosbags from field tests, it is important to use the `--clock` flag to indicate sim time, for example:

```sh
rosbag play -l --clock harveri-hpk_2023-05-05-09-25-20_6.bag
```

- Currently this module relies on the map to odom transformations supplied from other modules. I couldn't get these working super well on the field test data.

- The expectation is that you simlink the `treedet_ros` folder into the catkin_ws. The `treedet` folder only contains files related to training the machine learning model.

- For inference, copy an onnx file of your choice from the pretrained models into the root directory of `treedet_ros` as `treedet_ros/model.onnx`.

## Datasets

To train the network, you need the synthetic and real datasets.

| Dataset     | Description                            | Link                                                                                                 |
| ----------- | -------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Synth43k    | 43k computer generated images of trees | [Github](https://github.com/norlab-ulaval/PercepTreeV1)                                              |
| CanaTree100 | 100 real images from Canadian forests  | [Github](https://github.com/norlab-ulaval/PercepTreeV1)                                              |
| WikiTree    | 100 real images from forests worldwide | [Google Drive](https://drive.google.com/drive/folders/1CBwYHaWVl0_Li1czkcmdCc6V6Mw0ee8D?usp=sharing) |

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

This project has a fully configured VSCode Devcontainer to facilitate easy setup. If you so wish, the Docker container is in the `.devcontainer` folder for direct usage. It needs the following command to be run manually:

```sh
pip3 install --no-input -r requirements.txt && python3 setup.py develop
```

There is a list of commands in [`command.sh`](./commands.sh), which give examples for the syntax and things you can do.