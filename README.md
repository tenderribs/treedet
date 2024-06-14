# Treedet

The ROS related stuff is located in `treedet_ros` while the development code for the computer vision model is in `treedet`.

The computer vision subfolder is based on [edgeai-yolox](https://github.com/TexasInstruments/edgeai-yolox).

## Getting Started

Install the packages for running the rospy node running with

```sh
pip install optimum[onnxruntime-gpu] numpy scikit-learn scikit-image filterpy
```

The onnxruntime is set to use the CUDA execution provider, but might default to the CPU execution provider if CUDA isn't configured properly.

For a more detailed setup info, this project has a two configured VSCode Devcontainers in `.devcontainer` to facilitate automatic setup for both ROS and ML model development.

## ROS

### Nodes


```bash
roslaunch treedet_ros main.launch
```

At the time of the project deadline, the detected trees are published as HarveriDetectedTrees to provide compatibility with the existing software. Unfortunately this stays consistentent with the existing simplification that the trees are vertical cylinders, despite more accurate pose information being available. Truth be told, it is only a minor issue.

### Some remarks
- Look in the files `inference.py` and `cutting_data.py` for the core software.

- Tree cutting data is saved in the `tree_index` in this format (in the map frame):
```
[[pos_x, pos_y, pos_z, dim_x, dim_y, dim_z]]
```

The pos coordinates represent the cutting point of the tree (not the center of a cylinder).

- When running the code on rosbags from field tests, it is important to use the `--clock` flag to indicate sim time, for example:

```sh
rosbag play -l --clock *.bag
```

## Datasets

To train the network, you need the synthetic and real datasets.

| Dataset               | Description                                                       | Link                                                                                              |
| --------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Synth43k              | 43k computer generated images of trees                            | [Github](https://github.com/norlab-ulaval/PercepTreeV1)                                           |
| CanaTree100           | 100 real images from Canadian forests                             | [Github](https://github.com/norlab-ulaval/PercepTreeV1)                                           |
| CanaWikiSparseTree325 | CanaTree100 + 100 trees across the globe + 125 tree-sparse images | [Github](https://drive.google.com/drive/folders/1ipmGjLNhnr-HHqODUhc_dM78nxjyAbOg?usp=drive_link) |

- Annotations for Synth43k and CanaTree100 are provided in the folders on S3 and Onedrive respectively. Wikitrees has annotations on Google Drive and also has annotations of both real datasets merged together. Note that you should still symlink the image files of CanaTree100 into the WikiTree100 for the full CanaWikiTree200 dataset.

- The best performing model is already located in the `treedet/treedet_ros` folder in onnx format as `model.onnx` for your convenience. For the other model checkpoint files, check out this link: [Pretrained Model Checkpoint Files](https://drive.google.com/drive/folders/13LVyUGIS0vzHjzDNI97sHVZ7jDmVtHd1?usp=drive_link). Commands for creating onnx files are availble in `commands.sh`

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