# YOLOX Trees
This repository is based on [edgeai-yolox](https://github.com/TexasInstruments/edgeai-yolox) and contains a  of the YOLOX repository for supporting additional tasks and embedded friendly ti_lite models.

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