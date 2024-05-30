# scripts

Here are some scripts used to try out ideas and perform small tasks. Not mean to be executed in a ROS context.


## `test_cylinder_fit.py`

This script tests the code that fits a partial cylinder to the lidar pointcloud of a tree trunk in a sandboxed environment.


## `select_targets.py`

The idea is to select some trees for cutting based on data in the rosbags and save the targets for evaluation. This list of trees is supplied to the treedet_ros package, which tries to match the trees it finds with those in the target list. To conclude, an evaluation script compares the outputs of treedet_ros and the global list.

### Construct global map

Install [Open3d SLAM](https://open3d-slam.readthedocs.io/en/latest) and then run on the rosbags of your choice (with map to odom transformations available). You probably need to switch the parameters of the LiDAR module in the launch file.

```bash
roslaunch open3d_slam_ros mapping.launch
```

Once you are satisfied with the assembled map based on RVIZ, save the map to disk.

```bash
rosservice call /mapping_node/save_map
```

### Select trees to cut

First fire up the tree_detection module that extracts the bboxes from the global pointcloud. For some reason the default ground removal strategy doesn't work, so just cropbox instead.

```bash
roslaunch tree_detection_ros tree_detection.launch ground_removal_strategy:=cropbox launch_rviz:=False pcd_filepath:=<PCD FILEPATH>
```

With the above launch file still running, go ahead and run the command below. You can toggle the selction of a tree by clicking on its bounding box. When you kill the launch process, it saves the selected trees as a csv file.

```bash
roslaunch treedet_ros select_targets.launch
```

### Evaluate target acquisition

```bash
# save detections from treedet_ros inference script
roslaunch treedet_ros main.launch gui:=False
```

```sh
rosrun pcl_ros pcd_to_pointcloud /datasets/maps/straight_line.pcd 1 _frame_id:=map
```

## `map_to_map_o3d.py`

The `open3d_slam_ros` mapping software produces a map in the map_o3d frame. The map_o3d and range_sensor_o3d (a.k.a. PandarQT of the lidar) origins coincide. This should probably be the same case for the compslam package that was used to create the rosbags. Yet for some reason there is a translation and some yaw between the two produced maps. This script calculates the static transformation between the two coordinate frames.

With the ICP output you can update the transformation matrix in `pcl.py`.

```bash
rosbag play <whatever rosbag was used to generate map with o3d>

# make sure to update the path to the map_pcd file. this file listens to rosbag for the first five /tf and /hesai/pandar messages
python3 map_to_map_o3d.py
```