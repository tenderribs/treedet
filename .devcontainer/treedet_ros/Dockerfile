# https://github.com/ika-rwth-aachen/docker-ros-ml-images/tree/main
FROM rwthika/ros-torch:latest

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    ros-$ROS_DISTRO-rviz \
    ros-$ROS_DISTRO-image-view


# make a catkin workspace
RUN mkdir -p /root/catkin_ws/src && \
    cd /root/catkin_ws/src && \
    source /opt/ros/${ROS_DISTRO}/setup.bash && \
    catkin_init_workspace

WORKDIR /root/catkin_ws
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    catkin_make

# Set up the environment to be sourced when starting the terminal
RUN echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc