FROM nvcr.io/nvidia/pytorch:24.02-py3

# Install devtools for opencv and perfor
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends pip build-essential tmux

RUN apt-get install -y neovim

# Make sure mountpoint for datasets is available
RUN mkdir -p /datasets

# fetch some pip packages from the ./setup.sh script
RUN apt-get install -y libjpeg-dev zlib1g-dev cmake libffi-dev
RUN pip3 install --no-input --upgrade pip setuptools
RUN pip uninstall --yes pillow
RUN pip install --no-input -U --force-reinstall pillow-simd
RUN pip3 install --no-input cython wheel numpy
RUN pip3 install --no-input torchinfo pycocotools