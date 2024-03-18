#!/usr/bin/env bash

# Copyright (c) 2018-2021, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

######################################################################

# system packages
sudo apt-get install -y libjpeg-dev zlib1g-dev cmake libffi-dev protobuf-compiler

######################################################################
# upgrade pip
pip3 install --no-input --upgrade pip setuptools

######################################################################
echo "installing pytorch - use the applopriate index-url from https://pytorch.org/get-started/locally/"

# unfortunately needs to be nightly build stable torch binaries aren't built for "new" sm_89 architecture which includes my RTX 4070 (ada gen)
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121 # install more recent version of torch compiled against cuda 12.1

echo 'Installing python packages...'
# there as issue with installing pillow-simd through requirements - force it here
pip uninstall --yes pillow
pip install --no-input -U --force-reinstall pillow-simd
pip3 install --no-input cython wheel numpy
pip3 install --no-input torchinfo pycocotools opencv-python

echo "installing requirements"
pip3 install --no-input -r requirements.txt

######################################################################
echo "Installing mmcv"
# pip3 install --no-input mmcv-full==1.4.8 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.1/index.html
# kinda sus because actually using torch v2.2, not v2.1 but whatever this project is called yolo for a reason
pip3 install --no-input mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torc==1.23.0h2.1/index.html # newer version

######################################################################
# can we move this inside the requirements file is used.
# pip3 install --no-input protobuf==3.20.2 onnx==1.13.0
pip3 install --no-input protobuf==3.20.3

# add build flag, otherwise problems https://github.com/onnx/onnx/issues/4704
export CMAKE_ARGS="-DONNX_USE_PROTOBUF_SHARED_LIBS=ON"
pip3 install --no-input onnx

######################################################################
echo 'installing the python package...'
python3 setup.py develop

