# Driver Installation

Using RTX 4070 w/ Ada Lovelace arch on Ubuntu 20.04

- [Install GPU drivers](https://www.nvidia.com/Download/index.aspx)
- [Install Cuda](https://developer.nvidia.com/cuda-downloads)


Here is what I am running:

```sh
$ nvidia-smi
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 550.54.14              Driver Version: 550.54.14      CUDA Version: 12.4     |
    |-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  NVIDIA GeForce RTX 4070        Off |   00000000:2D:00.0 Off |                  N/A |
    |  0%   35C    P8              7W /  200W |      19MiB /  12282MiB |      0%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+

$ nvcc --version
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2024 NVIDIA Corporation
    Built on Tue_Feb_27_16:19:38_PST_2024
    Cuda compilation tools, release 12.4, V12.4.99
    Build cuda_12.4.r12.4/compiler.33961263_0
```