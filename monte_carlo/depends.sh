#!/bin/bash 

# 1) Download Miniforge installer
cd /tmp
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

# 2) Install to your home directory
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3

# 3) Activate conda
source ~/.bashrc

# 4) Confirm
conda --version

pip install "jax[cuda13]"  # Install JAX with CUDA 13 support
