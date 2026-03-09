#!/bin/bash
#########################################################
# Uncomment and change the variables below to your need:#
#########################################################

# Install directory without trailing slash
#install_dir="/home/$(whoami)"

# Name of the subdirectory
#clone_dir="stable-diffusion-webui"

# python3 executable - use system Python 3.12 (3.14 is too new for torch/numpy)
python_cmd="/usr/bin/python3.12"

# git executable
export GIT="git"

# python3 venv without trailing slash (defaults to ${install_dir}/${clone_dir}/venv)
venv_dir="venv"

# script to launch to start the app
export LAUNCH_SCRIPT="launch.py"

# install command for torch
# Let Forge auto-detect the right torch version for CUDA 12.8
#export TORCH_COMMAND="pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu128"

# Requirements file to use for stable-diffusion-webui
#export REQS_FILE="requirements_versions.txt"

# Fixed git repos
#export K_DIFFUSION_PACKAGE=""
#export GFPGAN_PACKAGE=""

# Fixed git commits
#export STABLE_DIFFUSION_COMMIT_HASH=""
#export CODEFORMER_COMMIT_HASH=""
#export BLIP_COMMIT_HASH=""

# Uncomment to enable accelerated launch
export ACCELERATE="True"

# Uncomment to disable TCMalloc
#export NO_TCMALLOC="True"

###########################################
# Model / LoRA / Embedding / Output paths
# (WSL2 auto-mounts Windows drives at /mnt/<letter>)
###########################################
export COMMANDLINE_ARGS="--listen --cuda-malloc --ckpt-dir /mnt/r/checkpoints --lora-dir /mnt/e/LoRAs --embeddings-dir /mnt/e/Embeddings"
