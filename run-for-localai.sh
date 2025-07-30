#!/usr/bin/env bash

# Resolve the full path of the script
SCRIPT_NAME=$(basename "$0")
ABS_SCRIPT_PATH=$(readlink -f "$0")
ABS_SCRIPT_DIR=$(dirname "$ABS_SCRIPT_PATH")

ABS_VENV_DIR="${ABS_SCRIPT_DIR}/venv/bin/activate"

echo ""
echo "Starting A1111 ..."
echo ""

echo "SCRIPT_NAME: $SCRIPT_NAME"
echo "ABS_SCRIPT_DIR: $ABS_SCRIPT_DIR"
echo "ABS_SCRIPT_PATH: $ABS_SCRIPT_PATH"

echo ""
echo "Changing to A1111 directory (${ABS_SCRIPT_DIR})"
echo ""
cd $ABS_SCRIPT_DIR || exit 1

echo ""
echo "Activating venv (${ABS_VENV_DIR})"
echo ""
source $ABS_VENV_DIR

echo ""
echo "Launching sdwebui (python3 ${ABS_SCRIPT_DIR}/launch.py)"
echo ""

# Use ONLY GPU 2 (index 2)
export CUDA_VISIBLE_DEVICES="GPU-65e1ea06-f2e4-c85c-4db5-6febbfe62843"

# Or if you want to set device order: map GPU 2 → become "device 0" for PyTorch
python3 launch.py \
  --port 7860 \
  --gpu-device-id=0 \
  --always-low-vram \
  --vae-in-fp32 \
  --force-upcast-attention \
  --api \
  --cuda-malloc \
  --pin-shared-memory
