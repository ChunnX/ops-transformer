#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

source /home/l00946875/Ascend/cann/set_env.sh 2>/dev/null

export PATH="/root/miniconda3/envs/mc2_py310_lbh/bin:$PATH"
CONDA_PREFIX="/root/miniconda3/envs/mc2_py310_lbh"

# Create ~/ascend/log to avoid CANN log warnings in command output
mkdir -p ~/ascend/log 2>/dev/null || true
sudo mkdir -p ~/ascend/log 2>/dev/null || true
sudo chmod -R 777 ~/ascend 2>/dev/null || true

TORCH_CMAKE=$(${CONDA_PREFIX}/bin/python3 -c "import torch;print(torch.utils.cmake_prefix_path)" 2>/dev/null | grep -E 'cmake' | tail -1)
TORCH_NPU_PATH=$(${CONDA_PREFIX}/bin/python3 -c "import torch_npu;print(torch_npu.__path__[0])" 2>/dev/null)

echo "Torch cmake: $TORCH_CMAKE"
echo "Torch NPU: $TORCH_NPU_PATH"

rm -rf "$SCRIPT_DIR/build"
cmake -S "$SCRIPT_DIR" -B "$SCRIPT_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DTorch_DIR="${TORCH_CMAKE}/Torch" \
    -DTORCH_NPU_PATH="${TORCH_NPU_PATH}"

cmake --build "$SCRIPT_DIR/build" -j"$(nproc)"
echo "Build OK: $SCRIPT_DIR/build/csrc/mc2_direct_launch.so"
