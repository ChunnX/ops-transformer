#!/bin/bash
set -e
cd "$(dirname "$0")"
source /home/l00946875/Ascend/cann/set_env.sh
export PATH="/root/miniconda3/envs/zzx_py310/bin:$PATH"

TORCH_CMAKE=$(python3 -c "import torch;print(torch.utils.cmake_prefix_path)")

rm -rf build && mkdir build && cd build
cmake .. -DNPU_ARCH=ascend950 -DCMAKE_BUILD_TYPE=Release -DTorch_DIR="${TORCH_CMAKE}/Torch"
make -j$(nproc)
echo "Build OK: $(pwd)/csrc/mc2_direct_launch.so"
