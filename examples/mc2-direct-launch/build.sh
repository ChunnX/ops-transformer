#!/bin/bash
set -e
cd "$(dirname "$0")"
# 编译前需完成:
#   1. source CANN 安装包中的 set_env.sh（如 source $ASCEND_HOME_PATH/set_env.sh）
#   2. 激活 conda 环境（如 conda activate zzx_py310），确保 python3 可用

TORCH_CMAKE=$(python3 -c "import torch;print(torch.utils.cmake_prefix_path)")

rm -rf build && mkdir build && cd build
cmake .. -DNPU_ARCH=ascend950 -DCMAKE_BUILD_TYPE=Release -DTorch_DIR="${TORCH_CMAKE}/Torch"
make -j$(nproc)
echo "Build OK: $(pwd)/csrc/mc2_direct_launch.so"
