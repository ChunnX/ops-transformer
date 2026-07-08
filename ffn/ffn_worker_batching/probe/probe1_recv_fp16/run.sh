#!/bin/bash
# ============================================================================
# 穿刺1 (RECV fp16) 一键: 编译 -> 生成数据+golden -> NPU 直调 -> 二进制比对
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

source /home/zl/Ascend/ascend-toolkit/set_env.sh

SKIP_BUILD=0
[ "${1:-}" = "--skip-build" ] && SKIP_BUILD=1

if [ "${SKIP_BUILD}" -eq 0 ]; then
    echo "=== [1/4] 编译 (dav-3510) ==="
    rm -rf build
    mkdir -p build
    cd build
    cmake ..
    make -j4
    cd ..
fi

cd build
echo "=== [2/4] 生成输入 + CPU golden ==="
python3 ../scripts/gen_data.py

echo "=== [3/4] NPU 直调 ==="
rm -f output/y.bin
./ffn_wb_recv_probe

echo "=== [4/4] 二进制一致比对 ==="
python3 ../scripts/verify_result.py output
echo "=== DONE ==="
