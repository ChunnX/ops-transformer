#!/bin/bash
# ============================================================================
# 迭代二 穿刺1 (int8 RECV) 一键:
#   编译(dav-3510) -> 生成数据+golden -> NPU 直调 -> 8 路二进制比对
# 验证目标: P2 int8 + 32B FP32 scale 连续 gather 在 RECV(TilingKey 101) 路径。
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
./ffn_wb_int8_recv_probe

echo "=== [4/4] 二进制一致比对 ==="
python3 ../scripts/verify_result.py output
echo "=== DONE ==="
