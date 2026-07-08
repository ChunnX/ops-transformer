#!/bin/bash
# ============================================================================
# 迭代二穿刺4 (RECV fp16 稀疏 ~50% mask) 一键:
#   编译 -> 生成数据+golden -> NPU 直调 -> 二进制比对 -> 连跑 3 次确定性 SHA256
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

source /home/zl/Ascend/ascend-toolkit/set_env.sh

SKIP_BUILD=0
[ "${1:-}" = "--skip-build" ] && SKIP_BUILD=1

if [ "${SKIP_BUILD}" -eq 0 ]; then
    echo "=== [1/5] 编译 (dav-3510) ==="
    rm -rf build
    mkdir -p build
    cd build
    cmake ..
    make -j4
    cd ..
fi

cd build
echo "=== [2/5] 生成输入 + CPU golden ==="
python3 ../scripts/gen_data.py

echo "=== [3/5] NPU 直调 ==="
rm -f output/y.bin
./ffn_wb_recv_probe

echo "=== [4/5] 二进制一致比对 (vs CPU golden) ==="
python3 ../scripts/verify_result.py output

echo "=== [5/5] 确定性: 连跑 3 次 8 路输出 SHA256 ==="
OUTS="y group_list session_ids micro_batch_ids token_ids expert_offsets dynamic_scale actual_token_num"
for run in 1 2 3; do
    rm -f output/y.bin
    ./ffn_wb_recv_probe > /dev/null
    line="run${run}:"
    : > "output/_all_run${run}.bin"
    for o in ${OUTS}; do
        h=$(sha256sum "output/${o}.bin" | cut -c1-16)
        line="${line} ${o}=${h}"
        cat "output/${o}.bin" >> "output/_all_run${run}.bin"
    done
    echo "${line}"
    sha256sum "output/_all_run${run}.bin"
done
if cmp -s output/_all_run1.bin output/_all_run2.bin && cmp -s output/_all_run2.bin output/_all_run3.bin; then
    echo "[DETERMINISM] PASS: 3 次 8 路合并输出 bitwise 一致"
else
    echo "[DETERMINISM] FAIL: 多次运行输出不一致"; exit 1
fi
echo "=== DONE ==="
