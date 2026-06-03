#!/bin/bash
#
# MoeEpDispatch ping-buffer 穿刺验证 一键编译运行脚本
#
# 用法:
#   ./run.sh              # 默认 dav-3510 (950PR / 950DT / 910PR)
#   ./run.sh dav-2201     # 910B / 910C / 910_93
#
# 流程:
#   [1/4] 生成随机输入 + golden (python3 scripts/gen_data.py)
#   [2/4] 编译 add_custom.asc (cmake + make)
#   [3/4] NPU 执行 demo (两段式 kernel + host 轮询 + 动态 alloc)
#   [4/4] 精度比对 (output_op1.bin / output_op2.bin vs golden.bin)
#
# 输出关键信息:
#   Register(MAPPED) syncDev 非空 → 零拷贝通道可用
#   pollCount > 0             → Host 确实在 kernel 跑着时自旋等待
#   error ratio: 0.0000       → 精度通过

set -e

ARCH="${1:-dav-3510}"
# dav-2201 → 910B / 910C / 910_93 (arch22)
# dav-3510 → 950PR / 950DT / 910PR (arch35)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

echo "=== [1/4] Generating test data ==="
python3 scripts/gen_data.py

echo "=== [2/4] Building ($ARCH) ==="
mkdir -p build && cd build
cmake -DCMAKE_ASC_ARCHITECTURES="$ARCH" .. && make -j

echo "=== [3/4] Running on NPU ==="
./demo

echo "=== [4/4] Verifying result ==="
cd "$SCRIPT_DIR"
python3 scripts/verify_result.py \
    output/output_op1.bin output/golden.bin \
    output/output_op2.bin output/golden.bin
