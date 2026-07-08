#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd. — CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------------------------------------
# 迭代一 穿刺2 (probe2_norm_multicore) 一键运行:  编译 -> 生成数据 -> NPU 运行 -> 二进制一致校验
#   bash run.sh              # 完整流程
#   bash run.sh --skip-build # 复用已有编译产物
# 目标平台: Ascend950DT_9584 (arch35 / DAV_3510)
# ----------------------------------------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="ffn_wb_probe"
SKIP_BUILD=0
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/4] 设置 CANN 环境 ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || source /home/zl/Ascend/ascend-toolkit/set_env.sh || die "无法 source CANN 环境"
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH 未设置"

if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_NAME}" ] || die "--skip-build 但 build/${OP_NAME} 不存在"
    echo "=== [2/4] 跳过编译 ==="
else
    echo "=== [2/4] 编译 (--npu-arch=dav-3510) ==="
    rm -rf build
    mkdir -p build
    cd build
    cmake .. || die "cmake 配置失败"
    make -j8  || die "make 编译失败"
    cd ..
fi

echo "=== [3/4] 生成测试数据 + golden ==="
cd build
python3 ../scripts/gen_data.py || die "gen_data.py 失败"

echo "=== [4/4] NPU 运行 (Ascend950DT_9584 / DAV_3510) ==="
rm -f output/*.bin.out 2>/dev/null || true
"./${OP_NAME}" || die "Kernel 运行失败 (exit $?)"

echo "=== 二进制一致校验 ==="
python3 ../scripts/verify_result.py output || die "二进制一致校验失败"

echo "=== 完成 ==="
exit 0
