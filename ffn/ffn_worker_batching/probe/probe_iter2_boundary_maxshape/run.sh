#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd. — CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------------------------------------
# 迭代二 穿刺2 (probe_iter2_boundary_maxshape) 一键运行:
#   编译 -> 查询 UB(--query-ub) -> 生成数据+golden -> NPU 运行 -> 8 路二进制一致校验
#   bash run.sh              # 完整流程
#   bash run.sh --skip-build # 复用已有编译产物
# 目标平台: Ascend950DT_9584 (arch35 / DAV_3510)
# 说明: shape (Y/SLM) 由 --query-ub 从 NPU 运行时真实 UB 派生并写 ub_params.txt，gen_data 读取；
#       禁写死 arch UB 常量 (对齐 MEMORY「UB size no hardcode」)。
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

echo "=== [1/5] 设置 CANN 环境 ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || source /home/zl/Ascend/ascend-toolkit/set_env.sh || die "无法 source CANN 环境"
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH 未设置"

if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_NAME}" ] || die "--skip-build 但 build/${OP_NAME} 不存在"
    echo "=== [2/5] 跳过编译 ==="
else
    echo "=== [2/5] 编译 (--npu-arch=dav-3510) ==="
    rm -rf build
    mkdir -p build
    cd build
    cmake .. || die "cmake 配置失败"
    make -j8  || die "make 编译失败"
    cd ..
fi

cd build
echo "=== [3/5] 查询 UB (kernel/host 两侧口径) + 派生 shape ==="
"./${OP_NAME}" --query-ub || die "UB 查询失败"
[ -f ub_params.txt ] || die "ub_params.txt 未生成"

echo "=== [4/5] 生成测试数据 + golden ==="
python3 ../scripts/gen_data.py || die "gen_data.py 失败"

echo "=== [5/5] NPU 运行 (Ascend950DT_9584 / DAV_3510) ==="
rm -f output/*.bin.out 2>/dev/null || true
"./${OP_NAME}" || die "Kernel 运行失败 (exit $?)"

echo "=== 二进制一致校验 ==="
python3 ../scripts/verify_result.py output || die "二进制一致校验失败"

echo "=== 完成 ==="
exit 0
