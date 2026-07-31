#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# run.sh — AlltoAllMatmulApace aclnn 测试一键脚本
# 用法:
#   bash run.sh --m 2048 --k 8192 --n 3584 --rank 4
#   bash run.sh                                    # 默认 2048/8192/3584/4
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
M=2048; K=8192; N=3584; RANK=4

usage() {
    echo "Usage: $0 [options]"
    echo "  --m M        per-rank M (default 2048)"
    echo "  --k K        total K   (default 8192)"
    echo "  --n N        N          (default 3584)"
    echo "  --rank R     rank count (default 4)"
    echo "  --skip-build skip cmake+build"
    echo "  --gen-only   仅生成数据"
    exit 0
}
SKIP_BUILD=0; GEN_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --m)   M="$2"; shift 2 ;;
        --k)   K="$2"; shift 2 ;;
        --n)   N="$2"; shift 2 ;;
        --rank)RANK="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        --gen-only)   GEN_ONLY=1; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown: $1"; usage ;;
    esac
done

echo "=========================================="
echo "AlltoAllMatmulApace aclnn 测试"
echo "  M=$M K=$K N=$N rank=$RANK"
echo "=========================================="

# CANN 环境
if [ -z "${ASCEND_HOME_PATH}" ]; then
    echo "ERROR: 请先 source CANN 的 set_env.sh"; exit 1
fi

# 自定义算子包环境
VENDOR_ENV="${ASCEND_HOME_PATH}/vendors/custom_transformer/bin/set_env.bash"
if [ -f "${VENDOR_ENV}" ]; then
    source "${VENDOR_ENV}" >/dev/null 2>&1
fi

# NPU 数量检查
NDEV=$(npu-smi info -l 2>/dev/null | grep -c "NPU ID" || echo "0")
if [ "${RANK}" -gt "${NDEV}" ]; then
    echo "ERROR: rank=${RANK} > NPU count=${NDEV}"; exit 1
fi

cd "${SCRIPT_DIR}"

# 1. 生成数据
echo "[1/4] 生成测试数据..."
rm -rf input output
mkdir -p input output
for r in $(seq 0 $((RANK-1))); do mkdir -p "input/$r" "output/$r"; done
python3 scripts/gen_data.py "${M}" "${K}" "${N}" "${RANK}"
echo "[1/4] 完成"

[ "${GEN_ONLY}" = "1" ] && exit 0

# 2. 编译
if [ "${SKIP_BUILD}" = "0" ]; then
    echo "[2/4] 编译..."
    mkdir -p "${BUILD_DIR}"
    cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DASCEND_HOME_PATH="${ASCEND_HOME_PATH}"
    cmake --build "${BUILD_DIR}" -j8
else
    echo "[2/4] 跳过编译"
fi

# 3. NPU 运行
echo "[3/4] NPU 运行 (${RANK} ranks)..."
rm -rf "${BUILD_DIR}/input" "${BUILD_DIR}/output"
cp -r input "${BUILD_DIR}/input"
cp -r output "${BUILD_DIR}/output"
mkdir -p "${BUILD_DIR}/output"
for r in $(seq 0 $((RANK-1))); do mkdir -p "${BUILD_DIR}/output/$r"; done

"${BUILD_DIR}/test_aclnn_allto_all_matmul_apace" "${M}" "${K}" "${N}" "${RANK}" "${BUILD_DIR}"

echo "[3/4] 完成"

# 4. 精度比对
echo "[4/4] 精度比对..."
python3 scripts/verify_result.py "${M}" "${N}" "${RANK}" "${BUILD_DIR}/output"

echo "=========================================="
echo "完成"
echo "=========================================="
