#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================
# FfnWorkerBatching 算子 ST 测试执行脚本（C++ 原生，Mock/Real 模式）
#
#   bash run.sh            # Real 模式（NPU 执行 + 二进制一致比对，需安装 A5 算子包）
#   bash run.sh --mock     # Mock 模式（CPU golden 自测，无需 NPU，迭代一验收口径）
#
# Real 模式前置：source /root/Ascend/ascend-toolkit/set_env.sh
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USE_MOCK=""
L0_CSV="${SCRIPT_DIR}/testcases/aclnnFfnWorkerBatching_l0_test_cases.csv"
REPORT_DIR="${SCRIPT_DIR}/../reports"

show_help() {
    echo "Usage: $0 [--mock|--real] [--help]"
    echo "  --mock   Mock 模式（CPU golden 自测，无需 NPU）"
    echo "  --real   Real 模式（NPU 执行 + 二进制一致比对，默认）"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --mock) USE_MOCK="-DUSE_MOCK=ON"; shift ;;
        --real) USE_MOCK=""; shift ;;
        --help|-h) show_help; exit 0 ;;
        *) echo "未知参数: $1"; show_help; exit 1 ;;
    esac
done

if [ -n "$USE_MOCK" ]; then
    BUILD_DIR="${SCRIPT_DIR}/build-mock"
    MODE="mock"
else
    BUILD_DIR="${SCRIPT_DIR}/build-real"
    MODE="real"
fi

mkdir -p "${REPORT_DIR}"
LOG_FILE="${REPORT_DIR}/st_cpp_${MODE}_$(date +%Y%m%d_%H%M%S).log"

echo "========================================"
echo "FfnWorkerBatching ST 测试 (C++, ${MODE} 模式)"
echo "工作目录: ${SCRIPT_DIR}"
echo "执行日志: ${LOG_FILE}"
echo "========================================"

# 依赖检查
command -v cmake >/dev/null || { echo "错误: 未找到 cmake"; exit 1; }
command -v g++ >/dev/null || { echo "错误: 未找到 g++"; exit 1; }
if [ -z "$USE_MOCK" ] && [ -z "$ASCEND_HOME_PATH" ]; then
    echo "警告: 未设置 ASCEND_HOME_PATH，请先 source /root/Ascend/ascend-toolkit/set_env.sh"
fi

{
    echo "==== FfnWorkerBatching ST C++ ${MODE} run @ $(date) ===="
    echo "SCRIPT_DIR=${SCRIPT_DIR}"
    echo "L0_CSV=${L0_CSV}"

    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"

    echo "---- CMake 配置 ----"
    if [ -n "$USE_MOCK" ]; then
        cmake .. -DUSE_MOCK=ON
    else
        cmake ..
    fi

    echo "---- 编译 ----"
    make -j"$(nproc)"

    echo "---- 执行测试 ----"
    ./test_aclnn_ffn_worker_batching "${L0_CSV}"
} 2>&1 | tee "${LOG_FILE}"

# tee 之后取被测程序退出码（管道首命令）
TEST_RESULT=${PIPESTATUS[0]}

echo ""
echo "========================================"
if [ "${TEST_RESULT}" -eq 0 ]; then
    echo "测试结果: PASS ✓   (日志: ${LOG_FILE})"
else
    echo "测试结果: FAIL ✗   (日志: ${LOG_FILE})"
fi
echo "========================================"
exit "${TEST_RESULT}"
