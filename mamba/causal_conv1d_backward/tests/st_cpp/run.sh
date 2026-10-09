#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================
# causal_conv1d_backward 算子 ST 测试执行脚本
#
# 用法:
#   bash run.sh --mock     # CPU golden 自检（无需 NPU）
#   bash run.sh --real     # NPU 执行（默认；需先构建并安装算子包）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USE_MOCK=""
BUILD_DIR=""

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --mock          使用 Mock 模式（CPU golden 验证，无需 NPU）"
    echo "  --real          使用 Real 模式（NPU 执行，默认）"
    echo "  --help          显示帮助信息"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --mock)
            USE_MOCK="-DUSE_MOCK=ON"
            shift
            ;;
        --real)
            USE_MOCK=""
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

if [ -n "$USE_MOCK" ]; then
    BUILD_DIR="${SCRIPT_DIR}/build-mock"
else
    BUILD_DIR="${SCRIPT_DIR}/build-real"
fi

echo "========================================"
echo "causal_conv1d_backward 算子 ST 测试 (C++)"
echo "========================================"
if [ -n "$USE_MOCK" ]; then
    echo "模式: Mock (CPU golden 验证)"
else
    echo "模式: Real (NPU 执行，默认)"
fi
echo "========================================"

if ! command -v cmake &> /dev/null; then
    echo "错误: 未找到 cmake"
    exit 1
fi
if ! command -v g++ &> /dev/null; then
    echo "错误: 未找到 g++"
    exit 1
fi

if [ -z "$USE_MOCK" ]; then
    if [ -z "$ASCEND_HOME_PATH" ]; then
        echo "警告: 未设置 ASCEND_HOME_PATH 环境变量"
        echo "建议设置: source set_env.sh (in CANN install directory)"
    fi

    # 定制定义算子包（build.sh 默认 vendor_name=custom → vendors/custom_transformer）
    CUSTOM_OPP_DIR=""
    for candidate_dir in \
        "${ASCEND_HOME_PATH}/opp/vendors/custom_transformer" \
        "${ASCEND_OPP_PATH}/vendors/custom_transformer" \
        "$(dirname "${ASCEND_HOME_PATH}")/latest/opp/vendors/custom_transformer"; do
        if [ -d "$candidate_dir" ]; then
            CUSTOM_OPP_DIR="$candidate_dir"
            break
        fi
    done
    if [ -n "$CUSTOM_OPP_DIR" ]; then
        export ASCEND_CUSTOM_OPP_PATH="${CUSTOM_OPP_DIR}"
        export LD_LIBRARY_PATH="${CUSTOM_OPP_DIR}/op_api/lib:${LD_LIBRARY_PATH}"
        echo "定制定义算子包: ${CUSTOM_OPP_DIR}"
    else
        echo "警告: 未找到 vendors/custom_transformer 定制定义算子包，Real 模式可能失败。"
        echo "      请先执行: bash build.sh -n causal_conv1d_backward -c ascend950 并安装输出包。"
    fi
    echo ""
fi

echo "创建构建目录..."
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "CMake 配置..."
if [ -n "$USE_MOCK" ]; then
    cmake .. -DUSE_MOCK=ON
else
    cmake ..
fi

echo "编译测试程序..."
make -j"$(nproc)"

echo "========================================"
echo "执行测试"
echo "========================================"

./test_causal_conv1d_backward
TEST_RESULT=$?

echo ""
echo "========================================"
if [ $TEST_RESULT -eq 0 ]; then
    echo "测试结果: PASS"
else
    echo "测试结果: FAIL"
fi
echo "========================================"

exit $TEST_RESULT
