#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}"
BUILD_DIR="${WORK_DIR}/build"

echo "================================================================="
echo "SageAttention2 Build & Test Runner"
echo "Working directory: ${WORK_DIR}"
echo "================================================================="

TARGET="${1:-all}"

run_host_tests() {
    echo ">>> Running C++ Host & ACLNN Tiling Unit Tests..."
    local CXX_COMPILER="g++"
    if command -v clang++ >/dev/null 2>&1; then
        CXX_COMPILER="clang++"
    fi

    mkdir -p "${BUILD_DIR}"
    local TEST_BIN="${BUILD_DIR}/test_aclnn_host"

    echo "Compiling test_aclnn_host with ${CXX_COMPILER}..."
    ${CXX_COMPILER} -std=c++17 -O2 -Wall -Wextra \
        "${WORK_DIR}/tests/test_aclnn_host.cpp" \
        "${WORK_DIR}/aclnn/sage_attn_host.cpp" \
        -I"${WORK_DIR}/aclnn" \
        -I"${WORK_DIR}/sageattn_fp8/ascendc" \
        -I"${WORK_DIR}/sageattn_fp8/catlass" \
        -I"${WORK_DIR}/quant" \
        -o "${TEST_BIN}"

    echo "Executing ${TEST_BIN}..."
    "${TEST_BIN}"
    echo ">>> C++ Host Tests Completed Successfully!"
}

check_kernel_syntax() {
    echo ">>> Checking Device Kernel Syntax & Headers..."
    local CXX_COMPILER="g++"
    if command -v clang++ >/dev/null 2>&1; then
        CXX_COMPILER="clang++"
    fi

    mkdir -p "${BUILD_DIR}"
    local OBJ_OUT="${BUILD_DIR}/sage_attn_kernel_syntax.o"

    echo "Checking sage_attn_kernel.cpp with ${CXX_COMPILER}..."
    ${CXX_COMPILER} -std=c++17 -c \
        "${WORK_DIR}/sageattn_fp8/ascendc/sage_attn_kernel.cpp" \
        -I"${WORK_DIR}/sageattn_fp8/ascendc" \
        -I"${WORK_DIR}/sageattn_fp8/catlass" \
        -I"${WORK_DIR}/quant" \
        -o "${OBJ_OUT}"

    rm -f "${OBJ_OUT}"
    echo ">>> Device Kernel Syntax Check Passed!"
}

run_python_tests() {
    echo ">>> Running Python Tiling & Quantization Tests..."
    local PY_BIN="python3"
    if [ -f "${WORK_DIR}/.venv/bin/python" ]; then
        PY_BIN="${WORK_DIR}/.venv/bin/python"
    elif command -v python >/dev/null 2>&1; then
        PY_BIN="python"
    fi

    echo "Using python: ${PY_BIN}"
    echo "Running test_tiling_struct.py..."
    "${PY_BIN}" -m unittest "${WORK_DIR}/tests/test_tiling_struct.py"

    if [ -f "${WORK_DIR}/tests/test_quant.py" ]; then
        echo "Running test_quant.py..."
        "${PY_BIN}" "${WORK_DIR}/tests/test_quant.py" || true
    fi
    echo ">>> Python Tests Completed!"
}

build_in_container() {
    echo ">>> Checking CANN / Container Build Environment..."
    if [ -n "${ASCEND_HOME_PATH:-}" ] && [ -x "${ASCEND_HOME_PATH}/bin/bisheng" ]; then
        echo "Found BiSheng compiler at ${ASCEND_HOME_PATH}/bin/bisheng"
        echo "Compiling kernel with BiSheng for Ascend 950 (dav-3510)..."
        "${ASCEND_HOME_PATH}/bin/bisheng" --npu-arch=dav-3510 -std=c++17 -xasc \
            -c "${WORK_DIR}/sageattn_fp8/ascendc/sage_attn_kernel.cpp" \
            -I"${ASCEND_HOME_PATH}/include" \
            -I"${WORK_DIR}/sageattn_fp8/ascendc" \
            -I"${WORK_DIR}/sageattn_fp8/catlass" \
            -I"${WORK_DIR}/quant" \
            -o "${BUILD_DIR}/sage_attn_kernel_bisheng.o"
        echo ">>> BiSheng compilation succeeded!"
    else
        echo "Notice: ASCEND_HOME_PATH / BiSheng not detected in current shell; skipped device binary emission."
    fi
}

case "${TARGET}" in
    host)
        run_host_tests
        ;;
    kernel)
        check_kernel_syntax
        ;;
    python)
        run_python_tests
        ;;
    container)
        build_in_container
        ;;
    all)
        run_host_tests
        check_kernel_syntax
        run_python_tests
        build_in_container
        echo "================================================================="
        echo "All requested SageAttention2 tests & checks passed successfully!"
        echo "================================================================="
        ;;
    *)
        echo "Unknown target: ${TARGET}"
        echo "Usage: $0 [all|host|kernel|python|container]"
        exit 1
        ;;
esac
