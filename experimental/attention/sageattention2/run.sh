#!/usr/bin/env bash

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

    echo "Compiling test_op_host (REG_OP & InferShape) with ${CXX_COMPILER}..."
    local OP_HOST_BIN="${BUILD_DIR}/test_op_host"
    ${CXX_COMPILER} -std=c++17 -O2 -Wall -Wextra \
        "${WORK_DIR}/tests/test_op_host.cpp" \
        "${WORK_DIR}/op_host/sage_attention2_def.cpp" \
        "${WORK_DIR}/op_host/sage_attention2_infershape.cpp" \
        -I"${WORK_DIR}/op_host" \
        -I"${WORK_DIR}/aclnn" \
        -o "${OP_HOST_BIN}"

    echo "Executing ${OP_HOST_BIN}..."
    "${OP_HOST_BIN}"

    echo "Compiling test_output_queue with ${CXX_COMPILER}..."
    local QUEUE_BIN="${BUILD_DIR}/test_output_queue"
    ${CXX_COMPILER} -std=c++17 -O2 -Wall -Wextra \
        "${WORK_DIR}/tests/test_output_queue.cpp" \
        -I"${WORK_DIR}/sageattn_fp8/ascendc" \
        -I"${WORK_DIR}/sageattn_fp8/catlass" \
        -I"${WORK_DIR}/quant" \
        -o "${QUEUE_BIN}"

    echo "Executing ${QUEUE_BIN}..."
    "${QUEUE_BIN}"

    echo "Compiling test_tiling_struct with ${CXX_COMPILER}..."
    local TILING_BIN="${BUILD_DIR}/test_tiling_struct"
    ${CXX_COMPILER} -std=c++17 -O2 -Wall -Wextra \
        "${WORK_DIR}/tests/test_tiling_struct.cpp" \
        -I"${WORK_DIR}/sageattn_fp8/ascendc" \
        -I"${WORK_DIR}/sageattn_fp8/catlass" \
        -I"${WORK_DIR}/quant" \
        -o "${TILING_BIN}"

    echo "Executing ${TILING_BIN}..."
    "${TILING_BIN}"

    echo "Compiling test_pipeline_guard with ${CXX_COMPILER}..."
    local GUARD_BIN="${BUILD_DIR}/test_pipeline_guard"
    ${CXX_COMPILER} -std=c++17 -O2 -Wall -Wextra \
        "${WORK_DIR}/tests/test_pipeline_guard.cpp" \
        -I"${WORK_DIR}/sageattn_fp8/ascendc" \
        -o "${GUARD_BIN}"

    echo "Executing ${GUARD_BIN}..."
    "${GUARD_BIN}"

    echo ">>> C++ Host, Tiling, Guard & OutputQueue Tests Completed Successfully!"
}

check_kernel_syntax() {
    echo ">>> Checking Device Kernel Syntax & Headers..."
    local CXX_COMPILER="g++"
    if command -v clang++ >/dev/null 2>&1; then
        CXX_COMPILER="clang++"
    fi

    mkdir -p "${BUILD_DIR}"
    local OBJ_OUT="${BUILD_DIR}/sageattention2_syntax.o"

    echo "Checking op_kernel/sageattention2.cpp with ${CXX_COMPILER}..."
    ${CXX_COMPILER} -std=c++17 -c \
        "${WORK_DIR}/op_kernel/sageattention2.cpp" \
        -I"${WORK_DIR}/sageattn_fp8/ascendc" \
        -I"${WORK_DIR}/sageattn_fp8/catlass" \
        -I"${WORK_DIR}/quant" \
        -o "${OBJ_OUT}"

    rm -f "${OBJ_OUT}"
    echo ">>> Device Kernel Syntax Check Passed!"
}

run_python_tests() {
    echo ">>> Running Python Tiling & Quantization Tests..."
    # 解释器选择链：SAGE_PYTHON 环境变量 → op 内 .venv → 仓库外 workspace 级 .venv-sage → PATH
    local PY_BIN=""
    for cand in "${SAGE_PYTHON:-}" \
                "${WORK_DIR}/.venv/bin/python" \
                "${WORK_DIR}/../../../../.venv-sage/bin/python" \
                "$(command -v python3 2>/dev/null)" \
                "$(command -v python 2>/dev/null)"; do
        if [ -n "${cand}" ] && [ -x "${cand}" ]; then
            PY_BIN="${cand}"
            break
        fi
    done
    if [ -z "${PY_BIN}" ]; then
        echo "ERROR: no python interpreter found (set SAGE_PYTHON)"
        return 1
    fi

    echo "Using python: ${PY_BIN}"
    echo "Running test_tiling_struct.py..."
    "${PY_BIN}" -m unittest "${WORK_DIR}/tests/test_tiling_struct.py"

    if [ -f "${WORK_DIR}/tests/test_quant.py" ]; then
        echo "Running test_quant.py..."
        "${PY_BIN}" "${WORK_DIR}/tests/test_quant.py"
    fi
    if [ -f "${WORK_DIR}/tests/test_kernel_twin.py" ]; then
        echo "Running test_kernel_twin.py (宿主孪生执行器与 kernel 宿主串行分支 vs golden)..."
        "${PY_BIN}" "${WORK_DIR}/tests/test_kernel_twin.py"
    fi
    echo ">>> Python Tests Completed!"
}

build_in_container() {
    echo ">>> Checking CANN / Container Build Environment..."
    mkdir -p "${BUILD_DIR}"
    if [ -n "${ASCEND_HOME_PATH:-}" ] && [ -x "${ASCEND_HOME_PATH}/bin/bisheng" ]; then
        echo "Found BiSheng compiler at ${ASCEND_HOME_PATH}/bin/bisheng"
        echo "Compiling kernel with BiSheng for Ascend 950 (dav-3510)..."
        "${ASCEND_HOME_PATH}/bin/bisheng" --npu-arch=dav-3510 -std=c++17 -xasc \
            -c "${WORK_DIR}/op_kernel/sageattention2.cpp" \
            -I"${ASCEND_HOME_PATH}/include" \
            -I"${WORK_DIR}/sageattn_fp8/ascendc" \
            -I"${WORK_DIR}/sageattn_fp8/catlass" \
            -I"${WORK_DIR}/quant" \
            -o "${BUILD_DIR}/sageattention2_bisheng.o"
        echo ">>> BiSheng compilation succeeded!"
    else
        echo "Notice: ASCEND_HOME_PATH / BiSheng not detected in current shell; skipped device binary emission."
    fi
}

run_opp_tests() {
    echo ">>> Running CANN Standard Custom OPP Packaging Test..."
    bash "${WORK_DIR}/build_opp.sh" --dry-run --clean
    bash "${WORK_DIR}/install_opp.sh" --mock --quiet
    echo ">>> Custom OPP Packaging Test Succeeded!"
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
    opp)
        run_opp_tests
        ;;
    container)
        build_in_container
        ;;
    all)
        run_host_tests
        check_kernel_syntax
        run_python_tests
        run_opp_tests
        build_in_container
        echo "================================================================="
        echo "All requested SageAttention2 tests & checks passed successfully!"
        echo "================================================================="
        ;;
    *)
        echo "Unknown target: ${TARGET}"
        echo "Usage: $0 [all|host|kernel|python|opp|container]"
        exit 1
        ;;
esac
