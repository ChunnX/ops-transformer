#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
BUILD_DIR=${BUILD_DIR:-$SCRIPT_DIR/build/pta}

: "${ASCEND_HOME_PATH:?ASCEND_HOME_PATH must point to the CANN installation root}"
: "${CUST_OPAPI_LIBRARY:?CUST_OPAPI_LIBRARY must be the absolute path to libcust_opapi.so}"
if [[ -z "${SHMEM_HOME_PATH:-}" && -z "${SHMEM_HOME:-}" ]]; then
    echo "SHMEM_HOME_PATH (source SDK layout) or SHMEM_HOME (HCCL-integrated layout) is required" >&2
    exit 1
fi
if [[ ! -x "$(command -v "$PYTHON_BIN" 2>/dev/null || true)" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -f "$CUST_OPAPI_LIBRARY" ]]; then
    echo "CUST_OPAPI_LIBRARY does not exist: $CUST_OPAPI_LIBRARY" >&2
    exit 1
fi

TORCH_CMAKE_PREFIX=$(
    "$PYTHON_BIN" -c 'import torch; print(torch.utils.cmake_prefix_path)'
)
TORCH_NPU_ROOT=$(
    "$PYTHON_BIN" -c 'import pathlib; import torch_npu; print(pathlib.Path(torch_npu.__file__).resolve().parent)'
)

cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DPython3_EXECUTABLE="$PYTHON_BIN" \
    -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PREFIX" \
    -DTORCH_NPU_ROOT="$TORCH_NPU_ROOT" \
    -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" \
    -DSHMEM_HOME_PATH="${SHMEM_HOME_PATH:-}" \
    -DSHMEM_HOME="${SHMEM_HOME:-}" \
    -DCUST_OPAPI_LIBRARY="$CUST_OPAPI_LIBRARY" \
    -DCUST_OPAPI_INCLUDE_DIR="${CUST_OPAPI_INCLUDE_DIR:-}"
cmake --build "$BUILD_DIR" --parallel "${BUILD_JOBS:-8}"

PTA_LIBRARY="$BUILD_DIR/liballto_allv_grouped_mat_mul_shmem_pta.so"
if [[ ! -f "$PTA_LIBRARY" ]]; then
    echo "Build completed without the expected library: $PTA_LIBRARY" >&2
    exit 1
fi
echo "PTA library: $(realpath "$PTA_LIBRARY")"
