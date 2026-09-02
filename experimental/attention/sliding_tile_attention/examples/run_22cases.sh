#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
SOC_VERSION="${SLIDING_TILE_ATTENTION_SOC_VERSION:-ascend910_93}"
DEVICE_ID="${SLIDING_TILE_ATTENTION_DEVICE_ID:-0}"
BUILD_THREADS="${SLIDING_TILE_ATTENTION_BUILD_THREADS:-16}"
PYTHON_BIN="${SLIDING_TILE_ATTENTION_PYTHON:-$(command -v python3 || true)}"
SUITE="${SLIDING_TILE_ATTENTION_SUITE:-reviewer22}"
VENDOR_NAME="sta_review"
RUNNER_BUILD_DIR="$(mktemp -d /tmp/sliding-tile-attention-runner.XXXXXX)"
INSTALL_ROOT="$(mktemp -d /tmp/sliding-tile-attention-opp.XXXXXX)"

cleanup() {
    chmod -R u+w "${INSTALL_ROOT}" 2>/dev/null || true
    rm -rf "${RUNNER_BUILD_DIR}" "${INSTALL_ROOT}"
}
trap cleanup EXIT

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "missing ASCEND_HOME_PATH; source the CANN set_env.sh first" >&2
    exit 1
fi
if [[ ! -f "${REPO_ROOT}/build.sh" ]]; then
    echo "missing repository build.sh: ${REPO_ROOT}/build.sh" >&2
    exit 1
fi
if ! [[ "${BUILD_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid SLIDING_TILE_ATTENTION_BUILD_THREADS=${BUILD_THREADS}" >&2
    exit 1
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "invalid SLIDING_TILE_ATTENTION_PYTHON=${PYTHON_BIN}" >&2
    exit 1
fi
if ! "${PYTHON_BIN}" -c 'import packaging' >/dev/null 2>&1; then
    echo "SLIDING_TILE_ATTENTION_PYTHON cannot import packaging: ${PYTHON_BIN}" >&2
    exit 1
fi
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export ASCEND_PYTHON_EXECUTABLE="${PYTHON_BIN}"

cd "${REPO_ROOT}"
for generatedRoot in "${REPO_ROOT}/build" "${REPO_ROOT}/output"; do
    if [[ -d "${generatedRoot}" ]]; then
        chmod -R u+w "${generatedRoot}"
    fi
done
SOURCE_TREE_SHA256="$({
    /usr/bin/find experimental/attention/sliding_tile_attention/CMakeLists.txt \
        experimental/attention/sliding_tile_attention/op_host \
        experimental/attention/sliding_tile_attention/op_kernel \
        experimental/attention/sliding_tile_attention/examples \
        experimental/attention/sliding_tile_attention/tests \
        cmake/func.cmake \
        -type f -print0 | /usr/bin/sort -z |
        /usr/bin/xargs -0 /usr/bin/sha256sum
} | /usr/bin/sha256sum | /usr/bin/awk '{print $1}')"
echo "SLIDING_TILE_ATTENTION_SOURCE_TREE_SHA256=${SOURCE_TREE_SHA256}"
bash build.sh --pkg --experimental "--soc=${SOC_VERSION}" --ops=sliding_tile_attention \
    "--vendor_name=${VENDOR_NAME}" "-j${BUILD_THREADS}"

PACKAGE_PATH="${REPO_ROOT}/build_out/cann-ops-transformer-${VENDOR_NAME}_linux-aarch64.run"
if [[ ! -f "${PACKAGE_PATH}" ]]; then
    echo "custom operator package was not generated: ${PACKAGE_PATH}" >&2
    exit 1
fi
bash "${PACKAGE_PATH}" "--install-path=${INSTALL_ROOT}" --quiet

CUSTOM_OPP_ROOT="${INSTALL_ROOT}/vendors/${VENDOR_NAME}_transformer"
CUSTOM_INCLUDE="${CUSTOM_OPP_ROOT}/op_api/include"
CUSTOM_LIBRARY="${CUSTOM_OPP_ROOT}/op_api/lib/libcust_opapi.so"
CANN_INCLUDE="${ASCEND_HOME_PATH}/aarch64-linux/include"
CANN_LIBRARY="${ASCEND_HOME_PATH}/aarch64-linux/lib64"
for required in "${CUSTOM_INCLUDE}/aclnnop/aclnn_sliding_tile_attention.h" "${CUSTOM_LIBRARY}" \
                "${CANN_INCLUDE}/acl/acl.h" "${CANN_LIBRARY}/libascendcl.so" "${CANN_LIBRARY}/libnnopbase.so"; do
    if [[ ! -e "${required}" ]]; then
        echo "missing runner dependency: ${required}" >&2
        exit 1
    fi
done

g++ -std=c++17 -O2 -Wall -Wextra -Werror \
    "${SCRIPT_DIR}/test_aclnn_sliding_tile_attention_22cases.cpp" \
    -I"${CANN_INCLUDE}" -I"${CUSTOM_INCLUDE}" \
    "${CUSTOM_LIBRARY}" "${CANN_LIBRARY}/libascendcl.so" "${CANN_LIBRARY}/libnnopbase.so" \
    -Wl,-rpath,"${CUSTOM_OPP_ROOT}/op_api/lib" -Wl,-rpath,"${CANN_LIBRARY}" \
    -o "${RUNNER_BUILD_DIR}/sliding_tile_attention_22cases"

export ASCEND_CUSTOM_OPP_PATH="${CUSTOM_OPP_ROOT}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
export LD_LIBRARY_PATH="${CUSTOM_OPP_ROOT}/op_api/lib:${CANN_LIBRARY}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export SLIDING_TILE_ATTENTION_DEVICE_ID="${DEVICE_ID}"
export SLIDING_TILE_ATTENTION_SUITE="${SUITE}"
export SLIDING_TILE_ATTENTION_WARMUP="${SLIDING_TILE_ATTENTION_WARMUP:-3}"
export SLIDING_TILE_ATTENTION_MEASURE_REPEAT="${SLIDING_TILE_ATTENTION_MEASURE_REPEAT:-5}"
export SLIDING_TILE_ATTENTION_REPORT_PATH="${SLIDING_TILE_ATTENTION_REPORT_PATH:-${REPO_ROOT}/sliding_tile_attention_self_validation.csv}"
"${RUNNER_BUILD_DIR}/sliding_tile_attention_22cases"
echo "SLIDING_TILE_ATTENTION_SELF_VALIDATION_REPORT=${SLIDING_TILE_ATTENTION_REPORT_PATH}"
