#!/usr/bin/env bash
# Copyright (c) 2026, edconeone
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Adapted from DeepEP; original copyright and license notices follow.

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
set -euo pipefail
operator_dir="$1"
shift
if [[ "${1:-}" == "--help" ]]; then
    printf '%s\n' 'Build the standalone Ascend950 operator and its Python bindings.' \
        'Usage: bash build.sh [additional CMake -D options]' \
        'Required environment: ASCEND_HOME_PATH, SHMEM_ROOT' \
        'Optional: PYTHON, MC2_BUILD_DIR, MC2_INSTALL_DIR, CMAKE_BUILD_PARALLEL_LEVEL'
    exit 0
fi
: "${ASCEND_HOME_PATH:?Source the CANN environment first}"
: "${SHMEM_ROOT:?Select the pinned SHMEM SDK; see common/deepep/README.md}"
build_dir="${MC2_BUILD_DIR:-${operator_dir}/build}"
install_dir="${MC2_INSTALL_DIR:-${operator_dir}/output}"
python_bin="${PYTHON:-python3}"
cmake -S "${operator_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${install_dir}" \
    -DPython_EXECUTABLE="$(command -v "${python_bin}")" "$@"
cmake --build "${build_dir}" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
cmake --install "${build_dir}"
printf 'Installed Python modules: %s/python\n' "${install_dir}"
