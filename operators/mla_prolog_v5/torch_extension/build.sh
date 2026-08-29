#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

set -e

BASE_PATH=$(
  cd "$(dirname $0)"
  pwd
)
BUILD_PATH="${BASE_PATH}/build"

mkdir -p "${BUILD_PATH}"
cd "${BUILD_PATH}"
TORCH_CMAKE_PREFIX=$(python3 -c "import torch; print(torch.__path__[0])")/share/cmake/Torch
cmake -DCMAKE_PREFIX_PATH="${TORCH_CMAKE_PREFIX}" ..
cmake --build . -- -j8

echo "[INFO] torch_extension built: ${BUILD_PATH}/libmla_prolog_v5_ext.so"
