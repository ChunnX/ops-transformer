#!/usr/bin/env bash
# Copyright (c) 2026 G_W_E. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail
op_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cann_root=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}
set +u
source "$cann_root/set_env.sh"
set -u
repo_dir=$(cd "$op_dir/../../.." && pwd)
export CATLASS_ROOT=${CATLASS_ROOT:-$repo_dir/../catlass}
cmake -S "$op_dir/native" -B "$op_dir/build/native" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCATLASS_ROOT="${CATLASS_ROOT}" \
  -DCMAKE_MODULE_PATH="$cann_root/tools/tikcpp/ascendc_kernel_cmake/asc_modules"
cmake --build "$op_dir/build/native" --parallel "${BUILD_JOBS:-4}"
