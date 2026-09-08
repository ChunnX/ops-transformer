#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cann=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}
custom="$root/build/opp/vendors/custom_transformer/op_api/lib"
mkdir -p "$root/build/benchmarks"
c++ -O2 -std=c++17 "$root/benchmarks/benchmark_aclnn.cpp" \
    -I"$cann/include" -I"$root/op_api" \
    -L"$custom" -L"$cann/lib64" \
    -Wl,-rpath,"$custom" -Wl,-rpath,"$cann/lib64" \
    -lcust_opapi -lopapi -lnnopbase -lascendcl \
    -o "$root/build/benchmarks/benchmark_aclnn"
