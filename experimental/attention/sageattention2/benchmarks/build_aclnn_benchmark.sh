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
if [ ! -d "${cann}" ] && [ -d "/usr/local/Ascend/ascend-toolkit/latest" ]; then
    cann="/usr/local/Ascend/ascend-toolkit/latest"
fi

custom="$root/build/opp/vendors/custom_transformer/op_api/lib"
out_dir="$root/build/benchmarks"
mkdir -p "$out_dir"

CXX_COMPILER="${CXX:-}"
if [ -z "${CXX_COMPILER}" ]; then
    if command -v clang++ >/dev/null 2>&1; then
        CXX_COMPILER="clang++"
    elif command -v g++ >/dev/null 2>&1; then
        CXX_COMPILER="g++"
    else
        CXX_COMPILER="c++"
    fi
fi

echo "================================================================="
echo "Building SageAttention2 Standalone ACLNN Benchmark Tool"
echo "Root directory:     ${root}"
echo "Compiler:           ${CXX_COMPILER}"
echo "Target binary:      ${out_dir}/benchmark_aclnn"
echo "================================================================="

if [ -f "${cann}/include/acl/acl.h" ]; then
    echo "Found CANN environment at ${cann}."
    echo "Compiling for Ascend NPU hardware runtime..."
    EXTRA_LIBS="-lascendcl -lopapi -lnnopbase -ldl"
    if [ -f "${custom}/libcust_opapi.so" ]; then
        EXTRA_LIBS="-lcust_opapi ${EXTRA_LIBS}"
    fi

    ${CXX_COMPILER} -O2 -std=c++17 -Wall -Wextra -Werror \
        "${root}/benchmarks/benchmark_aclnn.cpp" \
        -I"${cann}/include" \
        -I"${root}/aclnn" \
        -I"${root}/op_api" \
        -L"${custom}" -L"${cann}/lib64" \
        -Wl,-rpath,"${custom}" -Wl,-rpath,"${cann}/lib64" \
        ${EXTRA_LIBS} \
        -o "${out_dir}/benchmark_aclnn"
else
    echo "Notice: CANN environment not detected at ${cann}."
    echo "Compiling benchmark_aclnn in host mock/simulation mode..."

    ${CXX_COMPILER} -O2 -std=c++17 -Wall -Wextra -Werror \
        "${root}/benchmarks/benchmark_aclnn.cpp" \
        -I"${root}/aclnn" \
        -I"${root}/sageattn_fp8/ascendc" \
        -I"${root}/quant" \
        -o "${out_dir}/benchmark_aclnn"
fi

echo "Build successful! Generated binary: ${out_dir}/benchmark_aclnn"

# Execute smoke test verification
echo "-----------------------------------------------------------------"
echo "Executing self-check smoke test..."
# Exit code 2 means "SIMULATION run" (mock build without CANN): the tool ran fine
# but refuses to emit PASS/FAIL evidence; anything other than 0 or 2 is a failure.
set +e
"${out_dir}/benchmark_aclnn" --smoke-test --output "${out_dir}/smoke_test.json"
smoke_rc=$?
set -e
if [ "${smoke_rc}" -ne 0 ] && [ "${smoke_rc}" -ne 2 ]; then
    echo "Smoke test failed with exit code ${smoke_rc}"
    exit "${smoke_rc}"
fi
if [ "${smoke_rc}" -eq 2 ]; then
    echo "Smoke test ran in SIMULATION mode (exit 2): report carries NaN speedup, no verdict."
fi
echo "Smoke test passed successfully! Generated: ${out_dir}/smoke_test.json"
echo "================================================================="
