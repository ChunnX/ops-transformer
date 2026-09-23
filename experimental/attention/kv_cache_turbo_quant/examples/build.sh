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

# Build and run the verified KvCacheTurboQuant encode self-test.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CANN="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.1.0}"
if [[ -f "$CANN/set_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "$CANN/set_env.sh"
fi

INC=(
    -I"$CANN/aarch64-linux/ascendc/include/basic_api/interface"
    -I"$CANN/aarch64-linux/ascendc/include/basic_api/impl"
    -I"$CANN/aarch64-linux/ascendc/include/basic_api"
    -I"$CANN/aarch64-linux/ascendc/include/highlevel_api"
    -I"$CANN/aarch64-linux/ascendc/include/highlevel_api/lib"
    -I"$CANN/aarch64-linux/asc"
    -I"$CANN/aarch64-linux/tikcpp/tikcfw"
    -I"$CANN/include"
    -I"$CANN/pkg_inc"
    -I"$CANN/aarch64-linux/ascendc/include"
)

OUT="${1:-/tmp/kv_cache_turbo_quant_test}"
VEC_O="${OUT}.vec.o"
CUBE_O="${OUT}.cube.o"

bisheng -xcce --cce-aicore-arch=dav-c220-vec -O2 -std=c++17 "${INC[@]}" \
    -c "$ROOT/op_kernel/kv_cache_turbo_quant.cpp" -o "$VEC_O"
bisheng -xcce --cce-aicore-arch=dav-c220-cube -DASCENDC_CUBE_ONLY -O2 -std=c++17 "${INC[@]}" \
    -c "$ROOT/op_kernel/kv_cache_turbo_quant_cube.cpp" -o "$CUBE_O"
g++ -O2 -std=c++17 -I"$CANN/include" -I"$CANN/pkg_inc" -I"$CANN/aarch64-linux/include" \
    "$ROOT/examples/test_aclnn_kv_cache_turbo_quant.cpp" "$VEC_O" "$CUBE_O" \
    -Wl,--allow-multiple-definition \
    -L"$CANN/lib64" -Wl,--start-group \
    -ltiling_api -lplatform -lgraph -lgraph_base -lregister \
    -lascendcl -lruntime -lprofapi -lerror_manager -lunified_dlog -lc_sec \
    -lopapi -lnnopbase \
    -ldl -lpthread -lstdc++ -lm -Wl,--end-group \
    -o "$OUT"

echo "built $OUT"
if [[ "${KVTQ_BUILD_ONLY:-0}" != "1" ]]; then
    DUMP="${KVTQ_DUMP:-/tmp/kvtq_dump}"
    rm -rf "$DUMP"
    export KVTQ_DUMP="$DUMP"
    ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}" "$OUT"
    PY="${PYTHON:-/usr/local/python3.12.13/bin/python3}"
    "$PY" "$ROOT/tests/compare_golden.py" "$DUMP"
fi
