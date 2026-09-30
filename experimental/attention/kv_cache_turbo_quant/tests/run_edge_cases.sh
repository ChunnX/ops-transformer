#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 边界用例补充套件（tests/edge_cases.json）：与正式 5 case 相同口径的逐字节对拍（+ 可选 V3）。
#
# 用法:
#   export KCTQ_MATERIAL=<含 golden.py 的任务包目录>
#   export KCTQ_BIN=<构建出的 test_aclnn_kv_cache_turbo_quant 路径>
#   bash tests/run_edge_cases.sh
# 可选: KCTQ_WORK（默认 tests/_out）、PYTHON、KCTQ_GOLDEN
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python3}
MATERIAL=${KCTQ_MATERIAL:?"请设置 KCTQ_MATERIAL=<含 golden.py 的任务包目录>"}
GOLDEN=${KCTQ_GOLDEN:-$MATERIAL/golden.py}
BIN=${KCTQ_BIN:-/tmp/kctq_example/test_aclnn_kv_cache_turbo_quant}
WORK=${KCTQ_WORK:-$HERE/_out}

[[ -x "$BIN" ]] || { echo "找不到 $BIN（先 bash examples/build_and_run.sh <pkg>/packages）" >&2; exit 2; }
[[ -f "$GOLDEN" ]] || { echo "找不到 golden.py：$GOLDEN" >&2; exit 2; }

mkdir -p "$WORK/data" "$WORK/inputs" "$WORK/out"
[[ -f "$WORK/data/meta.json" ]] || "$PY" "$HERE/gen_test_data.py" --out "$WORK/data"
ROT="$WORK/data/rotation_matrix.bin"; QJL="$WORK/data/qjl_matrix.bin"

mapfile -t CASES < <("$PY" - "$HERE/edge_cases.json" <<'PY'
import json, sys
for c in json.load(open(sys.argv[1])):
    print(c["name"], c["tokens"], c["heads"], c["bits"], c.get("pattern", "uniform"))
PY
)

SUMMARY=(); FAILED=0
for line in "${CASES[@]}"; do
    read -r name tokens heads bits pattern <<<"$line"
    inp="$WORK/inputs/edge_${name}"; out="$WORK/out/edge_${name}"
    mkdir -p "$inp" "$out"
    echo "===== ${name} (tokens=${tokens}, heads=${heads}, bits=${bits}, pattern=${pattern}) ====="
    "$PY" "$HERE/gen_edge_input.py" --out "$inp" --tokens "$tokens" --heads "$heads" --pattern "$pattern" >/dev/null
    "$BIN" --kv "$inp/kv_vectors.bin" --rot "$ROT" --qjl "$QJL" --out "$out" \
        --tokens "$tokens" --heads "$heads" --bits "$bits" --warmup 3 --repeat 10 \
        | tee "$out/run.log" | grep -E "RESULT|latency_ms_median|label"
    compare_ok=1
    "$PY" "$HERE/kctq_compare.py" --workdir "$out" --kv "$inp/kv_vectors.bin" --rot "$ROT" --qjl "$QJL" \
        --tokens "$tokens" --heads "$heads" --bits "$bits" --material "$MATERIAL" \
        | tee "$out/compare.log" | grep -E "^\[V2\]|RESULT" || compare_ok=0
    grep -q "RESULT: PASS" "$out/compare.log" || { compare_ok=0; FAILED=1; }
    if [[ "$pattern" == "uniform" ]]; then
        "$PY" "$HERE/kctq_v3_precision.py" --workdir "$out" --kv "$inp/kv_vectors.bin" --rot "$ROT" --qjl "$QJL" \
            --tokens "$tokens" --heads "$heads" --bits "$bits" --golden "$GOLDEN" \
            --json "$out/v3.json" | tee "$out/v3.log" | grep -E "relMSE|内积 p95|判定" || true
    else
        echo "（pattern=${pattern}：跳过 V3 统计口径，逐字节对拍已覆盖该行内语义）"
    fi
    median=$(grep -o 'latency_ms_median: [0-9.]*' "$out/run.log" | awk '{print $2}')
    SUMMARY+=("$name bits=$bits median=${median}ms compare=$([[ $compare_ok == 1 ]] && echo PASS || echo FAIL)")
done

echo
echo "===== 边界用例汇总 ====="
printf '%s\n' "${SUMMARY[@]}"
echo "工作目录: $WORK"
exit "$FAILED"
