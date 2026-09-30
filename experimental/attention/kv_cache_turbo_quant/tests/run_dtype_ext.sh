#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 接口扩展功能回归：矩阵 {fp32,bf16} × metadata {bf16,fp16} × 3 个 shape（+ 2 个额外位宽）。
#
# 官方口径（2026-09-30 Q7）：正式五用例固定 kv=BF16/矩阵=FP32/metadata=BF16；
# 矩阵 BF16 与 metadata FP16 属接口扩展，需通过功能回归、不进性能门槛。
#
# 用法:
#   export KCTQ_MATERIAL=<含 golden.py 的任务包目录>
#   export KCTQ_BIN=<构建出的 test_aclnn_kv_cache_turbo_quant>
#   bash tests/run_dtype_ext.sh [输出根目录]
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
OP_DIR=$(cd "$HERE/.." && pwd)
PY=${PYTHON:-python3}
MATERIAL=${KCTQ_MATERIAL:?"请设置 KCTQ_MATERIAL=<含 golden.py 的任务包目录>"}
BIN=${KCTQ_BIN:?"请设置 KCTQ_BIN=<test_aclnn_kv_cache_turbo_quant 路径>"}
WORK=${1:-${KCTQ_WORK:-$HERE/_out}/dtype_ext}
DATA=${KCTQ_OFFICIAL_DATA:?"请设置 KCTQ_OFFICIAL_DATA=<官方 generate_data.py 输出根>"}

# 矩阵：任务包未随附，与 run_all.sh 共用同一份按固定种子生成的数据（sha256 见 meta.json），
# 这样扩展回归和正式回归面对的是完全相同的旋转 / 投影矩阵。
MATDIR=${KCTQ_MATRIX_DIR:-$HERE/_out/data}
mkdir -p "$MATDIR"
if [[ ! -f "$MATDIR/meta.json" ]]; then
  "$PY" "$HERE/gen_test_data.py" --out "$MATDIR"
fi
ROT="$MATDIR/rotation_matrix.bin"
QJL="$MATDIR/qjl_matrix.bin"

mkdir -p "$WORK"
fail=0

# case_name tokens heads bits
CASES=(
  "gqa_decode_b1 1 8 3"
  "gqa_decode_b64 64 8 3"
  "gqa_prefill_t2048 2048 8 3"
  "gqa_prefill_t2048 2048 8 2"
  "gqa_prefill_t2048 2048 8 4"
)
COMBOS=("fp32 bf16" "bf16 bf16" "fp32 fp16" "bf16 fp16")

for combo in "${COMBOS[@]}"; do
  read -r mat meta <<<"$combo"
  for row in "${CASES[@]}"; do
    read -r name tokens heads bits <<<"$row"
    out="$WORK/${name}_b${bits}_${mat}_${meta}"
    mkdir -p "$out"
    cp -f "$DATA/$name/kv_vectors.bin" "$out/kv_vectors.bin"
    "$BIN" --kv "$out/kv_vectors.bin" --rot "$ROT" \
           --qjl "$QJL" --out "$out" \
           --tokens "$tokens" --heads "$heads" --bits "$bits" \
           --matrix-dtype "$mat" --meta-dtype "$meta" \
           --warmup 2 --repeat 5 --label "${name}_b${bits}_${mat}_${meta}" \
           >"$out/run.log" 2>&1
    if ! grep -q "RESULT: PASS" "$out/run.log"; then
      echo "!! 驱动失败：${name}_b${bits}_${mat}_${meta}（见 $out/run.log）"
      fail=1
      continue
    fi
    "$PY" "$HERE/kctq_dtype_ext.py" --workdir "$out" --tokens "$tokens" --heads "$heads" \
        --bits "$bits" --matrix-dtype "$mat" --meta-dtype "$meta" \
        --material "$MATERIAL" --label "${name}_b${bits}_${mat}_${meta}" \
        --json "$out/ext.json" || fail=1
  done
done

echo
if [ "$fail" -eq 0 ]; then
  echo "DTYPE EXT REGRESSION: PASS（4 组合 × 5 case）"
else
  echo "DTYPE EXT REGRESSION: FAIL"
fi
exit "$fail"
