#!/usr/bin/env bash
# KvCacheTurboQuant 一键自测：按 case.json 跑 5 个 case 的
#   ① 算子执行（计时）→ ② 与 golden.py 逐字节对拍 → ③ V3 双口径精度
#
# 用法:
#   export KCTQ_MATERIAL=<含 golden.py 的任务包目录>
#   export KCTQ_BIN=<构建出的 test_aclnn_kv_cache_turbo_quant 路径>   # 默认 /tmp/kctq_example/...
#   bash tests/run_all.sh
# 可选环境变量: KCTQ_WORK（工作目录，默认 tests/_out）、PYTHON、KCTQ_GOLDEN
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
OP_DIR=$(cd "$HERE/.." && pwd)
PY=${PYTHON:-python3}
MATERIAL=${KCTQ_MATERIAL:?"请设置 KCTQ_MATERIAL=<含 golden.py 的任务包目录>"}
GOLDEN=${KCTQ_GOLDEN:-$MATERIAL/golden.py}
BIN=${KCTQ_BIN:-/tmp/kctq_example/test_aclnn_kv_cache_turbo_quant}
WORK=${KCTQ_WORK:-$HERE/_out}

if [[ ! -x "$BIN" ]]; then
    echo "找不到可执行文件 $BIN" >&2
    echo "请先：bash examples/build_and_run.sh <算子包>/packages /tmp/kctq_example" >&2
    exit 2
fi
if [[ ! -f "$GOLDEN" ]]; then
    echo "找不到 golden.py：$GOLDEN（用 KCTQ_GOLDEN 指定）" >&2
    exit 2
fi

mkdir -p "$WORK/data" "$WORK/inputs" "$WORK/out"

# 1) 矩阵（旋转/投影）：任务包未随附，按固定种子生成（统计等价，sha256 见 data/meta.json）
if [[ ! -f "$WORK/data/meta.json" ]]; then
    "$PY" "$HERE/gen_test_data.py" --out "$WORK/data"
fi
ROT="$WORK/data/rotation_matrix.bin"
QJL="$WORK/data/qjl_matrix.bin"

# 2) case 列表来自 case.json（shape[0]=tokens、shape[1]=heads、attr mse_bits）
mapfile -t CASES < <("$PY" - "$HERE/case.json" <<'PY'
import json, sys
for c in json.load(open(sys.argv[1])):
    t, h, _ = c["input_desc"][0]["shape"]
    bits = c["attr_desc"][0]["value"]
    print(c["case_name"], t, h, bits)
PY
)

SUMMARY=()
for line in "${CASES[@]}"; do
    read -r name tokens heads bits <<<"$line"
    inp="$WORK/inputs/t${tokens}_h${heads}"
    if [[ ! -f "$inp/kv_vectors.bin" ]]; then
        mkdir -p "$inp"
        "$PY" "$HERE/gen_case_input.py" --out "$inp" --tokens "$tokens" --heads "$heads"
    fi
    out="$WORK/out/${name}_b${bits}"
    mkdir -p "$out"
    echo "===== ${name} (tokens=${tokens}, heads=${heads}, bits=${bits}) ====="
    "$BIN" --kv "$inp/kv_vectors.bin" --rot "$ROT" --qjl "$QJL" --out "$out" \
        --tokens "$tokens" --heads "$heads" --bits "$bits" --warmup 3 --repeat 10 \
        | tee "$out/run.log"
    "$PY" "$HERE/kctq_compare.py" --workdir "$out" --kv "$inp/kv_vectors.bin" --rot "$ROT" --qjl "$QJL" \
        --tokens "$tokens" --heads "$heads" --bits "$bits" --material "$MATERIAL" \
        | tee "$out/compare.log"
    "$PY" "$HERE/kctq_v3_precision.py" --workdir "$out" --kv "$inp/kv_vectors.bin" --rot "$ROT" --qjl "$QJL" \
        --tokens "$tokens" --heads "$heads" --bits "$bits" --golden "$GOLDEN" \
        --json "$out/v3.json" | tee "$out/v3.log"
    median=$(grep -o 'latency_ms_median: [0-9.]*' "$out/run.log" | awk '{print $2}')
    SUMMARY+=("$name bits=$bits median=${median}ms")
done

echo
echo "===== 汇总（计时为中位数，含 workspace 申请；对拍/V3 见各 case 日志）====="
printf '%s\n' "${SUMMARY[@]}"
echo "工作目录: $WORK"
