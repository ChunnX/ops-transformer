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

# 1) 矩阵（旋转/投影）：
#    - 有官方共享件时优先用它（KCTQ_MATRIX_DIR，或 $KCTQ_OFFICIAL_DATA/shared）——
#      交付口径固定用官方件（sha256 见 2.2 日志抬头）；
#    - 否则按固定种子生成统计等价矩阵（seed 见 meta.json）。
MATDIR=${KCTQ_MATRIX_DIR:-}
if [[ -z "$MATDIR" && -n "${KCTQ_OFFICIAL_DATA:-}" && -f "$KCTQ_OFFICIAL_DATA/shared/rotation_matrix.bin" ]]; then
    MATDIR="$KCTQ_OFFICIAL_DATA/shared"
fi
if [[ -z "$MATDIR" ]]; then
    if [[ ! -f "$WORK/data/meta.json" ]]; then
        "$PY" "$HERE/gen_test_data.py" --out "$WORK/data"
    fi
    MATDIR="$WORK/data"
fi
ROT="$MATDIR/rotation_matrix.bin"
QJL="$MATDIR/qjl_matrix.bin"

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
    # 输入解析优先级（官方件 per-case 优先，避免多个同 shape 的 case 共用一份输入）：
    #   1) $KCTQ_OFFICIAL_DATA/<case_name>/{kv_vectors.bin,ip_query.bin}
    #   2) $WORK/inputs/<case_name>/{kv_vectors.bin,ip_query.bin}
    #   3) $WORK/inputs/t<tokens>_h<heads>/kv_vectors.bin（历史布局，兼容旧工作目录）
    #   4) 由 gen_case_input.py 生成（固定 seed，统计等价）
    inp="$WORK/inputs/$name"
    ipq=""
    off="${KCTQ_OFFICIAL_DATA:-}"
    if [[ -n "$off" && -f "$off/$name/kv_vectors.bin" ]]; then
        inp="$off/$name"
    elif [[ -f "$inp/kv_vectors.bin" ]]; then
        :
    elif [[ -f "$WORK/inputs/t${tokens}_h${heads}/kv_vectors.bin" ]]; then
        inp="$WORK/inputs/t${tokens}_h${heads}"
    else
        mkdir -p "$inp"
        "$PY" "$HERE/gen_case_input.py" --out "$inp" --tokens "$tokens" --heads "$heads"
    fi
    [[ -f "$inp/ip_query.bin" ]] && ipq="$inp/ip_query.bin"
    out="$WORK/out/${name}_b${bits}"
    mkdir -p "$out"
    echo "===== ${name} (tokens=${tokens}, heads=${heads}, bits=${bits}) ====="
    echo "input=${inp}/kv_vectors.bin sha256=$(sha256sum "$inp/kv_vectors.bin" | cut -d' ' -f1)"
    [[ -n "$ipq" ]] && echo "ip_query=${ipq} sha256=$(sha256sum "$ipq" | cut -d' ' -f1)"
    "$BIN" --kv "$inp/kv_vectors.bin" --rot "$ROT" --qjl "$QJL" --out "$out" \
        --tokens "$tokens" --heads "$heads" --bits "$bits" --warmup 3 --repeat 10 \
        | tee "$out/run.log"
    "$PY" "$HERE/kctq_compare.py" --workdir "$out" --kv "$inp/kv_vectors.bin" --rot "$ROT" --qjl "$QJL" \
        --tokens "$tokens" --heads "$heads" --bits "$bits" --material "$MATERIAL" \
        | tee "$out/compare.log"
    V3EXTRA=()
    [[ -n "$ipq" ]] && V3EXTRA=(--ip-query "$ipq")
    "$PY" "$HERE/kctq_v3_precision.py" --workdir "$out" --kv "$inp/kv_vectors.bin" --rot "$ROT" --qjl "$QJL" \
        --tokens "$tokens" --heads "$heads" --bits "$bits" "${V3EXTRA[@]}" --golden "$GOLDEN" \
        --json "$out/v3.json" | tee "$out/v3.log"
    # 4) 生态算子开源精度标准的混合容差判定（补充层，不替代任务书 §3.2 指标）
    "$PY" "$HERE/kctq_std_tolerance.py" --workdir "$out" --kv "$inp/kv_vectors.bin" --rot "$ROT" --qjl "$QJL" \
        --tokens "$tokens" --heads "$heads" --bits "$bits" --material "$MATERIAL" \
        --json "$out/std_tolerance.json" | tee "$out/std_tolerance.log"
    median=$(grep -o 'latency_ms_median: [0-9.]*' "$out/run.log" | awk '{print $2}')
    SUMMARY+=("$name bits=$bits median=${median}ms")
done

echo
echo "===== 汇总（计时为中位数，含 workspace 申请；对拍/V3 见各 case 日志）====="
printf '%s\n' "${SUMMARY[@]}"
echo "工作目录: $WORK"
