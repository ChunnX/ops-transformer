#!/bin/bash

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
QBSA_BATCH_DIR="/home/cann/Desktop/FA/ops-transformer-FIA-rdvcase/experimental/attention/quant_block_sparse_attn/tests/pytest/batch"

show_help() {
    echo "用法: $0 batch_single <csv文件> <i>"
    echo
    echo "参数说明："
    echo "  csv文件  QBSA新增CSV文件名、相对路径或绝对路径"
    echo "  i        CSV数据行号，从1开始，不包含表头"
}

resolve_csv_path() {
    local csv_arg="$1"

    if [[ "$csv_arg" = /* && -f "$csv_arg" ]]; then
        realpath "$csv_arg"
        return 0
    fi
    if [[ -f "$SCRIPT_DIR/$csv_arg" ]]; then
        realpath "$SCRIPT_DIR/$csv_arg"
        return 0
    fi
    if [[ -f "$SCRIPT_DIR/batch/$csv_arg" ]]; then
        realpath "$SCRIPT_DIR/batch/$csv_arg"
        return 0
    fi
    if [[ -f "$QBSA_BATCH_DIR/$csv_arg" ]]; then
        realpath "$QBSA_BATCH_DIR/$csv_arg"
        return 0
    fi

    echo "错误：找不到CSV文件 '$csv_arg'" >&2
    return 1
}

sanitize_name() {
    local name="$1"
    printf "%s" "$name" | sed 's/[^A-Za-z0-9._-]/_/g'
}

case_info() {
    local csv_path="$1"
    local case_index="$2"

    python3 - "$SCRIPT_DIR" "$csv_path" "$case_index" <<'PY'
import csv
import importlib
import math
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
case_index = int(sys.argv[3])

with csv_path.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
if case_index < 1 or case_index > len(rows):
    raise SystemExit(f"case index {case_index} out of range, total cases: {len(rows)}")
row = rows[case_index - 1]
case_name = (row.get("case_name") or f"case_{case_index:06d}").strip()
enable = (row.get("enable") or "TRUE").strip().upper()

sys.path.insert(0, str(script_dir / "fia_fullquant_mxfp8_test"))
sys.path.insert(0, str(script_dir / "fia_fullquant_gqa_test"))
mx = importlib.import_module("fia_fullquant_mxfp8_paramset_perf_rdv")
gqa = importlib.import_module("fia_fullquant_gqa_paramset_perf_rdv")
mx_names = {case["name"] for case in mx.CASES}
gqa_names = {case["name"] for case in gqa.CASES}
if case_name in mx_names:
    suite = "mxfp8"
elif case_name in gqa_names:
    suite = "gqa"
else:
    raise SystemExit(f"case '{case_name}' not found in FIA mxfp8/gqa perf paramsets")

def int_col(name, default=0):
    value = row.get(name)
    return int(value) if value not in (None, "") else default

batch = int_col("B")
n1 = int_col("N1")
s1 = int_col("S1")
s2 = int_col("S2")
head_dim = int_col("D")
q_block = int_col("sparse_q_block_size", 128)
kv_block = int_col("sparse_kv_block_size", 128)
sparse_count = int_col("sparse_count", math.ceil(s2 / kv_block))
mask_mode = int_col("mask_mode", 3)

def allowed_blocks(qb_idx):
    block_num = math.ceil(s2 / kv_block)
    if mask_mode == 0:
        return block_num
    if mask_mode != 3:
        raise SystemExit(f"unsupported mask_mode for FLOPS estimate: {mask_mode}")
    max_token = (qb_idx + 1) * q_block - 1 + s2 - s1
    if max_token < 0:
        return 0
    return min(block_num, max_token // kv_block + 1)

q_blocks = math.ceil(s1 / q_block)
basic_block_count = 0
for qb_idx in range(q_blocks):
    basic_block_count += min(sparse_count, allowed_blocks(qb_idx))
basic_block_count *= batch * n1

single_basic_block_compute = q_block * kv_block * head_dim
cube_compute_amount = basic_block_count * single_basic_block_compute * 2 * 2
cube_compute_capacity = 1000 * (16 * 16 * 32) * 1.65 * 32 * 2
mfu_time = cube_compute_amount / cube_compute_capacity

print(f"{suite}\t{case_name}\t{enable}\t{cube_compute_amount}\t{mfu_time:.6f}\t"
      f"{batch}\t{n1}\t{int_col('N2')}\t{s1}\t{s2}\t{head_dim}\t{sparse_count}\t"
      f"{basic_block_count}\t{single_basic_block_compute}")
PY
}

print_compute_log() {
    local case_name="$1"
    local flops="$2"
    local mfu_time="$3"
    local batch="$4"
    local n1="$5"
    local n2="$6"
    local s1="$7"
    local s2="$8"
    local head_dim="$9"
    local sparse_count="${10}"
    local basic_block_count="${11}"
    local single_block_compute="${12}"

    echo "case_name=$case_name"
    echo "FLOPS计算过程量:"
    echo "基本块数量: $basic_block_count"
    echo "单基本块计算量: $single_block_compute"
    echo "FLOPS计算公式: 基本块数($basic_block_count) * 单基本块计算量($single_block_compute) * 乘加计算(2) * 两次bmm计算(2) = $flops"
    echo "MFU*时间计算公式: FLOPS($flops) / 算力(865075200.000000) = MFU * 时间(us) = $mfu_time"
    echo "FLOPS=$flops"
    echo "MFU*时间(us)=$mfu_time"
    echo "shape: B=$batch, N1=$n1, N2=$n2, S1=$s1, S2=$s2, D=$head_dim, sparse_count=$sparse_count"
}

run_batch_single() {
    if [[ $# -ne 2 ]]; then
        show_help
        exit 1
    fi

    local csv_path case_index info
    csv_path=$(resolve_csv_path "$1") || exit 1
    case_index="$2"
    if ! [[ "$case_index" =~ ^[1-9][0-9]*$ ]]; then
        echo "错误：i必须是正整数，当前为 '$case_index'" >&2
        exit 1
    fi

    info=$(case_info "$csv_path" "$case_index") || exit 1
    local suite case_name enable flops mfu_time batch n1 n2 s1 s2 head_dim sparse_count basic_block_count single_block_compute
    IFS=$'\t' read -r suite case_name enable flops mfu_time batch n1 n2 s1 s2 head_dim sparse_count basic_block_count single_block_compute <<< "$info"
    if [[ "$enable" != "TRUE" ]]; then
        echo "case_name=$case_name enable=$enable，跳过"
        exit 0
    fi

    print_compute_log "$case_name" "$flops" "$mfu_time" "$batch" "$n1" "$n2" "$s1" "$s2" "$head_dim" \
        "$sparse_count" "$basic_block_count" "$single_block_compute"

    local safe_name cache_dir
    safe_name=$(sanitize_name "$case_name")
    cache_dir="$SCRIPT_DIR/.fia_batch_single_cache/${suite}_${safe_name}_${case_index}_$$"
    mkdir -p "$cache_dir"

    if [[ "$suite" == "mxfp8" ]]; then
        (
            cd "$SCRIPT_DIR/fia_fullquant_mxfp8_test" || exit 1
            python3 -m pytest -rA -s test_fia_fullquant_mxfp8_perf_rdv.py -v -m perf_rdv \
                --golden-mode=gen,npu --cache-dir "$cache_dir" -k "$case_name" \
                -W ignore::UserWarning -W ignore::DeprecationWarning
        )
    else
        (
            cd "$SCRIPT_DIR/fia_fullquant_gqa_test" || exit 1
            python3 -m pytest -rA -s test_fia_fullquant_gqa_perf_rdv.py -v -m perf_rdv \
                --golden-mode=gen,npu --cache-dir "$cache_dir" -k "$case_name" \
                -W ignore::UserWarning -W ignore::DeprecationWarning
        )
    fi
}

if [[ $# -lt 1 ]]; then
    show_help
    exit 1
fi

case "$1" in
    batch_single)
        shift
        run_batch_single "$@"
        ;;
    help|-h|--help)
        show_help
        ;;
    *)
        echo "错误：未知参数 '$1'" >&2
        show_help
        exit 1
        ;;
esac
