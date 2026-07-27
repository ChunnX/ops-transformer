#!/bin/bash

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
QBSA_BATCH_DIR="/home/cann/Desktop/FA/ops-transformer-FIA-rdvcase/experimental/attention/quant_block_sparse_attn/tests/pytest/batch"

show_help() {
    echo "用法: $0 <QBSA csv文件> [N] [归档目录]"
    echo
    echo "参数说明："
    echo "  QBSA csv文件  QBSA新增CSV文件名、相对路径或绝对路径"
    echo "  N             可选；执行第1到第N条数据行，不指定则执行全部"
    echo "  归档目录      默认 fia_prof_op_summary"
    echo
    echo "示例："
    echo "  $0 cases_fia_fullquant_mxfp8_perf_rdv.csv fia_prof_mxfp8"
    echo "  $0 cases_fia_fullquant_gqa_perf_rdv.csv 13 fia_prof_gqa"
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

resolve_output_dir() {
    local output_arg="$1"
    if [[ "$output_arg" = /* ]]; then
        echo "$output_arg"
    else
        echo "$SCRIPT_DIR/$output_arg"
    fi
}

csv_case_count() {
    python3 - "$1" <<'PY'
import csv
import sys
with open(sys.argv[1], "r", encoding="utf-8", newline="") as f:
    print(sum(1 for _ in csv.DictReader(f)))
PY
}

case_meta() {
    python3 - "$1" "$2" <<'PY'
import csv
import sys
csv_path = sys.argv[1]
case_index = int(sys.argv[2])
with open(csv_path, "r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
if case_index < 1 or case_index > len(rows):
    raise SystemExit(f"case index {case_index} out of range, total cases: {len(rows)}")
row = rows[case_index - 1]
print(f"{(row.get('case_name') or f'case_{case_index:06d}').strip()}\t{(row.get('enable') or 'TRUE').strip().upper()}\t{len(rows)}")
PY
}

sanitize_name() {
    printf "%s" "$1" | sed 's/[^A-Za-z0-9._-]/_/g'
}

latest_op_summary_after() {
    local marker="$1"
    find "$SCRIPT_DIR" \
        -path "$SCRIPT_DIR/PROF_*/mindstudio_profiler_output/op_summary_*.csv" \
        -type f -newer "$marker" -printf '%T@ %p\n' \
        | sort -n \
        | tail -1 \
        | cut -d' ' -f2-
}

append_manifest() {
    local manifest="$1"
    local index="$2"
    local case_name="$3"
    local status="$4"
    local op_summary="$5"
    local log_path="$6"
    python3 - "$manifest" "$index" "$case_name" "$status" "$op_summary" "$log_path" <<'PY'
import csv
import os
import sys
manifest, index, case_name, status, op_summary, log_path = sys.argv[1:]
exists = os.path.exists(manifest) and os.path.getsize(manifest) > 0
with open(manifest, "a", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    if not exists:
        writer.writerow(["index", "case_name", "status", "op_summary", "log"])
    writer.writerow([index, case_name, status, op_summary, log_path])
PY
}

run_one_case() {
    local csv_path="$1"
    local case_index="$2"
    local output_dir="$3"
    local manifest="$4"

    local meta case_name enable total_cases
    meta=$(case_meta "$csv_path" "$case_index") || {
        append_manifest "$manifest" "$case_index" "" "CSV_ERROR" "" ""
        return 1
    }
    IFS=$'\t' read -r case_name enable total_cases <<< "$meta"

    local case_file_name log_path summary_path
    case_file_name=$(sanitize_name "$case_name")
    log_path="$output_dir/${case_file_name}.log"
    summary_path="$output_dir/${case_file_name}.csv"

    if [[ "$enable" != "TRUE" ]]; then
        echo "第${case_index}/${total_cases}条：${case_name} enable=${enable}，跳过"
        append_manifest "$manifest" "$case_index" "$case_name" "DISABLED" "" "$log_path"
        return 0
    fi

    echo "===== 执行第${case_index}/${total_cases}条：${case_name} ====="
    echo "日志: $log_path"

    local marker
    marker=$(mktemp "$SCRIPT_DIR/.fia_prof_marker.XXXXXX")
    touch "$marker"

    (
        cd "$SCRIPT_DIR" || exit 1
        msprof bash test_run.sh batch_single "$csv_path" "$case_index"
    ) &> "$log_path"
    local run_status=$?

    local op_summary
    op_summary=$(latest_op_summary_after "$marker")
    python3 - "$marker" <<'PY'
import os
import sys
p = sys.argv[1]
if os.path.exists(p):
    os.unlink(p)
PY

    if [[ -z "$op_summary" ]]; then
        echo "第${case_index}条：未找到op_summary_*.csv，详见 $log_path"
        append_manifest "$manifest" "$case_index" "$case_name" "NO_OP_SUMMARY" "" "$log_path"
        return "$run_status"
    fi

    cp "$op_summary" "$summary_path"
    echo "第${case_index}条：op_summary已归档到 $summary_path"
    if [[ "$run_status" -eq 0 ]]; then
        append_manifest "$manifest" "$case_index" "$case_name" "PASS" "$summary_path" "$log_path"
    else
        append_manifest "$manifest" "$case_index" "$case_name" "RUN_FAILED" "$summary_path" "$log_path"
    fi
    return "$run_status"
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
    show_help
    exit 1
fi

CSV_PATH=$(resolve_csv_path "$1") || exit 1
TOTAL_CASES=$(csv_case_count "$CSV_PATH")
CASE_END="$TOTAL_CASES"
OUTPUT_ARG="fia_prof_op_summary"

if [[ $# -ge 2 ]]; then
    if [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
        CASE_END="$2"
        OUTPUT_ARG="${3:-fia_prof_op_summary}"
    else
        if [[ $# -eq 3 ]]; then
            echo "错误：第二个参数不是N时，不支持第三个参数" >&2
            show_help
            exit 1
        fi
        OUTPUT_ARG="$2"
    fi
fi

if [[ "$CASE_END" -gt "$TOTAL_CASES" ]]; then
    echo "错误：N=$CASE_END 超出CSV数据行数 $TOTAL_CASES" >&2
    exit 1
fi

OUTPUT_DIR=$(resolve_output_dir "$OUTPUT_ARG")
mkdir -p "$OUTPUT_DIR"
MANIFEST="$OUTPUT_DIR/summary_manifest.csv"
python3 - "$MANIFEST" "$OUTPUT_DIR/fused_infer_attention_score_summary.csv" <<'PY'
import os
import sys
for path in sys.argv[1:]:
    if os.path.exists(path):
        os.unlink(path)
PY

echo "pytest目录: $SCRIPT_DIR"
echo "CSV文件: $CSV_PATH"
echo "执行范围: 1..$CASE_END / CSV总行数: $TOTAL_CASES"
echo "归档目录: $OUTPUT_DIR"

failed_count=0
for ((case_index = 1; case_index <= CASE_END; case_index++)); do
    if ! run_one_case "$CSV_PATH" "$case_index" "$OUTPUT_DIR" "$MANIFEST"; then
        failed_count=$((failed_count + 1))
    fi
done

echo "批量执行完成，失败数量: $failed_count"
python3 "$SCRIPT_DIR/aggregate_fia_batch_single_prof.py" "$OUTPUT_DIR"
exit "$failed_count"
