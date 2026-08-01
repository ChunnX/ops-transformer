#!/bin/bash

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

show_help() {
    echo "用法: $0 <csv文件> [N] [归档目录]"
    echo
    echo "参数说明："
    echo "  csv文件   batch目录下的CSV文件名，或CSV绝对路径/相对路径"
    echo "  N         可选；批量执行第1到第N条CSV数据行（不包含表头），不指定则执行全部"
    echo "  归档目录  流水采集日志目录；相对路径按pytest目录解析，默认 pipe_timeline"
    echo
    echo "示例："
    echo "  $0 cases_fp8_9_perf.csv"
    echo "  $0 cases_fp8_9_perf.csv 3"
    echo "  $0 batch/cases_fp8_9_perf.csv 10 pipe_timeline_fp8"
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

case_meta() {
    local csv_path="$1"
    local case_index="$2"

    python3 - "$csv_path" "$case_index" <<'PY'
import csv
import sys

csv_path = sys.argv[1]
case_index = int(sys.argv[2])

with open(csv_path, "r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

if case_index < 1 or case_index > len(rows):
    print(f"case index {case_index} out of range, total cases: {len(rows)}", file=sys.stderr)
    sys.exit(2)

row = rows[case_index - 1]
case_name = (row.get("case_name") or f"case_{case_index:06d}").strip()
enable = (row.get("enable") or "TRUE").strip().upper()
print(f"{case_name}\t{enable}\t{len(rows)}")
PY
}

csv_case_count() {
    local csv_path="$1"

    python3 - "$csv_path" <<'PY'
import csv
import sys

with open(sys.argv[1], "r", encoding="utf-8", newline="") as f:
    print(sum(1 for _ in csv.DictReader(f)))
PY
}

sanitize_name() {
    local name="$1"
    printf "%s" "$name" | sed 's/[^A-Za-z0-9._-]/_/g'
}

append_manifest() {
    local manifest="$1"
    local index="$2"
    local case_name="$3"
    local status="$4"
    local log_path="$5"
    local trace_path="$6"

    python3 - "$manifest" "$index" "$case_name" "$status" "$log_path" "$trace_path" <<'PY'
import csv
import os
import sys

manifest, index, case_name, status, log_path, trace_path = sys.argv[1:]
exists = os.path.exists(manifest) and os.path.getsize(manifest) > 0
with open(manifest, "a", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    if not exists:
        writer.writerow(["index", "case_name", "status", "log", "trace_json"])
    writer.writerow([index, case_name, status, log_path, trace_path])
PY
}

latest_trace_after() {
    local marker="$1"

    find "$SCRIPT_DIR" \
        -path "$SCRIPT_DIR/OPPROF_*/trace.json" \
        -type f -newer "$marker" -printf '%T@ %p\n' \
        | sort -n \
        | tail -1 \
        | cut -d' ' -f2-
}

run_one_case() {
    local csv_path="$1"
    local case_index="$2"
    local output_dir="$3"
    local manifest="$4"

    local meta
    if ! meta=$(case_meta "$csv_path" "$case_index"); then
        echo "第${case_index}条：读取CSV失败，跳过"
        append_manifest "$manifest" "$case_index" "" "CSV_ERROR" "" ""
        return 1
    fi

    local case_name enable total_cases
    IFS=$'\t' read -r case_name enable total_cases <<< "$meta"
    local case_file_name
    case_file_name=$(sanitize_name "$case_name")
    local log_path="$output_dir/${case_file_name}.log"

    if [[ "$enable" != "TRUE" ]]; then
        echo "第${case_index}/${total_cases}条：${case_name} enable=${enable}，跳过"
        append_manifest "$manifest" "$case_index" "$case_name" "DISABLED" "$log_path" ""
        return 0
    fi

    echo "===== 执行第${case_index}/${total_cases}条：${case_name} ====="
    echo "日志: $log_path"

    local csv_key="${csv_path//\//_}"
    csv_key="${csv_key//,/_}"
    local testcase_dir="bsa_testcase/pipe_single_${csv_key}_${case_index}_$$"
    local marker
    marker=$(mktemp "$SCRIPT_DIR/.batch_single_pipe_marker.XXXXXX")
    touch "$marker"

    (
        cd "$SCRIPT_DIR" || exit 1
        export QBSA_BATCH_CSV="$csv_path"
        export QBSA_BATCH_CASE_INDEX="$case_index"
        export BSA_TESTCASE_PATH="$testcase_dir"
        export BSA_RESULT_PATH="result_pipe_single_${case_index}.xlsx"
        export BSA_SKIP_CPU_GOLDEN=1
        export BSA_SKIP_RESULT_COMPARE=1
        export BSA_PRINT_COMPUTE_LOG=1
        export BSA_PRINT_METADATA_AIC_BASIC_BLOCKS=1
        msprof op --aic-metrics=PipeTimeline bash run.sh batch
    ) &> "$log_path"
    local run_status=$?

    local trace_path
    trace_path=$(latest_trace_after "$marker")
    python3 - "$marker" <<'PY'
import os
import sys

path = sys.argv[1]
if os.path.exists(path):
    os.unlink(path)
PY

    local copied_trace=""
    if [[ -n "$trace_path" ]]; then
        copied_trace="$output_dir/${case_file_name}.json"
        cp "$trace_path" "$copied_trace"
        echo "第${case_index}条：trace.json已归档到 $copied_trace"
    else
        echo "第${case_index}条：未找到本次生成的OPPROF_*/trace.json，跳过trace归档"
    fi

    if [[ "$run_status" -eq 0 ]]; then
        append_manifest "$manifest" "$case_index" "$case_name" "PASS" "$log_path" "$copied_trace"
    else
        append_manifest "$manifest" "$case_index" "$case_name" "RUN_FAILED" "$log_path" "$copied_trace"
    fi
    return "$run_status"
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
    show_help
    exit 1
fi

CSV_ARG="$1"
CSV_PATH=$(resolve_csv_path "$CSV_ARG") || exit 1
TOTAL_CASES=$(csv_case_count "$CSV_PATH")

if [[ "$TOTAL_CASES" -lt 1 ]]; then
    echo "错误：CSV中没有可执行的数据行: $CSV_PATH" >&2
    exit 1
fi

CASE_END="$TOTAL_CASES"
OUTPUT_ARG="pipe_timeline"

if [[ $# -ge 2 ]]; then
    if [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
        CASE_END="$2"
        OUTPUT_ARG="${3:-pipe_timeline}"
    else
        if [[ $# -eq 3 ]]; then
            echo "错误：第二个参数不是N时，不支持第三个参数；当前第二个参数为 '$2'" >&2
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
MANIFEST="$OUTPUT_DIR/pipe_timeline_manifest.csv"
python3 - "$MANIFEST" <<'PY'
import os
import sys

path = sys.argv[1]
if os.path.exists(path):
    os.unlink(path)
PY

echo "pytest目录: $SCRIPT_DIR"
echo "CSV文件: $CSV_PATH"
echo "执行范围: 1..$CASE_END / CSV总行数: $TOTAL_CASES"
echo "日志目录: $OUTPUT_DIR"
echo "汇总清单: $MANIFEST"
echo "采集命令: msprof op --aic-metrics=PipeTimeline bash run.sh batch"

failed_count=0
for ((case_index = 1; case_index <= CASE_END; case_index++)); do
    if ! run_one_case "$CSV_PATH" "$case_index" "$OUTPUT_DIR" "$MANIFEST"; then
        failed_count=$((failed_count + 1))
    fi
done

echo "批量流水采集完成，失败数量: $failed_count"
exit "$failed_count"
