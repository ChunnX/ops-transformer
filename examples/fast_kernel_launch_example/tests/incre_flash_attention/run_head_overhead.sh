#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_ROOT="${1:-$SCRIPT_DIR/head_overhead_results}"
RUN_DIR="$OUTPUT_ROOT/run_$(date +%Y%m%d_%H%M%S)"
TIMING_CSV="$RUN_DIR/ifa_head_timing_raw.csv"
MERGED_CSV="$RUN_DIR/ifa_head_overhead.csv"
SUMMARY_CSV="$RUN_DIR/ifa_head_summary.csv"
HEAD_TIMING_POINT="${HEAD_TIMING_POINT:-mte2}"
HEAD_TIMING_LAUNCHES="${HEAD_TIMING_LAUNCHES:-250}"
HEAD_TIMING_WARMUP="${HEAD_TIMING_WARMUP:-50}"
HEAD_TIMING_SYSCNT_MHZ="${HEAD_TIMING_SYSCNT_MHZ:-1000}"

if ! command -v msprof >/dev/null 2>&1; then
    echo "msprof is not in PATH; source the CANN set_env.sh first" >&2
    exit 1
fi

mkdir -p "$RUN_DIR"
cd "$PROJECT_DIR"

HEAD_TIMING_LAUNCHES="$HEAD_TIMING_LAUNCHES" \
HEAD_TIMING_OUTPUT="$TIMING_CSV" \
HEAD_TIMING_POINT="$HEAD_TIMING_POINT" \
msprof \
    --output="$RUN_DIR" \
    --task-time=on \
    python3 "$SCRIPT_DIR/test.py"

PROFILE_DIR="$(find "$RUN_DIR" -maxdepth 1 -type d -name 'PROF_*' | sort | tail -1)"
if [[ -z "$PROFILE_DIR" ]]; then
    echo "No PROF_* directory found under $RUN_DIR" >&2
    exit 1
fi

python3 "$SCRIPT_DIR/analyze_head_overhead.py" \
    --profile-dir "$PROFILE_DIR" \
    --timing-csv "$TIMING_CSV" \
    --output "$MERGED_CSV" \
    --summary-output "$SUMMARY_CSV" \
    --warmup "$HEAD_TIMING_WARMUP" \
    --syscnt-mhz "$HEAD_TIMING_SYSCNT_MHZ"

chmod -R u+rwX,go+rX "$RUN_DIR"

echo "Raw timing: $TIMING_CSV"
echo "Merged timing: $MERGED_CSV"
echo "Summary: $SUMMARY_CSV"
