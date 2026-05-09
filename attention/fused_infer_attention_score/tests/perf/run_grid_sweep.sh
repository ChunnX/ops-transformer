#!/usr/bin/env bash
# Orchestrate the FIA performance grid sweep:
#   per case → msprof op → perf_summary.py → msprof_pipeline_analyze.py → CSV row
# Failures in a single case are recorded as bound_type=ERROR and the sweep continues.
#
# Usage:
#   bash run_grid_sweep.sh --cases cases.json --out-dir results/run_<TS> \
#       [--warm-up 10] [--launch-count 3] [--device 0] \
#       [--perf-summary <path>] [--pipeline-analyze <path>]

set -uo pipefail

# ---- defaults ----
WARM_UP=10
LAUNCH_COUNT=3
DEVICE=0
CASES=""
OUT_DIR=""

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default skill script paths (overridable via flags / env).
PERF_SUMMARY="${PERF_SUMMARY:-/mnt/workspace/gitCode/cann/skills/ops/skills/ops-profiling/scripts/perf_summary.py}"
PIPELINE_ANALYZE="${PIPELINE_ANALYZE:-/mnt/workspace/gitCode/cann/skills/ops/skills/ascendc-performance-analysis/scripts/msprof_pipeline_analyze.py}"

usage() {
    sed -n '2,12p' "${BASH_SOURCE[0]}"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cases)             CASES="$2"; shift 2 ;;
        --out-dir)           OUT_DIR="$2"; shift 2 ;;
        --warm-up)           WARM_UP="$2"; shift 2 ;;
        --launch-count)      LAUNCH_COUNT="$2"; shift 2 ;;
        --device)            DEVICE="$2"; shift 2 ;;
        --perf-summary)      PERF_SUMMARY="$2"; shift 2 ;;
        --pipeline-analyze)  PIPELINE_ANALYZE="$2"; shift 2 ;;
        -h|--help)           usage ;;
        *) echo "Unknown arg: $1" >&2; usage ;;
    esac
done

[[ -z "$CASES"   ]] && { echo "Error: --cases is required" >&2; exit 2; }
[[ -z "$OUT_DIR" ]] && { echo "Error: --out-dir is required" >&2; exit 2; }
[[ ! -f "$CASES"           ]] && { echo "Error: cases file not found: $CASES" >&2; exit 2; }
[[ ! -f "$PERF_SUMMARY"    ]] && { echo "Error: perf_summary.py not found: $PERF_SUMMARY" >&2; exit 2; }
[[ ! -f "$PIPELINE_ANALYZE" ]] && { echo "Error: msprof_pipeline_analyze.py not found: $PIPELINE_ANALYZE" >&2; exit 2; }

mkdir -p "$OUT_DIR/raw"
CASES_ABS="$(cd "$(dirname "$CASES")" && pwd)/$(basename "$CASES")"
OUT_ABS="$(cd "$OUT_DIR" && pwd)"

# Snapshot the cases manifest into OUT_DIR for reproducibility.
cp "$CASES_ABS" "$OUT_ABS/cases.json"

CSV="$OUT_ABS/grid_results.csv"
LOG="$OUT_ABS/sweep.log"

# Header (overwrite any prior file).
python3 "$THIS_DIR/collect_row.py" --header > "$CSV"
: > "$LOG"

NUM_CASES=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$CASES_ABS")
echo "[sweep] $(date -Is) — sweeping $NUM_CASES cases → $OUT_ABS" | tee -a "$LOG"

ok=0; fail=0
ids=$(python3 -c "import json,sys; print(' '.join(str(c['case_id']) for c in json.load(open(sys.argv[1]))))" "$CASES_ABS")

export FIA_PROFILE_DEVICE="$DEVICE"

for cid in $ids; do
    case_tag="case_${cid}"
    case_raw="$OUT_ABS/raw/$case_tag"
    mkdir -p "$case_raw"

    case_dir="$OUT_ABS/docs/perf/$case_tag"

    echo "[sweep] ── $case_tag ──" | tee -a "$LOG"

    # 1) msprof op
    if ! msprof op \
            --warm-up="$WARM_UP" --launch-count="$LAUNCH_COUNT" \
            --output="$case_raw" \
            python3 "$THIS_DIR/case_driver.py" --cases "$CASES_ABS" --case-id "$cid" --device "$DEVICE" \
            >>"$LOG" 2>&1; then
        echo "[sweep]   msprof op FAILED" | tee -a "$LOG"
        mkdir -p "$case_dir"
        python3 "$THIS_DIR/collect_row.py" --case-dir "$case_dir" --case-id "$cid" --cases "$CASES_ABS" \
                --status ERROR --error-msg "msprof op failed; see sweep.log" >> "$CSV"
        fail=$((fail+1))
        continue
    fi

    # 2) Locate the OPPROF dir produced by msprof.
    opprof_dir="$(ls -td "$case_raw"/OPPROF_* 2>/dev/null | head -1)"
    if [[ -z "$opprof_dir" || ! -d "$opprof_dir" ]]; then
        echo "[sweep]   OPPROF dir not found under $case_raw" | tee -a "$LOG"
        mkdir -p "$case_dir"
        python3 "$THIS_DIR/collect_row.py" --case-dir "$case_dir" --case-id "$cid" --cases "$CASES_ABS" \
                --status ERROR --error-msg "OPPROF dir not produced" >> "$CSV"
        fail=$((fail+1))
        continue
    fi

    # 3) perf_summary.py — uses ops_dir = $OUT_ABS so output lands at $OUT_ABS/docs/perf/$case_tag/
    if ! python3 "$PERF_SUMMARY" "$opprof_dir" "$OUT_ABS" --round-name="$case_tag" \
            >>"$LOG" 2>&1; then
        echo "[sweep]   perf_summary.py FAILED" | tee -a "$LOG"
        mkdir -p "$case_dir"
        python3 "$THIS_DIR/collect_row.py" --case-dir "$case_dir" --case-id "$cid" --cases "$CASES_ABS" \
                --status ERROR --error-msg "perf_summary.py failed" >> "$CSV"
        fail=$((fail+1))
        continue
    fi

    # 4) msprof_pipeline_analyze.py
    if ! python3 "$PIPELINE_ANALYZE" "$case_dir" -o "$case_dir/bound.md" \
            >>"$LOG" 2>&1; then
        echo "[sweep]   msprof_pipeline_analyze.py FAILED" | tee -a "$LOG"
        python3 "$THIS_DIR/collect_row.py" --case-dir "$case_dir" --case-id "$cid" --cases "$CASES_ABS" \
                --status ERROR --error-msg "msprof_pipeline_analyze failed" >> "$CSV"
        fail=$((fail+1))
        continue
    fi

    # 5) Append CSV row.
    python3 "$THIS_DIR/collect_row.py" --case-dir "$case_dir" --case-id "$cid" --cases "$CASES_ABS" >> "$CSV"
    ok=$((ok+1))
    echo "[sweep]   OK ($ok ok / $fail fail)" | tee -a "$LOG"
done

echo "[sweep] done — $ok ok, $fail fail" | tee -a "$LOG"

# 6) Aggregate.
python3 "$THIS_DIR/aggregate_results.py" "$OUT_ABS" | tee -a "$LOG"

echo "[sweep] CSV     : $CSV"
echo "[sweep] Summary : $OUT_ABS/grid_summary.md"
