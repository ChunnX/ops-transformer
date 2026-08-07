#!/bin/bash
# Batch profiling script for AllGatherQuantMatmul Prefill.
# Each msprof run produces 4 PROF dirs (one per rank).
# After each run, write case_info.txt + case_precision.txt into new dirs.
# If latency > threshold OR precision FAIL, delete and retry (max 3 attempts).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROF_DIR="$SCRIPT_DIR/prof"
PARSE_SCRIPT="$SCRIPT_DIR/scripts/parse_prof.py"
VERIFY_SCRIPT="$SCRIPT_DIR/scripts/verify_result.py"
PYTHON=/root/miniconda3/bin/python3
LATENCY_THRESHOLD=50000
MAX_RETRIES=3
RANK_NUM=4

# Ensure CANN environment
[ -z "$ASCEND_TOOLKIT_HOME" ] && source /home/h00951335/Ascend/cann/set_env.sh 2>/dev/null || true

# ---- helpers ----
get_existing_dirs() {
    find "$PROF_DIR" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null | sort || true
}

get_new_dirs() {
    local before="$1" after="$2"
    local new=""
    for d in $after; do
        if ! echo "$before" | grep -qFx "$d"; then
            new="$new $d"
        fi
    done
    echo "$new" | xargs
}

delete_dirs() {
    for d in $1; do
        rm -rf "$PROF_DIR/$d"
        echo "  [Deleted $d]"
    done
}

# Run one profiling attempt. Machine-readable result on the LAST line.
do_one_attempt() {
    local M=$1 K=$2 N=$3 ATTEMPT=$4 SKIP_BUILD=$5

    local skip_flag=""
    [ "$SKIP_BUILD" = "1" ] && skip_flag="--skip-build"

    echo "  [Attempt $ATTEMPT/$MAX_RETRIES] M=$M K=$K N=$N"

    # Clean stale output dirs so verify_result.py finds fresh data
    rm -rf "$SCRIPT_DIR/output"

    local before after new_dirs

    mkdir -p "$PROF_DIR"
    before=$(get_existing_dirs)
    msprof --output="$PROF_DIR" --application="bash run.sh --cli $M $K $N $RANK_NUM $skip_flag" 2>&1 | tail -5
    after=$(get_existing_dirs)

    new_dirs=$(get_new_dirs "$before" "$after")
    if [ -z "$new_dirs" ]; then
        echo "  [ERROR] No new PROF dirs created"
        echo "-1"
        return
    fi

    # Write case_info.txt into each new PROF dir
    for d in $new_dirs; do
        echo "M=$M K=$K N=$N" > "$PROF_DIR/$d/case_info.txt"
    done

    # Parse latency via parse_prof.py
    local parse_out latency
    parse_out=$("$PYTHON" "$PARSE_SCRIPT" --check-latest-threshold 999999999 --mkn "$M $K $N" --prof-dir "$PROF_DIR" 2>/dev/null)
    latency=$(echo "$parse_out" | grep -E '^[0-9]+\.[0-9]+$' | head -1)

    if [ -z "$latency" ]; then
        echo "  [ERROR] Cannot parse latency"
        echo "-1 $new_dirs"
        return
    fi

    # Check precision — run.sh copies output to $SCRIPT_DIR/output/
    local precision="FAIL"
    if [ -d "$SCRIPT_DIR/output" ]; then
        precision=$("$PYTHON" "$VERIFY_SCRIPT" "$M" "$N" "$RANK_NUM" "$SCRIPT_DIR/output" --check 2>/dev/null) || precision="FAIL"
    fi

    # Write precision into each new PROF dir
    for d in $new_dirs; do
        echo "$precision" > "$PROF_DIR/$d/case_precision.txt"
    done

    echo "  [latency=${latency}us, precision=$precision, dirs=$new_dirs]"
    # LAST LINE: machine-readable result
    echo "$latency $new_dirs $precision"
}

# Main: run one case with retry
test_case() {
    local M=$1 K=$2 N=$3
    local attempt=0
    local skip_build=0

    # If binary already exists, skip build from the start
    if [ -f "$SCRIPT_DIR/../build/all_gather_quant_matmul/apace_ag_qmm_st" ]; then
        skip_build=1
    fi

    while [ $attempt -lt $MAX_RETRIES ]; do
        attempt=$((attempt + 1))
        echo ""
        echo "===== Case M=$M K=$K N=$N | Attempt $attempt/$MAX_RETRIES ====="

        local all_output result_line
        all_output=$(do_one_attempt "$M" "$K" "$N" "$attempt" "$skip_build")

        # First run built the binary, subsequent runs skip build
        skip_build=1

        # Last line is the machine-readable result
        result_line=$(echo "$all_output" | tail -1)
        # Print diagnostic lines (everything except last)
        echo "$all_output" | head -n -1

        local latency new_dirs precision
        latency=$(echo "$result_line" | awk '{print $1}')
        precision=$(echo "$result_line" | awk '{print $NF}')
        new_dirs=$(echo "$result_line" | awk '{$1=""; $NF=""; print}' | xargs)

        # Check for errors
        if [ "$latency" = "-1" ] || [ -z "$latency" ]; then
            echo "  [RETRY] Attempt $attempt failed with error"
            [ -n "$new_dirs" ] && delete_dirs "$new_dirs"
            continue
        fi

        # Check precision
        if [ "$precision" != "PASS" ]; then
            echo "  [RETRY] Precision FAILED (latency=${latency}us)"
            delete_dirs "$new_dirs"
            continue
        fi

        # Check latency threshold
        local exceeds
        exceeds=$(awk "BEGIN { print ($latency > $LATENCY_THRESHOLD) ? 1 : 0 }")
        if [ "$exceeds" = "1" ]; then
            echo "  [RETRY] Latency ${latency}us > threshold ${LATENCY_THRESHOLD}us"
            delete_dirs "$new_dirs"
            continue
        fi

        echo "  [OK] Latency ${latency}us <= ${LATENCY_THRESHOLD}us, Precision PASS"
        return 0
    done

    echo ""
    echo "[FATAL] M=$M K=$K N=$N failed after $MAX_RETRIES attempts"
    exit 1
}

# ---- Profiling cases ----
echo "Threshold: ${LATENCY_THRESHOLD}us | Max retries: ${MAX_RETRIES}"
echo ""

test_case 1010 4096 2560
test_case 1010 4096 6656
test_case 2080 4096 2560
test_case 2080 4096 6656
test_case 2304 4096 2560
test_case 2304 4096 6656
test_case 2624 4096 2560
test_case 3136 4096 2560
test_case 3136 4096 6656
test_case 4040 4096 2560
test_case 4040 4096 6656

echo ""
echo "Done. Run: python3 scripts/parse_prof.py --all"