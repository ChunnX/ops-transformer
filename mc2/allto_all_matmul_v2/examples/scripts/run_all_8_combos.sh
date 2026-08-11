#!/bin/bash
# Run all 8 dtype combinations (or a single combo) for allto_all_matmul golden test
# Usage: qrun bash run_all_8_combos.sh [--case N] [--bs 1024] [--h 4096] [--n 8192]
#   --case 1   runs only combo 1 (1-indexed)
#   --case 0   runs all 8 (default)

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mc2_py310_mwx
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source /usr/local/Ascend/cann-9.1.0/vendors/custom_transformer/bin/set_env.bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PASS=0; FAIL=0; TOTAL=0
RESULTS=""
START_CASE=1; END_CASE=8

while [ $# -gt 0 ]; do
    case "$1" in
        --case) START_CASE="$2"; END_CASE="$2"; shift 2 ;;
        --bs|--h|--n|--rank-per-dev) EXTRA_ARGS="$EXTRA_ARGS $1 $2"; shift 2 ;;
        *) echo "Unknown: $1"; shift ;;
    esac
done

# Combo definitions: label + args
# 1: E4xE4 y=bf16   2: E5xE5 y=bf16   3: E4xE4 y=fp16   4: E5xE5 y=fp16
# 5: E4xE5 y=bf16   6: E5xE4 y=bf16   7: E4xE5 y=fp16   8: E5xE4 y=fp16
declare -A CMDS
CMDS[1]="--dtype e4m3fn --ydtype bf16"
CMDS[2]="--dtype e5m2    --ydtype bf16"
CMDS[3]="--dtype e4m3fn --ydtype fp16"
CMDS[4]="--dtype e5m2    --ydtype fp16"
CMDS[5]="--x1dtype e4m3fn --x2dtype e5m2 --ydtype bf16"
CMDS[6]="--x1dtype e5m2    --x2dtype e4m3fn --ydtype bf16"
CMDS[7]="--x1dtype e4m3fn --x2dtype e5m2 --ydtype fp16"
CMDS[8]="--x1dtype e5m2    --x2dtype e4m3fn --ydtype fp16"

run_one() {
    local label="$1"; shift
    TOTAL=$((TOTAL + 1))
    echo ""
    echo "============================================"
    echo "[$TOTAL/8] $label"
    echo "============================================"
    if python "$SCRIPT_DIR/test_golden_alltoallmm.py" "$@" $EXTRA_ARGS 2>&1; then
        echo "[PASS] $label"
        PASS=$((PASS + 1))
        RESULTS="$RESULTS\n[PASS] $label"
    else
        echo "[FAIL] $label"
        FAIL=$((FAIL + 1))
        RESULTS="$RESULTS\n[FAIL] $label"
    fi
}

echo "################################################################"
echo "# AlltoAllMatmulV2: dtype combinations precision test"
echo "# Cases: $START_CASE - $END_CASE"
echo "################################################################"

for i in $(seq $START_CASE $END_CASE); do
    run_one "case$i" ${CMDS[$i]}
done

echo ""
echo "################################################################"
echo "# FINAL RESULTS: $PASS / $TOTAL passed"
echo -e "$RESULTS"
echo "################################################################"