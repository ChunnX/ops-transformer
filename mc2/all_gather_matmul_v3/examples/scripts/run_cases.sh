#!/bin/bash
# run_cases.sh — AllGatherMatmulV3 torch golden test batch runner
#
# Reads a cases csv (columns: m,k,n,rank_num) and runs
# test_golden_allgathermm.py per row via qrun.
#
# 用法:
#   bash run_cases.sh                 # 跑 cases.csv 全部行
#   bash run_cases.sh 1 3 5           # 跑第 1/3/5 行
#   bash run_cases.sh 2-4             # 跑第 2~4 行
#   bash run_cases.sh --csv f.csv 1   # 指定 csv + 行号
#   EXTRA_ARGS="--dtype e5m2 --ydtype fp16" bash run_cases.sh 1
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV_FILE="${SCRIPT_DIR}/cases.csv"
# CSV_FILE="${SCRIPT_DIR}/cases_kehu.csv"
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --csv) CSV_FILE="$2"; shift 2 ;;
        -h|--help) sed -n '1,12p' "$0"; exit 0 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

DATA_LINES=$(grep -v '^#' "$CSV_FILE" | grep -v '^$' | tail -n +2)
TOTAL_LINES=$(echo "$DATA_LINES" | grep -c .)

expand_rows() {
    local result=()
    for item in "$@"; do
        if [ "$item" = "all" ]; then
            for ((i=1; i<=TOTAL_LINES; i++)); do result+=($i); done
        elif [[ "$item" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            for ((i=${BASH_REMATCH[1]}; i<=${BASH_REMATCH[2]}; i++)); do result+=($i); done
        elif [[ "$item" =~ ^[0-9]+$ ]]; then
            result+=("$item")
        fi
    done
    echo "${result[@]}"
}

if [ ${#ARGS[@]} -eq 0 ]; then
    SELECTED_ROWS=($(seq 1 $TOTAL_LINES))
else
    SELECTED_ROWS=($(expand_rows "${ARGS[@]}"))
fi

PASS_CNT=0; FAIL_CNT=0
RESULTS=()
for row in "${SELECTED_ROWS[@]}"; do
    SELECTED=$(echo "$DATA_LINES" | sed -n "${row}p")
    M=$(echo "$SELECTED" | awk -F, '{print $1}')
    K=$(echo "$SELECTED" | awk -F, '{print $2}')
    N=$(echo "$SELECTED" | awk -F, '{print $3}')
    RANK=$(echo "$SELECTED" | awk -F, '{print $4}')

    echo ""
    echo "########## row ${row}/${TOTAL_LINES}: M=$M K=$K N=$N rank=$RANK ##########"
    OUT=$(python3 test_golden_allgathermm.py --m $M --k $K --n $N --rank-per-dev $RANK ${EXTRA_ARGS:-} 2>&1)
    RC=$?
    echo "$OUT" | tail -8
    if [ $RC -eq 0 ] && echo "$OUT" | grep -q "\[PASS\] all ranks matched golden!"; then
        PASS_CNT=$((PASS_CNT + 1))
        RESULTS+=("PASS  row$row  M=$M K=$K N=$N rank=$RANK")
    else
        FAIL_CNT=$((FAIL_CNT + 1))
        RESULTS+=("FAIL  row$row  M=$M K=$K N=$N rank=$RANK")
    fi
done

echo ""
echo "=========================================="
echo "Summary: PASS=$PASS_CNT  FAIL=$FAIL_CNT  total=${#SELECTED_ROWS[@]}"
echo "=========================================="
for r in "${RESULTS[@]}"; do echo "  $r"; done
[ "$FAIL_CNT" -eq 0 ]
