#!/bin/bash
# run_one_msprof.sh — 单 case msprof 采集 (逐个排队用)
# 用法: bash run_one_msprof.sh <M> <K> <N> <R> <HEADM>
set -e
M=$1; K=$2; N=$3; R=$4; HEADM=$5
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source /home/l00947526/Ascend806/cann-9.2.0/set_env.sh 2>/dev/null

BUILD_DIR="${SCRIPT_DIR}/build/matmul_reduce_scatter"
EXE_NAME="apace_qmm_rs_urma_st"
OUT_ROOT="${SCRIPT_DIR}/profiling/st_direct_migrated"
RAW_DIR="$OUT_ROOT/raw"
RESULTS_FILE="$OUT_ROOT/msprof_summary.log"
mkdir -p "$RAW_DIR"
touch "$RESULTS_FILE"

CASE_DIR="$RAW_DIR/m${M}k${K}n${N}r${R}"
echo "########## M=$M K=$K N=$N R=$R headM=$HEADM ##########" >> "$RESULTS_FILE"

rm -rf input output
python3 scripts/gen_data.py "$M" "$K" "$N" "$R" > /dev/null 2>&1

rm -rf "$BUILD_DIR/input" "$BUILD_DIR/output"
cp -r input "$BUILD_DIR/input"
for r in $(seq 0 $((R-1))); do mkdir -p "$BUILD_DIR/output/$r"; done

cd "$BUILD_DIR"
rm -rf "$CASE_DIR"
msprof --output="$CASE_DIR" \
       --ai-core=on \
       --aic-mode=task-based \
       --aic-metrics=PipeUtilization \
       --task-time=on \
       --ascendcl=on \
       ./"$EXE_NAME" "$M" "$K" "$N" "$R" perf "$HEADM" > /dev/null 2>&1
cd "$SCRIPT_DIR"

for prof_subdir in "$CASE_DIR"/PROF_*/; do
    [ -d "$prof_subdir" ] || continue
    CSV=$(find "$prof_subdir" -name "op_summary_*.csv" -path "*/mindstudio_profiler_output/*" 2>/dev/null | head -1)
    if [ -n "$CSV" ]; then
        DEV_ID=$(tail -n +2 "$CSV" | tail -1 | cut -d, -f1)
        cp "$CSV" "$CASE_DIR/op_summary_dev${DEV_ID}.csv"
    fi
    rm -rf "$prof_subdir"
done

MAX_MED=0
for csv_file in "$CASE_DIR"/op_summary_dev*.csv; do
    [ -f "$csv_file" ] || continue
    DEV_ID=$(tail -n +2 "$csv_file" | tail -1 | cut -d, -f1)
    MED=$(tail -n +2 "$csv_file" | tail -5 | cut -d, -f10 | grep -v '^$' | sort -n | sed -n '3p')
    echo "  device $DEV_ID median(后5次) = ${MED}us" >> "$RESULTS_FILE"
    if [ "$(echo "$MED > $MAX_MED" | bc 2>/dev/null || echo 0)" = "1" ]; then MAX_MED=$MED; fi
done
echo "  M=$M K=$K N=$N R=$R MAX=${MAX_MED}us" >> "$RESULTS_FILE"
echo "CASE m${M}k${K}n${N}r${R} MAX=${MAX_MED}us"
