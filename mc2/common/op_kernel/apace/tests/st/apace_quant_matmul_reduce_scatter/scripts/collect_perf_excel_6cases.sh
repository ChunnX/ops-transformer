#!/bin/bash
source /home/l00947526/Ascend806/cann-9.2.0/set_env.sh 2>/dev/null

OP_DIR=/home/y00845930/cann_skills/plugins-official/ops-direct-invoke/operators/quantMatmulReduceScatter
cd "$OP_DIR"

PERF_DIR="$OP_DIR/docs/perf/round_excel_6cases"
mkdir -p "$PERF_DIR"
LOG="$PERF_DIR/msprof_task_duration_raw.log"
rm -f "$LOG"

CASES=(
    "case1_r2  8192 1536 12288 2"
    "case2_r4  8192 1536 12288 4"
    "case3_r2  8192 3696 8192 2"
    "case4_r4  8192 3696 8192 4"
    "case5_r2  8192 1024 8192 2"
    "case6_r4  8192 1024 8192 4"
)

echo "========================================" | tee -a "$LOG"
echo "quantMatmulReduceScatter 6 Case msprof" | tee -a "$LOG"
echo "Date: $(date -Iseconds)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

for scene in "${CASES[@]}"; do
    read -r NAME M K N RANK <<< "$scene"
    echo "" | tee -a "$LOG"
    echo "########## msprof: $NAME | M=$M K=$K N=$N rank=$RANK ##########" | tee -a "$LOG"

    python3 "$OP_DIR/scripts/gen_data.py" "$M" "$K" "$N" "$RANK" 2>/dev/null

    rm -rf "$OP_DIR/build/input" "$OP_DIR/build/output"
    cp -r "$OP_DIR/input" "$OP_DIR/build/input"
    mkdir -p "$OP_DIR/build/output"
    for r in $(seq 0 $((RANK-1))); do
        mkdir -p "$OP_DIR/build/output/$r"
    done

    MSPROF_OUT="$PERF_DIR/msprof_raw_${NAME}"
    rm -rf "$MSPROF_OUT"
    mkdir -p "$MSPROF_OUT"

    cd "$OP_DIR/build"
    msprof --ai-core=on --aic-mode=task-based --aic-metrics=PipeUtilization \
        --output="$MSPROF_OUT" \
        --application="./quant_matmul_reduce_scatter_st $M $K $N $RANK perf 512" \
        2>&1 | tee -a "$LOG" || true
    cd "$OP_DIR"

    echo "[msprof done] $NAME" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "Extracting op_summary..." | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

for scene in "${CASES[@]}"; do
    read -r NAME M K N RANK <<< "$scene"
    MSPROF_OUT="$PERF_DIR/msprof_raw_${NAME}"
    idx=0
    for csv in "$MSPROF_OUT"/PROF_*/mindstudio_profiler_output/op_summary_*.csv; do
        [ -f "$csv" ] || continue
        idx=$((idx+1))
        cp "$csv" "$PERF_DIR/op_summary_${NAME}_rank${idx}.csv"
        echo "  Copied: op_summary_${NAME}_rank${idx}.csv" | tee -a "$LOG"
    done
done

echo "" | tee -a "$LOG"
echo "Done. Performance data in: $PERF_DIR" | tee -a "$LOG"

# 复制原始日志到 logs 目录
cp "$LOG" "$OP_DIR/logs/msprof_raw_excel_6cases.log"
echo "Raw log copied to logs/msprof_raw_excel_6cases.log"
