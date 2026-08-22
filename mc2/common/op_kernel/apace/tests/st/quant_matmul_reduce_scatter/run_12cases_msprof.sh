#!/bin/bash
# run_12cases_msprof.sh — 12 case msprof 采集 (ops-transformer 仓, task-based)
# 每个 case: gen_data → msprof 采集 → 提取 op_summary CSV
# 输出到 profiling/st_direct/raw/, 采集后只保留 op_summary_*.csv
# 自适应 headMSize: R=2 → 512, R=4 → 128
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source /home/l00947526/Ascend806/cann-9.2.0/set_env.sh 2>/dev/null

BUILD_DIR="${SCRIPT_DIR}/build/quant_matmul_reduce_scatter"
EXE_NAME="quant_matmul_reduce_scatter"

CASES=(
    "8192 1536 12288 2"
    "8192 1536 12288 4"
    "8192 3696 8192 2"
    "8192 3696 8192 4"
    "8192 1024 8192 2"
    "8192 1024 8192 4"
    "4096 3584 4096 2"
    "4096 3584 4096 4"
    "2048 3584 4096 2"
    "2048 3584 4096 4"
    "4032 8192 4096 2"
    "4032 8192 4096 4"
)

RAW_DIR="$SCRIPT_DIR/../../../kernel/quant_matmul_reduce_scatter/profiling/st_direct/raw"
RESULTS_FILE="$SCRIPT_DIR/../../../kernel/quant_matmul_reduce_scatter/profiling/st_direct/msprof_summary.log"
echo "=== 12 Case msprof 采集 (ops-transformer, task-based) ===" > "$RESULTS_FILE"

for c in "${CASES[@]}"; do
    read -r M K N R <<< "$c"
    HEADM=$([ "$R" -le 2 ] && echo 512 || echo 128)
    CASE_DIR="$RAW_DIR/m${M}k${K}n${N}r${R}"

    echo "########## M=$M K=$K N=$N R=$R headM=$HEADM ##########"
    echo "" >> "$RESULTS_FILE"
    echo "########## M=$M K=$K N=$N R=$R headM=$HEADM ##########" >> "$RESULTS_FILE"

    # 1. 生成数据
    cd "$SCRIPT_DIR"
    rm -rf input output
    python3 scripts/gen_data.py "$M" "$K" "$N" "$R"

    # 2. 拷贝数据到 build 目录
    rm -rf "$BUILD_DIR/input" "$BUILD_DIR/output"
    cp -r input "$BUILD_DIR/input"
    for r in $(seq 0 $((R-1))); do mkdir -p "$BUILD_DIR/output/$r"; done

    # 3. msprof 采集 (task-based, PipeUtilization)
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

    # 4. 清理: 每个 PROF_*/ 目录下只保留 op_summary_*.csv
    for prof_subdir in "$CASE_DIR"/PROF_*/; do
        [ -d "$prof_subdir" ] || continue
        CSV=$(find "$prof_subdir" -name "op_summary_*.csv" -path "*/mindstudio_profiler_output/*" 2>/dev/null | head -1)
        if [ -n "$CSV" ]; then
            DEV_ID=$(tail -n +2 "$CSV" | tail -1 | cut -d, -f1)
            cp "$CSV" "$CASE_DIR/op_summary_dev${DEV_ID}.csv"
        fi
        rm -rf "$prof_subdir"
    done

    # 5. 解析: 取后 5 行 (稳态) 的 Task Duration 中位数, 跨 rank 取 max
    MAX_MED=0
    for csv_file in "$CASE_DIR"/op_summary_dev*.csv; do
        [ -f "$csv_file" ] || continue
        DEV_ID=$(tail -n +2 "$csv_file" | tail -1 | cut -d, -f1)
        MED=$(tail -n +2 "$csv_file" | tail -5 | cut -d, -f10 | grep -v '^$' | sort -n | sed -n '3p')
        echo "  device $DEV_ID median(后5次) = ${MED}us" >> "$RESULTS_FILE"
        echo "  device $DEV_ID median=${MED}us"
        if [ "$(echo "$MED > $MAX_MED" | bc 2>/dev/null || echo 0)" = "1" ]; then
            MAX_MED=$MED
        fi
    done
    echo "  >>> MAX across ranks = ${MAX_MED}us" >> "$RESULTS_FILE"
    echo "  >>> MAX = ${MAX_MED}us"
    echo "  M=$M K=$K N=$N R=$R MAX=${MAX_MED}us" >> "$RESULTS_FILE"
done

echo "" >> "$RESULTS_FILE"
echo "==================== 12 Case msprof 汇总 ====================" >> "$RESULTS_FILE"
echo ""
echo "==================== 12 Case msprof 汇总 ===================="
grep "^  M=" "$RESULTS_FILE"
