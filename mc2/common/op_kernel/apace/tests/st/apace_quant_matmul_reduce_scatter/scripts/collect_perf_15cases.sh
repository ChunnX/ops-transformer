#!/bin/bash
# collect_perf_15cases.sh — 15 用例完整 msprof profiling 采集
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PERF_DIR="$OP_DIR/docs/perf/round_15cases"

source "${ASCEND_HOME_PATH}/set_env.sh" >/dev/null 2>&1 || true

cd "$OP_DIR"

SCENES=(
    "std_r2_2048   2048 3584 4096 2"
    "std_r4_2048   2048 3584 4096 4"
    "largeM_r2     4096 3584 4096 2"
    "smallM_r2     1024 3584 4096 2"
    "largeM_r4     4096 3584 4096 4"
    "xlargeM_r2    8192 3584 4096 2"
    "xlargeM_r4    8192 3584 4096 4"
    "largeK_r4     4032 8192 4096 4"
    "largeN_r4     2048 3584 8192 4"
    "tiny_256      256  3584 256  2"
    "tiny_512      512  128  1024 2"
    "largeN_smallK 1024 128  8192 2"
    "smallN_r2     2048 3584 256  2"
    "smallK_r2     2048 512  4096 2"
    "smallN_r4     4096 3584 1024 4"
)

echo "========================================"
echo "quantMatmulReduceScatter 15-case msprof"
echo "Date: $(date -Iseconds)"
echo "NPU: Ascend950PR (dav-3510)"
echo "========================================"

rm -f "$PERF_DIR/msprof_task_duration_raw.log"
mkdir -p "$PERF_DIR"

for scene in "${SCENES[@]}"; do
    read -r NAME M K N RANK <<< "$scene"
    echo ""
    echo "########## msprof: $NAME | M=$M K=$K N=$N rank=$RANK ##########"

    echo "[gen_data]..."
    python3 scripts/gen_data.py "$M" "$K" "$N" "$RANK"

    rm -rf build/input build/output
    cp -r input build/input
    mkdir -p build/output
    for r in $(seq 0 $((RANK-1))); do
        mkdir -p "build/output/$r"
    done

    MSPROF_OUT="$PERF_DIR/msprof_raw_${NAME}"
    rm -rf "$MSPROF_OUT"
    mkdir -p "$MSPROF_OUT"

    cd build
    msprof --ai-core=on --aic-mode=task-based --aic-metrics=PipeUtilization \
        --output="$MSPROF_OUT" \
        --application="./quant_matmul_reduce_scatter_st $M $K $N $RANK perf 512" \
        2>&1 | tee -a "$PERF_DIR/msprof_task_duration_raw.log" || true
    cd "$OP_DIR"

    echo "[msprof done] $NAME"
done

echo ""
echo "========================================"
echo "All 15 scenes done. Extracting op_summary..."
echo "========================================"

for scene in "${SCENES[@]}"; do
    read -r NAME M K N RANK <<< "$scene"
    MSPROF_OUT="$PERF_DIR/msprof_raw_${NAME}"

    for dev_dir in "$MSPROF_OUT"/PROF_*/device_*/; do
        [ -d "$dev_dir" ] || continue
        dev_id=$(basename "$dev_dir")
        csv_src="$dev_dir/mindstudio_profiler_output/op_summary.csv"
        if [ -f "$csv_src" ]; then
            csv_dst="$PERF_DIR/op_summary_${NAME}_${dev_id}.csv"
            cp "$csv_src" "$csv_dst"
            echo "  Copied: op_summary_${NAME}_${dev_id}.csv"
        fi
    done
done

echo ""
echo "Done. Performance data in: $PERF_DIR"
