#!/bin/bash
set -euo pipefail

# ============================================================================
# batch_run.sh — 批量/单个跑 benchmark 用例, 从 input CSV 读取, 输出到 result CSV
# 仅支持 -m 2 (hcomm) 和 -m 3 (mc2_native)
#
# 用法:
#   bash batch_run.sh                          # 批跑所有用例 (默认 cases/input_cases.csv)
#   bash batch_run.sh -caseid 3                # 只跑 case_id=3 的用例
#   bash batch_run.sh -caseid 3,5,7            # 跑多个 case_id
#   bash batch_run.sh -i my_cases.csv          # 指定输入 CSV
#   bash batch_run.sh -o my_results.csv        # 指定输出 CSV
#   bash batch_run.sh -d 2                     # dev_count
#   bash batch_run.sh -- 0,1                   # dev_ids
#
# 输入 CSV 格式 (含表头):
#   case_id,sqe_mode,bench_mode,sqe_bytes,loop_count
#   1,mc2_native,batch_write,32,1
#   2,mc2_native,batch_write,65535,100
#   3,hcomm,write_drain_loop,32,500
#   ...
#
# sqe_mode 支持: mc2_native (=3), hcomm (=2)
# bench_mode 支持: batch_write (=0), write_drain_loop (=1)
#
# 输出 CSV 格式 (含表头):
#   case_id,sqe_mode,bench_mode,sqe_bytes,loop_count,assemble(us),commit/issue(us),drain(us),total(us),bw(GB/s),verify
# ============================================================================

CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")" && pwd)
CASES_DIR="${CURRENT_DIR}/cases"
INPUT_CSV="${CASES_DIR}/input_cases.csv"
OUTPUT_CSV="${CASES_DIR}/output_results.csv"
DEV_COUNT=2
DEV_IDS=""
CASE_IDS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
    -i)        INPUT_CSV="$2"; shift 2 ;;
    -o)        OUTPUT_CSV="$2"; shift 2 ;;
    -d)        DEV_COUNT="$2"; shift 2 ;;
    -caseid)   CASE_IDS="$2"; shift 2 ;;
    -h|--help) head -28 "$0" | tail -25; exit 0 ;;
    --)        shift; if [ $# -gt 0 ]; then DEV_IDS="$1"; fi; shift ;;
    *)         echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [ ! -f "$INPUT_CSV" ]; then
    echo "[ERROR] Input CSV not found: $INPUT_CSV"
    exit 1
fi

BIN="${CURRENT_DIR}/out/bin/test_urma_p2p_bench"
if [ ! -f "$BIN" ]; then
    echo "[ERROR] Binary not found: $BIN"
    echo "        Run 'bash run.sh -c ON' first to build."
    exit 1
fi

export LD_LIBRARY_PATH="${CURRENT_DIR}/out/lib:${CURRENT_DIR}/out/lib64:${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/lib64:$LD_LIBRARY_PATH"

# 构造 case_id 过滤条件
# CASE_IDS 为空 = 跑全部; CASE_IDS="3" = 只跑 3; CASE_IDS="3,5" = 跑 3 和 5
run_case() {
    local CASE_ID="$1"
    local SQE_MODE="$2"
    local BENCH_MODE="$3"
    local SQE_BYTES="$4"
    local LOOP_COUNT="$5"

    # sqe_mode 字符串转数字
    local SQE_MODE_NUM=""
    local MODE_NAME=""
    case "$SQE_MODE" in
        mc2_native|3) SQE_MODE_NUM=3; MODE_NAME="mc2_native" ;;
        hcomm|2)      SQE_MODE_NUM=2; MODE_NAME="hcomm" ;;
        *) echo "[WARN] Case $CASE_ID: skip sqe_mode=$SQE_MODE (only hcomm/mc2_native supported)"; return ;;
    esac

    # bench_mode 字符串转数字
    local BENCH_MODE_NUM=""
    local BENCH_NAME=""
    case "$BENCH_MODE" in
        batch_write|0)       BENCH_MODE_NUM=0; BENCH_NAME="batch_write" ;;
        write_drain_loop|1)  BENCH_MODE_NUM=1; BENCH_NAME="write_drain_loop" ;;
        *) BENCH_MODE_NUM="$BENCH_MODE"; BENCH_NAME="$BENCH_MODE" ;;
    esac

    echo "[INFO] Case $CASE_ID: $MODE_NAME $BENCH_NAME sqe_bytes=$SQE_BYTES loop=$LOOP_COUNT"

    local RUN_ARGS="--dev_count ${DEV_COUNT} --sqe_bytes ${SQE_BYTES} --loop_count ${LOOP_COUNT} --sqe_mode ${SQE_MODE_NUM} --bench_mode ${BENCH_MODE_NUM} --verify ON"
    if [ -n "$DEV_IDS" ]; then
        RUN_ARGS="$RUN_ARGS --dev_ids=$DEV_IDS"
    fi

    # 跑用例, 捕获输出
    local OUTPUT
    OUTPUT=$("$BIN" $RUN_ARGS 2>&1) || true

    # 从 Case Summary 提取数据 (取 avg 行)
    local ASSEMBLE ISSUE DRAIN TOTAL_TIME BW VERIFY
    ASSEMBLE=$(echo "$OUTPUT" | grep "assemble:" | head -1 | sed 's/.*assemble: *//;s/ *us.*//')
    ISSUE=$(echo "$OUTPUT" | grep -E " (commit|issue):" | head -1 | sed 's/.*: *//;s/ *us.*//')
    DRAIN=$(echo "$OUTPUT" | grep "drain:" | head -1 | sed 's/.*drain: *//;s/ *us.*//')
    TOTAL_TIME=$(echo "$OUTPUT" | grep "total:" | head -1 | sed 's/.*total: *//;s/ *us.*//')
    BW=$(echo "$OUTPUT" | grep "bw:" | head -1 | sed 's/.*bw: *//;s/ *GB\/s.*//')
    VERIFY=$(echo "$OUTPUT" | grep "verify:" | head -1 | sed 's/.*verify: *//')

    echo "  → assemble=${ASSEMBLE}us drain=${DRAIN}us total=${TOTAL_TIME}us bw=${BW}GB/s verify=${VERIFY}"
    echo ""

    echo "${CASE_ID},${SQE_MODE},${BENCH_MODE},${SQE_BYTES},${LOOP_COUNT},${ASSEMBLE},${ISSUE},${DRAIN},${TOTAL_TIME},${BW},${VERIFY}" >> "$OUTPUT_CSV"
}

# 判断 case_id 是否在过滤列表中
should_run_case() {
    local id="$1"
    if [ -z "$CASE_IDS" ]; then
        return 0  # 无过滤, 跑全部
    fi
    echo ",$CASE_IDS," | grep -q ",$id,"
}

# 统计要跑的用例数
TOTAL_CASES=$(awk -F',' 'NR>1 {print $1}' "$INPUT_CSV" | while read id; do should_run_case "$id" && echo x; done | wc -l | tr -d ' ')

echo "[INFO] Input:  $INPUT_CSV"
echo "[INFO] Output: $OUTPUT_CSV"
if [ -n "$CASE_IDS" ]; then
    echo "[INFO] Filter: -caseid $CASE_IDS ($TOTAL_CASES cases)"
else
    echo "[INFO] Filter: none ($TOTAL_CASES cases)"
fi
echo ""

# 写表头
echo "case_id,sqe_mode,bench_mode,sqe_bytes,loop_count,assemble(us),commit/issue(us),drain(us),total(us),bw(GB/s),verify" > "$OUTPUT_CSV"
echo "(sqe_mode: mc2_native/hcomm, bench_mode: batch_write/write_drain_loop)" >> "$OUTPUT_CSV"

# 读取 CSV, 跳过表头
IDX=0
while IFS=',' read -r CASE_ID SQE_MODE BENCH_MODE SQE_BYTES LOOP_COUNT; do
    # 跳过空行和表头
    [ -z "$CASE_ID" ] && continue
    [[ "$CASE_ID" == case_id* ]] && continue

    if ! should_run_case "$CASE_ID"; then
        continue
    fi

    IDX=$((IDX+1))
    echo "[INFO] Progress: $IDX/$TOTAL_CASES"
    run_case "$CASE_ID" "$SQE_MODE" "$BENCH_MODE" "$SQE_BYTES" "$LOOP_COUNT"

done < "$INPUT_CSV"

echo "[INFO] Done. Results saved to: $OUTPUT_CSV"
echo ""
echo "========== Results =========="
cat "$OUTPUT_CSV"
echo "============================="
