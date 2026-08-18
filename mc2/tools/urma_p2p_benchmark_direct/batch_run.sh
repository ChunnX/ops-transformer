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
#   case_id,sqe_mode,bench_mode,sqe_bytes,loop_count,write(us),commit(us),drain(us),total(us),bw(GB/s),verify
#   batch_write: write=assemble, commit=commit, drain=drain
#   write_drain_loop: write=write_and_commit(=assemble+commit), commit=空, drain=drain
# ============================================================================

CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")" && pwd)
CASES_DIR="${CURRENT_DIR}/cases"
INPUT_CSV="${CASES_DIR}/input_batch_write.csv"
# 输出 CSV 名根据输入 CSV 自动生成: input_xxx.csv → output_xxx.csv
INPUT_BASENAME=$(basename "$INPUT_CSV" .csv)
OUTPUT_CSV="${CASES_DIR}/output_${INPUT_BASENAME#input_}.csv"
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

# 批跑前自动编译
echo "[INFO] Building..."
bash "${CURRENT_DIR}/run.sh" -c ON || true
if [ ! -f "$BIN" ]; then
    echo "[ERROR] Build failed."
    exit 1
fi
echo "[INFO] Build done."
echo ""

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

    echo "----------------------------------------"
    echo "[CASE $CASE_ID] $MODE_NAME $BENCH_NAME"
    echo "  cmd: bash run.sh -r ON -d ${DEV_COUNT} -m $SQE_MODE_NUM -o $BENCH_MODE_NUM -s $SQE_BYTES -l $LOOP_COUNT -g ON"

    local RUN_ARGS="--dev_count ${DEV_COUNT} --sqe_bytes ${SQE_BYTES} --loop_count ${LOOP_COUNT} --sqe_mode ${SQE_MODE_NUM} --bench_mode ${BENCH_MODE_NUM} --verify ON"
    if [ -n "$DEV_IDS" ]; then
        RUN_ARGS="$RUN_ARGS --dev_ids=$DEV_IDS"
    fi

    # 跑用例, 捕获输出 (不让子进程退出码影响脚本)
    local OUTPUT
    OUTPUT=$("$BIN" $RUN_ARGS 2>&1) || true

    # 从 Case Summary 提取数据 (取 avg 行)
    # batch_write: assemble→write, commit/issue→commit, drain→drain
    # write_drain_loop: assemble→write_and_commit, drain→drain, commit=N/A
    local WRITE_VAL ISSUE_VAL DRAIN_VAL TOTAL_TIME BW VERIFY
    WRITE_VAL=$(echo "$OUTPUT" | grep "assemble:" | head -1 | sed 's/.*assemble: *//;s/ *us.*//')
    ISSUE_VAL=$(echo "$OUTPUT" | grep -E " (commit|issue):" | head -1 | sed 's/.*: *//;s/ *us.*//')
    DRAIN_VAL=$(echo "$OUTPUT" | grep "drain:" | head -1 | sed 's/.*drain: *//;s/ *us.*//')
    TOTAL_TIME=$(echo "$OUTPUT" | grep "total:" | head -1 | sed 's/.*total: *//;s/ *us.*//')
    BW=$(echo "$OUTPUT" | grep "bw:" | head -1 | sed 's/.*bw: *//;s/ *GB\/s.*//')
    VERIFY=$(echo "$OUTPUT" | grep "verify:" | head -1 | sed 's/.*verify: *//')

    # 解析不到数据 = 出错, 打印日志, 跳过此 case
    if [ -z "$WRITE_VAL" ]; then
        echo "  [ERROR] failed to parse result, last 30 lines:"
        echo "$OUTPUT" | tail -30
        echo "----------------------------------------"
        echo ""
        echo "${CASE_ID},${SQE_MODE},${BENCH_MODE},${SQE_BYTES},${LOOP_COUNT},ERROR,ERROR,ERROR,ERROR,ERROR,ERROR" >> "$OUTPUT_CSV"
        return 0
    fi

    # batch_write: write=assemble, commit=issue_val
    # write_drain_loop: write=assemble(含commit), commit=N/A
    local WRITE_COL COMMIT_COL
    if [ "$BENCH_MODE_NUM" == "0" ]; then
        WRITE_COL="$WRITE_VAL"
        COMMIT_COL="$ISSUE_VAL"
    else
        WRITE_COL="$WRITE_VAL"
        COMMIT_COL="N/A"
    fi

    echo "  result: write=${WRITE_COL}us commit=${COMMIT_COL}us drain=${DRAIN_VAL}us total=${TOTAL_TIME}us bw=${BW}GB/s verify=${VERIFY}"
    echo "----------------------------------------"
    echo ""

    echo "${CASE_ID},${SQE_MODE},${BENCH_MODE},${SQE_BYTES},${LOOP_COUNT},${WRITE_COL},${COMMIT_COL},${DRAIN_VAL},${TOTAL_TIME},${BW},${VERIFY}" >> "$OUTPUT_CSV"
}

# 判断 case_id 是否在过滤列表中
should_run_case() {
    local id="$1"
    if [ -z "$CASE_IDS" ]; then
        return 0  # 无过滤, 跑全部
    fi
    echo ",$CASE_IDS," | grep -q ",$id," || return 1
    return 0
}

# 统计要跑的用例数
TOTAL_CASES=$(awk -F',' 'NR>1 {print $1}' "$INPUT_CSV" | while read id; do should_run_case "$id" && echo x; done | wc -l | tr -d ' ' || true)

echo "[INFO] Input:  $INPUT_CSV"
echo "[INFO] Output: $OUTPUT_CSV"
if [ -n "$CASE_IDS" ]; then
    echo "[INFO] Filter: -caseid $CASE_IDS ($TOTAL_CASES cases)"
else
    echo "[INFO] Filter: none ($TOTAL_CASES cases)"
fi
echo ""

# 写表头
echo "case_id,sqe_mode,bench_mode,sqe_bytes,loop_count,write(us),commit(us),drain(us),total(us),bw(GB/s),verify" > "$OUTPUT_CSV"
echo "(batch_write: write/commit/drain; write_drain_loop: write=write_and_commit, commit=N/A)" >> "$OUTPUT_CSV"

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
    echo ""
    echo "[INFO] Progress: $IDX/$TOTAL_CASES"
    run_case "$CASE_ID" "$SQE_MODE" "$BENCH_MODE" "$SQE_BYTES" "$LOOP_COUNT"

done < "$INPUT_CSV"

echo "[INFO] Done. Results saved to: $OUTPUT_CSV"
echo ""
echo "========== Results =========="
cat "$OUTPUT_CSV"
echo "============================="
