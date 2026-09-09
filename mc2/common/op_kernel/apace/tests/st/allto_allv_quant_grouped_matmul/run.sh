#!/bin/bash
# run.sh — apace AllToAllvQuantGroupedMatmul ST
#
# 流程: 生成数据 -> 编译 -> 多 rank 运行 -> 精度比对
#
# 用法:
#   bash run.sh                        # 无参数: 运行 cases.csv 全部 case
#   bash run.sh 1                      # 运行 CSV 第 1 行
#   bash run.sh 1 3 5                  # 运行 CSV 第 1/3/5 行
#   bash run.sh 2-4                    # 运行 CSV 第 2~4 行
#   bash run.sh all                    # 运行 CSV 全部行
#   bash run.sh --cli m k n r e [trans_b] [dtype]   # 命令行模式(绕过 csv)
#   bash run.sh --skip-build ...       # 跳过编译
#   bash run.sh --gen-only ...         # 仅生成 CPU golden
#   bash run.sh --verify-only ...      # 仅精度比对
#   环境变量 KERNEL_TIMEOUT=<秒>       # 覆盖默认超时(600s)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ST_DIR}/build"
KERNEL_TIMEOUT=${KERNEL_TIMEOUT:-600}

SKIP_BUILD=0; GEN_ONLY=0; VERIFY_ONLY=0; CLI_MODE=0
CSV_FILE="${SCRIPT_DIR}/cases.csv"

# ---- 解析参数 ----
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-build)  SKIP_BUILD=1; shift ;;
        --gen-only)    GEN_ONLY=1; shift ;;
        --verify-only) VERIFY_ONLY=1; shift ;;
        --cli)         CLI_MODE=1; shift ;;
        -h|--help)     sed -n '2,15p' "$0"; exit 0 ;;
        *)             ARGS+=("$1"); shift ;;
    esac
done

if [ "$GEN_ONLY" -eq 1 ] && [ "$VERIFY_ONLY" -eq 1 ]; then
    echo "ERROR: --gen-only and --verify-only are mutually exclusive"; exit 1
fi

# ---- 将行号参数展开为列表 ----
expand_rows() {
    local input=("$@")
    local result=()
    for item in "${input[@]}"; do
        if [ "$item" = "all" ]; then
            for ((i=1; i<=TOTAL_LINES; i++)); do result+=($i); done
        elif [[ "$item" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            local start=${BASH_REMATCH[1]} end=${BASH_REMATCH[2]}
            for ((i=start; i<=end; i++)); do result+=($i); done
        elif [[ "$item" =~ ^[0-9]+$ ]]; then
            result+=($item)
        fi
    done
    echo "${result[@]}"
}

# ---- 环境检查 ----
if [ -n "${ASCEND_HOME_PATH:-}" ] && [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    source "${ASCEND_HOME_PATH}/set_env.sh" >/dev/null 2>&1 || true
else
    echo "ERROR: ASCEND_HOME_PATH not set or set_env.sh missing"; exit 1
fi
command -v python3 >/dev/null || { echo "ERROR: python3 not available"; exit 1; }

# ---- 单 case 执行函数 ----
run_single() {
    local M=$1 K=$2 N=$3 RANK_NUM=$4 EXPERT_NUM=$5 TRANS_B=$6 DTYPE=$7
    echo ""
    echo "=========================================="
    echo "apace AllToAllvQuantGroupedMatmul ST"
    echo "  M=$M K=$K N=$N rankNum=$RANK_NUM expertNum=$EXPERT_NUM transB=$TRANS_B dtype=$DTYPE"
    echo "=========================================="

    cd "$SCRIPT_DIR"
    rm -rf input output

    # ---- 1. 生成数据 ----
    echo "[1/4] Generate CPU golden + input data..."
    python3 scripts/gen_data.py "$M" "$K" "$N" "$RANK_NUM" "$EXPERT_NUM" "$TRANS_B" "$DTYPE"

    if [ "$GEN_ONLY" -eq 1 ]; then return 0; fi

    # ---- 2. 编译 ----
    if [ "$VERIFY_ONLY" -eq 0 ]; then
        if [ "$SKIP_BUILD" -eq 0 ]; then
            echo "[2/4] Build..."
            mkdir -p "$BUILD_DIR"
            cmake -S "$ST_DIR" -B "$BUILD_DIR" || { echo "ERROR: cmake failed"; return 1; }
            cmake --build "$BUILD_DIR" --target apace_a2av_gmm_st --parallel 4 || { echo "ERROR: build failed"; return 1; }
        else
            [ -x "$BUILD_DIR/allto_allv_quant_grouped_matmul/apace_a2av_gmm_st" ] || { echo "ERROR: --skip-build but binary missing"; return 1; }
            echo "[2/4] Skip build (--skip-build)"
        fi

        # ---- 3. 多 rank 运行 ----
        echo "[3/4] NPU run ($RANK_NUM ranks)..."
        local EXE_PATH="$BUILD_DIR/allto_allv_quant_grouped_matmul/apace_a2av_gmm_st"
        local EXE_DIR="$(dirname "$EXE_PATH")"
        rm -rf "$EXE_DIR/input" "$EXE_DIR/output"
        cp -r input "$EXE_DIR/input"
        for r in $(seq 0 $((RANK_NUM-1))); do
            mkdir -p "$EXE_DIR/output/$r"
        done

        MAX_RETRY=3
        RETRY=0
        while [ ${RETRY} -lt ${MAX_RETRY} ]; do
            if command -v fuser >/dev/null 2>&1; then
                fuser -k 8998/tcp 2>/dev/null || true
            else
                echo "  [WARN] fuser not installed, skip port cleanup"
            fi
            sleep 1

            if command -v timeout >/dev/null 2>&1; then
                set +e
                KERNEL_OUT=$(timeout ${KERNEL_TIMEOUT}s "$EXE_PATH" $M $K $N $RANK_NUM $EXPERT_NUM $TRANS_B "$DTYPE" 2>&1)
                KERNEL_RC=$?
                set -e
            else
                set +e
                KERNEL_OUT=$("$EXE_PATH" $M $K $N $RANK_NUM $EXPERT_NUM $TRANS_B "$DTYPE" 2>&1)
                KERNEL_RC=$?
                set -e
            fi
            echo "${KERNEL_OUT}"

            if [ ${KERNEL_RC} -eq 124 ]; then
                echo "ERROR: Kernel timeout after ${KERNEL_TIMEOUT}s"
                return 1
            fi

            if [ ${KERNEL_RC} -eq 0 ] && echo "${KERNEL_OUT}" | grep -q "Status: SUCCESS"; then
                echo "  Kernel finished!"
                break
            fi

            if echo "${KERNEL_OUT}" | grep -q "connect peers failed"; then
                RETRY=$((RETRY + 1))
                echo "  [retry ${RETRY}/${MAX_RETRY}] connect conflict..."
                sleep 2
                continue
            fi

            echo "ERROR: Kernel failed"
            return 1
        done

        if [ ${RETRY} -ge ${MAX_RETRY} ]; then
            echo "ERROR: Max retries exceeded"
            return 1
        fi

        cd "$SCRIPT_DIR"
        for r in $(seq 0 $((RANK_NUM-1))); do
            mkdir -p "output/$r"
            cp "$EXE_DIR/output/$r/npu_out.bin" "output/$r/npu_out.bin" 2>/dev/null || true
        done
    fi

    # ---- 4. 精度比对 ----
    echo "[4/4] Verify..."
    python3 scripts/verify_result.py "$N" "$RANK_NUM" "./output"
    return $?
}

# ---- 选择参数来源 ----
if [ "${CLI_MODE:-0}" -eq 1 ]; then
    M=2048; K=4096; N=4096; RANK_NUM=2; EXPERT_NUM=8; TRANS_B=1; DTYPE=e4m3_e4m3
    idx=0
    [ $idx -lt ${#ARGS[@]} ] && { M=${ARGS[$idx]}; idx=$((idx+1)); }
    [ $idx -lt ${#ARGS[@]} ] && { K=${ARGS[$idx]}; idx=$((idx+1)); }
    [ $idx -lt ${#ARGS[@]} ] && { N=${ARGS[$idx]}; idx=$((idx+1)); }
    [ $idx -lt ${#ARGS[@]} ] && { RANK_NUM=${ARGS[$idx]}; idx=$((idx+1)); }
    [ $idx -lt ${#ARGS[@]} ] && { EXPERT_NUM=${ARGS[$idx]}; idx=$((idx+1)); }
    [ $idx -lt ${#ARGS[@]} ] && { TRANS_B=${ARGS[$idx]}; idx=$((idx+1)); }
    [ $idx -lt ${#ARGS[@]} ] && { DTYPE=${ARGS[$idx]}; idx=$((idx+1)); }
    run_single "$M" "$K" "$N" "$RANK_NUM" "$EXPERT_NUM" "$TRANS_B" "$DTYPE"
    exit $?
fi

# ---- CSV 模式 ----
if [ ! -f "$CSV_FILE" ]; then
    echo "ERROR: CSV not found: $CSV_FILE"; exit 1
fi

DATA_LINES=$(grep -v '^#' "$CSV_FILE" | grep -v '^$' | tail -n +2)
TOTAL_LINES=$(echo "$DATA_LINES" | grep -c .)

if [ ${#ARGS[@]} -eq 0 ]; then
    echo "==== cases.csv ($TOTAL_LINES cases) ===="
    echo "Row  M      K     N     rank  expert  transB  dtype"
    local_idx=0
    while IFS= read -r line; do
        local_idx=$((local_idx + 1))
        printf "%-5s %s\n" "$local_idx" "$(echo "$line" | awk -F, '{printf "%-6s %-5s %-5s %-5s %-7s %-7s %s", $1,$2,$3,$4,$5,$6,$7}')"
    done <<< "$DATA_LINES"
    echo ""
    echo "Running all $TOTAL_LINES cases..."
    SELECTED_ROWS=($(seq 1 $TOTAL_LINES))
else
    SELECTED_ROWS=($(expand_rows "${ARGS[@]}"))
fi

for row in "${SELECTED_ROWS[@]}"; do
    if [ "$row" -lt 1 ] || [ "$row" -gt "$TOTAL_LINES" ]; then
        echo "ERROR: row $row out of range (1-$TOTAL_LINES)"; exit 1
    fi
done

echo "==== Running ${#SELECTED_ROWS[@]} case(s): ${SELECTED_ROWS[*]} ===="

PASS_CNT=0; FAIL_CNT=0
RESULTS=()

for row in "${SELECTED_ROWS[@]}"; do
    SELECTED=$(echo "$DATA_LINES" | sed -n "${row}p")
    M=$(echo "$SELECTED" | awk -F, '{print $1}')
    K=$(echo "$SELECTED" | awk -F, '{print $2}')
    N=$(echo "$SELECTED" | awk -F, '{print $3}')
    RANK_NUM=$(echo "$SELECTED" | awk -F, '{print $4}')
    EXPERT_NUM=$(echo "$SELECTED" | awk -F, '{print $5}')
    TRANS_B=$(echo "$SELECTED" | awk -F, '{print $6}')
    DTYPE=$(echo "$SELECTED" | awk -F, '{print $7}')

    echo ""
    echo "########## CSV row ${row}/${TOTAL_LINES} ##########"

    if run_single "$M" "$K" "$N" "$RANK_NUM" "$EXPERT_NUM" "$TRANS_B" "$DTYPE"; then
        PASS_CNT=$((PASS_CNT + 1))
        RESULTS+=("PASS  row$row  M=$M K=$K N=$N rank=$RANK_NUM e=$EXPERT_NUM transB=$TRANS_B $DTYPE")
    else
        FAIL_CNT=$((FAIL_CNT + 1))
        RESULTS+=("FAIL  row$row  M=$M K=$K N=$N rank=$RANK_NUM e=$EXPERT_NUM transB=$TRANS_B $DTYPE")
    fi
    SKIP_BUILD=1
done

echo ""
echo "=========================================="
echo "Summary: PASS=$PASS_CNT  FAIL=$FAIL_CNT  total=${#SELECTED_ROWS[@]}"
echo "=========================================="
for r in "${RESULTS[@]}"; do
    echo "  $r"
done

if [ "$FAIL_CNT" -eq 0 ]; then exit 0; else exit 1; fi
