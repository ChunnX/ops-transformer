#!/bin/bash
# run.sh — apace AllToAllQuantMatmul ST（CCU 直调变体）
#
# 流程: 编译 -> 多 rank 运行（rank 数 = 本机 NPU 数，进程内每卡一线程）-> 位级校验
#
# 用法:
#   bash run.sh                        # 无参数: 列出 cases.csv 全部 case，依次运行（precision）
#   bash run.sh 1                      # 运行 CSV 第 1 行
#   bash run.sh 1 3 5                  # 运行 CSV 第 1/3/5 行
#   bash run.sh 2-4                    # 运行 CSV 第 2~4 行
#   bash run.sh all                    # 运行 CSV 全部行
#   bash run.sh --csv <file> 1 3       # 指定 csv 文件 + 行号
#   bash run.sh --cli precision 512 128 256 256   # 命令行模式(绕过 csv)
#   bash run.sh --perf 4               # perf 模式运行 CSV 第 4 行(msprof 采集)
#   bash run.sh --skip-build ...       # 跳过编译
#   perf 模式跑完后: python3 scripts/parse_prof.py --all  解析性能数据
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ST_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"                       # tests/st
BUILD_DIR="${ST_DIR}/build"
PROF_DIR="${SCRIPT_DIR}/prof"
TARGET=apace_a2a_qmm_ccu_st

SKIP_BUILD=0; CLI_MODE=0; PERF_MODE=0
CSV_FILE="${SCRIPT_DIR}/cases.csv"

# ---- 解析参数 ----
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-build)  SKIP_BUILD=1; shift ;;
        --cli)         CLI_MODE=1; shift ;;
        --csv)         CSV_FILE="$2"; shift 2 ;;
        --perf)        PERF_MODE=1; shift ;;
        -h|--help)     sed -n '2,18p' "$0"; exit 0 ;;
        *)             ARGS+=("$1"); shift ;;
    esac
done

# ---- 环境检查 ----
if [ -n "${ASCEND_HOME_PATH:-}" ] && [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    source "${ASCEND_HOME_PATH}/set_env.sh" >/dev/null 2>&1 || true
else
    echo "ERROR: ASCEND_HOME_PATH not set or set_env.sh missing"; exit 1
fi

# ---- 将行号参数展开为列表（支持 "1 3 5"、"2-4"、"all"）----
TOTAL_LINES=$(( $(wc -l < "$CSV_FILE") - 1 ))
expand_rows() {
    local result=()
    for item in "$@"; do
        if [ "$item" = "all" ]; then
            for ((i=1; i<=TOTAL_LINES; i++)); do result+=($i); done
        elif [[ "$item" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            for ((i=${BASH_REMATCH[1]}; i<=${BASH_REMATCH[2]}; i++)); do result+=($i); done
        elif [[ "$item" =~ ^[0-9]+$ ]]; then
            result+=($item)
        fi
    done
    echo "${result[@]}"
}

# ---- 编译 ----
build() {
    if [ "$SKIP_BUILD" -eq 0 ]; then
        echo "[build] ${TARGET} ..."
        mkdir -p "$BUILD_DIR"
        cmake -S "$ST_DIR" -B "$BUILD_DIR" || { echo "ERROR: cmake failed"; return 1; }
        cmake --build "$BUILD_DIR" --target "$TARGET" --parallel 8 || { echo "ERROR: build failed"; return 1; }
    fi
    [ -x "$BUILD_DIR/all_to_all_quant_matmul/ccu/${TARGET}" ] || { echo "ERROR: binary missing (use --skip-build only after a build)"; return 1; }
}

# ---- 单 case 执行 ----
run_binary() {
    local MODE="$1"; shift
    local EXE="$BUILD_DIR/all_to_all_quant_matmul/ccu/${TARGET}"
    cd "$(dirname "$EXE")"
    export HCCL_OP_EXPANSION_MODE=CCU_SCHED
    if [ "$PERF_MODE" -eq 1 ] && [ "$MODE" = "perf" ]; then
        rm -rf "$PROF_DIR"; mkdir -p "$PROF_DIR"
        msprof --application="${EXE} $*" --output="${PROF_DIR}" --ai-core=on -disable-gpu-memory-block=false
        echo "perf data saved to: $PROF_DIR"
        echo "parse with: python3 scripts/parse_prof.py --all"
    else
        "$EXE" "$MODE" "$@"
    fi
}

# ---- 入口 ----
build

if [ "$CLI_MODE" -eq 1 ]; then
    run_binary "${ARGS[0]}" "${ARGS[@]:1}"
    exit $?
fi

[ -f "$CSV_FILE" ] || { echo "ERROR: csv file not found: $CSV_FILE"; exit 1; }
[ ${#ARGS[@]} -gt 0 ] || ARGS=(all)
ROW_LIST=($(expand_rows "${ARGS[@]}"))

for row in "${ROW_LIST[@]}"; do
    LINE=$(sed -n "$((row + 1))p" "$CSV_FILE")
    [ -n "$LINE" ] || { echo "ERROR: row $row out of range (total $TOTAL_LINES)"; continue; }
    IFS=',' read -r MODE M TILEM K N LOOP <<< "$LINE"
    echo ""
    echo "=========================================="
    echo "apace AllToAllQuantMatmul ST (CCU)  case $row"
    echo "  mode=$MODE M=$M tileM=$TILEM K=$K N=$N loop=$LOOP"
    echo "=========================================="
    run_binary "$MODE" "$M" "$TILEM" "$K" "$N" "$LOOP"
done

echo ""
echo "All requested cases finished."
