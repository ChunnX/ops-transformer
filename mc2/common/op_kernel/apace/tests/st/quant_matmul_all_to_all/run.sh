#!/bin/bash
# run.sh — apace QuantMatmulAllToAll ST 一键脚本
#
# 流程: 生成数据 -> 编译 -> 多 rank 运行 -> 精度比对
# 注意: 编译/运行步骤依赖 apace kernel 修复（见 README "已知阻塞"），未修复前
#       build/run 会失败；gen/verify 可独立运行。
#
# 用法:
#   bash run.sh [m] [k] [n] [rank_num] [tile_cnt] [local_matmul]
#   bash run.sh                       # 默认 2048 3584 4096 2 4 1
#   bash run.sh --skip-build ...      # 跳过编译
#   bash run.sh --gen-only ...        # 仅生成 CPU golden
#   bash run.sh --verify-only ...     # 仅精度比对（需已有 output）
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ST_DIR}/build"

M=2048; K=3584; N=4096; RANK_NUM=2; TILE_CNT=4; LOCAL_MATMUL=1
SKIP_BUILD=0; GEN_ONLY=0; VERIFY_ONLY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-build)  SKIP_BUILD=1; shift ;;
        --gen-only)    GEN_ONLY=1; shift ;;
        --verify-only) VERIFY_ONLY=1; shift ;;
        -h|--help)
            sed -n '1,20p' "$0"; exit 0 ;;
        *)
            M=$1; shift
            [ $# -gt 0 ] && { K=$1; shift; }
            [ $# -gt 0 ] && { N=$1; shift; }
            [ $# -gt 0 ] && { RANK_NUM=$1; shift; }
            [ $# -gt 0 ] && { TILE_CNT=$1; shift; }
            [ $# -gt 0 ] && { LOCAL_MATMUL=$1; shift; }
            ;;
    esac
done

# main.cpp 参数顺序: m k n rankNum tileCnt tileSize tailSize isLocalDelayed isPingPong
TILE_SIZE=$((M / TILE_CNT))
TAIL_SIZE=0
IS_PINGPONG=0

echo "=========================================="
echo "apace QuantMatmulAllToAll ST"
echo "  M=$M K=$K N=$N rankNum=$RANK_NUM tileCnt=$TILE_CNT localMatmul=$LOCAL_MATMUL"
echo "=========================================="

# ---- 环境 ----
if [ -n "${ASCEND_HOME_PATH:-}" ] && [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    source "${ASCEND_HOME_PATH}/set_env.sh" >/dev/null 2>&1 || true
else
    echo "ERROR: ASCEND_HOME_PATH 未设置"; exit 1
fi
command -v python3 >/dev/null || { echo "ERROR: python3 不可用"; exit 1; }

cd "$SCRIPT_DIR"
rm -rf input output

# ---- 1. 生成数据 ----
echo "[1/4] 生成 CPU golden + 输入数据..."
python3 scripts/gen_data.py "$M" "$K" "$N" "$RANK_NUM"

if [ "$GEN_ONLY" -eq 1 ]; then echo "[done] --gen-only"; exit 0; fi

# ---- 2. 编译 ----
if [ "$VERIFY_ONLY" -eq 0 ]; then
    if [ "$SKIP_BUILD" -eq 0 ]; then
        echo "[2/4] 编译..."
        mkdir -p "$BUILD_DIR"
        cmake -S "$ST_DIR" -B "$BUILD_DIR" || { echo "ERROR: cmake 失败（kernel 可能未修复，见 README）"; exit 1; }
        cmake --build "$BUILD_DIR" --target apace_a2a_qmm_st --parallel 4 || {
            echo "ERROR: 编译失败。apace kernel 当前存在 5 处编译 bug（见 README）。"
            echo "       修复 kernel 后重新运行本脚本。gen_data 已完成，可先用 --verify-only 走通比对流程。"
            exit 1; }
    else
        [ -x "$BUILD_DIR/quant_matmul_all_to_all/apace_a2a_qmm_st" ] || { echo "ERROR: --skip-build 但二进制不存在"; exit 1; }
        echo "[2/4] 跳过编译 (--skip-build)"
    fi

    # ---- 3. 多 rank 运行 ----
    echo "[3/4] NPU 运行 ($RANK_NUM ranks)..."
    EXE_PATH="$BUILD_DIR/quant_matmul_all_to_all/apace_a2a_qmm_st"
    EXE_DIR="$(dirname "$EXE_PATH")"
    rm -rf "$EXE_DIR/input"
    cp -r input "$EXE_DIR/input"
    mkdir -p "$EXE_DIR/output"
    for r in $(seq 0 $((RANK_NUM-1))); do
        mkdir -p "$EXE_DIR/output/$r"
    done
    cd "$EXE_DIR"
    "./apace_a2a_qmm_st" "$M" "$K" "$N" "$RANK_NUM" "$TILE_CNT" "$TILE_SIZE" "$TAIL_SIZE" "$LOCAL_MATMUL" "$IS_PINGPONG"
    # 回拷 npu_out
    cd "$SCRIPT_DIR"
    for r in $(seq 0 $((RANK_NUM-1))); do
        mkdir -p "output/$r"
        cp "$EXE_DIR/output/$r/npu_out.bin" "output/$r/npu_out.bin" 2>/dev/null || true
    done
fi

# ---- 4. 精度比对 ----
echo "[4/4] 精度比对..."
python3 scripts/verify_result.py "$M" "$N" "$RANK_NUM" "./output"
RC=$?
if [ $RC -eq 0 ]; then echo "==== PASS ===="; else echo "==== FAIL ===="; fi
exit $RC
