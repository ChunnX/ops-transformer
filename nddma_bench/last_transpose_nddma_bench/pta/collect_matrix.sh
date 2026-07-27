#!/bin/bash
# LastTranspose dim0=56 四配置全量采集：DB/SB × isMte3 0/1，三 dtype（FP16+FP32+B8），per-case dim1。
# 单个 msprof 会话/配置，prof_pta_matrix.py 跑当前 .so 变体（USE_PINGPONG 决定容量跳过）下所有
# block<=容量 的用例（FP16 20 + FP32 12/16 + B8 20），op_summary 按执行顺序对齐 RUN manifest。
#
# 用法: bash collect_matrix.sh <op_root> <matrix_out> [iters]
#   op_root    : 算子根目录（含 op_kernel/ build/ op_pta/）
#   matrix_out : 采集输出目录（内部建 db_rw/db_r/sb_rw/sb_r 子目录）
#   iters      : 每用例测量次数（默认 5，另加 1 预热）
#
# 前置: op_kernel .asc 的 USE_PINGPONG 决定 DB/SB；本脚本自动切换 + 重编 .so + 清 torch ext 缓存。
set -e

OP_ROOT="$(cd "${1:?need op_root}" && pwd)"
MATRIX_OUT="$2"
ITERS="${3:-5}"
mkdir -p "$MATRIX_OUT"
MATRIX_OUT="$(cd "$MATRIX_OUT" && pwd)"

# registry-invoke 迁移：脚本与 pta harness 同在 pta/；compute 逻辑迁至 op_kernel/arch35/*.h
# （USE_PINGPONG 开关随之迁到该头），launch .so 独立建在 pta/build（非算子根 build/）。
HERE="$OP_ROOT/pta"
KERNEL_ASC="$(ls "$OP_ROOT"/op_kernel/arch35/*.h | head -1)"
BUILD="$HERE/build"

set_pingpong() {  # $1 = true|false
    sed -i "s/constexpr bool USE_PINGPONG = \(true\|false\);/constexpr bool USE_PINGPONG = $1;/" "$KERNEL_ASC"
    echo "[toggle] USE_PINGPONG=$1"
    grep -n "USE_PINGPONG = " "$KERNEL_ASC"
}

rebuild_so() {
    rm -rf "$BUILD"
    mkdir -p "$BUILD"
    ( cd "$BUILD" && cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DBUILD_PTA_LAUNCH=ON "$HERE" >/dev/null && make -j nddma_transpose_launch >/dev/null )
    echo "[build] $(ls -la "$BUILD"/libnddma_transpose_launch.so | awk '{print $5}') bytes"
    find "$HOME/.cache/torch_extensions" -maxdepth 3 -name "nddma_transpose_ext" -type d -exec rm -rf {} + 2>/dev/null || true
}

collect_cfg() {  # $1=mode(db|sb) $2=cfg $3=isMte3
    local mode="$1" cfg="$2" mte3="$3"
    local out="$MATRIX_OUT/$cfg"
    rm -rf "$out"; mkdir -p "$out"
    ( cd "$HERE" && msprof --output="$out" \
        --aic-metrics=PipeUtilization \
        --application="python3 prof_pta_matrix.py $mode $mte3 $ITERS" )
    echo "[collect] $cfg -> $out"
}

echo "=========== DB variant ==========="
set_pingpong true
rebuild_so
collect_cfg db db_rw 1
collect_cfg db db_r  0

echo "=========== SB variant ==========="
set_pingpong false
rebuild_so
collect_cfg sb sb_rw 1
collect_cfg sb sb_r  0

echo "=========== restore DB default ==========="
set_pingpong true
rebuild_so
echo "[done] matrix at $MATRIX_OUT"
