#!/bin/bash
# DESIGN §9.6 5D 广播场景用例四配置全自动采集：DB/SB × isMte3 0/1（3D NDDMA 方案）。
# 单会话内串行 toggle→rebuild→clear-cache→collect→restore（DB 默认），不信任环境态。
# NDDMA_SUITE=5D 驱动 prof_pta_matrix.run_matrix_5d（RUN/SKIP 两判据 AND：tile=dim2*dim3*dim4≤cap
#   且 dim4*elem%8==0，§9.4）。AB_align_fp16 dim4=6 非对齐恒 SKIP，不发 kernel。
#
# 用法: bash collect_matrix_5d.sh <matrix_out> [iters]
#   matrix_out : 采集输出目录（内部建 db_rw/db_r/sb_rw/sb_r 子目录）
#   iters      : 每用例测量次数（默认 5，另加 1 预热）
set -e

OP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MATRIX_OUT="${1:?need matrix_out}"
ITERS="${2:-5}"
mkdir -p "$MATRIX_OUT"; MATRIX_OUT="$(cd "$MATRIX_OUT" && pwd)"

# registry-invoke 迁移：脚本与 pta harness 同在 pta/；compute 逻辑迁至 op_kernel/arch35/*.h
# （USE_PINGPONG 开关随之迁到该头），launch .so 独立建在 pta/build（非算子根 build/）。
HERE="$OP_ROOT/pta"
KERNEL_ASC="$(ls "$OP_ROOT"/op_kernel/arch35/*.h | head -1)"
BUILD="$HERE/build"
export PATH="/workspace/huangqiang/hq_claude/.local/bin:$PATH"
export NDDMA_SUITE=5D

set_pingpong() {  # $1 = true|false
    sed -i "s/constexpr bool USE_PINGPONG = \(true\|false\);/constexpr bool USE_PINGPONG = $1;/" "$KERNEL_ASC"
    echo "[toggle] USE_PINGPONG=$1  ($(grep -n 'USE_PINGPONG = ' "$KERNEL_ASC" | head -1))"
}

rebuild_so() {
    rm -rf "$BUILD"; mkdir -p "$BUILD"
    ( cd "$BUILD" && cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DBUILD_PTA_LAUNCH=ON "$HERE" >/dev/null \
        && make -j nddma_broadcast_launch >/dev/null )
    echo "[build] $(ls -la "$BUILD"/libnddma_broadcast_launch.so | awk '{print $5}') bytes  sha256=$(sha256sum "$BUILD"/libnddma_broadcast_launch.so | cut -c1-16)"
    find "$HOME/.cache/torch_extensions" -maxdepth 3 -name "nddma_broadcast_ext" -type d -exec rm -rf {} + 2>/dev/null || true
}

collect_cfg() {  # $1=mode(db|sb) $2=cfg $3=isMte3
    local mode="$1" cfg="$2" mte3="$3"
    local out="$MATRIX_OUT/$cfg"
    rm -rf "$out"; mkdir -p "$out"
    ( cd "$HERE" && NDDMA_SUITE=5D msprof --output="$out" \
        --aic-metrics=PipeUtilization \
        --application="python3 prof_pta_matrix.py $mode $mte3 $ITERS" )
    echo "[collect] $cfg -> $out"
}

echo "=========== DB variant (3D NDDMA, tile=dim2*dim3*dim4≤16384 + dim4 对齐) ==========="
set_pingpong true
rebuild_so
collect_cfg db db_rw 1
collect_cfg db db_r  0

echo "=========== SB variant (3D NDDMA, tile≤32768 + dim4 对齐) ==========="
set_pingpong false
rebuild_so
collect_cfg sb sb_rw 1
collect_cfg sb sb_r  0

echo "=========== restore DB default ==========="
set_pingpong true
rebuild_so
echo "[done] matrix5d at $MATRIX_OUT"
