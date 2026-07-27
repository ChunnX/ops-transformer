#!/bin/bash
# DESIGN §7.6 5D 核数扫描采集：4 代表 case（AB_0/BA_0/BAB_0/ABA_0）× blockNum={1,2,4,8,16,28,56}。
# 复用满核采集骨架：单会话 toggle→rebuild→clear-cache→collect→restore（DB 默认）。
# NDDMA_SUITE=5D NDDMA_CORESCAN=1 驱动 prof_pta_matrix.run_corescan_5d。
# 带宽分母取最慢核 t（整卡墙钟=Task Duration），分子恒满 numel_dst（§7.6b）。
#
# 用法: bash collect_corescan.sh <out_dir> [iters] [mode] [isMte3]
#   out_dir : 采集输出目录（内部建 <cfg> 子目录）
#   iters   : 每 (case,blockNum) 测量次数（默认 5，另加 1 预热）
#   mode    : db|sb（默认 db，DB 双缓冲代表并行扩展性）
#   isMte3  : 1 读写 / 0 纯读（默认 1）
set -e

OP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?need out_dir}"
ITERS="${2:-5}"
MODE="${3:-db}"
MTE3="${4:-1}"
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"

# registry-invoke 迁移：脚本与 pta harness 同在 pta/；compute 逻辑迁至 op_kernel/arch35/*.h
# （USE_PINGPONG 开关随之迁到该头），launch .so 独立建在 pta/build（非算子根 build/）。
HERE="$OP_ROOT/pta"
KERNEL_ASC="$(ls "$OP_ROOT"/op_kernel/arch35/*.h | head -1)"
BUILD="$HERE/build"
export PATH="/workspace/huangqiang/hq_claude/.local/bin:$PATH"
export NDDMA_SUITE=5D NDDMA_CORESCAN=1

set_pingpong() {
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

pp=$([ "$MODE" = "sb" ] && echo false || echo true)
cfg="${MODE}_$([ "$MTE3" = "1" ] && echo rw || echo r)"
out="$OUT/$cfg"; rm -rf "$out"; mkdir -p "$out"

echo "=========== corescan5d mode=$MODE isMte3=$MTE3 cfg=$cfg ==========="
set_pingpong "$pp"
rebuild_so
( cd "$HERE" && NDDMA_SUITE=5D NDDMA_CORESCAN=1 msprof --output="$out" \
    --aic-metrics=PipeUtilization \
    --application="python3 prof_pta_matrix.py $MODE $MTE3 $ITERS" )
echo "[collect] $cfg -> $out"

echo "=========== restore DB default ==========="
set_pingpong true
rebuild_so
echo "[done] corescan5d at $OUT"
