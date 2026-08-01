#!/bin/bash
# round_004 四配置采集：DB/SB × isMte3 0/1，per-case dim1。
# 用法: bash collect_single.sh <db|sb> <cfg_name:db_rw|db_r|sb_rw|sb_r> <isMte3> [iters]
# 前置: 已按 <db|sb> 编好 build/libnlast_transpose_launch.so（USE_PINGPONG 一致）。
set -e
MODE=${1:?need db|sb}
CFG=${2:?need cfg name}
ISMTE3=${3:?need isMte3}
ITERS=${4:-5}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
OUT="$ROOT/output/matrix/$CFG"
export PATH="/workspace/huangqiang/hq_claude/.local/bin:$PATH"

rm -rf "$OUT"
mkdir -p "$OUT"

cd "$HERE"
msprof --output="$OUT" \
       --aic-metrics=PipeUtilization \
       --application="python3 prof_pta_matrix.py $MODE $ISMTE3 $ITERS"
echo "[collect] $CFG done -> $OUT"
