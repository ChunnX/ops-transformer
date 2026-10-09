#!/bin/bash
# 只跑对照测试（包已装好）；注意不要 set -u：set_env.bash 会引用可能未定义的 ASCEND_CUSTOM_OPP_PATH
WORK=/home/wchao00893980/deepseek1009
PREV=/home/wchao00893980/scatter1009
SETENV_FIX=$WORK/opp-install/vendors/deepseek1009_transformer/bin/set_env.bash
SETENV_BASE=$PREV/opp-unmodified/vendors/scatter1009_transformer/bin/set_env.bash

export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate wangchao
cd $WORK
export ACLNN_CACHE_LIMIT=1024

run() {  # name setenv cases loglevel iters
  local name=$1 setenv=$2 cases=$3 loglevel=$4 iters=$5
  local outdir=$WORK/runs/$name
  local case_flags=""
  for c in $cases; do case_flags="$case_flags --case $c"; done
  rm -rf $outdir; mkdir -p $outdir
  (
    set +u
    source $setenv
    export ASCEND_GLOBAL_LOG_LEVEL=$loglevel
    export ASCEND_SLOG_PRINT_TO_STDOUT=1
    python $WORK/test_scatter_pa_kv_cache_paths.py $case_flags --iterations $iters \
      --label $name --output $outdir/results > $outdir/stdout.log 2>&1
    echo $? > $outdir/exit
  )
  echo
  echo "=== $name  exit=$(cat $outdir/exit 2>/dev/null)  (日志级别=$loglevel, 轮次=$iters) ==="
  printf "    same-address=%s  move-align=%s  Get-cache-key=%s  Add-cache-key=%s  relation-fail=%s  abandon(字面)=%s\n" \
    "$(grep -c 'same address' $outdir/stdout.log)" \
    "$(grep -c 'move align' $outdir/stdout.log)" \
    "$(grep -c 'Get op cache key' $outdir/stdout.log)" \
    "$(grep -c 'Add op cache key' $outdir/stdout.log)" \
    "$(grep -c 'tensor relation fail' $outdir/stdout.log)" \
    "$(grep -ci 'abandon' $outdir/stdout.log)"
  local hit
  hit=$(grep -m1 "same address" $outdir/stdout.log | cut -c1-200)
  [ -n "$hit" ] && echo "    命中示例: $hit"
  echo "    测试输出摘要:"
  sed -n '/===== 汇总 =====/,/结果目录/p' $outdir/stdout.log | head -14 | sed 's/^/     /'
}

echo "########## A) 连续 cache + DEBUG（分支证据） ##########"
run baseline-norm "$SETENV_BASE" norm_contiguous 0 5
run fixed-norm    "$SETENV_FIX"  norm_contiguous 0 5

echo
echo "########## B) 四场景全跑 + ERROR 级（数值 + 耗时对比） ##########"
run baseline-all "$SETENV_BASE" "norm_contiguous pa_nz_contiguous strided_cache key_only" 3 20
run fixed-all    "$SETENV_FIX"  "norm_contiguous pa_nz_contiguous strided_cache key_only" 3 20

echo
echo "=== 耗时对比（同级别；看第2..N轮中位数） ==="
for name in baseline-all fixed-all; do
  echo "--- $name"
  cat $WORK/runs/$name/results/summary.csv 2>/dev/null
done
