#!/bin/bash
# 安装改后包 + 改前/改后对照跑（含 DEBUG 分支证据与缓存命中证据）
set -u
WORK=/home/wchao00893980/deepseek1009
PREV=/home/wchao00893980/scatter1009
PKG=$WORK/source/build_out/cann-ops-transformer-deepseek1009_linux-aarch64.run
INSTALL=$WORK/opp-install

echo "=== 1. 安装改后包 ==="
rm -rf $INSTALL
mkdir -p $INSTALL
bash $PKG --quiet --install-path=$INSTALL > $WORK/install.log 2>&1
echo "install exit=$?"
tail -3 $WORK/install.log
SETENV_FIX=$INSTALL/vendors/deepseek1009_transformer/bin/set_env.bash
echo "--- set_env.bash"; cat $SETENV_FIX
echo "--- 包内新接口报错串（应为 WithAllocOut）"
strings $INSTALL/vendors/deepseek1009_transformer/op_api/lib/libcust_opapi.so | grep "AllocTensor for Scatter" | head -2
echo "--- 包内旧名残留（应为 0）"
strings $INSTALL/vendors/deepseek1009_transformer/op_api/lib/libcust_opapi.so | grep -c "OutOfPlace"
SETENV_BASE=$PREV/opp-unmodified/vendors/scatter1009_transformer/bin/set_env.bash
echo "--- 对照包（改前）"; cat $SETENV_BASE

echo
echo "=== 2. 环境准备 ==="
source /root/miniconda3/etc/profile.d/conda.sh
conda activate wangchao
cd $WORK
export ACLNN_CACHE_LIMIT=1024
python -c "import torch, torch_npu; print('torch', torch.__version__, 'torch_npu', torch_npu.__version__)"

run() {  # name setenv cases loglevel iters
  local name=$1 setenv=$2 cases=$3 loglevel=$4 iters=$5
  local outdir=$WORK/runs/$name
  local case_flags=""
  for c in $cases; do case_flags="$case_flags --case $c"; done
  mkdir -p $outdir
  ( source $setenv
    export ASCEND_GLOBAL_LOG_LEVEL=$loglevel
    export ASCEND_SLOG_PRINT_TO_STDOUT=1
    python $WORK/test_scatter_pa_kv_cache_paths.py $case_flags --iterations $iters \
      --label $name --output $outdir/results > $outdir/stdout.log 2>&1
    echo $? > $outdir/exit )
  echo
  echo "=== $name  exit=$(cat $outdir/exit)  (日志级别=$loglevel, 轮次=$iters) ==="
  printf "    same-address=%s  move-align=%s  Get-cache-key=%s  Add-cache-key=%s  relation-fail=%s  AbandonCache(字面)=%s\n" \
    "$(grep -c 'same address' $outdir/stdout.log)" \
    "$(grep -c 'move align' $outdir/stdout.log)" \
    "$(grep -c 'Get op cache key' $outdir/stdout.log)" \
    "$(grep -c 'Add op cache key' $outdir/stdout.log)" \
    "$(grep -c 'tensor relation fail' $outdir/stdout.log)" \
    "$(grep -ci 'abandon' $outdir/stdout.log)"
  echo "   测试输出摘要:"
  sed -n '/===== 汇总 =====/,/结果目录/p' $outdir/stdout.log | head -14 | sed 's/^/     /'
}

echo
echo "########## A) 连续 cache 场景 + DEBUG（抓分支证据） ##########"
run baseline-norm "$SETENV_BASE" norm_contiguous 0 5
run fixed-norm    "$SETENV_FIX"  norm_contiguous 0 5

echo
echo "########## B) 四个场景全跑（ERROR 级，用于数值 + 耗时对比） ##########"
run baseline-all "$SETENV_BASE" "norm_contiguous pa_nz_contiguous strided_cache key_only" 3 20
run fixed-all    "$SETENV_FIX"  "norm_contiguous pa_nz_contiguous strided_cache key_only" 3 20

echo
echo "=== 3. 耗时对比（同一日志级别，看第2..N轮中位数） ==="
for name in baseline-all fixed-all; do
  echo "--- $name"
  column -s, -t $WORK/runs/$name/results/summary.csv 2>/dev/null || cat $WORK/runs/$name/results/summary.csv
done
