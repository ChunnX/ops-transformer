#!/bin/bash
# (A) 查缓存为什么没启用  (B) 交替 A/B 计时（每侧多轮，报 min/median）
WORK=/home/wchao00893980/deepseek1009
PREV=/home/wchao00893980/scatter1009
R=$WORK/runs
SETENV_FIX=$WORK/opp-install/vendors/deepseek1009_transformer/bin/set_env.bash
SETENV_BASE=$PREV/opp-unmodified/vendors/scatter1009_transformer/bin/set_env.bash
export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

echo "=== A. 两份 DEBUG 日志里的缓存相关行全文 ==="
for name in baseline-norm fixed-norm; do
  echo "--- $name"
  grep -iE "op cache|OpExecCache|cacheLimit|Cannot (add|get) op cache" $R/$name/stdout.log \
    | grep -v "改前会打印" | sed 's/^/    /' | cut -c1-230 | head -12
done

echo
echo "=== B. 交替 A/B 计时（norm_contiguous，ERROR 级，3 轮 × 20 次） ==="
source /root/miniconda3/etc/profile.d/conda.sh
conda activate wangchao
export ACLNN_CACHE_LIMIT=1024
cd $WORK

one() {  # name setenv round
  local name=$1 setenv=$2 round=$3
  local out=$R/timing-$name-$round
  rm -rf $out; mkdir -p $out
  ( set +u; source $setenv
    export ASCEND_GLOBAL_LOG_LEVEL=3
    export ASCEND_SLOG_PRINT_TO_STDOUT=0
    python $WORK/test_scatter_pa_kv_cache_paths.py --case norm_contiguous --iterations 20 \
      --label $name-$round --output $out > $out/stdout.log 2>&1 )
  python3 - "$out/results.json" "$name-$round" <<'PY'
import json,sys,statistics
d=json.load(open(sys.argv[1]))
rows=d["cases"]["norm_contiguous"]["rows"]
host=[r["host_us"] for r in rows][1:]
print(f"{sys.argv[2]:18s} 轮次={len(host)+1} 第1轮={rows[0]['host_us']:9.1f}us  第2..N: min={min(host):8.1f} median={statistics.median(host):8.1f} max={max(host):9.1f}")
PY
}

for round in 1 2 3; do
  one baseline "$SETENV_BASE" $round
  one fixed    "$SETENV_FIX"  $round
done

echo
echo "=== C. 汇总（把 3 轮的第2..N轮数据合起来看） ==="
python3 - "$R" <<'PY'
import json,glob,statistics,os
root=sys.argv[1]
for side in ("baseline","fixed"):
    vals=[]
    for f in sorted(glob.glob(os.path.join(root,f"timing-{side}-*/results.json"))):
        rows=json.load(open(f))["cases"]["norm_contiguous"]["rows"]
        vals += [r["host_us"] for r in rows][1:]
    vals.sort()
    print(f"  {side:9s} n={len(vals):3d}  min={vals[0]:8.1f}  p50={statistics.median(vals):8.1f}  p90={vals[int(len(vals)*0.9)]:8.1f}  max={vals[-1]:9.1f}")
PY
