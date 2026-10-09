#!/bin/bash
# 细节分析：per-iteration 耗时、加载的库、缓存相关日志
WORK=/home/wchao00893980/deepseek1009
R=$WORK/runs

echo "=== 1. 每轮 host 耗时（norm_contiguous） ==="
for name in baseline-all fixed-all; do
  echo "--- $name"
  python3 - "$R/$name/results/results.json" <<'PY'
import json,sys,statistics
d=json.load(open(sys.argv[1]))
case=d["cases"]["norm_contiguous"]
rows=case["rows"]
host=[r["host_us"] for r in rows]
later=sorted(host[1:])
print(f"  轮次={len(host)}  第1轮={host[0]:.1f}us")
print(f"  第2..N: min={later[0]:.1f} median={statistics.median(later):.1f} max={later[-1]:.1f}")
print("  全部:", " ".join(f"{v:.0f}" for v in host))
PY
done

echo
echo "=== 2. 各 run 实际加载的 op_api 库（确认 A/B 没串） ==="
for name in baseline-norm fixed-norm baseline-all fixed-all; do
  echo "--- $name"
  grep -o "/home/[^ ]*libcust_opapi.so" $R/$name/stdout.log 2>/dev/null | sort -u | head -3
  grep -o "/home/[^ ]*libopapi_transformer.so" $R/$name/stdout.log 2>/dev/null | sort -u | head -2
done

echo
echo "=== 3. 缓存相关日志（DEBUG 的两份） ==="
for name in baseline-norm fixed-norm; do
  echo "--- $name"
  for pat in "op cache" "OpExecCache" "Cannot add op cache" "Cannot get op cache" "Disable op cache" "PTA" "SetOpCacheKey" "Add op cache" "Get op cache"; do
    n=$(grep -ci "$pat" $R/$name/stdout.log 2>/dev/null)
    [ "$n" != "0" ] && echo "    $pat: $n"
  done
  echo "    (只列出非 0 项)"
done

echo
echo "=== 4. baseline-norm 里 same-address 的真实日志行（前 4 条） ==="
grep "same address" $R/baseline-norm/stdout.log | grep -v "改前会打印" | head -4 | cut -c1-230
echo "计数（排除提示文本）: $(grep 'same address' $R/baseline-norm/stdout.log | grep -vc '改前会打印')"
echo "--- fixed-norm 里同样排除提示文本后的计数（应为 0）"
grep 'same address' $R/fixed-norm/stdout.log | grep -vc '改前会打印'

echo
echo "=== 5. 改后 DEBUG 日志里 ViewCopy 的行（看走了哪个分支） ==="
grep "ViewCopy" $R/fixed-norm/stdout.log | grep -E "contiguous.cpp|data addr" | head -6 | cut -c1-220
echo "--- 是否有 AddTensorRelation / BindWorkspace 类似痕迹"
grep -c "AddTensorRelation" $R/fixed-norm/stdout.log
grep -i "relation" $R/fixed-norm/stdout.log | head -3 | cut -c1-200

echo
echo "=== 6. key_only 报错是否两边一致（应为同样缺 siso binary） ==="
for name in baseline-all fixed-all; do
  echo "--- $name: $(grep -c 'Cannot find bin of op ScatterPaKvCache' $R/$name/stdout.log) 次"
done
