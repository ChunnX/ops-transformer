#!/bin/bash
# 在已跑的 DEBUG 日志里定位"缓存被谁关掉"
R=/home/wchao00893980/deepseek1009/runs
L=$R/baseline-norm/stdout.log

echo "=== 1. profiling 相关 ==="
grep -iE "RecordOpArgCallback|level2ProfilingFlag|opProfiling|cacheHasFull" $L | head -8 | cut -c1-200

echo
echo "=== 2. dump 相关 ==="
grep -iE "AdumpGetDumpSwitch|dump switch|isOpDumpEnable|DumpEnable|exception dump|overflow dump" $L | head -8 | cut -c1-200

echo
echo "=== 3. 缓存函数与命中尝试 ==="
for pat in "PTAGetExecCache" "SetPTAHashKey" "SetPTACacheHashKey" "CanUsePTACache" "InitPTACacheThreadLocal" "GetFromCache" "AddOpExecCache" "no op cache in context" "OpExecCache"; do
  printf "  %-26s %s\n" "$pat" "$(grep -c "$pat" $L)"
done

echo
echo "=== 4. 前 3 次调用之间的 opbase 日志（带时间戳，看有没有缓存尝试） ==="
grep -nE "op_cache|OpExecCache|PTACache|DumpToCache|CanUse" $L | head -12 | cut -c1-210

echo
echo "=== 5. 环境里是否有 profiling/dump 相关的 python 侧开关被 torch_npu 打开 ==="
grep -iE "profiling|profil" $L | grep -viE "register|kernel binary|op_impl" | head -8 | cut -c1-200
