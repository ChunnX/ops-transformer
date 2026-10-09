#!/bin/bash
# 找到 op-plugin 侧"是否请求缓存"的判定
SRC=/home/wchao00893980/deepseek1009/source
OP=$SRC/third_party/op-plugin

echo "=== op-plugin 目录 ==="
ls -d $OP 2>/dev/null || echo "不存在：$OP"
find $SRC/third_party -maxdepth 2 -iname "*op-plugin*" -o -maxdepth 2 -iname "*op_plugin*" 2>/dev/null | head

echo
echo "=== SetPTACacheHashKey 出现位置 ==="
grep -rn "SetPTACacheHashKey" $SRC/third_party 2>/dev/null | head -10

echo
echo "=== op_api_common.h 里的缓存判定（关键字） ==="
for f in $(find $SRC/third_party -name "op_api_common.h" 2>/dev/null | head -3); do
  echo "--- $f"
  grep -n -iE "cache|inplace|in_place|IsView|SetPTA" "$f" | head -40
done

echo
echo "=== scatter 的 op-plugin 适配文件 ==="
f=$(find $SRC/third_party -name "ScatterPaKvCacheNpuOpApi.cpp" 2>/dev/null | head -1)
echo "文件: $f"
[ -n "$f" ] && sed -n '1,90p' "$f"

echo
echo "=== torch_npu 里与缓存相关的符号/字符串 ==="
PY=/root/miniconda3/envs/wangchao/lib/python3.11/site-packages
ls $PY | grep -i torch_npu | head
grep -rl "SetPTACacheHashKey" $PY/torch_npu 2>/dev/null | head -5
grep -rn "SetPTACacheHashKey" $PY/torch_npu/*.py 2>/dev/null | head -5
strings $PY/torch_npu/lib/libtorch_npu.so 2>/dev/null | grep -iE "SetPTACacheHashKey|op cache|aclnn cache" | sort -u | head -10
