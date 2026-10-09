#!/bin/bash
# 看 op-plugin(op_api_common.h) 里何时请求 aclnn 缓存
PY=/root/miniconda3/envs/wangchao/lib/python3.11/site-packages
OP=$PY/torch_npu/include/third_party/op-plugin
F=$OP/op_plugin/utils/op_api_common.h

echo "=== 头文件树 ==="
find $OP -name "*.h" | head -20

echo
echo "=== op_api_common.h 里缓存相关行 ==="
grep -n -iE "cache|SetPTA" $F | head -60

echo
echo "=== 关键片段：EXEC_NPU_CMD 定义与缓存开关 ==="
grep -n "define EXEC_NPU_CMD\|define EXEC_NPU_CMD_FAST\|GetOpApiFuncAddr\|CheckInplace\|inplace" $F | head -30

echo
echo "=== 把 EXEC_NPU_CMD 前后 60 行打出来 ==="
awk '/define EXEC_NPU_CMD/{found=NR} found && NR>=found-10 && NR<=found+70 {print NR": "$0}' $F | head -90

echo
echo "=== torch_npu python 侧与 aclnn 缓存有关的开关 ==="
grep -rn -iE "aclnn.*cache|cache.*aclnn|SetPTACacheHashKey" $PY/torch_npu --include=*.py 2>/dev/null | head -10
strings $PY/torch_npu/lib/libtorch_npu.so | grep -iE "ACLNN_CACHE|cache_limit|disable.*cache|enable.*cache" | sort -u | head -10

echo
echo "=== libtorch_npu.so 里缓存判定相关字符串 ==="
strings $PY/torch_npu/lib/libtorch_npu.so | grep -iE "inplace|in_place" | sort -u | head -20
