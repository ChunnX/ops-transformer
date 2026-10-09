#!/bin/bash
# 确认 scatter 适配层用的宏是否请求缓存
PY=/root/miniconda3/envs/wangchao/lib/python3.11/site-packages
F=$PY/torch_npu/include/third_party/op-plugin/op_plugin/utils/op_api_common.h

echo "=== EXEC_NPU_NO_FORMAT_CHECK_CMD 定义位置 ==="
grep -n "define EXEC_NPU_NO_FORMAT_CHECK_CMD" $F

echo
echo "=== 该宏全文（从定义到下一个 #define） ==="
start=$(grep -n "define EXEC_NPU_NO_FORMAT_CHECK_CMD" $F | head -1 | cut -d: -f1)
sed -n "${start},$((start+95))p" $F | grep -nE "hit_cache|SetPTAHashKey|setPTAHashKeyFunc|InitPTACache|CanUse|getWorkspaceSize|do \{|while" | head -20

echo
echo "=== 对比：EXEC_NPU_CMD(435 行附近) 与 EXEC_NPU_CMD_V2 里是否调用 hit_cache ==="
awk '/define EXEC_NPU_CMD\(/{f=NR} f&&NR>=f&&NR<=f+90{print NR": "$0}' $F | grep -nE "hit_cache|SetPTAHashKey|setPTAHashKeyFunc\(0\)|CanUsePTACache" | head -10
echo "--- V2:"
awk '/define EXEC_NPU_CMD_V2/{f=NR} f&&NR>=f&&NR<=f+120{print NR": "$0}' $F | grep -nE "hit_cache|setPTACacheHashKeyFunc|CanUsePTACache" | head -10

echo
echo "=== 结论核对：装出来的 torch_npu 里，npu_scatter_pa_kv_cache 用的是哪个宏 ==="
strings $PY/torch_npu/lib/libtorch_npu.so | grep -c "EXEC_NPU_NO_FORMAT_CHECK_CMD"
strings $PY/torch_npu/lib/libtorch_npu.so | grep -oE "PTAGetExecCache|PTAFindExecCache|SetPTACacheHashKey" | sort | uniq -c
