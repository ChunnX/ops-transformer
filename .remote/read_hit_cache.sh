#!/bin/bash
# 读 op-plugin 的 hit_cache / hit_cache_v2 全文（找 inplace 判定）
PY=/root/miniconda3/envs/wangchao/lib/python3.11/site-packages
F=$PY/torch_npu/include/third_party/op-plugin/op_plugin/utils/op_api_common.h
echo "=== op_api_common.h 140-305 行 ==="
sed -n '140,305p' $F | cat -n | sed 's/^/   /'
