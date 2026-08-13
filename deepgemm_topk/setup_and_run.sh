#!/bin/bash
# setup_and_run.sh — DeepGEMM + TopK 融合 kernel 部署脚本
# 在 B200 环境上执行

set -e

echo "=== Step 1: Clone DeepGEMM ==="
git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git
cd DeepGEMM

echo "=== Step 2: Copy new files into tree ==="

# Device kernel files
cp /tmp/deepgemm_topk/topk_utils.cuh \
   deep_gemm/include/deep_gemm/topk_utils.cuh

cp /tmp/deepgemm_topk/sm100_mqa_logits_with_topk.cuh \
   deep_gemm/include/deep_gemm/impls/sm100_mqa_logits_with_topk.cuh

# Scheduler additions (merge into existing scheduler headers)
# 需要手动将 scheduler_additions.cuh 中的方法追加到:
#   deep_gemm/include/deep_gemm/scheduler/sm100_mqa_logits.cuh
#   deep_gemm/include/deep_gemm/scheduler/sm100_paged_mqa_logits.cuh
echo "NOTE: Manually merge scheduler_additions.cuh into scheduler headers"

# Host-side JIT dispatch
cp /tmp/deepgemm_topk/sm100_mqa_logits_with_topk.hpp \
   csrc/jit_kernels/impls/sm100_mqa_logits_with_topk.hpp

# Host-side API
cp /tmp/deepgemm_topk/attention_with_topk.hpp \
   csrc/apis/attention_with_topk.hpp

echo "=== Step 3: Patch C++ entry point ==="
# 在 csrc/python_api.cpp 中添加 include 和注册调用:
#   #include "apis/attention_with_topk.hpp"
#   deep_gemm::attention::register_topk_apis(m);

echo "=== Step 4: Patch Python __init__.py ==="
# 将 python_api_additions.py 中的函数追加到 deep_gemm/__init__.py

echo "=== Step 5: Build ==="
cat develop.sh
./develop.sh

echo "=== Step 6: Run test ==="
python /tmp/deepgemm_topk/test_topk.py

echo "=== Done ==="
