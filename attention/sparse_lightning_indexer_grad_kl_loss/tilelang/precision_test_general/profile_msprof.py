#!/usr/bin/env python3
"""
msprof 性能分析脚本 - SparseLightningIndexerGradKLLoss TileLang

用法:
  msprof --application="python profile_msprof.py" --output="./profiling_output" --model-execution=True

可选环境变量:
  PROFILE_CASE: 测试用例编号 (0-4), 默认 1
  PROFILE_ITERS: 迭代次数, 默认 5
"""

import torch
import torch_npu
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sparse_lightning_indexer_grad_kl_loss import sparse_lightning_indexer_grad_kl_loss
from precision_test_general import gen_inputs_corrected, TEST_CASES

CASE_IDX = int(os.environ.get("PROFILE_CASE", "1"))
ITERS = int(os.environ.get("PROFILE_ITERS", "5"))


def main():
    case = TEST_CASES[CASE_IDX]
    print(f"Profiling Case {CASE_IDX}: {case['name']}")
    print(f"配置: B={case['B']}, S1={case['S1']}, S2={case['S2']}, K={case['K']}, "
          f"N1={case['N1']}, Nidx1={case['Nidx1']}, layout={case['layout']}")
    print(f"迭代次数: {ITERS}")

    inputs = gen_inputs_corrected(case)

    # warmup
    print("Warmup...")
    for _ in range(3):
        sparse_lightning_indexer_grad_kl_loss(
            inputs["query"], inputs["key"],
            inputs["query_index"], inputs["key_index"],
            inputs["weights"], inputs["sparse_indices"],
            inputs["softmax_max"], inputs["softmax_sum"],
            inputs["scale_value"],
            query_rope=inputs["query_rope"],
            key_rope=inputs["key_rope"],
            layout_query=case["layout"],
            actual_seq_lengths_query=[case["S1"]] * case["B"] if case["layout"] == "BSND" else [case["S1"]],
            actual_seq_lengths_key=[case["S2"]] * case["B"] if case["layout"] == "BSND" else [case["S2"]],
            sparse_mode=3,
        )
        torch.npu.synchronize()

    # profiled iterations
    print(f"Running {ITERS} profiled iterations...")
    for i in range(ITERS):
        sparse_lightning_indexer_grad_kl_loss(
            inputs["query"], inputs["key"],
            inputs["query_index"], inputs["key_index"],
            inputs["weights"], inputs["sparse_indices"],
            inputs["softmax_max"], inputs["softmax_sum"],
            inputs["scale_value"],
            query_rope=inputs["query_rope"],
            key_rope=inputs["key_rope"],
            layout_query=case["layout"],
            actual_seq_lengths_query=[case["S1"]] * case["B"] if case["layout"] == "BSND" else [case["S1"]],
            actual_seq_lengths_key=[case["S2"]] * case["B"] if case["layout"] == "BSND" else [case["S2"]],
            sparse_mode=3,
        )
        torch.npu.synchronize()

    print("Done.")


if __name__ == "__main__":
    main()