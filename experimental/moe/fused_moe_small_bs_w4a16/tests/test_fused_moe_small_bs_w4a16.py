# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""fused_moe_small_bs_w4a16 正确性测试（含 NPU Graph 捕获验证）。

对照组为纯 PyTorch 参考实现（复用同一 int4 原始权重与分组 scale，FP32 累加）。
另外验证算子可被 torch.npu.NPUGraph 捕获且 replay 结果与 eager 一致，
这是接入 vLLM aclgraph decode 路径的前提。

运行方式（需先安装 custom_transformer 算子包，绑定为目录内自包含 JIT 扩展）：
    export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
    export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
    python3 test_fused_moe_small_bs_w4a16.py
    # 或 pytest test_fused_moe_small_bs_w4a16.py
"""

import os
import sys

import numpy as np
import torch
import torch_npu  # noqa: F401  (registers NPU device)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import fused_moe_w4a16_ref, pack_int4
from _binding import fused_moe_small_bs_w4a16

TOP_K = 8
NUM_EXPERTS = 16
HIDDEN_SIZE = 2048  # 128 的倍数
INTER_SIZE = 768    # 128 的倍数

DEVICE = "npu:0"
# 逻辑错误会产生 O(1) 量级偏差，fp16 分组累加误差远小于该阈值
MAX_ABS_DIFF_THRESHOLD = 0.5


def _make_inputs(batch_size, dtype, seed=42):
    torch.manual_seed(seed)
    x = torch.randn(batch_size, HIDDEN_SIZE, dtype=dtype, device=DEVICE) * (1.0 / HIDDEN_SIZE**0.5)
    expert_ids = torch.randint(0, NUM_EXPERTS, (batch_size, TOP_K), dtype=torch.int32, device=DEVICE)
    topk_weights = torch.softmax(torch.randn(batch_size, TOP_K, dtype=torch.float32, device=DEVICE), dim=-1)

    # 原始有符号 int4 权重（kernel 原生布局 [E, K, N]），CPU 生成后搬运
    raw_w13 = torch.randint(-8, 8, (NUM_EXPERTS, HIDDEN_SIZE, 2 * INTER_SIZE), dtype=torch.int8).to(DEVICE)
    raw_w2 = torch.randint(-8, 8, (NUM_EXPERTS, INTER_SIZE, HIDDEN_SIZE), dtype=torch.int8).to(DEVICE)

    # 打包为 kernel 布局 int32 [E, K, N//8]（int4 补码、低位在前）
    w13_weight = torch.from_numpy(pack_int4(raw_w13.cpu().numpy())).to(DEVICE)
    w2_weight = torch.from_numpy(pack_int4(raw_w2.cpu().numpy())).to(DEVICE)

    # 分组 scale [E, K//32, N]，取小的正值保证数值稳定
    scale13 = torch.rand(NUM_EXPERTS, HIDDEN_SIZE // 32, 2 * INTER_SIZE, dtype=torch.float32, device=DEVICE)
    scale13 = (scale13 * 0.025 + 0.005).to(dtype)
    scale2 = torch.rand(NUM_EXPERTS, INTER_SIZE // 32, HIDDEN_SIZE, dtype=torch.float32, device=DEVICE)
    scale2 = (scale2 * 0.025 + 0.005).to(dtype)

    return x, w13_weight, scale13, w2_weight, scale2, expert_ids, topk_weights, raw_w13, raw_w2


def _run_one(batch_size, dtype):
    (x, w13_weight, scale13, w2_weight, scale2, expert_ids, topk_weights,
     raw_w13, raw_w2) = _make_inputs(batch_size, dtype)

    # eager 调用（首次触发 JIT 编译）
    _ = fused_moe_small_bs_w4a16(
        x, w13_weight, scale13, w2_weight, scale2, expert_ids, topk_weights)
    y_custom = fused_moe_small_bs_w4a16(
        x, w13_weight, scale13, w2_weight, scale2, expert_ids, topk_weights)
    torch.npu.synchronize()

    # 纯 PyTorch 参考（FP32）
    y_ref = fused_moe_w4a16_ref(x, raw_w13, scale13, raw_w2, scale2, expert_ids, topk_weights)

    diff = (y_custom.float() - y_ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel = max_diff / (y_ref.abs().max().item() + 1e-6)
    print(
        f"[{dtype} BS={batch_size}] max_diff={max_diff:.4f} mean_diff={mean_diff:.6f} "
        f"rel_to_absmax={rel:.4f} (E={NUM_EXPERTS} K={TOP_K} H={HIDDEN_SIZE} I={INTER_SIZE})"
    )
    assert max_diff < MAX_ABS_DIFF_THRESHOLD, (
        f"fused_moe_small_bs_w4a16 ({dtype}, BS={batch_size}) max_diff {max_diff} too large")

    return x, w13_weight, scale13, w2_weight, scale2, expert_ids, topk_weights


def _run_npu_graph_check(batch_size, dtype):
    """验证算子可被 torch.npu.NPUGraph 捕获，replay 结果与 eager 一致。"""
    (x, w13_weight, scale13, w2_weight, scale2, expert_ids, topk_weights,
     _, _) = _make_inputs(batch_size, dtype, seed=123)

    def run_op():
        return fused_moe_small_bs_w4a16(
            x, w13_weight, scale13, w2_weight, scale2, expert_ids, topk_weights)

    # warmup + eager 对照
    for _ in range(3):
        y_eager = run_op()
    torch.npu.synchronize()

    # 捕获 graph 并 replay
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        y_graph = run_op()
    torch.npu.synchronize()
    g.replay()
    torch.npu.synchronize()

    diff = (y_graph.float() - y_eager.float()).abs().max().item()
    print(f"[{dtype} BS={batch_size}] npu_graph replay vs eager max_diff={diff:.6f}")
    assert diff < 1e-2, f"npu_graph replay mismatch: {diff}"
    print(f"[{dtype} BS={batch_size}] NPUGraph capture/replay OK")


def test_fused_moe_small_bs_w4a16_fp16():
    for bs in (1, 2, 4):
        _run_one(bs, torch.float16)


def test_fused_moe_small_bs_w4a16_bf16():
    for bs in (1, 2, 4):
        _run_one(bs, torch.bfloat16)


def test_fused_moe_small_bs_w4a16_npu_graph():
    _run_npu_graph_check(2, torch.float16)
    _run_npu_graph_check(2, torch.bfloat16)


if __name__ == "__main__":
    test_fused_moe_small_bs_w4a16_fp16()
    test_fused_moe_small_bs_w4a16_bf16()
    test_fused_moe_small_bs_w4a16_npu_graph()
    print("\nfused_moe_small_bs_w4a16 all tests PASSED")
