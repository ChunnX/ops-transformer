# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Benchmark: fused_moe_small_bs_w4a16 vs 官方 npu_fused_experts W4A16 路径。

对标 moe_ascend_npu/tests/benchmark_fused_moe.py：
自包含绑定的 ``fused_moe_small_bs_w4a16``（小 batch decode 路径）
与官方 ``npu_fused_experts`` W4A16 antiquant 路径（大 batch 路径）对比。

两条路径各自捕获在一个 NPU Graph 中，每次 replay 执行 16 个路由块，
用 NPU events 计时。

运行方式（需先安装 custom_transformer 算子包）：
    export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
    export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
    python3 benchmark_fused_moe_small_bs_w4a16.py [--batch-sizes 1 2 4 8] [--json-out out.json]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import pack_int4

from _binding import fused_moe_small_bs_w4a16

# ------------------------------------------------------------
# Fixed constants（与 moe_ascend_npu benchmark 对齐）
# ------------------------------------------------------------
TOP_K = 8
NUM_EXPERTS = 128
HIDDEN_SIZE = 2048
INTER_SIZE = 768
GROUP_SIZE = 32
ROUTING_BLOCKS = NUM_EXPERTS // TOP_K  # 16

# Timing configuration
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100

device = torch.device("npu:0")
dtype = torch.float16


# ------------------------------------------------------------
# NPU timing with events
# ------------------------------------------------------------
def measure_time_with_events(fn, num_iters: int = 100) -> float:
    """Measure NPU execution time using events."""
    for _ in range(WARMUP_ITERATIONS):
        fn()
    torch.npu.synchronize()

    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        fn()
    end_event.record()
    torch.npu.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / num_iters
    return elapsed_ms


# ------------------------------------------------------------
# Reference implementation (official npu_fused_experts W4A16 path)
# ------------------------------------------------------------
def npu_fused_experts(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    **kwargs,
):
    w13_offset = kwargs["w13_offset"]
    w2_offset = kwargs["w2_offset"]
    row_idx = kwargs["row_idx"]

    original_dtype = hidden_states.dtype

    # ---- init routing (row_idx MUST be 2D) ----
    hidden_states, expanded_row_idx, expanded_expert_idx = (
        torch_npu.npu_moe_init_routing(
            hidden_states,
            row_idx=row_idx,
            expert_idx=topk_ids,
            active_num=hidden_states.shape[0],
        )
    )

    expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
        expanded_expert_idx, w13.shape[0]
    ).to(torch.int64)

    # ---- W13 ----
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w13],
        antiquant_scale=[w13_scale],
        antiquant_offset=[w13_offset],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    hidden_states = torch_npu.npu_swiglu(hidden_states)

    # ---- W2 ----
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        antiquant_scale=[w2_scale],
        antiquant_offset=[w2_offset],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    # ---- finalize ----
    return torch_npu.npu_moe_finalize_routing(
        hidden_states,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights.to(hidden_states.dtype),
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids,
    )


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def pack_int4_weights_official(raw, device):
    """官方 npu_fused_experts 路径的 int4pack 打包（[E, K, N] -> [E, K, N//8]）。"""
    flat = raw.view(-1, raw.shape[-1]).to(device)
    packed = torch_npu.npu_convert_weight_to_int4pack(flat)
    return packed.view(*raw.shape[:-1], raw.shape[-1] // 8)


def build_row_idx(B):
    row_idx_len = B * TOP_K
    return (
        torch.arange(row_idx_len, dtype=torch.int32, device=device)
        .view(TOP_K, -1)
        .permute(1, 0)
        .contiguous()
    )


def build_routing_blocks():
    experts = torch.arange(NUM_EXPERTS, device=device, dtype=torch.int32)
    return experts.view(ROUTING_BLOCKS, TOP_K)


def build_weight_blocks():
    w = torch.randn((ROUTING_BLOCKS, TOP_K), device=device)
    return torch.softmax(w, dim=-1)


# ------------------------------------------------------------
# Capture graph: 16 forwards inside ONE graph
# ------------------------------------------------------------
def capture_16_forward_graph(fn, x, row_idx, topk_ids_all, topk_w_all):
    # warmup
    for i in range(ROUTING_BLOCKS):
        fn(x, row_idx, topk_ids_all[i], topk_w_all[i])
    torch.npu.synchronize()

    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        for i in range(ROUTING_BLOCKS):
            fn(x, row_idx, topk_ids_all[i], topk_w_all[i])
    torch.npu.synchronize()
    return g


# ------------------------------------------------------------
# Bytes accounting (weight traffic)
# ------------------------------------------------------------
def bytes_per_expert(w13, w13_scale, w2, w2_scale):
    # Symmetric quantization: no offset tensors
    return (
        w13[0].numel() * w13.element_size()
        + w13_scale[0].numel() * w13_scale.element_size()
        + w2[0].numel() * w2.element_size()
        + w2_scale[0].numel() * w2_scale.element_size()
    )


# ------------------------------------------------------------
# Benchmark per batch size
# ------------------------------------------------------------
def benchmark_bs(B, w13_custom, w13_official, w13_scale, w13_offset,
                 w2_custom, w2_official, w2_scale, w2_offset):
    x = torch.randn((B, HIDDEN_SIZE), device=device, dtype=dtype)
    row_idx = build_row_idx(B)

    blocks = build_routing_blocks()
    weights = build_weight_blocks()

    # build 16 routing sets
    topk_ids_all, topk_w_all = [], []
    for i in range(ROUTING_BLOCKS):
        ids = torch.empty((B, TOP_K), device=device, dtype=torch.int32)
        wts = torch.empty((B, TOP_K), device=device)
        for b in range(B):
            blk = (i + b) % ROUTING_BLOCKS
            ids[b].copy_(blocks[blk])
            wts[b].copy_(weights[blk])
        topk_ids_all.append(ids)
        topk_w_all.append(wts)

    # reference graph (official npu_fused_experts W4A16 antiquant path)
    g_ref = capture_16_forward_graph(
        lambda x, row_idx, ids, w: npu_fused_experts(
            hidden_states=x,
            w13=w13_official, w13_scale=w13_scale,
            w2=w2_official, w2_scale=w2_scale,
            topk_weights=w,
            topk_ids=ids,
            top_k=TOP_K,
            w13_offset=w13_offset,
            w2_offset=w2_offset,
            row_idx=row_idx,
        ),
        x, row_idx, topk_ids_all, topk_w_all,
    )

    # custom fused graph (fused_moe_small_bs_w4a16 自包含绑定)
    g_custom = capture_16_forward_graph(
        lambda x, row_idx, ids, w: fused_moe_small_bs_w4a16(
            x,
            w13_custom, w13_scale,
            w2_custom, w2_scale,
            ids, w,
        ),
        x, row_idx, topk_ids_all, topk_w_all,
    )

    def run_ref():
        g_ref.replay()

    def run_custom():
        g_custom.replay()

    # Event-based timing
    t_ref_ms = measure_time_with_events(run_ref, BENCHMARK_ITERATIONS)
    t_custom_ms = measure_time_with_events(run_custom, BENCHMARK_ITERATIONS)

    bytes_total = (
        B * TOP_K * ROUTING_BLOCKS *
        bytes_per_expert(w13_custom, w13_scale, w2_custom, w2_scale)
    )

    return {
        "B": B,
        "ref_us": t_ref_ms * 1e3 / ROUTING_BLOCKS,
        "custom_us": t_custom_ms * 1e3 / ROUTING_BLOCKS,
        "ref_GBs": bytes_total / (t_ref_ms * 1e-3) / 1e9,
        "custom_GBs": bytes_total / (t_custom_ms * 1e-3) / 1e9,
        "speedup": t_ref_ms / t_custom_ms,
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    torch.manual_seed(0)

    # 原始有符号 int4 权重 [E, K, N]（两条路径共用）
    raw_w13 = torch.randint(-8, 8, (NUM_EXPERTS, HIDDEN_SIZE, 2 * INTER_SIZE), dtype=torch.int8)
    raw_w2 = torch.randint(-8, 8, (NUM_EXPERTS, INTER_SIZE, HIDDEN_SIZE), dtype=torch.int8)

    # custom 路径：kernel 原生布局 int32 [E, K, N//8]（int4 补码、低位在前）
    print("packing weights for custom path ...")
    w13_custom = torch.from_numpy(pack_int4(raw_w13.numpy())).to(device)
    w2_custom = torch.from_numpy(pack_int4(raw_w2.numpy())).to(device)

    # 官方路径：CANN int4pack 布局
    print("packing weights for official path ...")
    w13_official = pack_int4_weights_official(raw_w13.to(torch.int32), device)
    w2_official = pack_int4_weights_official(raw_w2.to(torch.int32), device)
    del raw_w13, raw_w2

    w13_scale = torch.randn(
        (NUM_EXPERTS, HIDDEN_SIZE // GROUP_SIZE, 2 * INTER_SIZE),
        device=device, dtype=dtype
    ) * 0.125
    w13_offset = torch.zeros_like(w13_scale)

    w2_scale = torch.randn(
        (NUM_EXPERTS, INTER_SIZE // GROUP_SIZE, HIDDEN_SIZE),
        device=device, dtype=dtype
    ) * 0.125
    w2_offset = torch.zeros_like(w2_scale)

    print("BS | Ref(us) | Custom(us) | Ref GB/s | Custom GB/s | Speedup")
    print("-" * 68)

    results = []
    for B in args.batch_sizes:
        r = benchmark_bs(B, w13_custom, w13_official, w13_scale, w13_offset,
                         w2_custom, w2_official, w2_scale, w2_offset)
        results.append(r)
        print(
            f"{B:>2} | {r['ref_us']:>7.2f} | {r['custom_us']:>10.2f} | "
            f"{r['ref_GBs']:>8.2f} | {r['custom_GBs']:>10.2f} | {r['speedup']:>7.2f}x"
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "benchmark": "fused_moe_small_bs_w4a16",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_shape": {
                "hidden_size": HIDDEN_SIZE,
                "moe_intermediate_size": INTER_SIZE,
                "num_experts": NUM_EXPERTS,
                "top_k": TOP_K,
                "group_size": GROUP_SIZE,
            },
            "environment": {
                "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                "torch_version": torch.__version__,
                "torch_npu_version": torch_npu.__version__,
            },
            "settings": {
                "routing_blocks_per_graph": ROUTING_BLOCKS,
                "warmup_iterations": WARMUP_ITERATIONS,
                "benchmark_iterations": BENCHMARK_ITERATIONS,
            },
            "reference": "official npu_fused_experts W4A16 antiquant path",
            "custom": "fused_moe_small_bs_w4a16 (experimental self-contained binding)",
            "results": results,
        }
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"JSON result: {args.json_out}")


if __name__ == "__main__":
    main()
