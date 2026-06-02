# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
MoeDispatchPrefill standalone smoke test (layout -> fused notify + dispatch_normal).

Defaults match sgl-kernel-npu test_normal_and_fused_deep_moe.py / test_common.py.

Prerequisites on the NPU machine:
  1. Install the custom run package (notify_dispatch + dispatch_normal kernels).
  2. pip install / JIT npu_ops_transformer (rebuild after moe_dispatch_prefill.cpp changes).

Examples:
    torchrun --nproc_per_node=16 torch_extension/npu_ops_transformer/ops/test_dispatch_prefill.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
import torch_npu

import npu_ops_transformer  # noqa: F401
from npu_ops_transformer.ops.dispatch_layout import op_module as dispatch_layout_op_module
from npu_ops_transformer.ops.moe_dispatch_prefill import npu_moe_dispatch_prefill
from npu_ops_transformer.ops.moe_dispatch_prefill import op_module as prefill_op_module
from npu_ops_transformer.ops.test_utils import (
    SGL_HIDDEN,
    SGL_LOCAL_RANK_SIZE,
    SGL_NUM_EXPERTS,
    SGL_NUM_TOKENS,
    SGL_NUM_TOPK,
    SGL_PER_ROUND_TOKENS,
    SGL_ROUND,
    check_token_capacity,
    env_int,
    layout_round,
    make_moe_inputs,
)
from npu_ops_transformer.ops.update_context import update_context

EXPAND_IDX_PER_TOKEN = 3


class IntWrapper:
    def __init__(self, value: int = 0) -> None:
        self.value = value


def init_dist():
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    if world_size < 2:
        raise RuntimeError("WORLD_SIZE must be >= 2 for moe_dispatch_prefill communication test.")
    torch_npu.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    return local_rank, rank, world_size


def build_context(group, rank: int, world_size: int, ccl_buffer_size: IntWrapper) -> torch.Tensor:
    context_struct_size = 2 + 1024
    context_tensor_size = ((context_struct_size * 8) + 3) // 4
    context = torch.zeros((context_tensor_size,), dtype=torch.int32, device="npu")
    group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank, init_comm=False)
    ok = update_context(group_name, world_size, ccl_buffer_size, context)
    if not ok:
        raise RuntimeError("update_context failed.")
    if ccl_buffer_size.value <= 0:
        raise RuntimeError(f"invalid ccl_buffer_size from update_context: {ccl_buffer_size.value}")
    return context


def run_layout(
    topk_idx: torch.Tensor,
    num_tokens: int,
    topk: int,
    moe_expert_num: int,
    ep_world_size: int,
    local_rank_size: int,
    per_round_tokens: int,
    ep_rank_id: int,
):
    (
        num_tokens_per_rank,
        _,
        num_tokens_per_expert,
        is_token_in_rank,
        ext_handle_data,
    ) = dispatch_layout_op_module.npu_dispatch_layout(
        topk_idx,
        num_tokens,
        topk,
        moe_expert_num,
        ep_world_size,
        local_rank_size,
        per_round_tokens,
        ep_rank_id,
    )
    torch.npu.synchronize()
    layout_round_val = num_tokens_per_expert.numel() // moe_expert_num
    return (
        num_tokens_per_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        ext_handle_data,
        layout_round_val,
    )


def assert_prefill_outputs(
    recv_x: torch.Tensor,
    dynamic_scales: torch.Tensor,
    assist_info: torch.Tensor,
    recv_count: torch.Tensor,
    hidden_size: int,
    layout_round: int,
    moe_expert_num: int,
):
    assert recv_count.dim() == 1
    # sgl intranode_dispatch returns recv_count.sum(0) -> [num_experts]
    assert recv_count.numel() == moe_expert_num
    assert recv_count.dtype == torch.int32

    assert recv_x.dim() == 2
    num_recv = recv_x.size(0)
    assert recv_x.size(1) == hidden_size
    assert num_recv >= 1

    assert dynamic_scales.dim() == 1
    assert dynamic_scales.size(0) == num_recv
    assert dynamic_scales.dtype == torch.float32

    assert assist_info.dim() == 1
    assert assist_info.size(0) == num_recv * EXPAND_IDX_PER_TOKEN
    assert assist_info.dtype == torch.int32

    assert torch.isfinite(recv_x.float()).all()
    assert torch.isfinite(dynamic_scales).all()


def run_prefill_op_module(
    context: torch.Tensor,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    ext_handle_data,
    ep_world_size: int,
    ep_rank_id: int,
    local_rank_size: int,
    moe_expert_num: int,
    ccl_buffer_size: int,
    num_tokens_per_rank: torch.Tensor,
    is_token_in_rank: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    layout_round: int,
    per_round_tokens: int,
    num_worst_tokens: int,
    quant_mode: int = 0,
):
    recv_x, dynamic_scales, assist_info, recv_count = prefill_op_module.npu_moe_dispatch_prefill(
        context,
        x,
        topk_idx,
        ext_handle_data,
        ep_world_size,
        ep_rank_id,
        local_rank_size,
        moe_expert_num,
        ccl_buffer_size,
        None,
        None,
        num_tokens_per_rank,
        is_token_in_rank,
        num_tokens_per_expert,
        None,
        num_worst_tokens,
        quant_mode,
        layout_round,
        per_round_tokens,
    )
    torch.npu.synchronize()
    return recv_x, dynamic_scales, assist_info, recv_count


def run_prefill_torch_api(
    context: torch.Tensor,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    ext_handle_data,
    ep_world_size: int,
    ep_rank_id: int,
    local_rank_size: int,
    moe_expert_num: int,
    ccl_buffer_size: int,
    num_tokens_per_rank: torch.Tensor,
    is_token_in_rank: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    layout_round: int,
    per_round_tokens: int,
    num_worst_tokens: int,
    quant_mode: int = 0,
):
    recv_x, dynamic_scales, assist_info, recv_count = npu_moe_dispatch_prefill(
        context,
        x,
        topk_idx,
        ext_handle_data,
        ep_world_size,
        ep_rank_id,
        local_rank_size,
        moe_expert_num,
        ccl_buffer_size,
        x_scales=None,
        topk_weights=None,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        dispatch_wait_recv_cost_stats=None,
        num_worst_tokens=num_worst_tokens,
        quant_mode=quant_mode,
        round=layout_round,
        per_round_tokens=per_round_tokens,
    )
    torch.npu.synchronize()
    return recv_x, dynamic_scales, assist_info, recv_count


def main():
    _, rank, world_size = init_dist()
    group = dist.distributed_c10d._get_default_group()

    ep_world_size = env_int("MOE_DISPATCH_PREFILL_EP_WORLD_SIZE", world_size)
    local_rank_size = env_int("MOE_DISPATCH_PREFILL_LOCAL_RANK_SIZE", SGL_LOCAL_RANK_SIZE)
    ep_rank_id = env_int("MOE_DISPATCH_PREFILL_EP_RANK_ID", rank)
    moe_expert_num = env_int("MOE_DISPATCH_PREFILL_NUM_EXPERTS", SGL_NUM_EXPERTS)
    num_tokens = env_int("MOE_DISPATCH_PREFILL_NUM_TOKENS", SGL_NUM_TOKENS)
    topk = env_int("MOE_DISPATCH_PREFILL_TOPK", SGL_NUM_TOPK)
    per_round_tokens = env_int("MOE_DISPATCH_PREFILL_PER_ROUND_TOKENS", SGL_PER_ROUND_TOKENS)
    hidden_size = env_int("MOE_DISPATCH_PREFILL_HIDDEN", SGL_HIDDEN)
    num_worst_tokens = env_int("MOE_DISPATCH_PREFILL_NUM_WORST_TOKENS", num_tokens)
    configured_round = env_int("MOE_DISPATCH_PREFILL_ROUND", SGL_ROUND)
    expected_layout_round = layout_round(num_tokens, per_round_tokens)
    check_token_capacity(num_tokens, configured_round, per_round_tokens)

    if moe_expert_num % ep_world_size != 0:
        raise ValueError(f"moe_expert_num={moe_expert_num} must be divisible by ep_world_size={ep_world_size}")

    ccl_buffer_size = IntWrapper()
    context = build_context(group, rank, world_size, ccl_buffer_size)

    x, topk_idx, _ = make_moe_inputs(
        num_tokens,
        hidden_size,
        moe_expert_num,
        topk,
        rank=rank,
    )

    num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank, ext_handle_data, layout_round_val = run_layout(
        topk_idx,
        num_tokens,
        topk,
        moe_expert_num,
        ep_world_size,
        local_rank_size,
        per_round_tokens,
        ep_rank_id,
    )
    torch._check(
        layout_round_val == expected_layout_round,
        lambda: (
            f"layout round {layout_round_val} != expected {expected_layout_round}; "
            f"num_tokens={num_tokens} per_round_tokens={per_round_tokens}."
        ),
    )
    torch._check(
        layout_round_val <= configured_round,
        lambda: (
            f"layout round {layout_round_val} exceeds configured round {configured_round}; "
            f"increase MOE_DISPATCH_PREFILL_ROUND."
        ),
    )

    print(
        f"[rank {rank}] moe_dispatch_prefill (sgl-aligned): num_tokens={num_tokens} topk={topk} "
        f"hidden={hidden_size} experts={moe_expert_num} ep={ep_world_size} round={layout_round_val} "
        f"per_round_tokens={per_round_tokens} local_rank_size={local_rank_size} "
        f"ccl_buffer_size={ccl_buffer_size.value}",
        flush=True,
    )

    mod_rx, mod_scales, mod_assist, mod_count = run_prefill_op_module(
        context,
        x,
        topk_idx,
        ext_handle_data,
        ep_world_size,
        ep_rank_id,
        local_rank_size,
        moe_expert_num,
        ccl_buffer_size.value,
        num_tokens_per_rank,
        is_token_in_rank,
        num_tokens_per_expert,
        layout_round_val,
        per_round_tokens,
        num_worst_tokens,
    )
    assert_prefill_outputs(
        mod_rx, mod_scales, mod_assist, mod_count, hidden_size, layout_round_val, moe_expert_num
    )

    # Re-run layout before torch API path (avoid back-to-back collectives on stale buffers).
    topk_idx_api = topk_idx.clone()
    x_api = x.clone()
    ntr, nte, itir, ext_api, round_api = run_layout(
        topk_idx_api,
        num_tokens,
        topk,
        moe_expert_num,
        ep_world_size,
        local_rank_size,
        per_round_tokens,
        ep_rank_id,
    )
    assert round_api == layout_round_val

    api_rx, api_scales, api_assist, api_count = run_prefill_torch_api(
        context,
        x_api,
        topk_idx_api,
        ext_api,
        ep_world_size,
        ep_rank_id,
        local_rank_size,
        moe_expert_num,
        ccl_buffer_size.value,
        ntr,
        itir,
        nte,
        layout_round_val,
        per_round_tokens,
        num_worst_tokens,
    )
    assert_prefill_outputs(
        api_rx, api_scales, api_assist, api_count, hidden_size, layout_round_val, moe_expert_num
    )

    torch.testing.assert_close(mod_rx.cpu(), api_rx.cpu(), rtol=0, atol=0, msg="recv_x op_module vs torch API")
    torch.testing.assert_close(mod_scales.cpu(), api_scales.cpu(), rtol=0, atol=0, msg="dynamic_scales mismatch")
    torch.testing.assert_close(mod_assist.cpu(), api_assist.cpu(), rtol=0, atol=0, msg="assist_info mismatch")
    torch.testing.assert_close(mod_count.cpu(), api_count.cpu(), rtol=0, atol=0, msg="recv_count mismatch")

    print(
        f"[rank {rank}] OK prefill recv_x={tuple(mod_rx.shape)} "
        f"assist={tuple(mod_assist.shape)} recv_count_sum={int(mod_count.sum().item())}",
        flush=True,
    )

    dist.barrier()
    if rank == 0:
        print("[INFO] moe_dispatch_prefill standalone test finished.", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        raise
