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

Profiles (MOE_DISPATCH_PREFILL_PROFILE):
  auto             - smoke when ep_world_size <= 8, else sgl-scale
  smoke            - 2-card, round=1 (per_round_tokens=8, num_tokens=8)
  smoke_multiround - 2-card, round=4 (per_round_tokens=2); isolates multi-round NotifyDispatch
  sgl              - test_normal_and_fused_deep_moe.py scale (16-card)

NotifyDispatch inside moe_dispatch_prefill uses local_rank_size=ep_world_size (A3 intranode).
layout local_rank_size is dispatch_layout-only and must divide ep_world_size.

Examples:
    # 2-card smoke, single-round (recommended first)
    MOE_DISPATCH_PREFILL_PROFILE=smoke \\
    torchrun --nproc_per_node=2 torch_extension/npu_ops_transformer/ops/test_dispatch_prefill.py

    # smoke with deterministic topk (8 tokens, topk=4, 16 experts)
    MOE_DISPATCH_PREFILL_FIXED_TOPK=1 \\
    MOE_DISPATCH_PREFILL_PROFILE=smoke \\
    torchrun --nproc_per_node=2 torch_extension/npu_ops_transformer/ops/test_dispatch_prefill.py

    # multi-round isolation (may hit NotifyDispatch round>1 bugs)
    MOE_DISPATCH_PREFILL_PROFILE=smoke_multiround \\
    torchrun --nproc_per_node=2 torch_extension/npu_ops_transformer/ops/test_dispatch_prefill.py

    # sgl-scale 16-card
    export HCCL_BUFFSIZE=2048
    MOE_DISPATCH_PREFILL_PROFILE=sgl \\
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
from npu_ops_transformer.ops.update_context import update_context

# sgl test_normal_and_fused_deep_moe.py / deep_ep.cpp defaults (sgl profile)
SGL_NUM_TOKENS = 4096
SGL_HIDDEN = 7168
SGL_NUM_TOPK = 8
SGL_NUM_EXPERTS = 256
SGL_PER_ROUND_TOKENS = 8192
SGL_LAYOUT_LOCAL_RANK_SIZE = 8
SGL_MAX_TOTAL_TOKENS = 131072

# 2-card smoke defaults: single-round path (matches sgl round=1 style)
SMOKE_NUM_TOKENS = 8
SMOKE_HIDDEN = 1024
SMOKE_NUM_TOPK = 4
SMOKE_NUM_EXPERTS = 16
SMOKE_PER_ROUND_TOKENS = 8

# smoke_multiround: round=ceil(8/2)=4, for NotifyDispatch multi-round isolation
SMOKE_MULTIROUND_PER_ROUND_TOKENS = 2

EXPAND_IDX_PER_TOKEN = 3


def env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val is not None else default


def layout_round(num_tokens: int, per_round_tokens: int) -> int:
    return (num_tokens + per_round_tokens - 1) // per_round_tokens


def resolve_profile(ep_world_size: int) -> str:
    profile = os.getenv("MOE_DISPATCH_PREFILL_PROFILE", "auto").lower()
    if profile in ("smoke", "smoke_multiround", "sgl"):
        return profile
    return "smoke" if ep_world_size <= 8 else "sgl"


def default_layout_local_rank_size(ep_world_size: int) -> int:
    """dispatch_layout local_ranksize: ranks per server (sgl 16-card -> 8, else ep_world_size)."""
    for key in ("MOE_DISPATCH_PREFILL_LAYOUT_LOCAL_RANK_SIZE", "MOE_DISPATCH_PREFILL_LOCAL_RANK_SIZE"):
        val = os.getenv(key)
        if val is not None:
            return int(val)
    if ep_world_size >= 16 and ep_world_size % SGL_LAYOUT_LOCAL_RANK_SIZE == 0:
        return SGL_LAYOUT_LOCAL_RANK_SIZE
    return ep_world_size


def resolve_test_params(ep_world_size: int) -> dict:
    profile = resolve_profile(ep_world_size)
    if profile == "smoke_multiround":
        num_tokens = env_int("MOE_DISPATCH_PREFILL_NUM_TOKENS", SMOKE_NUM_TOKENS)
        per_round_tokens = env_int(
            "MOE_DISPATCH_PREFILL_PER_ROUND_TOKENS", SMOKE_MULTIROUND_PER_ROUND_TOKENS
        )
        defaults = {
            "num_tokens": num_tokens,
            "hidden_size": env_int("MOE_DISPATCH_PREFILL_HIDDEN", SMOKE_HIDDEN),
            "topk": env_int("MOE_DISPATCH_PREFILL_TOPK", SMOKE_NUM_TOPK),
            "moe_expert_num": env_int("MOE_DISPATCH_PREFILL_NUM_EXPERTS", SMOKE_NUM_EXPERTS),
            "per_round_tokens": per_round_tokens,
        }
    elif profile == "smoke":
        num_tokens = env_int("MOE_DISPATCH_PREFILL_NUM_TOKENS", SMOKE_NUM_TOKENS)
        per_round_tokens = env_int("MOE_DISPATCH_PREFILL_PER_ROUND_TOKENS", SMOKE_PER_ROUND_TOKENS)
        defaults = {
            "num_tokens": num_tokens,
            "hidden_size": env_int("MOE_DISPATCH_PREFILL_HIDDEN", SMOKE_HIDDEN),
            "topk": env_int("MOE_DISPATCH_PREFILL_TOPK", SMOKE_NUM_TOPK),
            "moe_expert_num": env_int("MOE_DISPATCH_PREFILL_NUM_EXPERTS", SMOKE_NUM_EXPERTS),
            "per_round_tokens": per_round_tokens,
        }
    else:
        num_tokens = env_int("MOE_DISPATCH_PREFILL_NUM_TOKENS", SGL_NUM_TOKENS)
        per_round_tokens = env_int("MOE_DISPATCH_PREFILL_PER_ROUND_TOKENS", SGL_PER_ROUND_TOKENS)
        defaults = {
            "num_tokens": num_tokens,
            "hidden_size": env_int("MOE_DISPATCH_PREFILL_HIDDEN", SGL_HIDDEN),
            "topk": env_int("MOE_DISPATCH_PREFILL_TOPK", SGL_NUM_TOPK),
            "moe_expert_num": env_int("MOE_DISPATCH_PREFILL_NUM_EXPERTS", SGL_NUM_EXPERTS),
            "per_round_tokens": per_round_tokens,
        }

    layout_round_val = layout_round(defaults["num_tokens"], defaults["per_round_tokens"])
    max_configured_round = env_int(
        "MOE_DISPATCH_PREFILL_MAX_ROUND",
        env_int("MOE_DISPATCH_PREFILL_ROUND", layout_round_val),
    )
    defaults["layout_local_rank_size"] = default_layout_local_rank_size(ep_world_size)
    defaults["max_configured_round"] = max_configured_round
    defaults["layout_round_val"] = layout_round_val
    defaults["profile"] = profile
    defaults["num_worst_tokens"] = env_int(
        "MOE_DISPATCH_PREFILL_NUM_WORST_TOKENS",
        defaults["num_tokens"] * defaults["topk"],
    )
    return defaults


def dump_layout_debug(
    rank: int,
    num_tokens: int,
    topk: int,
    topk_idx: torch.Tensor,
    num_tokens_per_rank: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    is_token_in_rank: torch.Tensor,
    ext_handle_data,
) -> None:
    if len(ext_handle_data) != 2:
        raise RuntimeError(
            f"[rank {rank}] ext_handle_data should be 2-tuple, got len={len(ext_handle_data)}"
        )
    notify_send_data, send_token_idx_small = ext_handle_data

    if notify_send_data.dtype != torch.int32:
        raise RuntimeError(
            f"[rank {rank}] notify_send_data dtype should be int32, got {notify_send_data.dtype}"
        )
    if send_token_idx_small.dtype != torch.int32:
        raise RuntimeError(
            f"[rank {rank}] send_token_idx_small dtype should be int32, got {send_token_idx_small.dtype}"
        )

    print(f"[rank {rank}] topk_idx = {topk_idx.cpu()}", flush=True)

    expected_assignments = num_tokens * topk
    actual_expert_sum = int(num_tokens_per_expert.sum().item())
    print(
        f"[rank {rank}] expert_sum={actual_expert_sum}, expected={expected_assignments}",
        flush=True,
    )
    if actual_expert_sum != expected_assignments:
        raise RuntimeError(
            f"[rank {rank}] num_tokens_per_expert sum mismatch: "
            f"{actual_expert_sum} != num_tokens*topk={expected_assignments}"
        )

    print(f"[rank {rank}] num_tokens_per_rank = {num_tokens_per_rank.cpu()}", flush=True)
    print(f"[rank {rank}] num_tokens_per_expert = {num_tokens_per_expert.cpu()}", flush=True)
    print(f"[rank {rank}] is_token_in_rank = {is_token_in_rank.cpu()}", flush=True)
    print(
        f"[rank {rank}] notify_send_data shape={tuple(notify_send_data.shape)} dtype={notify_send_data.dtype}",
        flush=True,
    )
    print(f"[rank {rank}] notify_send_data = {notify_send_data.cpu()}", flush=True)
    print(
        f"[rank {rank}] notify_send_data min/max="
        f"{int(notify_send_data.min().item())}/{int(notify_send_data.max().item())}",
        flush=True,
    )
    print(f"[rank {rank}] send_token_idx_small = {send_token_idx_small.cpu()}", flush=True)
    send_min = int(send_token_idx_small.min().item())
    send_max = int(send_token_idx_small.max().item())
    print(
        f"[rank {rank}] send_token_idx_small min/max={send_min}/{send_max}",
        flush=True,
    )
    if send_min < -1:
        raise RuntimeError(
            f"[rank {rank}] send_token_idx_small has invalid value < -1: min={send_min}"
        )
    if send_max >= num_tokens:
        raise RuntimeError(
            f"[rank {rank}] send_token_idx_small has invalid token index: "
            f"max={send_max}, num_tokens={num_tokens}"
        )


def print_prefill_call_args(
    rank: int,
    *,
    x,
    topk_idx,
    num_tokens_per_rank,
    num_tokens_per_expert,
    is_token_in_rank,
    ext_handle_data,
    ep_world_size,
    ep_rank_id,
    layout_local_rank_size,
    prefill_local_rank_size,
    moe_expert_num,
    layout_round_val,
    per_round_tokens,
    num_worst_tokens,
    ccl_buffer_size,
):
    send_token_idx = ext_handle_data[1] if ext_handle_data else None
    print(
        f"[rank {rank}] call npu_moe_dispatch_prefill (profile={resolve_profile(ep_world_size)}):",
        f"x={tuple(x.shape)} {x.dtype}",
        f"topk_idx={tuple(topk_idx.shape)} {topk_idx.dtype}",
        f"num_tokens_per_rank={tuple(num_tokens_per_rank.shape)} {num_tokens_per_rank.dtype}",
        f"num_tokens_per_expert={tuple(num_tokens_per_expert.shape)} {num_tokens_per_expert.dtype}",
        f"is_token_in_rank={tuple(is_token_in_rank.shape)} {is_token_in_rank.dtype}",
        f"send_token_idx_small={tuple(send_token_idx.shape)} {send_token_idx.dtype}"
        if send_token_idx is not None
        else "send_token_idx_small=None",
        f"ep_world_size={ep_world_size} ep_rank_id={ep_rank_id}",
        f"layout_local_rank_size={layout_local_rank_size} prefill_local_rank_size={prefill_local_rank_size}",
        f"experts={moe_expert_num} round={layout_round_val} per_round_tokens={per_round_tokens}",
        f"num_worst_tokens={num_worst_tokens} ccl_buffer_size={ccl_buffer_size}",
        flush=True,
    )


def check_token_capacity(
    num_tokens: int,
    round: int,
    per_round_tokens: int,
    *,
    max_total_tokens: int = SGL_MAX_TOTAL_TOKENS,
) -> None:
    capacity = round * per_round_tokens
    if num_tokens > capacity:
        raise ValueError(
            f"num_tokens={num_tokens} exceeds capacity round*per_round_tokens="
            f"{round}*{per_round_tokens}={capacity}"
        )
    if capacity > max_total_tokens:
        raise ValueError(
            f"round*per_round_tokens={capacity} exceeds max_total_tokens={max_total_tokens}"
        )


def make_moe_inputs(
    num_tokens: int,
    hidden: int,
    num_experts: int,
    num_topk: int,
    *,
    rank: int,
    device: str = "npu",
):
    torch.manual_seed(rank)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=device)
    if os.getenv("MOE_DISPATCH_PREFILL_FIXED_TOPK", "0") == "1":
        if num_tokens != 8 or num_topk != 4 or num_experts != 16:
            raise ValueError(
                "MOE_DISPATCH_PREFILL_FIXED_TOPK=1 only supports num_tokens=8, topk=4, num_experts=16"
            )
        topk_idx = torch.tensor(
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [8, 9, 10, 11],
                [12, 13, 14, 15],
                [12, 13, 14, 15],
            ],
            dtype=torch.int64,
            device=device,
        )
    else:
        scores = (
            torch.randn((num_tokens, num_experts), dtype=torch.float32, device=device).abs()
            + 1
        )
        topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device=device)
    return x, topk_idx, topk_weights


def topk_idx_for_prefill(topk_idx: torch.Tensor) -> torch.Tensor:
    """dispatch_layout.cpp requires int64; moe_dispatch_prefill.cpp accepts either and uses int32 for dispatch_normal."""
    return topk_idx.to(torch.int32)


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
    # init_comm must be True (default) so HCCL comm is registered for update_context / Mc2 kernels.
    group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
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
    prefill_local_rank_size: int,
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
        prefill_local_rank_size,
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
    prefill_local_rank_size: int,
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
        prefill_local_rank_size,
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
    ep_rank_id = env_int("MOE_DISPATCH_PREFILL_EP_RANK_ID", rank)
    params = resolve_test_params(ep_world_size)

    num_tokens = params["num_tokens"]
    topk = params["topk"]
    hidden_size = params["hidden_size"]
    moe_expert_num = params["moe_expert_num"]
    per_round_tokens = params["per_round_tokens"]
    layout_local_rank_size = params["layout_local_rank_size"]
    max_configured_round = params["max_configured_round"]
    expected_layout_round = params["layout_round_val"]
    num_worst_tokens = params["num_worst_tokens"]
    profile = params["profile"]
    prefill_local_rank_size = ep_world_size

    check_token_capacity(num_tokens, max_configured_round, per_round_tokens)

    min_worst_tokens = num_tokens * topk
    if num_worst_tokens < min_worst_tokens:
        raise ValueError(
            f"num_worst_tokens={num_worst_tokens} is too small; "
            f"expected at least num_tokens*topk={min_worst_tokens}"
        )

    if moe_expert_num % ep_world_size != 0:
        raise ValueError(f"moe_expert_num={moe_expert_num} must be divisible by ep_world_size={ep_world_size}")
    if ep_world_size % layout_local_rank_size != 0:
        raise ValueError(
            f"ep_world_size={ep_world_size} must be divisible by layout_local_rank_size="
            f"{layout_local_rank_size}; for {ep_world_size}-card smoke use "
            f"MOE_DISPATCH_PREFILL_LAYOUT_LOCAL_RANK_SIZE={ep_world_size}"
        )

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
        layout_local_rank_size,
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
        layout_round_val <= max_configured_round,
        lambda: (
            f"layout round {layout_round_val} exceeds max configured round {max_configured_round}; "
            f"increase MOE_DISPATCH_PREFILL_MAX_ROUND."
        ),
    )

    dump_layout_debug(
        rank,
        num_tokens,
        topk,
        topk_idx,
        num_tokens_per_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        ext_handle_data,
    )

    print(
        f"[rank {rank}] moe_dispatch_prefill profile={profile}: num_tokens={num_tokens} topk={topk} "
        f"hidden={hidden_size} experts={moe_expert_num} ep={ep_world_size} round={layout_round_val} "
        f"per_round_tokens={per_round_tokens} layout_local_rank_size={layout_local_rank_size} "
        f"prefill_local_rank_size={prefill_local_rank_size} num_worst_tokens={num_worst_tokens} "
        f"ccl_buffer_size={ccl_buffer_size.value}",
        flush=True,
    )

    topk_idx_prefill = topk_idx_for_prefill(topk_idx)

    print_prefill_call_args(
        rank,
        x=x,
        topk_idx=topk_idx_prefill,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        is_token_in_rank=is_token_in_rank,
        ext_handle_data=ext_handle_data,
        ep_world_size=ep_world_size,
        ep_rank_id=ep_rank_id,
        layout_local_rank_size=layout_local_rank_size,
        prefill_local_rank_size=prefill_local_rank_size,
        moe_expert_num=moe_expert_num,
        layout_round_val=layout_round_val,
        per_round_tokens=per_round_tokens,
        num_worst_tokens=num_worst_tokens,
        ccl_buffer_size=ccl_buffer_size.value,
    )

    mod_rx, mod_scales, mod_assist, mod_count = run_prefill_op_module(
        context,
        x,
        topk_idx_prefill,
        ext_handle_data,
        ep_world_size,
        ep_rank_id,
        prefill_local_rank_size,
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
        layout_local_rank_size,
        per_round_tokens,
        ep_rank_id,
    )
    assert round_api == layout_round_val

    topk_idx_api_prefill = topk_idx_for_prefill(topk_idx_api)

    print_prefill_call_args(
        rank,
        x=x_api,
        topk_idx=topk_idx_api_prefill,
        num_tokens_per_rank=ntr,
        num_tokens_per_expert=nte,
        is_token_in_rank=itir,
        ext_handle_data=ext_api,
        ep_world_size=ep_world_size,
        ep_rank_id=ep_rank_id,
        layout_local_rank_size=layout_local_rank_size,
        prefill_local_rank_size=prefill_local_rank_size,
        moe_expert_num=moe_expert_num,
        layout_round_val=layout_round_val,
        per_round_tokens=per_round_tokens,
        num_worst_tokens=num_worst_tokens,
        ccl_buffer_size=ccl_buffer_size.value,
    )

    api_rx, api_scales, api_assist, api_count = run_prefill_torch_api(
        context,
        x_api,
        topk_idx_api_prefill,
        ext_api,
        ep_world_size,
        ep_rank_id,
        prefill_local_rank_size,
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
