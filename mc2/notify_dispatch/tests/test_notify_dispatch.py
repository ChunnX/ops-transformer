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
NotifyDispatch step-by-step test (layout -> notify -> dispatch_normal).

Steps (MOE_NOTIFY_DISPATCH_STEP):
  layout  - dispatch_layout only + Python sanity checks
  notify  - layout + npu_notify_dispatch (stop before dispatch_normal)
  full    - layout + notify + dispatch_normal (default)

send_data modes (MOE_NOTIFY_SEND_DATA_MODE):
  zeros           - torch.zeros((send_count,)) [default]
  empty           - torch.empty((send_count,)) (uninitialized)
  layout_flat     - ext_handle_data[0][:send_count] 1D slice from layout scratch
  layout_2d       - layout scratch as (round, experts*3) then flatten for npu_notify_dispatch (1D API)
  prefill_empty_2d - empty({round, experts*3}) like moe_dispatch_prefill.cpp, flatten for API

Note: npu_notify_dispatch C++ wrapper requires send_data 1D. moe_dispatch_prefill allocates a fresh
empty({round, experts*3}) buffer and does NOT slice ext_handle_data[0].

x dtype (MOE_NOTIFY_X_DTYPE): bf16 (default, matches test_dispatch_prefill) or fp16.

Context init (MOE_NOTIFY_USE_HCCL_CHANNEL_CONTEXT):
  0 (default) - update_context (KFC + HcclGetRemoteIpcHcclBuf)
  1           - update_context_by_hccl_channel (HcclEngineCtx + channel buffer; mega_moe path)

Examples:
    # Step 2: notify with zeros send_data
    MOE_NOTIFY_DISPATCH_STEP=notify MOE_NOTIFY_SEND_DATA_MODE=zeros \\
    MOE_DISPATCH_PREFILL_FIXED_TOPK=1 \\
    torchrun --nproc_per_node=2 mc2/notify_dispatch/tests/test_notify_dispatch.py

    # Step 2: notify with layout scratch (2D view, flattened for API)
    MOE_NOTIFY_DISPATCH_STEP=notify MOE_NOTIFY_SEND_DATA_MODE=layout_2d \\
    MOE_DISPATCH_PREFILL_FIXED_TOPK=1 \\
    torchrun --nproc_per_node=2 mc2/notify_dispatch/tests/test_notify_dispatch.py
"""

import os
import struct
import sys

import torch
import torch_npu
import torch.distributed as dist
import npu_ops_transformer  # noqa: F401
from npu_ops_transformer.ops.update_context import init_mc2_context


class IntWrapper:
    def __init__(self, value=0):
        self.value = value


SEND_PER_GROUP = 3

# smoke (same as test_dispatch_prefill)
SMOKE_NUM_TOKENS = 8
SMOKE_TOPK = 4
SMOKE_NUM_EXPERTS = 16
SMOKE_PER_ROUND_TOKENS = 8
SMOKE_MULTIROUND_PER_ROUND_TOKENS = 2


def init_dist():
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    if world_size < 2:
        raise RuntimeError("WORLD_SIZE must be >= 2 for notify_dispatch communication test.")
    torch_npu.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    return local_rank, rank, world_size


def build_context(group, rank, world_size, ccl_buffer_size):
    context_struct_size = 2 + 1024
    context_tensor_size = ((context_struct_size * 8) + 3) // 4
    context = torch.zeros((context_tensor_size,), dtype=torch.int32, device="npu")
    group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    use_hccl_channel = os.getenv("MOE_NOTIFY_USE_HCCL_CHANNEL_CONTEXT", "0") == "1"
    if use_hccl_channel:
        os.environ["MC2_USE_HCCL_CHANNEL_CONTEXT"] = "1"
    ok = init_mc2_context(group_name, world_size, ccl_buffer_size, context)
    if use_hccl_channel:
        print(
            f"[rank {rank}] context via update_context_by_hccl_channel: "
            f"ccl_buffer_size={ccl_buffer_size.value}",
            flush=True,
        )
    else:
        print(
            f"[rank {rank}] context via update_context: ccl_buffer_size={ccl_buffer_size.value}",
            flush=True,
        )
    if not ok:
        raise RuntimeError("update_context failed.")
    if ccl_buffer_size.value <= 0:
        raise RuntimeError(f"invalid ccl_buffer_size from update_context: {ccl_buffer_size.value}")
    return context


def check_mc2_context(context, rank, world_size):
    """Lightweight sanity check: epRankId and epHcclBuffer IPC addresses."""
    print(
        f"[rank {rank}] context shape={tuple(context.shape)} dtype={context.dtype} device={context.device}",
        flush=True,
    )
    if os.getenv("MOE_NOTIFY_DEBUG_CONTEXT", "1") == "1":
        print(f"[rank {rank}] context head={context[:8].cpu().tolist()}", flush=True)

    ep_rank_id = int(context[0].item())
    if ep_rank_id != rank:
        raise RuntimeError(
            f"[rank {rank}] context epRankId={ep_rank_id} != rank={rank}"
        )

    raw = context.detach().cpu().view(torch.uint8).numpy().tobytes()
    for i in range(world_size):
        addr = struct.unpack_from("<Q", raw, 16 + i * 8)[0]
        print(f"[rank {rank}] epHcclBuffer[{i}]=0x{addr:x}", flush=True)
        if addr == 0:
            raise RuntimeError(f"[rank {rank}] epHcclBuffer[{i}] is zero after update_context")


def make_topk_idx(num_tokens, topk, num_experts, device):
    if os.getenv("MOE_DISPATCH_PREFILL_FIXED_TOPK", "0") == "1":
        if num_tokens != 8 or topk != 4 or num_experts != 16:
            raise ValueError(
                "MOE_DISPATCH_PREFILL_FIXED_TOPK=1 only supports num_tokens=8, topk=4, num_experts=16"
            )
        return torch.tensor(
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
    return torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int64, device=device)


def resolve_params(world_size):
    profile = os.getenv("MOE_DISPATCH_PREFILL_PROFILE", "smoke").lower()
    if profile == "smoke_multiround":
        per_round_tokens = int(
            os.getenv("MOE_DISPATCH_PREFILL_PER_ROUND_TOKENS", SMOKE_MULTIROUND_PER_ROUND_TOKENS)
        )
    else:
        per_round_tokens = int(os.getenv("MOE_DISPATCH_PREFILL_PER_ROUND_TOKENS", SMOKE_PER_ROUND_TOKENS))
    return {
        "profile": profile,
        "num_tokens": int(os.getenv("MOE_DISPATCH_PREFILL_NUM_TOKENS", SMOKE_NUM_TOKENS)),
        "topk": int(os.getenv("MOE_DISPATCH_PREFILL_TOPK", SMOKE_TOPK)),
        "moe_expert_num": int(os.getenv("MOE_DISPATCH_PREFILL_NUM_EXPERTS", SMOKE_NUM_EXPERTS)),
        "per_round_tokens": per_round_tokens,
        "hidden_size": int(os.getenv("MOE_DISPATCH_PREFILL_HIDDEN", 1024)),
    }


def check_layout_outputs(rank, num_tokens, topk, num_tokens_per_expert, ext_handle_data):
    if len(ext_handle_data) != 2:
        raise RuntimeError(
            f"[rank {rank}] ext_handle_data should be len=2, got len={len(ext_handle_data)}"
        )
    notify_send_data, send_token_idx_small = ext_handle_data
    if notify_send_data.dtype != torch.int32:
        raise RuntimeError(f"[rank {rank}] notify_send_data dtype must be int32")
    if send_token_idx_small.dtype != torch.int32:
        raise RuntimeError(f"[rank {rank}] send_token_idx_small dtype must be int32")

    expected = num_tokens * topk
    actual = int(num_tokens_per_expert.sum().item())
    print(f"[rank {rank}] expert_sum={actual}, expected={expected}", flush=True)
    if actual != expected:
        raise RuntimeError(
            f"[rank {rank}] num_tokens_per_expert sum mismatch: {actual} != {expected}"
        )

    send_min = int(send_token_idx_small.min().item())
    send_max = int(send_token_idx_small.max().item())
    print(f"[rank {rank}] send_token_idx_small min/max={send_min}/{send_max}", flush=True)
    if send_min < -1 or send_max >= num_tokens:
        raise RuntimeError(
            f"[rank {rank}] send_token_idx_small out of range: min={send_min}, max={send_max}, "
            f"num_tokens={num_tokens}"
        )

    print(f"[rank {rank}] num_tokens_per_expert = {num_tokens_per_expert.cpu()}", flush=True)
    print(
        f"[rank {rank}] notify_send_data min/max="
        f"{int(notify_send_data.min().item())}/{int(notify_send_data.max().item())}",
        flush=True,
    )
    head_len = min(128, notify_send_data.numel())
    print(
        f"[rank {rank}] notify_send_data[:{head_len}] = {notify_send_data[:head_len].cpu()}",
        flush=True,
    )


def resolve_x_dtype():
    name = os.getenv("MOE_NOTIFY_X_DTYPE", "bf16").lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "f16"):
        return torch.float16
    raise ValueError(f"MOE_NOTIFY_X_DTYPE must be bf16 or fp16, got {name!r}")


def make_send_data(
    mode: str,
    send_count: int,
    layout_round: int,
    moe_expert_num: int,
    notify_send_data: torch.Tensor,
    device: str,
):
    """Build send_data for npu_notify_dispatch (API requires 1D int32, numel == send_count)."""
    send_row = moe_expert_num * SEND_PER_GROUP
    if mode == "zeros":
        send_data = torch.zeros((send_count,), dtype=torch.int32, device=device)
        storage_desc = f"zeros 1D {tuple(send_data.shape)}"
    elif mode == "empty":
        send_data = torch.empty((send_count,), dtype=torch.int32, device=device)
        storage_desc = f"empty 1D {tuple(send_data.shape)}"
    elif mode == "layout_flat":
        if notify_send_data.numel() < send_count:
            raise RuntimeError(
                f"notify_send_data numel={notify_send_data.numel()} < send_count={send_count}"
            )
        send_data = notify_send_data[:send_count].contiguous()
        storage_desc = f"layout_flat slice [:send_count] -> 1D {tuple(send_data.shape)}"
    elif mode == "layout_2d":
        if send_count != layout_round * send_row:
            raise RuntimeError(
                f"layout_2d requires send_count==round*experts*3, got send_count={send_count}, "
                f"round={layout_round}, experts={moe_expert_num}"
            )
        if notify_send_data.numel() < send_count:
            raise RuntimeError(
                f"notify_send_data numel={notify_send_data.numel()} < send_count={send_count}"
            )
        send_data_2d = notify_send_data[:send_count].contiguous().view(layout_round, send_row)
        send_data = send_data_2d.reshape(-1)
        storage_desc = f"layout_2d view {tuple(send_data_2d.shape)} -> flat 1D {tuple(send_data.shape)}"
    elif mode == "prefill_empty_2d":
        if send_count != layout_round * send_row:
            raise RuntimeError(
                f"prefill_empty_2d requires send_count==round*experts*3, got send_count={send_count}"
            )
        send_data_2d = torch.empty((layout_round, send_row), dtype=torch.int32, device=device)
        send_data = send_data_2d.reshape(-1)
        storage_desc = (
            f"prefill_empty_2d empty {tuple(send_data_2d.shape)} -> flat 1D {tuple(send_data.shape)}"
        )
    else:
        raise ValueError(
            f"unknown MOE_NOTIFY_SEND_DATA_MODE={mode!r}; "
            f"use zeros|empty|layout_flat|layout_2d|prefill_empty_2d"
        )

    if send_data.dim() != 1 or send_data.numel() != send_count:
        raise RuntimeError(
            f"internal error: send_data must be 1D numel={send_count}, "
            f"got shape={tuple(send_data.shape)}"
        )
    return send_data, storage_desc


def main():
    step = os.getenv("MOE_NOTIFY_DISPATCH_STEP", "full").lower()
    if step not in ("layout", "notify", "full"):
        raise ValueError(f"MOE_NOTIFY_DISPATCH_STEP must be layout|notify|full, got {step!r}")

    _, rank, world_size = init_dist()
    group = dist.distributed_c10d._get_default_group()
    params = resolve_params(world_size)

    ep_world_size = world_size
    ep_rank_id = rank
    local_rank_size = world_size
    local_rank_id = ep_rank_id % local_rank_size
    num_tokens = params["num_tokens"]
    topk = params["topk"]
    moe_expert_num = params["moe_expert_num"]
    per_round_tokens = params["per_round_tokens"]
    hidden_size = params["hidden_size"]

    print(
        f"[rank {rank}] step={step} profile={params['profile']} "
        f"num_tokens={num_tokens} topk={topk} experts={moe_expert_num} ep={ep_world_size} "
        f"per_round_tokens={per_round_tokens} fixed_topk="
        f"{os.getenv('MOE_DISPATCH_PREFILL_FIXED_TOPK', '0')} "
        f"send_data_mode={os.getenv('MOE_NOTIFY_SEND_DATA_MODE', 'zeros')}",
        flush=True,
    )

    ccl_buffer_size = IntWrapper()
    context = build_context(group, rank, world_size, ccl_buffer_size)
    check_mc2_context(context, rank, world_size)

    x_dtype = resolve_x_dtype()
    topk_idx = make_topk_idx(num_tokens, topk, moe_expert_num, "npu")
    x = torch.randn((num_tokens, hidden_size), dtype=x_dtype, device="npu")

    from npu_ops_transformer.ops.dispatch_layout import op_module as dispatch_layout_op_module

    num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, ext_handle_data = (
        dispatch_layout_op_module.npu_dispatch_layout(
            topk_idx,
            num_tokens,
            topk,
            moe_expert_num,
            ep_world_size,
            local_rank_size,
            per_round_tokens,
            ep_rank_id,
        )
    )
    torch.npu.synchronize()
    layout_round = num_tokens_per_expert.numel() // moe_expert_num

    print(f"[rank {rank}] layout OK: round={layout_round}", flush=True)
    print(f"[rank {rank}] topk_idx = {topk_idx.cpu()}", flush=True)
    print(f"[rank {rank}] num_tokens_per_rank = {num_tokens_per_rank.cpu()}", flush=True)
    check_layout_outputs(rank, num_tokens, topk, num_tokens_per_expert, ext_handle_data)

    if step == "layout":
        dist.barrier()
        if rank == 0:
            print("[INFO] step=layout finished OK.", flush=True)
        dist.destroy_process_group()
        return 0

    send_token_idx = ext_handle_data[1]
    notify_send_data = ext_handle_data[0]
    send_count = SEND_PER_GROUP * moe_expert_num * layout_round
    send_data_mode = os.getenv("MOE_NOTIFY_SEND_DATA_MODE", "zeros")
    send_data, send_data_storage_desc = make_send_data(
        send_data_mode,
        send_count,
        layout_round,
        moe_expert_num,
        notify_send_data,
        "npu",
    )

    print(
        f"[rank {rank}] send_data_mode={send_data_mode} storage={send_data_storage_desc} "
        f"send_count={send_count} send_data.shape={tuple(send_data.shape)} "
        f"send_data.dtype={send_data.dtype} "
        f"send_data.min/max={int(send_data.min().item())}/{int(send_data.max().item())}",
        flush=True,
    )
    print(
        f"[rank {rank}] calling npu_notify_dispatch: layout_round={layout_round} "
        f"ccl_buffer_size={ccl_buffer_size.value}",
        flush=True,
    )

    (
        send_offset,
        recv_offset,
        expert_output_offset,
        total_recv_tokens,
        max_bs,
        recv_data,
        recv_tokens_per_expert,
    ) = torch.ops.npu_ops_transformer.npu_notify_dispatch(
        context,
        send_data,
        num_tokens_per_expert,
        send_count,
        num_tokens,
        ep_world_size,
        ep_rank_id,
        local_rank_size,
        local_rank_id,
        ccl_buffer_size.value,
        layout_round,
        per_round_tokens,
    )
    torch.npu.synchronize()

    num_recv_tokens = int(total_recv_tokens.item())
    real_max_bs = max(int(max_bs.item()), num_tokens)
    print(
        f"[rank {rank}] notify OK: send_offset={tuple(send_offset.shape)} "
        f"total_recv_tokens={num_recv_tokens} max_bs={real_max_bs} "
        f"recv_data={tuple(recv_data.shape)}",
        flush=True,
    )

    if step == "notify":
        dist.barrier()
        if rank == 0:
            print("[INFO] step=notify finished OK.", flush=True)
        dist.destroy_process_group()
        return 0

    if num_recv_tokens <= 0:
        num_recv_tokens = 1
    global_bs = min(per_round_tokens, real_max_bs) * ep_world_size

    recv_x, x_scales, assist_info_for_combine, _ = torch.ops.npu_ops_transformer.npu_dispatch_normal(
        context=context,
        x=x,
        topk_idx=topk_idx.to(torch.int32),
        send_offset=send_offset,
        send_token_idx=send_token_idx,
        recv_offset=recv_offset,
        expert_output_offset=expert_output_offset,
        ep_world_size=ep_world_size,
        ep_rank_id=ep_rank_id,
        moe_expert_num=moe_expert_num,
        ccl_buffer_size=ccl_buffer_size.value,
        num_recv_tokens=num_recv_tokens,
        quant_mode=0,
        real_max_bs=real_max_bs,
        global_bs=global_bs,
        round=layout_round,
        per_round_tokens=per_round_tokens,
    )
    torch.npu.synchronize()

    print(
        f"[rank {rank}] dispatch OK: recv_x={tuple(recv_x.shape)} "
        f"assist={tuple(assist_info_for_combine.shape)} "
        f"recv_tokens_per_expert={tuple(recv_tokens_per_expert.shape)}",
        flush=True,
    )

    dist.barrier()
    if rank == 0:
        print("[INFO] step=full pipeline finished OK.", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        raise
