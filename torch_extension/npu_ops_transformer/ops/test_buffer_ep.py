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
Buffer EP normal test (layout -> dispatch -> combine), aligned with sgl test_common.py.

Run with HCCL initialized, for example:
    torchrun --nproc_per_node=16 torch_extension/npu_ops_transformer/ops/test_buffer_ep.py
"""

import os

import torch
import torch_npu
import torch.distributed as dist
import npu_ops_transformer  # noqa: F401
from npu_ops_transformer.ops.buffer import Buffer
from npu_ops_transformer.ops.test_utils import (
    SGL_COMBINE_DIFF_TOL,
    SGL_HIDDEN,
    SGL_NUM_EXPERTS,
    SGL_NUM_TOKENS,
    SGL_NUM_TOPK,
    SGL_PER_ROUND_TOKENS,
    calc_diff,
    combine_golden,
    env_int,
    layout_round,
    make_moe_inputs,
)


def init_dist():
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    if world_size < 2:
        raise RuntimeError("WORLD_SIZE must be >= 2 for Buffer EP communication test.")
    torch_npu.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    return local_rank, rank, world_size


def main():
    _, rank, world_size = init_dist()
    group = dist.distributed_c10d._get_default_group()

    moe_expert_num = env_int("MOE_BUFFER_EP_NUM_EXPERTS", SGL_NUM_EXPERTS)
    num_tokens = env_int("MOE_BUFFER_EP_NUM_TOKENS", SGL_NUM_TOKENS)
    topk = env_int("MOE_BUFFER_EP_TOPK", SGL_NUM_TOPK)
    per_round_tokens = env_int("MOE_BUFFER_EP_PER_ROUND_TOKENS", SGL_PER_ROUND_TOKENS)
    hidden_size = env_int("MOE_BUFFER_EP_HIDDEN", SGL_HIDDEN)
    expected_round = env_int(
        "MOE_BUFFER_EP_ROUND",
        layout_round(num_tokens, per_round_tokens),
    )

    if moe_expert_num % world_size != 0:
        raise ValueError(f"moe_expert_num={moe_expert_num} must be divisible by world_size={world_size}")

    buf = Buffer(
        group,
        per_round_tokens=per_round_tokens,
        quant_mode=0,
    )

    x, topk_idx, topk_weights = make_moe_inputs(
        num_tokens,
        hidden_size,
        moe_expert_num,
        topk,
        rank=rank,
    )

    (
        num_tokens_per_rank,
        _,
        num_tokens_per_expert,
        is_token_in_rank,
        _event,
    ) = buf.npu_get_dispatch_layout(topk_idx, moe_expert_num)

    torch._check(
        buf.round == expected_round,
        lambda: (
            f"layout round {buf.round} != expected {expected_round}; "
            f"num_tokens={num_tokens} per_round_tokens={per_round_tokens}."
        ),
    )
    torch._check(
        num_tokens_per_expert.numel() == moe_expert_num * buf.round,
        lambda: (
            f"num_tokens_per_expert size {num_tokens_per_expert.numel()} "
            f"!= moe_expert_num * round ({moe_expert_num * buf.round})."
        ),
    )

    recv_x, recv_topk_idx, recv_topk_weights, _, handle, _ = buf.npu_dispatch(
        x,
        topk_idx=topk_idx,
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        topk_weights=topk_weights,
        num_worst_tokens=num_tokens,
    )

    if isinstance(recv_x, tuple):
        recv_x_tensor = recv_x[0]
    else:
        recv_x_tensor = recv_x

    out, _, _ = buf.npu_combine(recv_x_tensor, handle, topk_weights=recv_topk_weights)

    _, _, _, _, _, _, topk_idx_from_handle, topk_weights_from_handle = handle
    golden = combine_golden(x, topk_idx_from_handle, topk_weights_from_handle)
    diff = calc_diff(out.float(), golden.float())
    torch._check(
        diff < SGL_COMBINE_DIFF_TOL,
        lambda: f"combine calc_diff {diff} >= {SGL_COMBINE_DIFF_TOL}",
    )

    print(
        f"[rank {rank}] layout/dispatch/combine ok (sgl-aligned): "
        f"num_tokens={num_tokens} hidden={hidden_size} experts={moe_expert_num} topk={topk} "
        f"round={buf.round} per_round_tokens={per_round_tokens} "
        f"recv_x={tuple(recv_x_tensor.shape)} out={tuple(out.shape)} calc_diff={diff:.2e} "
        f"ccl_buffer_size={buf.ccl_buffer_size.value}",
        flush=True,
    )

    dist.barrier()
    if rank == 0:
        print("[INFO] Buffer EP test finished.", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
