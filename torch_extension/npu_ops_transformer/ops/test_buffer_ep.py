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
Buffer EP normal smoke test (layout -> dispatch -> combine).

Run with HCCL initialized, for example:
    torchrun --nproc_per_node=2 torch_extension/npu_ops_transformer/ops/test_buffer_ep.py
"""

import os
import torch
import torch_npu
import torch.distributed as dist
import npu_ops_transformer  # noqa: F401
from npu_ops_transformer.ops.buffer import Buffer


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

    moe_expert_num = 16
    num_tokens = 8
    round_num = 4
    # layout round = ceil(num_tokens / per_round_tokens); use 2 so 8 tokens -> 4 rounds
    per_round_tokens = 2
    topk = 4
    hidden_size = 1024

    buf = Buffer(
        group,
        local_ranksize=world_size,
        per_round_tokens=per_round_tokens,
        round_num=round_num,
        quant_mode=0,
    )

    topk_idx = torch.randint(0, moe_expert_num, (num_tokens, topk), dtype=torch.int64, device="npu")
    x = torch.randn((num_tokens, hidden_size), dtype=torch.float16, device="npu")

    (
        num_tokens_per_rank,
        _,
        num_tokens_per_expert,
        is_token_in_rank,
        _event,
    ) = buf.npu_get_dispatch_layout(topk_idx, moe_expert_num)

    torch._check(
        buf.round == round_num,
        lambda: f"layout round {buf.round} != expected {round_num}.",
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
        num_worst_tokens=num_tokens,
    )

    if isinstance(recv_x, tuple):
        recv_x_tensor = recv_x[0]
    else:
        recv_x_tensor = recv_x

    out, _, _ = buf.npu_combine(recv_x_tensor, handle, topk_weights=recv_topk_weights)

    print(
        f"[rank {rank}] layout/dispatch/combine ok: recv_x={tuple(recv_x_tensor.shape)}, "
        f"out={tuple(out.shape)}, ccl_buffer_size={buf.ccl_buffer_size.value}"
    )

    dist.barrier()
    if rank == 0:
        print("[INFO] Buffer EP smoke test finished.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
