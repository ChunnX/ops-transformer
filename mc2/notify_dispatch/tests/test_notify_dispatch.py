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
NotifyDispatch smoke test (layout -> notify -> dispatch_normal).

Run with HCCL initialized, for example:
    torchrun --nproc_per_node=2 mc2/notify_dispatch/tests/test_notify_dispatch.py
"""

import os
import torch
import torch_npu
import torch.distributed as dist
import npu_ops_transformer  # noqa: F401
from npu_ops_transformer.ops.update_context import update_context


class IntWrapper:
    def __init__(self, value=0):
        self.value = value


SEND_PER_GROUP = 3


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
    group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank, init_comm=False)
    ok = update_context(group_name, world_size, ccl_buffer_size, context)
    if not ok:
        raise RuntimeError("update_context failed.")
    if ccl_buffer_size.value <= 0:
        raise RuntimeError(f"invalid ccl_buffer_size from update_context: {ccl_buffer_size.value}")
    return context


def main():
    _, rank, world_size = init_dist()
    group = dist.distributed_c10d._get_default_group()

    ep_world_size = world_size
    ep_rank_id = rank
    local_rank_size = world_size
    local_rank_id = ep_rank_id % local_rank_size
    moe_expert_num = 16
    num_tokens = 8
    round_num = 4
    per_round_tokens = 2  # ceil(8/2) == 4 rounds, matches round_num
    topk = 4
    hidden_size = 1024

    ccl_buffer_size = IntWrapper()
    context = build_context(group, rank, world_size, ccl_buffer_size)

    topk_idx = torch.randint(0, moe_expert_num, (num_tokens, topk), dtype=torch.int64, device="npu")
    x = torch.randn((num_tokens, hidden_size), dtype=torch.float16, device="npu")

    _, _, num_tokens_per_expert, _, ext_handle_data = torch.ops.npu_ops_transformer.npu_dispatch_layout(
        topk_idx,
        ep_world_size,
        moe_expert_num,
        local_rank_size,
        per_round_tokens,
        ep_rank_id,
    )
    send_data, send_token_idx = ext_handle_data[0], ext_handle_data[1]
    layout_round = num_tokens_per_expert.numel() // moe_expert_num
    torch._check(
        layout_round == round_num,
        lambda: f"layout round {layout_round} != round_num {round_num}; fix per_round_tokens.",
    )
    send_count = SEND_PER_GROUP * moe_expert_num * layout_round

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

    num_recv_tokens = int(total_recv_tokens.item())
    if num_recv_tokens <= 0:
        num_recv_tokens = 1
    real_max_bs = max(int(max_bs.item()), num_tokens)
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

    print(
        f"[rank {rank}] notify: send_offset={tuple(send_offset.shape)}, "
        f"recv_data={tuple(recv_data.shape)}, total_recv_tokens={num_recv_tokens}, max_bs={real_max_bs}"
    )
    print(
        f"[rank {rank}] dispatch: recv_x={tuple(recv_x.shape)}, "
        f"x_scales={tuple(x_scales.shape)}, assist={tuple(assist_info_for_combine.shape)}, "
        f"recv_tokens_per_expert={tuple(recv_tokens_per_expert.shape)}"
    )

    dist.barrier()
    if rank == 0:
        print("[INFO] notify_dispatch pipeline smoke test finished.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
