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
DispatchNormal smoke test.

Run with HCCL initialized, for example:
    torchrun --nproc_per_node=2 mc2/dispatch_normal/tests/test_dispatch_normal.py
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


def init_dist():
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    if world_size < 2:
        raise RuntimeError("WORLD_SIZE must be >= 2 for dispatch_normal communication test.")
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
    moe_expert_num = 16
    ccl_buffer_size = IntWrapper()
    quant_mode = 0
    real_max_bs = 8
    global_bs = real_max_bs * ep_world_size
    round_num = 4
    per_round_tokens = 1024
    num_recv_tokens = 32
    hidden_size = 1024
    topk = 4

    context = build_context(group, rank, world_size, ccl_buffer_size)
    x = torch.randn((real_max_bs, hidden_size), dtype=torch.float16, device="npu")
    topk_idx = torch.randint(0, moe_expert_num, (real_max_bs, topk), dtype=torch.int32, device="npu")

    recv_offset = [
        torch.zeros((round_num * moe_expert_num,), dtype=torch.int32, device="npu"),
        torch.zeros((round_num * moe_expert_num,), dtype=torch.int32, device="npu"),
    ]
    expert_output_offset = [
        torch.zeros((moe_expert_num // ep_world_size,), dtype=torch.int32, device="npu"),
        torch.zeros((moe_expert_num,), dtype=torch.int32, device="npu"),
        torch.zeros((round_num * moe_expert_num,), dtype=torch.int32, device="npu"),
    ]

    send_offset = torch.zeros((round_num * moe_expert_num,), dtype=torch.int32, device="npu")
    send_token_idx = torch.zeros((round_num * moe_expert_num,), dtype=torch.int32, device="npu")

    recv_x, x_scales, assist_info_for_combine, ext_handle_data = torch.ops.npu_ops_transformer.npu_dispatch_normal(
        context=context,
        x=x,
        topk_idx=topk_idx,
        send_offset=send_offset,
        send_token_idx=send_token_idx,
        recv_offset=recv_offset,
        expert_output_offset=expert_output_offset,
        ep_world_size=ep_world_size,
        ep_rank_id=ep_rank_id,
        moe_expert_num=moe_expert_num,
        ccl_buffer_size=ccl_buffer_size.value,
        num_recv_tokens=num_recv_tokens,
        quant_mode=quant_mode,
        real_max_bs=real_max_bs,
        global_bs=global_bs,
        round=round_num,
        per_round_tokens=per_round_tokens,
        dispatch_wait_recv_cost_stats=None,
    )

    print(
        f"[rank {rank}] recv_x={tuple(recv_x.shape)}, "
        f"x_scales={tuple(x_scales.shape)}, "
        f"assist_info_for_combine={tuple(assist_info_for_combine.shape)}, "
        f"ext_handle_data={tuple(ext_handle_data.shape)}"
    )

    dist.barrier()
    if rank == 0:
        print("[INFO] dispatch_normal smoke test finished.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
