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
MoeCombineNormal smoke test.

Run with HCCL initialized, for example:
    torchrun --nproc_per_node=2 experimental/mc2/combine_normal/tests/test_moe_combine_normal.py
"""

import os
import torch
import torch_npu
import torch.distributed as dist
import npu_ops_transformer  # noqa: F401


def init_dist():
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    if world_size < 2:
        raise RuntimeError("WORLD_SIZE must be >= 2 for moe_combine_normal communication test.")
    torch_npu.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    return local_rank, rank, world_size


def main():
    _, rank, world_size = init_dist()
    group = dist.distributed_c10d._get_default_group()
    ep_group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank, init_comm=False)

    ep_world_size = world_size
    ep_rank_id = rank
    moe_expert_num = 16
    bs = 8
    hidden_size = 1024
    topk = 4
    global_bs = bs * ep_world_size

    num_recv_tokens = 32

    recv_x = torch.randn((num_recv_tokens, hidden_size), dtype=torch.float16, device="npu")
    token_src_info = torch.zeros((num_recv_tokens * 3,), dtype=torch.int32, device="npu")
    ep_recv_counts = torch.zeros((moe_expert_num,), dtype=torch.int32, device="npu")
    recv_topk_weights = torch.randn((bs, topk), dtype=torch.float, device="npu")

    x = torch.ops.npu_ops_transformer.npu_moe_combine_normal(
        recv_x=recv_x,
        token_src_info=token_src_info,
        ep_recv_counts=ep_recv_counts,
        recv_topk_weights=recv_topk_weights,
        tp_recv_counts=None,
        ep_group_name=ep_group_name,
        ep_world_size=ep_world_size,
        ep_rank_id=ep_rank_id,
        moe_expert_num=moe_expert_num,
        tp_group_name="",
        tp_world_size=0,
        tp_rank_id=0,
        global_bs=global_bs,
        combine_send_cost_stats=None,
    )

    print(f"[rank {rank}] x={tuple(x.shape)}")

    dist.barrier()
    if rank == 0:
        print("[INFO] moe_combine_normal smoke test finished.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()