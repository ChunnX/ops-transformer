# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Description: Example for notify_dispatch test.
# Note: notify_dispatch is an internal operator used by moe_dispatch_prefill.
#       For end-to-end testing, please use moe_prefill_sample.py instead.
# -----------------------------------------------------------------------------------------------------------

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'common', 'examples'))

import torch
import torch.distributed as dist
import torch_npu
from npu_ops_transformer import notify_dispatch
from utils import init_dist, bench


def test_notify_dispatch(args, num_local_ranks, local_rank, num_ranks, rank, group):
    num_tokens = args.num_tokens
    num_experts = args.num_experts
    
    ep_hcomm_info = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    ep_hcomm_info = ep_hcomm_info.encode('utf-8')
    
    experts_per_rank = num_experts // num_ranks
    
    topk_idx = torch.randint(0, num_experts, (num_tokens, args.num_topk), dtype=torch.int32, device="npu")
    
    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device="npu")
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    
    send_per_group = 3
    send_data = torch.zeros((num_experts * send_per_group), dtype=torch.int, device="npu")
    send_data_offset = torch.zeros((num_experts,), dtype=torch.int, device="npu")
    recv_data = torch.zeros((num_experts * send_per_group), dtype=torch.int, device="npu")
    total_recv_token = torch.zeros((1,), dtype=torch.int, device="npu")
    recv_count = torch.zeros((num_experts,), dtype=torch.int, device="npu")
    recv_offset = torch.zeros((num_experts,), dtype=torch.int, device="npu")
    max_bs = torch.zeros((1,), dtype=torch.int, device="npu")
    recv_tokens_per_expert = torch.zeros((experts_per_rank,), dtype=torch.long, device="npu")
    
    send_count = send_per_group * experts_per_rank * num_ranks
    
    if local_rank == 0:
        print(f"Testing notify_dispatch with num_tokens={num_tokens}, num_experts={num_experts}")
    
    notify_dispatch(
        sendData=send_data,
        numTokensPerExpert=num_tokens_per_expert,
        sendCount=send_count,
        numTokens=num_tokens,
        groupEp=ep_hcomm_info,
        numRanks=num_ranks,
        rank=rank,
        localRankSize=num_ranks,
        localRankId=rank,
        sendDataOffset=send_data_offset,
        recvData=recv_data,
        totalRecvToken=total_recv_token,
        recvCount=recv_count,
        recvOffset=recv_offset,
        maxBs=max_bs,
        recvTokensPerExpert=recv_tokens_per_expert
    )
    
    if local_rank == 0:
        print(f"notify_dispatch completed: total_recv_token={total_recv_token.item()}, max_bs={max_bs.item()}")
    
    t = bench(lambda: notify_dispatch(
        sendData=send_data,
        numTokensPerExpert=num_tokens_per_expert,
        sendCount=send_count,
        numTokens=num_tokens,
        groupEp=ep_hcomm_info,
        numRanks=num_ranks,
        rank=rank,
        localRankSize=num_ranks,
        localRankId=rank,
        sendDataOffset=send_data_offset,
        recvData=recv_data,
        totalRecvToken=total_recv_token,
        recvCount=recv_count,
        recvOffset=recv_offset,
        maxBs=max_bs,
        recvTokensPerExpert=recv_tokens_per_expert
    ))[0]
    
    if local_rank == 0:
        print(f"[tuning] notify_dispatch avg_t: {t * 1e6:.2f} us")


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank)
    test_notify_dispatch(args, num_local_ranks, local_rank, num_ranks, rank, group)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test notify_dispatch operator")
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=8)
    args = parser.parse_args()
    
    torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)