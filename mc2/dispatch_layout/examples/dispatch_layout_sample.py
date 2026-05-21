# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Description: Standalone test for dispatch_layout operator.
# This demonstrates computing dispatch layout from top-k expert indices.
# -----------------------------------------------------------------------------------------------------------

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'common', 'examples'))

import torch
import torch.distributed as dist
import torch_npu
from npu_ops_transformer import dispatch_layout, get_dispatch_layout
from utils import init_dist, bench


def test_dispatch_layout(args, num_local_ranks, local_rank, num_ranks, rank, group):
    num_tokens = args.num_tokens
    num_topk = args.num_topk
    num_experts = args.num_experts
    
    assert num_experts % num_ranks == 0
    experts_per_rank = num_experts // num_ranks
    
    if local_rank == 0:
        print(f"Testing dispatch_layout: num_tokens={num_tokens}, num_topk={num_topk}, num_experts={num_experts}")
    
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="npu").abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    
    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int32, device="npu")
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    
    recv_expand_per_expert = torch.zeros((num_experts,), dtype=torch.int32, device="npu")
    send_recv_map = torch.zeros((num_ranks, num_experts), dtype=torch.int32, device="npu")
    expert_tokens_map = torch.zeros((num_ranks, num_experts), dtype=torch.int32, device="npu")
    send_token_idx = torch.zeros((num_ranks, num_tokens), dtype=torch.int32, device="npu")
    recv_token_idx = torch.zeros((num_ranks, num_tokens), dtype=torch.int32, device="npu")
    
    ep_hcomm_info = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    ep_hcomm_info = ep_hcomm_info.encode('utf-8')
    
    dispatch_layout(
        topkIdx=topk_idx,
        numTokensPerExpert=num_tokens_per_expert,
        recvExpandPerExpert=recv_expand_per_expert,
        sendRecvMap=send_recv_map,
        expertTokensMap=expert_tokens_map,
        sendTokenIdx=send_token_idx,
        recvTokenIdx=recv_token_idx,
        groupEp=ep_hcomm_info,
        rank=rank,
        numRanks=num_ranks
    )
    
    if local_rank == 0:
        print("dispatch_layout completed successfully")
        print(f"recv_expand_per_expert shape: {recv_expand_per_expert.shape}")
        print(f"send_recv_map shape: {send_recv_map.shape}")
    
    t = bench(lambda: dispatch_layout(
        topkIdx=topk_idx,
        numTokensPerExpert=num_tokens_per_expert,
        recvExpandPerExpert=recv_expand_per_expert,
        sendRecvMap=send_recv_map,
        expertTokensMap=expert_tokens_map,
        sendTokenIdx=send_token_idx,
        recvTokenIdx=recv_token_idx,
        groupEp=ep_hcomm_info,
        rank=rank,
        numRanks=num_ranks
    ))[0]
    
    if local_rank == 0:
        print(f"[tuning] dispatch_layout avg_t: {t * 1e6:.2f} us")


def test_get_dispatch_layout(args, num_local_ranks, local_rank, num_ranks, rank, group):
    num_tokens = args.num_tokens
    num_topk = args.num_topk
    num_experts = args.num_experts
    
    if local_rank == 0:
        print(f"Testing get_dispatch_layout: num_tokens={num_tokens}, num_topk={num_topk}, num_experts={num_experts}")
    
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="npu").abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    
    num_tokens_per_expert, send_token_idx_small = get_dispatch_layout(
        topkIdx=topk_idx,
        numExperts=num_experts,
        numRanks=num_ranks
    )
    
    if local_rank == 0:
        print("get_dispatch_layout completed successfully")
        print(f"num_tokens_per_expert shape: {num_tokens_per_expert.shape}")
        print(f"send_token_idx_small shape: {send_token_idx_small.shape}")
    
    t = bench(lambda: get_dispatch_layout(
        topkIdx=topk_idx,
        numExperts=num_experts,
        numRanks=num_ranks
    ))[0]
    
    if local_rank == 0:
        print(f"[tuning] get_dispatch_layout avg_t: {t * 1e6:.2f} us")


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank)
    
    if args.test_type == "dispatch_layout":
        test_dispatch_layout(args, num_local_ranks, local_rank, num_ranks, rank, group)
    elif args.test_type == "get_dispatch_layout":
        test_get_dispatch_layout(args, num_local_ranks, local_rank, num_ranks, rank, group)
    else:
        test_dispatch_layout(args, num_local_ranks, local_rank, num_ranks, rank, group)
        test_get_dispatch_layout(args, num_local_ranks, local_rank, num_ranks, rank, group)
    
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test dispatch_layout operator")
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--test-type", type=str, default="both", 
                        choices=["dispatch_layout", "get_dispatch_layout", "both"],
                        help="Which test to run")
    args = parser.parse_args()
    
    torch.multiprocessing.spawn(test_loop, args=(args.num_processes, args), nprocs=args.num_processes)