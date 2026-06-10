# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MoE EP Dispatch/Combine 训练场景测试脚本

参照 test_moe_ep_combine_standalone.py 的 Process 模式，
验证 MoEElasticBuffer + EPHandle 的 dispatch/combine 功能。
注意：kernel 尚有 stub (printf+return)，仅验证形状和流程，不验证精度。

用法：
  python test_moe_ep_dispatch_combine.py

环境变量：
  HCCL_WHITELIST_DISABLE=1（脚本内自动设置）
"""

import os
import torch
import torch_npu
import torch.distributed as dist
from torch.multiprocessing import Process
from npu_ops_transformer.ops.elastic_buffer import MoEElasticBuffer, EPHandle

master_ip = '127.0.0.1'
world_size = 2
BASE_PORT = 50020


def run_test_dispatch_forward_upper_bound(rank):
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size,
                             init_method=f'tcp://{master_ip}:{BASE_PORT}')

    num_experts = world_size * 4
    num_max_tokens_per_rank = 128
    hidden = 4096
    top_k = 4

    group = dist.new_group(backend="hccl", ranks=list(range(world_size)))
    group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    buf = MoEElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=top_k,
        num_experts=num_experts,
    )

    num_tokens = 64
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=f"npu:{rank}")
    topk_idx = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=f"npu:{rank}")
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.float32, device=f"npu:{rank}")

    recv_x, recv_topk_idx, recv_topk_weights, handle = buf.dispatch(
        x, topk_idx, topk_weights, do_cpu_sync=False)

    torch.npu.synchronize()

    upper_a = world_size * num_max_tokens_per_rank
    assert recv_x.shape == (upper_a, hidden), f"recv_x shape={recv_x.shape}, expected ({upper_a}, {hidden})"
    assert recv_topk_idx.shape == (upper_a, top_k)
    assert recv_topk_weights.shape == (upper_a,)
    assert handle.dst_buffer_slot_idx.shape == (num_tokens, top_k)
    assert handle.recv_src_metadata.shape[1] == 3
    assert handle.num_recv_tokens_per_rank.shape[0] == world_size
    print(f"[rank {rank}] dispatch_forward_upper_bound: recv_x shape={recv_x.shape}, PASS")

    buf.destroy()
    dist.destroy_process_group()


def run_test_combine_forward_pure_sum(rank):
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size,
                             init_method=f'tcp://{master_ip}:{BASE_PORT + 1}')

    num_experts = world_size * 4
    num_max_tokens_per_rank = 128
    hidden = 4096
    top_k = 4

    group = dist.new_group(backend="hccl", ranks=list(range(world_size)))
    group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    buf = MoEElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=top_k,
        num_experts=num_experts,
    )

    num_tokens = 64
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=f"npu:{rank}")
    topk_idx = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=f"npu:{rank}")
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.float32, device=f"npu:{rank}")

    recv_x, recv_topk_idx, recv_topk_weights, handle = buf.dispatch(
        x, topk_idx, topk_weights, do_cpu_sync=False)

    torch.npu.synchronize()

    combined_x, combined_topk_weights = buf.combine(recv_x, handle)

    torch.npu.synchronize()
    assert combined_x.shape == (num_tokens, hidden), f"combined_x shape={combined_x.shape}"
    assert combined_x.dtype == torch.bfloat16
    assert combined_topk_weights is None
    print(f"[rank {rank}] combine_forward_pure_sum: combined_x shape={combined_x.shape}, PASS")

    buf.destroy()
    dist.destroy_process_group()


def run_test_combine_forward_with_weights(rank):
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size,
                             init_method=f'tcp://{master_ip}:{BASE_PORT + 2}')

    num_experts = world_size * 4
    num_max_tokens_per_rank = 128
    hidden = 4096
    top_k = 4

    group = dist.new_group(backend="hccl", ranks=list(range(world_size)))
    group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    buf = MoEElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=top_k,
        num_experts=num_experts,
    )

    num_tokens = 64
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=f"npu:{rank}")
    topk_idx = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=f"npu:{rank}")
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.float32, device=f"npu:{rank}")

    recv_x, recv_topk_idx, recv_topk_weights, handle = buf.dispatch(
        x, topk_idx, topk_weights, do_cpu_sync=False)

    torch.npu.synchronize()

    combined_x, combined_topk_weights = buf.combine(
        recv_x, handle, topk_weights=recv_topk_weights)

    torch.npu.synchronize()
    assert combined_x.shape == (num_tokens, hidden), f"combined_x shape={combined_x.shape}"
    assert combined_x.dtype == torch.bfloat16
    assert combined_topk_weights is not None
    assert combined_topk_weights.shape == (num_tokens, top_k)
    print(f"[rank {rank}] combine_forward_with_weights: combined_x={combined_x.shape}, "
          f"combined_topk_weights={combined_topk_weights.shape}, PASS")

    buf.destroy()
    dist.destroy_process_group()


def run_test_ephandle_properties(rank):
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size,
                             init_method=f'tcp://{master_ip}:{BASE_PORT + 3}')

    num_experts = world_size * 4
    num_max_tokens_per_rank = 128
    hidden = 4096
    top_k = 4
    num_local_experts = num_experts // world_size

    group = dist.new_group(backend="hccl", ranks=list(range(world_size)))
    group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    buf = MoEElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=top_k,
        num_experts=num_experts,
    )

    num_tokens = 64
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=f"npu:{rank}")
    topk_idx = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=f"npu:{rank}")
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.float32, device=f"npu:{rank}")

    _, _, _, handle = buf.dispatch(x, topk_idx, topk_weights, do_cpu_sync=False)
    torch.npu.synchronize()

    assert handle.num_experts == num_experts
    assert handle.expert_alignment == 1
    assert handle.num_max_tokens_per_rank == num_max_tokens_per_rank
    assert handle.topk_idx.shape == (num_tokens, top_k)
    assert handle.num_recv_tokens_per_rank.shape[0] == world_size
    assert handle.num_recv_tokens_per_expert.shape[0] == num_local_experts
    print(f"[rank {rank}] ephandle_properties: PASS")

    buf.destroy()
    dist.destroy_process_group()


def run_test_bias_rejected(rank):
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size,
                             init_method=f'tcp://{master_ip}:{BASE_PORT + 4}')

    num_experts = world_size * 4
    num_max_tokens_per_rank = 128
    hidden = 4096
    top_k = 4

    group = dist.new_group(backend="hccl", ranks=list(range(world_size)))
    group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    buf = MoEElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=top_k,
        num_experts=num_experts,
    )

    num_tokens = 64
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=f"npu:{rank}")
    topk_idx = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=f"npu:{rank}")
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.float32, device=f"npu:{rank}")

    recv_x, _, _, handle = buf.dispatch(x, topk_idx, topk_weights, do_cpu_sync=False)
    torch.npu.synchronize()

    bias = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=f"npu:{rank}")
    try:
        buf.combine(recv_x, handle, bias=bias)
        print(f"[rank {rank}] test_bias_rejected: FAIL - bias should be rejected")
    except (RuntimeError, AssertionError) as e:
        print(f"[rank {rank}] test_bias_rejected: PASS (bias correctly rejected)")

    buf.destroy()
    dist.destroy_process_group()


def run_test(func):
    os.environ['HCCL_WHITELIST_DISABLE'] = '1'
    processes = []
    for rank in range(world_size):
        p = Process(target=func, args=(rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == "__main__":
    tests = [
        ("Test 1: dispatch forward upper bound", run_test_dispatch_forward_upper_bound),
        ("Test 2: combine forward pure sum (no topk_weights)", run_test_combine_forward_pure_sum),
        ("Test 3: combine forward with topk_weights", run_test_combine_forward_with_weights),
        ("Test 4: ephandle properties", run_test_ephandle_properties),
        ("Test 5: bias rejected", run_test_bias_rejected),
    ]

    for name, func in tests:
        print("=" * 60)
        print(name)
        print("=" * 60)
        run_test(func)

    print("All dispatch/combine tests finished.")