"""
MoE EP Dispatch/Combine 训练场景测试脚本
参照 DeepEP V2 README 4-helper 训练样例，验证 MoEElasticBuffer + EPHandle 的 dispatch/combine 功能。

用法：
  mpirun -np 8 python test_moe_ep_dispatch_combine.py

环境变量：
  HCCL_BUFFSIZE 建议设为 512MB+（ep=8, nmt=2048, h=7168 约需 ~226MB）
"""

import os
import torch
import torch.distributed as dist
import torch_npu
from npu_ops_transformer import MoEElasticBuffer, EPHandle


def init_process_group():
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.npu.set_device(rank)
    return rank, world_size


def test_dispatch_forward_basic():
    """测试 dispatch 前向主路径（precise 模式，do_cpu_sync=True）"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_experts = world_size * 8
    num_max_tokens_per_rank = 256
    hidden = 7168
    top_k = 8

    group = dist.new_group(backend="hccl")
    buf = MoEElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=top_k,
        num_experts=num_experts,
    )

    num_tokens = 256
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=f"npu:{rank}")
    topk_idx = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=f"npu:{rank}")
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.float32, device=f"npu:{rank}")

    recv_x, recv_topk_idx, recv_topk_weights, handle = buf.dispatch(x, topk_idx, topk_weights)

    assert recv_x.dtype == torch.bfloat16
    assert recv_x.shape[0] <= world_size * num_max_tokens_per_rank
    assert recv_x.shape[1] == hidden
    assert recv_topk_idx.shape == (recv_x.shape[0], top_k)
    assert recv_topk_weights.shape == (recv_x.shape[0], top_k)
    assert isinstance(handle, EPHandle)
    assert handle.dst_buffer_slot_idx.shape == (num_tokens, top_k)
    assert handle.recv_src_metadata.shape[1] == 3
    assert handle.num_recv_tokens_per_rank.shape[0] == world_size

    print(f"[rank {rank}] dispatch_forward_basic: recv_x shape={recv_x.shape}, "
          f"actualA={handle.num_recv_tokens}, PASS")

    buf.destroy()


def test_dispatch_forward_upper_bound():
    """测试 dispatch 上界分配模式（do_cpu_sync=False）"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_experts = world_size * 4
    num_max_tokens_per_rank = 128
    hidden = 4096
    top_k = 4

    group = dist.new_group(backend="hccl")
    buf = MoEElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=top_k,
        num_experts=num_experts,
    )

    num_tokens = 128
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=f"npu:{rank}")
    topk_idx = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=f"npu:{rank}")
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.float32, device=f"npu:{rank}")

    recv_x, recv_topk_idx, recv_topk_weights, handle = buf.dispatch(
        x, topk_idx, topk_weights, do_cpu_sync=False)

    upper_a = world_size * num_max_tokens_per_rank
    assert recv_x.shape[0] == upper_a
    print(f"[rank {rank}] dispatch_forward_upper_bound: recv_x shape={recv_x.shape}, PASS")

    buf.destroy()


def test_combine_forward_pure_sum():
    """测试 combine 前向纯 sum（C-A 模板，topk_weights=None）"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_experts = world_size * 4
    num_max_tokens_per_rank = 128
    hidden = 4096
    top_k = 4

    group = dist.new_group(backend="hccl")
    buf = MoEElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=top_k,
        num_experts=num_experts,
    )

    num_tokens = 128
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=f"npu:{rank}")
    topk_idx = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=f"npu:{rank}")
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.float32, device=f"npu:{rank}")

    recv_x, recv_topk_idx, recv_topk_weights, handle = buf.dispatch(x, topk_idx, topk_weights)

    A = recv_x.shape[0]
    weighted_x = recv_x * recv_topk_weights.unsqueeze(-1)

    combined_x, combined_topk_weights = buf.combine(weighted_x, handle)

    assert combined_x.shape == (num_tokens, hidden)
    assert combined_x.dtype == torch.bfloat16
    assert combined_topk_weights is None
    print(f"[rank {rank}] combine_forward_pure_sum: combined_x shape={combined_x.shape}, PASS")

    buf.destroy()


def test_4helper_full_training_step():
    """完整 4-helper 训练 step（对照 DeepEP V2 README 样例）"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_experts = world_size * 4
    num_max_tokens_per_rank = 128
    hidden = 4096
    top_k = 4

    group = dist.new_group(backend="hccl")
    buf = MoEElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=top_k,
        num_experts=num_experts,
    )

    num_tokens = 128

    # ---- 前向 dispatch ----
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=f"npu:{rank}")
    topk_idx = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=f"npu:{rank}")
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.float32, device=f"npu:{rank}")

    recv_x, recv_topk_idx, recv_topk_weights, handle = buf.dispatch(x, topk_idx, topk_weights)

    # ---- 用户侧 FFN (mock) ----
    A = recv_x.shape[0]
    expert_out = torch.randn(A, hidden, dtype=torch.bfloat16, device=f"npu:{rank}")
    weighted_out = expert_out * recv_topk_weights.unsqueeze(-1)

    # ---- 前向 combine ----
    combined_x, _ = buf.combine(weighted_out, handle)
    assert combined_x.shape == (num_tokens, hidden)

    # ---- 反向 dispatch (dispatch_backward = combine with topk_weights) ----
    grad_recv_x = torch.randn(A, hidden, dtype=torch.bfloat16, device=f"npu:{rank}")
    grad_recv_topk_weights = torch.randn(A, top_k, dtype=torch.float32, device=f"npu:{rank}")
    grad_x, grad_topk_weights = buf.combine(
        grad_recv_x, handle=handle, topk_weights=grad_recv_topk_weights)
    assert grad_x.shape == (num_tokens, hidden)
    assert grad_topk_weights is not None
    assert grad_topk_weights.shape == (num_tokens, top_k)

    # ---- 反向 combine (combine_backward = dispatch cached) ----
    grad_combined_x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=f"npu:{rank}")
    grad_x2, _, _, handle2 = buf.dispatch(grad_combined_x, handle=handle)
    assert handle2.num_recv_tokens == handle.num_recv_tokens

    print(f"[rank {rank}] 4helper_full_training_step: PASS, "
          f"forward combined_x={combined_x.shape}, "
          f"backward grad_x={grad_x.shape}, grad_x2={grad_x2.shape}")

    buf.destroy()


def test_dispatch_cached_mode():
    """测试 cached dispatch（combine 反向 = dispatch handle=...）"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_experts = world_size * 4
    num_max_tokens_per_rank = 128
    hidden = 4096
    top_k = 4

    group = dist.new_group(backend="hccl")
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

    recv_x1, _, _, handle1 = buf.dispatch(x, topk_idx, topk_weights)
    actual_a = handle1.num_recv_tokens

    grad_combined_x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=f"npu:{rank}")
    recv_x2, _, _, handle2 = buf.dispatch(grad_combined_x, handle=handle1)
    assert recv_x2.shape[0] == actual_a
    assert handle2.num_recv_tokens == actual_a

    print(f"[rank {rank}] dispatch_cached_mode: actualA={actual_a}, "
          f"cached recv_x shape={recv_x2.shape}, PASS")

    buf.destroy()


def test_ephandle_properties():
    """测试 EPHandle 属性"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_experts = world_size * 4
    num_max_tokens_per_rank = 64
    hidden = 2048
    top_k = 4
    num_local_experts = num_experts // world_size

    group = dist.new_group(backend="hccl")
    buf = MoEElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=top_k,
        num_experts=num_experts,
    )

    num_tokens = 32
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=f"npu:{rank}")
    topk_idx = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=f"npu:{rank}")
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.float32, device=f"npu:{rank}")

    _, _, _, handle = buf.dispatch(x, topk_idx, topk_weights)

    assert handle.num_experts == num_experts
    assert handle.expert_alignment == 1
    assert handle.num_max_tokens_per_rank == num_max_tokens_per_rank
    assert handle.topk_idx.shape == (num_tokens, top_k)
    assert handle.num_recv_tokens > 0
    assert handle.num_recv_tokens_per_rank.shape[0] == world_size
    assert handle.num_recv_tokens_per_expert.shape[0] == num_local_experts

    print(f"[rank {rank}] ephandle_properties: num_recv_tokens={handle.num_recv_tokens}, PASS")

    buf.destroy()


def test_topology_info():
    """测试 topology 只读属性"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_experts = world_size * 4

    group = dist.new_group(backend="hccl")
    buf = MoEElasticBuffer(
        group,
        num_max_tokens_per_rank=128,
        hidden=4096,
        num_topk=4,
        num_experts=num_experts,
    )

    topo = buf.topology
    assert topo.num_scaleup_ranks == min(world_size, 128)
    assert topo.num_scaleout_ranks == -(-world_size // 128)
    assert topo.intra_bw_GBps == 400.0
    assert topo.inter_bw_GBps == 50.0

    print(f"[rank {rank}] topology: scaleup={topo.num_scaleup_ranks}, "
          f"scaleout={topo.num_scaleout_ranks}, PASS")

    buf.destroy()


def main():
    rank, world_size = init_process_group()
    dist.barrier()

    print(f"[rank {rank}] === Starting MoE EP Dispatch/Combine tests (world_size={world_size}) ===")

    test_dispatch_forward_basic()
    dist.barrier()

    test_dispatch_forward_upper_bound()
    dist.barrier()

    test_combine_forward_pure_sum()
    dist.barrier()

    test_4helper_full_training_step()
    dist.barrier()

    test_dispatch_cached_mode()
    dist.barrier()

    test_ephandle_properties()
    dist.barrier()

    test_topology_info()
    dist.barrier()

    print(f"[rank {rank}] === All tests PASSED ===")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()