import os
import torch
import torch_npu
import torch.distributed as dist
from torch.multiprocessing import Process
from npu_ops_transformer.ops.elastic_buffer import MoEElasticBuffer
from npu_ops_transformer.ops.moe_ep_combine import npu_moe_ep_combine

master_ip = '127.0.0.1'
world_size = 2


def _make_handle_tensors(rank, ep_world_size, num_tokens, top_k, num_experts,
                          num_max_tokens_per_rank, hidden):
    A = ep_world_size * num_max_tokens_per_rank
    device = f"npu:{rank}"
    num_local_experts = num_experts // ep_world_size

    topk_idx = torch.zeros(num_tokens, top_k, dtype=torch.int32, device=device)
    recv_src_metadata = torch.full((A, 3), -1, dtype=torch.int32, device=device)
    num_recv_tokens_per_expert = torch.zeros(num_local_experts, dtype=torch.int64, device=device)

    recv_src_metadata[:num_tokens, 0] = torch.arange(num_tokens, dtype=torch.int32, device=device)
    recv_src_metadata[:num_tokens, 1] = torch.tensor(rank, dtype=torch.int32, device=device)
    recv_src_metadata[:num_tokens, 2] = torch.arange(num_tokens, dtype=torch.int32, device=device)

    slot_val = torch.arange(num_tokens, dtype=torch.int32, device=device)
    topk_idx[:, 0] = slot_val
    if top_k > 1:
        topk_idx[:, 1] = slot_val

    return topk_idx, recv_src_metadata, num_recv_tokens_per_expert


def run_test_combine_basic(rank):
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size,
                             init_method=f'tcp://{master_ip}:50007')

    num_experts = world_size * 4
    num_max_tokens_per_rank = 16
    hidden = 1024
    top_k = 2

    group = dist.new_group(backend="hccl", ranks=list(range(world_size)))
    group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    buf = MoEElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=top_k,
        num_experts=num_experts,
    )

    num_tokens = 8
    A = world_size * num_max_tokens_per_rank

    topk_idx, recv_src_metadata, num_recv_per_expert = _make_handle_tensors(
        rank, world_size, num_tokens, top_k, num_experts, num_max_tokens_per_rank, hidden)

    x = torch.randn(A, hidden, dtype=torch.bfloat16, device=f"npu:{rank}")

    combined_x, combined_topk_weights = npu_moe_ep_combine(
        buf.context, x, topk_idx, recv_src_metadata, num_recv_per_expert,
        None, None, None,
        buf.ep_world_size, rank, num_experts, num_max_tokens_per_rank, buf.ccl_buffer_size,
    )

    torch.npu.synchronize()
    print(f"[rank {rank}] combined_x shape={combined_x.shape}, expected ({num_tokens}, {hidden})")
    assert combined_x.shape == (num_tokens, hidden), \
        f"combined_x shape mismatch: {combined_x.shape} vs ({num_tokens}, {hidden})"
    assert combined_topk_weights is None, \
        f"combined_topk_weights should be None when topk_weights not provided"
    print(f"[rank {rank}] test_combine_basic: PASS")

    buf.destroy()
    dist.destroy_process_group()


def run_test_combine_with_weights(rank):
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size,
                             init_method=f'tcp://{master_ip}:50008')

    num_experts = world_size * 4
    num_max_tokens_per_rank = 16
    hidden = 1024
    top_k = 2

    group = dist.new_group(backend="hccl", ranks=list(range(world_size)))
    group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    buf = MoEElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=top_k,
        num_experts=num_experts,
    )

    num_tokens = 8
    A = world_size * num_max_tokens_per_rank

    topk_idx, recv_src_metadata, num_recv_per_expert = _make_handle_tensors(
        rank, world_size, num_tokens, top_k, num_experts, num_max_tokens_per_rank, hidden)

    x = torch.randn(A, hidden, dtype=torch.bfloat16, device=f"npu:{rank}")
    topk_weights = torch.rand(A, dtype=torch.float32, device=f"npu:{rank}")

    combined_x, combined_topk_weights = npu_moe_ep_combine(
        buf.context, x, topk_idx, recv_src_metadata, num_recv_per_expert,
        topk_weights, None, None,
        buf.ep_world_size, rank, num_experts, num_max_tokens_per_rank, buf.ccl_buffer_size,
    )

    torch.npu.synchronize()
    print(f"[rank {rank}] combined_x shape={combined_x.shape}, expected ({num_tokens}, {hidden})")
    print(f"[rank {rank}] combined_topk_weights shape={combined_topk_weights.shape}, expected ({num_tokens}, {top_k})")
    assert combined_x.shape == (num_tokens, hidden), \
        f"combined_x shape mismatch: {combined_x.shape} vs ({num_tokens}, {hidden})"
    assert combined_topk_weights.shape == (num_tokens, top_k), \
        f"combined_topk_weights shape mismatch: {combined_topk_weights.shape} vs ({num_tokens}, {top_k})"
    print(f"[rank {rank}] test_combine_with_weights: PASS")

    buf.destroy()
    dist.destroy_process_group()


def run_test_combine_large_tokens(rank):
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size,
                             init_method=f'tcp://{master_ip}:50009')

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
    A = world_size * num_max_tokens_per_rank

    topk_idx, recv_src_metadata, num_recv_per_expert = _make_handle_tensors(
        rank, world_size, num_tokens, top_k, num_experts, num_max_tokens_per_rank, hidden)

    x = torch.randn(A, hidden, dtype=torch.bfloat16, device=f"npu:{rank}")
    topk_weights = torch.rand(A, dtype=torch.float32, device=f"npu:{rank}")

    combined_x, combined_topk_weights = npu_moe_ep_combine(
        buf.context, x, topk_idx, recv_src_metadata, num_recv_per_expert,
        topk_weights, None, None,
        buf.ep_world_size, rank, num_experts, num_max_tokens_per_rank, buf.ccl_buffer_size,
    )

    torch.npu.synchronize()
    print(f"[rank {rank}] combined_x shape={combined_x.shape}, expected ({num_tokens}, {hidden})")
    print(f"[rank {rank}] combined_topk_weights shape={combined_topk_weights.shape}, expected ({num_tokens}, {top_k})")
    assert combined_x.shape == (num_tokens, hidden)
    assert combined_topk_weights.shape == (num_tokens, top_k)
    print(f"[rank {rank}] test_combine_large_tokens: PASS")

    buf.destroy()
    dist.destroy_process_group()


def run_test_combine_bias_rejected(rank):
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size,
                             init_method=f'tcp://{master_ip}:50010')

    num_experts = world_size * 4
    num_max_tokens_per_rank = 16
    hidden = 1024
    top_k = 2

    group = dist.new_group(backend="hccl", ranks=list(range(world_size)))
    group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    buf = MoEElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=top_k,
        num_experts=num_experts,
    )

    num_tokens = 8
    A = world_size * num_max_tokens_per_rank

    topk_idx, recv_src_metadata, num_recv_per_expert = _make_handle_tensors(
        rank, world_size, num_tokens, top_k, num_experts, num_max_tokens_per_rank, hidden)

    x = torch.randn(A, hidden, dtype=torch.bfloat16, device=f"npu:{rank}")
    bias = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=f"npu:{rank}")

    try:
        combined_x, combined_topk_weights = npu_moe_ep_combine(
            buf.context, x, topk_idx, recv_src_metadata, num_recv_per_expert,
            None, bias, None,
            buf.ep_world_size, rank, num_experts, num_max_tokens_per_rank, buf.ccl_buffer_size,
        )
        torch.npu.synchronize()
        print(f"[rank {rank}] test_combine_bias_rejected: FAIL - bias should be rejected")
        assert False, "bias should have been rejected"
    except (RuntimeError, torch._C._LinAlgError) as e:
        print(f"[rank {rank}] test_combine_bias_rejected: PASS (bias correctly rejected: {e})")

    buf.destroy()
    dist.destroy_process_group()


def run_combine(func, port_offset):
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
        ("Test 1: combine basic (no topk_weights)", run_test_combine_basic, 50007),
        ("Test 2: combine with topk_weights", run_test_combine_with_weights, 50008),
        ("Test 3: combine large tokens", run_test_combine_large_tokens, 50009),
        ("Test 4: combine bias rejected", run_test_combine_bias_rejected, 50010),
    ]

    for name, func, port in tests:
        print("=" * 60)
        print(name)
        print("=" * 60)
        run_combine(func, port)

    print("All standalone combine tests finished.")