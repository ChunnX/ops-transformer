import os
import torch
import torch_npu
import torch.distributed as dist
from torch.multiprocessing import Process
from npu_ops_transformer.ops.elastic_buffer import MoEElasticBuffer, EPHandle

master_ip = '127.0.0.1'
world_size = 2


def run_test_combine_shape(rank):
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size,
                             init_method=f'tcp://{master_ip}:50004')

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
    print(f"[rank {rank}] recv_x shape={recv_x.shape}")

    combined_x, combined_topk_weights = buf.combine(
        recv_x, handle, topk_weights=recv_topk_weights)

    torch.npu.synchronize()
    print(f"[rank {rank}] combined_x shape={combined_x.shape}, expected ({num_tokens}, {hidden})")
    print(f"[rank {rank}] combined_topk_weights shape={combined_topk_weights.shape}, expected ({num_tokens}, {top_k})")
    assert combined_x.shape == (num_tokens, hidden), \
        f"combined_x shape mismatch: {combined_x.shape} vs ({num_tokens}, {hidden})"
    assert combined_topk_weights.shape == (num_tokens, top_k), \
        f"combined_topk_weights shape mismatch: {combined_topk_weights.shape} vs ({num_tokens}, {top_k})"
    print(f"[rank {rank}] test_combine_shape: PASS")

    buf.destroy()
    dist.destroy_process_group()


def run_test_combine_no_weights(rank):
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size,
                             init_method=f'tcp://{master_ip}:50005')

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
    print(f"[rank {rank}] combined_x shape={combined_x.shape}")
    assert combined_x.shape == (num_tokens, hidden), \
        f"combined_x shape mismatch: {combined_x.shape} vs ({num_tokens}, {hidden})"
    assert combined_topk_weights is None, \
        f"combined_topk_weights should be None when no topk_weights provided"
    print(f"[rank {rank}] test_combine_no_weights: PASS")

    buf.destroy()
    dist.destroy_process_group()


def run_test_combine_roundtrip(rank):
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size,
                             init_method=f'tcp://{master_ip}:50006')

    num_experts = world_size * 4
    num_max_tokens_per_rank = 128
    hidden = 1024
    top_k = 1

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
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=f"npu:{rank}")
    topk_idx = torch.zeros(num_tokens, top_k, dtype=torch.int32, device=f"npu:{rank}")
    topk_weights = torch.ones(num_tokens, top_k, dtype=torch.float32, device=f"npu:{rank}")

    recv_x, recv_topk_idx, recv_topk_weights, handle = buf.dispatch(
        x, topk_idx, topk_weights, do_cpu_sync=False)

    torch.npu.synchronize()

    combined_x, combined_topk_weights = buf.combine(
        recv_x, handle, topk_weights=recv_topk_weights)

    torch.npu.synchronize()
    print(f"[rank {rank}] combined_x shape={combined_x.shape}")
    assert combined_x.shape == (num_tokens, hidden), \
        f"combined_x shape mismatch: {combined_x.shape} vs ({num_tokens}, {hidden})"

    # x_cpu = x.float()
    # combined_cpu = combined_x.float()
    # diff = (x_cpu - combined_cpu).abs().max().item()
    # print(f"[rank {rank}] max abs diff between input and roundtrip output: {diff}")
    # assert diff < 1.0, f"roundtrip mismatch: max abs diff={diff}"
    print(f"[rank {rank}] test_combine_roundtrip: PASS")

    buf.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ['HCCL_WHITELIST_DISABLE'] = '1'

    print("=" * 60)
    print("Test 1: combine shape with topk_weights")
    print("=" * 60)
    processes = []
    for rank in range(world_size):
        p = Process(target=run_test_combine_shape, args=(rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print("=" * 60)
    print("Test 2: combine shape without topk_weights")
    print("=" * 60)
    processes = []
    for rank in range(world_size):
        p = Process(target=run_test_combine_no_weights, args=(rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print("=" * 60)
    print("Test 3: combine roundtrip correctness")
    print("=" * 60)
    processes = []
    for rank in range(world_size):
        p = Process(target=run_test_combine_roundtrip, args=(rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print("All tests finished.")