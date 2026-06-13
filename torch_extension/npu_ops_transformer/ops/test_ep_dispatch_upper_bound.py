import os
import torch
import torch_npu
import torch.distributed as dist
from torch.multiprocessing import Process
from npu_ops_transformer.ops.elastic_buffer import MoEElasticBuffer

master_ip = '127.0.0.1'
world_size = 2


def run_test_upper_bound(rank):
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size,
                             init_method=f'tcp://{master_ip}:50003')

    num_experts = world_size * 4
    num_max_tokens_per_rank = 128
    hidden = 4096
    top_k = 4
    upper_a = world_size * num_max_tokens_per_rank * min(top_k, num_experts // world_size)
    num_tokens = 128

    group = dist.new_group(backend="hccl", ranks=list(range(world_size)))
    group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

    buf = MoEElasticBuffer(
        group, num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden, num_topk=top_k, num_experts=num_experts)

    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=f"npu:{rank}")
    topk_idx = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=f"npu:{rank}")
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.float32, device=f"npu:{rank}")

    recv_x, _, _, handle = buf.dispatch(x, topk_idx, topk_weights, do_cpu_sync=False)
    torch.npu.synchronize()
    assert recv_x.shape[0] == upper_a, f"upper_bound: {recv_x.shape[0]} != {upper_a}"
    print(f"[rank {rank}] test_upper_bound: PASS (recv_x shape={recv_x.shape})")

    buf.destroy()
    dist.destroy_process_group()


def run_test_do_cpu_sync(rank):
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size,
                             init_method=f'tcp://{master_ip}:50004')

    num_experts = world_size * 4
    num_max_tokens_per_rank = 128
    hidden = 4096
    top_k = 4
    upper_a = world_size * num_max_tokens_per_rank * min(top_k, num_experts // world_size)
    num_tokens = 128

    group = dist.new_group(backend="hccl", ranks=list(range(world_size)))
    group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

    buf = MoEElasticBuffer(
        group, num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden, num_topk=top_k, num_experts=num_experts)

    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=f"npu:{rank}")
    topk_idx = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=f"npu:{rank}")
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.float32, device=f"npu:{rank}")

    recv_x, _, _, handle = buf.dispatch(x, topk_idx, topk_weights, do_cpu_sync=True)
    actual_a = recv_x.shape[0]
    torch.npu.synchronize()
    assert actual_a > 0, f"do_cpu_sync: actual_a should > 0, got {actual_a}"
    assert actual_a <= upper_a, f"do_cpu_sync: actual_a {actual_a} > upper_a {upper_a}"
    print(f"[rank {rank}] test_do_cpu_sync: PASS (actual_a={actual_a}, upper_a={upper_a})")

    buf.destroy()
    dist.destroy_process_group()


def launch(func):
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
        ("Test 1: upper_bound (do_cpu_sync=False)", run_test_upper_bound),
        ("Test 2: do_cpu_sync=True", run_test_do_cpu_sync),
    ]
    for name, func in tests:
        print("=" * 60)
        print(name)
        print("=" * 60)
        launch(func)
    print("All tests finished.")
