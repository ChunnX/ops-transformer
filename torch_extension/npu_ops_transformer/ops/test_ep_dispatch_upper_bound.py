import os
import torch
import torch_npu
import torch.distributed as dist
from torch.multiprocessing import Process
from npu_ops_transformer.ops.elastic_buffer import MoEElasticBuffer, EPHandle

master_ip = '127.0.0.1'
world_size = 2


def run_test(rank):
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size,
                             init_method=f'tcp://{master_ip}:50003')

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

    num_tokens = 128
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=f"npu:{rank}")
    topk_idx = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=f"npu:{rank}")
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.float32, device=f"npu:{rank}")

    print(f"[rank {rank}] BEFORE dispatch: ccl_buffer_size={buf.ccl_buffer_size.value}, ep_world_size={buf.ep_world_size.value}")
    print(f"[rank {rank}] context first 10 ints={buf.context.cpu()[:10].tolist()}")
    recv_x, recv_topk_idx, recv_topk_weights, handle = buf.dispatch(
        x, topk_idx, topk_weights, do_cpu_sync=False)

    upper_a = world_size * num_max_tokens_per_rank * min(top_k, num_experts // world_size)
    torch.npu.synchronize()
    print(f"[rank {rank}] recv_x shape={recv_x.shape}, expected upper_a={upper_a}")
    print(f"[rank {rank}] ccl_buffer_size={buf.ccl_buffer_size.value}, ep_world_size={buf.ep_world_size.value}")
    print(f"[rank {rank}] context first 5 ints={buf.context.cpu()[:5].tolist()}")
    assert recv_x.shape[0] == upper_a, f"shape mismatch: {recv_x.shape[0]} vs {upper_a}"
    print(f"[rank {rank}] test_dispatch_forward_upper_bound: PASS")

    buf.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ['HCCL_WHITELIST_DISABLE'] = '1'
    processes = []
    for rank in range(world_size):
        p = Process(target=run_test, args=(rank,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All processes finished.")