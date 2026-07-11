"""
EngramFetch 训练样例（单机 2 卡）

使用方法:
    torchrun --nproc_per_node=2 mc2/example_engram_train.py

或手动启动两个进程:
    NPU_VISIBLE_DEVICES=0 python mc2/example_engram_train.py --rank 0 --world_size 2 &
    NPU_VISIBLE_DEVICES=1 python mc2/example_engram_train.py --rank 1 --world_size 2 &

验证逻辑（对应设计文档示例）:
  前向: fetched[i] 应等于 storage[indices[i]]
  反向: grad=fetched 时, gradUnique[k] = count_k × local_embedding[uniqueLocalEntry[k]]
        count_k = 该 local index 被所有 rank 请求的总次数
"""

import os
import argparse
import torch
import torch_npu
import torch.distributed as dist
from collections import Counter
from cann_ops_transformer.ops.elastic_buffer import ElasticBuffer


def parse_args():
    parser = argparse.ArgumentParser(description="EngramFetch Training Example")
    parser.add_argument("--rank", type=int, default=0, help="当前 rank")
    parser.add_argument("--world_size", type=int, default=2, help="总 rank 数")
    parser.add_argument("--master_port", type=str, default="50029", help="master 端口")
    return parser.parse_args()


def main():
    args = parse_args()
    rank = args.rank
    world_size = args.world_size

    # ==================== 初始化 ====================
    torch_npu.npu.set_device(rank)
    dist.init_process_group(
        backend='hccl',
        rank=rank,
        world_size=world_size,
        init_method=f'tcp://127.0.0.1:{args.master_port}'
    )
    group = dist.new_group(backend='hccl', ranks=list(range(world_size)))
    device = f"npu:{rank}"

    # ==================== 参数设置 ====================
    num_entries_per_rank = 4        # 每 rank 4 行 embedding
    hidden_size = 128               # hidden 维度（必须 128 对齐）
    num_tokens = 3                  # 本 rank fetch 的 token 数
    storage_type = torch.bfloat16

    # ==================== 生成 embedding（各 rank 不同）====================
    # 全局 index 分配: rank r → [r*E, (r+1)*E)
    # 对应示例: rank0→[0,1,2,3], rank1→[4,5,6,7]
    torch.manual_seed(42 + rank)
    storage = torch.randn(num_entries_per_rank, hidden_size, dtype=storage_type)
    total_entries = num_entries_per_rank * world_size

    # ==================== 生成 indices（随机）====================
    torch.manual_seed(100 + rank)
    indices = torch.randint(0, total_entries, (num_tokens,), dtype=torch.int32)

    # 生成 golden: fetch_golden[i] = global_storage[indices[i]]
    global_storage = [None] * world_size
    dist.all_gather_object(global_storage, storage)
    full_storage = torch.cat(global_storage, dim=0)
    fetch_golden = full_storage[indices]

    print(f"[rank {rank}] indices={indices.tolist()}")
    print(f"[rank {rank}] storage shape={storage.shape}, fetch_golden shape={fetch_golden.shape}")

    # ==================== 创建 ElasticBuffer（with_grad=True）====================
    num_cpu_bytes = ElasticBuffer.get_engram_storage_size_hint(
        num_entries=num_entries_per_rank, hidden_size=hidden_size, dtype=storage_type)

    buffer = ElasticBuffer(group=group, num_cpu_bytes=num_cpu_bytes, with_grad=True)
    print(f"[rank {rank}] ElasticBuffer created (with_grad=True), num_cpu_bytes={num_cpu_bytes}")

    # ==================== write ====================
    buffer.engram_write(storage)
    dist.barrier()
    torch.npu.synchronize()
    print(f"[rank {rank}] engram_write completed")

    # ==================== 训练前向 ====================
    # engram_fetch(with_grad=True) → hook() → (fetched, fetch_ctx)
    # 内部: AllocateAndExchangeGm → aclnnEngramFetch(issue) → aclnnEngramFetchWait → FreeGm
    indices_npu = indices.to(device)
    hook = buffer.engram_fetch(indices_npu)
    fetched, fetch_ctx = hook()

    print(f"[rank {rank}] forward: fetched shape={fetched.shape}")
    print(f"[rank {rank}] forward: fetch_ctx.perm shape={fetch_ctx.perm.shape}, "
          f"sendCounts shape={fetch_ctx.sendCounts.shape}")

    # 验证前向: fetched[i] 应等于 full_storage[indices[i]]
    fetched_cpu = fetched.cpu()
    if torch.equal(fetched_cpu, fetch_golden):
        print(f"[rank {rank}] [SUCCESS] forward: fetched == storage[indices]")
    else:
        diff = (fetched_cpu.float() - fetch_golden.float()).abs()
        print(f"[rank {rank}] [FAIL] forward: max_diff={diff.max().item():.6f}")

    # ==================== 训练反向 ====================
    # 设 grad = fetched，验证 scatter_add 累加
    # gradUnique[k] = count_k × local_embedding[uniqueLocalEntry[k]]
    grad_fetched = fetched.clone()

    # all_gather 所有 rank 的 indices，用于验证累加计数
    all_indices = [torch.zeros_like(indices) for _ in range(world_size)]
    dist.all_gather(all_indices, indices_npu, group=group)
    all_indices_list = [t.cpu().tolist() for t in all_indices]

    grad_unique, unique_local_entry = buffer.engram_fetch_grad(grad_fetched, fetch_ctx)

    print(f"[rank {rank}] backward: grad_unique shape={grad_unique.shape} (fp32)")
    print(f"[rank {rank}] backward: unique_local_entry={unique_local_entry.cpu().tolist()}")

    # ==================== 验证反向 ====================
    # 统计本 rank 的每个 local index 被请求的次数
    my_local_indices = []
    for r in range(world_size):
        for idx in all_indices_list[r]:
            global_idx = int(idx)
            owner = global_idx // num_entries_per_rank
            if owner == rank:
                local_idx = global_idx % num_entries_per_rank
                my_local_indices.append(local_idx)

    counts = Counter(my_local_indices)
    print(f"[rank {rank}] backward: counts={dict(sorted(counts.items()))}")

    # 验证 gradUnique[k] = count × local_embedding[uniqueLocalEntry[k]]
    grad_unique_cpu = grad_unique.cpu()
    unique_le_cpu = unique_local_entry.cpu()
    all_pass = True
    for k in range(len(unique_le_cpu)):
        local_idx = int(unique_le_cpu[k].item())
        count = counts[local_idx]
        expected = storage[local_idx].float() * count
        actual = grad_unique_cpu[k]
        if torch.allclose(actual, expected, atol=1e-2, rtol=1e-2):
            print(f"[rank {rank}] [SUCCESS] backward[{k}]: "
                  f"localIdx={local_idx}, count={count}, "
                  f"gradUnique={actual.tolist()[:4]}... == {count}×emb[{local_idx}]")
        else:
            diff = (actual - expected).abs().max().item()
            print(f"[rank {rank}] [FAIL] backward[{k}]: "
                  f"localIdx={local_idx}, count={count}, max_diff={diff:.6f}")
            all_pass = False

    if all_pass:
        print(f"[rank {rank}] [SUCCESS] All backward verified!")
    else:
        print(f"[rank {rank}] [FAIL] Backward verification failed!")

    # ==================== 清理 ====================
    dist.barrier()
    torch.npu.synchronize()
    buffer.destroy()
    dist.destroy_process_group()
    print(f"[rank {rank}] cleanup done")


if __name__ == "__main__":
    main()
