"""
EngramFetch 训练样例（单机多卡）

使用方法:
    python mc2/example_engram_train.py

验证逻辑（对应设计文档示例）:
  前向: fetched[i] 应等于 storage[indices[i]]
  反向: grad=fetched 时, gradUnique[k] = count_k × local_embedding[uniqueLocalEntry[k]]
        count_k = 该 local index 被所有 rank 请求的总次数
"""

import os
import inspect
import torch
import torch_npu
import torch.distributed as dist
from torch.multiprocessing import Process
from collections import Counter
from cann_ops_transformer.ops.elastic_buffer import ElasticBuffer

# 诊断: 打印实际加载的模块路径和 __init__ 签名
import cann_ops_transformer.ops.elastic_buffer as _eb_mod
print(f"[DIAG] elastic_buffer loaded from: {_eb_mod.__file__}")
print(f"[DIAG] ElasticBuffer.__init__ params: {list(inspect.signature(ElasticBuffer.__init__).parameters.keys())}")

# ===================== 固定参数 =====================
WORLD_SIZE = 2
MASTER_IP = '127.0.0.1'
MASTER_PORT = 50029

num_entries_per_rank = 4
hidden_size = 128
num_tokens = 3
storage_type = torch.bfloat16


def set_device(rank):
    torch_npu.npu.set_device(rank)
    print(f"[INFO] rank: {rank}, device: {rank}")


def init_hccl_comm(rank, world_size):
    print(f"[INFO] device_{rank} 创建HCCL通信链路")
    dist.init_process_group(
        backend='hccl',
        rank=rank,
        world_size=world_size,
        init_method=f'tcp://{MASTER_IP}:{MASTER_PORT}'
    )
    print(f"[INFO] device_{rank} HCCL通信链路创建成功")
    global_group = dist.new_group(backend='hccl', ranks=list(range(world_size)))
    print(f"[INFO] device_{rank} 全局通信组创建成功")
    return global_group


def train_worker(rank):
    try:
        set_device(rank)
        group = init_hccl_comm(rank, WORLD_SIZE)
        device = f"npu:{rank}"

        # ==================== 生成 embedding（各 rank 不同）====================
        # 全局 index 分配: rank r → [r*E, (r+1)*E)
        torch.manual_seed(42 + rank)
        storage = torch.randn(num_entries_per_rank, hidden_size, dtype=storage_type)
        total_entries = num_entries_per_rank * WORLD_SIZE

        # ==================== 生成 indices（随机）====================
        torch.manual_seed(100 + rank)
        indices = torch.randint(0, total_entries, (num_tokens,), dtype=torch.int32)

        # 生成 golden: fetch_golden[i] = global_storage[indices[i]]
        global_storage = [None] * WORLD_SIZE
        dist.all_gather_object(global_storage, storage)
        full_storage = torch.cat(global_storage, dim=0)
        fetch_golden = full_storage[indices]

        print(f"[rank {rank}] indices={indices.tolist()}")
        print(f"[rank {rank}] storage shape={storage.shape}, fetch_golden shape={fetch_golden.shape}")

        # ==================== 创建 ElasticBuffer（with_grad=True）====================
        num_cpu_bytes = ElasticBuffer.get_engram_storage_size_hint(
            num_entries=num_entries_per_rank, hidden_size=hidden_size, dtype=storage_type)

        buffer = ElasticBuffer(group=group, num_cpu_bytes=num_cpu_bytes,
                               num_max_tokens_per_rank=num_tokens, with_grad=True)
        print(f"[rank {rank}] ElasticBuffer created (with_grad=True), num_cpu_bytes={num_cpu_bytes}")

        # ==================== write ====================
        buffer.engram_write(storage)
        dist.barrier()
        torch.npu.synchronize()
        print(f"[rank {rank}] engram_write completed")

        # ==================== 训练前向 ====================
        indices_npu = indices.to(device)
        fetched, fetch_ctx = buffer.engram_fetch(indices_npu)()

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
        grad_fetched = fetched.clone()

        # all_gather 所有 rank 的 indices，用于验证累加计数
        all_indices = [torch.zeros_like(indices) for _ in range(WORLD_SIZE)]
        dist.all_gather(all_indices, indices_npu, group=group)
        all_indices_list = [t.cpu().tolist() for t in all_indices]

        grad_unique, unique_local_entry = buffer.engram_fetch_grad(grad_fetched, fetch_ctx)

        print(f"[rank {rank}] backward: grad_unique shape={grad_unique.shape} (fp32)")
        print(f"[rank {rank}] backward: unique_local_entry={unique_local_entry.cpu().tolist()}")

        # ==================== 验证反向 ====================
        my_local_indices = []
        for r in range(WORLD_SIZE):
            for idx in all_indices_list[r]:
                global_idx = int(idx)
                owner = global_idx // num_entries_per_rank
                if owner == rank:
                    local_idx = global_idx % num_entries_per_rank
                    my_local_indices.append(local_idx)

        counts = Counter(my_local_indices)
        print(f"[rank {rank}] backward: counts={dict(sorted(counts.items()))}")

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
                      f"gradUnique={actual.tolist()[:4]}... == {count}x emb[{local_idx}]")
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

    except Exception as e:
        print(f"[ERROR] rank {rank} 执行失败: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        try:
            dist.barrier()
        except:
            pass
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except:
                pass
        torch.npu.empty_cache()
        print(f"[INFO] rank {rank} 资源清理完成", flush=True)


if __name__ == "__main__":
    print(f"[INFO] 开始 engram 训练测试, world_size={WORLD_SIZE}")
    torch.multiprocessing.set_start_method('forkserver', force=True)

    proc_list = []
    for rank in range(WORLD_SIZE):
        proc = Process(target=train_worker, args=(rank,))
        proc.start()
        proc_list.append(proc)

    for proc in proc_list:
        proc.join()

    print(f"\n{'='*60}")
    print("[INFO] 所有进程已完成")
    print(f"{'='*60}")
