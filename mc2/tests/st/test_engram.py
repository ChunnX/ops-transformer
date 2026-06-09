"""
Engram测试脚本 - 参考 DeepEP test_engram.py
https://github.com/deepseek-ai/DeepEP/blob/main/tests/elastic/test_engram.py

使用方法：
    python test_engram.py --num-processes 8 --num-entries 65536 --hidden 256 --num-tokens 4096
"""
import argparse
import torch
import torch.distributed as dist
import torch_npu


def test_engram(local_rank, num_local_ranks, args):
    """测试流程参考DeepEP"""
    rank = local_rank
    world_size = num_local_ranks
    device = f'npu:{rank}'
    
    print(f'[Rank {rank}] Testing engram: entries={args.num_entries}, hidden={args.hidden}, tokens={args.num_tokens}')
    
    try:
        # 1. 准备数据
        torch.manual_seed(42 + rank)
        local_storage = torch.randn(
            (args.num_entries, args.hidden),
            dtype=torch.bfloat16,
            device=device
        )
        
        global_storage = torch.empty(
            (world_size * args.num_entries, args.hidden),
            dtype=torch.bfloat16,
            device=device
        )
        dist.all_gather_into_tensor(global_storage, local_storage)
        
        # 2. 构造buffer
        from torch_npu.contrib import ElasticBuffer
        
        num_gpu_bytes, num_cpu_bytes = ElasticBuffer.get_engram_storage_size_hint(
            args.num_entries, args.hidden, args.num_tokens, torch.bfloat16
        )
        
        buffer = ElasticBuffer(
            group=dist.group.WORLD,
            num_bytes=num_gpu_bytes + num_cpu_bytes,
            num_cpu_bytes=num_cpu_bytes,
            explicitly_destroy=True,
            num_allocated_qps=args.num_qps,
            allow_hybrid_mode=args.allow_hybrid_mode,
            allow_multiple_reduction=False
        )
        
        # 3. engram_write
        buffer.engram_write(local_storage)
        
        # 4. engram_fetch
        indices = torch.randint(
            0, world_size * args.num_entries,
            (args.num_tokens,),
            dtype=torch.int,
            device=device
        )
        
        hook = buffer.engram_fetch(indices)  # issue
        fetched = hook()                      # wait
        
        # 5. 验证正确性
        ref_fetched = global_storage[indices]
        assert torch.equal(ref_fetched, fetched), \
            f'[Rank {rank}] Data mismatch: max diff {(ref_fetched - fetched).abs().max().item()}'
        
        print(f'[Rank {rank}] Test passed!')
        
        # 6. 清理
        buffer.destroy()
        
    except ImportError:
        print(f'[Rank {rank}] ElasticBuffer not available - waiting for torch_npu implementation')
    except Exception as e:
        print(f'[Rank {rank}] Test failed: {e}')
        raise


def main():
    parser = argparse.ArgumentParser(description='Test engram (DeepEP style)')
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-qps', type=int, default=8)
    parser.add_argument('--num-entries', type=int, default=65536)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--num-tokens', type=int, default=4096)
    parser.add_argument('--allow-hybrid-mode', type=int, default=0)
    
    args = parser.parse_args()
    
    # 初始化分布式
    dist.init_process_group(backend='hccl')
    
    # 运行测试
    if args.num_processes > 1:
        torch.multiprocessing.spawn(
            test_engram,
            args=(args.num_processes, args),
            nprocs=args.num_processes
        )
    else:
        test_engram(0, 1, args)
    
    dist.destroy_process_group()


if __name__ == '__main__':
    main()