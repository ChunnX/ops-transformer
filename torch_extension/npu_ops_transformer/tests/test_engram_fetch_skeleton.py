# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------

import torch
import torch_npu
import torch.distributed as dist
from torch.multiprocessing import Process
import torch.multiprocessing as mp
import sys
import os

# 设置正确的导入路径：项目根目录
# 从 tests 目录：../../../ = ops-transformer 根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
torch_ext_path = os.path.join(project_root, 'torch_extension')

# 确保优先导入项目代码，而不是系统安装的包
sys.path.insert(0, torch_ext_path)

from npu_ops_transformer.ops.elastic_buffer import ElasticBuffer


def run_engram_fetch_test(rank, world_size):
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size,
                            init_method=f'tcp://127.0.0.1:50007')
    group = dist.distributed_c10d._get_default_group()
    
    # 创建 ElasticBuffer
    num_cpu_bytes = ElasticBuffer.get_engram_storage_size_hint(
        num_entries=100, hidden=64, dtype=torch.bfloat16)
    
    eb = ElasticBuffer(group=group, num_cpu_bytes=num_cpu_bytes)
    # eb.__barrier(True)
# 测试 engram_write
    print(f"R{rank}: Testing engram_write...")
    
    # 创建测试数据（CPU tensor）- 使用小数据触发C++验证（32字节）
    test_data = torch.randn(1, 16, dtype=torch.bfloat16, device="cpu")
    print(f"R{rank}:   Created test data: shape={test_data.shape}, dtype={test_data.dtype}")
    print(f"R{rank}:   Values: {test_data[0].tolist()}")
    
    # Python端：打印十六进制字节（用于对比C++输出）
    # bfloat16 = 2 bytes per element, 1x16 = 32 bytes total
    test_bytes = test_data.view(torch.uint8).numpy().tobytes()
    print(f"R{rank}:   Python hex: {test_bytes.hex()}")
    
    # 写入到 host pinned memory
    eb.engram_write(test_data)
    print(f"R{rank}:   engram_write completed!")
    
    # 测试 engram_fetch 调用
    indices = torch.tensor([0, 1, 2, 3, 4,], dtype=torch.int32, device="npu")
    hidden_size = 64
    
    print(f"R{rank}: Testing engram_fetch...")
    wait_func = eb.engram_fetch(indices)
    
    # 调用 wait callback
    fetched = wait_func()
    
    print(f"R{rank}: engram_fetch completed!")
    print(f"R{rank}:   indices shape: {indices.shape}")
    print(f"R{rank}:   fetched shape: {fetched.shape}")
    print(f"R{rank}:   fetched dtype: {fetched.dtype}")
    
    # 验证shape正确
    assert fetched.shape == (5, 16), f"Expected shape (5, 16), got {fetched.shape}"
    assert fetched.dtype == torch.bfloat16, f"Expected dtype bfloat16, got {fetched.dtype}"
    
    print(f"R{rank}: All validations passed!")
    
    # 清理
    eb.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    print("=" * 60)
    print("Testing EngramFetch (Skeleton Version)")
    print("=" * 60)
    
    try:
        mp.set_start_method("forkserver", force=True)
    except RuntimeError:
        pass
    
    procs = [Process(target=run_engram_fetch_test, args=(r, 2)) for r in range(2)]
    for p in procs: p.start()
    for p in procs: p.join()
    
    if all(p.exitcode == 0 for p in procs):
        print("\nPASS - EngramFetch skeleton test passed!")
        print("  ✓ Python调用路径正确")
        print("  ✓ Callable返回机制正确")
        print("  ✓ Tensor shape验证通过")
        sys.exit(0)
    else:
        print("\nFAIL - EngramFetch test failed!")
        sys.exit(1)