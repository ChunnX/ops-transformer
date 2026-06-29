# ElasticBuffer

## 产品支持情况

| 产品 | 是否支持 |
|------|:--------:|
| <term>Ascend 950DT</term> | √ |

## 功能说明

- **API功能**：

  ElasticBuffer 用于分布式 Engram 存储管理，支持将本 rank 的表分片写入 host pinned 共享段，以及通过 RDMA 从远端 rank 抓取 Engram 数据。需与 [get_engram_storage_size_hint]配套使用。

## ElasticBuffer 类接口原型

```python
class ElasticBuffer:
    def __init__(self, group: torch.distributed.ProcessGroup, num_cpu_bytes: int)
    
    def engram_write(self, storage: torch.Tensor) -> None
    
    def engram_fetch(self, indices: torch.Tensor) -> Callable[[], torch.Tensor]
    
    def destroy(self) -> None
    
    @staticmethod
    def get_engram_storage_size_hint(num_entries: int, hidden_size: int,
                                      dtype: torch.dtype = torch.bfloat16) -> int
```

## 成员函数说明

### __init__

**功能**：构造 ElasticBuffer，分配 host pinned 内存并初始化通信上下文。

**输入参数**：
- **group** (`torch.distributed.ProcessGroup`)：必选参数，分布式进程组，用于跨 rank 通信和同步。
- **num_cpu_bytes** (`int`)：必选参数，CPU buffer 大小（字节），用于 host pinned 存储区分配，必须 2MB 对齐且非负。

**输出**：无返回值，构造 ElasticBuffer 实例。

### engram_write

**功能**：将本 rank 的表分片写入 host pinned 共享段。

**输入参数**：
- **storage** (`Tensor`)：必选参数，待写入的 CPU tensor，shape 为 `(num_entries, hidden_size)`，表示有 `num_entries` 个条目，每个条目维度为 `hidden_size`。

**输出**：无返回值，数据写入 host pinned 内存。

### engram_fetch

**功能**：通过 RDMA 从远端 rank 抓取 Engram 数据，返回 callable 实现异步获取。

**输入参数**：
- **indices** (`Tensor`)：必选参数，查询索引的 NPU tensor，shape 为 `(num_tokens,)`，表示要抓取的条目全局索引。数据类型支持 `int32`，数据格式为 $ND$。元素取值范围需在 `[0, world_size × num_entries)`。

**输出**：
- **wait_callable** (`Callable[[], Tensor]`)：返回一个 callable，调用时阻塞至 RDMA 完成并返回 fetched tensor。

调用 `wait_callable()` 返回：
- **fetched** (`Tensor`)：NPU tensor，shape 为 `(num_tokens, hidden_size)`，数据类型与 `engram_write` 的 `storage.dtype` 相同，数据格式为 $ND$。

### destroy

**功能**：释放 ElasticBuffer 资源，包括 host pinned 内存。

**输入参数**：无参数。

**输出**：无返回值，资源释放完成。

### get_engram_storage_size_hint（静态方法）

**功能**：计算 CPU buffer 大小。

**输入参数**：
- **num_entries** (`int`)：必选参数，Engram storage 的条目数，必须非负。
- **hidden_size** (`int`)：必选参数，每个条目的隐藏维度，必须 128 数量对齐且大于0。
- **dtype** (`torch.dtype`)：可选参数，数据类型，默认为 `torch.bfloat16`。仅在此处用于按 dtype 计算字节数。

**输出**：
- **num_cpu_bytes** (`int`)：CPU buffer 大小（字节），已 2MB 对齐。

## 约束说明

- **参数对齐约束**：
  - `num_cpu_bytes` 必须为 2MB 对齐（即能被 `2 × 1024 × 1024` 整除）。
  - `hidden_size` 必须为 128 数量对齐。
  - `get_engram_storage_size_hint` 返回值自动满足 2MB 对齐。

- **维度约束**：
  - `storage` 必须为 2 维张量。
  - `indices` 必须为 1 维张量。

- **dtype 约束**：
  - `storage.dtype` 仅支持 `bfloat16`、`float16`、`float32`。
  - `indices.dtype` 必须为 `int32`。

- **设备约束**：
  - `storage` 必须在 CPU 上。
  - `indices` 必须在 NPU 上。

- **调用顺序约束**：
  - 必须先调用 `engram_write` 至少一次，才能调用 `engram_fetch`。
  - 同一 `ElasticBuffer` 实例上不允许并发 `engram_fetch`（需等待上次 fetch 的 callable 执行完成）。

- **数值约束**：
  - `num_cpu_bytes`、`num_entries`、`hidden_size` 必须非负。
  - `storage.nbytes()` 必须 ≤ `num_cpu_bytes`。
  - 全局条目总数 `world_size × num_entries` 必须 < 2^31（int32 最大值），保证 indices 索引不溢出。

- **通信域约束**：
  - `world_size` 范围 [2, 1024]，支持多卡分布式场景。

- **特殊场景处理**：
  - 支持 `num_entries = 0`。
  - 支持 `num_tokens = 0`。
  - 二进制一致：EngramWrite 和 EngramFetch 全程纯数据搬运，输出与源必须逐字节相等，无任何容差。

## 调用示例

- **单算子模式调用（多卡分布式）**

```python
import os
import torch
import torch_npu
import torch.distributed as dist
from torch.multiprocessing import Process, Manager
import torch.multiprocessing as mp
from cann_ops_transformer.ops import ElasticBuffer

num_entries = 10000
hidden_size = 4096
dtype = torch.bfloat16
world_size = 2

def set_device(rank):
    torch_npu.npu.set_device(rank)
    print(f"current device set: {torch_npu.npu.current_device()}")

def init_hccl_comm(rank, world_size):
    print(f'[INFO] device_{rank} 创建HCCL通信链路')
    master_ip = '127.0.0.1'
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size,
                             init_method=f'tcp://{master_ip}:50001')
    print(f"device_{rank} init_process_group success")
    
    # 创建全局通信域
    group = dist.new_group(backend="hccl", ranks=list(range(world_size)))
    return group

def run_elastic_buffer(queue, rank, world_size, storage, indices):
    print(f"{os.getpid()=}{rank=}")
    set_device(rank)
    group = init_hccl_comm(rank, world_size)
    
    # 计算存储大小
    num_cpu_bytes = ElasticBuffer.get_engram_storage_size_hint(
        num_entries, hidden_size, dtype
    )
    print(f"[INFO] device_{rank} num_cpu_bytes={num_cpu_bytes}")
    
    # 构造 ElasticBuffer
    buffer = ElasticBuffer(group, num_cpu_bytes)
    
    # engram_write（storage 已在 CPU）
    print(f"[INFO] device_{rank} 执行 engram_write")
    buffer.engram_write(storage)
    
    # engram_fetch（indices 移到 NPU）
    print(f"[INFO] device_{rank} 执行 engram_fetch")
    indices_npu = indices.npu()
    wait_callable = buffer.engram_fetch(indices_npu)
    fetched = wait_callable()
    
    # 同步并释放
    torch.npu.synchronize()
    print(f"[INFO] device_{rank} fetched shape: {fetched.shape}")
    buffer.destroy()
    
    print(f"[INFO] device_{rank} finish")
    dist.destroy_process_group()
    
    queue.put([rank, fetched.cpu()])

if __name__ == "__main__":
    # 构造测试数据
    storage = torch.randn(num_entries, hidden_size, dtype=dtype)  # CPU tensor
    indices = torch.randint(0, num_entries * world_size, (1000,), 
                           dtype=torch.int32)  # CPU tensor，后续移到 NPU
    
    # 多进程启动
    manager = Manager()
    result_queue = manager.Queue()
    mp.set_start_method("forkserver", force=True)
    
    proc_list = []
    for rank in range(world_size):
        proc = Process(target=run_elastic_buffer, 
                      args=(result_queue, rank, world_size, 
                            storage.clone(), indices.clone()))
        proc.start()
        proc_list.append(proc)
    
    # 等待结果
    results = [None] * world_size
    for proc in proc_list:
        rank_id, fetched = result_queue.get()
        results[rank_id] = fetched
        print(f"[INFO] rank_{rank_id} 结果已收集")
    
    for proc in proc_list:
        proc.join()
    
    # 检查结果
    if None not in results:
        print("All ranks finished successfully")
        for i, result in enumerate(results):
            print(f"Rank {i} fetched shape: {result.shape}")
    else:
        print("[ERROR] Task failed! Please check the detailed error logs.")
        exit(1)
```