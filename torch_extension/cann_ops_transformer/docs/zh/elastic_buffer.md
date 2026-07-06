# ElasticBuffer

## 产品支持情况

| 接口 | 产品 | 是否支持 |
|------|------|:--------:|
| Engram 存储接口 | <term>Ascend 950DT</term> | √ |
| Dispatch 接口 | <term>Ascend 950DT</term> | √ |
| Combine 接口 | <term>Ascend 950DT</term> | √ |

## 功能说明

- **API功能**：

  ElasticBuffer 提供统一的分布式通信 buffer 管理能力：

  - Engram 存储接口用于分布式 Engram 存储管理，支持将本 rank 的表分片写入 host pinned 共享段，以及通过 RDMA 从远端 rank 抓取 Engram 数据。需与 [get_engram_storage_size_hint](#get-engram-storage-size-hint) 配套使用。
  - Dispatch/Combine 接口用于 MoE 的 Expert Parallelism（EP）并行部署，支持通过 [dispatch](#dispatch) 将 token 数据分发到对应专家卡，再通过 [combine](#combine) 将专家输出按原路由聚合回原始序列。需与 [get_moe_ep_ccl_buffer_size](#get-moe-ep-ccl-buffer-size) 配套使用。

## ElasticBuffer 类接口原型

```python
class ElasticBuffer:
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        *,
        num_cpu_bytes: int = 0,
        num_max_tokens_per_rank: Optional[int] = None,
        hidden: Optional[int] = None,
        num_topk: Optional[int] = None,
        num_experts: Optional[int] = None,
        ccl_buffer_size: int = 0,
        expert_alignment: int = 1,
    )

    def engram_write(self, storage: torch.Tensor) -> None

    def engram_fetch(self, indices: torch.Tensor) -> Callable[[], torch.Tensor]

    @staticmethod
    def get_engram_storage_size_hint(
        num_entries: int,
        hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> int

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        handle: Optional[EPHandle] = None,
        num_experts: Optional[int] = None,
        num_max_tokens_per_rank: Optional[int] = None,
        expert_alignment: Optional[int] = None,
        do_cpu_sync: Optional[bool] = None,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
               Optional[torch.Tensor], Optional[torch.Tensor], EPHandle]

    def combine(
        self,
        x: torch.Tensor,
        handle: EPHandle,
        *,
        topk_weights: Optional[torch.Tensor] = None,
        bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]

    @staticmethod
    def get_moe_ep_ccl_buffer_size(
        world_size: int,
        num_max_tokens_per_rank: int,
        hidden: int,
        num_experts: int,
        topk: int,
    ) -> int

    def destroy(self) -> None
```

## 成员函数说明

### __init__<a name="init"></a>

**功能**：构造 ElasticBuffer 实例，记录 Engram 和 Dispatch/Combine 所需配置。Engram 运行时资源在首次调用 [engram_write](#engram-write) 时初始化；Dispatch/Combine 通信上下文在首次调用 [dispatch](#dispatch) 或 [combine](#combine) 时初始化。

**输入参数**：

- **group** (`torch.distributed.ProcessGroup`)：必选参数，分布式进程组，用于跨 rank 通信和同步。
- <strong>*</strong>：其之前的变量是位置相关的；之后的变量是可选参数，需要使用键值对赋值，不赋值会使用默认值。
- **num_cpu_bytes** (`int`)：可选参数，CPU buffer 大小（字节），用于 host pinned 存储区分配。默认值为 0；使用 [engram_write](#engram-write) 时必须大于 0，且必须 2MB 对齐。
- **num_max_tokens_per_rank** (`int`)：可选参数，表示每张卡上的最大 token 数量上限。使用 [dispatch](#dispatch) 和 [combine](#combine) 时必须与 `hidden`、`num_topk`、`num_experts` 一起指定。
- **hidden** (`int`)：可选参数，hidden size 隐藏层大小。
- **num_topk** (`int`)：可选参数，表示选取 topK 个专家。
- **num_experts** (`int`)：可选参数，MoE 专家总数量。
- **ccl_buffer_size** (`int`)：可选参数，CCL buffer size 初始值，默认值为 0。
- **expert_alignment** (`int`)：可选参数，专家对齐数，默认值为 1。

**输出**：无返回值，构造 ElasticBuffer 实例。

### engram_write<a name="engram-write"></a>

**功能**：将本 rank 的表分片写入 host pinned 共享段。首次调用时会懒初始化 Engram 运行时资源和通信上下文。

**输入参数**：

- **storage** (`Tensor`)：必选参数，待写入的 CPU tensor，shape 为 `(num_entries, hidden_size)`，表示有 `num_entries` 个条目，每个条目维度为 `hidden_size`。

**输出**：无返回值，数据写入 host pinned 内存。

### engram_fetch<a name="engram-fetch"></a>

**功能**：通过 RDMA 从远端 rank 抓取 Engram 数据，返回 callable 实现异步获取。

**输入参数**：

- **indices** (`Tensor`)：必选参数，查询索引的 NPU tensor，shape 为 `(num_tokens,)`，表示要抓取的条目全局索引。数据类型支持 `int32`，数据格式为 $ND$。元素取值范围需在 `[0, world_size × num_entries)`。

**输出**：

- **wait_callable** (`Callable[[], Tensor]`)：返回一个 callable，调用时阻塞至 RDMA 完成并返回 fetched tensor。

调用 `wait_callable()` 返回：

- **fetched** (`Tensor`)：NPU tensor，shape 为 `(num_tokens, hidden_size)`，数据类型与 [engram_write](#engram-write) 的 `storage.dtype` 相同，数据格式为 $ND$。

### get_engram_storage_size_hint（静态方法）<a name="get-engram-storage-size-hint"></a>

**功能**：计算 Engram CPU buffer 大小。

**函数原型**：

```python
ElasticBuffer.get_engram_storage_size_hint(num_entries, hidden_size, dtype=torch.bfloat16) -> int
```

**输入参数**：

- **num_entries** (`int`)：必选参数，Engram storage 的条目数，必须非负。
- **hidden_size** (`int`)：必选参数，每个条目的隐藏维度，必须 128 数量对齐且大于 0。
- **dtype** (`torch.dtype`)：可选参数，数据类型，默认为 `torch.bfloat16`。仅在此处用于按 dtype 计算字节数。

**输出**：

- **num_cpu_bytes** (`int`)：CPU buffer 大小（字节），已 2MB 对齐。

### dispatch<a name="dispatch"></a>

**功能**：需与 [combine](#combine) 配套使用，完成 MoE 的 Expert Parallelism（EP）并行部署下的 token dispatch。该接口根据每个 token 的 topK 专家索引，将 token 数据通过 EP 域的 alltoallv 通信分发到对应的专家卡上。

- 支持 cached 模式，即第二次 dispatch 时可复用第一次的 handle，跳过 slot 分配阶段，实现更低延迟。
- 支持指定 `num_max_tokens_per_rank` 控制接收 buffer 大小上限。

**计算公式**：

$$alltoall\_x\_out = alltoallv(x)$$

$$dst\_buffer\_slot\_idx = SlotAssignment(topk\_idx)$$

$$recv\_src\_metadata = MetadataAssignment(alltoall\_x\_out, topk\_idx)$$

**函数原型**：

```python
ElasticBuffer.dispatch(
    x,
    *,
    topk_idx=None,
    topk_weights=None,
    handle=None,
    num_experts=None,
    num_max_tokens_per_rank=None,
    expert_alignment=None,
    do_cpu_sync=None,
) -> (Tensor, Tensor, Tensor, EPHandle)
```

**输入参数**：

- **x** (`Tensor` 或 `Tuple[Tensor, Tensor]`)：必选参数，表示计算使用的 token 数据，需根据 `topk_idx` 来发送给其他卡。Tensor 要求为 2 维张量，shape 为 `(BS, H)`，数据类型支持 `bfloat16`、`float16`、`float8_e5m2`、`float8_e4m3fn`，数据格式为 $ND$。当传入 tuple 时，第一个 Tensor 为 token 数据，第二个 Tensor 为 scales。
- <strong>*</strong>：其之前的变量是位置相关的；之后的变量是可选参数，需要使用键值对赋值，不赋值会使用默认值。
- **topk_idx** (`Tensor`)：可选参数，表示每个 token 的 topK 个专家索引，决定每个 token 要发给哪些专家。要求为 2 维张量，shape 为 `(BS, K)`，数据类型支持 `int32`，数据格式为 $ND$。张量里 value 取值范围为 `[0, num_experts)`。非 cached 模式下为必选参数，cached 模式下必须为 `None`。
- **topk_weights** (`Tensor`)：可选参数，表示每个 token 对应的 topK 专家权重。要求为 2 维张量，shape 为 `(BS, K)`，数据类型支持 `float32`，数据格式为 $ND$。非 cached 模式下为可选参数，cached 模式下必须为 `None`。
- **handle** (`EPHandle`)：可选参数，表示上一次 dispatch 返回的 handle 对象，用于 cached 模式。传入 handle 时，`topk_idx` 和 `topk_weights` 必须为 `None`，`do_cpu_sync` 必须为 `False`。默认为 `None`，即非 cached 模式。
- **num_experts** (`int`)：可选参数，MoE 专家总数量，覆盖初始化时的值。取值范围 `[2, 2048]`，且满足 `num_experts % ep_world_size = 0`。默认使用初始化时传入的值。
- **num_max_tokens_per_rank** (`int`)：可选参数，表示每张卡上的最大 token 数量上限，覆盖初始化时的值。默认使用初始化时传入的值。
- **expert_alignment** (`int`)：可选参数，表示专家对齐数，覆盖初始化时的值。默认使用初始化时传入的值。
- **do_cpu_sync** (`bool`)：可选参数，表示是否进行 CPU 同步等待。非 cached 模式默认为 `True`，cached 模式必须为 `False`。

**输出说明**：

- **recv_x** (`Tensor` 或 `Tuple[Tensor, Tensor]`)：表示本卡收到的 token 数据。Tensor shape 为 `(A, H)`，数据类型与 `x` 一致，数据格式为 $ND$。经专家网络处理后，作为 [combine](#combine) 的 `x` 输入。当 `x` 输入包含 scales 时，输出为 `(recv_x, recv_scales)`。
- **recv_topk_idx** (`None`)：当前版本始终为 `None`，预留参数。
- **recv_topk_weights** (`Tensor | None`)：表示本卡收到的 topK 权重。仅当输入 `topk_weights` 不为 `None` 时返回，否则为 `None`。要求为 1 维张量，shape 为 `(A,)`，数据类型为 `float32`，数据格式为 $ND$，作为 [combine](#combine) 的 `topk_weights` 输入。
- **handle** (`EPHandle`)：表示 dispatch 阶段生成的 handle 对象，包含 slot 索引、元数据等信息，需传递给 [combine](#combine) 使用。handle 的属性如下：
  - **dst_buffer_slot_idx** (`Tensor`)：slot 索引，shape 为 `(BS, K)`，dtype 为 `int32`。
  - **recv_src_metadata** (`Tensor`)：接收元数据，shape 为 `(A, 4)`，dtype 为 `int32`。
  - **num_recv_tokens_per_rank** (`Tensor`)：各卡接收 token 数量，shape 为 `(ep_world_size,)`，dtype 为 `int32`。
  - **num_recv_tokens_per_expert** (`Tensor`)：每个本地专家接收的 token 数量，shape 为 `(num_local_experts,)`，dtype 为 `int64`。
  - **num_experts** (`int`)：专家总数量。
  - **expert_alignment** (`int`)：专家对齐数。
  - **num_max_tokens_per_rank** (`int`)：每张卡最大 token 数量上限。
  - **topk_idx** (`Tensor`)：原始 topK 索引。

### combine<a name="combine"></a>

**功能**：需与 [dispatch](#dispatch) 配套使用，相当于按 dispatch 算子收集数据的路径原路返回。该接口将专家处理后的 token 数据根据 dispatch 阶段记录的元数据信息，通过逆向路由和加权聚合，将 token 数据组合还原为原始序列顺序。

- 支持带 `topk_weights` 加权聚合和纯累加两种模式。
- 当前版本不支持 bias 参数。

**计算公式**：

当提供 `topk_weights` 时：

$$combined\_x_i = \sum_{k=0}^{K-1} topk\_weights_{i,k} \times x_{slot(i,k)}$$

当不提供 `topk_weights` 时（纯累加）：

$$combined\_x_i = \sum_{k=0}^{K-1} x_{slot(i,k)}$$

**函数原型**：

```python
ElasticBuffer.combine(x, handle, *, topk_weights=None, bias=None) -> (Tensor, Tensor?)
```

**输入参数**：

- **x** (`Tensor`)：必选参数，表示经过专家计算后的 token 数据，即 [dispatch](#dispatch) 输出的 `recv_x` 经过专家网络处理后的结果。要求为 2 维张量，shape 为 `(A, H)`，数据类型仅支持 `bfloat16`，数据格式为 $ND$。
- **handle** (`EPHandle`)：必选参数，表示 [dispatch](#dispatch) 返回的 handle 对象，包含 slot 索引、接收元数据等信息。handle 的属性参见 [dispatch](#dispatch) 输出说明。
- <strong>*</strong>：其之前的变量是位置相关的；之后的变量是可选参数，需要使用键值对赋值，不赋值会使用默认值。
- **topk_weights** (`Tensor`)：可选参数，表示每个 token 对应的 topK 专家权重，用于加权聚合。要求为 1 维张量，shape 为 `(A,)`，数据类型支持 `float32`，数据格式为 $ND$，对应 [dispatch](#dispatch) 的 `recv_topk_weights` 输出。若不提供，则进行纯累加 combine，输出 `combined_topk_weights` 为 `None`。
- **bias** (`Tensor` 或 `Tuple[Tensor, Tensor]`)：可选参数，当前版本不支持 bias，传入 `None` 即可。预留支持单个 bias 张量或 `bias_0`、`bias_1` 双张量模式。

**输出说明**：

- **combined_x** (`Tensor`)：表示 combine 后的 token 数据，还原为原始序列顺序。要求为 2 维张量，shape 为 `(BS, H)`，数据类型为 `bfloat16`，数据格式为 $ND$，不支持非连续的 Tensor。
- **combined_topk_weights** (`Tensor | None`)：表示 combine 后的 topK 专家权重。当 `topk_weights` 输入不为 `None` 时，要求为 2 维张量，shape 为 `(BS, K)`，数据类型为 `float32`，数据格式为 $ND$；当 `topk_weights` 输入为 `None` 时，返回 `None`。

### get_moe_ep_ccl_buffer_size（静态方法）<a name="get-moe-ep-ccl-buffer-size"></a>

**功能**：需与 [dispatch](#dispatch) 和 [combine](#combine) 配套使用，用于计算 dispatch 和 combine 算子所需的 HCCL 通信 `buffer_size` 大小（单位：MB）。该接口为静态方法，可在初始化 ElasticBuffer 前调用。

**函数原型**：

```python
ElasticBuffer.get_moe_ep_ccl_buffer_size(world_size, num_max_tokens_per_rank, hidden, num_experts, topk) -> int
```

**输入参数**：

- **world_size** (`int`)：必选参数，表示 EP 通信域的大小（即参与 EP 通信的卡数）。取值范围 `[2, 768]`。
- **num_max_tokens_per_rank** (`int`)：必选参数，表示每张卡上的最大 token 数量上限。
- **hidden** (`int`)：必选参数，表示 hidden size 隐藏层大小。取值范围 `[1024, 8192]`。
- **num_experts** (`int`)：必选参数，MoE 专家总数量，取值范围 `[1, 2048]`，且满足 `num_experts % ep_world_size = 0`。
- **topk** (`int`)：必选参数，表示选取 topK 个专家，取值范围 `[1, 32]`。

**输出说明**：

- **ccl_buffer_size** (`int`)：计算得到的 `ccl_buffer_size` 大小，单位为 MB。将该值设置为 `HCCL_BUFFSIZE` 环境变量即可满足通信域的内存需求。

**计算公式**：

```text
local_experts_num = num_experts // world_size
state_buffer_size =
    world_size * Align512(local_experts_num * 4)
    + 2 * world_size * 512
    + num_max_tokens_per_rank * topk * 512
    + world_size * 512

metadata_bytes = Align32(topk * 4)
hidden_align = Align32(hidden * 2)
dispatch_per_slot_bytes = Align512(hidden_align + metadata_bytes * 2 + 32)
combine_per_slot_bytes = Align512(hidden_align + 32)

minimum_buffer_size =
    state_buffer_size
    + world_size * num_max_tokens_per_rank * dispatch_per_slot_bytes
    + num_max_tokens_per_rank * topk * combine_per_slot_bytes

ccl_buffer_size = Align2(Align1MB(minimum_buffer_size) / 1MB) / 2
```

其中 `AlignX(value) = ((value + X - 1) / X) * X`，公式中的 `/` 表示整除。

### destroy<a name="destroy"></a>

**功能**：释放 ElasticBuffer 资源，包括 host pinned 内存、Engram 运行时资源和 Dispatch/Combine 通信上下文。

**输入参数**：无参数。

**输出**：无返回值，资源释放完成。

## 约束说明

- **参数对齐约束**：
  - `num_cpu_bytes` 必须为 2MB 对齐（即能被 `2 × 1024 × 1024` 整除）。
  - `hidden_size` 必须为 128 数量对齐。
  - [get_engram_storage_size_hint](#get-engram-storage-size-hint) 返回值自动满足 2MB 对齐。

- **Engram 维度约束**：
  - `storage` 必须为 2 维张量。
  - `indices` 必须为 1 维张量。

- **Engram dtype 约束**：
  - `storage.dtype` 仅支持 `bfloat16`、`float16`、`float32`。
  - `indices.dtype` 必须为 `int32`。

- **Engram 设备约束**：
  - `storage` 必须在 CPU 上。
  - `indices` 必须在 NPU 上。

- **Engram 调用顺序约束**：
  - 必须先调用 [engram_write](#engram-write) 至少一次，才能调用 [engram_fetch](#engram-fetch)。
  - 同一 ElasticBuffer 实例上不允许并发 [engram_fetch](#engram-fetch)（需等待上次 fetch 的 callable 执行完成）。

- **Engram 数值约束**：
  - `num_cpu_bytes`、`num_entries`、`hidden_size` 必须非负。
  - 使用 [engram_write](#engram-write) 时，`num_cpu_bytes` 必须大于 0。
  - `storage.nbytes()` 必须小于等于 `num_cpu_bytes`。
  - 全局条目总数 `world_size × num_entries` 必须小于 2^31（int32 最大值），保证 indices 索引不溢出。

- **Dispatch/Combine 配套约束**：
  - [dispatch](#dispatch) 和 [combine](#combine) 必须配套使用。
  - 调用接口过程中使用的 `num_experts`、`num_max_tokens_per_rank` 参数取值所有卡需保持一致，且 [dispatch](#dispatch) 和 [combine](#combine) 对应参数也需保持一致。
  - 当前版本不支持 bias 参数，`bias` 必须传入 `None`。
  - cached 模式下，`topk_idx` 和 `topk_weights` 必须为 `None`，`do_cpu_sync` 必须为 `False`；非 cached 模式下，`topk_idx` 为必选参数，`topk_weights` 为可选参数。

- **Dispatch/Combine Shape 变量说明**：
  - `A`：表示本卡接收的最大 token 数量，`A = ep_world_size * num_max_tokens_per_rank * MIN(K, num_local_experts)`。
  - `H`：表示 hidden size 隐藏层大小。取值范围为 `(0, 8192]`。
  - `BS`：表示 batch sequence size，即本卡的 token 数量。
  - `K`：表示选取 topK 个专家，取值范围为 `1 ≤ K ≤ 32`。
  - `num_local_experts`：表示本卡专家数量，`num_local_experts = num_experts / ep_world_size`，应满足 `0 < num_local_experts * ep_world_size ≤ 2048`。

- **HCCL 通信域缓存区大小**：
  - 调用 [dispatch](#dispatch) 或 [combine](#combine) 前需检查 `HCCL_BUFFSIZE` 环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位 MB，不配置时默认为 200MB。
  - 通信域缓存区大小可通过调用 [get_moe_ep_ccl_buffer_size](#get-moe-ep-ccl-buffer-size) 计算。
  - 计算得到的 `ccl_buffer_size` 需通过环境变量 `HCCL_BUFFSIZE` 设置，每个通信域独占一组 `2 * HCCL_BUFFSIZE` 大小的内存。

- **通信域约束**：
  - Engram 通信域 `world_size` 范围 `[2, 1024]`，支持多卡分布式场景。
  - 一个模型中的 [dispatch](#dispatch) 和 [combine](#combine) 算子仅支持相同 EP 通信域，且该通信域中不允许有其他算子。

- **特殊场景处理**：
  - 支持 `num_entries = 0`。
  - 支持 `num_tokens = 0`。
  - 二进制一致：EngramWrite 和 EngramFetch 全程纯数据搬运，输出与源必须逐字节相等，无任何容差。

## 调用示例

### Engram 单算子模式调用（多卡分布式）

```python
import os
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Manager
from cann_ops_transformer.ops import ElasticBuffer

num_entries = 10000
hidden_size = 4096
dtype = torch.bfloat16
world_size = 2


def set_device(rank):
    torch_npu.npu.set_device(rank)
    print(f"current device set: {torch_npu.npu.current_device()}")


def init_hccl_comm(rank, world_size):
    print(f"[INFO] device_{rank} create HCCL communication link")
    master_ip = "127.0.0.1"
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://{master_ip}:50001",
    )
    print(f"device_{rank} init_process_group success")

    group = dist.new_group(backend="hccl", ranks=list(range(world_size)))
    return group


def run_elastic_buffer(queue, rank, world_size, storage, indices):
    print(f"{os.getpid()=}{rank=}")
    set_device(rank)
    group = init_hccl_comm(rank, world_size)

    num_cpu_bytes = ElasticBuffer.get_engram_storage_size_hint(
        num_entries, hidden_size, dtype
    )
    print(f"[INFO] device_{rank} num_cpu_bytes={num_cpu_bytes}")

    buffer = ElasticBuffer(group, num_cpu_bytes=num_cpu_bytes)

    print(f"[INFO] device_{rank} run engram_write")
    buffer.engram_write(storage)

    print(f"[INFO] device_{rank} run engram_fetch")
    indices_npu = indices.npu()
    wait_callable = buffer.engram_fetch(indices_npu)
    fetched = wait_callable()

    torch.npu.synchronize()
    print(f"[INFO] device_{rank} fetched shape: {fetched.shape}")
    buffer.destroy()
    dist.destroy_process_group()

    queue.put([rank, fetched.cpu()])


if __name__ == "__main__":
    storage = torch.randn(num_entries, hidden_size, dtype=dtype)
    indices = torch.randint(
        0,
        num_entries * world_size,
        (1000,),
        dtype=torch.int32,
    )

    manager = Manager()
    result_queue = manager.Queue()
    mp.set_start_method("forkserver", force=True)

    proc_list = []
    for rank in range(world_size):
        p = Process(
            target=run_elastic_buffer,
            args=(result_queue, rank, world_size, storage.clone(), indices.clone()),
        )
        p.start()
        proc_list.append(p)

    results = [None] * world_size
    for _ in range(world_size):
        rank_id, fetched = result_queue.get()
        results[rank_id] = fetched
        print(f"[INFO] rank_{rank_id} result collected")

    for p in proc_list:
        p.join()

    if all(result is not None for result in results):
        print("All ranks finished successfully")
        for rank, result in enumerate(results):
            print(f"Rank {rank} fetched shape: {result.shape}")
    else:
        print("[ERROR] Task failed. Please check the detailed error logs.")
        exit(1)
```

### Dispatch/Combine 单算子模式调用（多卡分布式）

```python
import os
import torch
import torch_npu
import torch.distributed as dist
from torch.multiprocessing import Process
from cann_ops_transformer.ops import ElasticBuffer

master_ip = "127.0.0.1"
world_size = 2
num_experts = world_size * 4
num_max_tokens_per_rank = 128
hidden = 4096
top_k = 4


def run_dispatch_combine(rank):
    torch_npu.npu.set_device(rank)
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://{master_ip}:50002",
    )

    group = dist.new_group(backend="hccl")
    buffer = ElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=top_k,
        num_experts=num_experts,
    )

    num_tokens = 64
    x = torch.randn(
        num_tokens,
        hidden,
        dtype=torch.bfloat16,
        device=f"npu:{rank}",
    )
    topk_idx = torch.randint(
        0,
        num_experts,
        (num_tokens, top_k),
        dtype=torch.int32,
        device=f"npu:{rank}",
    )
    topk_weights = torch.rand(
        num_tokens,
        top_k,
        dtype=torch.float32,
        device=f"npu:{rank}",
    )

    recv_x, _, recv_topk_weights, handle = buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
    )
    torch.npu.synchronize()

    expert_output = recv_x
    combined_x, combined_topk_weights = buffer.combine(
        expert_output,
        handle,
        topk_weights=recv_topk_weights,
    )

    torch.npu.synchronize()
    print(f"[rank {rank}] combined_x shape={combined_x.shape}, expected ({num_tokens}, {hidden})")
    assert combined_x.shape == (num_tokens, hidden)
    print(f"[rank {rank}] dispatch_combine PASS")

    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"
    os.environ["HCCL_BUFFSIZE"] = str(
        ElasticBuffer.get_moe_ep_ccl_buffer_size(
            world_size,
            num_max_tokens_per_rank,
            hidden,
            num_experts,
            top_k,
        )
    )

    processes = []
    for rank in range(world_size):
        p = Process(target=run_dispatch_combine, args=(rank,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All processes finished.")
```
