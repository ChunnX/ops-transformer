# 如何通过 deep_ep.py 调用 Dispatch / Combine：V2 与 V3 对比

本文说明业务代码如何使用 `MoeDistributeBuffer` 完成下面的 MoE 数据流：

```text
原始 token x
  -> Dispatch：按 top-k expert 将 token 分发到各 rank
  -> 本地 Expert / FFN 计算
  -> Combine：将 expert 输出发回原 rank，并按 top-k weight 合并
  -> 最终 token 输出
```

先明确适用范围：**当前 `torch_extension/cann_ops_transformer/ops/deep_ep.py` 的低时延接口实际调用 V3 算子，不提供一个让用户在 Python 参数中选择 V2/V3 的开关。**

- 当前路径：`MoeDistributeBuffer` 创建 context，最终进入 DispatchV3/CombineV3。
- V2 路径：传统 ACLNN 接口接收 `groupEp/groupTp`；当前 DeepEP Python 封装不直接暴露这条路径。
- 兼容例外：Ascend 950 非 CCU 场景调用旧 V2 ACLNN API 时，V2 host wrapper 会把 `groupEp` 转成 context，并在内部改走 V3。

## 1. V2 与 V3 调用流程总览

### 1.1 V2 传统流程

```text
用户/框架持有 groupEp、groupTp 字符串
  -> 每次调用 aclnnMoeDistributeDispatchV2(..., groupEp, groupTp, ...)
  -> inner MoeDistributeDispatchV2
  -> V2 kernel
  -> HcclContextHolder 解释框架下发的 HcclOpParam
  -> Expert / FFN
  -> 每次调用 aclnnMoeDistributeCombineV2(..., groupEp, groupTp, ...)
  -> inner MoeDistributeCombineV2
  -> V2 kernel
```

V2 的调用者显式传通信域名称，算子接口中没有 context Tensor：

- [DispatchV2 的 `group_ep/group_tp`](mc2/moe_distribute_dispatch_v2/op_host/moe_distribute_dispatch_v2_def.cpp#L98)
- [CombineV2 的 `group_ep/group_tp`](mc2/moe_distribute_combine_v2/op_host/moe_distribute_combine_v2_def.cpp#L138)

### 1.2 V3 / 当前 deep_ep.py 流程

```text
用户创建 EP ProcessGroup
  -> MoeDistributeBuffer(group)
  -> CommContextManager.create_context()，只创建/取得一次 context
  -> low_latency_dispatch() 自动传 context
  -> ACLNN DispatchV5 -> inner DispatchV3
  -> V3 kernel
  -> Mc2MoeContextHolder 解释 Mc2MoeContext
  -> Expert / FFN
  -> low_latency_combine() 复用同一 context
  -> ACLNN CombineV5 -> inner CombineV3
  -> V3 kernel
```

V3 的底层接口删除通信域字符串，改为显式 context Tensor：

- [DispatchV3 的 `context`](mc2/moe_distribute_dispatch_v3/op_host/moe_distribute_dispatch_v3_def.cpp#L23)
- [CombineV3 的 `context`](mc2/moe_distribute_combine_v3/op_host/moe_distribute_combine_v3_def.cpp#L23)

### 1.3 调用侧差异

| 调用环节 | V2 | V3 / deep_ep.py |
|---|---|---|
| Python 对外封装 | 当前 `deep_ep.py` 不直接提供 | `MoeDistributeBuffer` |
| 用户持有的通信对象 | `groupEp/groupTp` 名称 | PyTorch `ProcessGroup` 和 `MoeDistributeBuffer` |
| context 创建位置 | V2/HCCL 路径内部 | `MoeDistributeBuffer.__init__()` 中提前创建 |
| Dispatch 每次通信参数 | 传 `groupEp/groupTp` | `self.context` 自动传入，用户无须传 |
| Combine 每次通信参数 | 再传相同 `groupEp/groupTp` | 复用同一 `self.context` |
| buffer size | 不作为 V2 算子显式属性 | V3 必须有 `ccl_buffer_size`，由 buffer 对象保存 |
| kernel holder | `HcclContextHolder` | `Mc2MoeContextHolder` |
| Dispatch→Combine 辅助数据 | 必须成对传递 | 同样必须成对传递 |

无论 V2 还是 V3，业务数据流都没有改变：Dispatch 的辅助输出必须传给配套 Combine，Expert/FFN 位于二者之间。变化的是通信 context 的管理方式，而不是 MoE 的路由语义。

## 2. 当前 V3 的对外 Python 入口

`MoeDistributeBuffer` 已从 `cann_ops_transformer.ops` 导出：

- [导出位置](torch_extension/cann_ops_transformer/ops/__init__.py#L16)
- [类定义](torch_extension/cann_ops_transformer/ops/deep_ep.py#L453)

业务代码使用：

```python
from cann_ops_transformer.ops import MoeDistributeBuffer
```

两个主要方法是：

- [`low_latency_dispatch()`](torch_extension/cann_ops_transformer/ops/deep_ep.py#L617)
- [`low_latency_combine()`](torch_extension/cann_ops_transformer/ops/deep_ep.py#L652)

## 3. 当前 V3 的完整调用示例

下面展示最主要的数据流。代码假设 HCCL 分布式进程组已经初始化，并且每个进程已经设置对应 NPU device。

```python
import torch
import torch.distributed as dist
import torch_npu

from cann_ops_transformer.ops import MoeDistributeBuffer


# 本进程基本信息
rank = dist.get_rank()
world_size = dist.get_world_size()

# 创建 EP 通信域；示例中所有 rank 都属于同一个 EP group
ep_ranks = list(range(world_size))
ep_group = dist.new_group(ranks=ep_ranks, backend="hccl")

# 模型参数
num_tokens = 8
hidden_size = 7168
topk = 8
num_experts = 32
max_tokens_per_rank = 16

# 输入必须位于 NPU
x = torch.randn(
    num_tokens,
    hidden_size,
    dtype=torch.bfloat16,
    device="npu",
)
topk_idx = torch.randint(
    low=0,
    high=num_experts,
    size=(num_tokens, topk),
    dtype=torch.int32,
    device="npu",
)
topk_weights = torch.randn(
    num_tokens,
    topk,
    dtype=torch.float32,
    device="npu",
)

# 1. 可选：先计算通信 buffer 大小
ccl_buffer_size = MoeDistributeBuffer.get_low_latency_ccl_buffer_size(
    world_size=world_size,
    num_max_dispatch_tokens_per_rank=max_tokens_per_rank,
    hidden=hidden_size,
    num_moe_expert=num_experts,
    topk=topk,
)

# 2. 创建一次，后续 dispatch/combine 复用同一个 context
buffer = MoeDistributeBuffer(
    group=ep_group,
    ccl_buffer_size=ccl_buffer_size,
)

# V3 context 已经在 buffer 内部创建；业务代码一般不直接访问它
# 等价关系：buffer.context == CommContextManager(...).create_context() 的结果

# 3. Dispatch
(
    expand_x,
    dynamic_scales,
    assist_info_for_combine,
    expert_token_nums,
    ep_recv_counts,
    expand_scales,
) = buffer.low_latency_dispatch(
    x=x,
    topk_idx=topk_idx,
    num_experts=num_experts,
    quant_mode=0,
    num_max_dispatch_tokens_per_rank=max_tokens_per_rank,
)

# 4. Expert / FFN 计算
# 实际模型应根据 expert_token_nums 使用本 rank 上对应的 expert 权重。
# 这里仅用占位函数表达数据流。
expert_output = local_expert_ffn(
    expand_x,
    expert_token_nums=expert_token_nums,
    dynamic_scales=dynamic_scales,
)

# 5. Combine
output = buffer.low_latency_combine(
    x=expert_output,
    topk_idx=topk_idx,
    topk_weights=topk_weights,
    assist_info_for_combine=assist_info_for_combine,
    ep_send_counts=ep_recv_counts,
    num_experts=num_experts,
    expand_scales=expand_scales,
    num_max_dispatch_tokens_per_rank=max_tokens_per_rank,
)
```

其中 `local_expert_ffn()` 是示意函数，不是本仓库提供的 API。真实模型需要在 Dispatch 和 Combine 中间调用自己的 Expert/FFN 实现。

### 3.1 为什么示例里没有写 `context=`

上面的示例是 V3，但调用者看不到 `context=`，因为 `MoeDistributeBuffer` 做了两层封装。

第一层，在构造 `MoeDistributeBuffer` 时创建并保存 context：

```python
# deep_ep.py 内部逻辑，不需要用户重复编写
self._ctx_manager = CommContextManager(
    self.group_name,
    self.world_size,
    ...,
)
self.context = self._ctx_manager.create_context()
```

对应源码：

- [创建 `CommContextManager`](torch_extension/cann_ops_transformer/ops/deep_ep.py#L459)
- [保存 `self.context`](torch_extension/cann_ops_transformer/ops/deep_ep.py#L466)

第二层，调用 Dispatch 和 Combine 时自动传入这个 context：

```python
# buffer.low_latency_dispatch(...) 内部
torch.ops.cann_ops_transformer.npu_moe_distribute_dispatch(
    context=self.context,
    x=x,
    expert_ids=topk_idx,
    ...,
)

# buffer.low_latency_combine(...) 内部
torch.ops.cann_ops_transformer.npu_moe_distribute_combine(
    context=self.context,
    expand_x=expert_output,
    expert_ids=topk_idx,
    ...,
)
```

对应源码：

- [Dispatch 传 `context=self.context`](torch_extension/cann_ops_transformer/ops/deep_ep.py#L625)
- [Combine 传 `context=self.context`](torch_extension/cann_ops_transformer/ops/deep_ep.py#L664)

所以业务代码：

```python
buffer.low_latency_dispatch(...)
buffer.low_latency_combine(...)
```

实际等价于：

```text
创建一次 context
  -> DispatchV3(context)
  -> Expert / FFN
  -> CombineV3(同一个 context)
```

这正是 V3 的调用方式。V2 不会在算子输入中出现 `context` Tensor，而是在 ACLNN 接口中显式传 `groupEp/groupTp`。

## 4. V2/V3 对比：初始化分布式环境

在创建 `MoeDistributeBuffer` 前，需要完成：

```python
torch_npu.npu.set_device(local_rank)

dist.init_process_group(
    backend="hccl",
    rank=rank,
    world_size=world_size,
    init_method="tcp://master_ip:port",
)

ep_group = dist.new_group(
    ranks=list(range(world_size)),
    backend="hccl",
)
```

`MoeDistributeBuffer` 从 `ep_group` 自动取得：

- 当前 EP rank
- EP world size
- HCCL communicator name

源码位置：

- [取得 rank 和 world size](torch_extension/cann_ops_transformer/ops/deep_ep.py#L454)
- [取得 HCCL communicator name](torch_extension/cann_ops_transformer/ops/deep_ep.py#L458)

对比 V2：V2 ACLNN 调用者通常需要取得并在每次 Dispatch/Combine 调用中传入通信域名称 `groupEp`；V3 的 Python 调用者只把 ProcessGroup 交给 `MoeDistributeBuffer`，后者负责取得名称并创建 context。

## 5. V2/V3 对比：CCL buffer 大小

推荐根据模型形状预先计算：

```python
ccl_buffer_size = MoeDistributeBuffer.get_low_latency_ccl_buffer_size(
    world_size=world_size,
    num_max_dispatch_tokens_per_rank=max_tokens_per_rank,
    hidden=hidden_size,
    num_moe_expert=num_experts,
    topk=topk,
    num_shared_expert=0,
    num_shared_expert_ranks=0,
    comm_alg="",
)
```

实现位置：

- [`get_low_latency_ccl_buffer_size()`](torch_extension/cann_ops_transformer/ops/deep_ep.py#L469)

这里的 `num_max_dispatch_tokens_per_rank` 是**每个 rank 的最大原始 token 数**。`deep_ep.py` 调用底层算子时会把它乘以 `world_size`，生成底层使用的 `global_bs`：

- [Dispatch 计算 `global_bs`](torch_extension/cann_ops_transformer/ops/deep_ep.py#L642)
- [Combine 计算 `global_bs`](torch_extension/cann_ops_transformer/ops/deep_ep.py#L692)

因此调用者不要提前再乘一次 `world_size`。

V2 接口的 OpDef 没有显式 `ccl_buffer_size` 属性；V3 将其设为必选属性：

- [DispatchV3 `ccl_buffer_size`](mc2/moe_distribute_dispatch_v3/op_host/moe_distribute_dispatch_v3_def.cpp#L106)
- [CombineV3 `ccl_buffer_size`](mc2/moe_distribute_combine_v3/op_host/moe_distribute_combine_v3_def.cpp#L146)

## 6. V2/V3 对比：通信对象的生命周期

```python
buffer = MoeDistributeBuffer(
    group=ep_group,
    ccl_buffer_size=ccl_buffer_size,
)
```

初始化过程会创建 `CommContextManager`，然后生成 DispatchV3/CombineV3 使用的 context Tensor：

- [`MoeDistributeBuffer.__init__()`](torch_extension/cann_ops_transformer/ops/deep_ep.py#L453)
- [`CommContextManager`](torch_extension/cann_ops_transformer/ops/deep_ep.py#L459)
- [`create_context()`](torch_extension/cann_ops_transformer/ops/deep_ep.py#L466)

同一个 `buffer` 实例应被重复使用，不要在每一层或每次前向中重新创建。这样 Dispatch 和 Combine 才能复用同一个通信 context。

生命周期差异可以概括为：

```text
V2：长期保存 group name，每次算子调用再由底层取得 HCCL context
V3：初始化时取得/创建 context Tensor，后续每次算子直接复用
```

## 7. V2/V3 对比：调用 Dispatch

最简调用：

```python
(
    expand_x,
    dynamic_scales,
    assist_info_for_combine,
    expert_token_nums,
    ep_recv_counts,
    expand_scales,
) = buffer.low_latency_dispatch(
    x=x,
    topk_idx=topk_idx,
    num_experts=num_experts,
    num_max_dispatch_tokens_per_rank=max_tokens_per_rank,
)
```

### 7.1 主要输入

| 参数 | 含义 |
|---|---|
| `x` | 本 rank 的原始 token，通常形状为 `[BS, H]` |
| `topk_idx` | 每个 token 选择的 expert ID，通常为 `[BS, K]`、`int32` |
| `num_experts` | EP 通信域中的 MoE expert 总数 |
| `quant_mode` | Dispatch 通信量化模式，`0` 表示非量化 |
| `topk_weights` | 可选，某些场景需要把权重随 token 一起 dispatch |
| `num_max_dispatch_tokens_per_rank` | 每 rank 最大 token 数，用于容量和 buffer 规划 |

完整方法签名见 [`low_latency_dispatch()`](torch_extension/cann_ops_transformer/ops/deep_ep.py#L617)。

### 7.2 Dispatch 内部调用

该方法把初始化时保存的 context、rank、world size 和 buffer size自动补入底层 torch op：

- [`torch.ops.cann_ops_transformer.npu_moe_distribute_dispatch`](torch_extension/cann_ops_transformer/ops/deep_ep.py#L624)
- [`context=self.context`](torch_extension/cann_ops_transformer/ops/deep_ep.py#L625)
- [自动传入 EP world size/rank](torch_extension/cann_ops_transformer/ops/deep_ep.py#L628)

用户不需要直接调用 `torch.ops`，也不需要手工传 context。

V2 调用概念上则是：

```cpp
aclnnMoeDistributeDispatchV2GetWorkspaceSize(
    x,
    expertIds,
    /* ... */, 
    groupEp,
    epWorldSize,
    epRankId,
    moeExpertNum,
    groupTp,
    /* outputs and executor */
);
```

V2 的关键差异是 `groupEp/groupTp` 出现在每次 ACLNN 调用参数中；当前 V3 Python 接口把这部分封装成 `buffer.context`。

### 7.3 Dispatch 输出

| 输出 | 后续用途 |
|---|---|
| `expand_x` | 已按目标 expert/rank 排列的 token，送给本地 Expert/FFN |
| `dynamic_scales` | 动态量化 scale；非量化场景通常不参与 FFN |
| `assist_info_for_combine` | Combine 恢复 token 顺序需要的辅助信息，必须原样传回 |
| `expert_token_nums` | 本 rank 各 local expert 收到的 token 数，用于切分 `expand_x` |
| `ep_recv_counts` | 本 rank从各 EP rank 收到的 token 数；Combine 时作为 `ep_send_counts` |
| `expand_scales` | 随 dispatch 展开的权重/scale 信息，按场景传入 Combine |

注意：`deep_ep.py` 底层 torch op 实际产生 7 个输出，但公开方法不返回 `tp_recv_counts`：

- [底层解包 7 个输出](torch_extension/cann_ops_transformer/ops/deep_ep.py#L623)
- [公开返回 6 个输出](torch_extension/cann_ops_transformer/ops/deep_ep.py#L650)

## 8. V2/V3 共同流程：执行本地 Expert / FFN

Dispatch 完成后，`expand_x` 已经被发送到拥有目标 expert 的 rank，并按 expert 组织。典型逻辑是：

```python
offset = 0
expert_outputs = []

for local_expert_id, token_count in enumerate(expert_token_nums):
    expert_input = expand_x[offset : offset + token_count]
    expert_output = experts[local_expert_id](expert_input)
    expert_outputs.append(expert_output)
    offset += token_count

expert_output = torch.cat(expert_outputs, dim=0)
```

这只是概念代码。实际实现通常使用 grouped matmul、融合 FFN 或固定容量布局，不能直接假设 `expert_token_nums` 可以在 Python 中无代价地逐项读取。

关键要求是：传给 Combine 的 `x` 必须保持 Dispatch 约定的 expert/token 布局，不能随意重新排序；如果 FFN 改变了顺序，相应辅助信息也必须同步调整。

这一段不属于 context 切换：V2 和 V3 都是在 Dispatch 后执行本地 expert，再调用配套 Combine。V3 更换 context 管理方式不会改变 FFN 的数学计算。

## 9. V2/V3 对比：调用 Combine

```python
output = buffer.low_latency_combine(
    x=expert_output,
    topk_idx=topk_idx,
    topk_weights=topk_weights,
    assist_info_for_combine=assist_info_for_combine,
    ep_send_counts=ep_recv_counts,
    num_experts=num_experts,
    expand_scales=expand_scales,
    num_max_dispatch_tokens_per_rank=max_tokens_per_rank,
)
```

### 9.1 Dispatch 输出与 Combine 输入的对应关系

| Dispatch 产生的数据 | Combine 参数 | 要求 |
|---|---|---|
| FFN 处理后的 `expand_x` | `x` | token 布局必须与 dispatch 约定一致 |
| 原始 `topk_idx` | `topk_idx` | 与原始 token 路由一致 |
| 原始 top-k 权重 | `topk_weights` | 用于多个 expert 结果加权合并 |
| `assist_info_for_combine` | `assist_info_for_combine` | 原样传递，不应由业务逻辑解释或修改 |
| `ep_recv_counts` | `ep_send_counts` | 名字不同，但这就是成对传递关系 |
| `expand_scales` | `expand_scales` | 量化/权重相关场景按原样传递 |

代码中的必选检查：

- [`assist_info_for_combine` 不能为空](torch_extension/cann_ops_transformer/ops/deep_ep.py#L658)
- [`ep_send_counts` 不能为空](torch_extension/cann_ops_transformer/ops/deep_ep.py#L660)

Combine 随后调用底层 torch op，并再次使用同一个 context：

- [`torch.ops...npu_moe_distribute_combine`](torch_extension/cann_ops_transformer/ops/deep_ep.py#L663)
- [`context=self.context`](torch_extension/cann_ops_transformer/ops/deep_ep.py#L664)

V2 调用概念上需要再次传入与 Dispatch 相同的通信域：

```cpp
aclnnMoeDistributeCombineV2GetWorkspaceSize(
    expertOutput,
    expertIds,
    assistInfoForCombine,
    epSendCounts,
    expertScales,
    /* ... */,
    groupEp,
    epWorldSize,
    epRankId,
    moeExpertNum,
    groupTp,
    /* output and executor */
);
```

V3 不再重复传通信域名称，而是 `low_latency_dispatch()` 和 `low_latency_combine()` 都读取同一 `MoeDistributeBuffer.context`。

## 10. V2/V3 共同能力：非量化和动态量化示例

### 10.1 非量化

```python
dispatch_outputs = buffer.low_latency_dispatch(
    x=x,
    topk_idx=topk_idx,
    num_experts=num_experts,
    quant_mode=0,
    num_max_dispatch_tokens_per_rank=max_tokens_per_rank,
)
```

FFN 输入通常直接使用 `expand_x`。

### 10.2 动态量化

```python
dispatch_outputs = buffer.low_latency_dispatch(
    x=x,
    topk_idx=topk_idx,
    num_experts=num_experts,
    quant_mode=2,
    x_smooth_scale=x_smooth_scale,
    num_max_dispatch_tokens_per_rank=max_tokens_per_rank,
)
```

此时要结合 `dynamic_scales` 和 FFN 支持的量化格式处理 `expand_x`。不要只把量化后的 `expand_x` 强制转换回浮点就认为等价；具体解量化/量化融合方式应与所使用的 FFN 算子一致。

## 11. 常见错误

### 11.1 每次调用都重新创建 MoeDistributeBuffer

错误：

```python
def forward(...):
    buffer = MoeDistributeBuffer(ep_group)
```

推荐在模型或通信模块初始化阶段创建一次，并保存为成员：

```python
class MoeLayer(torch.nn.Module):
    def __init__(self, ep_group, ccl_buffer_size):
        super().__init__()
        self.distribute_buffer = MoeDistributeBuffer(ep_group, ccl_buffer_size)
```

### 11.2 把 `ep_recv_counts` 忘记传给 Combine

Dispatch 的 `ep_recv_counts` 对应 Combine 的 `ep_send_counts`。这是通信方向反转后的命名，不是两个无关参数。

### 11.3 修改 assist_info_for_combine

该 Tensor 是 Dispatch 和 Combine 之间的内部协议。业务代码应把它视为 opaque data，只负责保存和原样传递。

### 11.4 Dispatch 和 Combine 参数不一致

两次调用必须保持以下配置一致：

- `num_experts`
- `shared_expert_num`
- `shared_expert_rank_num`
- `zero_expert_num/copy_expert_num/const_expert_num`
- `num_max_dispatch_tokens_per_rank`
- `comm_alg`
- 相关量化配置

### 11.5 传错 max token 语义

`num_max_dispatch_tokens_per_rank` 是每 rank 最大 token 数。`deep_ep.py` 会自行乘 `world_size` 生成 `global_bs`，调用端不应重复乘。

## 12. 为什么 deep_ep.py 明确走 V3

业务代码虽然只调用：

```python
buffer.low_latency_dispatch(...)
buffer.low_latency_combine(...)
```

但 `MoeDistributeBuffer` 内部创建的 context 会被传给底层 DispatchV3/CombineV3。简化调用链如下：

```text
MoeDistributeBuffer.low_latency_dispatch/combine
  -> torch.ops.cann_ops_transformer.npu_moe_distribute_*
  -> C++ Torch Extension
  -> ACLNN DispatchV5/CombineV5
  -> inner DispatchV3/CombineV3
```

V2/V3 的 context 差异见 [dispatch_combine_v2_v3_context_diff.md](dispatch_combine_v2_v3_context_diff.md)。

### 12.1 V5 API 与 V3 算子的映射

这里的 V5 是 ACLNN API 版本，V3 是 GE 算子版本：

- [DispatchV5 转 inner DispatchV3](mc2/moe_distribute_dispatch_v3/op_api/aclnn_moe_distribute_dispatch_v5.cpp#L42)
- [CombineV5 转 inner CombineV3](mc2/moe_distribute_combine_v3/op_api/aclnn_moe_distribute_combine_v5.cpp#L42)

### 12.2 V2 兼容入口也可能内部转 V3

在 Ascend 950 且 `commAlg != "ccu"` 时，旧 V2 ACLNN wrapper 会：

1. 根据 `groupEp` 调用 `GetMc2ContextTensor()`；
2. 得到 `Mc2MoeContext` Tensor；
3. 转发给 inner V3。

源码：

- [DispatchV2 wrapper 的分支](mc2/moe_distribute_dispatch_v2/op_api/moe_distribute_dispatch_v2_base.cpp#L190)
- [Dispatch 从 group 创建 context](mc2/moe_distribute_dispatch_v2/op_api/moe_distribute_dispatch_v2_base.cpp#L201)
- [Dispatch 转 inner V3](mc2/moe_distribute_dispatch_v2/op_api/moe_distribute_dispatch_v2_base.cpp#L203)
- [CombineV2 wrapper 的分支](mc2/moe_distribute_combine_v2/op_api/moe_distribute_combine_v2_base.cpp#L187)
- [Combine 从 group 创建 context](mc2/moe_distribute_combine_v2/op_api/moe_distribute_combine_v2_base.cpp#L200)
- [Combine 转 inner V3](mc2/moe_distribute_combine_v2/op_api/moe_distribute_combine_v2_base.cpp#L202)

所以“调用了 V2 API”与“最终一定执行 V2 kernel”不是完全等价的；当前代码会根据平台和通信算法做兼容路由。

## 13. 官方仓库内参考

- [Low Latency Dispatch 接口文档](torch_extension/cann_ops_transformer/docs/zh/low_latency_dispatch.md#L129)
- [Low Latency Dispatch 完整示例](torch_extension/cann_ops_transformer/docs/zh/low_latency_dispatch.md#L553)
- [Low Latency Combine 接口文档](torch_extension/cann_ops_transformer/docs/zh/low_latency_combine.md#L45)
- [Low Latency Combine 完整示例](torch_extension/cann_ops_transformer/docs/zh/low_latency_combine.md#L450)
- [CCL buffer size 接口文档](torch_extension/cann_ops_transformer/docs/zh/get_low_latency_ccl_buffer_size.md#L51)
