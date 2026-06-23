# sparse_flash_mla / sparse_flash_mla_metadata

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | × |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

`sparse_flash_mla`是基于`torch_npu`的`cann_ops_transformer`扩展接口，用于调用`SparseFlashMla`算子完成共享KV（Key和Value使用同一份输入）的稀疏注意力计算。该接口支持以下三类计算模式：

- **SWA（Sliding Window Attention）**：仅使用`ori_kv`，对原始KV做滑动窗口注意力。
- **CFA（Compressed Flash Attention）**：同时使用`ori_kv`和`cmp_kv`，对原始KV窗口和连续压缩KV段共同做注意力。
- **SCFA（Sparse Compressed Flash Attention）**：同时使用`ori_kv`、`cmp_kv`和`cmp_sparse_indices`，对原始KV窗口和TopK选择出的压缩KV共同做注意力。

计算公式如下：

$$
O = \text{softmax}(Q \cdot \tilde{K}^{T} \cdot \text{softmax\_scale}) \cdot \tilde{V}
$$

其中$\tilde{K}=\tilde{V}$，由`ori_kv`的滑动窗口部分和`cmp_kv`的压缩部分共同组成，实际参与计算的KV范围由`cmp_ratio`、`ori_mask_mode`、`cmp_mask_mode`、`ori_win_left`、`ori_win_right`以及`cmp_sparse_indices`决定。

`sparse_flash_mla_metadata`是`SparseFlashMlaMetadata`的torch扩展接口，用于在主算子执行前生成metadata。metadata记录AICore/AIVCore的任务切分结果，主算子必须传入该metadata。典型调用流程如下：

1. 准备`q`、`ori_kv`、`cmp_kv`、序列长度、block table、sinks等输入。
2. 调用`sparse_flash_mla_metadata`生成`metadata`。
3. 调用`sparse_flash_mla`，将上一步得到的`metadata`传入主算子。

> [!NOTE]
>
> `cmp_residual_kv`同时是`sparse_flash_mla`和`sparse_flash_mla_metadata`的可选输入。该参数用于恢复压缩前KV长度：`ori_len_for_cmp_mask = cmp_len * cmp_ratio + cmp_residual_kv[b]`。

## 函数原型

```python
from cann_ops_transformer.ops import sparse_flash_mla

sparse_flash_mla(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    softmax_scale=1.0,
    cmp_ratio=1,
    ori_mask_mode=0,
    cmp_mask_mode=0,
    ori_win_left=-1,
    ori_win_right=-1,
    layout_q="BSND",
    layout_kv="BSND",
    topk_value_mode=1,
    return_softmax_lse=False
) -> (Tensor, Tensor)
```

```python
from cann_ops_transformer.ops import sparse_flash_mla_metadata

sparse_flash_mla_metadata(
    num_heads_q,
    num_heads_kv,
    head_dim,
    *,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_ori_kv=None,
    max_seqlen_cmp_kv=None,
    ori_topk=0,
    cmp_topk=0,
    cmp_ratio=1,
    ori_mask_mode=0,
    cmp_mask_mode=0,
    ori_win_left=-1,
    ori_win_right=-1,
    layout_q="BSND",
    layout_kv="BSND",
    has_ori_kv=True,
    has_cmp_kv=True
) -> Tensor
```

## 参数说明

### sparse_flash_mla

| 参数名 | 输入/输出 | 描述 | 数据类型 | 维度 |
| :--- | :--- | :--- | :--- | :--- |
| q | 必选输入 | Query输入。 | `float16`、`bfloat16` | `layout_q="BSND"`时为`[B, S1, N1, D]`；`layout_q="TND"`时为`[T1, N1, D]`。 |
| ori_kv | 可选输入 | 原始KV输入，Key和Value共享同一份数据。SWA/CFA/SCFA模式需要传入。 | 与`q`一致 | `layout_kv="BSND"`时为`[B, S2, N2, D]`；`layout_kv="TND"`时为`[T2, N2, D]`；`layout_kv="PA_BBND"`时为`[ori_block_num, ori_block_size, N2, D]`。 |
| cmp_kv | 可选输入 | 压缩KV输入，Key和Value共享同一份数据。CFA/SCFA模式需要传入，SWA模式不传。 | 与`q`一致 | `layout_kv="BSND"`时为`[B, S3, N2, D]`；`layout_kv="TND"`时为`[T3, N2, D]`；`layout_kv="PA_BBND"`时为`[cmp_block_num, cmp_block_size, N2, D]`。 |
| ori_sparse_indices | 可选输入 | 原始KV稀疏索引。当前主流程不支持，建议传`None`。 | `int32` | - |
| cmp_sparse_indices | 可选输入 | 压缩KV TopK索引。SCFA模式必须传入，SWA/CFA模式传`None`。无效位置填`-1`。 | `int32` | `layout_q="BSND"`时为`[B, S1, N2, K]`；`layout_q="TND"`时为`[T1, N2, K]`，`K`支持512或1024。 |
| ori_block_table | 可选输入 | PA场景下`ori_kv`的block映射表。 | `int32` | `layout_kv="PA_BBND"`时为`[B, ori_max_block_num_per_batch]`。 |
| cmp_block_table | 可选输入 | PA场景下`cmp_kv`的block映射表。 | `int32` | CFA/SCFA且`layout_kv="PA_BBND"`时为`[B, cmp_max_block_num_per_batch]`。 |
| cu_seqlens_q | 可选输入 | TND场景下q的前缀和序列长度。 | `int32` | `[B + 1]`。 |
| cu_seqlens_ori_kv | 可选输入 | TND场景下ori_kv的前缀和序列长度。 | `int32` | `[B + 1]`。 |
| cu_seqlens_cmp_kv | 可选输入 | TND场景下cmp_kv的前缀和序列长度。 | `int32` | `[B + 1]`。 |
| seqused_q | 可选输入 | BSND场景下每个batch实际参与计算的q长度。 | `int32` | `[B]`。 |
| seqused_ori_kv | 可选输入 | 每个batch实际参与计算的ori_kv长度。`layout_kv="PA_BBND"`时必须传入；`layout_kv="BSND"`时可作为每个batch的ori_kv有效长度覆盖值；`layout_kv="TND"`时使用`cu_seqlens_ori_kv`表达序列边界。 | `int32` | `[B]`。 |
| seqused_cmp_kv | 可选输入 | 每个batch实际参与计算的cmp_kv长度。显式传入时作为cmp逻辑有效长度，优先于layout推导，`layout_kv="BSND"`、`layout_kv="TND"`、`layout_kv="PA_BBND"`均可使用；未传时使用`cmp_kv` shape、`cu_seqlens_cmp_kv`或PA block table相关语义推导。 | `int32` | `[B]`。 |
| cmp_residual_kv | 可选输入 | 每个batch的压缩余数，用于按`cmp_len * cmp_ratio + residual`恢复cmp侧mask使用的压缩前长度。主算子和metadata均可传入；在CFA/SCFA、`cmp_ratio != 1`且`cmp_mask_mode = 3`场景必传，`layout_kv="BSND"`、`layout_kv="TND"`、`layout_kv="PA_BBND"`均可使用。 | `int32` | `[B]`。 |
| ori_topk_length | 可选输入 | 预留输入，当前版本不支持传入非空Tensor；请传`None`。 | `int32` | - |
| cmp_topk_length | 可选输入 | 预留输入，当前版本不支持传入非空Tensor；请传`None`，SCFA有效位置通过`cmp_sparse_indices`中的`-1`表达。 | `int32` | - |
| sinks | 可选输入 | 每个Query head的sink值，作为online softmax初始项。 | `float32` | `[N1]`。 |
| metadata | 必选输入 | `sparse_flash_mla_metadata`生成的任务切分结果。 | `int32` | `[1024]`。 |
| softmax_scale | 可选属性 | QK矩阵乘后的缩放系数，通常为`1 / sqrt(D)`。 | `float` | - |
| cmp_ratio | 可选属性 | 压缩率。A2/A3支持`1`、`4`、`128`，`0`为非法值；A5支持`1~128`。 | `int` | - |
| ori_mask_mode | 可选属性 | ori_kv侧mask模式。当前有效配置为`4`，表示band模式。 | `int` | - |
| cmp_mask_mode | 可选属性 | cmp_kv侧mask模式。CFA/SCFA中当前有效配置为`3`，表示rightDownCausal模式。 | `int` | - |
| ori_win_left | 可选属性 | ori_kv滑动窗口左侧长度。当前支持`127`。 | `int` | - |
| ori_win_right | 可选属性 | ori_kv滑动窗口右侧长度。当前支持`0`。 | `int` | - |
| layout_q | 可选属性 | q的数据排布格式。 | `str` | 支持`"BSND"`、`"TND"`。 |
| layout_kv | 可选属性 | ori_kv/cmp_kv的数据排布格式，默认值为`"BSND"`。 | `str` | 支持`"BSND"`、`"TND"`、`"PA_BBND"`。 |
| topk_value_mode | 可选属性 | TopK索引取值模式。当前支持`1`。 | `int` | - |
| return_softmax_lse | 可选属性 | 是否返回softmax_lse。 | `bool` | 默认`False`。 |

### sparse_flash_mla_metadata

| 参数名 | 输入/输出 | 描述 | 数据类型 | 维度 |
| :--- | :--- | :--- | :--- | :--- |
| num_heads_q | 必选属性 | Query head数。 | `int` | - |
| num_heads_kv | 必选属性 | KV head数。当前仅支持`1`。 | `int` | - |
| head_dim | 必选属性 | 每个注意力头的维度，当前支持`512`。 | `int` | - |
| cu_seqlens_q | 可选输入 | TND场景下q的前缀和序列长度。 | `int32` | `[B + 1]`。 |
| cu_seqlens_ori_kv | 可选输入 | TND场景下ori_kv的前缀和序列长度。 | `int32` | `[B + 1]`。 |
| cu_seqlens_cmp_kv | 可选输入 | TND场景下cmp_kv的前缀和序列长度。 | `int32` | `[B + 1]`。 |
| seqused_q | 可选输入 | BSND场景下每个batch实际参与计算的q长度。 | `int32` | `[B]`。 |
| seqused_ori_kv | 可选输入 | 每个batch实际参与计算的ori_kv长度。PA_BBND场景必须传入；BSND场景可作为每个batch的ori_kv有效长度覆盖值；TND场景使用`cu_seqlens_ori_kv`表达序列边界。 | `int32` | `[B]`。 |
| seqused_cmp_kv | 可选输入 | 每个batch实际参与计算的cmp_kv长度。显式传入时作为cmp逻辑有效长度，BSND、TND、PA_BBND场景均可使用。 | `int32` | `[B]`。 |
| cmp_residual_kv | 可选输入 | 每个batch的压缩余数，用于按`cmp_len * cmp_ratio + residual`恢复cmp侧mask使用的压缩前长度。 | `int32` | `[B]`。 |
| ori_topk_length | 可选输入 | 预留输入，当前版本不支持传入非空Tensor；请传`None`。 | `int32` | - |
| cmp_topk_length | 可选输入 | 预留输入，当前版本不支持传入非空Tensor；请传`None`。 | `int32` | - |
| batch_size | 可选属性 | batch大小。BSND且未传`seqused_q`时需要显式指定。 | `int` | - |
| max_seqlen_q | 可选属性 | 所有batch中q的最大有效长度。 | `int` | - |
| max_seqlen_ori_kv | 可选属性 | 所有batch中ori_kv的最大有效长度。 | `int` | - |
| max_seqlen_cmp_kv | 可选属性 | 所有batch中cmp_kv的最大有效长度。 | `int` | - |
| ori_topk | 可选属性 | 原始KV TopK长度。当前主流程不支持，传`0`。 | `int` | - |
| cmp_topk | 可选属性 | 压缩KV TopK长度。CFA/SWA传`0`；SCFA支持`512`或`1024`。 | `int` | - |
| cmp_ratio | 可选属性 | 压缩率。A2/A3支持`1`、`4`、`128`，`0`为非法值；A5支持`1~128`。 | `int` | - |
| ori_mask_mode | 可选属性 | ori_kv侧mask模式。当前有效配置为`4`。 | `int` | - |
| cmp_mask_mode | 可选属性 | cmp_kv侧mask模式。CFA/SCFA中当前有效配置为`3`。 | `int` | - |
| ori_win_left | 可选属性 | ori_kv滑动窗口左侧长度，当前支持`127`。 | `int` | - |
| ori_win_right | 可选属性 | ori_kv滑动窗口右侧长度，当前支持`0`。 | `int` | - |
| layout_q | 可选属性 | q的数据排布格式。 | `str` | 支持`"BSND"`、`"TND"`。 |
| layout_kv | 可选属性 | ori_kv/cmp_kv的数据排布格式，默认值为`"BSND"`。 | `str` | 支持`"BSND"`、`"TND"`、`"PA_BBND"`。 |
| has_ori_kv | 可选属性 | 主算子是否会传入`ori_kv`。 | `bool` | - |
| has_cmp_kv | 可选属性 | 主算子是否会传入`cmp_kv`。未传`cmp_kv`时按SWA处理；传入`cmp_kv`时保留cmp段语义。 | `bool` | - |
| metadata | 输出 | 任务切分metadata。 | `int32` | `[1024]`。 |

## 返回值说明

- **attention_out**：`sparse_flash_mla`的第一个输出，shape和`q`一致，dtype和`q`一致。
- **softmax_lse**：`sparse_flash_mla`的第二个输出。`return_softmax_lse=False`时返回FLOAT32标量占位Tensor；`return_softmax_lse=True`时返回FLOAT32的log-sum-exp结果。
- **metadata**：`sparse_flash_mla_metadata`的输出，shape固定为`[1024]`，dtype为`int32`。

## 约束说明

- 该接口支持推理场景。
- `q`、`ori_kv`、`cmp_kv`的数据类型必须一致，支持`float16`和`bfloat16`。
- `head_dim`当前支持`512`。
- `num_heads_kv`当前仅支持`1`；`num_heads_q / num_heads_kv`在A2/A3上必须为`[1,128]`范围内的2的幂，在A5上支持`[1,128]`范围内任意整数。
- `layout_q`支持`BSND`和`TND`；`layout_kv`支持`BSND`、`TND`和`PA_BBND`，默认值为`BSND`。
- `layout_q="TND"`时必须传入`cu_seqlens_q`。
- `layout_kv="TND"`时必须传入`cu_seqlens_ori_kv`；若同时存在`cmp_kv`，还必须传入`cu_seqlens_cmp_kv`。
- `layout_kv="PA_BBND"`时必须传入`seqused_ori_kv`和`ori_block_table`；CFA/SCFA还必须传入`cmp_block_table`。`layout_kv="BSND"`时可选传入`seqused_ori_kv`覆盖每个batch的ori_kv有效长度；`layout_kv="TND"`时使用`cu_seqlens_ori_kv`表达ori_kv序列边界。
- `seqused_cmp_kv`为所有`layout_kv`下的可选输入，显式传入时用于覆盖cmp侧逻辑有效长度。
- `cmp_sparse_indices`只在SCFA模式传入，最后一维`K`支持512或1024。
- `cmp_residual_kv`用于修正cmp侧mask语义，主算子和metadata均可传入；长度必须等于batch大小。
- `ori_kv`和`cmp_kv`允许存在行间padding类非连续内存，接口会通过aclnn获取stride信息传给底层算子。

## 调用示例

### SWA，BSND输入

```python
import math
import torch
import torch_npu
from cann_ops_transformer.ops import sparse_flash_mla
from cann_ops_transformer.ops import sparse_flash_mla_metadata

torch_npu.npu.set_device(0)

dtype = torch.bfloat16
B = 1
S1 = 16
S2 = 64
N1 = 64
N2 = 1
D = 512
cmp_ratio = 1  # SWA示例不传cmp_kv，cmp_ratio使用合法默认值1。

q = torch.randn(B, S1, N1, D, dtype=dtype, device="npu")
ori_kv = torch.randn(B, S2, N2, D, dtype=dtype, device="npu")
sinks = torch.zeros(N1, dtype=torch.float32, device="npu")

metadata = sparse_flash_mla_metadata(
    N1,
    N2,
    D,
    batch_size=B,
    max_seqlen_q=S1,
    max_seqlen_ori_kv=S2,
    ori_topk=0,
    cmp_topk=0,
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=0,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="BSND",
    layout_kv="BSND",
    has_ori_kv=True,
    has_cmp_kv=False,
)

attn_out, softmax_lse = sparse_flash_mla(
    q,
    ori_kv=ori_kv,
    sinks=sinks,
    metadata=metadata,
    softmax_scale=1.0 / math.sqrt(D),
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=0,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="BSND",
    layout_kv="BSND",
    return_softmax_lse=False,
)
torch_npu.npu.synchronize()
assert attn_out.shape == q.shape
assert attn_out.dtype == q.dtype
assert softmax_lse.shape == torch.Size([])
assert torch.isfinite(attn_out.float()).all().item()
```

### CFA，BSND输入

```python
import math
import torch
import torch_npu
from cann_ops_transformer.ops import sparse_flash_mla
from cann_ops_transformer.ops import sparse_flash_mla_metadata

torch_npu.npu.set_device(0)

dtype = torch.bfloat16
B = 1
S1 = 16
S2 = 64
S3 = 16
N1 = 64
N2 = 1
D = 512
cmp_ratio = 4

q = torch.randn(B, S1, N1, D, dtype=dtype, device="npu")
ori_kv = torch.randn(B, S2, N2, D, dtype=dtype, device="npu")
cmp_kv = torch.randn(B, S3, N2, D, dtype=dtype, device="npu")
sinks = torch.zeros(N1, dtype=torch.float32, device="npu")

metadata = sparse_flash_mla_metadata(
    N1,
    N2,
    D,
    batch_size=B,
    max_seqlen_q=S1,
    max_seqlen_ori_kv=S2,
    max_seqlen_cmp_kv=S3,
    ori_topk=0,
    cmp_topk=0,
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="BSND",
    layout_kv="BSND",
    has_ori_kv=True,
    has_cmp_kv=True,
)

attn_out, softmax_lse = sparse_flash_mla(
    q,
    ori_kv=ori_kv,
    cmp_kv=cmp_kv,
    sinks=sinks,
    metadata=metadata,
    softmax_scale=1.0 / math.sqrt(D),
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="BSND",
    layout_kv="BSND",
    return_softmax_lse=False,
)
torch_npu.npu.synchronize()
assert attn_out.shape == q.shape
assert attn_out.dtype == q.dtype
assert softmax_lse.shape == torch.Size([])
assert torch.isfinite(attn_out.float()).all().item()
```

### SCFA，TND输入并使能cmp_residual_kv

```python
import math
import torch
import torch_npu
from cann_ops_transformer.ops import sparse_flash_mla
from cann_ops_transformer.ops import sparse_flash_mla_metadata

torch_npu.npu.set_device(0)

dtype = torch.float16
B = 1
q_lens = [1]
ori_lens = [6]
cmp_lens = [1]
N1 = 64
N2 = 1
D = 512
K = 512
cmp_ratio = 4

cu_q = torch.tensor([0, 1], dtype=torch.int32, device="npu")
cu_ori = torch.tensor([0, 6], dtype=torch.int32, device="npu")
cu_cmp = torch.tensor([0, 1], dtype=torch.int32, device="npu")
cmp_residual_kv = torch.tensor([2], dtype=torch.int32, device="npu")

q = torch.randn(sum(q_lens), N1, D, dtype=dtype, device="npu")
ori_kv = torch.randn(sum(ori_lens), N2, D, dtype=dtype, device="npu")
cmp_kv = torch.randn(sum(cmp_lens), N2, D, dtype=dtype, device="npu")
sinks = torch.zeros(N1, dtype=torch.float32, device="npu")

cmp_sparse_indices = torch.full((sum(q_lens), N2, K), -1, dtype=torch.int32, device="npu")
cmp_sparse_indices[:, :, :1] = torch.arange(1, dtype=torch.int32, device="npu").view(1, 1, 1)

metadata = sparse_flash_mla_metadata(
    N1,
    N2,
    D,
    cu_seqlens_q=cu_q,
    cu_seqlens_ori_kv=cu_ori,
    cu_seqlens_cmp_kv=cu_cmp,
    max_seqlen_q=max(q_lens),
    max_seqlen_ori_kv=max(ori_lens),
    max_seqlen_cmp_kv=max(cmp_lens),
    ori_topk=0,
    cmp_topk=K,
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="TND",
    layout_kv="TND",
    has_ori_kv=True,
    has_cmp_kv=True,
)

attn_out, softmax_lse = sparse_flash_mla(
    q,
    ori_kv=ori_kv,
    cmp_kv=cmp_kv,
    cmp_sparse_indices=cmp_sparse_indices,
    cu_seqlens_q=cu_q,
    cu_seqlens_ori_kv=cu_ori,
    cu_seqlens_cmp_kv=cu_cmp,
    cmp_residual_kv=cmp_residual_kv,
    sinks=sinks,
    metadata=metadata,
    softmax_scale=1.0 / math.sqrt(D),
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="TND",
    layout_kv="TND",
    return_softmax_lse=False,
)
torch_npu.npu.synchronize()
assert attn_out.shape == q.shape
assert attn_out.dtype == q.dtype
assert softmax_lse.shape == torch.Size([])
assert torch.isfinite(attn_out.float()).all().item()
```

### CP切分示例，TND + SCFA，rank0切开第二个seq

下面示例用单进程顺序模拟两个CP rank，说明全局TND数据与每个rank入参之间的关系。假设全局有2个序列，`cmp_ratio=4`：

| 视角 | q范围 | ori_kv范围 | cmp_kv范围 | cu_seqlens_q | cu_seqlens_ori_kv | cu_seqlens_cmp_kv | cmp_residual_kv |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 全局 | seq0 `[0,16)`，seq1 `[0,18)` | seq0 `[0,16)`，seq1 `[0,18)` | seq0 `[0,4)`，seq1 `[0,4)` | `[0,16,34]` | `[0,16,34]` | `[0,4,8]` | `[0,2]` |
| rank0 | seq0 `[0,16)`，seq1 `[0,8)` | seq0 `[0,16)`，seq1 `[0,8)` | seq0 `[0,4)`，seq1 `[0,2)` | `[0,16,24]` | `[0,16,24]` | `[0,4,6]` | `[0,0]` |
| rank1 | seq1 `[8,18)` | seq1 `[0,18)` | seq1 `[0,4)` | `[0,10]` | `[0,18]` | `[0,4]` | `[2]` |

rank1虽然只计算seq1的`[8,18)`，但`ori_kv`和`cmp_kv`需要传到当前位置结束为止的前缀。此时`ori_prefix_len - q_len = 18 - 10 = 8`，kernel推导出的q起点正好是CP切分点。每个本地batch都需要满足`cmp_len * cmp_ratio + cmp_residual_kv[b] == ori_prefix_len`。

```python
import math
import torch
import torch_npu
from cann_ops_transformer.ops import sparse_flash_mla
from cann_ops_transformer.ops import sparse_flash_mla_metadata


torch_npu.npu.set_device(0)

dtype = torch.float16
cmp_ratio = 4
K = 512
N1 = 64
N2 = 1
D = 512

# 全局packed TND视角：seq0长度16，seq1长度18。
global_q_lens = [16, 18]
global_ori_lens = [16, 18]
global_cmp_lens = [4, 4]
global_cmp_residual = [0, 2]

q_global = torch.randn(sum(global_q_lens), N1, D, dtype=dtype, device="npu")
ori_global = torch.randn(sum(global_ori_lens), N2, D, dtype=dtype, device="npu")
cmp_global = torch.randn(sum(global_cmp_lens), N2, D, dtype=dtype, device="npu")
sinks = torch.zeros(N1, dtype=torch.float32, device="npu")


def make_cu(lengths):
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    return torch.tensor(cu, dtype=torch.int32, device="npu")


def make_cmp_sparse_indices(q_lens, ori_prefix_lens, cmp_lens):
    indices = torch.full((sum(q_lens), N2, K), -1, dtype=torch.int32, device="npu")
    q_base = 0
    for q_len, ori_prefix_len, cmp_len in zip(q_lens, ori_prefix_lens, cmp_lens):
        q_start = ori_prefix_len - q_len
        for row in range(q_len):
            q_pos = q_start + row
            cmp_end = min(cmp_len, (q_pos + 1) // cmp_ratio)
            if cmp_end > 0:
                indices[q_base + row, :, :cmp_end] = torch.arange(
                    cmp_end, dtype=torch.int32, device="npu"
                ).view(1, cmp_end)
        q_base += q_len
    return indices


def run_one_rank(name, q, ori_kv, cmp_kv, q_lens, ori_prefix_lens, cmp_lens, residuals):
    cu_q = make_cu(q_lens)
    cu_ori = make_cu(ori_prefix_lens)
    cu_cmp = make_cu(cmp_lens)
    cmp_residual_kv = torch.tensor(residuals, dtype=torch.int32, device="npu")
    cmp_sparse_indices = make_cmp_sparse_indices(q_lens, ori_prefix_lens, cmp_lens)

    metadata = sparse_flash_mla_metadata(
        N1,
        N2,
        D,
        cu_seqlens_q=cu_q,
        cu_seqlens_ori_kv=cu_ori,
        cu_seqlens_cmp_kv=cu_cmp,
        max_seqlen_q=max(q_lens),
        max_seqlen_ori_kv=max(ori_prefix_lens),
        max_seqlen_cmp_kv=max(cmp_lens),
        ori_topk=0,
        cmp_topk=K,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="TND",
        has_ori_kv=True,
        has_cmp_kv=True,
    )

    attn_out, softmax_lse = sparse_flash_mla(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        cmp_sparse_indices=cmp_sparse_indices,
        cu_seqlens_q=cu_q,
        cu_seqlens_ori_kv=cu_ori,
        cu_seqlens_cmp_kv=cu_cmp,
        cmp_residual_kv=cmp_residual_kv,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=1.0 / math.sqrt(D),
        cmp_ratio=cmp_ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="TND",
        return_softmax_lse=False,
    )
    torch_npu.npu.synchronize()
    assert attn_out.shape == q.shape, name
    assert attn_out.dtype == q.dtype, name
    assert softmax_lse.shape == torch.Size([]), name
    assert torch.isfinite(attn_out.float()).all().item(), name
    return attn_out


# rank0：包含完整seq0，并切到seq1前8个token。
rank0_q = torch.cat([q_global[0:16], q_global[16:24]], dim=0)
rank0_ori = torch.cat([ori_global[0:16], ori_global[16:24]], dim=0)
rank0_cmp = torch.cat([cmp_global[0:4], cmp_global[4:6]], dim=0)
rank0_out = run_one_rank(
    "rank0", rank0_q, rank0_ori, rank0_cmp,
    q_lens=[16, 8], ori_prefix_lens=[16, 8], cmp_lens=[4, 2], residuals=[0, 0]
)

# rank1：只算seq1后10个token，但ori_kv/cmp_kv传seq1到18为止的前缀。
rank1_q = q_global[24:34]
rank1_ori = ori_global[16:34]
rank1_cmp = cmp_global[4:8]
rank1_out = run_one_rank(
    "rank1", rank1_q, rank1_ori, rank1_cmp,
    q_lens=[10], ori_prefix_lens=[18], cmp_lens=[4], residuals=[2]
)

seq0_out = rank0_out[:16]
seq1_out = torch.cat([rank0_out[16:24], rank1_out], dim=0)
assert seq0_out.shape == (16, N1, D)
assert seq1_out.shape == (18, N1, D)
```
