# dense\_lightning\_indexer\_softmax\_lse\_v2\_metadata

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明

- 接口功能：

  `dense_lightning_indexer_softmax_lse_metadata` 算子是 `dense_lightning_indexer_softmax_lse` 的前置算子，用于计算分核负载均衡信息。根据主算子的序列长度、排列格式、mask 模式等信息，计算各核的工作量分配方案，输出固定 shape (64,) 的 int32 张量，供主算子的 `metadata` 输入使用。

  主要计算过程为：

  1. 根据输入参数计算总的 work item 数量（BSND 场景为 `batch_size * max_seqlen_q`，TND 场景为 `cu_seqlens_q` 最后一个元素的值）。
  2. 按照可用 AIC 核数对 work item 进行均匀分配，使各核负载均衡。
  3. 将分配结果写入固定长度为 64 的 int32 张量的前 4 个字段，其余字段置 0。

  metadata 输出布局：
  - `metadata[0]`：前置核数（forecore_num），即多分配 1 个 work item 的核数。
  - `metadata[1]`：尾部核数（tail_core_num），即少分配 1 个 work item 的核数。
  - `metadata[2]`：前置核每核 work item 数（b_s1_per_core）。
  - `metadata[3]`：尾部核每核 work item 数（b_s1_per_tail_core）。
  - `metadata[4]~metadata[63]`：预留字段，固定为 0。

- 计算公式：

  设总 work item 数为 $N_{total}$，可用 AIC 核数为 $N_{cores}$：

  1. 当 $N_{total} < N_{cores}$ 时：

  $$
  forecore\_num = N_{total},\quad tail\_core\_num = 0
  $$

  $$
  b\_s1\_per\_core = 1,\quad b\_s1\_per\_tail\_core = 0
  $$

  2. 当 $N_{total} \geq N_{cores}$ 时，设 $base = \lfloor N_{total} / N_{cores} \rfloor$，$rem = N_{total} \bmod N_{cores}$：

  - 若 $rem = 0$：

  $$
  forecore\_num = N_{cores},\quad tail\_core\_num = 0
  $$

  $$
  b\_s1\_per\_core = base,\quad b\_s1\_per\_tail\_core = 0
  $$

  - 若 $rem \neq 0$：

  $$
  forecore\_num = rem,\quad tail\_core\_num = N_{cores} - rem
  $$

  $$
  b\_s1\_per\_core = base + 1,\quad b\_s1\_per\_tail\_core = base
  $$

  其中，BSND 场景下 $N_{total} = B \times S1$；TND 场景下 $N_{total} = cu\_seqlens\_q[B]$。

## 函数原型

```python
cann_ops_transformer.dense_lightning_indexer_softmax_lse_metadata(
    num_heads_q,
    num_heads_k,
    head_dim,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    layout_q=None,
    layout_k=None,
    mask_mode=None,
    cmp_ratio=None
) -> Tensor
```

## 参数说明

>**说明：**<br>
>
>- num_heads_q、num_heads_k、head_dim 参数含义：num_heads_q 表示 query 的多头数，num_heads_k 表示 key 的多头数，head_dim 表示注意力头的维度。num_heads_q 必须能被 num_heads_k 整除。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| num_heads_q | int | 必选 | query 的多头数，必须为正数且能被 num_heads_k 整除。 | int | - |
| num_heads_k | int | 必选 | key 的多头数，必须为正数。 | int | - |
| head_dim | int | 必选 | 注意力头的维度，必须为正数。当前仅支持 128。 | int | - |
| cu_seqlens_q | Tensor | 可选 | 当前 Batch 及前序 Batch 中 q 的有效 token 数的累加和。仅 layout_q 为 TND 场景下必传，第一个值固定为 0。数据格式为 ND。 | int32 | (B+1,) |
| cu_seqlens_k | Tensor | 可选 | 当前 Batch 及前序 Batch 中 k 的有效 token 数的累加和。仅 layout_k 为 TND 场景下必传，第一个值固定为 0。数据格式为 ND。 | int32 | (B+1,) |
| seqused_q | Tensor | 可选 | 不同 Batch 中 q 的实际使用长度。数据格式为 ND。 | int32 | (B,) |
| seqused_k | Tensor | 可选 | 不同 Batch 中 k 的实际使用长度。数据格式为 ND。 | int32 | (B,) |
| cmp_residual_k | Tensor | 可选 | 表示 k 压缩前 token 数量除以 cmp_ratio 的余数。需在 mask_mode=3 且 cmp_ratio≠1 时传入。数据格式为 ND。 | int32 | (B,) |
| batch_size | int | 可选 | batch 数量。BSND 场景下必传；TND 场景可填 None，自动从 cu_seqlens_q 推导。 | int | - |
| max_seqlen_q | int | 可选 | query 的最大序列长度。BSND 场景下必传；TND 场景可填 None。 | int | - |
| max_seqlen_k | int | 可选 | key 的最大序列长度。BSND 场景下必传；TND 场景可填 None。 | int | - |
| layout_q | str | 可选 | 用于标识输入 q 的数据排布格式，支持 BSND、TND，默认值为 BSND。 | string | - |
| layout_k | str | 可选 | 用于标识输入 k 的数据排布格式，支持 BSND、TND，默认值为 BSND。layout_q 与 layout_k 必须一致。 | string | - |
| mask_mode | int | 可选 | 表示 mask 的模式，0 代表 defaultMask 模式，3 代表 rightDownCausal 模式，默认值为 0。 | int | - |
| cmp_ratio | int | 可选 | 表示 k 的压缩倍数。取值范围 [1, 128]，默认值为 1，表示无压缩。 | int | - |

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| metadata | Tensor | 必选 | 分核负载均衡信息。前 4 个字段依次为前置核数、尾部核数、前置核每核 work item 数、尾部核每核 work item 数，其余字段为 0。不支持空 tensor。数据格式为 ND。 | int32 | (64,) |

## 约束说明

- 该接口支持推理和训练场景下使用。
- 该接口支持单算子模式和 aclgraph 图模式调用。
- 参数 num_heads_q 必须为正数且能被 num_heads_k 整除。
- 参数 num_heads_k 必须为正数。
- 参数 head_dim 当前仅支持 128。
- 参数 cmp_ratio 支持 1~128，其中 1 表示无压缩。
- 参数 mask_mode 支持 0（defaultMask 模式）和 3（rightDownCausal 模式）。mask_mode 的具体含义参见[sparse_mode 参数说明](../../../../docs/zh/context/sparse_mode参数说明.md)。
- 参数 layout_q 与 layout_k 必须一致。
- 参数 cu_seqlens_q、cu_seqlens_k 要求其值为当前 Batch 与前序 Batch 有效 token 数的累加值，后一个元素的值必须大于等于前一个元素的值。
- 参数 seqused_q、seqused_k 要求其值不大于各 Batch 的实际序列长度。
  - BSND 场景下：seqused_q ≤ max_seqlen_q，seqused_k ≤ max_seqlen_k。
  - TND 场景下：seqused_q ≤ cu_seqlens_q[i+1] - cu_seqlens_q[i]，seqused_k ≤ cu_seqlens_k[i+1] - cu_seqlens_k[i]。
- 参数 cmp_residual_k 需满足 cmp_residual_k[i] < cmp_ratio。
- 当 cmp_ratio > 1 且 mask_mode = 3 时，必须传入 cmp_residual_k；其余情况不需要传入。
- <term>Ascend 950PR/Ascend 950DT</term>：
  - 当 layout_q 为 BSND 时，必须传入 batch_size 和 max_seqlen_q；不支持传入 cu_seqlens_q。
  - 当 layout_k 为 BSND 时，必须传入 max_seqlen_k；不支持传入 cu_seqlens_k。
  - 当 layout_q 为 TND 时，必须传入 cu_seqlens_q 和 cu_seqlens_k；batch_size 和 max_seqlen_q 可填 None，自动从 cu_seqlens_q 推导。
  - 当 cmp_ratio > 1 且 mask_mode = 3 时，必须传入 cmp_residual_k。

## 确定性计算

- 默认支持确定性计算。

## 调用示例

- 单算子模式调用（BSND 场景）：

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import dense_lightning_indexer_softmax_lse_metadata

  B = 2
  S1 = 4096
  S2 = 1024
  N1 = 32
  N2 = 1
  D = 128

  # 执行 metadata 算子
  metadata = dense_lightning_indexer_softmax_lse_metadata(
      num_heads_q=N1,
      num_heads_k=N2,
      head_dim=D,
      batch_size=B,
      max_seqlen_q=S1,
      max_seqlen_k=S2,
      layout_q="BSND",
      layout_k="BSND",
      mask_mode=0,
      cmp_ratio=1
  )
  print(f"metadata shape: {metadata.shape}")  # (64,)
  print(f"metadata[0:4]: {metadata[:4].tolist()}")  # [forecore_num, tail_core_num, b_s1_per_core, b_s1_per_tail_core]
  ```

- 单算子模式调用（TND 场景）：

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import dense_lightning_indexer_softmax_lse_metadata

  N1 = 32
  N2 = 1
  D = 128

  cu_seqlens_q = torch.tensor([0, 4096, 8192], dtype=torch.int32).npu()
  cu_seqlens_k = torch.tensor([0, 1024, 2048], dtype=torch.int32).npu()

  # 执行 metadata 算子
  metadata = dense_lightning_indexer_softmax_lse_metadata(
      num_heads_q=N1,
      num_heads_k=N2,
      head_dim=D,
      cu_seqlens_q=cu_seqlens_q,
      cu_seqlens_k=cu_seqlens_k,
      layout_q="TND",
      layout_k="TND",
      mask_mode=0,
      cmp_ratio=1
  )
  print(f"metadata shape: {metadata.shape}")  # (64,)
  print(f"metadata[0:4]: {metadata[:4].tolist()}")
  ```

- 配合主算子调用：

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import (
      dense_lightning_indexer_softmax_lse_metadata,
      dense_lightning_indexer_softmax_lse,
  )

  B = 2
  S1 = 4096
  S2 = 1024
  N1 = 32
  N2 = 1
  D = 128

  q = torch.randn(B, S1, N1, D, dtype=torch.float16).npu()
  k = torch.randn(B, S2, N2, D, dtype=torch.float16).npu()
  w = torch.randn(B, S1, N1, dtype=torch.float32).npu()

  # 先执行 metadata 算子获取分核信息
  metadata = dense_lightning_indexer_softmax_lse_metadata(
      num_heads_q=N1,
      num_heads_k=N2,
      head_dim=D,
      batch_size=B,
      max_seqlen_q=S1,
      max_seqlen_k=S2,
      layout_q="BSND",
      layout_k="BSND",
      mask_mode=0,
      cmp_ratio=1
  )

  # 再执行主算子
  lse = dense_lightning_indexer_softmax_lse(
      q, k, w,
      metadata=metadata,
      layout_q="BSND",
      layout_k="BSND",
      mask_mode=0,
      cmp_ratio=1
  )
  print(f"lse shape: {lse.shape}")
  ```

- aclgraph 图模式调用：

  ```python
  import torch
  import torch_npu
  import torchair
  from cann_ops_transformer.ops import dense_lightning_indexer_softmax_lse_metadata

  B = 2
  S1 = 4096
  N1 = 32
  N2 = 1
  D = 128

  class MetadataNetwork(torch.nn.Module):
      def __init__(self):
          super(MetadataNetwork, self).__init__()

      def forward(self):
          return torch.ops.cann_ops_transformer.dense_lightning_indexer_softmax_lse_metadata(
              N1,
              N2,
              D,
              batch_size=B,
              max_seqlen_q=S1,
              max_seqlen_k=1024,
              layout_q="BSND",
              layout_k="BSND",
              mask_mode=0,
              cmp_ratio=1
          )

  from torchair.configs.compiler_config import CompilerConfig
  config = CompilerConfig()
  config.mode = "reduce-overhead"
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  torch._dynamo.reset()
  npu_mode = torch.compile(
      MetadataNetwork(), fullgraph=True, backend=npu_backend, dynamic=False
  )
  metadata = npu_mode()
  print(f"metadata shape: {metadata.shape}")
  ```
