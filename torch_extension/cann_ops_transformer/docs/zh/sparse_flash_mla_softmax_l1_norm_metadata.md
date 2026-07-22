# sparse\_flash\_mla\_softmax\_l1\_norm\_metadata

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明

- 接口功能：

  `sparse_flash_mla_softmax_l1_norm_metadata` 算子是 `sparse_flash_mla_softmax_l1_norm` 的前置算子，用于计算分核负载均衡信息。根据主算子的序列长度、排列格式、mask 模式、压缩比例等信息，计算各核的工作量分配方案，输出固定 shape (64,) 的 int32 张量，供主算子的 `metadata` 输入使用。

  主要计算过程为：

  1. 根据输入参数计算参与负载均衡的 seq 总数（BSND 场景为 `batch_size * max_seqlen_q`，TND 场景为 `cu_seqlens_q` 最后一个元素的值；当 `seqused_q` 存在时，按每个 batch 实际用到的 seq 累加）。
  2. 按照可用 AIC 核数（取 `min(totalNum, aicCoreNum, 36)`）对 seq 进行 strided 方式均匀分配，使各核负载均衡。
  3. 将分配结果写入固定长度为 64 的 int32 张量的前 5 个字段，其余字段置 0。

  metadata 输出布局：
  - `metadata[0]`：totalNum，参与负载均衡的 seq 总数。
  - `metadata[1]`：formerCoreProcessNum，常规核处理的 seq 数，即 `ceil(totalNum / totalCoreNum)`。
  - `metadata[2]`：remainCoreProcessNum，尾核处理的 seq 数，即 `floor(totalNum / totalCoreNum)`。
  - `metadata[3]`：remainCoreNum，尾核数目。当 totalNum 能被 totalCoreNum 整除时为 0，否则为 `totalCoreNum - (totalNum % totalCoreNum)`。
  - `metadata[4]`：totalCoreNum，实际使用的 AIC 核数，取 `min(totalNum, aicCoreNum, 36)`。
  - `metadata[5]~metadata[63]`：预留字段，固定为 0。

  采用 strided 切分方式，每个核处理的 seq 索引按核间步进 totalCoreNum 排列。例如 totalNum=72，totalCoreNum=36 时，核 0 处理 seq[0] 和 seq[36]，核 1 处理 seq[1] 和 seq[37]，以此类推。

- 计算公式：

  设总 seq 数为 $N_{total}$，可用 AIC 核数为 $N_{cores}$，实际使用核数 $N_{used} = \min(N_{total}, N_{cores}, 36)$：

  1. 每核处理量：

  $$
  former\_core\_process\_num = \lceil N_{total} / N_{used} \rceil
  $$

  $$
  remain\_core\_process\_num = \lfloor N_{total} / N_{used} \rfloor
  $$

  2. 尾核数目：

  - 若 $N_{total} \bmod N_{used} = 0$：

  $$
  remain\_core\_num = 0
  $$

  - 若 $N_{total} \bmod N_{used} \neq 0$：

  $$
  remain\_core\_num = N_{used} - (N_{total} \bmod N_{used})
  $$

  其中，BSND 场景下 $N_{total} = B \times S1$；TND 场景下 $N_{total} = cu\_seqlens\_q[B]$；当 `seqused_q` 存在时，$N_{total} = \sum_{i=0}^{B-1} seqused\_q[i]$。

## 函数原型

```python
cann_ops_transformer.sparse_flash_mla_softmax_l1_norm_metadata(
    num_heads_q,
    num_heads_k,
    head_dim,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    topk_length=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    topk=None,
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
| head_dim | int | 必选 | 注意力头的维度，必须为正数。 | int | - |
| cu_seqlens_q | Tensor | 可选 | 当前 Batch 及前序 Batch 中 q 的有效 token 数的累加和。仅 layout_q 为 TND 场景下必传，第一个值固定为 0。数据格式为 ND。 | int32 | (B+1,) |
| cu_seqlens_k | Tensor | 可选 | 当前 Batch 及前序 Batch 中 k 的有效 token 数的累加和。仅 layout_k 为 TND 场景下必传，第一个值固定为 0。数据格式为 ND。 | int32 | (B+1,) |
| seqused_q | Tensor | 可选 | 不同 Batch 中 q 的实际使用长度。当该入参存在时，seq 总数按每个 batch 实际用到的 seq 累加计算。数据格式为 ND。 | int32 | (B,) |
| seqused_k | Tensor | 可选 | 预留参数，不同 Batch 中 k 的实际使用长度。当前 kernel 路径暂不使用。数据格式为 ND。 | int32 | (B,) |
| cmp_residual_k | Tensor | 可选 | 预留参数，表示 k 的 sequence length 与 cmp_ratio 相关的残差。当前 kernel 路径暂不使用。数据格式为 ND。 | int32 | (B,) |
| topk_length | Tensor | 可选 | 每行 query 对应的 key 实际可选的 topk 长度。mask_mode=0 且存在稀疏索引时需要传，且必须为准确值。数据格式为 ND。 | int32 | (B,S1) 或 (T1,) |
| batch_size | int | 可选 | batch 数量。BSND 场景下必传；TND 场景可填 None，自动从 cu_seqlens_q 推导。 | int | - |
| max_seqlen_q | int | 可选 | query 的最大序列长度。BSND 场景下必传；TND 场景可填 None。 | int | - |
| max_seqlen_k | int | 可选 | key 的最大序列长度。BSND 场景下必传；TND 场景可填 None。 | int | - |
| topk | int | 可选 | 从 key 中筛选出的关键 token 个数，0 表示无稀疏。 | int | - |
| layout_q | str | 可选 | 用于标识输入 q 的数据排布格式，支持 BSND、TND，默认值为 BSND。 | string | - |
| layout_k | str | 可选 | 用于标识输入 k 的数据排布格式，支持 BSND、TND，默认值为 BSND。建议与 layout_q 保持一致。 | string | - |
| mask_mode | int | 可选 | 表示 mask 的模式，0 代表 No mask 模式，3 代表 rightDownCausal 模式，默认值为 0。 | int | - |
| cmp_ratio | int | 可选 | 表示 k 的压缩倍数。取值范围 [1, 128]，默认值为 1，表示无压缩。 | int | - |

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| metadata | Tensor | 必选 | 分核负载均衡信息。前 5 个字段依次为 totalNum、formerCoreProcessNum、remainCoreProcessNum、remainCoreNum、totalCoreNum，其余字段为 0。不支持空 tensor。数据格式为 ND。 | int32 | (64,) |

## 约束说明

- 该接口支持推理和训练场景下使用。
- 该接口支持单算子模式和 aclgraph 图模式调用。
- 参数 num_heads_q 必须为正数且能被 num_heads_k 整除。
- 参数 num_heads_k 必须为正数。
- 参数 head_dim 必须为正数。
- 参数 cmp_ratio 支持 1~128，其中 1 表示无压缩。
- 参数 mask_mode 支持 0（No mask 模式）和 3（rightDownCausal 模式）。mask_mode 的具体含义参见[sparse_mode 参数说明](../../../../docs/zh/context/sparse_mode参数说明.md)。
- 参数 layout_q 必须为 BSND 或 TND。layout_k 支持 BSND 和 TND，建议与 layout_q 保持一致。
- 参数 cu_seqlens_q、cu_seqlens_k 要求其值为当前 Batch 与前序 Batch 有效 token 数的累加值，后一个元素的值必须大于等于前一个元素的值。
- 参数 seqused_q 要求其值不大于各 Batch 的实际序列长度。
  - BSND 场景下：seqused_q ≤ max_seqlen_q。
  - TND 场景下：seqused_q ≤ cu_seqlens_q[i+1] - cu_seqlens_q[i]。
- 参数 topk 支持非负数，0 表示无稀疏。
- <term>Ascend 950PR/Ascend 950DT</term>：
  - 当 layout_q 为 BSND 时，必须传入 batch_size 和 max_seqlen_q；不支持传入 cu_seqlens_q。
  - 当 layout_k 为 BSND 时，必须传入 max_seqlen_k；不支持传入 cu_seqlens_k。
  - 当 layout_q 为 TND 时，必须传入 cu_seqlens_q 和 cu_seqlens_k；batch_size 和 max_seqlen_q 可填 None，自动从 cu_seqlens_q 推导。
  - 当 mask_mode=0 且存在稀疏索引时，必须传入 topk_length 且必须为准确值。

## 确定性计算

- 默认支持确定性计算。确定性计算配置不会改变其输出规则。

## 调用示例

- 单算子模式调用（BSND 场景）：

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import sparse_flash_mla_softmax_l1_norm_metadata

  B = 4
  S1 = 128
  S2 = 128
  N1 = 8
  N2 = 1
  D = 128

  seqused_q = torch.tensor([128, 100, 64, 32], dtype=torch.int32).npu()

  # 执行 metadata 算子
  metadata = sparse_flash_mla_softmax_l1_norm_metadata(
      num_heads_q=N1,
      num_heads_k=N2,
      head_dim=D,
      seqused_q=seqused_q,
      batch_size=B,
      max_seqlen_q=S1,
      max_seqlen_k=S2,
      topk=64,
      layout_q="BSND",
      layout_k="BSND",
      mask_mode=0,
      cmp_ratio=1
  )
  print(f"metadata shape: {metadata.shape}")  # (64,)
  print(f"metadata[0:5]: {metadata[:5].tolist()}")  # [totalNum, formerCoreProcessNum, remainCoreProcessNum, remainCoreNum, totalCoreNum]
  ```

- 单算子模式调用（TND 场景）：

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import sparse_flash_mla_softmax_l1_norm_metadata

  N1 = 8
  N2 = 1
  D = 128

  cu_seqlens_q = torch.tensor([0, 128, 228, 292, 324], dtype=torch.int32).npu()
  cu_seqlens_k = torch.tensor([0, 128, 228, 292, 324], dtype=torch.int32).npu()

  # 执行 metadata 算子
  metadata = sparse_flash_mla_softmax_l1_norm_metadata(
      num_heads_q=N1,
      num_heads_k=N2,
      head_dim=D,
      cu_seqlens_q=cu_seqlens_q,
      cu_seqlens_k=cu_seqlens_k,
      topk=64,
      layout_q="TND",
      layout_k="TND",
      mask_mode=3,
      cmp_ratio=4
  )
  print(f"metadata shape: {metadata.shape}")  # (64,)
  print(f"metadata[0:5]: {metadata[:5].tolist()}")
  ```

- aclgraph 图模式调用：

  ```python
  import torch
  import torch_npu
  import torchair
  from cann_ops_transformer.ops import sparse_flash_mla_softmax_l1_norm_metadata

  B = 2
  S1 = 128
  S2 = 128
  N1 = 8
  N2 = 1
  D = 128

  class MetadataNetwork(torch.nn.Module):
      def __init__(self):
          super(MetadataNetwork, self).__init__()

      def forward(self):
          return torch.ops.cann_ops_transformer.sparse_flash_mla_softmax_l1_norm_metadata(
              N1,
              N2,
              D,
              batch_size=B,
              max_seqlen_q=S1,
              max_seqlen_k=S2,
              topk=0,
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
