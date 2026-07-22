# sparse_flash_mla_softmax_l1_norm / sparse_flash_mla_softmax_l1_norm_metadata

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明

- API功能：

    * `sparse_flash_mla_softmax_l1_norm`：计算`SparseFlashMla`注意力的Softmax L1Norm结果，支持Compressed Attention以及Sparse Compressed Attention场景。该接口为`dense_lightning_indexer_kl_loss_grad`反向算子的配套接口，输出可用于反向梯度计算。
    * `sparse_flash_mla_softmax_l1_norm_metadata`：接口用于生成任务切分metadata，完成多核负载均衡计算，供后续`sparse_flash_mla_softmax_l1_norm`算子使用。

- 计算公式：

    阶段一：根据是否为sparse场景，对输入key进行选择

    * 当为sparse场景时：

    $$
    selectedK\text{ }=\text{ }Gather \left( K,sparseIndices \left[ i \left]  \left) ,\text{ }0\text{ } < =i < \text{ }selectBlockCount\right. \right. \right. \right.
    $$

    * else:

    $$
    selectedK\text{ }=\text{ }K
    $$

    阶段二：计算P（SimpleSoftmax）

    $$
    P = SimpleSoftmax(Mask(Q \text{ }@\text{ } selectedK^{{T}} \cdot \text{ } scale), softmax\_lse)
    $$

    阶段三：计算Softmax L1Norm

    $$
    softmaxL1Norm = \frac{ReduceSum(P, dim=G)}{G}
    $$

    其中，$G$ 为group数（$G = N1 / N2$），$ReduceSum$ 在G维度（q head group维度）上对softmax概率$P$求和后取平均。

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
```python
cann_ops_transformer.sparse_flash_mla_softmax_l1_norm(
    q,
    k,
    softmax_lse,
    *,
    sparse_indices=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    topk_length=None,
    metadata=None,
    softmax_scale=1.0,
    max_seqlen_k=0,
    cmp_ratio=1,
    mask_mode=0,
    layout_q="BSND",
    layout_k="BSND"
) -> Tensor
```

## 参数说明

### sparse_flash_mla_softmax_l1_norm

- **q**（`Tensor`）：必选参数，对应公式中的$Q$。`layout_q`="BSND" 时 shape 为 `[B, S1, N1, D]`；`layout_q`="TND" 时 shape 为 `[T1, N1, D]`。B：支持泛化；S1：支持泛化；D：512；T1：B × S1。支持非连续，数据格式支持ND，数据类型支持`bfloat16`和`float16`。

- **k**（`Tensor`）：必选参数，对应公式中的$K$。`layout_k`="BSND" 时 shape 为 `[B, S2, N2, D]`；`layout_k`="TND" 时 shape 为 `[T2, N2, D]`。B：与 q 的 B 保持一致；S2：支持泛化；N2：1；D：512；T2：B × S2。支持非连续，数据格式支持ND，数据类型与 q 保持一致。

- **softmax_lse**（`Tensor`）：必选参数，注意力正向计算的输出 softmax_lse。`layout_q`="BSND" 时 shape 为 `[B, N2, S1, G]`；`layout_q`="TND" 时 shape 为 `[N2, T1, G]`。B：与 q 的 B 保持一致；N2：1；S1：与 q 的 S1 保持一致；G：N1/N2；T1：B × S1。支持非连续，数据格式支持ND，数据类型支持`float32`。

- **sparse_indices**（`Tensor`）：可选参数，稀疏场景下选择的 k 中权重较高的注意力索引。`layout_q`="BSND" 时 shape 为 `[B, S1, N2, K]`；`layout_q`="TND" 时 shape 为 `[T1, N2, K]`。B：与 q 的 B 保持一致；S1：与 q 的 S1 保持一致；N2：1；K：支持泛化；T1：B × S1。支持非连续，数据格式支持ND，数据类型支持`int32`。

- **cu_seqlens_q**（`Tensor`）：可选参数，每个 Batch 中 q 的有效 token 数的累加和形式。`layout_q`="TND" 时必传。shape 为 `[B+1]`，累加和与 T1 保持一致。支持非连续，数据格式支持ND，数据类型支持`int32`。

- **cu_seqlens_k**（`Tensor`）：可选参数，每个 Batch 中 k 的有效 token 数的累加和形式。`layout_k`="TND" 时必传。shape 为 `[B+1]`，累加和与 T2 保持一致。支持非连续，数据格式支持ND，数据类型支持`int32`。

- **seqused_q**（`Tensor`）：可选参数，表示不同 batch 中 q 实际参与运算的 token 数。shape 为 `[B]`。支持非连续，数据格式支持ND，数据类型支持`int32`。

- **seqused_k**（`Tensor`）：可选参数，表示不同 batch 中 k 实际参与运算的 token 数。shape 为 `[B]`。支持非连续，数据格式支持ND，数据类型支持`int32`。

- **cmp_residual_k**（`Tensor`）：可选参数，表示每个 batch S2 // cmpRatio 后的余数。shape 为 `[B]`。当 k 不为空且 mask_mode=3 时必须传入。支持非连续，数据格式支持ND，数据类型支持`int32`。

- **topk_length**（`Tensor`）：可选参数，表示每行 q 对应的 k 实际可选的 topk 长度。`layout_q`="BSND" 时 shape 为 `[B, S1, N2]`；`layout_q`="TND" 时 shape 为 `[T1, N2]`。当 mask_mode=0 且 sparse_indices 不为 None 时必须传入且必须为准确值。支持非连续，数据格式支持ND，数据类型支持`int32`。

- **metadata**（`Tensor`）：必选参数，表示 tiling 下沉的 aicpu 算子输出结果，由 `sparse_flash_mla_softmax_l1_norm_metadata` 算子生成。支持非连续，数据格式支持ND，数据类型支持`int32`。

- **softmax_scale**（`float`）：可选参数，代表缩放系数。数据类型支持`float`，默认值：1.0。

- **max_seqlen_k**（`int`）：可选参数，表示 k 的最大序列长度，TND dense 场景用于输出 shape 推导。数据类型支持`int`，默认值：0。

- **cmp_ratio**（`int`）：可选参数，代表压缩率，取值范围 1~128。数据类型支持`int`，默认值：1。

- **mask_mode**（`int`）：可选参数，q 和 k 计算的 mask 模式。模式 0 为不做 mask 操作；模式 3 为 rightDownCausal。数据类型支持`int`。

- **layout_q**（`str`）：可选参数，q 的数据排布格式。支持 "BSND"、"TND"，需与 layout_k 保持一致。数据类型支持`str`。

- **layout_k**（`str`）：可选参数，k 的数据排布格式。支持 "BSND"、"TND"，需与 layout_q 保持一致。数据类型支持`str`。

### sparse_flash_mla_softmax_l1_norm_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| num_heads_q | int | 必选 | 表示公式中$Q$的头数（即N1），当前支持[1, 128]。 | int32 | - |
| num_heads_k | int | 必选 | 表示公式中$K$的头数（即N2），当前仅支持1。 | int32 | - |
| head_dim | int | 必选 | 表示头的维度（即D），当前仅支持512。 | int32 | - |
| cu_seqlens_q | Tensor | 可选 | 代表每个Batch中，q的有效token数的累加和形式，当`layout_q`为TND时该参数必传，支持非连续，数据格式支持ND，累加和与T1保持一致。 | int32 | (B+1, ) |
| cu_seqlens_k | Tensor | 可选 | 代表每个Batch中，k的有效token数的累加和形式，当`layout_k`为TND时该参数必传，支持非连续，数据格式支持ND，累加和与T2保持一致。 | int32 | (B+1, ) |
| seqused_q | Tensor | 可选 | 表示不同batch中q实际参与运算的token数。 | int32 | (B, ) |
| seqused_k | Tensor | 可选 | 预留接口参数，表示不同batch中k实际参与运算的token数，当前kernel路径暂不使用。 | int32 | (B, ) |
| cmp_residual_k | Tensor | 可选 | 预留接口参数，表示不同batch中k的sequence length与cmpRatio相关的残差，当前kernel路径暂不使用。 | int32 | (B, ) |
| topk_length | Tensor | 可选 | 表示每行q对应的k实际可选的topk长度。 | int32 | (B, S1, N2)或(T1, N2) |
| batch_size | int | 可选 | 表示输入样本批量大小（即B），默认None（自动推导或由max_seqlen_q决定）。BSND场景必须为正数。 | int32 | - |
| max_seqlen_q | int | 可选 | 表示输入q的最大序列长度，默认None。BSND场景必须为正数。 | int32 | - |
| max_seqlen_k | int | 可选 | 表示输入k的最大序列长度，默认None。BSND场景必须为正数。 | int32 | - |
| topk | int | 可选 | 表示从k中筛选出的关键token个数，0表示无稀疏，默认None。 | int32 | - |
| cmp_ratio | int | 可选 | 代表压缩率，取值范围：1~128，默认None（内部转为1）。 | int32 | - |
| mask_mode | int | 可选 | 表示q和k计算的mask模式，0表示No mask，3表示rightDownCausal模式，默认None（内部转为0）。 | int32 | - |
| layout_q | str | 可选 | 表示q的数据排布格式，支持"BSND"、"TND"，默认None（内部转为BSND）。 | string | - |
| layout_k | str | 可选 | 表示k的数据排布格式，支持"BSND"、"TND"，默认None（内部转为BSND）。 | string | - |

## 返回值说明

### sparse_flash_mla_softmax_l1_norm

- **softmax_l1_norm**（`Tensor`）：表示 q 与 k 计算得出的 softmax L1Norm 结果，公式为 reduceG(softmax)/G。若存在 sparse_indices，则该输出不为空；其他场景下该参数输出为空。数据类型为`float32`。`layout_q`="BSND" 时 shape 为 `[B, S1, N2, S2]`；`layout_q`="TND" 时 shape 为 `[T1, N2, T2]`。

### sparse_flash_mla_softmax_l1_norm_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| metadata | Tensor | 必选 | 负载均衡结果输出，采用strided切分方式，每个核处理的seq索引按核间步进排列。数据格式为ND，不支持非连续的Tensor。 | int32 | (64, ) |

Metadata字段布局：

| 字段 | index | 说明 |
|:---|:---|:---|
| totalNum | 0 | 参与负载均衡的seq总数。 |
| formerCoreProcessNum | 1 | 常规核处理的seq数，即ceil(totalNum / totalCoreNum)。 |
| remainCoreProcessNum | 2 | 尾核处理的seq数，即floor(totalNum / totalCoreNum)。 |
| remainCoreNum | 3 | 尾核数目。 |
| totalCoreNum | 4 | 实际使用的AIC核数，取min(totalNum, aicCoreNum, 36)。 |

## 约束说明

- 该接口支持训练场景下使用。
- 该接口支持单算子模式和aclgraph模式。
- 参数 q、k 的数据类型必须保持一致。
- 入参为空的场景处理：q 为空 Tensor 时直接返回。
- **确定性计算**：该接口默认确定性实现。
- **Mask 模式支持**：

    | 模式 | 含义 | 备注 |
    | :--- | :--- | :--- |
    | 0 | 不做 mask 操作 | 支持 |
    | 3 | rightDownCausal | 支持 |

- **规格约束**：

    | 规格项 | 规格 | 规格说明 |
    | :--- | :--- | :--- |
    | B | 支持泛化 | - |
    | S1、S2 | 支持泛化 | 支持S1、S2不等长。 |
    | N1 | 1~128 | num_heads_q 必须能被 num_heads_k 整除。 |
    | N2 | 1 | 当前仅支持N2=1。 |
    | D | 512 | q、k 最后一维需保持一致。 |
    | layout_q/k | BSND / TND，必须一致 | - |
    | cmp_ratio | 1~128 | - |

  - metadata 必须传入，由 `sparse_flash_mla_softmax_l1_norm_metadata` 算子生成。
  - BSND 场景：必传 batch_size、max_seqlen_q 和 max_seqlen_k 参数。
  - TND 场景：必传 cu_seqlens_q 和 cu_seqlens_k 参数，batch_size 可为0（通过 cu_seqlens_q 推导）。

## 确定性计算

- 默认支持确定性计算

## 调用示例

- 单算子模式调用

    ```python
    import math
    import torch
    import torch_npu
    import cann_ops_transformer

    torch_npu.npu.set_device(0)

    S1 = 16
    S2 = 2048
    cmp_ratio = 128
    actual_seq_q = [S1]
    actual_seq_k = [S2]

    T1 = sum(actual_seq_q)
    T2 = sum(actual_seq_k)

    B = 1
    N1 = 128
    N2 = 1
    D = 512
    scaleValue = 1.0 / math.sqrt(D)
    dtype = torch.float16
    input_layout = "TND"

    q_shape = (T1, N1, D)
    key_shape = (T2, N2, D)
    lse_shape = (N2, T1, N1 // N2)
    softmax_l1_norm_shape = (T1, N2, T2)

    cu_seq_qlen = [0] + [sum(actual_seq_q[:x+1]) for x in range(len(actual_seq_q))]
    cu_seq_klen = [0] + [sum(actual_seq_k[:x+1]) for x in range(len(actual_seq_k))]

    q = (torch.rand(q_shape).to(dtype)) * 2
    key = (torch.rand(key_shape).to(dtype)) * 2
    lse = (torch.rand(lse_shape).to(torch.float32))

    cu_seq_qlen_tensor = torch.tensor(cu_seq_qlen).to(torch.int32).npu()
    cu_seq_klen_tensor = torch.tensor(cu_seq_klen).to(torch.int32).npu()

    cmp_residual_k = torch.zeros(B, dtype=torch.int32, device="npu")

    metadata = cann_ops_transformer.sparse_flash_mla_softmax_l1_norm_metadata(
        N1, N2, D,
        cu_seqlens_q=cu_seq_qlen_tensor,
        cu_seqlens_k=cu_seq_klen_tensor,
        cmp_residual_k=cmp_residual_k,
        max_seqlen_q=S1,
        max_seqlen_k=S2,
        topk=0,
        cmp_ratio=cmp_ratio,
        mask_mode=3,
        layout_q=input_layout,
        layout_k=input_layout,
    )

    softmax_l1_norm = cann_ops_transformer.sparse_flash_mla_softmax_l1_norm(
        q.npu(),
        key.npu(),
        lse.npu(),
        cu_seqlens_q=cu_seq_qlen_tensor,
        cu_seqlens_k=cu_seq_klen_tensor,
        cmp_residual_k=cmp_residual_k,
        metadata=metadata,
        scale_value=scaleValue,
        max_seqlen_k=S2,
        cmp_ratio=cmp_ratio,
        mask_mode=3,
        layout_q=input_layout,
        layout_k=input_layout,
    )

    torch_npu.npu.synchronize()
    assert softmax_l1_norm.shape == softmax_l1_norm_shape
    assert softmax_l1_norm.dtype == torch.float32
    assert torch.isfinite(softmax_l1_norm.float()).all().item()
    ```
