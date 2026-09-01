# generic_block_sparse_attention

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- **接口功能**：

  `generic_block_sparse_attention`是`cann_ops_transformer`扩展torch接口，用于调用`GenericBlockSparseAttention`算子完成任意粒度块稀疏注意力计算。通过`sparse_block_idx`指定每个Q块选择的KV块，`sparse_block_count`指定每个Q块保留的KV块数量。

  `generic_block_sparse_attention_metadata`是主算子的前置metadata生成接口，用于生成AICPU侧对稀疏索引与序列布局的预处理结果。`metadata`必须由与主算子完全一致的参数生成，并传入`generic_block_sparse_attention`。当前场景必须传入该metadata。典型调用流程如下：

  1. 准备`query`、`key`、`value`、`sparse_block_idx`、`sparse_block_count`、`cu_seq_lengths_q`、`cu_seq_lengths_kv`、`block_table`等输入。
  2. 调用`generic_block_sparse_attention_metadata`生成`metadata`。
  3. 调用`generic_block_sparse_attention`，并传入`generic_block_sparse_attention_metadata`的计算结果。

- **计算公式**：

  稀疏块大小为$blockShapeX \times blockShapeY$。

  $$
  attentionOut = Softmax(scaleValue \cdot query \cdot key_{sparse}^{T} + atten\_mask) \cdot value_{sparse}
  $$

  当前仅支持`layout_q="TND"`、`layout_kv="PAGED_BBND"`。

> [!NOTE]
>
> TND中的N为query的headNum（记为Nq），PAGED_BBND中的N为key/value的headNum（记为Nkv）。GQA下Nq与Nkv可以不同。Tq表示所有batch query token数的累加和；totalQBlocks = Σ ceilDiv(qStorageLen_i, blockShapeX)。

## 函数原型

调用`generic_block_sparse_attention`接口前，请先调用前置接口`generic_block_sparse_attention_metadata`完成metadata生成。两次调用中所有对应的Tensor和属性须保持一致。

> [!NOTE]
>
> `*`之后的所有参数仅支持以关键字方式传入。
>
> 主算子原型中`metadata`默认值为`None`，但当前落地场景必须传入由`generic_block_sparse_attention_metadata`生成的有效`metadata`。
>
> 当前仅支持`layout_q="TND"`、`layout_kv="PAGED_BBND"`，因此`cu_seq_lengths_q`、`cu_seq_lengths_kv`及`block_table`虽在原型中为可选，但实际调用时必须传入。

```python
cann_ops_transformer.generic_block_sparse_attention_metadata(
    sparse_block_idx,
    sparse_block_count,
    max_q_seq_len,
    max_kv_seq_len,
    num_q_heads,
    num_kv_heads,
    head_dim,
    block_shape,
    *,
    cu_seq_lengths_q=None,
    cu_seq_lengths_kv=None,
    seqused_q=None,
    seqused_kv=None,
    is_packed_gqa=1,
    layout_q="TND",
    layout_kv="PAGED_BBND",
    mask_type=1,
    quant_type=0,
    softmax_precision=1,
    win_left=-1,
    win_right=-1,
) -> Tensor
```

```python
cann_ops_transformer.generic_block_sparse_attention(
    query,
    key,
    value,
    sparse_block_idx,
    sparse_block_count,
    block_shape,
    *,
    metadata=None,
    atten_mask=None,
    q_dequant_scale=None,
    k_dequant_scale=None,
    v_dequant_scale=None,
    p_quant_scale=None,
    cu_seq_lengths_q=None,
    cu_seq_lengths_kv=None,
    seqused_q=None,
    seqused_kv=None,
    block_table=None,
    is_packed_gqa=1,
    layout_q="TND",
    layout_kv="PAGED_BBND",
    scale_value=0.0,
    mask_type=1,
    quant_type=0,
    dst_type_max=0.0,
    softmax_precision=1,
    win_left=-1,
    win_right=-1,
    return_softmax_lse=False,
    attention_out_dtype=None,
) -> (Tensor, Tensor)
```

## 参数说明

### 基准信息说明

| 命名 | 含义 |
| :--- | :--- |
| B | 表示输入样本batch大小。 |
| Nq / Nkv | query的头数 / kv的头数。 |
| Tq | TND布局下query所有batch token数的累加和。 |
| D | 每个注意力头的维度。 |
| topK | `sparse_block_idx`最后一维，表示每个Q块最多选择的KV块数。 |
| totalQBlocks | Σ ceilDiv(qStorageLen_i, blockShapeX)，i为batch索引。 |
| numBlocks | Paged KV Cache物理页总数。 |
| blockSize | Paged KV Cache单页token数，须等于`block_shape[1]`。 |

### generic_block_sparse_attention_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| sparse_block_idx | Tensor | 必选 | 每个Q块选择的KV块索引 | int32 | ND | is_packed_gqa=1: [Nkv, totalQBlocks, topK] | × |
| sparse_block_count | Tensor | 必选 | 每个Q块实际保留的KV块数量 | int32 | ND | is_packed_gqa=1: [Nkv, totalQBlocks] | × |
| max_q_seq_len | int | 必选 | query序列长度上限，须与主算子query及长度类输入一致 | int32 | - | - | - |
| max_kv_seq_len | int | 必选 | kv序列长度上限，须与主算子key/value及长度类输入一致 | int32 | - | - | - |
| num_q_heads | int | 必选 | query head数，须与主算子query的Nq维一致 | int32 | - | - | - |
| num_kv_heads | int | 必选 | key/value head数，须与主算子key/value的Nkv维一致 | int32 | - | - | - |
| head_dim | int | 必选 | 每个注意力头的维度，须与主算子query的D维一致 | int32 | - | - | - |
| block_shape | list[int] | 必选 | 稀疏块形状 `[block_x, block_y]` | int64 | - | 长度为2 | - |
| cu_seq_lengths_q | Tensor | TND必选 | query序列长度前缀和，首元素须为0 | int64 | ND | (B+1,) | × |
| cu_seq_lengths_kv | Tensor | TND必选 | kv序列长度前缀和，首元素须为0 | int64 | ND | (B+1,) | × |
| seqused_q | Tensor | 可选 | 每个batch实际使用的query序列长度 | int32 | ND | (B,) | × |
| seqused_kv | Tensor | 可选 | 每个batch实际使用的kv序列长度 | int32 | ND | (B,) | × |
| is_packed_gqa | int | 可选 | 1表示同group共享稀疏pattern，默认1；当前仅支持1 | int32 | - | - | - |
| layout_q | string | 可选 | 默认`"TND"`，当前仅支持该值 | string | - | - | - |
| layout_kv | string | 可选 | 默认`"PAGED_BBND"`，当前仅支持该值 | string | - | - | - |
| mask_type | int | 可选 | 掩码类型，默认1；当前仅支持1（内置causal mask） | int32 | - | - | - |
| quant_type | int | 可选 | 量化类型，默认0（非量化）；Ascend 950可选5，取值1~4当前不支持 | int32 | - | - | - |
| softmax_precision | int | 可选 | Softmax精度模式，默认1；Ascend 950仅支持1；A2/A3上FP16支持0或1，BF16仅支持0 | int32 | - | - | - |
| win_left | int | 可选 | 滑窗左边界，默认-1（当前仅支持-1） | int32 | - | - | - |
| win_right | int | 可选 | 滑窗右边界，默认-1（当前仅支持-1） | int32 | - | - | - |
| metadata | Tensor | 输出 | 主算子使用的预处理metadata | int32 | ND | (1024,) | × |

### generic_block_sparse_attention

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| query | Tensor | 必选 | 公式中的query | float16/bfloat16/float8_e4m3fn | ND | TND: (Tq, Nq, D) | × |
| key | Tensor | 必选 | Paged KV Cache中的key | float16/bfloat16/float8_e4m3fn | ND | PAGED_BBND: (numBlocks, blockSize, Nkv, D) | √（dim0） |
| value | Tensor | 必选 | Paged KV Cache中的value | float16/bfloat16/float8_e4m3fn | ND | 与key一致 | √（dim0） |
| sparse_block_idx | Tensor | 必选 | 每个Q块选择的KV块索引 | int32 | ND | is_packed_gqa=1: [Nkv, totalQBlocks, topK] | × |
| sparse_block_count | Tensor | 必选 | 每个Q块实际保留的KV块数量 | int32 | ND | is_packed_gqa=1: [Nkv, totalQBlocks] | × |
| block_shape | list[int] | 必选 | 稀疏块形状 `[block_x, block_y]` | int64 | - | 长度为2 | - |
| metadata | Tensor | 必选 | 由`generic_block_sparse_attention_metadata`生成 | int32 | ND | (1024,) | × |
| atten_mask | Tensor | 可选 | 当前不支持，须为None；`mask_type=1`时使用算子内置causal mask | bool | ND | - | - |
| q_dequant_scale | Tensor | 可选 | query的反量化缩放因子；非量化场景为None | float32/float8_e8m0 | ND | 见「约束说明」 | × |
| k_dequant_scale | Tensor | 可选 | key的反量化缩放因子；非量化场景为None | float32/float8_e8m0 | ND | 见「约束说明」 | × |
| v_dequant_scale | Tensor | 可选 | value的反量化缩放因子；非量化场景为None | float32/float8_e8m0 | ND | 见「约束说明」 | × |
| p_quant_scale | Tensor | 可选 | 当前须为None | float32 | ND | - | × |
| cu_seq_lengths_q | Tensor | TND必选 | query序列长度前缀和，首元素须为0 | int64 | ND | (B+1,) | × |
| cu_seq_lengths_kv | Tensor | TND必选 | kv序列长度前缀和，首元素须为0 | int64 | ND | (B+1,) | × |
| seqused_q | Tensor | 可选 | 每个batch实际使用的query序列长度 | int32 | ND | (B,) | × |
| seqused_kv | Tensor | 可选 | 每个batch实际使用的kv序列长度 | int32 | ND | (B,) | × |
| block_table | Tensor | PAGED必选 | 物理页映射表 | int32 | ND | (B, maxNumBlocksPerBatch) | × |
| is_packed_gqa | int | 可选 | 1表示同group共享稀疏pattern，默认1；当前仅支持1 | int32 | - | - | - |
| layout_q | string | 可选 | 默认`"TND"`，当前仅支持该值 | string | - | - | - |
| layout_kv | string | 可选 | 默认`"PAGED_BBND"`，当前仅支持该值 | string | - | - | - |
| scale_value | float | 可选 | 注意力缩放系数；默认0.0时算子内回落为`D^-0.5` | float32 | - | - | - |
| mask_type | int | 可选 | 掩码类型，默认1；当前仅支持1（内置causal mask），须与metadata一致 | int32 | - | - | - |
| quant_type | int | 可选 | 量化类型，默认0（非量化）；Ascend 950可选5，取值1~4当前不支持；须与metadata一致 | int32 | - | - | - |
| dst_type_max | float | 可选 | 量化相关参数，默认0.0 | float32 | - | - | - |
| softmax_precision | int | 可选 | Softmax精度模式，默认1；须与metadata一致；Ascend 950仅支持1 | int32 | - | - | - |
| win_left | int | 可选 | 滑窗左边界，默认-1（当前仅支持-1） | int32 | - | - | - |
| win_right | int | 可选 | 滑窗右边界，默认-1（当前仅支持-1） | int32 | - | - | - |
| return_softmax_lse | bool | 可选 | 是否返回softmaxLse，默认False（当前仅支持False） | bool | - | - | - |
| attention_out_dtype | dtype | 可选 | FP8输入时必填，须为float16或bfloat16 | ScalarType | - | - | - |
| attention_out | Tensor | 输出 | 注意力输出，shape与query一致 | float16/bfloat16 | ND | 与query一致 | - |
| softmax_lse | Tensor | 输出 | `return_softmax_lse=True`时为`(Tq, Nq, 1)`，否则为空张量 | float32 | ND | TND: (Tq, Nq, 1) 或空 | - |

## 返回值说明

### generic_block_sparse_attention_metadata

| 参数名 | 参数类型 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| metadata | Tensor | 主算子调度所需的预处理数据，调用者不应解析或修改 | int32 | ND | (1024,) |

### generic_block_sparse_attention

| 参数名 | 参数类型 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| attention_out | Tensor | 注意力输出，shape与query一致（FP8场景dtype由`attention_out_dtype`指定） | float16/bfloat16 | ND | 与query一致 |
| softmax_lse | Tensor | `return_softmax_lse=True`时为`(Tq, Nq, 1)`（TND），否则为空张量 | float32 | ND | TND: (Tq, Nq, 1) 或空 |

**说明**

- `attention_out`：限制：该输出参数的shape与入参query的shape保持一致；FP8输入时dtype由`attention_out_dtype`指定。
- `softmax_lse`：`return_softmax_lse=False`时返回空张量；当前仅支持False。
- `metadata`：为不透明数据，调用者不应解析或修改其中的内容。

## 约束说明

- 声明
  - `cu_seq_lengths_q`、`cu_seq_lengths_kv`、`seqused_q`、`seqused_kv`、`sparse_block_idx`、`sparse_block_count`及`block_table`均为Tensor。算子在Tiling阶段无法校验其具体值，用户必须保证其合法性；传入非法值可能导致精度异常或非法内存访问。
  - `cu_seq_lengths_q`、`cu_seq_lengths_kv`须满足首元素为0，且序列整体呈非递减排列。
  - `seqused_q`、`seqused_kv`可选。不传时，实际长度等于对应cu前缀和差分得到的存储长度；与cu同时传入时，分核/任务空间按各batch实际有效长度（seqused）累加，稀疏块分块与QKV寻址仍按cu存储长度计算，每个元素须不大于对应batch的存储长度。
  - 当`layout_kv`为`PAGED_BBND`时，`key`和`value`支持dim0非连续（FIA fused cache布局，stride0 = 2 × page）。
  - `generic_block_sparse_attention_metadata`和`generic_block_sparse_attention`分两段调用。两次调用中所有对应的Tensor和属性须完全一致（含`sparse_block_idx`、`sparse_block_count`、`block_shape`、`cu_seq_lengths_q`、`cu_seq_lengths_kv`、`seqused_q`、`seqused_kv`及`is_packed_gqa`、`layout_q`、`layout_kv`、`mask_type`、`quant_type`、`softmax_precision`、`win_left`、`win_right`等）；不一致时可能产生未定义行为。
  - 须先调用`generic_block_sparse_attention_metadata`生成`metadata`，且每次主算子调用前重新生成。
  - 本接口支持单算子模式和TorchAir图模式调用，可用于训练和推理场景。

### 特性参数组

| 特性参数组 | 参数字段名称 |
| :--- | :--- |
| 公共参数组 | layout_q、layout_kv、query、key、value、sparse_block_idx、sparse_block_count、block_shape、metadata、scale_value、attention_out |
| metadata参数组 | max_q_seq_len、max_kv_seq_len、num_q_heads、num_kv_heads、head_dim、metadata |
| Sparse参数组 | sparse_block_idx、sparse_block_count、block_shape |
| SeqLens参数组 | cu_seq_lengths_q、cu_seq_lengths_kv、seqused_q、seqused_kv |
| Paged Attention参数组 | block_table、key、value |
| Mask参数组 | mask_type、win_left、win_right |
| Quant参数组 | quant_type、q_dequant_scale、k_dequant_scale、v_dequant_scale、dst_type_max、attention_out_dtype |
| SoftmaxLSE参数组 | return_softmax_lse、softmax_lse、softmax_precision |

### 其他约束

- `atten_mask`、`p_quant_scale`当前须为None。
- quant_type=0（非量化）时，`q_dequant_scale`、`k_dequant_scale`、`v_dequant_scale`须为None；量化场景shape与dtype约束见`aclnnGenericBlockSparseAttention`文档。
- `num_q_heads`须满足`num_q_heads >= num_kv_heads`且`num_q_heads % num_kv_heads == 0`；`groupSize = num_q_heads / num_kv_heads`当前须不超过128。
- `topK`（`sparse_block_idx`最后一维）须不小于`sparse_block_count`中所有元素的最大值，且当前不超过256。
- 当前落地：`D=128`；`block_shape=[1, 128]`；KV页`blockSize=128`且等于`block_y`；`return_softmax_lse`仅支持False；Ascend 950上`softmax_precision`仅支持1。
- 详细约束见`aclnnGenericBlockSparseAttentionMetadata`与`aclnnGenericBlockSparseAttention`文档。

## 调用示例

```python
import math
import torch
import torch_npu
import cann_ops_transformer

torch_npu.npu.set_device(0)

B, Nq, Nkv, Sq, Skv, D = 1, 32, 8, 128, 256, 128
block_x, block_y = 1, 128
block_size = 128
top_k = 16

Tq = B * Sq
num_blocks = math.ceil(Skv / block_size)
total_q_blocks = math.ceil(Sq / block_x)

query = torch.randn(Tq, Nq, D, dtype=torch.float16, device="npu")
key = torch.randn(num_blocks, block_size, Nkv, D, dtype=torch.float16, device="npu")
value = torch.randn(num_blocks, block_size, Nkv, D, dtype=torch.float16, device="npu")

sparse_block_idx = torch.randint(
    0, num_blocks, (Nkv, total_q_blocks, top_k), dtype=torch.int32, device="npu"
)
sparse_block_count = torch.full((Nkv, total_q_blocks), top_k, dtype=torch.int32, device="npu")
cu_seq_lengths_q = torch.tensor([0, Sq], dtype=torch.int64, device="npu")
cu_seq_lengths_kv = torch.tensor([0, Skv], dtype=torch.int64, device="npu")
block_table = torch.arange(num_blocks, dtype=torch.int32, device="npu").view(1, -1)
block_shape = [block_x, block_y]

metadata = cann_ops_transformer.generic_block_sparse_attention_metadata(
    sparse_block_idx,
    sparse_block_count,
    Sq,
    Skv,
    Nq,
    Nkv,
    D,
    block_shape,
    cu_seq_lengths_q=cu_seq_lengths_q,
    cu_seq_lengths_kv=cu_seq_lengths_kv,
    is_packed_gqa=1,
    layout_q="TND",
    layout_kv="PAGED_BBND",
    mask_type=1,
    quant_type=0,
    softmax_precision=1,
)

attention_out, _ = cann_ops_transformer.generic_block_sparse_attention(
    query,
    key,
    value,
    sparse_block_idx,
    sparse_block_count,
    block_shape,
    metadata=metadata,
    cu_seq_lengths_q=cu_seq_lengths_q,
    cu_seq_lengths_kv=cu_seq_lengths_kv,
    block_table=block_table,
    is_packed_gqa=1,
    layout_q="TND",
    layout_kv="PAGED_BBND",
    scale_value=1.0 / (D ** 0.5),
    mask_type=1,
    quant_type=0,
    softmax_precision=1,
)
torch_npu.npu.synchronize()
assert attention_out.shape == (Tq, Nq, D)
```
