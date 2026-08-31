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

  `generic_block_sparse_attention`是基于`TorchNPU`的`cann_ops_transformer`扩展接口，用于调用`GenericBlockSparseAttention`算子完成任意粒度块稀疏注意力计算。通过`sparse_block_idx`指定每个Q块选择的KV块，`sparse_block_count`指定每个Q块保留的KV块数量。

  须先调用`generic_block_sparse_attention_metadata`生成`metadata`，再调用本接口。

- **计算公式**：

  稀疏块大小为$blockShapeX \times blockShapeY$。

  $$
  attentionOut = Softmax(scaleValue \cdot query \cdot key_{sparse}^{T} + atten\_mask) \cdot value_{sparse}
  $$

  当前仅支持`q_input_layout="TND"`、`kv_input_layout="PAGED_BBND"`。

## 函数原型

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
    seq_used_q=None,
    seq_used_kv=None,
    block_table=None,
    is_packed_gqa=1,
    q_input_layout="TND",
    kv_input_layout="PAGED_BBND",
    scale_value=0.0,
    mask_type=1,
    quant_type=0,
    dst_type_max=0.0,
    softmax_precision=1,
    window_size_left=-1,
    window_size_right=-1,
    return_softmax_lse=False,
    attention_out_dtype=None,
) -> (Tensor, Tensor)
```

> [!NOTE]
>
> `*`之后的所有参数仅支持以关键字方式传入。

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| query | Tensor | 必选 | 公式中的query | float16/bfloat16/float8_e4m3fn | ND | TND: [Tq, Nq, D] |
| key | Tensor | 必选 | Paged KV Cache 中的key | float16/bfloat16/float8_e4m3fn | ND | PAGED_BBND: [numBlocks, blockSize, Nkv, D] |
| value | Tensor | 必选 | Paged KV Cache 中的value | float16/bfloat16/float8_e4m3fn | ND | 与key一致 |
| sparse_block_idx | Tensor | 必选 | 每个Q块选择的KV块索引 | int32 | ND | is_packed_gqa=1: [Nkv, totalQBlocks, topK] |
| sparse_block_count | Tensor | 必选 | 每个Q块实际保留的KV块数量 | int32 | ND | is_packed_gqa=1: [Nkv, totalQBlocks] |
| block_shape | list[int] | 必选 | 稀疏块形状 `[block_x, block_y]` | int64 | - | 长度为2 |
| metadata | Tensor | 必选 | 由`generic_block_sparse_attention_metadata`生成 | int32 | ND | (1024,) |
| atten_mask | Tensor | 可选 | 当前不支持，须为None | bool | ND | - |
| q_dequant_scale / k_dequant_scale / v_dequant_scale | Tensor | 可选 | 反量化缩放；非量化场景为None | float32等 | ND | 见aclnn文档 |
| p_quant_scale | Tensor | 可选 | 当前须为None | float32 | ND | - |
| cu_seq_lengths_q / cu_seq_lengths_kv | Tensor | TND必选 | query/kv序列长度前缀和 | int64 | ND | [B+1] |
| seq_used_q / seq_used_kv | Tensor | 可选 | 实际使用的序列长度 | int32 | ND | [B] |
| block_table | Tensor | PAGED必选 | 物理页映射表 | int32 | ND | [B, maxNumBlocksPerBatch] |
| is_packed_gqa | int | 可选 | 1表示同group共享稀疏pattern，默认1 | int64 | - | - |
| q_input_layout | string | 可选 | 默认`"TND"`，当前仅支持该值 | string | - | - |
| kv_input_layout | string | 可选 | 默认`"PAGED_BBND"`，当前仅支持该值 | string | - | - |
| scale_value | float | 可选 | 注意力缩放系数；默认0.0时算子内回落为`D^-0.5` | double | - | - |
| mask_type | int | 可选 | 掩码类型，默认1（当前仅支持1） | int64 | - | - |
| quant_type | int | 可选 | 量化类型，默认0（非量化） | int64 | - | - |
| dst_type_max | float | 可选 | 量化相关参数，默认0.0 | double | - | - |
| softmax_precision | int | 可选 | Softmax精度模式，默认1。Ascend 950仅支持1；须与metadata一致 | int64 | - | - |
| window_size_left / window_size_right | int | 可选 | 滑窗左右窗口，默认-1（当前仅支持-1） | int64 | - | - |
| return_softmax_lse | bool | 可选 | 是否返回softmaxLse，默认False（当前仅支持False） | bool | - | - |
| attention_out_dtype | dtype | 可选 | FP8输入时必填，须为float16或bfloat16 | ScalarType | - | - |

## 返回值说明

- `attention_out`：注意力输出，shape与query一致（FP8场景dtype由`attention_out_dtype`指定）。
- `softmax_lse`：`return_softmax_lse=True`时为`[T, N, 1]`（TND），否则为空张量。

## 约束说明

- 须先调用`generic_block_sparse_attention_metadata`生成`metadata`，且每次主算子调用前重新生成；`softmax_precision`等属性须与metadata一致。
- `cu_seq_lengths_q` / `cu_seq_lengths_kv` 必须为 **int64**。
- 当前落地：`D=128`；`block_shape=[1, 128]`；KV页`blockSize=128`且等于`block_y`；`groupSize≤128`；`topK≤256`；`return_softmax_lse`仅支持False。
- 详细约束见`aclnnGenericBlockSparseAttention`文档。

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
    cu_seq_lengths=cu_seq_lengths_q,
    cu_seq_lengths_kv=cu_seq_lengths_kv,
    is_packed_gqa=1,
    q_input_layout="TND",
    kv_input_layout="PAGED_BBND",
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
    q_input_layout="TND",
    kv_input_layout="PAGED_BBND",
    scale_value=1.0 / (D ** 0.5),
    mask_type=1,
    quant_type=0,
    softmax_precision=1,
)
torch_npu.npu.synchronize()
assert attention_out.shape == (Tq, Nq, D)
```
