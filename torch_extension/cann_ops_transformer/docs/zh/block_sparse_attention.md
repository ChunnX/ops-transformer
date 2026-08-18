# block_sparse_attention

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

> [!NOTE]
>
> FP8（`quant_mode=1`）、MXFP4（`quant_mode=2/3`）量化输入以及BSND布局仅在<term>Ascend 950PR/Ascend 950DT</term>（arch35）上支持；<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>仅支持float16/bfloat16输入与TND/BNSD布局。

## 功能说明

- **接口功能**：

  `block_sparse_attention`是基于`TorchNPU`的`cann_ops_transformer`扩展接口，用于调用`BlockSparseAttention`算子完成块级稀疏注意力计算。通过`block_sparse_mask`指定每个Q块需要参与计算的KV块，实现高效的稀疏注意力计算，训练推理归一化。

  当前支持以下量化场景：

  - **非量化场景**（`quant_mode=0`，默认）：Q/K/V 均为 float16 或 bfloat16；
  - **FP8场景**（`quant_mode=1`）：Q/K/V 均为 float8_e4m3fn，携带 FP32 反量化缩放因子，输出反量化为 float16 或 bfloat16；
  - **MXFP4场景**（`quant_mode=2/3`）：Q/K/V 为 float4_e2m1fn_x2（以 uint8 传入），携带 float8_e8m0fnu descale 张量，输出反量化为 float16 或 bfloat16。

- **计算公式**：

  self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n \times d$的矩阵。

  self-attention的计算公式一般定义如下，其中$Q、K、V$为输入样本的重要属性元素，是输入样本经过空间变换得到，且可以统一到一个特征空间中。公式及算子名称中的"Attention"为"self-attention"的简写。稀疏块大小为$blockShapeX \times blockShapeY$，由`block_sparse_mask`指定稀疏模式。

  $$
  attentionOut = Softmax(scale \cdot query \cdot key_{sparse}^T) \cdot value_{sparse}
  $$

  其中$scale$为缩放系数，一般取$scale = D^{-0.5}$，$key_{sparse}$、$value_{sparse}$为`block_sparse_mask`选取的KV块。

> [!NOTE]
>
> query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小batch_size、S（Seq-Length）表示输入样本序列长度、H（Hidden-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸headdim，且满足D=H/N、T表示B和S合轴紧密排列的长度（Total tokens）。Q_S表示输入query tensor的序列长度，Q_N表示输入query tensor的头数，KV_S表示输入key/value tensor的序列长度，KV_N表示输入key/value tensor的头数。

## 函数原型

```python
cann_ops_transformer.block_sparse_attention(
    query,
    key,
    value,
    block_sparse_mask,
    block_shape,
    actual_seq_lengths,
    actual_seq_lengths_kv,
    num_heads,
    num_key_value_heads,
    q_input_layout="BNSD",
    kv_input_layout="BNSD",
    mask_type=0,
    scale_value=1.0,
    inner_precise=0,
    block_size=0,
    pre_tokens=2147483647,
    next_tokens=2147483647,
    softmax_lse_flag=False,
    quant_mode=0,
    dst_type_max=0,
    q_dequant_scale=None,
    k_dequant_scale=None,
    v_dequant_scale=None,
    p_quant_scale=None,
    attention_out_dtype=None
) -> (Tensor, Tensor)
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| query | Tensor | 必选 | 公式中的query | float16/bfloat16/float8_e4m3fn/uint8 | ND | <ul><li>TND: [Tq, Nq, D]</li><li>BNSD: [B, Nq, Sq, D]</li><li>BSND: [B, Sq, Nq, D]</li></ul> | × |
| key | Tensor | 必选 | 公式中的key | float16/bfloat16/float8_e4m3fn/uint8 | ND | <ul><li>TND: [Tkv, Nkv, D]</li><li>BNSD: [B, Nkv, Skv, D]</li><li>BSND: [B, Skv, Nkv, D]</li></ul> | × |
| value | Tensor | 必选 | 公式中的value | float16/bfloat16/float8_e4m3fn/uint8 | ND | <ul><li>TND: [Tkv, Nkv, D]</li><li>BNSD: [B, Nkv, Skv, D]</li><li>BSND: [B, Skv, Nkv, D]</li></ul> | × |
| block_sparse_mask | Tensor | 必选 | 块级稀疏mask，1表示保留该块参与计算，0表示跳过该块 | int8 | ND | [B, N, ceil(Sq/block_x), ceil(Skv/block_y)] | × |
| block_shape | int[] | 必选 | 稀疏块形状，必须包含两个元素[block_x, block_y] | int64 | - | (2,) | - |
| actual_seq_lengths | int[] | TND时必选/BNSD、BSND时可选 | 每个batch对应的query实际序列长度 | int64 | - | (B,) | - |
| actual_seq_lengths_kv | int[] | TND时必选/BNSD、BSND时可选 | 每个batch对应的key/value实际序列长度 | int64 | - | (B,) | - |
| num_heads | int | 必选 | query的head个数 | int64 | - | - | - |
| num_key_value_heads | int | 必选 | key/value的head个数 | int64 | - | - | - |
| q_input_layout | string | 可选 | 输入query张量的数据排布格式 | string | - | - | - |
| kv_input_layout | string | 可选 | 输入key/value张量的数据排布格式 | string | - | - | - |
| mask_type | int | 可选 | attention计算中的掩码类型，当前仅支持0 | int64 | - | - | - |
| scale_value | float | 可选 | 公式中的scale，一般设置为D^-0.5 | double | - | - | - |
| inner_precise | int | 可选 | Softmax计算采取的精度级别，支持0、1、4 | int64 | - | - | - |
| block_size | int | 可选 | PagedAttention的block大小，当前不支持PagedAttention，仅支持0 | int64 | - | - | - |
| pre_tokens | int | 可选 | 滑窗attention需要向前包含的token数，当前仅支持2147483647 | int64 | - | - | - |
| next_tokens | int | 可选 | 滑窗attention需要向后包含的token数，当前仅支持2147483647 | int64 | - | - | - |
| softmax_lse_flag | bool | 可选 | 是否使能softmaxLse输出 | BOOL | - | - | - |
| quant_mode | int | 可选 | 量化模式，支持0、1、2、3 | int64 | - | - | - |
| dst_type_max | float | 可选 | MXFP4 CX量化时传入的自定义量化量程 | double | - | - | - |
| q_dequant_scale | Tensor | 可选 | query的反量化缩放因子 | float32/float8_e8m0fnu | ND | 见「约束说明」 | × |
| k_dequant_scale | Tensor | 可选 | key的反量化缩放因子 | float32/float8_e8m0fnu | ND | 见「约束说明」 | × |
| v_dequant_scale | Tensor | 可选 | value的反量化缩放因子 | float32/float8_e8m0fnu | ND | 见「约束说明」 | × |
| p_quant_scale | Tensor | 可选 | softmax结果p的量化缩放因子，预留参数，当前不支持，必须为None | float32 | ND | - | × |
| attention_out_dtype | ScalarType | 可选 | attentionOut的数据类型，默认None | - | - | - | - |

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| attention_out | Tensor | 必选 | 公式中的attentionOut | float16/bfloat16 | ND | 与query一致（MXFP4输入时最后一维为2*D） | × |
| softmax_lse | Tensor | 可选 | Softmax计算的log-sum-exp中间结果，`softmax_lse_flag`为True时输出，否则输出空Tensor {0} | float32 | ND | <ul><li>TND: [Tq, Nq, 1]</li><li>BNSD: [B, Nq, Sq, 1]</li><li>BSND: [B, Sq, Nq, 1]</li></ul> | × |

## 约束说明

- q_input_layout与kv_input_layout必须保持一致，且仅支持"TND"、"BNSD"、"BSND"。
- query、key、value的数据类型必须一致，支持float16和bfloat16；FP8（`quant_mode=1`）时三者必须同时为float8_e4m3fn；MXFP4（`quant_mode=2/3`）时三者必须同时为uint8（float4_e2m1fn_x2伪装）。
- query、key、value的D轴当前仅支持64或128；MXFP4场景下D仅支持64/128。
- num_heads（N1）与num_key_value_heads（N2）需满足 N1 >= N2 && N1 % N2 == 0。
- block_shape必须包含两个元素[block_x, block_y]，值必须大于0，且block_y必须为128的倍数；MXFP4场景下block_x仅支持64的倍数。
- block_sparse_mask当前必须传入，且shape必须为[batch, headNum, ceilDiv(maxQS, block_x), ceilDiv(maxKVS, block_y)]。
- actual_seq_lengths在q_input_layout为"TND"时必选；actual_seq_lengths_kv在kv_input_layout为"TND"时必选；两者必须同时配置或同时不配置，仅配置其中之一将被拦截。
- qSeqlen和kvSeqlen不需要被block_shape整除，支持非对齐场景，实际分块数通过向上取整计算。
- innerPrecise仅支持配置0、1或4：
  - 0：online softmax和rescale全部采取fp32数据类型；
  - 1：仅支持query、key、value均为fp16时配置，全部采取fp16数据类型，性能更好但精度较低；
  - 4：混合精度运算，online softmax采取fp16/bf16，rescale采取fp32。
  - <term>Ascend 950PR/Ascend 950DT</term>仅支持配置为4；<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>仅支持配置为0或1。
- mask_type当前仅支持0，表示不加mask。
- block_size当前仅支持0，表示不支持paged cache。
- pre_tokens和next_tokens当前仅支持2147483647，表示当前token的前后所有token都参与attention运算，即不支持滑窗attention。
- p_quant_scale为预留参数，当前必须为None。
- quant_mode=0（非量化）时，q_dequant_scale、k_dequant_scale、v_dequant_scale必须为None。

### quant_mode=1（FP8量化）相关约束

- 仅<term>Ascend 950PR/Ascend 950DT</term>支持。
- query、key、value必须同时为float8_e4m3fn，且必须同时提供q_dequant_scale、k_dequant_scale、v_dequant_scale三个反量化缩放因子。
- 反量化缩放因子的数据类型必须为float32：
  - q_dequant_scale：shape为[Batch, HeadNum, ceilDiv(maxQSeqLength, 128), 1]，在QK矩阵乘法时对query进行反量化；
  - k_dequant_scale：shape为[Batch, KVHeadNum, ceilDiv(maxKVSeqLength, 256), 1]或[Batch, KVHeadNum, ceilDiv(maxKVSeqLength, 512), 1]，在QK矩阵乘法时对key进行反量化；
  - v_dequant_scale：shape与k_dequant_scale一致，在PV矩阵乘法时对value进行反量化。
- block_shape必须传入，且q、kv的量化块大小必须与block_shape的两个元素大小分别保持一致。
- attention_out_dtype必须为float16或bfloat16（不存在fp8→fp8的输出二进制）。

### quant_mode=2/3（MXFP4量化）相关约束

- 仅<term>Ascend 950PR/Ascend 950DT</term>支持。
- query、key、value为float4_e2m1fn_x2（以uint8伪装传入），需要提供float8_e8m0fnu类型的descale张量：
  - q_dequant_scale：MX量化模式，仅支持Rowwise E8M0 Micro Scaling，shape为：
    - BNSD: [B, Nq, Sq, ceilDiv(D, 64), 2]；
    - BSND: [B, Sq, Nq, ceilDiv(D, 64), 2]；
    - TND: [Tq, Nq, ceilDiv(D, 64), 2]。
  - k_dequant_scale：MX量化模式，仅支持Rowwise E8M0 Micro Scaling，shape为：
    - BNSD: [B, Nkv, Skv, ceilDiv(D, 64), 2]；
    - BSND: [B, Skv, Nkv, ceilDiv(D, 64), 2]；
    - TND: [Tkv, Nkv, ceilDiv(D, 64), 2]。
  - v_dequant_scale：MX量化模式，仅支持Columnwise E8M0 Micro Scaling，shape为：
    - BNSD: [B, Nkv, ceilDiv(Skv, 64), D, 2]；
    - BSND: [B, ceilDiv(Skv, 64), Nkv, D, 2]；
    - TND: [BTkv//64, Nkv, D, 2]，其中$BTkv = \sum_{b=0}^{B-1}\lceil\frac{S2_b}{64}\rceil$。
- attention_out_dtype必须为float16或bfloat16，attentionOut的最后一维为2*D（两个fp4数值打包进一个uint8字节）。
- dst_type_max当quant_mode=2/3时可设置0.0或[6.0, 12.0]：0.0代表Amax(DType)为量化结果数据类型的最大值；6.0~12.0代表Amax(DType)为传入值。
- 暂不支持attenMask、PagedAttention、softmaxLse等高阶特性。

## 调用示例

```python
import torch
import torch_npu
import cann_ops_transformer

B, N, Nkv, S, D = 1, 32, 32, 128, 128
block_x, block_y = 128, 128

query = torch.randn(B * S, N, D, dtype=torch.float16, device="npu")
key = torch.randn(B * S, Nkv, D, dtype=torch.float16, device="npu")
value = torch.randn(B * S, Nkv, D, dtype=torch.float16, device="npu")

# block_sparse_mask: [B, N, ceil(S/block_x), ceil(Skv/block_y)]
mask = torch.ones(B, N, S // block_x, S // block_y, dtype=torch.int8, device="npu")

attn_out, softmax_lse = cann_ops_transformer.block_sparse_attention(
    query, key, value, mask,
    [block_x, block_y],
    actual_seq_lengths=[S],
    actual_seq_lengths_kv=[S],
    num_heads=N,
    num_key_value_heads=Nkv,
    q_input_layout="TND",
    kv_input_layout="TND",
    scale_value=1.0 / (D ** 0.5),
    softmax_lse_flag=True,
)
```
