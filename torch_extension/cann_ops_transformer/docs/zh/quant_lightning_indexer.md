# quant\_lightning\_indexer\_metadata / quant\_lightning\_indexer

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明

- 接口功能：

  `quant_lightning_indexer_metadata`接口用于生成一个任务列表，包含每个AIcore的Attention计算任务的起止点的Batch、Head、以及Q和K的分块的索引，供后续`quant_lightning_indexer`算子使用。

## 函数原型

```python
cann_ops_transformer.quant_lightning_indexer_metadata(
  num_heads_q,
  num_heads_k,
  head_dim,
  topk,
  quant_mode,
  *,
  cu_seqlens_q=None,
  cu_seqlens_k=None,
  seqused_q=None,
  seqused_k=None,
  cmp_residual_k=None,
  batch_size=0,
  max_seqlen_q=-1,
  max_seqlen_k=-1,
  layout_q="BSND",
  layout_k="BSND",
  mask_mode=0,
  cmp_ratio=1
) -> Tensor
```

## 参数说明

### quant_lightning_indexer_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| num_heads_q | int | 必选 | 表示Query的head个数，当前仅支持32/64。 | int32 | - |
| num_heads_k | int | 必选 | 表示Key的head个数，当前仅支持1。 | int32 | - |
| head_dim | int | 必选 | 表示注意力头的维度，当前仅支持128。 | int32 | - |
| topk | int | 必选 | 表示从Query中筛选出的关键稀疏token的个数，当前仅支持[1, 2048]。 | int32 | - |
| quant_mode | int | 必选 | 表示量化模式，当前仅支持1/2/4。1表示qk: fp8(e4m3) per-token-head, scale: fp32；2表示qk: int8 per-token-head, scale: fp16, w: fp16；4表示qk: hif8 per-tensor, scale: fp32。 | int32 | - |
| cu_seqlens_q | Tensor | 可选 | 表示不同Batch中Query的有效Sequence Length，仅layout_q为TND场景下必传，第一个值固定为0。数据格式为ND，支持非连续的Tensor。 | int32 | (B+1, ) |
| cu_seqlens_k | Tensor | 可选 | 表示不同Batch中Key的有效Sequence Length，仅layout_k为TND场景下必传，第一个值固定为0。数据格式为ND，支持非连续的Tensor。 | int32 | (B+1, ) |
| seqused_q | Tensor | 可选 | 表示不同Batch中Query实际参与运算的Sequence Length。数据格式为ND，支持非连续的Tensor。 | int32 | (B, ) |
| seqused_k | Tensor | 可选 | 表示不同Batch中Key实际参与运算的Sequence Length。数据格式为ND，支持非连续的Tensor。 | int32 | (B, ) |
| cmp_residual_k | Tensor | 可选 | 表示不同Batch中cmp_kv压缩后Sequence Length的余数，配合cmp_ratio实现cmp_kv部分的mask和负载计算。cmp_ratio不为1且mask_mode为3场景下必传。数据格式为ND，支持非连续的Tensor。 | int32 | (B, ) |
| batch_size | int | 可选 | 表示Batch数量，默认值为0。 | int32 | - |
| max_seqlen_q | int | 可选 | 表示Query的最长Sequence Length，-1表示任意可能长度，默认值为-1。 | int32 | - |
| max_seqlen_k | int | 可选 | 表示Key的最长Sequence Length，-1表示任意可能长度，默认值为-1。 | int32 | - |
| layout_q | str | 可选 | 表示Query的排列格式，支持BSND、TND，默认值为BSND。 | string | - |
| layout_k | str | 可选 | 表示Key的排列格式，支持BSND、TND、PA_BBND，默认值为BSND。 | string | - |
| mask_mode | int | 可选 | 表示sparse模式，0表示No mask，3表示rightDownCausal模式，默认值为0。 | int32 | - |
| cmp_ratio | int | 可选 | 表示Key的压缩率，取值范围[1, 128]，默认值为1，表示无压缩。 | int32 | - |

- <term>Ascend 950PR/Ascend 950DT</term>：不支持quant_mode = 2。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持num_heads_q = 32，不支持quant_mode = 1/4，不支持layout_k = BSND/TND，不支持cmp_ratio在[1, 128]任意取值，仅支持cmp_ratio = 1/2/4/8/16/32/64/128。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持num_heads_q = 32，不支持quant_mode = 1/4，不支持layout_k = BSND/TND，不支持cmp_ratio在[1, 128]任意取值，仅支持cmp_ratio = 1/2/4/8/16/32/64/128。

## 返回值说明

### quant_lightning_indexer_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| metadata | Tensor | 必选 | 每个AIcore的Attention计算任务的Batch、Head、以及Q和K的分块的索引。数据格式为ND，不支持非连续的Tensor。 | int32 | (1024, ) |

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和aclgraph模式。
- quant_lightning_indexer_metadata接口需与quant_lightning_indexer算子配套使用。
- B（Batch）表示输入样本批量大小。
- 参数cu_seqlens_q、cu_seqlens_k要求其值为当前Batch与前序Batch有效token数的累加值，后一个元素的值必须大于等于前一个元素的值。
- 参数seqused_q、seqused_k要求其值表示每个Batch中的有效token数。
- 参数cmp_residual_k需满足cmp_residual_k\[i\] < cmp_ratio。
- mask_mode所表示的mask模式的详细介绍见[sparse_mode参数说明](../../../../docs/zh/context/sparse_mode_introduction.md)。

## 确定性计算

- 默认支持确定性计算
