# mixed\_quant\_sparse\_flash\_mla\_metadata / mixed\_quant\_sparse\_flash\_mla

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明

- 接口功能：

  `mixed_quant_sparse_flash_mla_metadata`接口用于生成一个任务列表，包含每个AIcore的Attention计算任务的起止点的Batch、Head、以及Q和K的分块的索引，供后续mixed_quant_sparse_flash_mla算子使用。
  `mixed_quant_sparse_flash_mla`是基于`torch_npu`的`cann_ops_transformer`扩展接口，用于调用`MixedQuantSparseFlashMla`算子完成共享KV（Key和Value使用同一份输入）的稀疏注意力计算。该接口支持以下三类计算模式：

  - **SWA（Sliding Window Attention）**：仅使用`ori_kv`，对原始KV做滑动窗口注意力。
  - **CSA（Compressed Sparse Attention）**：同时使用`ori_kv`、`cmp_kv`和`cmp_sparse_indices`，对原始KV窗口和TopK选择出的压缩KV共同做注意力。
  - **HCA（Heavily Compressed Attention）**：同时使用`ori_kv`和`cmp_kv`，对原始KV窗口和连续压缩KV段共同做注意力。

  `mixed_quant_sparse_flash_mla_metadata`是`MixedQuantSparseFlashMla`的torch扩展接口，用于在主算子执行前生成metadata。metadata记录AICore/AIVCore的任务切分结果，主算子必须传入该metadata。典型调用流程如下：

  1. 准备`q`、`ori_kv`、`cmp_kv`、序列长度、`block table`、`sinks`等输入。
  2. 调用`mixed_quant_sparse_flash_mla_metadata`生成`metadata`。
  3. 调用`mixed_quant_sparse_flash_mla`，将上一步得到的`metadata`传入主算子。

- 计算公式：

  $$
  O = \text{softmax}(Q@\tilde{K}^T \cdot \text{softmax\_scale})@\tilde{V}
  $$

  其中$\tilde{K}=\tilde{V}$为基于入参控制的实际参与计算的$KV$，由`ori_kv`的滑动窗口部分和`cmp_kv`的压缩部分共同组成，实际参与计算的KV范围由`cmp_ratio`、`ori_mask_mode`、`cmp_mask_mode`、`ori_win_left`、`ori_win_right`以及`cmp_sparse_indices`决定。

> [!NOTE]
>
> `cmp_residual_kv`同时是`sparse_flash_mla`和`sparse_flash_mla_metadata`的可选输入。该参数用于恢复压缩前KV长度：`ori_len_for_cmp_mask = cmp_len * cmp_ratio + cmp_residual_kv[b]`。

## 函数原型

```python
cann_ops_transformer.mixed_quant_sparse_flash_mla_metadata(num_heads_q, num_heads_kv, head_dim, quant_mode, *, cu_seqlens_q=None, cu_seqlens_ori_kv=None, cu_seqlens_cmp_kv=None, seqused_q=None, seqused_ori_kv=None, seqused_cmp_kv=None, cmp_residual_kv=None, ori_topk_length=None, cmp_topk_length=None, batch_size=0, max_seqlen_q=0, max_seqlen_ori_kv=0, max_seqlen_cmp_kv=0, ori_topk=0, cmp_topk=0, rope_head_dim=64, cmp_ratio=1, ori_mask_mode=0, cmp_mask_mode=0, ori_win_left=-1, ori_win_right=-1, layout_q="BSND", layout_kv="BSND", has_ori_kv=True, has_cmp_kv=True) -> Tensor
```

## 参数说明

### mixed_quant_sparse_flash_mla_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| num_heads_q | int | 必选 | 表示Query的head个数，当前仅支持2/4/8/16/32/64/128。 | int32 | - |
| num_heads_kv | int | 必选 | 表示Key和Value对应的多头数，当前仅支持1。 | int32 | - |
| head_dim | int | 必选 | 表示注意力头的维度，当前仅支持512。 | int32 | - |
| quant_mode | int | 必选 | 表示量化模式，1表示K、V nope为per-token-group量化，scale类型为bfloat16，2表示K、V nope为per-token-group量化，scale类型为float8_e8m0。 | int32 | - |
| cu_seqlens_q | Tensor | 可选 | 表示不同Batch中Query的有效Sequence Length，仅layout_q为TND场景需传入。数据格式为ND，支持非连续的Tensor。 | int32 | B+1 |
| cu_seqlens_ori_kv | Tensor | 可选 | 表示不同Batch中ori_kv的有效Sequence Length，仅layout_kv为TND场景需传入。数据格式为ND，支持非连续的Tensor。 | int32 | B+1 |
| cu_seqlens_cmp_kv | Tensor | 可选 | 表示不同Batch中cmp_kv的有效Sequence Length，仅layout_kv为TND场景需传入。数据格式为ND，支持非连续的Tensor。 | int32 | B+1 |
| seqused_q | Tensor | 可选 | 表示不同Batch中Query实际参与运算的Sequence Length。数据格式为ND，支持非连续的Tensor。 | int32 | B |
| seqused_ori_kv | Tensor | 可选 | 表示不同Batch中ori_kv实际参与运算的Sequence Length。数据格式为ND，支持非连续的Tensor。 | int32 | B |
| seqused_cmp_kv | Tensor | 可选 | 表示不同Batch中cmp_kv实际参与运算的Sequence Length。数据格式为ND，支持非连续的Tensor。 | int32 | B |
| cmp_residual_kv | Tensor | 可选 | 表示不同Batch中cmp_kv压缩后Sequence Length的余数，配合cmp_ratio实现cmp_kv部分的mask和负载计算。cmp_mask_mode=3且cmp_ratio≠1时必须传入。数据格式为ND，支持非连续的Tensor。 | int32 | B |
| ori_topk_length | Tensor | 可选 | 预留参数，当前不生效。数据格式为ND，支持非连续的Tensor。 | int32 | (B, S1, N2)或(T1, N2) |
| cmp_topk_length | Tensor | 可选 | 预留参数，当前不生效。数据格式为ND，支持非连续的Tensor。 | int32 | (B, S1, N2)或(T1, N2) |
| batch_size | int | 可选 | 表示Batch数量，默认值为0。 | int32 | - |
| max_seqlen_q | int | 可选 | 表示Query的最长Sequence Length，默认值为0。 | int32 | - |
| max_seqlen_ori_kv | int | 可选 | 表示ori_kv的最长Sequence Length，默认值为0。 | int32 | - |
| max_seqlen_cmp_kv | int | 可选 | 表示cmp_kv的最长Sequence Length，默认值为0。 | int32 | - |
| ori_topk | int | 可选 | 预留参数，当前不生效，表示ori_kv中筛选出的关键稀疏token的个数，0表示非稀疏场景，默认值为0。 | int32 | - |
| cmp_topk | int | 可选 | 表示cmp_kv中筛选出的关键稀疏token的个数，0表示非稀疏场景，默认值为0，当前仅支持512/1024。 | int32 | - |
| rope_head_dim | int | 可选 | 表示rope头的维度，默认值为64，当前仅支持64。 | int32 | - |
| cmp_ratio | int | 可选 | 表示对cmp_kv的压缩率，取值范围[1, 128]，默认值为1，当前仅支持4/128。 | int32 | - |
| ori_mask_mode | int | 可选 | 表示q和ori_kv计算的mask模式，0表示No mask，3表示rightDownCausal模式，4表示sliding window模式，默认值为0，当前仅支持4。 | int32 | - |
| cmp_mask_mode | int | 可选 | 表示q和cmp_kv计算的mask模式，0表示No mask，3表示rightDownCausal模式，默认值为0，当前仅支持3。 | int32 | - |
| ori_win_left | int | 可选 | 表示q和ori_kv计算中q对过去token计算的数量，-1表示无穷大，默认值为-1，当前仅支持127。 | int32 | - |
| ori_win_right | int | 可选 | 表示q和ori_kv计算中q对未来token计算的数量，-1表示无穷大，默认值为-1，当前仅支持0。 | int32 | - |
| layout_q | str | 可选 | 表示Query的排列格式，支持"BSND"、"TND"，默认值为"BSND"。 | string | - |
| layout_kv | str | 可选 | 表示Key的排列格式，支持"BSND"、"TND"、"PA_BBND"，默认值为"BSND"。 | string | - |
| has_ori_kv | bool | 可选 | 用于标识是否含有ori_kv，默认值为True。 | bool | - |
| has_cmp_kv | bool | 可选 | 用于标识是否含有cmp_kv，默认值为True。 | bool | - |

## 返回值说明

### mixed_quant_sparse_flash_mla_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| metadata | Tensor | 必选 | 每个cube核上FlashAttention计算任务的Batch、Head、以及 Q 和 K 的分块的索引，以及每个vector核上FlashDecode的规约任务索引。数据格式为ND，不支持非连续的Tensor。 | int32 | 1024 |

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和aclgraph模式。
- mixed_quant_sparse_flash_mla_metadata接口需与mixed_quant_sparse_flash_mla算子配套使用。
- `layout_q`支持"BSND"和"TND"；`layout_q="BSND"`时，`q`必须为4维；`layout_q="TND"`时，`q`必须为3维且必须传入`cu_seqlens_q`。
- `layout_kv`支持"BSND"、"TND"和"PA_BBND"；`layout_kv="BSND"`或`layout_kv="PA_BBND"`时，`ori_kv`和`cmp_kv`必须为4维；`layout_kv="TND"`时，`ori_kv`和`cmp_kv`必须为3维。
- `layout_kv="TND"`时必须传入`cu_seqlens_ori_kv`；传入`cmp_kv`时，还必须传入`cu_seqlens_cmp_kv`。
- B（Batch）表示输入样本批量大小。
- 参数`cu_seqlens_q`、`cu_seqlens_ori_kv`及`cu_seqlens_cmp_kv`要求其值为当前Batch与前序Batch有效token数的累加值，后一个元素的值必须大于等于前一个元素的值。
- 参数`seqused_q`、`seqused_ori_kv`、`seqused_cmp_kv`要求其值表示每个Batch中的有效token数。
- 参数`cmp_residual_kv`需满足`cmp_residual_kv\[i\]` < `cmp_ratio`。
- `layout_kv="PA_BBND"`时必须传入`seqused_ori_kv`和`ori_block_table`；传入`cmp_kv`时，还必须传入`cmp_block_table`。
- `seqused_cmp_kv`为所有`layout_kv`下的可选输入，显式传入时用于覆盖cmp侧逻辑有效长度。
- `ori_mask_mode`及`cmp_mask_mode`所表示的mask模式的详细介绍见[sparse_mode参数说明](../../../../docs/zh/context/sparse_mode参数说明.md)。
- `metadata`固定为1024个INT32元素，`topk_value_mode`仅支持1，`ori_sparse_indices`、`ori_topk_length`和`cmp_topk_length`当前版本不支持传入非空Tensor。
- `ori_kv`和`cmp_kv`允许存在行间padding类非连续内存，接口会通过aclnn获取stride信息传给底层算子。

## 确定性计算

- 默认支持确定性计算