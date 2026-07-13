# mixed_quant_sparse_flash_mla

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明

- 接口功能:

  `mixed_quant_sparse_flash_mla`是基于`torch_npu`的`cann_ops_transformer`扩展接口，用于调用`MixedQuantSparseFlashMla`算子完成量化和稀疏场景下的MLA（Multi-head Latent Attention）注意力计算，训练推理归一化。该接口支持以下三类计算模式：

  - **SWA（Sliding Window Attention）**：仅传入`ori_kv`，对原始KV做滑动窗口注意力。
  - **CSA（Compressed Sparse Attention）**：同时传入`ori_kv`、`cmp_kv`和`cmp_sparse_indices`，对原始KV窗口和TopK选择出的压缩KV共同做注意力。
  - **HCA（Heavily Compressed Attention）**：同时传入`ori_kv`和`cmp_kv`，对原始KV窗口和连续压缩KV段共同做注意力。

  `mixed_quant_sparse_flash_mla_metadata`是`mixed_quant_sparse_flash_mla`的元数据生成接口，用于在主算子执行前生成metadata。metadata记录AICore/AIVCore的任务切分结果，主算子必须传入该metadata。典型调用流程如下：

  1. 准备`q`、`ori_kv`、`cmp_kv`、`block_table`、`sinks`等输入。
  2. 调用`mixed_quant_sparse_flash_mla_metadata`生成`metadata`。
  3. 调用`mixed_quant_sparse_flash_mla`，将上一步得到的`metadata`传入主算子。

- 计算公式:

  mixed_quant_sparse_flash_mla采用MLA（Multi-head Latent Attention）对KV共享输入的稀疏注意力进行计算，其原理是对输入的KV进行选择性压缩与量化处理，再将Query与拼接后的KV计算结果通过Softmax得到注意力权重。

  MLA的计算公式一般定义如下，其中$\tilde{K}=\tilde{V}$为基于入参控制的实际参与计算的KV，由`ori_kv`的滑动窗口部分和`cmp_kv`的压缩部分共同组成，实际参与计算的KV范围由`cmp_ratio`、`ori_mask_mode`、`cmp_mask_mode`、`ori_win_left`、`ori_win_right`以及`cmp_sparse_indices`决定。

  $$
  O = \text{softmax}(Q\tilde{K}^T \cdot \text{softmax\_scale}) \tilde{V}
  $$

> [!NOTE]
>
> q、k、v数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小batch_size、S（Seq-Length）表示输入样本序列长度、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸headdim。
> Q_S表示seqlen_q，Q_N表示nheads_q，Q_D表示dheads_q，KV_S表示seqlen_kv，KV_N表示nheads_kv，KV_D表示dheads_kv。
>
> `cmp_residual_kv`同时是`mixed_quant_sparse_flash_mla`和`mixed_quant_sparse_flash_mla_metadata`的可选输入。该参数用于恢复压缩前KV长度：`ori_len_for_cmp_mask[b] = cmp_len[b] * cmp_ratio + cmp_residual_kv[b]`。

## 函数原型

调用mixed_quant_sparse_flash_mla接口之前，请先调用前置接口mixed_quant_sparse_flash_mla_metadata，完成mixed_quant_sparse_flash_mla负载均衡的计算。

```python
cann_ops_transformer.ops.mixed_quant_sparse_flash_mla_metadata(
    num_heads_q,
    num_heads_kv,
    head_dim,
    quant_mode,
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
    batch_size=0,
    max_seqlen_q=0,
    max_seqlen_ori_kv=0,
    max_seqlen_cmp_kv=0,
    ori_topk=0,
    cmp_topk=0,
    rope_head_dim=64,
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

```python
cann_ops_transformer.ops.mixed_quant_sparse_flash_mla(
    q,
    quant_mode,
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
    rope_head_dim=64,
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



## 参数说明

### mixed_quant_sparse_flash_mla_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| num_heads_q | int | 必选 | Query head数 | int32 | - | - | - |
| num_heads_kv | int | 必选 | Key/Value head数 | int32 | - | - | - |
| head_dim | int | 必选 | 每个注意力头的维度 | int32 | - | - | - |
| quant_mode | int | 必选 | 表示量化模式，当前仅支持1和2 | int32 | - | - | - |
| cu_seqlens_q | Tensor | 可选 | 累积序列长度，用于处理变长序列，第一个元素必须为0 | int32 | ND | (B+1,) | √ |
| cu_seqlens_ori_kv | Tensor | 可选 | 累积序列长度，用于处理变长序列，第一个元素必须为0 | int32 | ND | (B+1,) | √ |
| cu_seqlens_cmp_kv | Tensor | 可选 | 累积序列长度，用于处理变长序列，第一个元素必须为0 | int32 | ND | (B+1,) | √ |
| seqused_q | Tensor | 可选 | 指定每batch中实际使用的序列长度，截断冗余运算 | int32 | ND | (B,) | √ |
| seqused_ori_kv | Tensor | 可选 | 指定每batch中实际使用的序列长度，截断冗余运算 | int32 | ND | (B,) | √ |
| seqused_cmp_kv | Tensor | 可选 | 指定每batch中实际使用的序列长度，截断冗余运算 | int32 | ND | (B,) | √ |
| cmp_residual_kv | Tensor | 可选 | 不同batch中cmp_kv压缩后序列长度的余数，配合cmp_ratio实现cmp_kv部分的mask和负载计算 | int32 | ND | (B,) | √ |
| ori_topk_length | Tensor | 可选 | 预留参数，当前不生效 | int32 | ND | <ul><li>(B, Q_S, KV_N)</li><li>(Q_T, KV_N)</li></ul> | √ |
| cmp_topk_length | Tensor | 可选 | 预留参数，当前不生效 | int32 | ND | <ul><li>(B, Q_S, KV_N)</li><li>(Q_T, KV_N)</li></ul> | √ |
| batch_size | int | 可选 | batch大小。若未传入，则从cu_seqlens_q或seqused_q推导。默认值为0 | int32 | - | - | - |
| max_seqlen_q | int | 可选 | 指定查询q序列的长度上限 | int32 | - | - | - |
| max_seqlen_ori_kv | int | 可选 | 指定ori_kv序列的长度上限 | int32 | - | - | - |
| max_seqlen_cmp_kv | int | 可选 | 指定cmp_kv序列的长度上限 | int32 | - | - | - |
| ori_topk | int | 可选 | 表示ori_kv中筛选出的关键稀疏token的个数，0表示非稀疏场景。默认值为0 | int32 | - | - | - |
| cmp_topk | int | 可选 | 表示cmp_kv中筛选出的关键稀疏token的个数。0表示非稀疏场景。默认值为0 | int32 | - | - | - |
| rope_head_dim | int | 可选 | rope头的维度。默认值为64 | int32 | - | - | - |
| cmp_ratio | int | 可选 | 表示对cmp_kv的压缩率。默认值为1 | int32 | - | - | - |
| ori_mask_mode | int | 可选 | 表示q和ori_kv计算的mask模式。0：No mask。3：rightDownCausal模式。4：sliding window模式。默认值为0 | int32 | - | - | - |
| cmp_mask_mode | int | 可选 | 表示q和cmp_kv计算的mask模式。0：No mask。3：rightDownCausal模式。默认值为0 | int32 | - | - | - |
| ori_win_left | int | 可选 | 表示q和ori_kv计算中q对过去token计算的数量，-1表示无穷大。默认值为-1 | int32 | - | - | - |
| ori_win_right | int | 可选 | 表示q和ori_kv计算中q对未来token计算的数量，-1表示无穷大。默认值为-1 | int32 | - | - | - |
| layout_q | string | 可选 | 定义输入q张量的布局格式 | string | - | - | - |
| layout_kv | string | 可选 | 定义输入k和v张量的布局格式 | string | - | - | - |
| has_ori_kv | bool | 可选 | 用于标识是否含有ori_kv。默认值为True | bool | - | - | - |
| has_cmp_kv | bool | 可选 | 用于标识是否含有cmp_kv。默认值为True | bool | - | - | - |

### mixed_quant_sparse_flash_mla

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| q | Tensor | 必选 | 公式中的q | bfloat16 | ND |  <ul><li>(B, Q_N, Q_S, D)</li><li>(Q_T, Q_N, D)</li></ul> | × |
| quant_mode | int | 必选 | 表示量化模式。当前仅支持1和2 | int32 | - | - | - |
| ori_kv | Tensor | 可选 | 原始量化KV输入，Key和Value共享同一份数据。 | fp8_e4m3 | ND | <ul><li>(B, KV_S, KV_N, D)</li><li>(B, KV_N, KV_S, D)</li><li>(KV_T, KV_N, D)</li><li>(block_nums, block_size, KV_N, D)</li></ul> | × |
| cmp_kv | Tensor | 可选 | 压缩量化KV输入，Key和Value共享同一份数据。 | fp8_e4m3 | ND | <ul><li>(B, KV_S, KV_N, D)</li><li>(B, KV_N, KV_S, D)</li><li>(KV_T, KV_N, D)</li><li>(block_nums, block_size, KV_N, D)</li></ul> | × |
| ori_sparse_indices | Tensor | 可选 | 原始KV稀疏索引，无效位置填-1 | int32 | ND | <ul><li>(Q_T, KV_N, K)</li><li>(B, Q_S, KV_N, K)</li></ul> | × |
| cmp_sparse_indices | Tensor | 可选 | 压缩KV TopK索引，无效位置填-1 | int32 | ND | <ul><li>(Q_T, KV_N, K)</li><li>(B, Q_S, KV_N, K)</li></ul> | × |
| ori_block_table | Tensor | 可选 | PageAttention场景下ori_kv使用的block映射表 | int32 | ND | (B, Sori_max/block_size1) | × |
| cmp_block_table | Tensor | 可选 | PageAttention场景下cmp_kv使用的block映射表 | int32 | ND | (B, Scmp_max/block_size2) | × |
| cu_seqlens_q | Tensor | 可选 | 累积序列长度，用于处理变长序列，第一个元素必须为0 | int32 | ND | (B+1,) | × |
| cu_seqlens_ori_kv | Tensor | 可选 | 累积序列长度，用于处理变长序列，第一个元素必须为0 | int32 | ND | (B+1,) | × |
| cu_seqlens_cmp_kv | Tensor | 可选 | 累积序列长度，用于处理变长序列，第一个元素必须为0 | int32 | ND | (B+1,) | × |
| seqused_q | Tensor | 可选 | 指定每batch中实际使用的序列长度，截断冗余运算 | int32 | ND | (B,) | × |
| seqused_ori_kv | Tensor | 可选 | 指定每batch中实际使用的序列长度，截断冗余运算 | int32 | ND | (B,) | × |
| seqused_cmp_kv | Tensor | 可选 | 指定每batch中实际使用的序列长度，截断冗余运算 | int32 | ND | (B,) | × |
| cmp_residual_kv | Tensor | 可选 | 不同batch中cmp_kv压缩后序列长度的余数，配合cmp_ratio实现cmp_kv部分的mask和负载计算 | int32 | ND | (B,) | × |
| ori_topk_length | Tensor | 可选 | 参数ori_kv进行稀疏计算时，实际参与运算的元素数。 | int32 | ND | <ul><li>(B, Q_S, KV_N)</li><li>(Q_T, KV_N)</li></ul> | × |
| cmp_topk_length | Tensor | 可选 | 参数cmp_kv进行稀疏计算时，实际参与运算的元素数。 | int32 | ND | <ul><li>(B, Q_S, KV_N)</li><li>(Q_T, KV_N)</li></ul> | × |
| sinks | Tensor | 可选 | attention sinks输入 | float32 | ND | (Q_N,) | × |
| metadata | Tensor | 必选 | mixed_quant_sparse_flash_mla_metadata生成的任务切分结果，传入后可优化调度 | int32 | ND | (1024,) | × |
| rope_head_dim | int | 可选 | rope头的维度。默认值为64 | int32 | - | - | - |
| softmax_scale | float | 可选 | 可显式设置缩放因子，覆盖默认计算。默认值为1.0 | float32 | - | - | - |
| cmp_ratio | int | 可选 | 表示cmp_kv相对于压缩前KV长度的压缩倍率，用于恢复cmp侧mask使用的压缩前KV长度。默认值为1 | int32 | - | - | - |
| ori_mask_mode | int | 可选 | 表示q和ori_kv计算的mask模式。0：No mask。3：rightDownCausal模式。4：sliding window模式。默认值为0 | int32 | - | - | - |
| cmp_mask_mode | int | 可选 | 表示q和cmp_kv计算的mask模式。0：No mask。3：rightDownCausal模式。默认值为0 | int32 | - | - | - |
| ori_win_left | int | 可选 | 表示q和ori_kv计算中q对过去token计算的数量，-1表示无穷大。默认值为-1 | int32 | - | - | - |
| ori_win_right | int | 可选 | 表示q和ori_kv计算中q对未来token计算的数量，-1表示无穷大。默认值为-1 | int32 | - | - | - |
| layout_q | string | 可选 | 定义输入q张量的布局格式 | string | - | - | - |
| layout_kv | string | 可选 | 定义输入k和v张量的布局格式 | string | - | - | - |
| topk_value_mode | int | 可选 | TopK索引取值模式。默认值为1 | int32 | - | - | - |
| return_softmax_lse | bool | 可选 | 是否返回softmax的lse结果。默认值为False | bool | - | - | - |

## 返回值说明

### mixed_quant_sparse_flash_mla_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| metadata | Tensor | 必选 | mixed_quant_sparse_flash_mla的任务切分数据 | int32 | ND | (1024,) | × |

### mixed_quant_sparse_flash_mla

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| attention_out | Tensor | 必选 | mixed_quant_sparse_flash_mla的计算输出 | bfloat16 | ND | <ul><li>(B, Q_N, Q_S, D)</li><li>(Q_T, Q_N, D)</li></ul> | × |
| softmax_lse | Tensor | 可选 | softmax的lse结果 | float32 | ND | <ul><li>(B, KV_N, Q_S, Q_N/KV_N)</li><li>(KV_N, Q_T, Q_N/KV_N)</li></ul> | × |

**说明**
-   attention_out：Tensor类型，公式中的输出，数据类型支持bfloat16。数据格式支持ND。限制：该输出参数的shape与入参q的shape保持一致，dtype与q一致。
-   softmax_lse：Tensor类型，对query乘key的结果先取max得到softmax_max，query乘key的结果减去softmax_max后取exp再取sum得到softmax_sum，最后对softmax_sum取log再加上softmax_max得到的结果。return_softmax_lse=False时返回shape为[1]的值为0的Tensor；return_softmax_lse=True时返回FLOAT32的log-sum-exp结果。

## 约束说明

- 声明
  - 参数cu_seqlens_q、cu_seqlens_ori_kv、cu_seqlens_cmp_kv、seqused_q、seqused_ori_kv、seqused_cmp_kv、cmp_residual_kv、ori_block_table、cmp_block_table、等输入属于tensor。由于算子在Tiling阶段无法获取tensor的具体数值，tiling侧不对值进行校验，正确性需要用户自行保证。若上述参数传入非法值，会触发未定义行为（精度问题、非法内存访问导致的程序崩溃等）。
  - mixed_quant_sparse_flash_mla_metadata和mixed_quant_sparse_flash_mla的入参在调用时应该保持一致。由于算子分为两个接口分段调用，算子无法自行校验，正确性需要由用户自行保证。若接口传入参数不一致，会发生未定义行为（精度问题、非法内存访问导致的程序崩溃等）。

### 特性参数组

|      特性参数组      |     参数字段名称     |    字段分组    |  字段类型  |
| :-------------------: | :-------------------: | :-------------: | :--------: |
|      公共参数组      |         q         |      INPUT      |   Tensor   |
|                      |      quant_mode      |      INPUT      |    int    |
|                      |      ori_kv      |      INPUT(OPTIONAL)      |   Tensor   |
|                      |      cmp_kv      |      INPUT(OPTIONAL)      |   Tensor   |
|                      |      metadata      |       INPUT(OPTIONAL)      |   Tensor   |
|                      |      rope_head_dim      |      ATTR(OPTIONAL)      |    int    |
|                      |      softmax_scale      |      ATTR(OPTIONAL)      |   float   |
|                      |      layout_q      |      ATTR(OPTIONAL)      |   string   |
|                      |      layout_kv      |      ATTR(OPTIONAL)      |   string   |
|                      |      attention_out      |      OUTPUT      |   Tensor   |
|      Mask参数组      |      ori_mask_mode      |      ATTR(OPTIONAL)      |    int    |
|                      |      cmp_mask_mode      |      ATTR(OPTIONAL)      |    int    |
|                      |      ori_win_left      |      ATTR(OPTIONAL)      |    int    |
|                      |      ori_win_right      |      ATTR(OPTIONAL)      |    int    |
|   SeqLens参数组   |      cu_seqlens_q      |      INPUT(OPTIONAL)      |   Tensor   |
|                      |      cu_seqlens_ori_kv      |      INPUT(OPTIONAL)      |   Tensor   |
|                      |      cu_seqlens_cmp_kv      |      INPUT(OPTIONAL)      |   Tensor   |
|                      |      seqused_q      |      INPUT(OPTIONAL)      |   Tensor   |
|                      |      seqused_ori_kv      |      INPUT(OPTIONAL)      |   Tensor   |
|                      |      seqused_cmp_kv      |      INPUT(OPTIONAL)      |   Tensor   |
|   稀疏压缩参数组    |      cmp_ratio      |      ATTR(OPTIONAL)      |    int    |
|                      |      cmp_residual_kv      |      INPUT(OPTIONAL)      |   Tensor   |
|                      |      ori_sparse_indices      |      INPUT(OPTIONAL)      |   Tensor   |
|                      |      cmp_sparse_indices      |      INPUT(OPTIONAL)      |   Tensor   |
|                      |      ori_topk_length      |      INPUT(OPTIONAL)      |   Tensor   |
|                      |      cmp_topk_length      |      INPUT(OPTIONAL)      |   Tensor   |
|                      |      topk_value_mode      |      ATTR(OPTIONAL)      |    int    |
| Paged Attention参数组 |      ori_block_table      |      INPUT(OPTIONAL)      |   Tensor   |
|                      |      cmp_block_table      |      INPUT(OPTIONAL)      |   Tensor   |
|   Sinks参数组   |      sinks      |      INPUT(OPTIONAL)      |   Tensor   |
|   SoftmaxLse参数组   |      return_softmax_lse      |      ATTR(OPTIONAL)      |    bool    |
|                      |      softmax_lse      |      OUTPUT(OPTIONAL)      |   Tensor   |

### 基准信息说明

资料约束中，常见字段释义如下：

|    命名    |                            含义                            |
| :---------: | :---------------------------------------------------------: |
|      B      |                Batch，表示输入样本批量大小                |
|     Q_S     |      输入q tensor的序列长度，对应q shape中的S      |
|    KV_S    |  输入ori_kv/cmp_kv tensor的序列长度，对应ori_kv shape中的S  |
|     Q_N     |        输入q tensor的头数，对应q shape中的N        |
|    KV_N    |    输入ori_kv/cmp_kv tensor的头数，对应ori_kv/cmp_kv shape中的N    |
|      Q_D      |          输入q tensor以及输出attention_out隐藏层最小的单元尺寸headdim         |
|      KV_D      |          输入ori_kv/cmp_kv tensor隐藏层最小的单元尺寸headdim         |
|     Q_T     |          输入q tensor所有batch序列长度的累加和          |
|     KV_T    |          输入ori_kv/cmp_kv tensor所有batch序列长度的累加和          |
|      K1      |           ori_sparse_indices中TopK选出的token个数         |
|      K2      |           cmp_sparse_indices中TopK选出的token个数         |

### 计算模式说明
|    命名    |                            必传参数                           |
| :---------: | :---------------------------------------------------------: |
|      SWA      |                ori_kv                |
|      HCA      |                ori_kv、cmp_kv                |
|      CSA      |                ori_kv、cmp_kv、cmp_sparse_indices                |
|      ALL_CSA      |                ori_kv、cmp_kv、cmp_sparse_indices、ori_sparse_indices                |

### 参数组约束

#### 公共参数组
- 入参为空的场景处理：
    - 空Tensor指必选输入、计算模式（SWA\CFA\HCA）下必传输入和输出的shape size为0，即有任意轴为0。
    - 触发空tensor的用例将全部拦截报错。


- q、ori_kv、cmp_kv、attention_out校验
<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 147px">
<col style="width: 232px">
<col style="width: 232px">
<col style="width: 293px">
<col style="width: 185px">
</colgroup>
<thead>
<tr>
    <th>参数</th>
    <th>单参数校验</th>
    <th>存在性校验</th>
    <th>一致性校验</th>
    <th>特性交叉校验</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>q</td>
        <td>
            <ul>
                <li>tensor_type支持BFLOAT16</li>
                <li>BSND -> (B, Q_S, Q_N, Q_D)</li>
                <li>TND -> (Q_T, Q_N, Q_D)</li>
            </ul>
        </td>
        <td rowspan="3">
            必须存在
        </td>
        <td rowspan="4">
            <ul>
                <li>q、attention_out的dtype、shape需相同</li>
                <li>若cmp_kv传入，ori_kv与cmp_kv的shape需一致</li>
                <li>layout_kv不为PA_BBND时，layout_q和layout_kv需保持一致</li>
                <li>layout_kv为PA_BBND时，layout_q可为BSND或TND</li>
            </ul>
        </td>
        <td rowspan="4">
            轴校验：
            <ul>
                <li>B > 0</li>
                <li>Q_T > 0</li>
                <li>KV_T > 0</li>
                <li>Q_N > 0</li>
                <li>KV_N = 1</li>
                <li>Q_S > 0</li>
                <li>KV_S > 0</li>
                <li>Q_D = 512</li>
                <li>quant_mode=1时 KV_D=608，quant_mode=2时 KV_D=584</li>
                <li>Q_N % KV_N == 0且Q_N / KV_N > 0</li>
                <li>block_nums1 > 0</li>
                <li>block_nums2 > 0</li>
                <li>block_size1 > 0</li>
                <li>block_size2 > 0</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>ori_kv</td>
        <td>
            <ul>
                <li>由rope、nope、scale、padding按quant_mode拼接而成的packed格式</li>
                <li>BSND -> (B, KV_S, KV_N, KV_D)</li>
                <li>TND -> (KV_T, KV_N, KV_D)</li>
                <li>PA_BBND -> (block_nums1, block_size1, KV_N, KV_D)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>attention_out</td>
        <td>
            <ul>
                <li>data_type支持BFLOAT16</li>
                <li>BSND -> (B, KV_S, KV_N, KV_D)</li>
                <li>TND -> (KV_T, KV_N, KV_D)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>cmp_kv</td>
        <td>
            <ul>
                <li>由rope、nope、scale、padding按quant_mode拼接而成的packed格式</li>
                <li>BSND -> (B, KV_S, KV_N, KV_D)</li>
                <li>TND -> (KV_T, KV_N, KV_D)</li>
                <li>PA_BBND -> (block_nums2, block_size2, KV_N, KV_D)</li>
            </ul>
        </td>
        <td>
            SWA计算模式下不传入；CSA/HCA计算模式下必须存在
        </td>
    </tr>
</tbody>
</table>
layout匹配关系表：
<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 400px">
<col style="width: 400px">
<col style="width: 400px">
<col style="width: 400px">
</colgroup>
<thead>
<tr>
    <th>layout_q</th>
    <th>layout_kv</th>
    <th>layout_out</th>
    <th>layout_softmax_lse</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>BSND</td>
        <td>
          <li>BSND</li>
          <li>PA_BBND</li>
        </td>
        <td>BSND</td>
        <td>(B, KV_N, Q_S, Q_N/KV_N)</td>
    </tr>
    <tr>
        <td>TND</td>
        <td>
          <li>TND</li>
          <li>PA_BBND</li>
        </td>
        <td>TND</td>
        <td>(KV_N, Q_T, Q_N/KV_N)</td>
    </tr>
</tbody>
</table>

metadata校验
<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>metadata</td>
            <td>
                <ul>
                    <li>tensor_type仅支持INT32</li>
                    <li>shape由flash_attn_metadata动态计算</li>
                    <li>当前不支持不传入，未传入将发出拦截报警</li>
                </ul>
            </td>
            <td>可选参数</td>
            <td>无</td>
            <td>传入时需与flash_attn_metadata生成的结果一致</td>
        </tr>
    </tbody>
</table>

#### mask_mode参数组
<ul>
    <li>ori_mask_mode/cmp_mask_mode=0，全计算模式（默认值）</li>
    <li>ori_mask_mode/cmp_mask_mode=3，Causal模式</li>
    <li>ori_mask_mode=4，SlidingWindow模式</li>
</ul>

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 200px">
        <col style="width: 100px">
        <col style="width: 200px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>ori_mask_mode</td>
            <td>
                <ul>
                    <li>data_type支持INT</li>
                    <li>支持输入范围仅为0、3、4，默认值为0</li>
                </ul>
            </td>
            <td>
                可选输入，如果不传该参数，默认值为0
            </td>
            <td>
                <ul>
                    无
                </ul>
            </td>
            <td>
                <ul>
                    <li>只有ALL_CSA情景下，cmp_mask_mode为0和ori_mask_mode必须为0</li>
                    <li>SWA场景下，ori_mask_mode为3、4</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cmp_mask_mode</td>
            <td>
                <ul>
                    <li>data_type支持INT</li>
                    <li>支持输入范围仅为0、3，默认值为0</li>
                </ul>
            </td>
            <td>
                可选输入，如果不传该参数，默认值为0
            </td>
            <td>
                <ul>
                    无
                </ul>
            </td>
            <td>
                    <li>只有ALL_CSA情景下，cmp_mask_mode为0和ori_mask_mode必须为0</li>
                    <li>SWA情景下，cmp_mask_mode必须为默认值0</li>
                    <li>当ori_mask_mode为3时，cmp_mask_mode必须为3</li>
                    <li>当ori_mask_mode为4时，cmp_mask_mode必须为3</li>
            </td>
        </tr>
        <tr>
            <td>ori_win_left</td>
            <td>
                <ul>
                    <li>data_type支持INT</li>
                    <li>支持输入范围仅为-1，>=0</li>
                </ul>
            </td>
            <td>
                可选输入，如果不传该参数，默认值为-1
            </td>
            <td>
                <ul>
                    无
                </ul>
            </td>
            <td>
                    <li>只有ori_mask_mode为4时，ori_win_left可以>=0</li>
            </td>
        </tr>
        <tr>
            <td>ori_win_right</td>
            <td>
                <ul>
                    <li>data_type支持INT</li>
                    <li>支持输入范围仅为-1，>=0</li>
                </ul>
            </td>
            <td>
                可选输入，如果不传该参数，默认值为-1
            </td>
            <td>
                <ul>
                    无
                </ul>
            </td>
            <td>
                    <li>只有ori_mask_mode为4时，ori_win_right可以>=0</li>
            </td>
        </tr>
    </tbody>
</table>


#### SeqLens参数组

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>seqused_q</td>
            <td >
                <ul>
                    <li>tensor_type支持INT32</li>
                    <li>tensor_shape为(B,)</li>
                    <li>仅支持非负整数</li>
                    <li>seqused_q中的值需小于等于Q_S</li>
                </ul>
            </td>
            <td>当layout_q为BSND时，必须传入</td>
            <td>无</td>
            <td>无</td>
        </tr>
        <tr>
            <td>seqused_ori_kv</td>
            <td >
                <ul>
                    <li>tensor_type支持INT32</li>
                    <li>tensor_shape为(B,)</li>
                    <li>仅支持非负整数</li>
                    <li>seqused_ori_kv中的值需小于等于KV_S</li>
                </ul>
            </td>
            <td >
                <ul>
                    <li>当layout_kv为BSND时，必须传入(待确认，可以不传，S为每个batch)</li>
                    <li>当layout_kv为PA_BBND时，必须传入</li>
                </ul>
            </td>
            <td>无</td>
            <td>无</td>
        </tr>
        <tr>
            <td>seqused_cmp_kv</td>
            <td >
                <ul>
                    <li>tensor_type支持INT32</li>
                    <li>tensor_shape为(B,)</li>
                    <li>仅支持非负整数</li>
                    <li>seqused_ori_kv中的值需小于等于KV_S</li>
                </ul>
            </td>
            <td>当layout_kv为BSND时且非SWA场景下，必须传入</td>
            <td>无</td>
            <td>无</td>
        </tr>
        <tr>
            <td>cu_seqlens_q</td>
            <td>
                <ul>
                    <li>tensor_type支持INT32</li>
                    <li>tensor_shape为(B+1,)</li>
                    <li>值仅支持非负整数</li>
                    <li>其值应非递减（大于等于前一个值）排列，第一个元素为0且最后一个元素等于Q_T</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>当layout_q为TND时，必须传入</li>
                    <li>当layout_q不为TND时，不支持传入</li>
                </ul>
            </td>
            <td>无</td>
            <td>无</td>
        </tr>
        <tr>
            <td>cu_seqlens_ori_kv</td>
            <td>
                <ul>
                    <li>tensor_type支持INT32</li>
                    <li>tensor_shape为(B+1,)</li>
                    <li>值仅支持非负整数</li>
                    <li>其值应非递减（大于等于前一个值）排列，第一个元素为0且最后一个元素等于KV_T</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>当layout_kv为TND时，必须传入</li>
                    <li>当layout_kv不为TND时，不支持传入</li>
                </ul>
            </td>
            <td>无</td>
            <td>无</td>
        </tr>
        <tr>
            <td>cu_seqlens_cmp_kv</td>
            <td>
                <ul>
                    <li>tensor_type支持INT32</li>
                    <li>tensor_shape为(B+1,)</li>
                    <li>值仅支持非负整数</li>
                    <li>其值应非递减（大于等于前一个值）排列，第一个元素为0且最后一个元素等于KV_T</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>当layout_kv为TND时，必须传入</li>
                    <li>当layout_kv不为TND时，不支持传入</li>
                </ul>
            </td>
            <td>无</td>
            <td>无</td>
        </tr>
    </tbody>
</table>

#### 稀疏压缩参数组

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>cmp_ratio</td>
            <td>
                <ul>
                    <li>data_type支持INT</li>
                    <li>表示cmp_kv相对于压缩前KV长度的压缩倍率，需大于0</li>
                    <li>默认值为1</li>
                </ul>
            </td>
            <td>可选属性，默认值为1</td>
            <td>无</td>
            <td>
                <ul>
                    <li>配合cmp_residual_kv使用，用于恢复cmp侧mask使用的压缩前KV长度</li>
                    <li>在SWA场景，仅支持默认值1。</li>
                    <li>在CSA\HCA场景，支持1-128。</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cmp_residual_kv</td>
            <td>
                <ul>
                    <li>tensor_type支持INT32</li>
                    <li>tensor_shape为(B,)</li>
                    <li>值仅支持非负整数</li>
                </ul>
            </td>
            <td>无</td>
            <td>
                <ul>
                    <li>配合cmp_ratio使用，用于恢复压缩前KV长度</li>
                </ul>
            </td>
            <td>无</td>
        </tr>
        <tr>
            <td>ori_sparse_indices</td>
            <td>
                <ul>
                    <li>tensor_type支持INT32</li>
                    <li>tensor_shape为(Q_T, KV_N, K1)或(B, Q_S, KV_N, K1)</li>
                    <li>无效位置填-1，其余为非负整数</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>可选参数</li>
                    <li>在CSA情景下，该参数必传。</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>当layout_q为TND时，该tensor_shape为(Q_T, KV_N, K1)</li>
                    <li>当layout_q为BSND时，该tensor_shape为(B, Q_S, KV_N, K1)</li>
                </ul>
            </td>
            <td>无</td>
        </tr>
        <tr>
            <td>cmp_sparse_indices</td>
            <td>
                <ul>
                    <li>tensor_type支持INT32</li>
                    <li>tensor_shape为(Q_T, KV_N, K2)或(B, Q_S, KV_N, K2)</li>
                    <li>无效位置填-1，其余为非负整数</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>可选参数</li>
                    <li>在CSA情景下，该参数必传。</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>当layout_q为TND时，该tensor_shape为(Q_T, KV_N, K2)</li>
                    <li>当layout_q为BSND时，该tensor_shape为(B, Q_S, KV_N, K2)</li>
                </ul>
            </td>
            <td>无</td>
        </tr>
        <tr>
            <td>ori_topk_length</td>
            <td rowspan="2">
                <ul>
                    <li>tensor_type支持INT32</li>
                    <li>tensor_shape为(B, Q_S, KV_N)或(Q_T, KV_N)</li>
                </ul>
            </td>
            <td rowspan="2">可选参数</td>
            <td rowspan="2">
                <ul>
                    <li>当layout_q为TND时，该tensor_shape为(Q_T, KV_N)</li>
                    <li>当layout_q为BSND时，该tensor_shape为(B, Q_S, KV_N)</li>
                </ul>
            </td>
            <td rowspan="2">无</td>
        </tr>
        <tr>
            <td>cmp_topk_length</td>
        </tr>
        <tr>
            <td>topk_value_mode</td>
            <td>
                <ul>
                    <li>data_type支持INT</li>
                    <li>TopK索引取值模式，默认值为1</li>
                </ul>
            </td>
            <td>可选属性，默认值为1</td>
            <td>无</td>
            <td>无</td>
        </tr>
    </tbody>
</table>

#### Paged Attention参数组
当ori_block_table不为空时，开启Paged Attention
<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>ori_block_table</td>
            <td>
                <ul>
                    <li>tensor_type仅支持INT32</li>
                    <li>tensor_shape为(B, Sori_max/block_size1)</li>
                    <li>值只能为正整数</li>
                </ul>
            </td>
            <td>可选参数</td>
            <td>当layout_kv为PA_BBND时，该参数必传。</td>
            <td>
                <ul>
                    <li>ori_block_table存在时，必须传入seqused_ori_kv</li>
                    <li>PagedAttention开启情况下，block_table必须不为空</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cmp_block_table</td>
            <td>
                <ul>
                    <li>tensor_type仅支持INT32</li>
                    <li>tensor_shape为(B, Scmp_max/block_size2)</li>
                    <li>值只能为正整数</li>
                </ul>
            </td>
            <td>可选参数</td>
            <td>无</td>
            <td>
                <ul>
                    <li>cmp_block_table存在时，必须传入seqused_cmp_kv</li>
                    <li>PagedAttention开启情况下，block_table必须不为空</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>

#### Sinks参数组

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>sinks</td>
            <td>
                <ul>
                    <li>tensor_type支持FLOAT。</li>
                    <li>tensor_shape为(B, )。</li>
                </ul>
            </td>
            <td> 可选参数 </td>
            <td> 无 </td>
            <td> 暂不支持 </td>
        </tr>
    </tbody>
</table>

#### SoftmaxLSE参数组

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>return_softmax_lse</td>
            <td>
                <ul>
                    <li>data_type仅支持BOOL</li>
                    <li>值仅支持true和false，true代表开启softmax_lse，false代表关闭softmax_lse</li>
                </ul>
            </td>
            <td>可选属性，默认值为false</td>
            <td rowspan="2">
                <ul>
                    <li>当return_softmax_lse为false时，输出shape为[1]的值为0的Tensor(待确认)</li>
                    <li>当return_softmax_lse为true时，输出shape见layout匹配关系表</li>
                </ul>
            </td>
            <td>无</td>
        </tr>
        <tr>
            <td>softmax_lse</td>
            <td>
                <ul>
                    <li>data_type仅支持FLOAT。</li>
                </ul>
            </td>
            <td>无</td>
            <td>无</td>
        </tr>
    </tbody>
</table>