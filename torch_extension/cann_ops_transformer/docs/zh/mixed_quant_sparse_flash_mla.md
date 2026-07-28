# mixed_quant_sparse_flash_mla

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明

- 接口功能：

  `mixed_quant_sparse_flash_mla`是基于`torch_npu`的`cann_ops_transformer`扩展接口，用于调用`MixedQuantSparseFlashMla`算子完成量化和稀疏场景下的MLA（Multi-head Latent Attention）注意力计算，训练推理归一化。该算子的三种典型场景：

  - **SWA（Sliding Window Attention）**：仅传入`ori_kv`，对原始KV做滑动窗口注意力。
  - **CSA（Compressed Sparse Attention）**：同时传入`ori_kv`、`cmp_kv`和`cmp_sparse_indices`，对原始KV窗口和topK选择出的压缩KV共同做注意力。
  - **HCA（Heavily Compressed Attention）**：同时传入`ori_kv`和`cmp_kv`，对原始KV窗口和连续压缩KV段共同做注意力。

  `mixed_quant_sparse_flash_mla_metadata`是`mixed_quant_sparse_flash_mla`的分核信息，在主算子执行前生成。当前版本主算子必须传入该metadata。典型调用流程如下：

  1. 根据调用场景准备对应输入。
  2. 调用`mixed_quant_sparse_flash_mla_metadata`生成`metadata`，作为`mixed_quant_sparse_flash_mla`的入参。
  3. 调用`mixed_quant_sparse_flash_mla`，生成计算结果。

- 计算公式：

  mixed_quant_sparse_flash_mla采用MLA对KV共享输入的稀疏注意力进行计算，其原理是对输入的KV进行选择性压缩与量化处理，再将Query与拼接后的KV计算结果通过Softmax得到注意力权重。

  MLA的计算公式一般定义如下，其中$\tilde{K}=\tilde{V}$为基于入参控制的实际参与计算的KV，由`ori_kv`的滑动窗口部分和`cmp_kv`的压缩部分共同组成，实际参与计算的KV范围由`cmp_ratio`、`ori_mask_mode`、`cmp_mask_mode`、`ori_win_left`、`ori_win_right`以及`cmp_sparse_indices`决定。

  $$
  O = \text{softmax}(Q\tilde{K}^T \cdot \text{softmax\_scale}) \tilde{V}
  $$

## 函数原型

调用mixed_quant_sparse_flash_mla接口之前，请先调用前置接口mixed_quant_sparse_flash_mla_metadata。

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
) -> tensor
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
) -> (tensor, tensor)
```



## 参数说明

### 常见字段释义

|    命名    |                            含义                            |
| :---------: | :---------------------------------------------------------: |
|      B      |                表示输入样本batch大小                |
|     Q_S     |      输入q的序列长度      |
|    ORI_KV_S    |  输入ori_kv的序列长度  |
|    CMP_KV_S    |  输入cmp_kv的序列长度  |
|     Q_N     |        输入q的头数        |
|    KV_N    |    输入ori_kv/cmp_kv的头数    |
|      Q_D      |          输入q的注意力头的维度         |
|      KV_D      |          输入ori_kv/cmp_kv的注意力头的维度         |
|     Q_T     |          输入q所有batch序列长度的累加和          |
|     ORI_KV_T    |          输入ori_kv所有batch序列长度的累加和          |
|     CMP_KV_T    |          输入cmp_kv所有batch序列长度的累加和          |
|      ORI_KV_K      |           输入ori_sparse_indices中topK选出的token个数         |
|      CMP_KV_K      |           输入cmp_sparse_indices中topK选出的token个数         |
|      ori_kv_s_max      |           输入ori_kv的最大序列长度         |
|      cmp_kv_s_max      |           输入cmp_kv的最大序列长度         |
|      ori_kv_block_size      |           输入ori_kv在PagedAttention场景下的block大小         |
|      cmp_kv_block_size      |           输入cmp_kv在PagedAttention场景下的block大小         |
|      ori_kv_block_nums      |           输入ori_kv在PagedAttention场景下的block数量         |
|      cmp_kv_block_nums      |           输入cmp_kv在PagedAttention场景下的block数量         |

### mixed_quant_sparse_flash_mla_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度
| :--- | :--- | :--- | :--- | :--- | :--- | :---
| num_heads_q | int | 必选 | 表示q 头数 | int32 | - | -
| num_heads_kv | int | 必选 | 表示ori_kv/cmp_kv 头数 | int32 | - | -
| head_dim | int | 必选 | 表示每个注意力头的维度 | int32 | - | -
| quant_mode | int | 必选 | 表示量化模式，当前支持1和2 | int32 | - | -
| cu_seqlens_q | tensor | 可选 | 表示处理变长序列的累积序列长度 | int32 | ND | (B+1,)
| cu_seqlens_ori_kv | tensor | 可选 | 表示处理变长序列的累积序列长度 | int32 | ND | (B+1,)
| cu_seqlens_cmp_kv | tensor | 可选 | 表示处理变长序列的累积序列长度 | int32 | ND | (B+1,)
| seqused_q | tensor | 可选 | 表示每batch中实际参与运算的序列长度 | int32 | ND | (B,)
| seqused_ori_kv | tensor | 可选 | 表示每batch中实际参与运算的序列长度 | int32 | ND | (B,)
| seqused_cmp_kv | tensor | 可选 | 表示每batch中实际参与运算的序列长度 | int32 | ND | (B,)
| cmp_residual_kv | tensor | 可选 | 表示每batch中cmp_kv压缩后序列长度的余数 | int32 | ND | (B,)
| ori_topk_length | tensor | 可选 | 表示ori_sparse_indices实际参与计算的长度 | int32 | ND | <ul><li>(B, Q_S, KV_N)</li><li>(Q_T, KV_N)</li></ul>
| cmp_topk_length | tensor | 可选 | 表示cmp_sparse_indices实际参与计算的长度 | int32 | ND | <ul><li>(B, Q_S, KV_N)</li><li>(Q_T, KV_N)</li></ul>
| batch_size | int | 可选 | 表示batch大小 | int32 | - | -
| max_seqlen_q | int | 可选 | 表示查询q序列的长度上限 | int32 | - | -
| max_seqlen_ori_kv | int | 可选 | 表示ori_kv序列的长度上限 | int32 | - | -
| max_seqlen_cmp_kv | int | 可选 | 表示cmp_kv序列的长度上限 | int32 | - | -
| ori_topk | int | 可选 | 表示ori_kv中筛选出的关键稀疏token的个数。0表示非稀疏场景。默认值为0 | int32 | - | -
| cmp_topk | int | 可选 | 表示cmp_kv中筛选出的关键稀疏token的个数。0表示非稀疏场景。默认值为0 | int32 | - | -
| rope_head_dim | int | 可选 | 表示rope头的维度。默认值为64 | int32 | - | -
| cmp_ratio | int | 可选 | 表示对cmp_kv的压缩率。默认值为1 | int32 | - | -
| ori_mask_mode | int | 可选 | 表示q和ori_kv计算的mask模式。0：No mask。3：rightDownCausal模式。4：sliding window模式。默认值为0 | int32 | - | -
| cmp_mask_mode | int | 可选 | 表示q和cmp_kv计算的mask模式。0：No mask。3：rightDownCausal模式。默认值为0 | int32 | - | -
| ori_win_left | int | 可选 | 表示q和ori_kv计算中q对过去token计算的数量，-1表示无穷大，即全部参与运算。默认值为-1 | int32 | - | -
| ori_win_right | int | 可选 | 表示q和ori_kv计算中q对未来token计算的数量，-1表示无穷大，即全部参与运算。默认值为-1 | int32 | - | -
| layout_q | string | 可选 | 表示输入q的布局格式，默认值为BSND | string | - | -
| layout_kv | string | 可选 | 表示输入ori_kv/cmp_kv的布局格式，默认值为BSND | string | - | -
| has_ori_kv | bool | 可选 | 表示是否含有ori_kv。默认值为True | bool | - | -
| has_cmp_kv | bool | 可选 | 表示是否含有cmp_kv。默认值为True | bool | - | -

### mixed_quant_sparse_flash_mla

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度
| :--- | :--- | :--- | :--- | :--- | :--- | :---
| q | tensor | 必选 | 表示公式中的q | bfloat16 | ND |  <ul><li>(B, Q_S, Q_N, D)</li><li>(Q_T, Q_N, D)</li></ul>
| quant_mode | int | 必选 | 表示量化模式。当前仅支持1和2 | int32 | - | -
| ori_kv | tensor | 可选 | 表示原始量化KV输入，Key和Value共享同一份数据 | fp8_e4m3 | ND | <ul><li>(B, ORI_KV_S, KV_N, D)</li><li>(ORI_KV_T, KV_N, D)</li><li>(ori_kv_block_nums, ori_kv_block_size, KV_N, D)</li></ul>
| cmp_kv | tensor | 可选 | 表示压缩量化KV输入，Key和Value共享同一份数据 | fp8_e4m3 | ND | <ul><li>(B, CMP_KV_S, KV_N, D)</li><li>(CMP_KV_T, KV_N, D)</li><li>(cmp_kv_block_nums, cmp_kv_block_size, KV_N, D)</li></ul>
| ori_sparse_indices | tensor | 可选 | 表示原始KV topK索引，无效位置填-1 | int32 | ND | <ul><li>(Q_T, KV_N, ORI_KV_K)</li><li>(B, Q_S, KV_N, ORI_KV_K)</li></ul>
| cmp_sparse_indices | tensor | 可选 | 表示压缩KV topK索引，无效位置填-1 | int32 | ND | <ul><li>(Q_T, KV_N, CMP_KV_K)</li><li>(B, Q_S, KV_N, CMP_KV_K)</li></ul>
| ori_block_table | tensor | 可选 | 表示PageAttention场景下ori_kv使用的block映射表 | int32 | ND | (B, ceil(ori_kv_s_max/ori_kv_block_size))
| cmp_block_table | tensor | 可选 | 表示PageAttention场景下cmp_kv使用的block映射表 | int32 | ND | (B, ceil(cmp_kv_s_max/cmp_kv_block_size))
| cu_seqlens_q | tensor | 可选 | 表示处理变长序列的累积序列长度 | int32 | ND | (B+1,)
| cu_seqlens_ori_kv | tensor | 可选 | 表示处理变长序列的累积序列长度 | int32 | ND | (B+1,)
| cu_seqlens_cmp_kv | tensor | 可选 | 表示处理变长序列的累积序列长度 | int32 | ND | (B+1,)
| seqused_q | tensor | 可选 | 表示每batch中实际参与运算的序列长度 | int32 | ND | (B,)
| seqused_ori_kv | tensor | 可选 | 表示每batch中实际参与运算的序列长度 | int32 | ND | (B,)
| seqused_cmp_kv | tensor | 可选 | 表示每batch中实际参与运算的序列长度 | int32 | ND | (B,)
| cmp_residual_kv | tensor | 可选 | 表示m每batch中cmp_kv压缩后序列长度的余数 | int32 | ND | (B,)
| ori_topk_length | tensor | 可选 | 表示ori_sparse_indices实际参与计算的长度 | int32 | ND | <ul><li>(B, Q_S, KV_N)</li><li>(Q_T, KV_N)</li></ul>
| cmp_topk_length | tensor | 可选 | 表示cmp_sparse_indices实际参与计算的长度 | int32 | ND | <ul><li>(B, Q_S, KV_N)</li><li>(Q_T, KV_N)</li></ul>
| sinks | tensor | 可选 | 为各注意力头设置独立可学习虚拟偏移项，从而维持长文本推理时的稳定性 | float32 | ND | (Q_N,)
| metadata | tensor | 可选 | 表示mixed_quant_sparse_flash_mla_metadata生成的分核信息 | int32 | ND | (1024,)
| rope_head_dim | int | 可选 | 表示rope头的维度。默认值为64 | int32 | - | -
| softmax_scale | float | 可选 | 可显式设置缩放因子，覆盖默认计算。默认值为1.0 | float32 | - | -
| cmp_ratio | int | 可选 | 表示cmp_kv相对于压缩前KV长度的压缩倍率，可恢复cmp侧mask使用的压缩前KV长度。默认值为1 | int32 | - | -
| ori_mask_mode | int | 可选 | 表示q和ori_kv计算的mask模式。0：No mask。3：rightDownCausal模式。4：sliding window模式。默认值为0 | int32 | - | -
| cmp_mask_mode | int | 可选 | 表示q和cmp_kv计算的mask模式。0：No mask。3：rightDownCausal模式。默认值为0 | int32 | - | -
| ori_win_left | int | 可选 | 表示q和ori_kv计算中q对历史token计算的数量，-1表示无穷大，即全部参与运算。默认值为-1 | int32 | - | -
| ori_win_right | int | 可选 | 表示q和ori_kv计算中q对未来token计算的数量，-1表示无穷大，即全部参与运算。默认值为-1 | int32 | - | -
| layout_q | string | 可选 | 表示输入q的布局格式，默认值为BSND | string | - | -
| layout_kv | string | 可选 | 表示输入ori_kv/cmp_kv的布局格式，默认值为BSND | string | - | -
| topk_value_mode | int | 可选 | 表示topK索引取值模式。默认值为1 | int32 | - | -
| return_softmax_lse | bool | 可选 | 表示是否返回softmax的lse结果。默认值为False | bool | - | -

## 返回值说明

### mixed_quant_sparse_flash_mla_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度
| :--- | :--- | :--- | :--- | :--- | :--- | :---
| metadata | tensor | 必选 | mixed_quant_sparse_flash_mla的分核信息 | int32 | ND | (1024,)

### mixed_quant_sparse_flash_mla

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度
| :--- | :--- | :--- | :--- | :--- | :--- | :---
| attention_out | tensor | 必选 | mixed_quant_sparse_flash_mla的计算输出 | bfloat16 | ND | <ul><li>(B, Q_N, Q_S, D)</li><li>(Q_T, Q_N, D)</li></ul>
| softmax_lse | tensor | 可选 | softmax的lse结果 | float32 | ND | <ul><li>(B, KV_N, Q_S, Q_N/KV_N)</li><li>(KV_N, Q_T, Q_N/KV_N)</li></ul>

**说明**
-   attention_out：tensor类型，公式中的输出，数据类型支持bfloat16。数据格式支持ND。限制：该输出参数的shape与入参q的shape保持一致，dtype与q一致。
-   softmax_lse：tensor类型，对query乘key的结果先取max得到softmax_max，query乘key的结果减去softmax_max后取exp再取sum得到softmax_sum，最后对softmax_sum取log再加上softmax_max得到的结果。return_softmax_lse=False时返回shape为[1]的值为0的tensor；return_softmax_lse=True时返回float32的log-sum-exp结果。
- cmp_residual_kv每个batch的值需要小于cmp_ratio。
- cu_seqlens_q、cu_seqlens_ori_kv、cu_seqlens_cmp_kv须满足首元素为0，且序列整体呈非递减排列，即任一元素不小于其前一个元素。
- 当layout_kv为PA_BBND时，ori_kv和cmp_kv支持0轴非连续。

## 约束说明

- 声明
  - 参数cu_seqlens_q、cu_seqlens_ori_kv、cu_seqlens_cmp_kv、seqused_q、seqused_ori_kv、seqused_cmp_kv、cmp_residual_kv、ori_block_table、cmp_block_table等输入属于tensor。由于算子在Tiling阶段无法获取tensor的具体数值，tiling侧不对值进行校验，正确性需要用户自行保证。若上述参数传入非法值，会触发未定义行为（精度问题、非法内存访问导致的程序崩溃等）。
  - mixed_quant_sparse_flash_mla_metadata和mixed_quant_sparse_flash_mla的入参在调用时应该保持一致。由于算子分为两个接口分段调用，算子无法自行校验，正确性需要由用户自行保证。若接口传入参数不一致，会发生未定义行为（精度问题、非法内存访问导致的程序崩溃等）。
  - ori_topk_length、cmp_topk_length表示ori/cmp sparse_indices实际参与计算的长度。其值不能大于sparse_indices的最后一维大小，且当seqused_q传入时，topk_length对应有效部分的值需要大于等于0。
  - 当ori_mask_mode/cmp_mask_mode为0时，ORI_KV_K/CMP_KV_K需要大于等于ori_topk_length/cmp_topk_lengthd的最大值。
  - 各参数shape中以相同符号表示的维度，其对应轴的实际数值需保持一致。

### 特性参数组

|      特性参数组      |     参数字段名称     |
| :-------------------: | :-------------------: |
|      公共参数组      | q、quant_mode、ori_kv、cmp_kv、metadata、rope_head_dim、softmax_scale、layout_q、layout_kv、attention_out |
|      Mask参数组      | ori_mask_mode、cmp_mask_mode、ori_win_left、ori_win_right |
|   SeqLens参数组   | cu_seqlens_q、cu_seqlens_ori_kv、cu_seqlens_cmp_kv、seqused_q、seqused_ori_kv、seqused_cmp_kv |
|   稀疏压缩参数组    | cmp_ratio、cmp_residual_kv、ori_sparse_indices、cmp_sparse_indices、ori_topk_length、cmp_topk_length、topk_value_mode |
| Paged Attention参数组 | ori_block_table、cmp_block_table |
|   Sinks参数组   | sinks |
|   SoftmaxLse参数组   | return_softmax_lse、softmax_lse |

### 基准信息说明


### 计算模式说明
|    命名    |    常见场景    |                            需传参数                           |
| :---------: | :---------: | :---------------------------------------------------------: |
|      SWA      |     典型场景     |                ori_kv                |
|      HCA      |     典型场景     |                ori_kv、cmp_kv                |
|      CSA      |     典型场景     |                ori_kv、cmp_kv、cmp_sparse_indices                |
|      ORI_SPARSE      |     全稀疏场景     |                ori_kv、ori_sparse_indices|
|      ORI_CMP_SPARSE      |     全稀疏场景     |                ori_kv、cmp_kv、ori_sparse_indices、cmp_sparse_indices|

### 参数组约束

#### 公共参数组
- 入参为空的场景处理：
    - 空tensor指必选输入、某调用场景下下必传输入和输出的shape size为0，即有任意轴为0。
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
                <li>dtype支持bfloat16</li>
                <li>layout_q为BSND时，q的shape为(B, Q_S, Q_N, Q_D)</li>
                <li>layout_q为TND时，q的shape为(Q_T, Q_N, Q_D)</li>
            </ul>
        </td>
        <td>
            必须传入
        </td>
        <td rowspan="4">
            <ul>
                <li>q、attention_out的dtype、shape需相同</li>
                <li>若cmp_kv传入，ori_kv与cmp_kv的dtype需一致</li>
                <li>layout_kv不为PA_BBND时，layout_q和layout_kv需保持一致</li>
                <li>layout_kv为PA_BBND时，layout_q可为BSND或TND</li>
            </ul>
        </td>
        <td rowspan="4">
            轴校验：
            <ul>
                <li>B > 0</li>
                <li>Q_S > 0</li>
                <li>0 < Q_N <= 128</li>
                <li>Q_D = 512</li>
                <li>Q_T > 0</li>
                <li>ORI_KV_S > 0</li>
                <li>CMP_KV_S > 0</li>
                <li>KV_N = 1</li>
                <li>quant_mode=1时 KV_D=608，quant_mode=2时 KV_D=584</li>
                <li>ORI_KV_T > 0</li>
                <li>CMP_KV_T > 0</li>
                <li>ori_kv_block_nums > 0</li>
                <li>cmp_kv_block_nums > 0</li>
                <li>1 <= ori_kv_block_size <= 1024</li>
                <li>1 <= cmp_kv_block_size <= 1024</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>ori_kv</td>
        <td>
            <ul>
                <li>dtype支持fp8_e4m3</li>
                <li>layout_kv为BSND时，ori_kv的shape为(B, ORI_KV_S, KV_N, KV_D)</li>
                <li>layout_kv为TND时，ori_kv的shape为(ORI_KV_T, KV_N, KV_D)</li>
                <li>layout_kv为PA_BBND时，ori_kv的shape为(ori_kv_block_nums, ori_kv_block_size, KV_N, KV_D)</li>
            </ul>
        </td>
        <td>
            当前版本必传
        </td>
    </tr>
    <tr>
        <td>attention_out</td>
        <td>
            <ul>
                <li>dtype支持bfloat16</li>
                <li>layout_q为BSND时，attention_out的shape为(B, Q_S, Q_N, Q_D)</li>
                <li>layout_q为TND时，attention_out的shape为(Q_T, Q_N, Q_D)</li>
            </ul>
        </td>
        <td>
            必须传入
        </td>
    </tr>
    <tr>
        <td>cmp_kv</td>
        <td>
            <ul>
                <li>dtype支持fp8_e4m3</li>
                <li>layout_kv为BSND时，cmp_kv的shape为(B, CMP_KV_S, KV_N, KV_D)</li>
                <li>layout_kv为TND时，cmp_kv的shape为(CMP_KV_T, KV_N, KV_D)</li>
                <li>layout_kv为PA_BBND时，cmp_kv的shape为(cmp_kv_block_nums, cmp_kv_block_size, KV_N, KV_D)</li>
            </ul>
        </td>
        <td>
            可选输入
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
                    <li>dtype仅支持int32</li>
                    <li>shape由mixed_quant_sparse_flash_metadata动态计算</li>
                </ul>
            </td>
            <td>当前版本必传</td>
            <td>无</td>
            <td>传入时需与mixed_quant_sparse_flash_metadata生成的结果一致</td>
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
                    <li>dtype支持int32</li>
                    <li>支持输入范围仅为0、3、4，默认值为0</li>
                </ul>
            </td>
            <td>
                可选，如果不传该参数，默认值为0
            </td>
            <td>
                <ul>
                    <li>无</li>
                </ul>
            </td>
            <td>
                <ul>
                     <li>只有ORI_SPARSE和ORI_CMP_SPARSE情景下，cmp_mask_mode为0和ori_mask_mode必须为0</li>
                     <li>SWA场景下，ori_mask_mode为0、3、4</li>
                 </ul>
             </td>
         </tr>
         <tr>
             <td>cmp_mask_mode</td>
             <td>
                 <ul>
                     <li>dtype支持int32</li>
                     <li>支持输入范围仅为0、3，默认值为0</li>
                 </ul>
             </td>
             <td>
                 可选，如果不传该参数，默认值为0
             </td>
             <td>
                 <ul>
                     <li>无</li>
                 </ul>
             </td>
             <td>
                     <li>当ori_kv/cmp_kv/cmp_sparse_indices/ori_sparse_indices传入时，cmp_mask_mode为0和ori_mask_mode必须为0</li>
                    <li>当cmp_kv不传时，ori_mask_mode为3、4</li>
                    <li>当ori_mask_mode为3时，cmp_mask_mode必须为3</li>
                    <li>当ori_mask_mode为4时，cmp_mask_mode必须为3</li>
            </td>
        </tr>
        <tr>
            <td>ori_win_left/ori_win_right</td>
            <td>
                <ul>
                    <li>dtype支持int32</li>
                    <li>支持输入范围仅为-1，>=0</li>
                </ul>
            </td>
            <td>
                可选，如果不传该参数，默认值为-1
            </td>
            <td>
                <ul>
                    无
                </ul>
            </td>
            <td>
                    <li>只有ori_mask_mode为4时，ori_win_left/ori_win_right可以>=0</li>
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
                    <li>dtype支持int32</li>
                    <li>shape为(B,)</li>
                    <li>取值仅支持非负整数</li>
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
                    <li>dtype支持int32</li>
                    <li>shape为(B,)</li>
                    <li>取值仅支持非负整数</li>
                    <li>seqused_ori_kv中的值需小于等于KV_S</li>
                </ul>
            </td>
            <td >
                <ul>
                    <li>当layout_kv为BSND时，可选传入</li>
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
                    <li>dtype支持int32</li>
                    <li>shape为(B,)</li>
                    <li>取值仅支持非负整数</li>
                    <li>seqused_ori_kv中的值需小于等于KV_S</li>
                </ul>
            </td>
            <td >
                <ul>
                    <li>当layout_kv为BSND时，可选传入</li>
                    <li>当layout_kv为PA_BBND且cmp_kv传入时，必须传入</li>
                    <li>ORI_CMP_SPARSE计算模式下，可以不传</li>
                </ul>
            </td>
            <td>无</td>
            <td>无</td>
        </tr>
        <tr>
            <td>cu_seqlens_q</td>
            <td>
                <ul>
                    <li>dtype支持int32</li>
                    <li>shape为(B+1,)</li>
                    <li>取值仅支持非负整数</li>
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
                    <li>dtype支持int32</li>
                    <li>shape为(B+1,)</li>
                    <li>取值仅支持非负整数</li>
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
                    <li>dtype支持int32</li>
                    <li>shape为(B+1,)</li>
                    <li>取值仅支持非负整数</li>
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
                    <li>data_type支持int32</li>
                    <li>表示cmp_kv相对于压缩前KV长度的压缩倍率，需大于0</li>
                    <li>默认值为1</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>可选，默认值为1</li>
                </ul>
            </td>
            <td>无</td>
            <td>
                <ul>
                    <li>在SWA典型场景，仅支持默认值1。</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cmp_residual_kv</td>
            <td>
                <ul>
                    <li>dtype支持int32</li>
                    <li>shape为(B,)</li>
                    <li>取值仅支持非负整数</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>只有cmp_kv传入才校验</li>
                    <li>可选</li>
                    <li>当cmp_mask_mode=3且cmp_ratio!=1时，必传</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>配合cmp_ratio使用，可恢复压缩前KV长度</li>
                </ul>
            </td>
            <td>无</td>
        </tr>
        <tr>
            <td>ori_sparse_indices</td>
            <td>
                <ul>
                    <li>dtype支持int32</li>
                    <li>shape为(Q_T, KV_N, ORI_KV_K)或(B, Q_S, KV_N, ORI_KV_K)</li>
                    <li>无效位置填-1，其余为非负整数</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>可选</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>当layout_q为TND时，该shape为(Q_T, KV_N, ORI_KV_K)</li>
                    <li>当layout_q为BSND时，该shape为(B, Q_S, KV_N, ORI_KV_K)</li>
                </ul>
            </td>
            <td>无</td>
        </tr>
        <tr>
            <td>cmp_sparse_indices</td>
            <td>
                <ul>
                    <li>dtype支持int32</li>
                    <li>shape为(Q_T, KV_N, CMP_KV_K)或(B, Q_S, KV_N, CMP_KV_K)</li>
                    <li>无效位置填-1，其余为非负整数</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>只有cmp_kv传入才校验</li>
                    <li>可选</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>当layout_q为TND时，该shape为(Q_T, KV_N, CMP_KV_K)</li>
                    <li>当layout_q为BSND时，该shape为(B, Q_S, KV_N, CMP_KV_K)</li>
                </ul>
            </td>
            <td>无</td>
        </tr>
        <tr>
            <td>ori_topk_length</td>
            <td>
                <ul>
                    <li>dtype支持int32</li>
                    <li>shape为(B, Q_S, KV_N)或(Q_T, KV_N)</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>ori_mask_mode=0且ori_sparse_indices不为空时，必须传入</li>
                    <li>其他场景不支持传入</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>当layout_q为TND时，该shape为(Q_T, KV_N)</li>
                    <li>当layout_q为BSND时，该shape为(B, Q_S, KV_N)</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>当ori_mask_mode不为0时，不支持传入</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cmp_topk_length</td>
            <td>
                <ul>
                    <li>dtype支持int32</li>
                    <li>shape为(B, Q_S, KV_N)或(Q_T, KV_N)</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>只有cmp_kv传入才校验</li>
                    <li>cmp_mask_mode=0且cmp_sparse_indices不为空时，必须传入</li>
                    <li>其他场景不支持传入</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>当layout_q为TND时，该shape为(Q_T, KV_N)</li>
                    <li>当layout_q为BSND时，该shape为(B, Q_S, KV_N)</li>
                </ul>
            </td>
            <td>无</td>
        </tr>
        <tr>
            <td>topk_value_mode</td>
            <td>
                <ul>
                    <li>data_type支持int32</li>
                    <li>topK索引取值模式，默认值为1</li>
                </ul>
            </td>
            <td>可选属性，默认值为1</td>
            <td>无</td>
            <td>无</td>
        </tr>
    </tbody>
</table>

#### Paged Attention参数组
当layout_kv为PA_BBND时，开启Paged Attention
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
                    <li>dtype仅支持int32</li>
                    <li>shape为(B, ori_kv_s_max/ori_kv_block_size)</li>
                    <li>值只能为正整数</li>
                </ul>
            </td>
            <td>可选</td>
            <td>无</td>
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
                    <li>dtype仅支持int32</li>
                    <li>shape为(B, cmp_kv_s_max/cmp_kv_block_size)</li>
                    <li>值只能为正整数</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>只有cmp_kv传入才校验</li>
                    <li>可选</li>
                </ul>
            </td>
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
                    <li>dtype支持float32</li>
                    <li>shape为(Q_N, )</li>
                </ul>
            </td>
            <td> 当前版本必传 </td>
            <td> 无 </td>
            <td> 无 </td>
        </tr>
    </tbody>
</table>

#### SoftmaxLse参数组

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
                    <li>data_type仅支持bool</li>
                    <li>true代表开启softmax_lse，false代表关闭softmax_lse</li>
                </ul>
            </td>
            <td>可选，默认值为false</td>
            <td rowspan="2">
                <ul>
                     <li>当return_softmax_lse为false时，输出shape为[1]的值为0的tensor</li>
                    <li>当return_softmax_lse为true时，softmax_lse的shape与layout_q的关系如下：<ul><li>layout_q为BSND时，softmax_lse的shape为(B, KV_N, Q_S, Q_N/KV_N)</li><li>layout_q为TND时，softmax_lse的shape为(KV_N, Q_T, Q_N/KV_N)</li></ul></li>
                </ul>
            </td>
            <td>无</td>
        </tr>
        <tr>
            <td>softmax_lse</td>
            <td>
                <ul>
                    <li>data_type仅支持float32</li>
                </ul>
            </td>
            <td>无</td>
            <td>无</td>
        </tr>
    </tbody>
</table>