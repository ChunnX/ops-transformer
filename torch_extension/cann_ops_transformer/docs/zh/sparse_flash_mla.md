# sparse_flash_mla <a name="ZH-CN_TOPIC_SPARSE_FLASH_MLA"></a>

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明<a name="zh-cn_topic_sparse_flash_mla_function"></a>

- 接口功能：

  `sparse_flash_mla`是基于`torch_npu`的`cann_ops_transformer`扩展接口，用于调用`SparseFlashMla`算子完成共享KV（Key和Value使用同一份输入）的稀疏注意力计算。接口支持训练和推理场景，并通过`ori_kv`和可选的`cmp_kv`组合KV上下文。

  `sparse_flash_mla_metadata`是主算子的前置元数据生成接口，用于生成AI Core/AI Vector Core的任务切分结果。`metadata`必须由与主算子完全一致的参数生成，并传入`sparse_flash_mla`；典型调用流程如下：

  1. 准备`q`、`ori_kv`、可选的`cmp_kv`及序列长度、稀疏索引或Block Table等输入。
  2. 调用`sparse_flash_mla_metadata`生成`metadata`。
  3. 调用`sparse_flash_mla`，并传入上一步生成的`metadata`。

  依据`cmp_kv`和`cmp_sparse_indices`的组合，接口支持以下三类计算模式：

  - **SWA（Sliding Window Attention）**：仅使用`ori_kv`，对原始KV做滑动窗口注意力。
  - **CSA（Compressed Sparse Attention）**：同时使用`ori_kv`、`cmp_kv`和`cmp_sparse_indices`，对原始KV窗口和TopK选择出的压缩KV共同做注意力。
  - **HCA（Heavily Compressed Attention）**：同时使用`ori_kv`和`cmp_kv`，对原始KV窗口和连续压缩KV段共同做注意力。

- 计算公式：

  $$
  O = \text{softmax}(Q \cdot \tilde{K}^{T} \cdot \text{softmax\_scale}) \cdot \tilde{V}
  $$

  其中$\tilde{K}=\tilde{V}$，由`ori_kv`的滑动窗口部分与`cmp_kv`的压缩部分共同组成。实际参与计算的KV范围由`cmp_ratio`、`ori_mask_mode`、`cmp_mask_mode`、`ori_win_left`、`ori_win_right`以及`cmp_sparse_indices`共同决定。

  开启`return_softmax_lse`后，第二个输出为每个Query位置的log-sum-exp：

  $$
  \operatorname{LSE}=\log\sum\exp(S-\max(S))+\max(S),\quad S=Q\tilde{K}^{T}\cdot\text{softmax\_scale}
  $$

> [!NOTE]
>
> `cmp_residual_kv`同时是`sparse_flash_mla`和`sparse_flash_mla_metadata`的可选输入，用于恢复压缩前KV长度：`ori_len_for_cmp_mask = cmp_len * cmp_ratio + cmp_residual_kv[b]`。当`cmp_kv`存在且`cmp_mask_mode=3`时，该参数必须传入。
>
> 文中B表示Batch大小；N1、N2分别表示q和KV的头数；S1、S2、S3分别表示q、`ori_kv`、`cmp_kv`的序列长度；T1、T2、T3分别表示对应TND布局下各Batch序列长度之和；D表示每个头的维度。

## 函数原型<a name="zh-cn_topic_sparse_flash_mla_prototype"></a>

调用`sparse_flash_mla`接口前，请先调用前置接口`sparse_flash_mla_metadata`完成任务切分。两次调用中参与任务切分的参数必须保持一致。

> [!NOTE]
>
> `sparse_flash_mla_metadata`的mask和窗口参数默认值为`ori_mask_mode=0`、`cmp_mask_mode=0`、`ori_win_left=-1`、`ori_win_right=-1`，而`sparse_flash_mla`的对应默认值为`4`、`3`、`127`、`0`。为保证两段调用的参数一致，请显式传入这四个参数，不要同时依赖两接口的默认值。

```python
cann_ops_transformer.sparse_flash_mla_metadata(
    num_heads_q,
    num_heads_kv,
    head_dim,
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
cann_ops_transformer.sparse_flash_mla(
    q,
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
    softmax_scale=1.0,
    cmp_ratio=1,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="BSND",
    layout_kv="BSND",
    topk_value_mode=1,
    return_softmax_lse=False
) -> (Tensor, Tensor)
```

## 参数说明<a name="zh-cn_topic_sparse_flash_mla_parameters"></a>

### sparse_flash_mla_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| num_heads_q | int | 必选 | Query的头数，支持1、2、4、8、16、32、64、128。 | int32 | - | - |
| num_heads_kv | int | 必选 | `ori_kv`和`cmp_kv`的头数，仅支持1。 | int32 | - | - |
| head_dim | int | 必选 | 每个注意力头的维度，仅支持512。 | int32 | - | - |
| cu_seqlens_q | Tensor | 可选 | TND场景下q各Batch有效token数的累积和，第一个元素必须为0。 | int32 | ND | (B+1,) |
| cu_seqlens_ori_kv | Tensor | 可选 | TND场景下`ori_kv`各Batch有效token数的累积和，第一个元素必须为0。 | int32 | ND | (B+1,) |
| cu_seqlens_cmp_kv | Tensor | 可选 | TND场景下`cmp_kv`各Batch有效token数的累积和，第一个元素必须为0。 | int32 | ND | (B+1,) |
| seqused_q | Tensor | 可选 | 每个Batch中q实际参与计算的token数。 | int32 | ND | (B,) |
| seqused_ori_kv | Tensor | 可选 | 每个Batch中`ori_kv`实际参与计算的token数。 | int32 | ND | (B,) |
| seqused_cmp_kv | Tensor | 可选 | 每个Batch中`cmp_kv`实际参与计算的token数。 | int32 | ND | (B,) |
| cmp_residual_kv | Tensor | 可选 | 每个Batch压缩前KV长度除以`cmp_ratio`的余数，用于恢复cmp侧mask长度。 | int32 | ND | (B,) |
| ori_topk_length | Tensor | 可选 | 预留字段，当前版本不支持传入非空Tensor。 | int32 | ND | - |
| cmp_topk_length | Tensor | 可选 | 预留字段，当前版本不支持传入非空Tensor。 | int32 | ND | - |
| batch_size | int | 可选 | Batch大小；BSND场景使用该值，默认值为0。 | int32 | - | - |
| max_seqlen_q | int | 可选 | q的最大有效序列长度，TND场景需与实际最大长度一致。默认值为0。 | int32 | - | - |
| max_seqlen_ori_kv | int | 可选 | `ori_kv`的最大有效序列长度，默认值为0。 | int32 | - | - |
| max_seqlen_cmp_kv | int | 可选 | `cmp_kv`的最大有效序列长度，默认值为0。 | int32 | - | - |
| ori_topk | int | 可选 | 原始KV的TopK长度，当前支持传0。 | int32 | - | - |
| cmp_topk | int | 可选 | 压缩KV的TopK长度；CSA场景应与索引最后一维对应。默认值为0。 | int32 | - | - |
| cmp_ratio | int | 可选 | `cmp_kv`相对于压缩前KV的压缩倍率，取值范围1~128；SWA场景固定为1。默认值为1。 | int32 | - | - |
| ori_mask_mode | int | 可选 | q与`ori_kv`的mask模式：0（No Mask）、3（RightDownCausal）或4（Band）。 | int32 | - | - |
| cmp_mask_mode | int | 可选 | q与`cmp_kv`的mask模式：0（No Mask）或3（RightDownCausal）。 | int32 | - | - |
| ori_win_left | int | 可选 | `ori_kv`滑动窗口左边界，-1表示不限制。 | int32 | - | - |
| ori_win_right | int | 可选 | `ori_kv`滑动窗口右边界，-1表示不限制。 | int32 | - | - |
| layout_q | string | 可选 | q的数据布局，支持`BSND`和`TND`。默认值为`BSND`。 | string | - | - |
| layout_kv | string | 可选 | `ori_kv`和`cmp_kv`的数据布局，支持`BSND`、`TND`和`PA_BBND`。默认值为`BSND`。 | string | - | - |
| has_ori_kv | bool | 可选 | 主算子是否传入`ori_kv`。默认值为True。 | bool | - | - |
| has_cmp_kv | bool | 可选 | 主算子是否传入`cmp_kv`。默认值为True。 | bool | - | - |
| metadata | Tensor | 输出 | 主算子使用的任务切分结果。 | int32 | ND | (1024,) |

### sparse_flash_mla

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| q | Tensor | 必选 | 公式中的Query。 | bfloat16/float16 | ND | `BSND`：(B, S1, N1, D)<br>`TND`：(T1, N1, D) |
| ori_kv | Tensor | 可选 | 原始KV输入，Key和Value共享同一份数据。 | bfloat16/float16 | ND | `BSND`：(B, S2, N2, D)<br>`TND`：(T2, N2, D)<br>`PA_BBND`：(num_blocks, block_size, N2, D) |
| cmp_kv | Tensor | 可选 | 压缩KV输入，Key和Value共享同一份数据。 | bfloat16/float16 | ND | `BSND`：(B, S3, N2, D)<br>`TND`：(T3, N2, D)<br>`PA_BBND`：(num_blocks, block_size, N2, D) |
| ori_sparse_indices | Tensor | 可选 | 原始KV稀疏索引；预留字段，当前版本不支持传入非空Tensor。 | int32 | ND | - |
| cmp_sparse_indices | Tensor | 可选 | 压缩KV的TopK索引，无效位置填-1；仅CSA场景传入。 | int32 | ND | `BSND`：(B, S1, N2, K)<br>`TND`：(T1, N2, K) |
| ori_block_table | Tensor | 可选 | PageAttention场景下`ori_kv`使用的Block映射表。 | int32 | ND | (B, max_num_blocks_per_seq) |
| cmp_block_table | Tensor | 可选 | PageAttention场景下`cmp_kv`使用的Block映射表。 | int32 | ND | (B, max_num_blocks_per_seq) |
| cu_seqlens_q | Tensor | 可选 | TND场景下q各Batch有效token数的累积和。 | int32 | ND | (B+1,) |
| cu_seqlens_ori_kv | Tensor | 可选 | TND场景下`ori_kv`各Batch有效token数的累积和。 | int32 | ND | (B+1,) |
| cu_seqlens_cmp_kv | Tensor | 可选 | TND场景下`cmp_kv`各Batch有效token数的累积和。 | int32 | ND | (B+1,) |
| seqused_q | Tensor | 可选 | 每个Batch中q实际参与计算的token数。 | int32 | ND | (B,) |
| seqused_ori_kv | Tensor | 可选 | 每个Batch中`ori_kv`实际参与计算的token数。 | int32 | ND | (B,) |
| seqused_cmp_kv | Tensor | 可选 | 每个Batch中`cmp_kv`实际参与计算的token数。 | int32 | ND | (B,) |
| cmp_residual_kv | Tensor | 可选 | 每个Batch的压缩余数；`cmp_kv`存在且`cmp_mask_mode=3`时必须传入。 | int32 | ND | (B,) |
| ori_topk_length | Tensor | 可选 | 预留字段，当前版本不支持传入非空Tensor。 | int32 | ND | - |
| cmp_topk_length | Tensor | 可选 | 预留字段，当前版本不支持传入非空Tensor。 | int32 | ND | - |
| sinks | Tensor | 可选 | Attention Sink输入。 | float32 | ND | (N1,) |
| metadata | Tensor | 必选 | 由`sparse_flash_mla_metadata`生成的任务切分结果。 | int32 | ND | (1024,) |
| softmax_scale | float | 可选 | QK矩阵乘后的缩放系数。默认值为1.0。 | float32 | - | - |
| cmp_ratio | int | 可选 | 压缩倍率，取值范围1~128。默认值为1。 | int32 | - | - |
| ori_mask_mode | int | 可选 | q与`ori_kv`的mask模式。默认值为4。 | int32 | - | - |
| cmp_mask_mode | int | 可选 | q与`cmp_kv`的mask模式。默认值为3。 | int32 | - | - |
| ori_win_left | int | 可选 | `ori_kv`滑动窗口左边界。默认值为127。 | int32 | - | - |
| ori_win_right | int | 可选 | `ori_kv`滑动窗口右边界。默认值为0。 | int32 | - | - |
| layout_q | string | 可选 | q的数据布局，支持`BSND`和`TND`。默认值为`BSND`。 | string | - | - |
| layout_kv | string | 可选 | KV的数据布局，支持`BSND`、`TND`和`PA_BBND`。默认值为`BSND`。 | string | - | - |
| topk_value_mode | int | 可选 | TopK索引取值模式，仅支持1。默认值为1。 | int32 | - | - |
| return_softmax_lse | bool | 可选 | 是否返回softmax的log-sum-exp结果。默认值为False。 | bool | - | - |
| attention_out | Tensor | 必选 | Attention计算输出，shape和数据类型与q一致。 | bfloat16/float16 | ND | 与q一致 |
| softmax_lse | Tensor | 可选 | softmax的log-sum-exp结果。 | float32 | ND | `BSND`：(B, N2, S1, N1/N2)<br>`TND`：(N2, T1, N1/N2) |

## 返回值说明<a name="zh-cn_topic_sparse_flash_mla_returns"></a>

### sparse_flash_mla_metadata

| 参数名 | 参数类型 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| metadata | Tensor | `sparse_flash_mla`使用的任务切分数据。 | int32 | ND | (1024,) |

### sparse_flash_mla

| 参数名 | 参数类型 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| attention_out | Tensor | 注意力计算输出，shape和数据类型与q一致。 | bfloat16/float16 | ND | 与q一致 |
| softmax_lse | Tensor | `return_softmax_lse=True`时返回softmax的log-sum-exp；否则返回float32标量占位Tensor。 | float32 | ND | `BSND`：(B, N2, S1, N1/N2)<br>`TND`：(N2, T1, N1/N2) |

## 约束说明<a name="zh-cn_topic_sparse_flash_mla_constraints"></a>

- 声明
  - `cu_seqlens_q`、`cu_seqlens_ori_kv`、`cu_seqlens_cmp_kv`、`seqused_q`、`seqused_ori_kv`、`seqused_cmp_kv`、`cmp_residual_kv`、稀疏索引及Block Table均为Tensor。算子在Tiling阶段无法校验其具体值，用户必须保证其合法性；传入非法值可能导致精度异常或非法内存访问。
  - `sparse_flash_mla_metadata`和`sparse_flash_mla`分两段调用。两次调用中参与任务切分的入参必须一致；不一致时可能产生未定义行为。
  - 本接口支持单算子模式和TorchAir图模式调用，可用于训练和推理场景。

### 特性参数组

| 特性参数组 | 参数字段名称 | 字段分组 | 字段类型 |
| :--- | :--- | :--- | :--- |
| 公共参数组 | q、`ori_kv`、`cmp_kv`、metadata、softmax_scale、layout_q、layout_kv、attention_out | INPUT / ATTR / OUTPUT | Tensor / float / string |
| 稀疏压缩参数组 | cmp_ratio、`ori_sparse_indices`、`cmp_sparse_indices`、`ori_topk_length`、`cmp_topk_length`、ori_topk、cmp_topk、topk_value_mode | INPUT(OPTIONAL) / ATTR(OPTIONAL) | Tensor / int |
| Mask和窗口参数组 | ori_mask_mode、cmp_mask_mode、ori_win_left、ori_win_right、`cmp_residual_kv` | INPUT(OPTIONAL) / ATTR(OPTIONAL) | Tensor / int |
| SeqLengths参数组 | `cu_seqlens_q`、`cu_seqlens_ori_kv`、`cu_seqlens_cmp_kv`、`seqused_q`、`seqused_ori_kv`、`seqused_cmp_kv`、max_seqlen_q、max_seqlen_ori_kv、max_seqlen_cmp_kv | INPUT(OPTIONAL) / ATTR(OPTIONAL) | Tensor / int |
| Paged Attention参数组 | `ori_block_table`、`cmp_block_table` | INPUT(OPTIONAL) | Tensor |
| Sinks参数组 | sinks | INPUT(OPTIONAL) | Tensor |
| SoftmaxLSE参数组 | return_softmax_lse、softmax_lse | ATTR(OPTIONAL) / OUTPUT | bool / Tensor |

### 基准信息说明

| 命名 | 含义 |
| :--- | :--- |
| B | Batch大小。 |
| N1 / N2 | q的头数 / KV的头数。N2当前仅支持1。 |
| S1 / S2 / S3 | q、`ori_kv`、`cmp_kv`的单Batch序列长度。 |
| T1 / T2 / T3 | TND布局下q、`ori_kv`、`cmp_kv`所有Batch序列长度的累加和。 |
| D | 每个注意力头的维度，当前仅支持512。 |
| K | CSA场景下`cmp_sparse_indices`的最后一维，即压缩KV的TopK长度。 |

### 参数组约束

#### 公共参数组

- `q`、`ori_kv`、`cmp_kv`的数据类型必须一致，且仅支持`float16`和`bfloat16`。
- `layout_q`和`layout_kv`仅支持`BSND`/`BSND`、`TND`/`TND`、`BSND`/`PA_BBND`、`TND`/`PA_BBND`组合。非`PA_BBND`场景下，两者必须一致。
- `layout_q="BSND"`时q必须为4维；`layout_q="TND"`时q必须为3维，并且必须传入`cu_seqlens_q`。
- `layout_kv="BSND"`或`layout_kv="PA_BBND"`时，KV必须为4维；`layout_kv="TND"`时，KV必须为3维。`layout_kv="TND"`时必须传入`cu_seqlens_ori_kv`；传入`cmp_kv`时，还必须传入`cu_seqlens_cmp_kv`。
- `metadata`必须为1024个`int32`元素；`topk_value_mode`仅支持1。`ori_sparse_indices`、`ori_topk_length`和`cmp_topk_length`当前不支持传入非空Tensor。
- `ori_kv`和`cmp_kv`允许存在行间padding类非连续内存，接口会通过aclNN获取stride信息并传递给底层算子。

#### 计算模式参数组

| 场景 | 必选输入与属性 | 不允许或固定的输入与属性 |
| :--- | :--- | :--- |
| SWA | 仅传入`ori_kv`；`cmp_ratio=1`。 | 不传`cmp_kv`、`cmp_sparse_indices`和`cmp_block_table`；`cmp_topk=0`、`cmp_mask_mode=0`。 |
| CSA | 传入`ori_kv`、`cmp_kv`、`cmp_sparse_indices`和`cmp_residual_kv`；`cmp_mask_mode=3`且`cmp_topk`为非0。 | - |
| HCA | 传入`ori_kv`、`cmp_kv`和`cmp_residual_kv`；`cmp_mask_mode=3`。 | 不传`cmp_sparse_indices`；`cmp_topk=0`。 |

#### SeqLengths和Mask参数组

- `cu_seqlens_q`、`cu_seqlens_ori_kv`和`cu_seqlens_cmp_kv`必须为当前Batch与前序Batch有效token数的累加值，第一个元素为0，后一个元素不得小于前一个元素。
- `seqused_q`、`seqused_ori_kv`和`seqused_cmp_kv`表示各Batch的实际有效token数。`seqused_cmp_kv`在所有KV布局下均可选，显式传入时用于覆盖cmp侧逻辑有效长度。
- `cmp_residual_kv[i]`必须小于`cmp_ratio`；CSA和HCA场景下，其长度必须等于Batch大小。
- `ori_mask_mode`及`cmp_mask_mode`的详细含义请参见[sparse_mode参数说明](../../../../docs/zh/context/sparse_mode_introduction.md)。当前规格中`ori_mask_mode`仅支持4，`ori_win_left`仅支持127，`ori_win_right`仅支持0。

#### Paged Attention参数组

- `layout_kv="PA_BBND"`时，必须传入`seqused_ori_kv`和`ori_block_table`；传入`cmp_kv`时，还必须传入`cmp_block_table`。
- PageAttention的`block_size`支持1~1024。对于<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>和<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>，`block_size`必须为16的倍数且不超过1024。

#### 产品型号和稀疏规格

- `head_dim`仅支持512，`num_heads_kv`仅支持1；`num_heads_q / num_heads_kv`支持1、2、4、8、16、32、64、128。
- <term>Ascend 950PR/Ascend 950DT</term>不支持`num_heads_q / num_heads_kv=1`。
- 对于<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>和<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：SWA场景`cmp_ratio`固定为1，CSA仅支持`cmp_ratio=4`，HCA仅支持`cmp_ratio=128`；`cmp_sparse_indices`最后一维K支持0、512、1024。

## 确定性计算<a name="zh-cn_topic_sparse_flash_mla_deterministic"></a>

- 默认支持确定性计算。
- 默认支持batch invariance。

## 调用示例<a name="zh-cn_topic_sparse_flash_mla_examples"></a>

### SWA场景（BSND输入）

```python
import math
import torch
import torch_npu
import cann_ops_transformer

torch_npu.npu.set_device(0)

dtype = torch.bfloat16
B = 1
S1 = 16
S2 = 64
N1 = 64
N2 = 1
D = 512
cmp_ratio = 1  # SWA示例仅传ori_kv，cmp_ratio不参与压缩KV计算，保持默认值1。

q = torch.randn(B, S1, N1, D, dtype=dtype, device="npu")
ori_kv = torch.randn(B, S2, N2, D, dtype=dtype, device="npu")
sinks = torch.zeros(N1, dtype=torch.float32, device="npu")

metadata = cann_ops_transformer.sparse_flash_mla_metadata(
    N1,
    N2,
    D,
    batch_size=B,
    max_seqlen_q=S1,
    max_seqlen_ori_kv=S2,
    ori_topk=0,
    cmp_topk=0,
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=0,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="BSND",
    layout_kv="BSND",
    has_ori_kv=True,
    has_cmp_kv=False,
)

attn_out, softmax_lse = cann_ops_transformer.sparse_flash_mla(
    q,
    ori_kv=ori_kv,
    sinks=sinks,
    metadata=metadata,
    softmax_scale=1.0 / math.sqrt(D),
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=0,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="BSND",
    layout_kv="BSND",
    return_softmax_lse=False,
)
torch_npu.npu.synchronize()
assert attn_out.shape == q.shape
assert attn_out.dtype == q.dtype
assert softmax_lse.shape == torch.Size([])
assert torch.isfinite(attn_out.float()).all().item()
```

### HCA场景（BSND输入）

```python
import math
import torch
import torch_npu
import cann_ops_transformer

torch_npu.npu.set_device(0)

dtype = torch.bfloat16
B = 1
S1 = 16
S2 = 128
S3 = 1
N1 = 64
N2 = 1
D = 512
cmp_ratio = 128

q = torch.randn(B, S1, N1, D, dtype=dtype, device="npu")
ori_kv = torch.randn(B, S2, N2, D, dtype=dtype, device="npu")
cmp_kv = torch.randn(B, S3, N2, D, dtype=dtype, device="npu")
cmp_residual_kv = torch.zeros(B, dtype=torch.int32, device="npu")
sinks = torch.zeros(N1, dtype=torch.float32, device="npu")

metadata = cann_ops_transformer.sparse_flash_mla_metadata(
    N1,
    N2,
    D,
    batch_size=B,
    max_seqlen_q=S1,
    max_seqlen_ori_kv=S2,
    max_seqlen_cmp_kv=S3,
    cmp_residual_kv=cmp_residual_kv,
    ori_topk=0,
    cmp_topk=0,
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="BSND",
    layout_kv="BSND",
    has_ori_kv=True,
    has_cmp_kv=True,
)

attn_out, softmax_lse = cann_ops_transformer.sparse_flash_mla(
    q,
    ori_kv=ori_kv,
    cmp_kv=cmp_kv,
    cmp_residual_kv=cmp_residual_kv,
    sinks=sinks,
    metadata=metadata,
    softmax_scale=1.0 / math.sqrt(D),
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="BSND",
    layout_kv="BSND",
    return_softmax_lse=False,
)
torch_npu.npu.synchronize()
assert attn_out.shape == q.shape
assert attn_out.dtype == q.dtype
assert softmax_lse.shape == torch.Size([])
assert torch.isfinite(attn_out.float()).all().item()
```

### CSA场景（TND输入，使用cmp_residual_kv）

```python
import math
import torch
import torch_npu
import cann_ops_transformer

torch_npu.npu.set_device(0)

dtype = torch.float16
B = 1
q_lens = [1]
ori_lens = [6]
cmp_lens = [1]
N1 = 64
N2 = 1
D = 512
K = 512
cmp_ratio = 4

cu_q = torch.tensor([0, 1], dtype=torch.int32, device="npu")
cu_ori = torch.tensor([0, 6], dtype=torch.int32, device="npu")
cu_cmp = torch.tensor([0, 1], dtype=torch.int32, device="npu")
cmp_residual_kv = torch.tensor([2], dtype=torch.int32, device="npu")

q = torch.randn(sum(q_lens), N1, D, dtype=dtype, device="npu")
ori_kv = torch.randn(sum(ori_lens), N2, D, dtype=dtype, device="npu")
cmp_kv = torch.randn(sum(cmp_lens), N2, D, dtype=dtype, device="npu")
sinks = torch.zeros(N1, dtype=torch.float32, device="npu")

cmp_sparse_indices = torch.full((sum(q_lens), N2, K), -1, dtype=torch.int32, device="npu")
cmp_sparse_indices[:, :, :1] = torch.arange(1, dtype=torch.int32, device="npu").view(1, 1, 1)

metadata = cann_ops_transformer.sparse_flash_mla_metadata(
    N1,
    N2,
    D,
    cu_seqlens_q=cu_q,
    cu_seqlens_ori_kv=cu_ori,
    cu_seqlens_cmp_kv=cu_cmp,
    max_seqlen_q=max(q_lens),
    max_seqlen_ori_kv=max(ori_lens),
    max_seqlen_cmp_kv=max(cmp_lens),
    ori_topk=0,
    cmp_topk=K,
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    cmp_residual_kv=cmp_residual_kv,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="TND",
    layout_kv="TND",
    has_ori_kv=True,
    has_cmp_kv=True,
)

attn_out, softmax_lse = cann_ops_transformer.sparse_flash_mla(
    q,
    ori_kv=ori_kv,
    cmp_kv=cmp_kv,
    cmp_sparse_indices=cmp_sparse_indices,
    cu_seqlens_q=cu_q,
    cu_seqlens_ori_kv=cu_ori,
    cu_seqlens_cmp_kv=cu_cmp,
    cmp_residual_kv=cmp_residual_kv,
    sinks=sinks,
    metadata=metadata,
    softmax_scale=1.0 / math.sqrt(D),
    cmp_ratio=cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="TND",
    layout_kv="TND",
    return_softmax_lse=False,
)
torch_npu.npu.synchronize()
assert attn_out.shape == q.shape
assert attn_out.dtype == q.dtype
assert softmax_lse.shape == torch.Size([])
assert torch.isfinite(attn_out.float()).all().item()
```

### CP切分示例（TND + CSA，rank0切开第二个序列）

下面示例用单进程顺序模拟两个CP rank，说明全局TND数据与每个rank入参之间的关系。假设全局有2个序列，`cmp_ratio=4`：

| 视角 | q范围 | ori_kv范围 | cmp_kv范围 | cu_seqlens_q | cu_seqlens_ori_kv | cu_seqlens_cmp_kv | cmp_residual_kv |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 全局 | seq0 `[0,16)`，seq1 `[0,18)` | seq0 `[0,16)`，seq1 `[0,18)` | seq0 `[0,4)`，seq1 `[0,4)` | `[0,16,34]` | `[0,16,34]` | `[0,4,8]` | `[0,2]` |
| rank0 | seq0 `[0,16)`，seq1 `[0,8)` | seq0 `[0,16)`，seq1 `[0,8)` | seq0 `[0,4)`，seq1 `[0,2)` | `[0,16,24]` | `[0,16,24]` | `[0,4,6]` | `[0,0]` |
| rank1 | seq1 `[8,18)` | seq1 `[0,18)` | seq1 `[0,4)` | `[0,10]` | `[0,18]` | `[0,4]` | `[2]` |

rank1虽然只计算seq1的`[8,18)`，但`ori_kv`和`cmp_kv`需要传到当前位置结束为止的前缀。此时`ori_prefix_len - q_len = 18 - 10 = 8`，kernel推导出的q起点正好是CP切分点。每个本地batch都需要满足`cmp_len * cmp_ratio + cmp_residual_kv[b] == ori_prefix_len`。

```python
import math
import torch
import torch_npu
import cann_ops_transformer


torch_npu.npu.set_device(0)

dtype = torch.float16
cmp_ratio = 4
K = 512
N1 = 64
N2 = 1
D = 512

# 全局packed TND视角：seq0长度16，seq1长度18。
global_q_lens = [16, 18]
global_ori_lens = [16, 18]
global_cmp_lens = [4, 4]
global_cmp_residual = [0, 2]

q_global = torch.randn(sum(global_q_lens), N1, D, dtype=dtype, device="npu")
ori_global = torch.randn(sum(global_ori_lens), N2, D, dtype=dtype, device="npu")
cmp_global = torch.randn(sum(global_cmp_lens), N2, D, dtype=dtype, device="npu")
sinks = torch.zeros(N1, dtype=torch.float32, device="npu")


def make_cu(lengths):
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    return torch.tensor(cu, dtype=torch.int32, device="npu")


def make_cmp_sparse_indices(q_lens, ori_prefix_lens, cmp_lens):
    indices = torch.full((sum(q_lens), N2, K), -1, dtype=torch.int32, device="npu")
    q_base = 0
    for q_len, ori_prefix_len, cmp_len in zip(q_lens, ori_prefix_lens, cmp_lens):
        q_start = ori_prefix_len - q_len
        for row in range(q_len):
            q_pos = q_start + row
            cmp_end = min(cmp_len, (q_pos + 1) // cmp_ratio)
            if cmp_end > 0:
                indices[q_base + row, :, :cmp_end] = torch.arange(
                    cmp_end, dtype=torch.int32, device="npu"
                ).view(1, cmp_end)
        q_base += q_len
    return indices


def run_one_rank(name, q, ori_kv, cmp_kv, q_lens, ori_prefix_lens, cmp_lens, residuals):
    cu_q = make_cu(q_lens)
    cu_ori = make_cu(ori_prefix_lens)
    cu_cmp = make_cu(cmp_lens)
    cmp_residual_kv = torch.tensor(residuals, dtype=torch.int32, device="npu")
    cmp_sparse_indices = make_cmp_sparse_indices(q_lens, ori_prefix_lens, cmp_lens)

    metadata = cann_ops_transformer.sparse_flash_mla_metadata(
        N1,
        N2,
        D,
        cu_seqlens_q=cu_q,
        cu_seqlens_ori_kv=cu_ori,
        cu_seqlens_cmp_kv=cu_cmp,
        max_seqlen_q=max(q_lens),
        max_seqlen_ori_kv=max(ori_prefix_lens),
        max_seqlen_cmp_kv=max(cmp_lens),
        ori_topk=0,
        cmp_topk=K,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        cmp_residual_kv=cmp_residual_kv,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="TND",
        has_ori_kv=True,
        has_cmp_kv=True,
    )

    attn_out, softmax_lse = cann_ops_transformer.sparse_flash_mla(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        cmp_sparse_indices=cmp_sparse_indices,
        cu_seqlens_q=cu_q,
        cu_seqlens_ori_kv=cu_ori,
        cu_seqlens_cmp_kv=cu_cmp,
        cmp_residual_kv=cmp_residual_kv,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=1.0 / math.sqrt(D),
        cmp_ratio=cmp_ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="TND",
        return_softmax_lse=False,
    )
    torch_npu.npu.synchronize()
    assert attn_out.shape == q.shape, name
    assert attn_out.dtype == q.dtype, name
    assert softmax_lse.shape == torch.Size([]), name
    assert torch.isfinite(attn_out.float()).all().item(), name
    return attn_out


# rank0：包含完整seq0，并切到seq1前8个token。
rank0_q = torch.cat([q_global[0:16], q_global[16:24]], dim=0)
rank0_ori = torch.cat([ori_global[0:16], ori_global[16:24]], dim=0)
rank0_cmp = torch.cat([cmp_global[0:4], cmp_global[4:6]], dim=0)
rank0_out = run_one_rank(
    "rank0", rank0_q, rank0_ori, rank0_cmp,
    q_lens=[16, 8], ori_prefix_lens=[16, 8], cmp_lens=[4, 2], residuals=[0, 0]
)

# rank1：只算seq1后10个token，但ori_kv/cmp_kv传seq1到18为止的前缀。
rank1_q = q_global[24:34]
rank1_ori = ori_global[16:34]
rank1_cmp = cmp_global[4:8]
rank1_out = run_one_rank(
    "rank1", rank1_q, rank1_ori, rank1_cmp,
    q_lens=[10], ori_prefix_lens=[18], cmp_lens=[4], residuals=[2]
)

seq0_out = rank0_out[:16]
seq1_out = torch.cat([rank0_out[16:24], rank1_out], dim=0)
assert seq0_out.shape == (16, N1, D)
assert seq1_out.shape == (18, N1, D)
```
