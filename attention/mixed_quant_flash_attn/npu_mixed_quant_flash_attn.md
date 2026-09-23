# mixed_quant_flash_attn <a name="ZH-CN_TOPIC_0000002309174912"></a>

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明<a name="zh-cn_topic_0000002168254826_section14441124184110"></a>

- 接口功能:

  `mixed_quant_flash_attn`是基于`torch_npu`的`npu_ops_transformer`扩展接口，用于调用`MixedQuantFlashAttn`算子完成key、value的伪量化注意力计算，训练推理归一化。该算子采用伪量化策略：Q保持高精度数据类型（BFLOAT16/FLOAT16）不量化，key、value使用低精度量化存储（数据类型由`quant_compute_mode`决定，当前支持MXFP4和HiFP4），通过`k_descale`和`v_descale`缩放因子进行反量化恢复。

  `mixed_quant_flash_attn_metadata`是`mixed_quant_flash_attn`的元数据生成接口，用于在主算子执行前生成metadata。metadata记录AICore/AIVCore的任务切分结果，主算子可选择传入该metadata以优化调度。典型调用流程如下：

  1. 准备`q`、`k`、`v`等输入。
  2. 调用`mixed_quant_flash_attn_metadata`生成`metadata`。
  3. 调用`mixed_quant_flash_attn`，将上一步得到的`metadata`传入主算子。

- 计算公式:

  self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n \times d$的矩阵。

  self-attention的计算公式一般定义如下，其中$Q、K、V$为输入样本的重要属性元素，是输入样本经过空间变换得到，且可以统一到一个特征空间中。公式及算子名称中的"Attention"为"self-attention"的简写。

  $$
  Attention(Q,K,V)=Score(Q,K)V
  $$

  本算子中Score函数采用Softmax函数，self-attention计算公式为:

  $$
  Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
  $$

  其中$Q$和$K^T$的乘积代表输入$x$的注意力，为避免该值变得过大，通常除以$\sqrt{d}$进行缩放，并对每行进行softmax归一化，与$V$相乘后得到一个$n \times d$的矩阵。

  增加**sink**之后计算逻辑如下所示，主要修改相关softmax_max和softmax_sum逻辑计算部分。
    
  $$
  S = \frac{QK^T}{\sqrt{d}}
  $$

  $$
  m = max(sink, max(S))
  $$

  $$
  Attention = \frac{e^{S - m} \times V}{\sum e^{S-m} + e^{sink - m}}
  $$

开启**return_softmax_lse**之后，返回值softmax_lse计算逻辑如下所示：

  $$
  S = \frac{QK^T}{\sqrt{d}}
  $$
  
  $$
  softmax\_max = max(S)
  $$

  $$
  softmax\_lse = log{\sum e^{S-softmax\_max}} + softmax\_max
  $$

  本算子采用**伪量化**（Mixed Quantization）策略，query保持原始精度不量化，key、value使用低精度量化存储。计算时通过反量化缩放因子恢复key、value精度，伪量化计算公式如下：

  $$
  K_{dequant} = K_{quant} \times k\_descale
  $$

  $$
  V_{dequant} = V_{quant} \times v\_descale
  $$

  $$
  Attention(Q,K_{dequant},V_{dequant})=Softmax(\frac{Q \cdot K_{dequant}^T}{\sqrt{d}}) \cdot V_{dequant}
  $$

## 函数原型<a name="zh-cn_topic_0000002168254826_section45077510411"></a>

调用mixed_quant_flash_attn接口之前，请先调用前置接口mixed_quant_flash_attn_metadata，完成mixed_quant_flash_attn负载均衡的计算。

```python
npu_ops_transformer.ops.mixed_quant_flash_attn_metadata(
    num_heads_q,
    num_heads_kv,
    head_dim,
    quant_compute_mode,
    *,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_kv=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    mask_mode=None,
    win_left=None,
    win_right=None,
    layout_q=None,
    layout_kv=None,
    layout_attn_out=None
) -> Tensor
```

```python
npu_ops_transformer.ops.mixed_quant_flash_attn(
    q,
    k,
    v,
    k_descale,
    v_descale,
    block_table=None,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    attn_mask=None,
    metadata=None,
    quant_compute_mode=1,
    softmax_scale=1.0,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="PA_BBND",
    layout_attn_out="BSND",
    return_softmax_lse=False
) -> (Tensor, Tensor)
```

## 参数说明<a name="zh-cn_topic_0000002168254826_section112637109429"></a>

### 基准信息说明

资料约束中，常见字段释义如下：

|    命名    |                            含义                            |
| :---------: | :---------------------------------------------------------: |
|      B      |                Batch,表示输入样本批量大小                |
|     Q_N     |        输入query tensor的头数，对应query shape中的N        |
|    KV_N    |    输入key/value tensor的头数，对应key/value shape中的N    |
|     Q_S     |      输入query tensor的序列长度，对应q shape中的S      |
|    KV_S    |  输入key/value tensor的序列长度，对应key/value shape中的S  |
|     Q_T     |          输入query tensor所有batch序列长度的累加和          |
|     D     |          输入query/key/value tensor以及输出attn_out隐藏层最小的单元尺寸headdim         |
|     block_size     |          PageAttention中key、value存储每个block中最大的token个数         |
|     block_num     |          PageAttention中block的数量         |
|     max_block_num_per_seq     |          每个batch中最大KV_S对应的block数量         |

### mixed_quant_flash_attn_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| num_heads_q | int | 必选 | Query head数 | int32 | - | - | - |
| num_heads_kv | int | 必选 | Key/Value head数 | int32 | - | - | - |
| head_dim | int | 必选 | 每个注意力头的维度 | int32 | - | - | - |
| quant_compute_mode | int | 必选 | 量化模式 | int32 | - | - | - |
| cu_seqlens_q | Tensor | 可选 | 累积序列长度，用于处理变长序列，第一个元素必须为0。仅在layout_q为TND时支持配置 | int32 | ND | (B+1,) | x |
| seqused_q | Tensor | 可选 | 指定每batch中实际使用的序列长度，截断冗余运算 | int32 | ND | (B,) | x |
| seqused_kv | Tensor | 可选 | 指定每batch中实际使用的序列长度，截断冗余运算 | int32 | ND | (B,) | x |
| batch_size | int | 可选 | batch大小。若未传入，则从cu_seqlens_q或seqused_q推导。默认值为None | int32 | - | - | - |
| max_seqlen_q | int | 可选 | 指定查询q序列的长度上限 | int32 | - | - | - |
| max_seqlen_kv | int | 可选 | 指定键k和值v序列的长度上限 | int32 | - | - | - |
| mask_mode | int | 可选 | 掩码模式 | int32 | - | - | - |
| win_left | int | 可选 | window左界限 | int32 | - | - | - |
| win_right | int | 可选 | window右界限 | int32 | - | - | - |
| layout_q | string | 可选 | 定义输入q张量的布局格式 | string | - | - | - |
| layout_kv | string | 可选 | 定义输入k和v张量的布局格式 | string | - | - | - |
| layout_attn_out | string | 可选 | 定义输出张量的布局格式 | string | - | - | - |

### mixed_quant_flash_attn

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| q | Tensor | 必选 | 公式中的q | bfloat16/float16 | ND | <ul><li>(B, Q_S, Q_N, D)</li><li>(B, Q_N, Q_S, D)</li></ul> | x |
| k | Tensor | 必选 | 公式中的k | float4_e2m1/hifloat4 | ND | <ul><li>PA_BBND: (block_num, block_size, KV_N, D)</li><li>PA_BNBD: (block_num, KV_N, block_size, D)</li><li>PA_NZ: (block_num, KV_N, D/64, block_size, 64)</li></ul> | x |
| v | Tensor | 必选 | 公式中的v | float4_e2m1/hifloat4 | ND | <ul><li>PA_BBND: (block_num, block_size, KV_N, D)</li><li>PA_BNBD:(block_num, KV_N, block_size, D)</li><li>PA_NZ: (block_num, KV_N, D/64, block_size, 64)</li></ul> | x |
| k_descale | Tensor | 必选 | k的反量化缩放因子 | float8_e8m0/hifloat4_scale | ND | <ul><li>quant_compute_mode=1(MXFP4，group_size=32)：<ul><li>PA_BBND: (block_num, block_size, KV_N, D/32)</li><li>PA_BNBD: (block_num, KV_N, block_size, D/32)</li><li>PA_NZ: (block_num, KV_N, block_size/16, ceil(D/64), 16, 2)</li></ul></li><li>quant_compute_mode=2(HIFP4，group_size=64)：<ul><li>PA_BBND: (block_num, block_size, KV_N, D/64)</li><li>PA_BNBD: (block_num, KV_N, block_size, D/64)</li><li>PA_NZ: (block_num, KV_N, block_size/16, ceil(D/64), 16)</li></ul></li></ul> | x |
| v_descale | Tensor | 必选 | v的反量化缩放因子 | float8_e8m0/hifloat4_scale | ND | <ul><li>quant_compute_mode=1(MXFP4，group_size=32)：<ul><li>PA_BBND: (block_num, KV_N, D, block_size/32)</li><li>PA_BNBD: (block_num, KV_N, D, block_size/32)</li><li>PA_NZ: (block_num, KV_N, D/16, ceil(block_size/64), 16, 2)</li></ul></li><li>quant_compute_mode=2(HIFP4，group_size=64)：<ul><li>PA_BBND: (block_num, KV_N, D, block_size/64)</li><li>PA_BNBD: (block_num, KV_N, D, block_size/64)</li><li>PA_NZ: (block_num, KV_N, D/16, ceil(block_size/64), 16)</li></ul></li></ul> | x |
| block_table | Tensor | 可选 | 用于分块注意力计算中的块索引映射，当前仅支持PA场景，必须传入 | int32 | ND | (B, max_block_num_per_seq) | x |
| cu_seqlens_q | Tensor | 可选 | 累积序列长度，用于处理变长序列，第一个元素必须为0。仅在layout_q为TND时支持配置 | int32 | ND | (B+1,) | x |
| seqused_q | Tensor | 可选 | 指定每batch中实际使用的序列长度，截断冗余运算 | int32 | ND | (B,) | x |
| seqused_kv | Tensor | 可选 | 指定每batch中实际使用的序列长度，截断冗余运算 | int32 | ND | (B,) | x |
| sinks | Tensor | 可选 | 表示通过可学习的"Sink Token"起到吸收Attention Score的作用 | float32 | ND | (Q_N,) | x |
| attn_mask | Tensor | 可选 | 注意力掩码矩阵 | int8 | ND | (2048, 2048) | x |
| metadata | Tensor | 可选 | `mixed_quant_flash_attn_metadata`生成的任务切分结果，传入后可优化调度 | int32 | ND | (max_schedule_size,) | x |
| quant_compute_mode | int | 必选 | 量化模式 | int32 | - | - | - |
| softmax_scale | float | 可选 | 可显式设置缩放因子，覆盖默认计算 | float32 | - | - | - |
| mask_mode | int | 可选 | 掩码模式，取值0/3分别代表全计算模式、Causal模式 | int32 | - | - | - |
| win_left | int | 可选 | window左界限 | int32 | - | - | - |
| win_right | int | 可选 | window右界限 | int32 | - | - | - |
| max_seqlen_q | int | 可选 | 指定查询q序列的长度上限 | int32 | - | - | - |
| max_seqlen_kv | int | 可选 | 指定键k和值v序列的长度上限 | int32 | - | - | - |
| layout_q | string | 可选 | 定义输入q张量的布局格式 | string | - | - | - |
| layout_kv | string | 可选 | 定义输入k和v张量的布局格式，默认PA_BBND | string | - | - | - |
| layout_attn_out | string | 可选 | 定义输出张量的布局格式 | string | - | - | - |
| return_softmax_lse | bool | 可选 | 是否需要获取softmax的LSE结果 | bool | - | - | - |

## 返回值说明<a name="zh-cn_topic_0000002168254826_section22231435517"></a>

### mixed_quant_flash_attn_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| metadata | Tensor | 必选 | mixed_quant_flash_attn的任务切分数据 | int32 | ND | shape根据batch_size和num_heads_kv动态计算 | x |

### mixed_quant_flash_attn

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| attn_out | Tensor | 必选 | mixed_quant_flash_attn的计算输出 | bfloat16/float16 | ND | <ul><li>(B, Q_S, Q_N, D)</li><li>(B, Q_N, Q_S, D)</li></ul> | x |
| softmax_lse | Tensor | 可选 | softmax的LSE结果 | float32 | ND | (B, Q_N, Q_S) | x |

**说明**
-   attn_out：Tensor类型，公式中的输出，数据类型支持float16、bfloat16。数据格式支持ND。限制：该输出参数的D维度与value的D保持一致，其余维度需要与入参query的shape保持一致。
-   softmax_lse：Tensor类型，公式中的softmax_lse。数据类型支持float32，return_softmax_lse为True时，输出shape为(B, Q_N, Q_S)的Tensor；return_softmax_lse为False时，则输出shape为[0]的空Tensor。


## 约束说明<a name="zh-cn_topic_0000002168254826_section12345537164214"></a>

- 声明
  - 参数cu_seqlens_q、seqused_q、seqused_kv、block_table及attn_mask属于tensor。由于算子在Tiling阶段无法获取tensor的具体数值，tiling侧不对值进行校验，正确性需要用户自行保证。若上述参数传入非法值，会触发未定义行为（精度问题、非法内存访问导致的程序崩溃等）。

### 约束类型说明
mixed_quant_flash_attn算子约束分为4个档位，按约束复杂程度递增分为单参数约束、存在性约束、一致性约束和特性交叉约束，各档位约束内容和示例如下：
- 单参数约束：对于单个接口参数的约束，包含mixed_quant_flash_attn算子接口中的Tensor、TensorList、Array和Attributes
  - 对于Tensor、TensorList、Array，单参数约束中包含如下校验
    - 校验shape，包括shape维度dim、每一维度dim value
    - 校验dtype
    - 校验format
  - 对于属性Attribute
    - 校验属性取值
- 存在性约束：约束特定场景下，特性参数组内，必须传入某参数，或不支持传入某参数
- 一致性约束：特性参数组内，各个参数间约束。
- 特性交叉约束：涉及多个参数组，不同参数组间交叉约束

### 特性参数组

|      特性参数组      |     参数字段名称     |    字段分组    |  字段类型  |
| :-------------------: | :-------------------: | :-------------: | :--------: |
|      公共参数组      |         q         |      INPUT      |   Tensor   |
|                      |          k          |      INPUT      | Tensor |
|                      |         v         |      INPUT      | Tensor |
|                      |         metadata        |      INPUT(OPTIONAL)      | Tensor |
|                      |      softmax_scale      | ATTR(OPTIONAL) |   float   |
|                      |      layout_q      | ATTR(OPTIONAL) |   string   |
|                      |      layout_kv      | ATTR(OPTIONAL) |   string   |
|                      |      layout_attn_out      | ATTR(OPTIONAL) |   string   |
|                      |     attn_out     |     OUTPUT     |   Tensor   |
|                      |    quant_compute_mode    | ATTR(REQUIRED) |    int    |
|                      |    k_descale    |      INPUT      |   Tensor   |
|                      |    v_descale    |      INPUT      |   Tensor   |
|      Mask参数组      |       mask_mode       | ATTR(OPTIONAL) |   int   |
|                      |       win_left       | ATTR(OPTIONAL) |   int   |
|                      |      win_right      | ATTR(OPTIONAL) |   int   |
|                      |      attn_mask      | INPUT(OPTIONAL) |   Tensor   |
| SeqLens参数组  |   cu_seqlens_q   | INPUT(OPTIONAL) |  Tensor  |
|                      |  seqused_q  | INPUT(OPTIONAL) |  Tensor  |
|                      |  seqused_kv  | INPUT(OPTIONAL) |  Tensor  |
|                      |  max_seqlen_q  | ATTR(OPTIONAL) |  int  |
|                      |  max_seqlen_kv  | ATTR(OPTIONAL) |  int  |
| Paged Attention参数组 |      block_table      | INPUT(OPTIONAL) |   Tensor   |
|  Sinks参数组  |     sinks     | INPUT(OPTIONAL) |   Tensor   |
|   SoftmaxLSE参数组   |    return_softmax_lse    | ATTR(OPTIONAL) |    bool    |
|                      |      softmax_lse      |     OUTPUT     |   Tensor   |

### 参数组约束

#### 公共参数组
- 入参为空的场景处理：
    - 空Tensor指必选输入和输出的shape size为0,即有任意轴为0。
    - 触发空tensor的用例将全部拦截报错。

- q、k、v、attn_out校验
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
                <li>tensor_type支持BFLOAT16和FLOAT16</li>
                <li>shape维度数量支持4(BSND、BNSD)</li>
            </ul>
        </td>
        <td rowspan="7">必选参数</td>
        <td rowspan="4">
            <ul>
                <li>q、attn_out的数据类型需相同</li>
                <li>k、v的shape需相同</li>
                <li>Layout校验规则见layout匹配关系表</li>
            </ul>
        </td>
        <td rowspan="6">
                轴校验：
            <ul>
                <li>65536 > B > 0</li>
                <li>Q_N > 0</li>
                <li>KV_N > 0</li>
                <li>Q_S > 0</li>
                <li>KV_S > 0</li>
                <li>D = 128</li>
                <li>Q_N % KV_N == 0且Q_N / KV_N > 0</li>
                <li>block_size需在[64, 2048]范围内且为64的倍数</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>k</td>
        <td rowspan="2">
            <ul>
                <li>tensor_type支持FLOAT4_E2M1和HIFLOAT4</li>
                <li>shape维度数量支持4、5(PA_BBND、PA_BNBD、PA_NZ)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>v</td>
    </tr>
    <tr>
        <td>attn_out</td>
        <td>
            <ul>
                <li>tensor_type支持BFLOAT16和FLOAT16</li>
                <li>shape维度数量支持4(BSND、BNSD)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>k_descale</td>
        <td rowspan="2">
            <ul>
                <li>tensor_type支持FLOAT8_E8M0和HIFLOAT4_SCALE</li>
                <li>shape维度数量支持4、5、6(PA_BBND、PA_BNBD、PA_NZ)</li>
            </ul>
        </td>
        <td rowspan="3">
            <ul>
                <li>当quant_compute_mode=1(A16C4_KV_MXFP4_SOFTMAX_FP32)时:</li>
                <li>
                    <ul>
                        <li>key、value的数据类型为FLOAT4_E2M1</li>
                        <li>k_descale、v_descale的数据类型必须为FLOAT8_E8M0</li>
                        <li>k_descale的shape:</li>
                        <ul>
                            <li>PA_BBND: (block_num, block_size, KV_N, D/32)</li>
                            <li>PA_BNBD: (block_num, KV_N, block_size, D/32)</li>
                            <li>PA_NZ: (block_num, KV_N, block_size/16, ceil(D/64), 16, 2)</li>
                        </ul>
                        <li>v_descale的shape:</li>
                        <ul>
                            <li>PA_BBND: (block_num, KV_N, D, block_size/32)</li>
                            <li>PA_BNBD: (block_num, KV_N, D, block_size/32)</li>
                            <li>PA_NZ: (block_num, KV_N, D/16, ceil(block_size/64), 16, 2)</li>
                        </ul>
                    </ul>
                </li>
                <li>当quant_compute_mode=2(A16C4_KV_HIF4_SOFTMAX_FP32)时:</li>
                <li>
                    <ul>
                        <li>key、value的数据类型为HIFLOAT4(FLOAT4_E1M2)</li>
                        <li>k_descale、v_descale的数据类型必须为HIFLOAT4_SCALE</li>
                        <li>k_descale的shape:</li>
                        <ul>
                            <li>PA_BBND: (block_num, block_size, KV_N, D/64)</li>
                            <li>PA_BNBD: (block_num, KV_N, block_size, D/64)</li>
                            <li>PA_NZ: (block_num, KV_N, block_size/16, ceil(D/64), 16)</li>
                        </ul>
                        <li>v_descale的shape:</li>
                        <ul>
                            <li>PA_BBND: (block_num, KV_N, D, block_size/64)</li>
                            <li>PA_BNBD: (block_num, KV_N, D, block_size/64)</li>
                            <li>PA_NZ: (block_num, KV_N, D/16, ceil(block_size/64), 16)</li>
                        </ul>
                    </ul>
                </li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>v_descale</td>
    </tr>
    <tr>
        <td>quant_compute_mode</td>
        <td>
            <ul>
                <li>data_type仅支持INT(Enum)</li>
                <li>取值支持1(A16C4_KV_MXFP4_SOFTMAX_FP32)、2(A16C4_KV_HIF4_SOFTMAX_FP32)</li>
            </ul>
        </td>
        <td>无</td>
    </tr>
</tbody>
</table>




layout匹配关系表：
<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 247px">
<col style="width: 132px">
<col style="width: 232px">
<col style="width: 293px">
<col style="width: 185px">
<col style="width: 119px">
<col style="width: 272px">
<col style="width: 145px">
</colgroup>
<thead>
<tr>
    <th>layout_q</th>
    <th>layout_kv</th>
    <th>layout_attn_out</th>
    <th>layout_softmax_lse</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>BNSD</td>
        <td>
          PA_BBND
          PA_BNBD
          PA_NZ
        </td>
        <td>
          BSND
          BNSD
        </td>
        <td>(B, Q_N, Q_S)</td>
    </tr>
    <tr>
        <td>BSND</td>
        <td>
          PA_BBND
          PA_BNBD
          PA_NZ
        </td>
        <td>BSND</td>
        <td>(B, Q_N, Q_S)</td>
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
                    <li>当前必须传入，未传入将发出拦截报警</li>
                </ul>
            </td>
            <td>可选参数</td>
            <td>无</td>
            <td>传入时需与mixed_quant_flash_attn_metadata生成的结果一致</td>
        </tr>
    </tbody>
</table>

#### Mask参数组

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
            <td>mask_mode</td>
            <td>
                <ul>
                    <li>data_type支持INT(Enum)</li>
                    <li>支持输入范围仅为0(ALL)、3(CAUSAL)，默认值为0(ALL)</li>
                </ul>
            </td>
            <td>
                可选输入，如果不传该参数，默认值为0
            </td>
            <td rowspan="4">
                <ul>
                    <li>当mask_mode为0(ALL)时，win_left和win_right必须为-1且不支持传入attn_mask</li>
                    <li>当mask_mode为3(CAUSAL)时，win_left和win_right必须为-1且必须传入attn_mask矩阵</li>
                </ul>
            </td>
            <td rowspan="4">
                <ul>
                    <li>无</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>attn_mask</td>
            <td>
                <ul>
                    <li>tensor_type支持INT8</li>
                    <li>不传时为空，传入时，tensor_shape维度为(2048, 2048)，该矩阵固定为2048*2048的下三角矩阵对角线全0</li>
                </ul>
            </td>
            <td rowspan="3">可选参数</td>
        </tr>
        <tr>
            <td>win_left</td>
            <td rowspan="2">
                <ul>
                    <li>data_type支持INT</li>
                    <li>不传或者传入-1,代表该值为正无穷</li>
                    <li>传入时，值需要 >= 0</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>win_right</td>
        </tr>
    </tbody>
</table>

#### SeqLengths参数组

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
            <td rowspan="2">
                <ul>
                    <li>tensor_type支持INT32</li>
                    <li>tensor_shape为(B,)</li>
                    <li>仅支持非负整数</li>
                    <li>seqused_q中的值需小于等于Q_S</li>
                    <li>seqused_kv中的值需小于等于KV_S</li>
                </ul>
            </td>
            <td rowspan="6">可选参数</td>
            <td rowspan="6">
序列长度相关参数分为3组：
<ul>
<li>A组：seqused_q、seqused_kv</li>
<li>B组：max_seqlen_q、max_seqlen_kv</li>
<li>C组：cu_seqlens_q</li>
</ul>

约束规则：
<ul>
<li>A组和B组至少传入一组（也可两组都传），C组不能传入</li>
</ul>
</td>
            <td rowspan="2">无</td>
        </tr>
        <tr>
            <td>seqused_kv</td>
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
                    <li>仅在layout_q为TND时支持配置</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>max_seqlen_q</td>
            <td rowspan="2">
                <ul>
                    <li>data_type支持INT</li>
                    <li>默认值为-1</li>
                </ul>
            </td>
            <td rowspan="2">
                <ul>
                    <li>A组(seqused_q、seqused_kv)和B组(max_seqlen_q、max_seqlen_kv)参数至少传入一组</li>
                    <li>若只传了B组(max_seqlen_q、max_seqlen_kv)参数，则max_seqlen_q需等于Q_S，max_seqlen_kv需等于KV_S</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>max_seqlen_kv</td>
        </tr>
    </tbody>
</table>

#### Paged Attention参数组
mixed_quant_flash_attn的key、value仅支持PA_BBND/PA_BNBD/PA_NZ分页注意力layout，block_table为必选参数，用于分块注意力计算中的块索引映射。
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
            <td>block_table</td>
            <td>
                <ul>
                    <li>tensor_type支持INT32</li>
                    <li>传入时，tensor_shape为(B, max_block_num_per_seq)</li>
                    <li>值只能为正整数</li>
                </ul>
            </td>
            <td>可选参数</td>
            <td>无</td>
            <td>
                <ul>
                    <li>当前仅支持PA场景，PA场景下block_table必须传入</li>
                    <li>PA场景下，必须同时传入seqused_kv</li>
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
                    <li>暂不支持，必须为空</li>
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
                </ul>
            </td>
            <td>可选属性，默认值为False</td>
            <td rowspan="2">
                <ul>
                    <li>当return_softmax_lse为False时，输出softmax_lse为空</li>
                    <li>当return_softmax_lse为True时，输出shape见layout匹配关系表</li>
                </ul>
            </td>
            <td>无</td>
        </tr>
        <tr>
            <td>softmax_lse</td>
            <td>
                <ul>
                    <li>data_type仅支持FLOAT32</li>
                </ul>
            </td>
            <td>无</td>
            <td>无</td>
        </tr>
    </tbody>
</table>

## 调用示例<a name="zh-cn_topic_0000002168254826_section14459801435"></a>

-   mixed_quant_flash_attn_metadata + mixed_quant_flash_attn 联合调用示例（BSND + PA_BBND）

    ```python
    import torch
    import torch_npu
    import npu_ops_transformer

    torch_npu.npu.set_device(0)

    q_dtype = torch.bfloat16
    B = 2
    S = 16
    Q_N = 8
    KV_N = 2
    D = 128
    block_size = 64
    block_num = 4

    q = torch.randn(B, S, Q_N, D, dtype=q_dtype, device="npu")
    k = torch.empty(block_num, block_size, KV_N, D, dtype=torch.float4_e2m1, device="npu")
    v = torch.empty(block_num, block_size, KV_N, D, dtype=torch.float4_e2m1, device="npu")
    group_size = 32
    k_descale = torch.empty(block_num, block_size, KV_N, D // group_size, dtype=torch.float8_e8m0, device="npu")
    v_descale = torch.empty(block_num, KV_N, D, block_size // group_size, dtype=torch.float8_e8m0, device="npu")
    block_table = torch.zeros(B, block_num, dtype=torch.int32, device="npu")

    metadata = npu_ops_transformer.ops.mixed_quant_flash_attn_metadata(
        Q_N,
        KV_N,
        D,
        1,
        batch_size=B,
        max_seqlen_q=S,
        max_seqlen_kv=S,
        mask_mode=0,
        win_left=-1,
        win_right=-1,
        layout_q="BSND",
        layout_kv="PA_BBND",
        layout_attn_out="BSND",
    )

    attn_out, softmax_lse = npu_ops_transformer.ops.mixed_quant_flash_attn(
        q, k, v, k_descale, v_descale,
        block_table=block_table,
        metadata=metadata,
        quant_compute_mode=1,
        softmax_scale=1.0 / (D ** 0.5),
        mask_mode=0,
        win_left=-1,
        win_right=-1,
        layout_q="BSND",
        layout_kv="PA_BBND",
        layout_attn_out="BSND",
        return_softmax_lse=False,
    )
    torch_npu.npu.synchronize()
    assert attn_out.shape == q.shape
    assert attn_out.dtype == q.dtype
    assert torch.isfinite(attn_out.float()).all().item()
    ```