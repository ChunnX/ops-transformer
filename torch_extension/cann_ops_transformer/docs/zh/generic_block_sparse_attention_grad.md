# generic_block_sparse_attention_grad

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
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

  `generic_block_sparse_attention_grad`是基于`TorchNPU`的`cann_ops_transformer`扩展接口，用于调用`GenericBlockSparseAttentionGrad`算子完成通用块稀疏注意力的反向梯度计算。通过`rsvd_block_idx`指定每个KV块选择的Q块/token索引，`rsvd_block_count`指定每个KV块实际保留的Q数量，仅在被选中的稀疏块上计算并回传`dq`/`dk`/`dv`。

  `generic_block_sparse_attention_grad_metadata`是`generic_block_sparse_attention_grad`的元数据生成接口，用于在主算子执行前生成`metadata`。`metadata`记录AICore任务切分与负载均衡结果，主算子必须传入该`metadata`。典型调用流程如下：

  1. 准备`query`、`key`、`value`、`dout`、`out`、`lse`、`rsvd_block_idx`、`rsvd_block_count`、`cu_seq_lengths`、`cu_seq_lengths_kv`等输入。
  2. 调用`generic_block_sparse_attention_grad_metadata`生成`metadata`。
  3. 调用`generic_block_sparse_attention_grad`，将上一步得到的`metadata`传入主算子。

- **计算公式**：

  稀疏块大小为$blockShapeX \times blockShapeY$。依据`rsvd_block_idx`/`rsvd_block_count`选取参与计算的Q–KV块对后：

  $$
  P = SimpleSoftmax(Mask(Q @ selectedK^{T} \cdot scale), lse)
  $$

  $$
  dP = dO @ selectedV^{T}
  $$

  $$
  dS = P \odot (dP - SoftmaxGrad(dO, O))
  $$

  $$
  dQ = dS @ selectedK \cdot scale
  $$

  $$
  dK = dS^{T} @ Q \cdot scale
  $$

  $$
  dV = P^{T} @ dO
  $$

  其中$scale$由`scale_value`指定，建议值为$D^{-0.5}$。

> [!NOTE]
>
> query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸headdim。
> TND中的N为query的headNum时记为N1（Q_N），key/value的headNum记为N2（KV_N）。T1表示所有query Batch输入样本序列长度的累加和；T2表示所有kv Batch输入样本序列长度的累加和。
> $J = \lceil max\_kv\_seqlen / blockShapeY \rceil$。与正向`generic_block_sparse_attention`不同，本反向接口的稀疏索引为**KV→Q**方向（`rsvd_block_*`），而非Q→KV方向（`sparse_block_*`）。

## 函数原型

调用`generic_block_sparse_attention_grad`接口之前，请先调用前置接口`generic_block_sparse_attention_grad_metadata`，完成负载均衡与分核信息的计算。

```python
cann_ops_transformer.generic_block_sparse_attention_grad_metadata(
    rsvd_block_idx,
    rsvd_block_count,
    *,
    cu_seq_lengths=None,
    cu_seq_lengths_kv=None,
    seqused_q=None,
    seqused_kv=None,
    max_q_seqlen=0,
    max_kv_seqlen=0,
    num_q_heads=0,
    num_kv_heads=0,
    head_dim=128,
    block_shape=[1, 128],
    is_packed_gqa=1,
    q_input_layout="TND",
    kv_input_layout="TND",
    mask_type=0,
    softmax_precision=0,
    window_size_left=-1,
    window_size_right=-1,
) -> Tensor
```

```python
cann_ops_transformer.generic_block_sparse_attention_grad(
    query,
    key,
    value,
    dout,
    out,
    lse,
    rsvd_block_idx,
    rsvd_block_count,
    metadata,
    *,
    atten_mask=None,
    cu_seq_lengths=None,
    cu_seq_lengths_kv=None,
    seqused_q=None,
    seqused_kv=None,
    block_shape=[0],
    is_packed_gqa=1,
    q_input_layout="TND",
    kv_input_layout="TND",
    scale_value=1.0,
    mask_type=0,
    softmax_precision=0,
    window_size_left=-1,
    window_size_right=-1,
) -> (Tensor, Tensor, Tensor)
```

## 参数说明

### generic_block_sparse_attention_grad_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| rsvd_block_idx | Tensor | 必选 | 稀疏块索引数组，指定每个KV块选择的Q块/token索引 | int32 | ND | TND：`(B, N2, J, maxS1)`；BNSD/BSND：`(B, N2, J, S1)` |
| rsvd_block_count | Tensor | 必选 | 指定每个KV块实际选择的Q数量 | int32 | ND | `(B, N2, J)` |
| cu_seq_lengths | Tensor | 可选 | query累积序列长度，第一个元素必须为0 | int64 | ND | `(B+1,)` |
| cu_seq_lengths_kv | Tensor | 可选 | key/value累积序列长度，第一个元素必须为0 | int64 | ND | `(B+1,)` |
| seqused_q | Tensor | 可选 | 各batch中query实际使用的序列长度 | int32 | ND | `(B,)` |
| seqused_kv | Tensor | 可选 | 各batch中kv实际使用的序列长度 | int32 | ND | `(B,)` |
| max_q_seqlen | int | 必选 | 所有batch中query序列长度的最大值 | int64 | - | - |
| max_kv_seqlen | int | 必选 | 所有batch中kv序列长度的最大值，用于计算$J$ | int64 | - | - |
| num_q_heads | int | 必选 | Query head数（N1） | int64 | - | - |
| num_kv_heads | int | 必选 | Key/Value head数（N2） | int64 | - | - |
| head_dim | int | 可选 | 每个注意力头的维度，当前固定128，默认128 | int64 | - | - |
| block_shape | list[int] | 可选 | 稀疏块形状`[block_x, block_y]`，当前仅支持`[1, 128]`，默认`[1, 128]` | int64 | - | 长度为2 |
| is_packed_gqa | int | 可选 | 同一group内qHead是否共享稀疏pattern，当前仅支持1，默认1 | int64 | - | - |
| q_input_layout | string | 可选 | query布局，支持`"TND"`/`"BNSD"`/`"BSND"`，默认`"TND"` | string | - | - |
| kv_input_layout | string | 可选 | key/value布局，须与`q_input_layout`一致，默认`"TND"` | string | - | - |
| mask_type | int | 可选 | 掩码类型，当前仅支持1（RIGHT_DOWN_CAUSAL） | int64 | - | - |
| softmax_precision | int | 可选 | Softmax精度级别，取值0或1，当前实现传0，默认0 | int64 | - | - |
| window_size_left | int | 可选 | 滑窗向前包含token数，不使能时必须为-1，默认-1 | int64 | - | - |
| window_size_right | int | 可选 | 滑窗向后包含token数，不使能时必须为-1，默认-1 | int64 | - | - |

### generic_block_sparse_attention_grad

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| query | Tensor | 必选 | 公式中的$Q$ | bfloat16/float16 | ND | TND：`(T1, N1, D)`；BNSD：`(B, N1, S1, D)`；BSND：`(B, S1, N1, D)` |
| key | Tensor | 必选 | 公式中的$K$ | bfloat16/float16 | ND | TND：`(T2, N2, D)`；BNSD：`(B, N2, S2, D)`；BSND：`(B, S2, N2, D)` |
| value | Tensor | 必选 | 公式中的$V$，shape与key相同 | bfloat16/float16 | ND | 与key一致 |
| dout | Tensor | 必选 | 注意力输出梯度$dO$ | bfloat16/float16 | ND | 与query一致 |
| out | Tensor | 必选 | 注意力正向输出$O$ | bfloat16/float16 | ND | 与query一致 |
| lse | Tensor | 必选 | 正向保存的softmax LSE | float32 | ND | TND：`(T1, N1, 1)`；BNSD：`(B, N1, S1, 1)`；BSND：`(B, S1, N1, 1)` |
| rsvd_block_idx | Tensor | 必选 | 稀疏块索引数组（KV→Q） | int32 | ND | 与metadata接口一致 |
| rsvd_block_count | Tensor | 必选 | 每个KV块实际选择的Q数量 | int32 | ND | 与metadata接口一致 |
| metadata | Tensor | 必选 | `generic_block_sparse_attention_grad_metadata`生成的分核结果 | int64 | ND | `(metaSize,)`，见返回值说明 |
| atten_mask | Tensor | 可选 | 掩码矩阵，当前暂不支持，应传`None` | bool | ND | - |
| cu_seq_lengths | Tensor | 可选 | query累积序列长度；`q_input_layout="TND"`时必传 | int64 | ND | `(B+1,)` |
| cu_seq_lengths_kv | Tensor | 可选 | kv累积序列长度；`kv_input_layout="TND"`时必传 | int64 | ND | `(B+1,)` |
| seqused_q | Tensor | 可选 | 各batch中query实际使用的序列长度 | int32 | ND | `(B,)` |
| seqused_kv | Tensor | 可选 | 各batch中kv实际使用的序列长度 | int32 | ND | `(B,)` |
| block_shape | list[int] | 可选 | 稀疏块形状`[block_x, block_y]`，当前仅支持`[1, 128]` | int64 | - | 长度为2 |
| is_packed_gqa | int | 可选 | Packed GQA开关，当前仅支持1，默认1 | int64 | - | - |
| q_input_layout | string | 可选 | query布局，支持`"TND"`/`"BNSD"`/`"BSND"`，默认`"TND"` | string | - | - |
| kv_input_layout | string | 可选 | key/value布局，须与`q_input_layout`一致，默认`"TND"` | string | - | - |
| scale_value | float | 可选 | 缩放因子，建议值$1/\sqrt{D}$，默认1.0 | float32 | - | - |
| mask_type | int | 可选 | 掩码类型，当前仅支持1 | int64 | - | - |
| softmax_precision | int | 可选 | Softmax精度级别，取值0或1，当前实现传0，默认0 | int64 | - | - |
| window_size_left | int | 可选 | 滑窗向前包含token数，不使能时必须为-1，默认-1 | int64 | - | - |
| window_size_right | int | 可选 | 滑窗向后包含token数，不使能时必须为-1，默认-1 | int64 | - | - |

## 返回值说明

### generic_block_sparse_attention_grad_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| metadata | Tensor | 必选 | 主算子分核与负载均衡数据 | int64 | ND | `(metaSize,)` |

其中：

$$
metaSize = 80 + B \times N1 \times J \times 4,\quad J = \lceil max\_kv\_seqlen / blockShapeY \rceil
$$

### generic_block_sparse_attention_grad

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| dq | Tensor | 必选 | query的梯度 | bfloat16/float16 | ND | 与query一致 |
| dk | Tensor | 必选 | key的梯度 | bfloat16/float16 | ND | 与key一致 |
| dv | Tensor | 必选 | value的梯度 | bfloat16/float16 | ND | 与value一致 |

**说明**

- `dq`/`dk`/`dv`的dtype与对应输入`query`/`key`/`value`保持一致。
- `metadata`长度须满足`shape[0] ≥ 80 + B × num_q_heads × J × 4`；任务数上界$B \times N1 \times J \le 1048576$。

## 约束说明

- 确定性计算：`generic_block_sparse_attention_grad`默认为非确定性实现，暂不支持确定性实现，确定性计算配置后不会生效。
- 参数`cu_seq_lengths`、`cu_seq_lengths_kv`、`seqused_q`、`seqused_kv`、`rsvd_block_idx`、`rsvd_block_count`属于tensor。由于算子在Tiling阶段无法获取tensor的具体数值，tiling侧不对值进行校验，正确性需要用户自行保证。若上述参数传入非法值，会触发未定义行为（精度问题、非法内存访问导致的程序崩溃等）。
- `generic_block_sparse_attention_grad_metadata`和`generic_block_sparse_attention_grad`的入参在调用时应该保持一致。由于算子分为两个接口分段调用，算子无法自行校验，正确性需要由客户自行保证。若接口传入参数不一致，会发生未定义行为（精度问题、非法内存访问导致的程序崩溃等）。
- 须先成功调用`generic_block_sparse_attention_grad_metadata`生成`metadata`，再调用主算子；`metadata`为必选输入。

### 特性参数组

|      特性参数组      |     参数字段名称     |    字段分组    |  字段类型  |
| :-------------------: | :-------------------: | :-------------: | :--------: |
|      公共参数组      |       query       |      INPUT      |   Tensor   |
|                      |         key         |      INPUT      |   Tensor   |
|                      |        value        |      INPUT      |   Tensor   |
|                      |        dout         |      INPUT      |   Tensor   |
|                      |         out         |      INPUT      |   Tensor   |
|                      |         lse         |      INPUT      |   Tensor   |
|                      |  rsvd_block_idx  |      INPUT      |   Tensor   |
|                      | rsvd_block_count  |      INPUT      |   Tensor   |
|                      |      metadata      |      INPUT      |   Tensor   |
|                      |     block_shape     |   ATTR(OPTIONAL) |  int[]  |
|                      |    scale_value    | ATTR(OPTIONAL) |   float   |
|                      |   q_input_layout   | ATTR(OPTIONAL) |   string   |
|                      |  kv_input_layout  | ATTR(OPTIONAL) |   string   |
|                      |         dq         |     OUTPUT     |   Tensor   |
|                      |         dk         |     OUTPUT     |   Tensor   |
|                      |         dv         |     OUTPUT     |   Tensor   |
|   metadata参数组   |   max_q_seqlen   | ATTR(REQUIRED) |   int   |
|                      |  max_kv_seqlen  | ATTR(REQUIRED) |   int   |
|                      |   num_q_heads   | ATTR(REQUIRED) |   int   |
|                      |  num_kv_heads  | ATTR(REQUIRED) |   int   |
|                      |     head_dim     | ATTR(OPTIONAL) |   int   |
|                      |      metadata      |     OUTPUT     |   Tensor   |
|      Mask参数组      |      mask_type      | ATTR(OPTIONAL) |   int   |
|                      |  window_size_left  | ATTR(OPTIONAL) |   int   |
|                      | window_size_right | ATTR(OPTIONAL) |   int   |
|                      |     atten_mask     | INPUT(OPTIONAL) |   Tensor   |
| SeqLens参数组  |   cu_seq_lengths   | INPUT(OPTIONAL) |  Tensor  |
|                      |  cu_seq_lengths_kv  | INPUT(OPTIONAL) |  Tensor  |
|                      |    seqused_q    | INPUT(OPTIONAL) |  Tensor  |
|                      |   seqused_kv   | INPUT(OPTIONAL) |  Tensor  |
|   Softmax参数组   |  softmax_precision  | ATTR(OPTIONAL) |   int   |
|                      |   is_packed_gqa   | ATTR(OPTIONAL) |   int   |

### 基准信息说明

资料约束中，常见字段释义如下：

|    命名    |                            含义                            |
| :---------: | :---------------------------------------------------------: |
|      B      |                Batch,表示输入样本批量大小                |
|     N1 / Q_N     |        输入query tensor的头数        |
|    N2 / KV_N    |    输入key/value tensor的头数    |
|     S1 / Q_S     |      输入query序列长度      |
|    S2 / KV_S    |  输入key/value序列长度  |
|     T1     |          输入query所有batch序列长度的累加和          |
|     T2     |          输入kv所有batch序列长度的累加和          |
|     D     |          headdim，当前固定128         |
|      J      | $\lceil max\_kv\_seqlen / blockShapeY \rceil$，KV块数 |
|   maxS1   | `rsvd_block_idx`最后一维容量，须≥`max_q_seqlen` |

### 参数组约束

#### 公共参数组

- 入参为空的场景处理：
  - 空Tensor指必选输入和输出的shape size为0，即有任意轴为0。
  - 触发空Tensor的用例将全部拦截报错。

- query、key、value、dout、out、dq、dk、dv校验:

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
        <td>query</td>
        <td>
            <ul>
                <li>tensor_type支持bfloat16、float16</li>
                <li>TND → (T1, N1, D)；BNSD → (B, N1, S1, D)；BSND → (B, S1, N1, D)</li>
            </ul>
        </td>
        <td rowspan="5">必须存在</td>
        <td rowspan="5">
            <ul>
                <li>query、key、value、dout、out、dq、dk、dv的数据类型需相同</li>
                <li>dout、out、dq的shape与query一致</li>
                <li>key、value、dk、dv的shape需相同</li>
                <li>q_input_layout与kv_input_layout须一致，支持TND/BNSD/BSND</li>
            </ul>
        </td>
        <td rowspan="5">
            轴校验：
            <ul>
                <li>当前D仅支持128</li>
                <li>N1、N2取值范围[1, 128]</li>
                <li>N1 % N2 == 0</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>key / value</td>
        <td>
            <ul>
                <li>tensor_type支持bfloat16、float16</li>
                <li>TND → (T2, N2, D)；BNSD → (B, N2, S2, D)；BSND → (B, S2, N2, D)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>dout / out</td>
        <td>
            <ul>
                <li>dtype与query一致</li>
                <li>shape与query一致</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>dq</td>
        <td>shape/dtype与query一致</td>
    </tr>
    <tr>
        <td>dk / dv</td>
        <td>shape/dtype与key一致</td>
    </tr>
    </tbody>
    </table>

- lse校验:

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
        <td>lse</td>
        <td>
            <ul>
                <li>tensor_type仅支持float32</li>
                <li>TND：(T1, N1, 1)；BNSD：(B, N1, S1, 1)；BSND：(B, S1, N1, 1)</li>
            </ul>
        </td>
        <td>必须存在</td>
        <td>head/seq轴语义须与query布局一致</td>
        <td>无</td>
    </tr>
    </tbody>
    </table>

- rsvd_block_idx、rsvd_block_count、block_shape校验:

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
        <td>rsvd_block_idx</td>
        <td>
            <ul>
                <li>tensor_type仅支持int32</li>
                <li>TND：(B, N2, J, maxS1)；BNSD/BSND：(B, N2, J, S1)</li>
            </ul>
        </td>
        <td rowspan="3">必须存在</td>
        <td rowspan="3">
            <ul>
                <li>metadata接口与主算子须传入相同的rsvd_block_idx、rsvd_block_count、block_shape</li>
                <li>J须等于ceilDiv(max_kv_seqlen, block_y)</li>
                <li>maxS1 / S1须≥max_q_seqlen；且≥rsvd_block_count中所有元素的最大值</li>
            </ul>
        </td>
        <td rowspan="3">
            <ul>
                <li>当前block_shape=[1, 128]</li>
                <li>is_packed_gqa当前仅支持1</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>rsvd_block_count</td>
        <td>
            <ul>
                <li>tensor_type仅支持int32</li>
                <li>shape为(B, N2, J)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>block_shape</td>
        <td>
            <ul>
                <li>长度为2的int列表</li>
            </ul>
        </td>
    </tr>
    </tbody>
    </table>

- metadata校验:

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
                        <li>tensor_type仅支持int64</li>
                        <li>shape[0] ≥ 80 + B×N1×J×4</li>
                    </ul>
                </td>
                <td>必须存在</td>
                <td>须为generic_block_sparse_attention_grad_metadata的输出</td>
                <td>任务数上界B×N1×J ≤ 1048576</td>
            </tr>
        </tbody>
    </table>

#### Mask参数组

mask_type参数解释：

- mask_type=0：不加mask（当前不支持）
- mask_type=1：RIGHT_DOWN_CAUSAL（当前默认且唯一支持）
- window_size_left / window_size_right：不使能时必须为-1；atten_mask当前应传`None`

#### SeqLens参数组

- `q_input_layout="TND"`时，`cu_seq_lengths`必选；`kv_input_layout="TND"`时，`cu_seq_lengths_kv`必选。
- BNSD/BSND布局下，上述累积序列长度传`None`。
- `cu_seq_lengths`/`cu_seq_lengths_kv`首元素必须为0，末元素分别等于T1/T2。

#### Softmax参数组

- `softmax_precision`仅支持0或1；当前实现传0。
- `is_packed_gqa`当前仅支持1。

## 调用示例

- generic_block_sparse_attention_grad_metadata + generic_block_sparse_attention_grad联合调用示例（BNSD）

    ```python
    import math
    import torch
    import torch_npu
    import cann_ops_transformer

    torch_npu.npu.set_device(0)

    B, N1, N2, S1, S2, D = 1, 1, 1, 128, 128, 128
    block_x, block_y = 1, 128
    J = math.ceil(S2 / block_y)
    scale = 1.0 / math.sqrt(D)
    block_shape = [block_x, block_y]

    query = torch.randn(B, N1, S1, D, dtype=torch.float16, device="npu")
    key = torch.randn(B, N2, S2, D, dtype=torch.float16, device="npu")
    value = torch.randn(B, N2, S2, D, dtype=torch.float16, device="npu")
    dout = torch.randn(B, N1, S1, D, dtype=torch.float16, device="npu")
    out = torch.randn(B, N1, S1, D, dtype=torch.float16, device="npu")
    lse = torch.randn(B, N1, S1, 1, dtype=torch.float32, device="npu")

    rsvd_block_idx = torch.full((B, N2, J, S1), -1, dtype=torch.int32, device="npu")
    rsvd_block_count = torch.zeros((B, N2, J), dtype=torch.int32, device="npu")
    for q in range(S1):
        rsvd_block_idx[0, 0, 0, q] = q
    rsvd_block_count[0, 0, 0] = S1

    metadata = cann_ops_transformer.generic_block_sparse_attention_grad_metadata(
        rsvd_block_idx,
        rsvd_block_count,
        max_q_seqlen=S1,
        max_kv_seqlen=S2,
        num_q_heads=N1,
        num_kv_heads=N2,
        head_dim=D,
        block_shape=block_shape,
        is_packed_gqa=1,
        q_input_layout="BNSD",
        kv_input_layout="BNSD",
        mask_type=1,
        softmax_precision=0,
        window_size_left=-1,
        window_size_right=-1,
    )

    dq, dk, dv = cann_ops_transformer.generic_block_sparse_attention_grad(
        query,
        key,
        value,
        dout,
        out,
        lse,
        rsvd_block_idx,
        rsvd_block_count,
        metadata,
        atten_mask=None,
        block_shape=block_shape,
        is_packed_gqa=1,
        q_input_layout="BNSD",
        kv_input_layout="BNSD",
        scale_value=scale,
        mask_type=1,
        softmax_precision=0,
        window_size_left=-1,
        window_size_right=-1,
    )
    torch_npu.npu.synchronize()
    assert dq.shape == query.shape
    assert dk.shape == key.shape
    assert dv.shape == value.shape
    assert dq.dtype == query.dtype
    ```
