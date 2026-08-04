# lightning\_indexer

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

  `lightning_indexer_metadata`接口用于生成一个任务列表，包含每个AIcore的Attention计算任务的起止点的Batch、Head、以及Q和K的分块的索引，供后续`lightning_indexer`算子使用。

  `lightning_indexer`接口基于一系列操作得到每一个token对应的top-k个位置。主要计算过程为：

  1. 将某个token对应的输入参数`q`（$Q_{index}\in\R^{g\times d}$）乘以给定上下文`k`（$K_{index}\in\R^{S_{k}\times d}$），得到相关性。
  2. 通过激活函数$ReLU$过滤无效负相关信号后，得到当前Token与所有前序Token的相关性分数向量。
  3. 将其与权重系数`w`（$W$）相乘后，沿g的方向，选取前$Top-k$个索引值得到输出$sparseIndices$，并输出对应的$sparseValues$，作为Attention的输入。

- **计算公式**：

  $$
  Top-k \left\{  \left[ 1 \left] \mathop{{}}\nolimits_{{1 \times \text{ }g}}\text{@} \left[  \left( W\text{@} \left[ 1 \left] \mathop{{}}\nolimits_{{1\text{ } \times \text{ }S\mathop{{}}\nolimits_{{k}}}} \left) \text{ } \odot \text{ }ReLU \left( Q\mathop{{}}\nolimits_{{index}}\text{@}K\mathop{{}}\nolimits_{{T}}^{{index}} \left)  \left]  \right\} \right. \right. \right. \right. \right. \right. \right. \right. \right. \right.
  $$

## 函数原型

调用lightning_indexer接口之前，先调用前置接口lightning_indexer_metadata，完成lightning_indexer负载均衡的计算。

```python
cann_ops_transformer.lightning_indexer_metadata(
  num_heads_q,
  num_heads_k,
  head_dim,
  topk,
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

```python
cann_ops_transformer.lightning_indexer(
  q,
  k,
  w,
  topk,
  *,
  cu_seqlens_q=None,
  cu_seqlens_k=None,
  seqused_q=None,
  seqused_k=None,
  cmp_residual_k=None,
  block_table=None,
  output_idx_offset=None,
  metadata=None,
  max_seqlen_q=-1,
  layout_q="BSND",
  layout_k="BSND",
  mask_mode=0,
  cmp_ratio=1,
  return_value=0
) -> (Tensor, Tensor)
```

## 参数说明

>**说明：**<br>
>
>- q、k、w参数维度含义：B（Batch Size）表示输入样本批量大小、S1表示q的输入样本序列长度、S2表示k的输入样本序列长度、N1表示q的多头数、N2表示k的多头数、D（Head Dim）表示注意力头的维度、T1表示q的输入样本序列长度的累加和、T2表示k的输入样本序列长度的累加和。参数q中的D和参数k中的D值相等，当前仅支持128。

### lightning_indexer_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| num_heads_q | int | 必选 | 表示Query的head个数，当前仅支持32/64。 | int32 | - |
| num_heads_k | int | 必选 | 表示Key的head个数，当前仅支持1。 | int32 | - |
| head_dim | int | 必选 | 表示注意力头的维度，当前仅支持128。 | int32 | - |
| topk | int | 必选 | 表示从Query中筛选出的关键稀疏token的个数，当前支持[1, 8192]。 | int32 | - |
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

### lightning_indexer

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| q | Tensor | 必选 | 公式中的输入Q。不支持空tensor。数据格式为ND。 | bfloat16、float16 | layout_q为BSND时shape为(B,S1,N1,D)；layout_q为TND时shape为(T1,N1,D) |
| k | Tensor | 必选 | 公式中的输入K。不支持空tensor。数据格式为ND，支持非连续的Tensor（仅PA_BBND场景下0轴支持非连续）。 | bfloat16、float16 | layout_k为BSND时shape为(B,S2,N2,D)；layout_k为TND时shape为(T2,N2,D)；layout_k为PA_BBND时shape为(block_num,block_size,N2,D) |
| w | Tensor | 必选 | 公式中的输入W。不支持空tensor。数据格式为ND。 | float | layout_q为BSND时shape为(B,S1,N1)；layout_q为TND时shape为(T1,N1) |
| topk | int | 必选 | topK阶段需要保留的block数量，当前支持[1, 8192]。 | int32 | - |
| cu_seqlens_q | Tensor | 可选 | 当前Batch及前序Batch中q的有效token数的累加和。仅layout_q为TND场景下必传，第一个值固定为0。数据格式为ND。 | int32 | (B+1,) |
| cu_seqlens_k | Tensor | 可选 | 当前Batch及前序Batch中k的有效token数的累加和。仅layout_k为TND场景下必传，第一个值固定为0。数据格式为ND。 | int32 | (B+1,) |
| seqused_q | Tensor | 可选 | 不同Batch中q的真实使用长度。数据格式为ND。 | int32 | (B,) |
| seqused_k | Tensor | 可选 | 不同Batch中k的真实使用长度。数据格式为ND。 | int32 | (B,) |
| cmp_residual_k | Tensor | 可选 | 表示k压缩前token数量除以cmp_ratio的余数。需要在mask_mode等于3、cmp_ratio不等于1的场景下使用。数据格式为ND。 | int32 | (B,) |
| block_table | Tensor | 可选 | 表示PageAttention中KV存储使用的block映射表。不支持空tensor。数据格式为ND。 | int32 | (B, S2_max/block_size) |
| output_idx_offset | Tensor | 可选 | 表示topK结果输出索引所需要加上的偏移。值必须大于0，加上偏移后topK index不能超过int32最大值。数据格式为ND。 | int32 | (B,) |
| metadata | Tensor | 可选 | 由lightning_indexer_metadata得到的分核信息，包含使用核数、分块大小以及每个核处理数据的起始点等内容。不支持空tensor。数据格式为ND。| int32 | (1024,) |
| max_seqlen_q | int | 可选 | q的最大序列长度。-1表示任意可能长度，默认值为-1。 | int32 | - |
| layout_q | str | 可选 | 用于标识输入q的数据排布格式，支持BSND、TND，默认值为BSND。 | string | - |
| layout_k | str | 可选 | 用于标识输入k的数据排布格式，支持BSND、TND、PA_BBND，默认值为BSND。 | string | - |
| mask_mode | int | 可选 | 表示mask的模式，0代表defaultMask模式，3代表rightDownCausal模式，默认值为0。 | int32 | - |
| cmp_ratio | int | 可选 | 用于稀疏计算，表示k的压缩倍数。支持1-128，默认值为4。 | int32 | - |
| return_value | int | 可选 | 代表是否需要返回Indices对应的Values值。0代表不返回，1代表返回值，默认值为0。 | int32 | - |

## 返回值说明

### lightning_indexer_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| metadata | Tensor | 必选 | 每个AIcore的Attention计算任务的Batch、Head、以及Q和K的分块的索引。数据格式为ND，不支持非连续的Tensor。 | int32 | shape为(1024,)  |

### lightning_indexer

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| sparse_indices | Tensor | 必选 | 公式中的Indices输出。不支持空tensor。无效部分填-1。数据格式为ND。 | int32 | layout_q为BSND时shape为(B,S1,N2,topk)；layout_q为TND时shape为(T1,N2,topk) |
| sparse_values | Tensor | 条件输出 | 公式中的Indices对应的Values输出。当return_value为1时输出对应值；当return_value为0时输出shape为[0]的空tensor。无效部分填-inf。数据格式为ND。 | float | layout_q为BSND时shape为(B,S1,N2,topk)；layout_q为TND时shape为(T1,N2,topk)；return_value为0时shape为(0,) |

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和TorchAir（aclgraph）图模式调用。
- lightning_indexer_metadata接口需与lightning_indexer算子配套使用。
- B（Batch）表示输入样本批量大小。
- 参数cu_seqlens_q、cu_seqlens_k要求其值为当前Batch与前序Batch有效token数的累加值，后一个元素的值必须大于等于前一个元素的值。
- 参数seqused_q、seqused_k要求其值表示每个Batch中的有效token数。
- 参数cmp_residual_k需满足cmp_residual_k\[i\] < cmp_ratio。
- mask_mode所表示的mask模式的详细介绍见[sparse_mode参数说明](../../../../docs/zh/context/sparse_mode_introduction.md)。
- pa_kv_cache支持0轴非连续；pa_block_size支持1~1024，满足block大小32B对齐。
- 参数q、k的数据类型应保持一致。
- sparse_indices无效部分填-1；sparse_values无效部分填-inf。
<!-- npu="A3,910b" id7 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>:
  - topk取值范围当前仅支持[1, 2048]，以及3072、4096、5120、6144、7168、8192。
  - 当前不支持seqused_q、output_idx_offset、max_seqlen_q功能，不建议传入这些参数。
  - 当layout_k为PA_BBND时，必须传入seqused_k；当layout_k不为PA_BBND时，不支持seqused_k功能，不建议传入该参数。
<!-- end id7 -->
<!-- npu="950" id8 -->
- <term>Ascend 950PR/Ascend 950DT</term>:
  - 当layout_q为BSND时，不支持传入cu_seqlens_q；当layout_k为BSND或PA_BBND时，不支持传入cu_seqlens_k。
  - 当传入output_idx_offset时，只支持大于0的索引偏移值；且应满足约束：加上传入的索引偏移值后，得到的sparseIndice值不超过INT32的最大值。
  - 当layout_q为TND时，必须传入cu_seqlens_q，如果也传入seqused_q，应保证由seqused_q传入的各个batch的query长度不超过根据cu_seqlens_q计算出的各个batch的q序列长度。当某个batch由seqused_q传入的q序列长度seqlen1小于由cu_seqlens_q计算出的query长度seqlen2时，会启用TND Padding功能，将该batch的从seqlen1 + 1到seqlen2的query输出的sparse_indices和sparse_values全部置为无效值。
  - 当传入的cmp_ratio > 1且mask_mode = 3时，必须传入cmp_residual_k，其余情况不传入。
  - 参数metadata必须传入。
<!-- end id8 -->

### 特性参数组

|      特性参数组      |     参数字段名称     |    字段分组    |  字段类型  |
| :-------------------: | :-------------------: | :-------------: | :--------: |
|      公共参数组      |         q         |      INPUT      |   Tensor   |
|                      |          k          |      INPUT      | Tensor |
|                      |          w          |      INPUT      | Tensor |
|                      |         metadata        |      INPUT(OPTIONAL)      | Tensor |
|                      |         output_idx_offset  |      INPUT(OPTIONAL)      | Tensor |
|                      |      topk      | ATTR(REQUIRED) |   int   |
|                      |      layout_q      | ATTR(OPTIONAL) |   string   |
|                      |      layout_k      | ATTR(OPTIONAL) |   string   |
|                      |     sparse_indices     |     OUTPUT     |   Tensor   |
|                      |     sparse_values     |     OUTPUT     |   Tensor   |
|      Mask参数组      |       mask_mode       | ATTR(OPTIONAL) |   int   |
| SeqLens参数组  |   cu_seqlens_q   | INPUT(OPTIONAL) |  Tensor  |
|                      |  cu_seqlens_k  | INPUT(OPTIONAL) |  Tensor  |
|                      |  seqused_q  | INPUT(OPTIONAL) |  Tensor  |
|                      |  seqused_k  | INPUT(OPTIONAL) |  Tensor  |
|                      |  max_seqlen_q  | ATTR(OPTIONAL) |  int  |
| Paged Attention参数组 |      block_table      | INPUT(OPTIONAL) |   Tensor   |
|   compress参数组   |    cmp_ratio    | ATTR(OPTIONAL) |    int    |
|                      |      cmp_residual_k      |     INPUT(OPTIONAL)     |   Tensor   |

### 基准信息说明

资料约束中，常见字段释义如下：

|    命名    |                            含义                            |
| :---------: | :---------------------------------------------------------: |
|      B      |                Batch,表示输入样本批量大小                |
|     Q_N     |        输入q tensor的头数，对应q shape中的N        |
|    K_N    |    输入k tensor的头数，对应k/v shape中的N    |
|     Q_S     |      输入q tensor的序列长度，对应q shape中的S      |
|    K_S    |  输入k tensor的序列长度，对应k/v shape中的S  |
|     Q_T     |          输入q tensor所有batch序列长度的累加和          |
|     K_T     |          输入k tensor所有batch序列长度的累加和          |
|     D     |          输入q/k tensor隐藏层最小的单元尺寸headdim         |
|     topk     |          表示输出前topk个索引         |

### 参数组约束

#### 公共参数组
- 入参为空的场景处理：
    - 空Tensor指必选输入和输出的shape size为0,即有任意轴为0。
    - 触发空tensor的用例将全部拦截报错。

- q、k、sparse_indices、sparse_values校验
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
                <li>BSND -> (B, S_Q, N_Q, D)</li>
                <li>TND -> (T_Q, N_Q, D)</li>
            </ul>
        </td>
        <td rowspan="4">
            必须存在
        </td>
        <td rowspan="4">
            <ul>
                <li>q、k的数据类型需相同</li>
                <li>Layout校验规则见layout匹配关系表</li>
            </ul>
        </td>
        <td rowspan="4">
            轴校验：
            <ul>
                <li>65536 > B > 0</li>
                <li>Q_T > 0</li>
                <li>K_T > 0</li>
                <li>Q_N > 0</li>
                <li>K_N = 1</li>
                <li>Q_S > 0</li>
                <li>K_S > 0</li>
                <li>D = 128</li>
                <li>Q_N % K_N == 0且Q_N / K_N > 0</li>
                <li>1 <= topk <= 8192</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>k</td>
        <td rowspan="1">
            <ul>
                <li>tensor_type支持BFLOAT16和FLOAT16</li>
                <li>BSND -> (B, S_K, N_K, D)</li>
                <li>TND -> (T_K, N_K, D)</li>
                <li>PA_BBND -> (num_blocks, block_size, K_N, D)</li>
                <li>1024 >= block_size >= 16，block_size % 16 == 0</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>sparse_indices</td>
        <td rowspan="1">
            <ul>
                <li>tensor_type支持INT32</li>
                <li>layout_q为BSND时，sparse_indices的shape为(B, Q_S, K_N, topk)</li>
                <li>layout_q为TND时，sparse_indices的shape为(Q_T, K_N, topk)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>sparse_values</td>
        <td rowspan="1">
            <ul>
                <li>tensor_type支持BFLOAT16</li>
                <li>layout_q为BSND时，sparse_indices的shape为(B, Q_S, K_N, topk)</li>
                <li>layout_q为TND时，sparse_indices的shape为(Q_T, K_N, topk)</li>
            </ul>
        </td>
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
    <th>layout_k</th>
    <th>layout_out</th>
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
    </tr>
    <tr>
        <td>TND</td>
        <td>
          <li>TND</li>
          <li>PA_BBND</li>
        </td>
        <td>TND</td>
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
                    <li>shape由lightning_indexer_v2_metadata动态计算</li>
                    <li>当前不支持不传入，未传入将发出拦截报警</li>
                </ul>
            </td>
            <td>可选参数</td>
            <td>无</td>
            <td>传入时需与lightning_indexer_v2_metadata生成的结果一致</td>
        </tr>
    </tbody>
</table>

mask_mode参数解释
<ul>
    <li>mask_mode=0，全计算模式（默认值）</li>
    <li>mask_mode=3，Causal模式</li>
</ul>

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
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>mask_mode</td>
            <td>
                <ul>
                    <li>data_type支持INT</li>
                    <li>支持输入范围仅为0、3，默认值为0</li>
                </ul>
            </td>
            <td>
                可选输入，如果不传该参数，默认值为0
            </td>
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
                    <li>seqused_k中的值需小于等于K_S</li>
                </ul>
            </td>
            <td rowspan="6">可选参数</td>
            <td rowspan="6">无</td>
            <td rowspan="2">无</td>
        </tr>
        <tr>
            <td>seqused_k</td>
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
        </tr>
        <tr>
            <td>cu_seqlens_k</td>
            <td>
                <ul>
                    <li>tensor_type支持INT32</li>
                    <li>tensor_shape为(B+1,)</li>
                    <li>值仅支持非负整数</li>
                    <li>其值应非递减（大于等于前一个值）排列，第一个元素为0且最后一个元素等于K_T</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>当layout_k为TND时，必须传入</li>
                    <li>当layout_k不为TND时，不支持传入</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>max_seqlen_q</td>
            <td rowspan="2">
                <ul>
                    <li>data_type支持INT</li>
                    <li>暂不生效，仅支持-1</li>
                    <li>默认值为-1</li>
                </ul>
            </td>
            <td rowspan="2">
                <ul>
                    <li>暂不生效，仅支持传入-1</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>max_seqlen_k</td>
        </tr>
    </tbody>
</table>

#### Paged Attention参数组
当block_table不为空时，开启Paged Attention
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
                    <li>tensor_type仅支持INT32</li>
                    <li>tensor_shape为(B, max_num_blocks_per_seq)</li>
                    <li>值只能为正整数</li>
                </ul>
            </td>
            <td>可选参数</td>
            <td>无</td>
            <td>
                <ul>
                    <li>PagedAttention开启情况下，必须传入seqused_k</li>
                    <li>PagedAttention开启情况下，block_table必须不为空</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>


## 确定性计算

默认支持确定性计算。

## 调用示例

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  import numpy as np
  from cann_ops_transformer.ops import lightning_indexer, lightning_indexer_metadata

  B = 2
  S1 = 64
  S2 = 130
  N1 = 16
  N2 = 1
  D = 128
  topk = 32
  cmp_ratio = 4

  # 计算压缩后S2长度及余数
  S2_compressed = S2 // cmp_ratio
  res_k = S2 % cmp_ratio

  # 构造输入
  q = torch.randn(B, S1, N1, D, dtype=torch.float16).npu()
  k = torch.randn(B, S2_compressed, N2, D, dtype=torch.float16).npu()
  w = torch.randn(B, S1, N1, dtype=torch.float32).npu()
  cmp_residual_k = torch.tensor([res_k] * B, dtype=torch.int32).npu()

  # 生成metadata
  metadata = lightning_indexer_metadata(
      num_heads_q=N1,
      num_heads_k=N2,
      head_dim=D,
      topk=topk,
      cmp_residual_k=cmp_residual_k,
      batch_size=B,
      max_seqlen_q=S1,
      max_seqlen_k=S2_compressed,
      layout_q="BSND",
      layout_k="BSND",
      mask_mode=3,
      cmp_ratio=cmp_ratio
  )

  # 执行lightning_indexer
  sparse_indices, sparse_values = lightning_indexer(
      q, k, w, topk,
      cmp_residual_k=cmp_residual_k,
      metadata=metadata,
      max_seqlen_q=-1,
      layout_q="BSND",
      layout_k="BSND",
      mask_mode=3,
      cmp_ratio=cmp_ratio,
      return_value=0
  )
  print(f"sparse_indices shape: {sparse_indices.shape}")
  ```

- TorchAir（aclgraph）图模式调用：

  ```python
  import torch
  import torch_npu
  import numpy as np
  import torchair
  from cann_ops_transformer.ops import lightning_indexer, lightning_indexer_metadata

  B = 2
  S1 = 64
  S2 = 130
  N1 = 16
  N2 = 1
  D = 128
  topk = 32
  cmp_ratio = 4

  S2_compressed = S2 // cmp_ratio
  res_k = S2 % cmp_ratio

  q = torch.randn(B, S1, N1, D, dtype=torch.float16).npu()
  k = torch.randn(B, S2_compressed, N2, D, dtype=torch.float16).npu()
  w = torch.randn(B, S1, N1, dtype=torch.float32).npu()
  cmp_residual_k = torch.tensor([res_k] * B, dtype=torch.int32).npu()

  class LightningIndexerNetwork(torch.nn.Module):
      def __init__(self):
          super(LightningIndexerNetwork, self).__init__()

      def forward(self, q, k, w, cmp_residual_k):
          metadata = torch.ops.cann_ops_transformer.lightning_indexer_metadata(
              num_heads_q=N1,
              num_heads_k=N2,
              head_dim=D,
              topk=topk,
              cmp_residual_k=cmp_residual_k,
              batch_size=B,
              max_seqlen_q=S1,
              max_seqlen_k=S2_compressed,
              layout_q="BSND",
              layout_k="BSND",
              mask_mode=3,
              cmp_ratio=cmp_ratio
          )

          return torch.ops.cann_ops_transformer.lightning_indexer(
              q, k, w, topk,
              cmp_residual_k=cmp_residual_k,
              metadata=metadata,
              max_seqlen_q=-1,
              layout_q="BSND",
              layout_k="BSND",
              mask_mode=3,
              cmp_ratio=cmp_ratio,
              return_value=1
          )

  from torchair.configs.compiler_config import CompilerConfig
  config = CompilerConfig()
  config.mode = "reduce-overhead"
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  torch._dynamo.reset()
  npu_mode = torch.compile(LightningIndexerNetwork(), fullgraph=True, backend=npu_backend, dynamic=False)
  sparse_indices, sparse_values = npu_mode(q, k, w, cmp_residual_k)
  print(f"sparse_indices shape: {sparse_indices.shape}")
  ```
