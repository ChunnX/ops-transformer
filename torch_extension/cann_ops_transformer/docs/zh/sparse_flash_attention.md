# sparse_flash_attention

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->

## 功能说明

- **接口功能**：`sparse_flash_attention`（SFA）是针对大序列长度推理场景的高效稀疏注意力计算接口，通过“只计算关键部分”大幅减少计算量。该接口是`cann_ops_transformer`扩展 torch 接口，底层调度`SparseFlashAttention`算子，在支持稀疏选择、MLA-absorb、Paged Attention 等能力的同时，针对离散访存进行了指令缩减及搬运聚合优化。

- **计算公式**：

  $$
  O = \text{softmax}\left(\frac{Q@\tilde{K}^{T}}{\sqrt{d_k}}\right)@\tilde{V}
  $$

  其中$\tilde{K},\tilde{V}$为基于某种选择算法（如`lightning_indexer`）得到的重要性较高的 Key 和 Value，一般具有稀疏或分块稀疏的特征，$d_k$为$Q,\tilde{K}$每一个头的维度。`sparse_indices`中的索引值`b`表示选择 KV 序列中从`b * sparse_block_size`开始的一段连续 token。将`key/value`按`sparse_indices`收集后，即得到公式中的$\tilde{K},\tilde{V}$。

  当`attention_mode = 2`时，表示 MLA-absorb 模式。此时原公式中的$Q$和$\tilde{K}$由`nope`部分和`rope`部分拼接而成：

  $$
  Q = [Q_{nope}, Q_{rope}], \quad \tilde{K} = [\tilde{K}_{nope}, \tilde{K}_{rope}]
  $$

  其中，`query/key`提供$Q_{nope}, \tilde{K}_{nope}$，`query_rope/key_rope`提供$Q_{rope}, \tilde{K}_{rope}$，且 key 与 value 共享同一份底层张量数据。

  不同`sparse_mode`下，公式中$\tilde{K},\tilde{V}$对应的“有效 KV 集合”有所不同：

  - 当`sparse_mode = 0`时，仅按照`sparse_indices`选择出的 KV 位置构造$\tilde{K},\tilde{V}$，即只做稀疏选择，不额外施加因果 mask。
  - 当`sparse_mode = 3`时，在`sparse_indices`选择出的 KV 位置基础上，继续叠加`rightDownCausal`约束，限定当前 query 允许访问的因果范围。若当前 batch 的 query 和 kv 有效长度分别为$L_q$和$L_{kv}$，则对第$i$个 query 位置，允许访问的 kv 位置$j$满足：

    $$
    j \le (L_{kv} - L_q) + i
    $$

- **典型使用流程**：

  1. 使用选择算法（如`lightning_indexer`）根据 Query 和 Key 计算出重要 token 的索引；
  2. 将索引填充为`sparse_indices`参数所要求的格式；
  3. 将`sparse_indices`传入本接口，完成稀疏注意力计算。

  > [!NOTE]
  >
  > `lightning_indexer` 是一种稀疏 KV 选择算法，用于识别长序列推理中与当前查询最相关的 Key/Value 区块。它的输出即为本接口所需的`sparse_indices`张量。若不使用`lightning_indexer`，用户需自行实现索引选择逻辑，保证`sparse_indices`满足[参数说明](#参数说明)中的格式要求。

## 函数原型

```python
cann_ops_transformer.sparse_flash_attention(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    *,
    block_table=None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    query_rope=None,
    key_rope=None,
    sinks=None,
    sparse_block_size=1,
    layout_query="BSND",
    layout_kv="BSND",
    sparse_mode=3,
    pre_tokens=9223372036854775807,
    next_tokens=9223372036854775807,
    attention_mode=2,
    return_softmax_lse=False,
) -> (Tensor, Tensor, Tensor)
```

> [!NOTE]
>
> - `query`、`key`、`value`、`sparse_indices`、`scale_value`为必选位置参数，必须按照顺序传入；`*`之后为可选关键字参数，不赋值时使用默认值。
> - `value`为必选位置参数，类型为`Tensor`，支持传入`None`。传入`None`时，底层将 D 和 V 合并处理，减少冗余访存与计算，提升性能。`value=None`仅在 CANN >= 9.2.0 时支持，更低版本会抛出`NOT_SUPPORT`错误。

## 参数说明

### 常见字段释义

|    命名     |                           含义                           |
| :---------: | :------------------------------------------------------: |
|      B      |                输入样本batch大小                          |
|     Q_S     |                输入query的序列长度                        |
|    KV_S     |                输入key/value的序列长度                     |
|     Q_N     |                输入query的头数                            |
|    KV_N     |                输入key/value的头数                        |
|      D      |                注意力头的nope维度，典型取512              |
|     Dr      |                注意力头的rope维度，典型取64               |
|     Q_T     |     输入query所有batch序列长度的累加和（Total Tokens）     |
|    KV_T     |  输入key/value所有batch序列长度的累加和（Total Tokens）    |
| sparse_size |              一次离散选取的block数                         |
|  block_num  |            PageAttention场景下的block总数                 |
| block_size  |      PageAttention场景下每个block的token数                |

### sparse_flash_attention

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| query | Tensor | 必选 | attention结构的Q输入，对应公式中的$Q$，不支持空tensor和非连续Tensor。Q_N支持1~128。 | float16、bfloat16 | ND | `BSND`：(B, Q_S, Q_N, D)<br>`TND`：(Q_T, Q_N, D) |
| key | Tensor | 必选 | attention结构的K输入，对应公式中的$\tilde{K}$，不支持空tensor。`layout_kv`为`PA_BSND`时支持0轴非连续，block_num为Paged Attention时block总数。block_size为一个block的token数，取值为16的倍数，最大支持1024。KV_N仅支持1。 | float16、bfloat16 | ND | `BSND`：(B, KV_S, KV_N, D)<br>`TND`：(KV_T, KV_N, D)<br>`PA_BSND`：(block_num, block_size, KV_N, D) |
| value | Tensor | 必选 | attention结构的V输入，对应公式中的$\tilde{V}$，不支持空tensor。`layout_kv`为`PA_BSND`时支持0轴非连续，block_size为16的倍数，最大支持1024。仅在CANN >= 9.2.0时支持`value=None`，更低版本会抛出`NOT_SUPPORT`错误。 | float16、bfloat16 | ND | `BSND`：(B, KV_S, KV_N, D)<br>`TND`：(KV_T, KV_N, D)<br>`PA_BSND`：(block_num, block_size, KV_N, D) |
| sparse_indices | Tensor | 必选 | 代表离散取kvCache的索引，通常由稀疏选择算法（如`lightning_indexer`）生成，不支持空tensor和非连续Tensor。sparse_size为一次离散选取的block数，每行有效值均在前半部分、无效值均在后半部分，且sparse_size需大于0。 | int32 | ND | `BSND`：(B, Q_S, KV_N, sparse_size)<br>`TND`：(Q_T, KV_N, sparse_size) |
| scale_value | double | 必选 | 代表缩放系数，作为query和key矩阵乘后Muls的scalar值。 | double | - | - |
| block_table | Tensor | 可选 | 表示PageAttention中kvCache存储使用的block映射表。shape第一维长度为B，第二维长度不小于所有batch中最大的KV_S对应的block数量，即`KV_S_max/block_size`向上取整。默认值为`None`。 | int32 | ND | (B, ceil(KV_S_max/block_size)) |
| actual_seq_lengths_query | Tensor | 可选 | 表示不同Batch中query的有效token数。不传时表示与query的Q_S长度相同，每个Batch的有效token数不超过Q_S且不小于0。当`layout_query`为`TND`时该入参必须传入，并以该入参元素的数量作为B值；该入参每个元素的值表示当前batch与之前所有batch的token数总和（前缀和），后一个元素的值必须大于等于前一个元素的值。默认值为`None`。 | int32 | ND | (B,) |
| actual_seq_lengths_kv | Tensor | 可选 | 表示不同Batch中key和value的有效token数。不传时表示与key的KV_S长度相同，每个Batch的有效token数不超过KV_S且不小于0。当`layout_kv`为`TND`或`PA_BSND`时该入参必须传入；`layout_kv`为`TND`时该入参每个元素的值表示前缀和，后一个元素的值必须大于等于前一个元素的值。默认值为`None`。 | int32 | ND | (B,) |
| query_rope | Tensor | 可选 | 表示MLA结构中的query的rope信息，不支持非连续Tensor。shape与query的B/S/N对齐，最后一维为rope维度。默认值为`None`。 | float16、bfloat16 | ND | `BSND`：(B, Q_S, Q_N, Dr)<br>`TND`：(Q_T, Q_N, Dr) |
| key_rope | Tensor | 可选 | 表示MLA结构中的key的rope信息。shape与key的layout对齐，最后一维为rope维度。默认值为`None`。 | float16、bfloat16 | ND | `BSND`：(B, KV_S, KV_N, Dr)<br>`TND`：(KV_T, KV_N, Dr)<br>`PA_BSND`：(block_num, block_size, KV_N, Dr) |
| sinks | Tensor | 可选 | 表示attention结构中的sinks信息，每个query head使用对应sink值参与Online Softmax的全局max和分母sum。传入有效Tensor时调用`aclnnSparseFlashAttentionV2`接口，不传入时调用原`aclnnSparseFlashAttention`接口。不支持空Tensor和非连续Tensor。默认值为`None`。 | float32 | ND | (Q_N,) |
| sparse_block_size | int64 | 可选 | 代表sparse阶段的block大小，在计算importance score时使用。仅支持默认值1，表示Token-wise稀疏化场景，将每个token视为独立单元。 | int64 | - | - |
| layout_query | string | 可选 | 用于标识输入query的数据排布格式，支持传入`BSND`和`TND`。默认值为`BSND`。 | string | - | - |
| layout_kv | string | 可选 | 用于标识输入key、value的数据排布格式，支持传入`BSND`、`TND`和`PA_BSND`，其中`PA_BSND`在开启PageAttention时使用。默认值为`BSND`。 | string | - | - |
| sparse_mode | int64 | 可选 | 表示sparse的模式。为0时代表全部计算；为3时代表rightDownCausal模式的mask，对应以右下顶点往左上为划分线的下三角场景。默认值为3。 | int64 | - | - |
| pre_tokens | int64 | 可选 | 用于稀疏计算，表示attention需要和前几个Token计算关联，仅支持默认值$2^{63}-1$。 | int64 | - | - |
| next_tokens | int64 | 可选 | 用于稀疏计算，表示attention需要和后几个Token计算关联，仅支持默认值$2^{63}-1$。 | int64 | - | - |
| attention_mode | int64 | 可选 | 表示attention的模式，仅支持传入2，表示MLA-absorb模式，即计算过程中会将query和key的nope部分分别和query_rope、key_rope的rope部分沿头维度（D）拼接，合并形成最终的query和key用于后续计算，且key和value共享同一份底层张量数据。默认值为2。 | int64 | - | - |
| return_softmax_lse | bool | 可选 | 表示是否返回softmax_max与softmax_sum。为True时返回，为False时返回空的占位Tensor。默认值为False。 | bool | - | - |

## 返回值说明

### sparse_flash_attention

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| attention_out | Tensor | 必选 | 代表公式中的输出Attention，shape与入参query保持一致，D为query的nope维度，不包含rope部分。 | float16、bfloat16 | ND | `BSND`：(B, Q_S, Q_N, D)<br>`TND`：(Q_T, Q_N, D) |
| softmax_max | Tensor | 可选 | Attention算法对query乘key的结果取max得到softmax_max。`return_softmax_lse=True`时返回有效结果，否则返回shape为[0]的空Tensor。 | float32 | ND | `BSND`：(B, KV_N, Q_S, Q_N/KV_N)<br>`TND`：(KV_N, Q_T, Q_N/KV_N) |
| softmax_sum | Tensor | 可选 | Attention算法query乘key的结果减去softmax_max后取exp并求sum得到softmax_sum。`return_softmax_lse=True`时返回有效结果，否则返回shape为[0]的空Tensor。 | float32 | ND | `BSND`：(B, KV_N, Q_S, Q_N/KV_N)<br>`TND`：(KV_N, Q_T, Q_N/KV_N) |

## 约束说明

- 该接口支持单算子模式和图模式。
- 参数 query、key、value 中的 D 值相等为 512，参数 query_rope、key_rope 的 Dr 值相等为 64。
- 参数 query、key、value 的数据类型必须保持一致。
- `value=None`仅在 CANN >= 9.2.0 时支持，更低版本会抛出`NOT_SUPPORT`错误。
- `layout_kv`为 PA_BSND 时，`layout_query`和`layout_kv`无需一致；`layout_kv`为 BSND 或 TND 时，`layout_query`和`layout_kv`需保持一致。
- Ascend 950PR/Ascend 950DT：
  - Q\_N 支持 1~128。
  - sparse\_block\_size 仅支持 1。
  - 仅在`layout_kv`为 PA_BSND 时，key、value 和 key_rope 支持 0 轴非连续。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import cann_ops_transformer
    import numpy as np
    import random

    torch_npu.npu.set_device(0)

    query_type = torch.float16
    scale_value = 1.0 / ((512 + 64) ** 0.5)
    sparse_block_size = 1
    sparse_block_count = 2048
    b = 4
    s1 = 1
    s2 = 8192
    n1 = 128
    n2 = 1
    dn = 512
    dr = 64
    s2_act = 4096
    attention_mode = 2
    return_softmax_lse = False

    query = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dn))).to(query_type).npu()
    key = torch.tensor(np.random.uniform(-5, 10, (b, s2, n2, dn))).to(query_type).npu()
    value = key.clone()
    idxs = random.sample(range(s2_act - s1 + 1), sparse_block_count)
    sparse_indices = torch.tensor([idxs for _ in range(b * s1 * n2)]).reshape(b, s1, n2, sparse_block_count).to(torch.int32).npu()
    query_rope = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dr))).to(query_type).npu()
    key_rope = torch.tensor(np.random.uniform(-10, 10, (b, s2, n2, dr))).to(query_type).npu()
    act_seq_q = torch.tensor([s1] * b).to(torch.int32).npu()
    act_seq_kv = torch.tensor([s2_act] * b).to(torch.int32).npu()

    attention_out, softmax_max, softmax_sum = cann_ops_transformer.sparse_flash_attention(
        query, key, value, sparse_indices, scale_value,
        actual_seq_lengths_query=act_seq_q, actual_seq_lengths_kv=act_seq_kv,
        query_rope=query_rope, key_rope=key_rope,
        sparse_block_size=sparse_block_size,
        layout_query="BSND", layout_kv="BSND", sparse_mode=3,
        pre_tokens=9223372036854775807, next_tokens=9223372036854775807,
        attention_mode=attention_mode, return_softmax_lse=return_softmax_lse)
    ```

- 图模式调用

    ```python
    import torch
    import cann_ops_transformer
    import torchair
    import torch.nn as nn
    import numpy as np
    import random

    query_type = torch.float16
    scale_value = 1.0 / ((512 + 64) ** 0.5)
    sparse_block_size = 1
    sparse_block_count = 2048
    b = 1
    s1 = 1
    s2 = 8192
    n1 = 128
    n2 = 1
    dn = 512
    dr = 64
    s2_act = 4096

    query = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dn))).to(query_type).npu()
    key = torch.tensor(np.random.uniform(-5, 10, (b, s2, n2, dn))).to(query_type).npu()
    value = key.clone()
    idxs = random.sample(range(s2_act - s1 + 1), sparse_block_count)
    sparse_indices = torch.tensor([idxs for _ in range(b * s1 * n2)]).reshape(b, s1, n2, sparse_block_count).to(torch.int32).npu()
    query_rope = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dr))).to(query_type).npu()
    key_rope = torch.tensor(np.random.uniform(-10, 10, (b, s2, n2, dr))).to(query_type).npu()
    act_seq_q = torch.tensor([s1] * b).to(torch.int32).npu()
    act_seq_kv = torch.tensor([s2_act] * b).to(torch.int32).npu()

    class Network(nn.Module):
        def forward(self, query, key, value, sparse_indices, scale_value,
                    actual_seq_lengths_query, actual_seq_lengths_kv,
                    query_rope, key_rope, sparse_block_size, attention_mode):
            attention_out, softmax_max, softmax_sum = cann_ops_transformer.sparse_flash_attention(
                query, key, value, sparse_indices, scale_value,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                query_rope=query_rope, key_rope=key_rope,
                sparse_block_size=sparse_block_size,
                layout_query="BSND", layout_kv="BSND", sparse_mode=3,
                pre_tokens=9223372036854775807, next_tokens=9223372036854775807,
                attention_mode=attention_mode, return_softmax_lse=False)
            return attention_out, softmax_max, softmax_sum

    from torchair.configs.compiler_config import CompilerConfig
    torch._dynamo.reset()
    npu_backend = torchair.get_npu_backend(compiler_config=CompilerConfig())
    mod = torch.compile(Network().npu(), backend=npu_backend, fullgraph=True, dynamic=True)
    attention_out, softmax_max, softmax_sum = mod(query, key, value, sparse_indices,
                        scale_value, act_seq_q, act_seq_kv, query_rope, key_rope,
                        sparse_block_size, 2)
    ```
