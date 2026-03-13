# KvQuantSparseFlashAttention

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列加速卡产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- API功能：`kv_quant_sparse_flash_attention`在`sparse_flash_attention`的基础上支持了[Per-Token-Head-Tile-128量化]输入。随着大模型上下文长度的增加，Sparse Attention的重要性与日俱增，这一技术通过“只计算关键部分”大幅减少计算量，然而会引入大量的离散访存，造成数据搬运时间增加，进而影响整体性能。

- 计算公式：

    $$
    Attention=\text{softmax}(\frac{Q @ \text{Dequant}({\tilde{K}^{INT8}},{Scale_K})^T}{\sqrt{d_k}})@\text{Dequant}(\tilde{V}^{INT8},{Scale_V}),
    $$

    其中$\tilde{K},\tilde{V}$为基于某种选择算法（如`LightningIndexer`）得到的重要性较高的Key和Value，一般具有稀疏或分块稀疏的特征，$d_k$为$Q,\tilde{K}$每一个头的维度，$\text{Dequant}(\cdot,\cdot)$为反量化函数。
本次公布的`kv_quant_sparse_flash_attention`是面向Sparse Attention的全新算子，针对离散访存进行了指令缩减及搬运聚合的细致优化。

## 参数说明

**说明：**<br> 

> - query、key、value参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
> - Q\_S和S1表示query shape中的S，KV\_S和S2表示key shape中的S，Q\_N表示num\_query\_heads，KV\_N表示num\_key\_value\_heads，Q\_T表示query shape中的T，KV\_T表示key shape中的T。

- **query**（`Tensor`）：必选参数，表示attention结构的Q输入，不支持非连续，数据格式支持$ND$，数据类型支持`bfloat16`和`float16`，query由相同dtype的q_nope和q_rope按D维度拼接得到。`layout_query`为BSND时shape为[B,S1,Q\_N,D]，当`layout_query`为TND时shape为[Q\_T,Q\_N,D]，其中Atlas A3 推理系列产品数据Q\_N支持1/2/4/8/16/32/64/128，Atlas A5 推理系列产品数据Q\_N支持1/2/4/8/16/32/48/64。

- **key**（`Tensor`）：必选参数，表示attention结构的K输入，不支持非连续，数据格式支持$ND$，Atlas A3 推理系列产品数据类型支持`int8`，Atlas A5 推理系列产品数据类型支持`float8_e4m3`和`hifloat8`，`int8`/`float8_e4m3`/`hifloat8`的k_nope、query相同dtype的k_rope和`float32`的量化参数按D维度拼接得到，layout\_kv为PA\_BSND时shape为[block\_num, block\_size, KV\_N, D]，其中block\_num为PageAttention时block总数，block\_size为一个block的token数，block\_size取值为16的整数倍，最大支持到1024。`layout_kv`为BSND时shape为[B, S2, KV\_N, D]，`layout_kv`为TND时shape为[KV\_T, KV\_N, D]，其中KV\_N只支持1。

- **value**（`Tensor`）：必选参数，表示attention结构的V输入，不支持非连续，数据格式支持$ND$，Atlas A3 推理系列产品数据类型支持`int8`，Atlas A5 推理系列产品数据类型支持`float8_e4m3`和`hifloat8`。value的N仅支持1。

- **sparse\_indices**（`Tensor`）：必选参数，代表离散取kvCache的索引，不支持非连续，数据格式支持$ND$，数据类型支持`int32`，当`layout_query`为BSND时，shape需要传入[B, Q\_S, KV\_N, sparse\_size]，当`layout_query`为TND时，shape需要传入[Q\_T, KV\_N, sparse\_size]，其中sparse\_size为一次离散选取的block数，需要保证每行有效值均在前半部分，无效值均在后半部分，且需要满足sparse\_size大于0。

- **scale\_value**（`float`）：必选参数，公式中$d_k$开根号的倒数，代表缩放系数，作为query和key矩阵乘后Muls的scalar值，数据类型支持`float`。

- **key\_quant\_mode**（`int`）：必选参数，代表key的量化模式，数据类型支持`int64`，仅支持传入2，代表per_tile量化模式。

- **value\_quant\_mode**（`int`）：必选参数，代表value的量化模式，数据类型支持`int64`，仅支持传入2，代表per_tile量化模式。

- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。

- **key\_dequant\_scale**（`Tensor`）：可选参数，预留参数，仅支持默认值。

- **value\_dequant\_scale**（`Tensor`）：可选参数，预留参数，仅支持默认值。

- **block\_table**（`Tensor`）：可选参数，表示PageAttention中kvCache存储使用的block映射表。数据格式支持$ND$，数据类型支持`int32`，shape为2维，其中第一维长度为B，第二维长度不小于所有batch中最大的s2对应的block数量，即s2\_max / block\_size向上取整。

- **actual\_seq\_lengths\_query**（`Tensor`）：可选参数，表示不同Batch中`query`的有效token数，数据类型支持`int32`。如果不指定seqlen可传入None，表示和`query`的shape的S长度相同。该参数中每个Batch的有效token数不超过`query`中的维度S大小且不小于0。支持长度为B的一维tensor。<br>当`layout_query`为TND时，该入参必须传入，且以该入参元素的数量作为B值，该入参中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。

- **actual\_seq\_lengths\_kv**（`Tensor`）：可选参数，表示不同Batch中`key`和`value`的有效token数，数据类型支持`int32`。如果不指定None，表示和key的shape的S长度相同。该参数中每个Batch的有效token数不超过`key/value`中的维度S大小且不小于0。支持长度为B的一维tensor。<br>当`layout_kv`为TND或PA_BSND时，该入参必须传入，`layout_kv`为TND，该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。

- **sparse\_block\_size**（`int`）：可选参数，代表sparse阶段的block大小，在计算importance score时使用，数据类型支持`int64`，Atlas A3 推理系列产品支持范围为[1, 16]，且为2的幂次方, Atlas A5 推理系列产品支持1。

- **layout\_query**（`str`）：可选参数，用于标识输入`query`的数据排布格式，默认值"BSND"，支持传入BSND和TND。

- **layout\_kv**（`str`）：可选参数，用于标识输入`key`的数据排布格式，默认值"BSND"，支持传入BSND、TND和PA\_BSND，PA\_BSND在使能PageAttention时使用。

- **sparse\_mode**（`int`）：可选参数，表示sparse的模式。数据类型支持`int64`。
    - sparse\_mode为0时，代表全部计算。
    - sparse\_mode为3时，代表rightDownCausal模式的mask，对应以右下顶点往左上为划分线的下三角场景。

- **pre\_tokens**（`int`）：可选参数，用于稀疏计算，表示attention需要和前几个Token计算关联。数据类型支持`int64`，仅支持默认值2^63-1。

- **next\_tokens**（`int`）：可选参数，用于稀疏计算，表示attention需要和后几个Token计算关联。数据类型支持`int64`，仅支持默认值2^63-1。

- **attention\_mode**（`int`）：可选参数，表示attention的模式。数据类型支持`int64`，仅支持传入2，表示MLA-absorb模式，即QK的D包含rope和nope两部分，且KV是同一份，默认值为0。

- **quant\_scale\_repo\_mode**（`int`）：可选参数，表示量化参数的存放模式。数据类型支持`int64`，仅支持传入1，表示combine模式，即量化参数和数据混合存放，默认值1。

- **tile\_size**（`int`）：可选参数，表示per_tile时每个参数对应的数据块大小，仅在per_tile时有效。数据类型支持`int64`，仅支持默认值128。

- **rope\_head\_dim**（`int`）：可选参数，表示MLA架构下的rope\_head\_dim大小，仅在attention\_mode为2时有效。数据类型支持`int64`，仅支持默认值64。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持图模式。
- 参数query中的D值为576，即nope\+rope=512\+64。
- 参数key、value中的D值为656，即nope\+rope\*2\+dequant\_scale\*4=512\+64\*2\+4\*4。
- 支持sparse\_block\_size整除block\_size。
- 非PageAttention场景layout\_query和layout\_kv需要保持一致。
