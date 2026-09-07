# TurboQuantSparseFlashAttention

## 产品支持情况
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明
- API功能：TurboQuantSparseFlashAttention算子面向 MLA（Multi-head Latent Attention）推理场景，对 4bit 量化存储的 KV latent 执行稀疏 Attention 计算。算子内部完成 KV 的反量化与 Attention 计算的融合，避免将反量化结果落盘到 GM，从而在 4bit KV 压缩比下保持访存效率。

- 计算公式：

    $$
    O = \text{softmax}(Q@\tilde{K}^T \cdot \text{scale\_value})@\tilde{V}
    $$

    其中 $\tilde{K}=\tilde{V}$ 为由 `sparse_indices` 选中、并经 4bit 码本反量化还原得到的 KV latent。MLA 场景下 K 与 V 共用同一份 latent。

- 量化方案：latent 每个 token 先做 L2 归一化，各元素按 16 个码本中心量化为 4bit，两个 nibble 打包为 1 字节；每 token 附加 1 个 `float16` 的归一化系数 $s_t$；rope 分量以与 query 相同的 `float16` 或 `bfloat16` dtype 存放 $rope / s_t$，即**预先除以 $s_t$**（原因见下方约束说明）。

## 参数说明

| 参数名           |输入/输出/属性|    描述    | 数据类型    |数据格式|
|-------------|------------|------|-----|-----|
|query|输入|公式中的 $Q$，layout_query 仅支持 TND，shape 为 [T1, N1, D]，最后一维 D 含 rope 分量|FLOAT16、BFLOAT16|ND|
|key|输入|公式中 $\tilde{K}$ 的 4bit 量化存储，以 INT8 承载打包后的 nibble、rope 与归一化系数|INT8|ND|
|value|输入|公式中的 $\tilde{V}$。MLA 场景下与 key 为同一份 latent，本算子不单独读取该输入，保留是为与 c8 稀疏注意力路径的接口保持一致；调用方传入与 key 相同的张量即可|INT8|ND|
|sparse_indices|输入|稀疏选取的 KV 索引，标识每个 query 实际参与计算的 KV 位置|INT32|ND|
|key_dequant_scale|可选输入|key 的反量化系数。quant_scale_repo_mode 为 1（COMBINE，当前唯一支持值）时该系数合并存放于 KV slot 内，本输入不被消费，可不传|FLOAT32|ND|
|value_dequant_scale|可选输入|value 的反量化系数。与 key_dequant_scale 同，COMBINE 模式下不被消费，可不传|FLOAT32|ND|
|block_table|输入|PageAttention 中 KV cache 存储使用的 block 映射表|INT32|ND|
|actual_seq_lengths_query|输入|各 Batch 中 query 的有效 token 数。TND 场景下为累加和形式，即前序 Batch 与当前 Batch 有效 token 数的累加值|INT32|ND|
|actual_seq_lengths_kv|输入|各 Batch 中 KV 的有效 token 数|INT32|ND|
|scale_value|属性|缩放系数，作为 $Q$ 与 $\tilde{K}$ 矩阵乘后的 Muls 标量值，默认值 1.0|FLOAT|-|
|key_quant_mode|属性|key 的量化模式，当前仅支持 3（TQ4 码本量化），默认值 3|INT32|-|
|value_quant_mode|属性|value 的量化模式，当前仅支持 3（TQ4 码本量化），默认值 3|INT32|-|
|sparse_block_size|可选属性|稀疏选取的 block 粒度，默认值 1|INT32|-|
|layout_query|可选属性|query 的数据排布格式，当前仅支持 TND，默认值 TND|STRING|-|
|layout_kv|可选属性|KV 的数据排布格式，支持 PA_BSND 和 PA_BNBD（PageAttention），默认值 PA_BSND|STRING|-|
|sparse_mode|可选属性|mask 模式，默认值 3，表示以右顶点为划分的下三角场景|INT32|-|
|pre_tokens|可选属性|query 对过去 token 的计算数量，默认值 INT64_MAX|INT32|-|
|next_tokens|可选属性|query 对未来 token 的计算数量，默认值 INT64_MAX|INT32|-|
|attention_mode|可选属性|Attention 计算模式，当前仅支持 2（MLA-absorb），默认值 2|INT32|-|
|quant_scale_repo_mode|可选属性|量化系数的存放模式。当前仅支持 1（COMBINE），即反量化系数与 Nope、Rope 合并存放于同一个 KV slot 内，默认值 1|INT32|-|
|tile_size|可选属性|量化粒度，默认值 128|INT32|-|
|rope_head_dim|可选属性|rope 分量的 head dim，默认值 64|INT32|-|
|return_softmax_lse|可选属性|是否返回 softmax_lse，默认值 false|BOOL|-|
|attention_out|输出|Attention 计算结果，layout 为 TND，shape 为 [T1, N1, D - rope_head_dim]，即最后一维为 query 最后一维减去 rope 分量，数据类型与 query 相同|FLOAT16、BFLOAT16|ND|
|softmax_max|输出|softmax 过程中按行取得的最大值|FLOAT32|ND|
|softmax_sum|输出|softmax 过程中按行取得的求和值|FLOAT32|ND|

## 约束说明
-   该接口支持推理场景下使用。
-   数据类型支持范围：query 与 attention_out 支持 `FLOAT16` 和 `BFLOAT16`，且两者必须同型；key 与 value 仅支持 `INT8`；sparse_indices 仅支持 `INT32`。
-   value 的数据类型与 shape 均须与 key 一致。MLA 场景下 K 与 V 为同一份 latent，kernel 的 MM2 复用已反量化的合并缓冲，不单独读取 value。
-   数据排布支持范围：query 与 attention_out 仅支持 `TND`；key 与 value 支持 `PA_BSND [Bn, Bs, 1, 386]` 和 `PA_BNBD [Bn, 1, Bs, 386]`，默认仍为 `PA_BSND`。Bn 为物理 block 数，Bs 为 block size，KV head 数固定为 1。
-   两种 KV 布局均要求 Bs 为 [16, 1024] 范围内的 16 的倍数，且可被 `sparse_block_size` 整除。layout 属性必须与输入 shape 的轴顺序匹配；不接受 `PA_BNSD` 或 `BNBD` 字符串别名，也不支持多 KV head。
-   连续内存中 N=1 的两种布局具有相同的字节顺序，Host 按各自的轴解析后复用同一个 PA Kernel，无需转置或复制 KV cache。386 字节 slot 的内容、block_table 和 sparse_indices 的语义不变；已有 vLLM cache 默认布局不受影响。
-   两种 PA 布局均原生支持 KV cache 第 0 轴（物理 block 轴）非连续，页内仍须连续。`stride(0)` 以 INT8 字节为单位传入 Kernel，不要求是 386 的倍数，支持间隔取页、页间 padding 和合法的非零 `storage_offset`，无需先将整块 cache 转连续。
-   多页 cache 要求 `stride(0) >= Bs * 386`，不支持重叠页、负 stride 或页内非连续；长度为 1 的轴不约束其 stride。寻址跨度 `(Bn - 1) * stride(0) + Bs * 386` 不得超过 `INT64_MAX`，调用方须提供足够的底层存储。空 query 也执行这些布局校验。
-   页地址使用 64 位字节偏移；合并搬运的源间距超出 DMA 参数范围时退回单条搬运。此改动增加了内部 Tiling 字段，发布时必须配套更新 Host 与 Kernel，不能混用旧二进制；增量构建需重新生成 ACLNN 入口。
-   query 的最后一维包含 rope 分量，attention_out 的最后一维为 query 最后一维减去 `rope_head_dim`，即 $D_{out} = D_{query} - rope\_head\_dim$。
-   quant_scale_repo_mode 仅支持 1（COMBINE）：反量化系数与 Nope、Rope 合并存放于同一个 KV slot 的尾部 2 字节，由 kernel 就地读取；key_dequant_scale 与 value_dequant_scale 两个可选输入在该模式下不被消费。
-   MLA 场景下 key 与 value 指向同一份 latent，"对 K 按行缩放"与"对 score/P 按列缩放"在数学上等价。kernel 采用后者，即把 $s_t$ 作用在**整列 score** 上（softmax 前与 softmax 后各一次）。
-   由于该列同时包含 nope 与 rope 两部分的贡献，**调用方必须按 $rope / s_t$ 存放 rope 分量**，kernel 乘回 $s_t$ 后才还原出真实的 rope 贡献。若按原始 rope 存放，误差为 ($s_t$ - 1) × q_rope · rope：在 rope 幅值小或 $s_t$ 接近 1 时该误差不可见，故 examples 中以 rope 主导且 $s_t$ 远离 1 的用例专门覆盖。
-   `actual_seq_lengths_query` 与 `actual_seq_lengths_kv` 的语义**不对称**，两者不可互换：前者为 TND query 的**累加和**（kernel 按相邻元素差分还原每 batch 长度，故末元素必须等于 query 的 T）；后者在两种 PA 布局下均为**每 batch 的实际 KV 长度**而非累加和，kernel 直接按 batch 下标取值。
-   下列数值约束由调用方保证：两个长度数组均非负；`actual_seq_lengths_query` 单调不减；每个 batch 的 KV 长度不超过其 `block_table` 行所能覆盖的容量（`block_table.dim1 × block_size`）；`block_table` 中的 block id 落在 KV cache 的block 数范围内；`sparse_indices` 为 **batch 内局部**的 KV 位置，超出有效长度的位置以 `-1` 填充。
    这些取值在推理场景下均为 device 张量，tiling 阶段无法读取，因此 host 侧只校验其数据类型与元素个数，数值本身不做校验——传入越界值会导致错误的地址计算，请调用方自行保证。
-   某个 batch 的 KV 长度允许为 0。两种布局下该 batch 的 `attention_out=0`；请求 softmax 输出时，`softmax_max=-2e38`、`softmax_sum=0`。未请求时两个 softmax 输出均为 shape [0]；请求时 shape 均为 [1, T1, N1]。
-   某个 batch 的 query 长度允许为 0，对应 `actual_seq_lengths_query` 的累计值可以重复，该 batch 不产生输出行。空 query batch 和零 KV batch 均跳过有效 Attention 计算，不影响同一调用内其他 batch 的输出。
-   支持整体 query 的 T1=0：返回 shape [0, N1, 512] 的空输出；请求 softmax 输出时其 shape 为 [1, 0, N1]，否则为 [0]。此时不分配算子 workspace，Kernel 不读取输入数据。Batch 数、query head 数、KV 物理 block 数、block size 和 block_table 列数仍须满足非空输入的约束；长度数组的所有 query 累计值须为 0。实际 KV 长度为 0 不代表支持空的物理 KV cache。
-   KV 长度非零但 sparse_indices 全为 -1 的行，与零 KV 长度不同：输出仍为 0，softmax_max 为 -2e38，但既有 Kernel 使用有限 mask 值，softmax_sum 不保证为 0。两种布局保持该行为，调用方不可仅用 softmax_sum 是否为 0 判断行有效性。
-   KV latent 的 headDim 为 512，rope_head_dim 默认 64。每个 KV slot 由 256 字节打包 nibble、128 字节与 query 同 dtype 的 rope 分量和 2 字节 `float16` 归一化系数顺序拼接组成，总宽度固定为 386 字节。
-   sparse_indices 中的无效位置以 -1 填充，需保证每行有效值均在前半部分。
-   参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、N（Head Num）表示多头数、D（Head Dim）表示 hidden 层最小的单元尺寸、T 表示所有 Batch 输入样本序列长度的累加和。
-   S1 表示 query shape 中的 S，N1 表示 num_query_heads，T1 表示 query shape 中的 T。

## 精度说明
-   本算子为 4bit 量化算子，其输出与全精度 Attention 之间存在量化本身引入的固有偏差，不属于算子实现缺陷。因此精度对标的标杆取**与算子采用相同量化方案的高精度 CPU 实现**（量化与反量化过程按 float64 计算），以剥离量化误差、只考察算子实现误差。
-   判定沿用 [experimental 精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) 的双条件：
    -   匹配率：满足 $|actual - golden| \le atol + rtol \times |golden|$ 的元素占比 $\ge 0.99$；
    -   最大绝对误差不超过对应数据类型的硬上限。
