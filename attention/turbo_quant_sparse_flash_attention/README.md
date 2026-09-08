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

- API功能：TurboQuantSparseFlashAttention算子面向 DeepSeek Sparse Attention（DSA）推理场景，对 TQ4 量化存储的分页 KV cache 执行融合反量化与稀疏 Attention 计算。

- 计算公式：

    $$
    O = \text{softmax}(Q@\tilde{K}^T \cdot \text{scale\_value})@\tilde{V}
    $$

    其中 $Q=[Q_{latent},Q_{rope}]$，$\tilde{K}=[\tilde{K}_{latent},K_{rope}]$，$\tilde{V}=\tilde{K}_{latent}$。稀疏索引选择参与计算的 KV；K 与 V 共用 latent，rope 只参与 QK 计算。

- 量化方案：latent 每个 token 先做 L2 归一化，各元素按 16 个码本中心量化为 4bit，两个 nibble 打包为 1 字节；每 token 附加 1 个 `float16` 的归一化系数 $s_t$；rope 分量以与 query 相同的 `float16` 或 `bfloat16` dtype 存放 $rope / s_t$。

## 参数说明

| 参数名           |输入/输出/属性|    描述    | 数据类型    |数据格式|
|-------------|------------|------|-----|-----|
|query|输入|公式中的 $Q$，layout_query 仅支持 TND，shape 为 [T1, N1, D]，最后一维 D 含 rope 分量|FLOAT16、BFLOAT16|ND|
|key|输入|公式中 $\tilde{K}$ 的 4bit 量化存储，以 INT8 承载打包后的 nibble、rope 与归一化系数|INT8|ND|
|value|输入|请传入与 key 相同的张量|INT8|ND|
|sparse_indices|输入|稀疏选取的 KV 索引，标识每个 query 实际参与计算的 KV 位置|INT32|ND|
|key_dequant_scale|可选输入|key 的反量化系数。当前 COMBINE 模式下系数存放于 KV slot 内，本输入不参与计算，可不传|FLOAT32|ND|
|value_dequant_scale|可选输入|value 的反量化系数。当前 COMBINE 模式下不参与计算，可不传|FLOAT32|ND|
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
|pre_tokens|可选属性|query 对过去 token 的计算数量，默认值 INT64_MAX|INT64|-|
|next_tokens|可选属性|query 对未来 token 的计算数量，默认值 INT64_MAX|INT64|-|
|attention_mode|可选属性|底层 Attention 计算模式，当前 DSA 路径仅支持 2（MLA-absorb），默认值 2|INT32|-|
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
-   数据排布支持范围：query 与 attention_out 仅支持 `TND`；key 与 value 支持 `PA_BSND [Bn, Bs, 1, 386]` 和 `PA_BNBD [Bn, 1, Bs, 386]`，默认值为 `PA_BSND`。Bn 为物理 block 数，Bs 为 block size，KV head 数固定为 1。
-   两种 KV 布局均要求 Bs 为 [16, 1024] 范围内的 16 的倍数，且可被 `sparse_block_size` 整除。layout 属性必须与输入 shape 的轴顺序匹配；不接受 `PA_BNSD` 或 `BNBD` 字符串别名，也不支持多 KV head。
-   两种 PA 布局均支持 KV cache 第 0 轴（物理 block 轴）非连续，页内仍须连续。`stride(0)` 以字节为单位，不要求是 386 的倍数，支持间隔取页、页间 padding 和合法的非零 `storage_offset`。
-   多页 cache 要求 `stride(0) >= Bs * 386`，不支持重叠页、负 stride 或页内非连续；长度为 1 的轴不约束其 stride。寻址跨度 `(Bn - 1) * stride(0) + Bs * 386` 不得超过 `INT64_MAX`，调用方须提供足够的底层存储。空 query 也执行这些布局校验。
-   query 的最后一维包含 rope 分量，attention_out 的最后一维为 query 最后一维减去 `rope_head_dim`，即 $D_{out} = D_{query} - rope\_head\_dim$。
-   quant_scale_repo_mode 仅支持 1（COMBINE）：反量化系数存放于 KV slot 尾部 2 字节；key_dequant_scale 与 value_dequant_scale 两个可选输入不参与计算，可不传。
-   调用方必须按 $rope / s_t$ 存放 rope 分量，数据类型与 query 相同。
-   `actual_seq_lengths_query` 为各 batch query 长度的累加和，末元素必须等于 query 的 T；`actual_seq_lengths_kv` 为每个 batch 的实际 KV 长度，不是累加和。
-   下列数值约束由调用方保证：两个长度数组均非负；`actual_seq_lengths_query` 单调不减；每个 batch 的 KV 长度不超过其 `block_table` 行所能覆盖的容量（`block_table.dim1 × block_size`）；`block_table` 中的 block id 落在 KV cache 的block 数范围内；`sparse_indices` 为 **batch 内局部**的 KV 位置，超出有效长度的位置以 `-1` 填充。
    接口不校验这些设备张量中的数值，传入越界值可能导致非法内存访问。
-   某个 batch 的 KV 长度允许为 0。两种布局下该 batch 的 `attention_out=0`；请求 softmax 输出时，`softmax_max=-2e38`、`softmax_sum=0`。未请求时两个 softmax 输出均为 shape [0]；请求时 shape 均为 [1, T1, N1]。
-   某个 batch 的 query 长度允许为 0，对应 `actual_seq_lengths_query` 的累计值可以重复，该 batch 不产生输出行。空 query batch 和零 KV batch 不影响其他 batch 的输出。
-   支持整体 query 的 T1=0：返回 shape [0, N1, 512] 的空输出；请求 softmax 输出时其 shape 为 [1, 0, N1]，否则为 [0]。Batch 数、query head 数、KV 物理 block 数、block size 和 block_table 列数仍须满足非空输入的约束；长度数组的所有 query 累计值须为 0。实际 KV 长度为 0 不代表支持空的物理 KV cache。
-   KV 长度非零但 sparse_indices 全为 -1 的行，输出为 0，softmax_max 为 -2e38，softmax_sum 不保证为 0。调用方不可仅用 softmax_sum 是否为 0 判断行有效性。
-   KV latent 的 headDim 为 512，rope_head_dim 默认 64。每个 KV slot 由 256 字节打包 nibble、128 字节与 query 同 dtype 的 rope 分量和 2 字节 `float16` 归一化系数顺序拼接组成，总宽度固定为 386 字节。
-   sparse_indices 中的无效位置以 -1 填充，需保证每行有效值均在前半部分。
-   参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、N（Head Num）表示多头数、D（Head Dim）表示 hidden 层最小的单元尺寸、T 表示所有 Batch 输入样本序列长度的累加和。
-   S1 表示 query shape 中的 S，N1 表示 num_query_heads，T1 表示 query shape 中的 T。

## 接口与使用

- [ACLNN 两阶段接口及 C++ 示例](docs/aclnnTurboQuantSparseFlashAttention.md)。
- [标准 PyTorch 接口](../../torch_extension/cann_ops_transformer/docs/zh/turbo_quant_sparse_flash_attention.md)：`cann_ops_transformer.turbo_quant_sparse_flash_attention`。
- [旧 Python 入口兼容说明](torch_ops_extension/README.md)。
- [构建与安装](docs/install.md)。
- [测试说明](tests/README.md)。
- [实现说明](docs/design.md)。

接口仅支持推理前向，支持 Meta/FakeTensor，不提供 GE Converter 或反向实现。ACL graph 捕获前需完成首次调用及 JIT 编译；回放时保持张量的 shape 和地址不变，仅更新内容。

## 精度说明

精度参考采用相同量化方案的 CPU float64 实现，从实际打包的 KV cache 解码后计算 Attention。比较方法与回归判据见[测试说明](tests/README.md)。
