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

- API功能：TurboQuantSparseFlashAttention算子面向DeepSeek Sparse Attention（DSA）推理场景，对TQ4量化存储的分页KV cache执行融合反量化与稀疏Attention计算。

- 计算公式：

    $$
    O = \text{softmax}(Q@\tilde{K}^T \cdot \text{scale\_value})@\tilde{V}
    $$

    其中 $Q=[Q_{latent},Q_{rope}]$，$\tilde{K}=[\tilde{K}_{latent},K_{rope}]$，$\tilde{V}=\tilde{K}_{latent}$。稀疏索引选择参与计算的KV；K与V共用latent，rope只参与QK计算。

- 量化方案：latent每个token先做L2归一化，各元素按16个固定码本中心量化为4bit，两个nibble打包为1字节，低4bit对应偶数维。设码本查表重建向量为c_t，写cache时准备的scale为`s_t = 原始latent的L2范数 / c_t的L2范数`，以float16存放。latent反量化结果为`s_t * c_t`；rope按同一个已舍入的s_t预除，以与query相同的dtype存放`rope / s_t`。零latent的scale和除法保护由压缩端处理，需保证写入值有限且rope预除有效。

- 坐标域约定：query的latent与码本重建latent必须位于同一坐标域，attention_out也位于该域。若压缩端使用正交旋转R，则调用方应传入`query_latent @ R`，并在需要原始latent坐标时对输出应用`R`的逆变换。本算子不执行该旋转及逆变换，rope分量不参与latent旋转。

## 参数说明

| 参数名           |输入/输出/属性|    描述    | 数据类型    |数据格式|
|-------------|------------|------|-----|-----|
|query|输入|公式中的 $Q$，layout_query仅支持TND，shape为 [T1, N1, D]，最后一维D含rope分量。|FLOAT16、BFLOAT16|ND|
|key|输入|公式中 $\tilde{K}$ 的4bit量化存储，以INT8承载打包后的nibble、rope与反量化系数。|INT8|ND|
|value|输入|请传入与key相同的张量。|INT8|ND|
|sparse_indices|输入|batch内局部稀疏块索引，索引i表示从i × sparse_block_size开始的token块。|INT32|ND|
|key_dequant_scale|可选输入|key的反量化系数。当前COMBINE模式下系数存放于KV slot内，本输入不参与计算，可不传。|FLOAT|ND|
|value_dequant_scale|可选输入|value的反量化系数。当前COMBINE模式下不参与计算，可不传。|FLOAT|ND|
|block_table|输入|PageAttention中KV cache存储使用的block映射表。|INT32|ND|
|actual_seq_lengths_query|输入|各Batch中query的有效token数。TND场景下为累加和形式，即前序Batch与当前Batch有效token数的累加值。|INT32|ND|
|actual_seq_lengths_kv|输入|各Batch中KV的有效token数。|INT32|ND|
|attention_out|输出|Attention计算结果，layout为TND，shape为 [T1, N1, D - rope_head_dim]，即最后一维为query最后一维减去rope分量，数据类型与query相同。|FLOAT16、BFLOAT16|ND|
|softmax_max|输出|softmax过程中按行取得的最大值。|FLOAT|ND|
|softmax_sum|输出|softmax过程中按行取得的指数和。|FLOAT|ND|
|scale_value|属性|QK分数缩放系数，默认值1.0。|FLOAT|-|
|key_quant_mode|属性|key的量化模式，当前仅支持3（TQ4码本量化），默认值3。|INT64|-|
|value_quant_mode|属性|value的量化模式，当前仅支持3（TQ4码本量化），默认值3。|INT64|-|
|sparse_block_size|可选属性|稀疏选取的block粒度，支持1、2、4、8、16，默认值1。|INT64|-|
|layout_query|可选属性|query的数据排布格式，当前仅支持TND，默认值TND。|STRING|-|
|layout_kv|可选属性|KV的数据排布格式，支持PA_BSND和PA_BNBD（PageAttention），默认值PA_BSND。|STRING|-|
|sparse_mode|可选属性|mask模式，0表示无因果mask，3表示右下对齐因果mask，默认值3。|INT64|-|
|pre_tokens|可选属性|query对过去token的计算数量，默认值INT64_MAX。|INT64|-|
|next_tokens|可选属性|query对未来token的计算数量，默认值INT64_MAX。|INT64|-|
|attention_mode|可选属性|底层Attention计算模式，当前DSA路径仅支持2（MLA-absorb），默认值2。|INT64|-|
|quant_scale_repo_mode|可选属性|量化系数的存放模式。当前仅支持1（COMBINE），即反量化系数与Nope、Rope合并存放于同一个KV slot内，默认值1。|INT64|-|
|tile_size|可选属性|量化粒度，默认值128。|INT64|-|
|rope_head_dim|可选属性|rope分量的head dim，默认值64。|INT64|-|
|return_softmax_lse|可选属性|是否输出softmax_max和softmax_sum，不直接返回log-sum-exp；默认值false。|BOOL|-|

## 约束说明

-   该接口支持推理场景下使用。
-   数据类型支持范围：query与attention_out支持`FLOAT16`和`BFLOAT16`，且两者必须同型；key与value仅支持`INT8`；sparse_indices仅支持`INT32`。
-   数据排布支持范围：query与attention_out仅支持`TND`；key与value支持`PA_BSND [Bn, Bs, 1, 386]`和`PA_BNBD [Bn, 1, Bs, 386]`，默认值为`PA_BSND`。Bn为物理block数，Bs为block size，KV head数固定为1。
-   两种KV布局均要求Bs为 [16, 1024] 范围内的16的倍数，且可被`sparse_block_size`整除。layout属性必须与输入shape的轴顺序匹配；不接受`PA_BNSD`或`BNBD`字符串别名，也不支持多KV head。
-   两种PA布局均支持KV cache第0轴（物理block轴）非连续，页内仍须连续。`stride(0)`以字节为单位，不要求是386的倍数，支持间隔取页、页间padding和合法的非零`storage_offset`。
-   多页cache要求`stride(0) >= Bs * 386`，不支持重叠页、负stride或页内非连续；长度为1的轴不约束其stride。寻址跨度`(Bn - 1) * stride(0) + Bs * 386`不得超过`INT64_MAX`，调用方须提供足够的底层存储。空query也须满足这些布局约束。
-   query的最后一维包含rope分量，attention_out的最后一维为query最后一维减去`rope_head_dim`，即 $D_{out} = D_{query} - rope\_head\_dim$。
-   quant_scale_repo_mode仅支持1（COMBINE）：反量化系数存放于KV slot尾部2字节；key_dequant_scale与value_dequant_scale两个可选输入不参与计算，可不传。
-   调用方必须按 $rope / s_t$ 存放rope分量，数据类型与query相同。
-   `actual_seq_lengths_query`为各batch query长度的累加和，末元素必须等于query的T；`actual_seq_lengths_kv`为每个batch的实际KV长度，不是累加和。
-   下列数值约束由调用方保证：两个长度数组均非负；`actual_seq_lengths_query`单调不减；每个batch的KV长度不超过其`block_table`行所能覆盖的容量（`block_table.dim1 × block_size`）；`block_table`中的block id落在KV cache的block数范围内；`sparse_indices`为batch内局部稀疏块索引，有效前缀后以`-1`填充。
    接口不校验这些设备张量中的数值，传入越界值可能导致非法内存访问。
-   某个batch的KV长度允许为0。两种布局下该batch的`attention_out=0`；请求softmax输出时，`softmax_max=-2e38`、`softmax_sum=0`。未请求时两个softmax输出均为shape [0]；请求时shape均为 [1, T1, N1]。
-   某个batch的query长度允许为0，对应`actual_seq_lengths_query`的累计值可以重复，该batch不产生输出行。空query batch和零KV batch不影响其他batch的输出。
-   支持整体query的T1=0：返回shape [0, N1, 512] 的空输出；请求softmax输出时其shape为 [1, 0, N1]，否则为 [0]。Batch数、query head数、KV物理block数、block size和block_table列数仍须满足非空输入的约束；长度数组的所有query累计值须为0。实际KV长度为0不代表支持空的物理KV cache。
-   KV长度非零但sparse_indices全为 -1的行，输出为0，softmax_max为 -2e38，softmax_sum不保证为0。调用方不可仅用softmax_sum是否为0判断行有效性。
-   KV latent的headDim为512，rope_head_dim默认64。每个KV slot由256字节打包nibble、128字节与query同dtype的rope分量和2字节`float16`归一化系数顺序拼接组成，总宽度固定为386字节。
-   sparse_indices中的无效位置以 -1填充，需保证每行有效值均在前半部分。
-   参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸、T表示所有Batch输入样本序列长度的累加和。
-   S1表示query shape中的S，N1表示num_query_heads，T1表示query shape中的T。

## 调用说明

- [ACLNN两阶段接口及C++ 示例](docs/aclnnTurboQuantSparseFlashAttention.md)。
- [标准PyTorch接口](../../torch_extension/cann_ops_transformer/docs/zh/turbo_quant_sparse_flash_attention.md)：`cann_ops_transformer.turbo_quant_sparse_flash_attention`。
- [旧Python入口兼容说明](torch_ops_extension/README.md)。
- [构建与安装](docs/install.md)。
- [测试说明](tests/README.md)。

接口支持推理前向，具体框架调用方式和约束见PyTorch接口文档。

## 精度说明

精度参考采用相同量化方案的CPU float64实现，从实际打包的KV cache解码后计算Attention。比较方法与回归判据见[测试说明](tests/README.md)。
