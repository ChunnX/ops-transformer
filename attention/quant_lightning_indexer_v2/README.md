# QuantLightningIndexerV2

## 产品支持情况
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列加速卡产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

-   API功能：QuantLightningIndexerV2是推理场景下，稀疏attention前处理的计算，选出关键的稀疏token，并对输入query和key进行量化实现存8算8，获取最大收益。

-   计算公式：
    $$out = \text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(\left(Scale_Q@Scale_K^T\right)\odot\left(Q_{index}^{Quant}@{\left(K_{index}^{Quant}\right)}^T\right)\right)\right]\right\}$$
    主要计算过程为：
    1. 将某个token对应的输入参数`query`（$Q_{index}^{Quant}\in\R^{g\times d}$）乘以给定上下文`key`（$K_{index}^{Quant}\in\R^{S_{k}\times d}$），得到相关性。
    2. 相关性结果与`query`和`key`对应的反量化系数`query_dequant_scale`（$Scale_Q$）和`key_dequant_scale`（$Scale_K^T$）相乘，通过激活函数$ReLU$过滤无效负相关信号后，得到当前Token与所有前序Token的相关性分数向量。
    3. 将其与权重系数`weights`（$W$）相乘后，沿g的方向，选取前$Top-k$个索引值得到输出$out$，作为Attention的输入。

## 参数说明
| 参数名                     | 输入/输出/属性 | 描述  | 数据类型       | 数据格式   |
|----------------------------|-----------|----------------------------------------------------------------------|----------------|------------|
| query                     | 输入      | 公式中的$Q_{index}^{Quant}\in\R^{g\times d}$，表示输入Index Query，不支持非连续。layout_q为BSND时shape为[B, S1, N1, D]，layout_q为TND时shape为[T1, N1, D]。D仅支持128；N1在Atlas A3/A2上仅支持64，在Ascend 950PR/950DT上支持32或64 | INT8、FLOAT8_e4m3fn、HIFLOAT8 | ND         |
| key                   | 输入      | 公式中的$K_{index}^{Quant}\in\R^{S_{k}\times d}$，表示压缩后的输入Index Key，支持0轴非连续。layout_k为PA_BSND时shape为[block_num, block_size, N2, D]；layout_k为BSND时shape为[B, S2, N2, D]；layout_k为TND时shape为[T2, N2, D]。D仅支持128；N2仅支持1 | INT8、FLOAT8_e4m3fn、HIFLOAT8 | ND |
| weights                 | 输入      | 公式中的$W$，表示权重系数，不支持非连续。layout_q为BSND时shape为[B, S1, N1]，layout_q为TND时shape为[T1, N1] | FLOAT16、FLOAT32 | ND |
| query_dequant_scale             | 输入      | 公式中的$Scale_Q$，表示Index Query的反量化系数，不支持非连续。shape与weights一致 | FLOAT16、FLOAT32     | ND         |
| key_dequant_scale            | 输入      | 公式中的$Scale_K$，表示Index Key的反量化系数，支持0轴非连续。layout_k为PA_BSND时shape为[block_num, block_size, N2] | FLOAT16、FLOAT32       | ND         |
| cu_seqlens_q                    | 可选输入      | layout_q为TND时必须传入，表示每个Batch中`query`的有效token数前缀和，shape为[B+1]，每个元素表示当前batch与之前所有batch的token数总和，后一个元素的值必须大于等于前一个元素的值，不能出现负值；layout_q为BSND时不能传入 | INT32       | ND         |
| cu_seqlens_k                    | 可选输入      | layout_k为TND时必须传入，表示每个Batch中`key`的有效token数前缀和，shape为[B+1]；layout_k为PA_BSND或BSND时不能传入 | INT32       | ND         |
| seqused_q                    | 可选输入      | layout_q为BSND时可选传入，表示每个Batch中`query`的有效token数，shape为[B]，可传入None表示与query的S长度相同 | INT32       | ND         |
| seqused_k                    | 可选输入      | layout_k为PA_BSND或BSND时使用，表示每个Batch中`key`的有效token数，shape为[B]。layout_k为PA_BSND时必须传入 | INT32       | ND         |
| cmp_residual_k                    | 可选输入      | 压缩场景下Key的残余长度，shape为[B]。当`cmp_ratio`不为1且`mask_mode`不为0时必须传入，否则不能传入 | INT32       | ND         |
| block_table                    | 可选输入      | 表示PageAttention中KV存储使用的block映射表，shape为[B, block_num]。layout_k为PA_BSND时必须传入，layout_k不为PA_BSND时不能传入。block_size取值为16的整数倍，最大支持1024 | INT32       | ND         |
| output_idx_offset                    | 可选输入      | 输出索引的偏移量 | INT32       | ND         |
| metadata                    | 可选输入      | QuantLightningIndexerV2Metadata算子传入的分核信息，包含使用核数、分块大小以及每个核处理数据的起始点等内容，shape大小为[1024]，必须传入，不支持传空 | INT32       | ND         |
| quant_mode                 | 属性      | 用于标识输入的量化模式。Atlas A3/A2仅支持传入2（Per-Token-Head量化）；Ascend 950PR/950DT支持1（FLOAT8_e4m3fn量化）和4（HIFLOAT8量化） | INT32          | -         |
| max_seqlen_q                 | 可选属性| Query的最大序列长度，默认值-1表示任意可能长度 | INT32 | -         |
| layout_q                 | 可选属性| 用于标识输入`query`的数据排布格式，当前支持BSND、TND，默认值"BSND" | STRING | -         |
| layout_k      | 可选属性      | 用于标识输入`key`的数据排布格式。Atlas A3/A2仅支持PA_BSND，默认值"PA_BSND"；Ascend 950PR/950DT支持PA_BSND、BSND、TND，默认值"PA_BSND"。layout_k不为PA_BSND时，layout_q和layout_k必须一致 | STRING          | -         |
| topk  | 属性      | 代表topK阶段需要保留的索引数量，支持[1, 2048]，默认值2048| INT32          | -         |
| mask_mode | 可选属性      | 表示mask的模式，支持0/3。mask_mode为0时代表defaultMask模式；mask_mode为3时代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。默认值0 | INT32          | -         |
| cmp_ratio      | 可选属性      | 用于稀疏计算，表示key的压缩倍数。Atlas A3/A2支持2的幂次方且范围为[1, 128]，即1/2/4/8/16/32/64/128；Ascend 950PR/950DT支持(0, 128]范围内的正整数。默认值1 | INT32          | -         |
| return_value      |  可选属性     | 表示是否输出`sparse_values`。True表示输出，False表示不输出，默认值为False。仅Ascend 950PR/950DT支持设为True，Atlas A3/A2不支持 | BOOL          | -         |
| sparse_indices     | 输出      | 公式中的输出Out，参与稀疏attention计算的token索引值。layout_q为BSND时shape为[B, S1, N2, topk]，layout_q为TND时shape为[T1, N2, topk] | INT32          | ND         |
| sparse_values           | 输出      | 公式中的Indices输出对应的value值。shape与sparse_indices一致。仅当return_value为True时输出有效值，否则输出bf16负无穷 | BFLOAT16         | ND          |

> **维度说明：** B（Batch Size）表示输入样本批量大小；S/S1/S2（Sequence Length）表示输入样本序列长度，其中S1对应query的序列长度，S2对应key的序列长度；N/N1/N2（Head Num）表示多头数，其中N1对应query的head数量，N2对应key的head数量；D（Head Dim）表示hidden层最小的单元尺寸，满足D=H/N；T/T1/T2（Total Tokens）表示所有Batch输入样本序列长度的累加和，其中T1对应query，T2对应key；g表示GQA的group size，g=N1/N2；block_num和block_size分别表示PageAttention场景下的block总数和每个block的token数。

- <term>Ascend 950PR/Ascend 950DT</term>：query、key在quant_mode为1时支持FLOAT8_e4m3fn，quant_mode为4时支持HIFLOAT8，不支持INT8；weights、query_dequant_scale和key_dequant_scale支持FLOAT32，不支持FLOAT16。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：query、key支持INT8，不支持FLOAT8_e4m3fn和HIFLOAT8；weights、query_dequant_scale和key_dequant_scale支持FLOAT16，不支持FLOAT32。


## 约束说明
-   该接口支持图模式。
-   该接口要求$W \odot Scale_Q$的结果在`float16`（Atlas A3/Atlas A2）/`float32`（Ascend 950PR/Ascend 950DT）的表示范围内。
-   该接口的TopK过程对NAN排序是未定义行为。
-   key和key_dequant_scale仅支持0轴非连续。
-   当`quant_mode`为4时（仅Ascend 950PR/Ascend 950DT），`query_dequant_scale`和`key_dequant_scale`的shape维度必须为1且shape[0]=1（即Per-Tensor量化）。

## 调用示例

| 调用方式 | 调用样例 | 说明 |
|----------|----------|------|
| aclnn调用 | [test_aclnn_quant_lightning_indexer_v2](./examples/test_aclnn_quant_lightning_indexer_v2.cpp) | 通过[aclnnQuantLightningIndexerV2](./docs/aclnnQuantLightningIndexerV2.md)接口方式调用QuantLightningIndexerV2算子。 |
| PyTorch API | - | 通过[torch.ops.cann_ops_transformer.quant_lightning_indexer](../../torch_extension/cann_ops_transformer/docs/zh/quant_lightning_indexer.md)接口调用QuantLightningIndexerV2算子。 |
