# GenericLightningIndexer

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：`generic_lightning_indexer`对**已经完成前处理**的 Index Query / Index Key 做打分与选择。计算分成四段：相关性、逐元素变换、沿 head 规约、TopK（含可选展开与拼接）。当前优先实现 Qwen Compressed Lightweight Indexer：ReLU、不加权求和、`sparse_mode=4`、展开后拼接当前未完成的 micro-block。不包含 RoPE、RMSNorm、AvgPool。仅支持 Ascend 950。

- 计算公式：

    $$
    idx = \mathrm{Post}\Bigl(\mathrm{Top}\text{-}k\bigl(\mathrm{Reduce}_{h}(\phi(\alpha\, Q K^{\top}))\odot M\bigr)\Bigr)
    $$

    - $\phi$：`act_type=0` 为 $\mathrm{ReLU}$，`act_type=1` 为 Identity（MSA）。
    - $\mathrm{Reduce}_{h}$：TopK 路径下 `has_weighted_sum=true` 时为 $W$ 加权和，否则为沿 head 累加和；MSA 路径保留 N1，不做 head 规约。
    - $\alpha$：`score_scale`，PKI 场景传 $1/\sqrt{d}$。
    - $M$：由 `sparse_mode` 与 `cmp_ratio` 决定的可见性掩码。
    - $\mathrm{Post}$：identity / Expand / Expand+AppendTail。
    - $k=\texttt{topk}$ 一律表示**打分轴**选出的个数；旧 PKI 的 token 预算需先除以 `pool_size` 再传入。
    - MSA：`select_mode=1` 时不做 TopK，对每个 head 沿 Key 做窗宽 `cmp_ratio` 的 Maxpool，输出 `[B,S1,N1,Align16(\lceil S2/r\rceil)]`。

    主要计算过程为：

    1. 将某个 token 对应的 `query`（$Q_{index}\in\R^{g\times d}$）乘以上下文 `key`（$K_{index}\in\R^{S_{k}\times d}$，已压缩或未压缩），再乘以 `score_scale`，得到相关性分数。
    2. 量化场景先结合 `q_descale` / `k_descale` 反量化。
    3. 通过激活函数 $\phi$ 过滤无效负相关信号。
    4. 与权重系数 `weights`（$W$）相乘后沿 $g$ 聚合，或直接沿 head 累加。
    5. 按 `sparse_mode` 做可见性过滤，无效位置置 $-\infty$。
    6. TopK 路径：在打分轴选取前 $\mathrm{Top}\text{-}k$ 个索引；`expand_indices` 为 true 时展开为原始 token 索引，`append_tail` 为 true 时再追加 `cmp_residual_k` 指定的尾部有效 token。
    7. MSA 路径：保留每个 head 的分数，按 `cmp_ratio` 做窗口 Maxpool，可选叠加 `init_blocks` / `local_blocks` 的 local_mask；`sparse_indices` 为空。

## 参数说明

> **说明：**<br>
> 参数维度含义：B 表示 Batch Size、S1 和 S2 分别表示 query 和 key 的 Sequence Length（S2 为压缩后长度）、N1 和 N2 分别表示 query 和 key 的 Head Num、D 表示 Head Dim（仅支持 128）、T1 和 T2 分别表示 query 和 key 的 Total Tokens、block_num 和 block_size 分别表示 PageAttention 场景下的 block 总数和每个 block 包含的压缩 token 数。N2 仅支持 1。

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|---------------|------|----------|----------|
| query | 输入 | 公式中的输入 Q。不支持空 tensor。layout_q 为 BSND 时，shape 为 [B, S1, N1, D]；layout_q 为 TND 时，shape 为 [T1, N1, D]。 | FLOAT16、BFLOAT16、INT8、FLOAT8_E4M3FN、HIFLOAT8、FLOAT4_E2M1、FLOAT8_E5M2 | ND |
| key | 输入 | 公式中的输入 K（已压缩或未压缩）。不支持空 tensor。layout_k 为 PA_BBND / PA_BSND / BBND 时，shape 为 [block_num, block_size, N2, D]；BNBD 为 [block_num, N2, block_size, D]（N2=1 时与 BBND 等价），支持 0 轴非连续；layout_k 为 BSND 时，shape 为 [B, S2, N2, D]；layout_k 为 TND 时，shape 为 [T2, N2, D]。默认与 query 同 dtype；MSA 额外支持 query=FLOAT16 且 key=INT8。 | 同 query，以及 MSA 的 INT8 | ND |
| weights | 输入 | 公式中的输入 W。`has_weighted_sum=true` 时必传且不支持空 tensor；`has_weighted_sum=false` 时必须为空。layout_q 为 BSND 时，shape 为 [B, S1, N1]；layout_q 为 TND 时，shape 为 [T1, N1]。 | FLOAT16、BFLOAT16、FLOAT | ND |
| actual_seq_q | 输入 | 每个 Batch 中 Query 的有效长度。layout_q 为 TND 时必传。支持 LI exclusive prefix `[B]`（元素 i 为 batch 0..i 的 token 总和）以及 LIv2/MSA cu_seqlens `[B+1]`（首元素为 0，长度 = v[i+1]-v[i]）。 | INT32、INT64 | ND |
| actual_seq_k | 输入 | 每个 Batch 中 key 的有效压缩长度。layout_k 为 TND 时必传（同上两种格式）；layout_k 为 PA 时必传（非前缀和 `[B]`，若配合 seqused 也可为 cu `[B+1]`）。 | INT32、INT64 | ND |
| seqused_q | 输入 | 每个 Batch 中 Query 的真实有效 token 数（非前缀和），用于 TND padding。可选。shape 为 [B, ]。 | INT32、INT64 | ND |
| seqused_k | 输入 | 每个 Batch 中 Key 的真实有效长度（非前缀和）。layout_k 为 PA 时建议传入。shape 为 [B, ]。 | INT32、INT64 | ND |
| cmp_residual_k | 输入 | 每个 Batch 中压缩残余（尾部不完整块）的有效 token 数。$cmp\_ratio>1$ 且 `sparse_mode!=0` 时必传。取值范围 [0, cmp_ratio-1]。shape 为 [B, ]。 | INT32、INT64 | ND |
| block_table | 输入 | PageAttention 中 KV 存储使用的 block 映射表。layout_k 为 PA_BBND / PA_BSND 时必传，shape 为 [B, maxBlockNumPerSeq]。 | INT32 | ND |
| q_descale | 输入 | Query 的反量化系数。仅 quant_mode>=0 时有效。 | FLOAT、FLOAT16、FLOAT8_E8M0 | ND |
| k_descale | 输入 | Key 的反量化系数。quant_mode>=0 或 INT8 key 时有效。MSA FP16×INT8 只要求 k_descale。仅 PA 布局下支持 0 轴非连续。 | 同 q_descale | ND |
| output_idx_offset | 输入 | 输出索引的逐 batch 偏移，写回 sparse_indices 前与选中下标相加。可选。shape 为 [B, ]。 | INT32 | ND |
| metadata | 输入 | 预留 LIv2 式负载均衡任务表。Step 1 可不传。shape 为 [1024, ]。 | INT32 | ND |
| start_loc | 输入 | MSA：query 所在逻辑 block，用于 local_mask。可选。shape 为 [B, ]。 | INT32、INT64 | ND |
| layout_q | 属性 | 用于标识输入 Query 的数据排布格式。支持 BSND、TND，默认值为 BSND。 | STRING | - |
| layout_k | 属性 | 用于标识输入 key 的数据排布格式。支持 BSND、TND、PA_BBND、PA_BSND、BBND、BNBD（前四个 PA 别名中 PA_BSND/BBND 视为 PA_BBND；BNBD 为 [BN,N2,BS,D]），默认值为 BSND。 | STRING | - |
| topk | 属性 | 打分轴需要保留的个数。支持 [1, 2048] 以及 3072、4096、5120、6144、7168、8192。默认值为 2048。 | INT64 | - |
| cmp_ratio | 属性 | 压缩比 / pool 大小。支持 [1, 128]，默认值为 1。取值为 1 时表示未压缩。 | INT64 | - |
| sparse_mode | 属性 | 表示 mask 的模式。0 表示 defaultMask，3 表示 rightDownCausal，4 表示 blockCompleteCausal（Qwen）。默认值为 0。 | INT64 | - |
| act_type | 属性 | 激活类型。0 表示 ReLU，1 表示 Identity（仅 MSA 成套属性）。默认值为 0。 | INT64 | - |
| has_weighted_sum | 属性 | true 表示乘 W 再沿 head 求和；false 表示沿 head 累加和且不得传 weights。默认值为 true。 | BOOL | - |
| expand_indices | 属性 | TopK 后是否将打分轴下标展开为原始 token 下标。默认值为 false。 | BOOL | - |
| append_tail | 属性 | 展开后是否追加残余 token。仅 `expand_indices=true` 时允许为 true。默认值为 false。 | BOOL | - |
| quant_mode | 属性 | 量化模式。-1 不量化；0/1 为 FP8 per-token-head（兼容 PKI）；2 INT8；3 MXFP8；4 HiFP8；5 MXFP4。默认值为 -1。 | INT64 | - |
| return_value | 属性 | 是否输出 sparse_values。True 表示输出，False 表示输出 shape 为 (0,) 的空 tensor。默认值为 False。 | BOOL | - |
| select_mode | 属性 | 选择模式。0 表示 fused TopK；1 表示 none（MSA，须与 act_type=1、key_reduce_mode=1 成套）。默认值为 0。 | INT64 | - |
| key_reduce_mode | 属性 | Key 轴规约。0 表示 none；1 表示 maxpool（MSA）。默认值为 0。 | INT64 | - |
| score_scale | 属性 | QK 缩放系数。默认值为 1.0。PKI 传 1/sqrt(128)。 | FLOAT | - |
| key_stride0 | 属性 | key 第 0 轴的 stride（元素单位），用于 PA 场景下 0 轴非连续 key 的寻址。-1 表示未指定，按连续输入由 shape 推导。默认值为 -1。 | INT64 | - |
| k_descale_stride0 | 属性 | k_descale 第 0 轴的 stride（元素单位），用于 PA 场景下 0 轴非连续 k_descale 的寻址。-1 表示未指定。默认值为 -1。 | INT64 | - |
| max_seqlen_q | 属性 | TND 场景下每个 batch Query 的最大长度，供 tiling / padding 使用。-1 表示未指定。默认值为 -1。 | INT64 | - |
| init_blocks | 属性 | MSA local_mask 的 sink 块数。默认值为 0。 | INT64 | - |
| local_blocks | 属性 | MSA local_mask 的窗口块数。默认值为 0。 | INT64 | - |
| sparse_indices | 输出 | 选中的稀疏索引，无效部分填 -1。layout_q 为 BSND 时，shape 为 [B, S1, L_out]；layout_q 为 TND 时，shape 为 [T1, L_out]。L_out 在不展开时为 topk，展开时为 topk * cmp_ratio，再追加尾时为 topk * cmp_ratio + cmp_ratio - 1。MSA（select_mode=1）时 shape 为 (0,)。 | INT32 | ND |
| sparse_values | 输出 | 选中位置对应的分数。return_value 为 True 时，layout_q 为 BSND 时 shape 为 [B, S1, topk]，layout_q 为 TND 时 shape 为 [T1, topk]；return_value 为 False 时，shape 为 (0,)。MSA 时 BSND 为 [B, S1, N1, Align16(ceil(S2/cmp_ratio))]，TND 为 [N1, T1, nOut]（对齐 MsaIndexScore）。 | FLOAT | ND |

## 约束说明

- 该接口支持推理场景下使用，支持单算子模式、TorchAir（GE 图）模式和 TorchAir（aclgraph）图模式调用。
- query 的 N1 支持小于等于 64，key 的 N2 仅支持 1，D（HeadDim）仅支持 128。
- 参数 query、key 的数据类型应保持一致。非量化场景下，weights 可为 FLOAT16 / BFLOAT16 / FLOAT；量化场景下 weights 为 FLOAT16 / BFLOAT16 / FLOAT（对齐各 indexer 组合，见设计文档数据类型表）。
- topk 支持 [1, 2048] 以及 3072、4096、5120、6144、7168、8192；cmp_ratio 支持 [1, 128]。
- 当 cmp_ratio=1 且 expand_indices=false 时，退化为 lightning_indexer 行为，输出维度为 topk。
- 当 layout_k 为 PA_BBND / PA_BSND / BBND / BNBD 时，必须传入 block_table 和 actual_seq_k（或 seqused_k）；当 layout_k 不为 PA 时，不支持传入 block_table。
- 当 layout_q 为 TND 时，必须传入 actual_seq_q；当 layout_k 为 TND 时，必须传入 actual_seq_k。
- PA 场景下，block_size 取值为 16 的倍数，最大支持 2048。
- key、k_descale 仅支持在 PA 布局下 0 轴非连续（需配合 key_stride0 / k_descale_stride0），BSND / TND 布局要求连续输入。
- 当 quant_mode=-1 时，不支持传入 q_descale 和 k_descale。
- `has_weighted_sum=true` 必须传 weights，反之必须为空。
- `append_tail=true` 需要 `expand_indices=true`。
- `act_type=1`、`key_reduce_mode=1`、`select_mode=1` 必须成套出现（MSA）；非 MSA 路径拒绝单独打开其中任一开关。MSA 要求 `has_weighted_sum=false` 且不得 `expand_indices`。`cmp_ratio` 在此剖面表示 maxpool 窗。
- sparse_indices 无效部分填 -1；sparse_values 无效部分填 -inf。
- sparse_mode 所表示的 mask 模式的详细介绍见 [sparse_mode 参数说明](../../docs/zh/context/sparse_mode_introduction.md)。
- <term>Ascend 950PR/Ascend 950DT</term>：
  - query、key 支持 FLOAT16、BFLOAT16、INT8、FLOAT8_E4M3FN、HIFLOAT8、FLOAT4_E2M1、FLOAT8_E5M2。
  - quant_mode 支持 -1 / 0 / 1 / 2 / 3 / 4 / 5。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|----------|----------|------|
| aclnn 接口 | [test_aclnn_generic_lightning_indexer](./examples/test_aclnn_generic_lightning_indexer.cpp) | 通过 [aclnnGenericLightningIndexer](./docs/aclnnGenericLightningIndexer.md) 接口方式调用算子 |
| PyTorch 接口 | — | 通过 [torchapi_generic_lightning_indexer](./docs/torchapi_generic_lightning_indexer.md) 接口方式调用算子 |
