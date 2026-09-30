# aclnnGenericLightningIndexer

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/generic_lightning_indexer)

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

- 接口功能：`aclnnGenericLightningIndexer`对已经完成前处理的 Index Query / Index Key 做 `Matmul → 激活 → 沿 head 规约 → 掩码 → TopK →（可选 Expand / 追加尾 token）`，得到每一个 token 对应的 Top-$k$ 位置。不包含 RoPE、RMSNorm、AvgPool、KeyPool、CSA 压缩。压缩后的 Key 通过 `cmpRatio` 消费。

  该接口用于收编 `lightning_indexer`、`lightning_indexer_v2`、`quant_lightning_indexer`、`quant_lightning_indexer_v2`、`pool_key_indexer`、Qwen Compressed Lightweight Indexer 以及 `MsaIndexScore`。MSA 通过 `actType=1` / `keyReduceMode=1` / `selectMode=1` 成套开启 identity + Maxpool，输出 block score、不融合 TopK。

- 计算公式：

  $$
  idx = \mathrm{Post}\Bigl(\mathrm{Top}\text{-}k\bigl(\mathrm{Reduce}_{h}(\phi(\alpha\, Q K^{\top}))\odot M\bigr)\Bigr)
  $$

  - $\phi$：`actType=0` 为 ReLU，`actType=1` 为 Identity。
  - $\mathrm{Reduce}_{h}$：`hasWeightedSum=true` 时为 $W$ 加权和，否则为沿 head 累加和。
  - $\alpha$：`scoreScale`。
  - $M$：由 `sparseMode` 与 `cmpRatio` 决定的可见性。
  - $\mathrm{Post}$：identity / Expand / Expand+AppendTail。
  - `topk` 表示**打分轴**选出的个数。旧 PKI 的 token 预算需先除以 `pool_size` 再传入。

  量化场景（`quantMode>=0`）先结合 `qDescale` / `kDescale` 反量化。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用 `aclnnGenericLightningIndexerGetWorkspaceSize` 获取 workspace 大小以及执行器，再调用 `aclnnGenericLightningIndexer` 执行计算。

```Cpp
aclnnStatus aclnnGenericLightningIndexerGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *weightsOptional,
    const aclTensor *actualSeqQOptional,
    const aclTensor *actualSeqKOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKOptional,
    const aclTensor *cmpResidualKOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *qDescaleOptional,
    const aclTensor *kDescaleOptional,
    const aclTensor *outputIdxOffsetOptional,
    const aclTensor *metadataOptional,
    const aclTensor *startLocOptional,
    int64_t          topk,
    int64_t          cmpRatio,
    char            *layoutQOptional,
    char            *layoutKOptional,
    int64_t          sparseMode,
    int64_t          actType,
    bool             hasWeightedSum,
    bool             expandIndices,
    bool             appendTail,
    int64_t          quantMode,
    bool             returnValue,
    int64_t          selectMode,
    int64_t          keyReduceMode,
    float            scoreScale,
    int64_t          keyStride0,
    int64_t          kDescaleStride0,
    int64_t          maxSeqlenQ,
    int64_t          initBlocks,
    int64_t          localBlocks,
    const aclTensor *sparseIndicesOut,
    const aclTensor *sparseValuesOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor);
```

```Cpp
aclnnStatus aclnnGenericLightningIndexer(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    const aclrtStream stream);
```

## aclnnGenericLightningIndexerGetWorkspaceSize

- **参数说明：**

  > [!NOTE]
  >
  > - 参数维度含义：B 表示 Batch Size、S1 / S2 表示 query / key 的 Sequence Length（S2 为压缩后长度）、N1 / N2 表示 Head Num、D 表示 Head Dim（仅支持 128）、T1 / T2 表示 Total Tokens。N2 仅支持 1。
  > - 序列类可选输入（actualSeq*、seqused*、cmpResidualK、startLoc）为 INT32 或 INT64 tensor。TND 的 actualSeq 同时支持 LI 式 exclusive prefix `[B]` 与 LIv2/MSA 式 cu_seqlens `[B+1]`（首元素为 0）。seqused / startLoc / block_table 为 `[B]`。

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|---------------|------|----------|----------|
| query | 输入 | 公式中的 Q。不支持空 tensor。BSND：[B,S1,N1,D]；TND：[T1,N1,D]。 | FLOAT16、BFLOAT16、INT8、FLOAT8_E4M3FN、HIFLOAT8、FLOAT4_E2M1、FLOAT8_E5M2 | ND |
| key | 输入 | 公式中的 K（已压缩或未压缩）。PA_BBND / PA_BSND / BBND：[block_num, block_size, N2, D]；BNBD：[block_num, N2, block_size, D]（N2=1 时与 BBND 等价）；支持 0 轴非连续；BSND：[B,S2,N2,D]；TND：[T2,N2,D]。默认与 query 同 dtype；MSA 额外支持 FP16×INT8。 | 同 query，MSA 可为 INT8 | ND |
| weightsOptional | 输入 | 公式中的 W。hasWeightedSum=true 时必选；false 时必须为空。BSND：[B,S1,N1]；TND：[T1,N1]。 | FLOAT16、BFLOAT16、FLOAT | ND |
| actualSeqQOptional | 输入 | TND 时必选。LI exclusive prefix `[B]` 或 cu_seqlens `[B+1]`。 | INT32、INT64 | ND |
| actualSeqKOptional | 输入 | TND / PA 时必选。TND 同上；PA 为每 batch 压缩长度 `[B]` 或 cu `[B+1]`。 | INT32、INT64 | ND |
| sequsedQOptional | 输入 | 每 batch Query 真实长度（非前缀和），用于 padding。[B] | INT32、INT64 | ND |
| sequsedKOptional | 输入 | 每 batch Key 真实长度。[B] | INT32、INT64 | ND |
| cmpResidualKOptional | 输入 | 压缩残余 token 数。cmpRatio>1 且 sparseMode!=0 时必选。[B] | INT32、INT64 | ND |
| blockTableOptional | 输入 | PA 时必选。[B, maxBlockNumPerSeq] | INT32 | ND |
| qDescaleOptional | 输入 | Query 反量化系数。quantMode>=0 时必选（MSA FP16×INT8 可省略）。含 QLI FP8+FP16 scale。 | FLOAT、FLOAT16、FLOAT8_E8M0 | ND |
| kDescaleOptional | 输入 | Key 反量化系数。quantMode>=0 或 INT8 key 时必选。PA 下支持 0 轴非连续。 | 同 qDescale | ND |
| outputIdxOffsetOptional | 输入 | 输出索引逐 batch 偏移。[B] | INT32 | ND |
| metadataOptional | 输入 | 预留负载均衡表。[1024] | INT32 | ND |
| startLocOptional | 输入 | MSA：query 所在逻辑 block，用于 local_mask。[B] | INT32、INT64 | ND |
| topk | 属性 | 打分轴选出的个数。[1,2048] ∪ {3072,4096,5120,6144,7168,8192} | INT64 | - |
| cmpRatio | 属性 | 压缩比 / pool 大小。[1,128]，1 表示未压缩 | INT64 | - |
| layoutQOptional | 属性 | BSND / TND，默认 BSND | STRING | - |
| layoutKOptional | 属性 | BSND / TND / PA_BBND / PA_BSND / BBND / BNBD | STRING | - |
| sparseMode | 属性 | 0 default / 3 rightDownCausal / 4 blockCompleteCausal | INT64 | - |
| actType | 属性 | 0 ReLU / 1 Identity | INT64 | - |
| hasWeightedSum | 属性 | true 乘 W 再求和 | BOOL | - |
| expandIndices | 属性 | TopK 后是否展开为 token 下标 | BOOL | - |
| appendTail | 属性 | 展开后是否追加残余 token | BOOL | - |
| quantMode | 属性 | -1 不量化；0/1 FP8 PTH；2 INT8；3 MXFP8；4 HiFP8；5 MXFP4 | INT64 | - |
| returnValue | 属性 | 是否输出 sparseValues | BOOL | - |
| selectMode | 属性 | 0 fused TopK；1 none（MSA） | INT64 | - |
| keyReduceMode | 属性 | 0 none；1 maxpool（MSA） | INT64 | - |
| scoreScale | 属性 | QK 缩放，默认 1.0 | FLOAT | - |
| keyStride0 | 属性 | PA 下 key 0 轴 stride，-1 由 shape 推导 | INT64 | - |
| kDescaleStride0 | 属性 | PA 下 k_descale 0 轴 stride，-1 由 shape 推导 | INT64 | - |
| maxSeqlenQ | 属性 | TND 每 batch 最大 Q 长度，-1 未指定 | INT64 | - |
| initBlocks | 属性 | MSA local_mask sink 块数 | INT64 | - |
| localBlocks | 属性 | MSA local_mask 窗口块数 | INT64 | - |
| sparseIndicesOut | 输出 | 选中下标，无效填 -1。BSND：[B,S1,L_out]；TND：[T1,L_out]；MSA 为 (0,) | INT32 | ND |
| sparseValuesOut | 输出 | returnValue=true 时为 [B,S1,topk] 或 [T1,topk]；false 时 (0,)；MSA BSND 为 [B,S1,N1,Align16(ceil(S2/cmpRatio))]，MSA TND 为 [N1,T1,nOut] | FLOAT | ND |
| workspaceSize | 输出 | 需要的 workspace 字节数 | UINT64 | - |
| executor | 输出 | 算子执行器 | aclOpExecutor* | - |

- **返回值：**

  返回 `aclnnStatus` 状态码，具体参见 [aclnn 返回码](../../../docs/zh/context/aclnn_return_code.md)。

## aclnnGenericLightningIndexer

- **参数说明：**

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| workspace | 输入 | 工作空间地址 |
| workspaceSize | 输入 | 工作空间大小 |
| executor | 输入 | GetWorkspaceSize 返回的执行器 |
| stream | 输入 | 执行流 |

## 约束说明

- HeadDim 仅支持 128，N2 仅支持 1，N1∈[1,64]。
- `appendTail=true` 需要 `expandIndices=true`。
- `hasWeightedSum=true` 必须传 weights，反之必须为空。
- `actType=1`、`keyReduceMode=1`、`selectMode=1` 必须成套（MSA）。
- layout_k 为 PA（PA_BBND / PA_BSND / BBND / BNBD）时必须传 blockTable 与 actualSeqK（或 sequsedK）；非 PA 不支持 blockTable。
- layout_q / layout_k 为 TND 时必须传对应 actualSeq。
- quantMode=-1 时不支持传入 qDescale / kDescale。
- <term>Ascend 950PR/Ascend 950DT</term>：支持 FLOAT16 / BFLOAT16 / INT8 / FLOAT8_E4M3FN / HIFLOAT8 / FLOAT4_E2M1 / FLOAT8_E5M2。
- <term>Atlas A2 / A3</term>：不支持。

## 调用示例

见 [test_aclnn_generic_lightning_indexer.cpp](../examples/test_aclnn_generic_lightning_indexer.cpp)。
