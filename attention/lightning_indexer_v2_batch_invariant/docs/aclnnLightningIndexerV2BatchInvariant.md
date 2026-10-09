# aclnnLightningIndexerV2BatchInvariant

## 产品支持情况

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持。

## 功能说明

本接口的计算功能和参数含义与 `aclnnLightningIndexerV2` 一致，并默认保证确定性和 batch 一致性。
相同输入重复执行的输出完全一致；对于完全相同的单条样本输入和算子属性，该样本的输出不受 batch 大小或 batch 内位置变化影响。

本算子使用两段式接口：先获取 workspace 大小和执行器，再执行算子。

```cpp
aclnnStatus aclnnLightningIndexerV2BatchInvariantGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensKOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKOptional,
    const aclTensor *cmpResidualKOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *outputIdxOffsetOptional,
    const aclTensor *metadataOptional,
    int64_t topk,
    int64_t maxSeqlenQ,
    char *layoutQ,
    char *layoutK,
    int64_t maskMode,
    int64_t cmpRatio,
    int64_t returnValue,
    const aclTensor *sparseIndices,
    const aclTensor *sparseValues,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

```cpp
aclnnStatus aclnnLightningIndexerV2BatchInvariant(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## 参数说明

| 参数 | 输入/输出 | 数据类型 | 格式 | 说明 |
| :-- | :--: | :-- | :--: | :-- |
| `q` | 输入 | FLOAT16、BFLOAT16 | ND | Q；BSND 为 `(B,S1,N1,D)`，TND 为 `(T1,N1,D)`。 |
| `k` | 输入 | FLOAT16、BFLOAT16 | ND | K；支持 BSND、TND 和 PA_BBND 对应 shape。 |
| `w` | 输入 | FLOAT32 | ND | 权重；BSND 为 `(B,S1,N1)`，TND 为 `(T1,N1)`。 |
| `cuSeqlensQOptional` | 可选输入 | INT32 | ND | q 的累积序列长度，shape 为 `(B+1,)`。 |
| `cuSeqlensKOptional` | 可选输入 | INT32 | ND | k 的累积序列长度，shape 为 `(B+1,)`。 |
| `sequsedQOptional` | 可选输入 | INT32 | ND | 各 batch 中 q 的实际使用长度，shape 为 `(B,)`。 |
| `sequsedKOptional` | 可选输入 | INT32 | ND | 各 batch 中 k 的实际使用长度，shape 为 `(B,)`。 |
| `cmpResidualKOptional` | 可选输入 | INT32 | ND | k 压缩前 token 数除以 `cmpRatio` 的余数，shape 为 `(B,)`。 |
| `blockTableOptional` | 可选输入 | INT32 | ND | PageAttention 的 block 映射表。 |
| `outputIdxOffsetOptional` | 可选输入 | INT32 | ND | top-k 索引输出偏移。 |
| `metadataOptional` | 可选输入 | INT32 | ND | 兼容 LightningIndexerV2 的 metadata 参数。 |
| `topk` | 属性 | INT64 | - | 保留的 top-k 数量。 |
| `maxSeqlenQ` | 属性 | INT64 | - | q 的最大序列长度。 |
| `layoutQ` | 属性 | STRING | - | q 的 layout，支持 `BSND`、`TND`。 |
| `layoutK` | 属性 | STRING | - | k 的 layout，支持 `BSND`、`TND`、`PA_BBND`。 |
| `maskMode` | 属性 | INT64 | - | mask 模式。 |
| `cmpRatio` | 属性 | INT64 | - | k 的压缩倍数。 |
| `returnValue` | 属性 | INT64 | - | 是否返回 top-k 分数。 |
| `sparseIndices` | 输出 | INT32 | ND | top-k 索引。 |
| `sparseValues` | 输出 | FLOAT32 | ND | top-k 分数；`returnValue=0` 时为空 Tensor。 |

## 约束说明

- `q` 和 `k` 数据类型必须一致，且不支持空 Tensor。
- head dim 仅支持 128；`q` 的 head 数支持 1~64，`k` 的 head 数仅支持 1。
- `topk` 支持 [1, 2048]，以及 3072、4096、5120、6144、7168、8192。
- 当 `layoutK` 为 `PA_BBND` 时，必须传入 `sequsedKOptional`。
- 当 `maskMode=3` 且 `cmpRatio>1` 时，传入的 `cmpResidualKOptional` 中每个值必须小于 `cmpRatio`。
- A2/A3 上不支持 `sequsedQOptional`、`outputIdxOffsetOptional`、`maxSeqlenQ` 功能，不建议传入。
- 无效位置的 `sparseIndices` 填充 -1，`sparseValues` 填充 -inf。

## 返回值

返回 `ACL_SUCCESS` 表示成功，其他返回值表示失败。
