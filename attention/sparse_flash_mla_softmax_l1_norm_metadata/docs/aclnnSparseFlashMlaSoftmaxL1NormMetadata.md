# aclnnSparseFlashMlaSoftmaxL1NormMetadata

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                                   |    √     |
| Atlas A3训练系列产品/Atlas A3推理系列产品                 |    x     |
| Atlas A2训练系列产品/Atlas A2推理系列产品                 |    x     |
| Atlas 200I/500 A2推理产品                                  |    x     |
| Atlas推理系列产品                                          |    x     |
| Atlas训练系列产品                                          |    x     |

## 功能说明

- 算子功能：该算子为AICPU算子，是aclnnSparseFlashMlaSoftmaxL1Norm算子的前置算子。根据aclnnSparseFlashMlaSoftmaxL1Norm算子的输入shape、layout、mask等信息，计算并输出分核切分metadata。输出结果可作为aclnnSparseFlashMlaSoftmaxL1Norm算子的metadataOptional输入，减少主算子tiling阶段对host array的访问。

  **该算子不建议单独使用，建议与aclnnSparseFlashMlaSoftmaxL1Norm算子配合使用，形成完整的工作流。**
    1. 接收主算子的shape信息，包括batchSize、maxSeqLenQ、maxSeqLenK、numHeadsQ、numHeadsK、headDim、topk、layout和mask信息。
    2. 根据layout和可选输入seqUsedQOptional计算参与负载均衡的seq总数，采用strided方式将任务均衡切分到可用AIC核上。
    3. 输出metadata后，后续作为aclnnSparseFlashMlaSoftmaxL1Norm算子的metadataOptional输入使用。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用"aclnnSparseFlashMlaSoftmaxL1NormMetadataGetWorkspaceSize"获取workspace大小，再调用"aclnnSparseFlashMlaSoftmaxL1NormMetadata"执行计算。

``` cpp
aclnnStatus aclnnSparseFlashMlaSoftmaxL1NormMetadataGetWorkspaceSize(
    const aclTensor *cuSeqLensQOptional,
    const aclTensor *cuSeqLensKOptional,
    const aclTensor *seqUsedQOptional,
    const aclTensor *seqUsedKOptional,
    const aclTensor *cmpResidualKOptional,
    const aclTensor *topkLengthOptional,
    int64_t batchSize,
    int64_t maxSeqLenQ,
    int64_t maxSeqLenK,
    int64_t numHeadsQ,
    int64_t numHeadsK,
    int64_t headDim,
    int64_t topk,
    int64_t cmpRatio,
    int64_t maskMode,
    char *layoutQ,
    char *layoutK,
    const aclTensor *metadata,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

``` cpp
aclnnStatus aclnnSparseFlashMlaSoftmaxL1NormMetadata(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

<!-- TODO: 待补充完整参数说明、约束说明、Metadata输出布局、调用示例等 -->
