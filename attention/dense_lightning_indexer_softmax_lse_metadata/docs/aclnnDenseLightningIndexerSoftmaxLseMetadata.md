# aclnnDenseLightningIndexerSoftmaxLseMetadata

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

- 算子功能：该算子为AICPU算子，是aclnnDenseLightningIndexerSoftmaxLse算子的前置算子。根据aclnnDenseLightningIndexerSoftmaxLse算子的输入shape、layout、mask等信息，计算并输出分核切分metadata。输出结果可作为aclnnDenseLightningIndexerSoftmaxLse算子的metadataOptional输入，减少主算子tiling阶段对host array的访问。

  **该算子不建议单独使用，建议与aclnnDenseLightningIndexerSoftmaxLse算子配合使用，形成完整的工作流。**

  <!-- TODO: 待补充详细功能说明 -->

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnDenseLightningIndexerSoftmaxLseMetadataGetWorkspaceSize"获取workspace大小，再调用"aclnnDenseLightningIndexerSoftmaxLseMetadata"执行计算。

``` cpp
// TODO: 待确认参数列表
aclnnStatus aclnnDenseLightningIndexerSoftmaxLseMetadataGetWorkspaceSize(
    // const aclTensor *xxxOptional,
    // int64_t xxx,
    const aclTensor *metadata,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

``` cpp
aclnnStatus aclnnDenseLightningIndexerSoftmaxLseMetadata(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

<!-- TODO: 待补充参数说明、约束说明、Metadata输出布局、调用示例等 -->
