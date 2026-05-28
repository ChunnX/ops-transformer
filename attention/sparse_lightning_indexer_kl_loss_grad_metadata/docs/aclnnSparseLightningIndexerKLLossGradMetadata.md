# aclnnSparseLightningIndexerKLLossGradMetadata

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 | × |
| Atlas 训练系列产品 | × |

## 功能说明

`aclnnSparseLightningIndexerKLLossGradMetadata` 用于生成
`aclnnSparseLightningIndexerKLLossGrad` 的运行时 metadata。metadata 中记录了 batch 与 S1 合轴后的多核切分信息，
主 kernel 可直接读取该 tensor 完成分核调度，从而避免在主 kernel tiling 阶段访问 host array。

该接口通常在主 kernel 前调用，输出 `metadata` 作为
`aclnnSparseLightningIndexerKLLossGradGetWorkspaceSize` 的 `metadataOptional` 输入。

## 函数原型

每个 aclnn 算子分为两段式接口。必须先调用
`aclnnSparseLightningIndexerKLLossGradMetadataGetWorkspaceSize` 获取 workspace 大小和 executor，再调用
`aclnnSparseLightningIndexerKLLossGradMetadata` 执行计算。

```c++
aclnnStatus aclnnSparseLightningIndexerKLLossGradMetadataGetWorkspaceSize(
    const aclTensor *cuSeqLensQOptional,
    const aclTensor *cuSeqLensKOptional,
    const aclTensor *seqUsedQOptional,
    const aclTensor *seqUsedKOptional,
    const aclTensor *cmpResidualKOptional,
    int64_t batchSize,
    int64_t maxSeqLenQ,
    int64_t maxSeqLenK,
    int64_t numHeadsQ,
    int64_t numHeadsK,
    int64_t headDim,
    int64_t topk,
    char *layoutQ,
    char *layoutK,
    int64_t maskMode,
    int64_t cmpRatio,
    const aclTensor *metadata,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

```c++
aclnnStatus aclnnSparseLightningIndexerKLLossGradMetadata(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnSparseLightningIndexerKLLossGradMetadataGetWorkspaceSize 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | Shape / 取值 | 是否支持非连续 |
| --- | --- | --- | --- | --- | --- | :---: |
| cuSeqLensQOptional | 可选输入 | TND layout 下 q 的累积序列长度。 | INT32 | ND | `(B+1,)` | × |
| cuSeqLensKOptional | 可选输入 | TND layout 下 k 的累积序列长度。 | INT32 | ND | `(B+1,)` | × |
| seqUsedQOptional | 可选输入 | 预留字段，表示每个 batch 实际使用的 q 长度。 | INT32 | ND | `(B,)` | × |
| seqUsedKOptional | 可选输入 | 预留字段，表示每个 batch 实际使用的 k 长度。 | INT32 | ND | `(B,)` | × |
| cmpResidualKOptional | 可选输入 | `maskMode=3` 且 `cmpRatio!=1` 时的预留字段。 | INT32 | ND | `(B,)` | × |
| batchSize | 输入属性 | batch 数。TND layout 下可由 `cuSeqLensQOptional` 推导；上层 Python 接口传 `None` 时映射为 `0`。 | INT64 | - | 大于等于 0 | - |
| maxSeqLenQ | 输入属性 | q 的最大序列长度。BSND layout 下需要传入；TND layout 下由 `cuSeqLensQOptional` 参与计算。 | INT64 | - | 大于等于 0 | - |
| maxSeqLenK | 输入属性 | k 的最大序列长度。BSND layout 下需要传入；TND layout 下由 `cuSeqLensKOptional` 参与计算。 | INT64 | - | 大于等于 0 | - |
| numHeadsQ | 输入属性 | q 的 head 数，对应主 kernel 的 `N1`。 | INT64 | - | 大于 0 | - |
| numHeadsK | 输入属性 | k 的 head 数，对应主 kernel 的 `N2`。 | INT64 | - | 大于 0 | - |
| headDim | 输入属性 | q/k 的 head dim，对应主 kernel 的 `D`。 | INT64 | - | 大于 0 | - |
| topk | 输入属性 | top-k 大小，对应主 kernel 的 `K`。上层 Python 接口传 `None` 时映射为 `0`。 | INT64 | - | 大于 0 | - |
| layoutQ | 输入属性 | q 侧 layout。 | STRING | - | `BSND` / `TND` | - |
| layoutK | 输入属性 | k 侧 layout。 | STRING | - | `BSND` / `TND` | - |
| maskMode | 输入属性 | mask 模式。`0` 表示无 mask，`3` 表示右下角 causal mask。 | INT64 | - | `0` / `3` | - |
| cmpRatio | 输入属性 | 压缩比。 | INT64 | - | `[1,128]` | - |
| metadata | 输出 | 分核 metadata。 | INT32 | ND | `(x,)`，当前实现为 `(64,)` | - |
| workspaceSize | 输出 | 返回用户需要在 device 侧申请的 workspace 大小。当前实现返回 `0`。 | UINT64 | - | 标量 | - |
| executor | 输出 | 返回 op 执行器，包含算子计算流程。 | aclOpExecutor* | - | - | - |

## 约束与限制

- `cuSeqLensQOptional/cuSeqLensKOptional/seqUsedQOptional/seqUsedKOptional/cmpResidualKOptional/metadata`
  数据类型均为 `INT32`。
- `layoutQ` 和 `layoutK` 支持 `BSND` 和 `TND`。
- 当 `layoutQ=TND` 时，必须传入 `cuSeqLensQOptional` 和 `cuSeqLensKOptional`，且长度至少为 `batchSize + 1`。
- 当 `batchSize` 为 0 且 `layoutQ=TND` 时，AICPU kernel 会尝试通过 `cuSeqLensQOptional` 的长度推导 batch。
- 当 `layoutQ=BSND` 时，`batchSize/maxSeqLenQ/maxSeqLenK/topk` 必须为正数。
- `numHeadsQ/numHeadsK/headDim` 必须为正数，且 `numHeadsQ` 必须能被 `numHeadsK` 整除。
- `maskMode` 当前支持 `0` 和 `3`。
- `cmpRatio` 范围为 `[1,128]`。
- 输出 metadata 当前布局如下：

| 字段 | index | 说明 |
| --- | --- | --- |
| coreNum | 0 | 主 kernel 实际使用的 cube core 数。 |
| totalSize | 1 | 按 batch 与 S1 合轴后的任务总行数。 |
| splitFactorSize | 2 | 均分场景下的分核粒度。 |
| reserved | 3-7 | 预留字段，当前置 0。 |
| bS1Index | 8-32 | 每个 cube core 的起始 B/S1 合轴索引。 |

## 与主 kernel 的配合方式

典型调用顺序如下：

1. 调用 `aclnnSparseLightningIndexerKLLossGradMetadataGetWorkspaceSize` 和
   `aclnnSparseLightningIndexerKLLossGradMetadata` 生成 `metadata`。
2. 将 `metadata` 作为 `metadataOptional` 传给
   `aclnnSparseLightningIndexerKLLossGradGetWorkspaceSize`。
3. 调用 `aclnnSparseLightningIndexerKLLossGrad` 执行反向计算。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_lightning_indexer_kl_loss_grad_metadata.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

void PrintIntResult(const std::vector<int64_t> &shape, void *deviceAddr)
{
    auto elemCount = GetShapeSize(shape);
    std::vector<int32_t> resultData(elemCount, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(int32_t), deviceAddr,
                           resultData.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < std::min<int64_t>(elemCount, 8); ++i) {
        LOG_PRINT("metadata[%ld] is: %d\n", i, resultData[i]);
    }
}

int main()
{
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    const int64_t batchSize = 1;
    const int64_t maxSeqLenQ = 16;
    const int64_t maxSeqLenK = 4;
    const int64_t numHeadsQ = 8;
    const int64_t numHeadsK = 1;
    const int64_t headDim = 128;
    const int64_t topk = 512;
    const int64_t maskMode = 3;
    const int64_t cmpRatio = 4;

    std::vector<int64_t> cuSeqLensShape = {batchSize + 1};
    std::vector<int64_t> metadataShape = {64};
    std::vector<int32_t> cuSeqLensQHostData = {0, static_cast<int32_t>(maxSeqLenQ)};
    std::vector<int32_t> cuSeqLensKHostData = {0, static_cast<int32_t>(maxSeqLenK)};
    std::vector<int32_t> metadataHostData(GetShapeSize(metadataShape), 0);

    void *cuSeqLensQDeviceAddr = nullptr;
    void *cuSeqLensKDeviceAddr = nullptr;
    void *metadataDeviceAddr = nullptr;

    aclTensor *cuSeqLensQ = nullptr;
    aclTensor *cuSeqLensK = nullptr;
    aclTensor *metadata = nullptr;

    ret = CreateAclTensor(cuSeqLensQHostData, cuSeqLensShape, &cuSeqLensQDeviceAddr, aclDataType::ACL_INT32,
                          &cuSeqLensQ);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cuSeqLensKHostData, cuSeqLensShape, &cuSeqLensKDeviceAddr, aclDataType::ACL_INT32,
                          &cuSeqLensK);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadata);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnSparseLightningIndexerKLLossGradMetadataGetWorkspaceSize(
        cuSeqLensQ, cuSeqLensK, nullptr, nullptr, nullptr, batchSize, maxSeqLenQ, maxSeqLenK, numHeadsQ, numHeadsK,
        headDim, topk, const_cast<char *>("TND"), const_cast<char *>("TND"), maskMode, cmpRatio, metadata,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnSparseLightningIndexerKLLossGradMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnSparseLightningIndexerKLLossGradMetadata(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnSparseLightningIndexerKLLossGradMetadata failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    PrintIntResult(metadataShape, metadataDeviceAddr);

    aclDestroyTensor(cuSeqLensQ);
    aclDestroyTensor(cuSeqLensK);
    aclDestroyTensor(metadata);

    aclrtFree(cuSeqLensQDeviceAddr);
    aclrtFree(cuSeqLensKDeviceAddr);
    aclrtFree(metadataDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
