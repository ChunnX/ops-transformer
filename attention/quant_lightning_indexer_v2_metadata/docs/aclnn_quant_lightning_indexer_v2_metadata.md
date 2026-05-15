# aclnnQuantLightingIndexerV2Metadata

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT |      √     |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品     |    x     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    x     |
| Atlas 200I/500 A2 推理产品                      |    x     |
| Atlas 推理系列产品                              |    x     |
| Atlas 训练系列产品                              |    x     |

## 功能说明

- 算子功能：该算子为AICPU算子，是aclnnQuantLightningIndexerV2算子的前置算子。根据aclnnQuantLightningIndexerV2算子的输入信息，计算并输出负载均衡结果。输出结果可以作为aclnnQuantLightningIndexerV2算子的输入，减少aclnnQuantLightningIndexerV2算子的执行耗时。

  **该算子不建议单独使用，建议与aclnnQuantLightningIndexerV2算子配合使用，形成完整的工作流。**
    1. 接受aclnnQuantLightningIndexerV2算子输入数据shape信息，包含batchSize，qSeqlen，kSeqlen, mask。通过对输入分块并模拟计算耗时，均匀分配分块到可用核上，以降低aclnnQuantLightningIndexerV2算子的整体计算耗时，并提高硬件利用率。
    2. 分配结果输出后，后续作为输入供aclnnQuantLightningIndexerV2算子使用。
    3. 分配结果包含每个AIC核基本块的起始点和终止点，已经每个AIV核的FD任务信息。详细内容可以参考[调用示例](#调用示例)。

## 函数原型
每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用"aclnnLightingIndexerV2MetadataGetWorkspaceSize"获取workspace大小，在调用"aclnnLightingIndexerV2Metadata"执行计算

``` cpp
aclnnStatus aclnnQuantLightningIndexerV2MetadataGetWorkspaceSize(
    const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensKOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKOptional,
    const aclTensor *cmpResidualKOptional,
    int64_t numHeadsQ,
    int64_t numHeadsK,
    int64_t headDim,
    int64_t topk,
    int64_t qQuantMode,
    int64_t kQuantMode,
    int64_t batchSizeOptional,
    int64_t maxSeqlenQOptional,
    int64_t maxSeqlenKOptional,
    char *layoutQOptional,
    char *layoutKOptional,
    int64_t maskModeOptional,
    int64_t cmpRatioOptional,
    const aclTensor *metaData,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

``` cpp
aclnnStatus aclnnQuantLightningIndexerV2Metadata(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnQuantLightingIndexerV2MetadataGetWorkspaceSize
## 参数说明

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1600px"><colgroup>
  <col style="width: 150px">
  <col style="width: 100px">
  <col style="width: 350px">
  <col style="width: 150px">
  <col style="width: 70px">
  <col style="width: 70px">
  <col style="width: 190px">
  <col style="width: 80px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>cuSeqlensQOptional</td>
      <td>输入</td>
      <td>表示不同batch中query的有效Sequence Length</br>TND场景下必传，以该入参的数量作为Batch值</br>第一个值为额外值并固定为0。</td>
      <td>支持空Tensor</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(B+1,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cuSeqlensKOptional</td>
      <td>输入</td>
      <td>表示不同batch中key的有效Sequence Length</br>TND场景下必传，以该入参的数量作为Batch值</br>第一个值为额外值并固定为0</td>
      <td>支持空Tensor</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(B+1,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sequsedQOptional</td>
      <td>输入</td>
      <td>表示不同batch中query实际参与运算的Sequence Length</td>
      <td>支持空Tensor</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(B,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sequsedKOptional</td>
      <td>输入</td>
      <td>表示不同batch中key实际参与运算的Sequence Length</td>
      <td>支持空Tensor</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(B,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cmpResidualKOptional</td>
      <td>输入</td>
      <td>预留接口参数。表示不同batch中key的Sequence Length的余数</td>
      <td>支持空Tensor</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(B,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>numHeadsQ</td>
      <td>输入</td>
      <td>表示query的head个数</td>
      <td>支持非负数</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsK</td>
      <td>输入</td>
      <td>表示key的head个数</td>
      <td>支持非负数</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim</td>
      <td>输入</td>
      <td>表示toekn数</td>
      <td>支持非负数</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topk</td>
      <td>输入</td>
      <td>表示从query中筛选出的关键稀疏token的个数</td>
      <td>支持非负数</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>qQuantMode</td>
      <td>输入</td>
      <td>表示query的量化模式</td>
      <td>2: per-toekn-head</br> 3: group-scaling</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kQuantMode</td>
      <td>输入</td>
      <td>表示key的量化模式</td>
      <td>2: per-toekn-head</br> 3: group-scaling</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batchSizeOptional</td>
      <td>输入</td>
      <td>表示batch数量</td>
      <td>支持非负数</br>建议值为0</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenQOptional</td>
      <td>输入</td>
      <td>表示query的最长seqlen</td>
      <td>支持非负数</br>建议值为0</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenKOptional</td>
      <td>输入</td>
      <td>表示key的最长seqlen</td>
      <td>支持非负数</br>建议值为0</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQOptional</td>
      <td>输入</td>
      <td>表示query的排列格式</td>
      <td>支持 BSND/TND</br>建议值为BSND</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKOptional</td>
      <td>输入</td>
      <td>表示key的排列格式</td>
      <td>支持 BSND/TND/PA_BBND</br>建议值为BSND</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskModeOptional</td>
      <td>输入</td>
      <td>表示sparse模式</td>
      <td>0: No mask</br>3: Causal </br>建议值为0 </td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatioOptional</td>
      <td>输入</td>
      <td>预留接口参数,表示key的压缩率</td>
      <td>取值范围[1, 128]</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metaData</td>
      <td>输出</td>
      <td>表示负载均衡结果输出</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(1024)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## aclnnQuantLightingIndexerV2Metadata
- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 168px">
    <col style="width: 128px">
    <col style="width: 854px">
    </colgroup>
    <thead>
        <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
        </tr></thead>
    <tbody>
        <tr>
        <td>workspace</td>
        <td>输入</td>
        <td>在Device侧申请的workspace内存地址。</td>
        </tr>
        <tr>
        <td>workspaceSize</td>
        <td>输入</td>
        <td>在Device侧申请的workspace大小，由第一段接口aclnnQuantLightingIndexerV2MetadataGetWorkspaceSize获取。</td>
        </tr>
        <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
        </tr>
        <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的Stream。</td>
        </tr>
    </tbody>
    </table>

- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
  - aclnnQuantLightingIndexerV2Metadata默认为确定性实现，暂不支持非确定性实现，确定性计算配置也不会生效。

  - BSND场景
    - 必传batchSizeOptional，maxSeqlenQOptional，maxSeqlenKOptional参数，已获取shape信息
  - TND场景
    - 必传cuSeqlensQOptional, cuSeqlensKOptional参数，已获取正确shape信息

<details>
<summary><a id="Mask"></a>Mask</summary>
    &nbsp;&nbsp;<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
        <col style="width: 165px">
        <col style="width: 625px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>mask_mode</th>
                <th>含义</th>
                <th>备注</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>0</td>
            <td>无mask</td>
            <td>-</td>
        </tr>
        <tr>
            <td>3</td>
            <td>rightDownCausal模式的mask，对应以右顶点为划分的下三角场景，需要传入优化后的attenmask矩阵</td>
            <td>-</td>
        </tr>
        </tbody>
    </table>
</details>

<details>
<summary><a id="特殊约束"></a>特殊约束</summary>

  - Batch取值规则
    - 优先获取sequsedKOptional中的batch的shape信息
    - 如果未传入sequsedKOptional，优先获取cuSeqlensQOptional中的batch的shape信息
    - 如果未传入sequsedKOptional，且layoutQOptional为TND，则必获取cuSeqlensQOptional中的batch的shape信息
    - 除上所述，使用batchSizeOptional
  - Seqlen取值规则
    - 优先获取sequsedKOptional中的seqlen的shape信息
    - 如果未传入sequsedKOptional，且layoutKOptional为TND，则必获取cuSeqlensQOptional中的batch的shape信息
    - 除上所述，使用maxSeqlenQOptional
    - key与query的获取规则一致
  - cmpResidualKOptional启用规则
    - 预留参数，暂无效果，只做约束
    - 如果cmpRatioOptional不为1，且mask为3，则必须传入cmpResidualKOptional
  - layout约束
    - 当layoutKOptional为PA_BBND时，layoutQOptional可以任意取值
    - 除上所述，layoutQOptional必须与layoutKOptional保持一致

</details>

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

``` cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <limits>
#include <functional>
#include <utility>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_lightning_indexer_v2_metadata.h"

#define CHECK_LOG_RET(cond, ret_val, fmt, ...)      \
    do {                                            \
        if (!(cond)) {                              \
            printf(fmt "\n", ##__VA_ARGS__);        \
            return (ret_val);                       \
        }                                           \
    } while (0)

// 参考 quant_lightning_indexer_v2_metadata.h
constexpr uint32_t AIC_CORE_NUM = 36;
constexpr uint32_t AIV_CORE_NUM = 72;
constexpr uint32_t QLI_V2_METADATA_SIZE = 8;
constexpr uint32_t QLD_V2_METADATA_SIZE = 8;

// QLI Metadata Index Definitions
constexpr uint32_t QLI_V2_CORE_ENABLE_INDEX = 0;
constexpr uint32_t QLI_V2_BN2_START_INDEX = 1;
constexpr uint32_t QLI_V2_M_START_INDEX = 2;
constexpr uint32_t QLI_V2_S2_START_INDEX = 3;
constexpr uint32_t QLI_V2_BN2_END_INDEX = 4;
constexpr uint32_t QLI_V2_M_END_INDEX = 5;
constexpr uint32_t QLI_V2_S2_END_INDEX = 6;
constexpr uint32_t QLI_V2_FIRST_QLD_V2_DATA_WORKSPACE_IDX_INDEX = 7;

// QLD Metadata Index Definitions
constexpr uint32_t QLD_V2_CORE_ENABLE_INDEX = 0;
constexpr uint32_t QLD_V2_BN2_IDX_INDEX = 1;
constexpr uint32_t QLD_V2_M_IDX_INDEX = 2;
constexpr uint32_t QLD_V2_WORKSPACE_IDX_INDEX = 3;
constexpr uint32_t QLD_V2_WORKSPACE_NUM_INDEX = 4;
constexpr uint32_t QLD_V2_M_START_INDEX = 5;
constexpr uint32_t QLD_V2_M_NUM_INDEX = 6;

struct QliV2MetaData {
    uint32_t faData[AIC_CORE_NUM][QLI_V2_METADATA_SIZE];
    uint32_t fdData[AIV_CORE_NUM][QLD_V2_METADATA_SIZE];
};

struct ScopeGuard
{
    explicit ScopeGuard(std::function<void()> onExitScope) : m_exitFunc(std::move(onExitScope)),
        m_isDismissed(false) {}
    // 禁止拷贝
    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;

    ~ScopeGuard()
    {
        if (!m_isDismissed) {
            m_exitFunc();
        }
    }

    void Dismiss()
    {
        m_isDismissed = true;
    }

    std::function<void()> m_exitFunc;
    bool m_isDismissed;
};

struct Tensor {
    void *hostAddr { nullptr };
    void *deviceAddr { nullptr };
    aclTensor *data { nullptr };
};

struct ArgScenario {
    bool hasCuSeq { false };
    bool hasSeqused { false };
};

struct ArgContext {
    // required input
    int64_t numHeadsQ { 0 };
    int64_t numHeadsK { 0 };
    int64_t headDim { 0 };
    int64_t topk { 0 };
    int64_t qQuantMode { 2 };
    int64_t kQuantMode { 2 };
    // optional input
    Tensor cuSeqlensQOptional {};
    Tensor cuSeqlensKOptional {};
    Tensor sequsedQOptional {};
    Tensor sequsedKOptional {};
    Tensor cmpResidualKOptional {};
    int64_t batchSizeOptional { 0 };
    int64_t maxSeqlenQOptional { 0 };
    int64_t maxSeqlenKOptional { 0 };
    char *layoutQOptional { nullptr };
    char *layoutKOptional { nullptr };
    int64_t maskModeOptional { 0 };
    int64_t cmpRatioOptional { 0 };
    // output
    Tensor metaData {};
};

int64_t GetShapeSize(const std::vector<int64_t>& shape) 
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

aclnnStatus Init(int32_t deviceId, aclrtStream* stream) 
{
    // 固定写法，初始化
    auto ret = aclInit(nullptr);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclInit failed. ERROR: %d", ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSetDevice failed. ERROR: %d", ret);
    ret = aclrtCreateStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtCreateStream failed. ERROR: %d", ret);
    return ACL_SUCCESS;
}

void Finalize(int32_t deviceId, aclrtStream stream) 
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

aclnnStatus CreateTensor(aclDataType dataType, const std::vector<int64_t> &shape, Tensor &tensor)
{
    auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
    // 调用aclrtMallocHost申请host侧内存
    auto ret = aclrtMallocHost(&(tensor.hostAddr), size);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMallocHost failed. ERROR: %d", ret);
    memset(tensor.hostAddr, 0, size);
    // 调用aclrtMalloc申请device侧内存
    ret = aclrtMalloc(&(tensor.deviceAddr), size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMalloc failed. ERROR: %d", ret);
    // 调用aclCreateTensor接口创建aclTensor
    tensor.data = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), tensor.deviceAddr);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(tensor.deviceAddr, size, tensor.hostAddr, size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d", ret);
    return ACL_SUCCESS;
}

void DestroyTensor(Tensor &tensor)
{
    if (tensor.data != nullptr) {
        aclDestroyTensor(tensor.data);
        tensor.data = nullptr;
    }
    if (tensor.deviceAddr != nullptr) {
        aclrtFree(tensor.deviceAddr);
        tensor.deviceAddr = nullptr;
    }
    if (tensor.hostAddr != nullptr) {
        aclrtFreeHost(tensor.hostAddr);
        tensor.hostAddr = nullptr;
    }
}

void DestroyArgs(ArgContext &context)
{
    DestroyTensor(context.metaData);
    DestroyTensor(context.cuSeqlensQOptional);
    DestroyTensor(context.cuSeqlensKOptional);
    DestroyTensor(context.sequsedQOptional);
    DestroyTensor(context.sequsedKOptional);
    DestroyTensor(context.cmpResidualKOptional);

    if (context.layoutQOptional != nullptr) {
        free(context.layoutQOptional);
        context.layoutQOptional = nullptr;
    }
    if (context.layoutKOptional != nullptr) {
        free(context.layoutKOptional);
        context.layoutKOptional = nullptr;
    }
}

aclnnStatus CreateArgs(const ArgScenario &scenario, ArgContext &context)
{
    ScopeGuard argsGuard([&] { DestroyArgs(context); });
    aclnnStatus ret;
    int64_t batchSize = 4;

    context.numHeadsQ = 1;
    context.numHeadsK = 1;
    context.headDim = 128;
    context.topk = 0;
    context.qQuantMode = 2; // 2: per-token-head / 3: group-scaling
    context.kQuantMode = 2; // 2: per-token-head / 3: group-scaling
    ret = CreateTensor(aclDataType::ACL_INT32, { 1024 }, context.metaData);     // 1024: Fix size
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create meta failed. Error: %d", ret);

    context.maskModeOptional = 0;                   // 0: no mask, 3: causal
    context.cmpRatioOptional = 1;                   // [1, 128], 1: no compress
    context.layoutQOptional = (char *)malloc(sizeof(char) * 16);
    context.layoutKOptional = (char *)malloc(sizeof(char) * 16);
    strcpy(context.layoutQOptional, "BSND");                // BSND,TND
    strcpy(context.layoutKOptional, "BSND");                // BSND,TND,PA_BBND

    if (!scenario.hasCuSeq && !scenario.hasSeqused) {
        context.batchSizeOptional = batchSize;
        context.maxSeqlenKOptional = 1024;
        context.maxSeqlenQOptional = 1024;
        return ACL_SUCCESS;
    }

    if (scenario.hasCuSeq) {
        // (B+1,), first element is always 0
        ret = CreateTensor(aclDataType::ACL_INT32, { batchSize + 1 }, context.cuSeqlensQOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensQOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, { batchSize + 1 }, context.cuSeqlensKOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensKOptional failed. Error: %d", ret);
    }

    if (scenario.hasSeqused) {
        // (B,)
        ret = CreateTensor(aclDataType::ACL_INT32, { batchSize }, context.sequsedQOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedQOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, { batchSize }, context.sequsedKOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedKOptional failed. Error: %d", ret);
    }

    argsGuard.Dismiss();
    return ACL_SUCCESS;
}

int main() {
    // 1. （固定写法）device/stream初始化，参考对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Init acl failed. ERROR: %d", ret);
    ScopeGuard sysGuard([&] { Finalize(deviceId, stream); });

    // 2. 构造输入与输出，需要根据API的接口定义构造
    ArgScenario scenario {};
    scenario.hasCuSeq = true;
    scenario.hasSeqused = true;
    ArgContext context {};
    ret = CreateArgs(scenario, context);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create input arguments failed. ERROR: %d", ret);
    ScopeGuard argsGuard([&] { DestroyArgs(context); });

    // 3. 调用CANN算子库API，需要修改为具体的API
    // 调用aclnnLightningIndexerV2Metadata第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
    ret = aclnnQuantLightningIndexerV2MetadataGetWorkspaceSize(
        context.cuSeqlensQOptional.data, context.cuSeqlensKOptional.data, context.sequsedQOptional.data,
        context.sequsedKOptional.data, context.cmpResidualKOptional.data,
        context.numHeadsQ, context.numHeadsK, context.headDim, context.topk, context.qQuantMode, context.kQuantMode,
        context.batchSizeOptional, context.maxSeqlenQOptional, context.maxSeqlenKOptional, context.layoutQOptional,
        context.layoutKOptional, context.maskModeOptional, context.cmpRatioOptional,
        context.metaData.data, &workspaceSize, &executor);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret,
        "aclnnQuantLightningIndexerV2MetadataGetWorkspaceSize failed. ERROR: %d\n", ret);

    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "allocate workspace failed. ERROR: %d\n", ret);
    }
    ScopeGuard workspaceGuard([&] {
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
            workspaceAddr = nullptr;
        }
    });
    
    // 调用aclnnLightningIndexerV2Metadata第二段接口
    ret = aclnnQuantLightningIndexerV2Metadata(workspaceAddr, workspaceSize, executor, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclnnQuantLightningIndexerV2Metadata failed. ERROR: %d\n", ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSynchronizeStream failed. ERROR: %d\n", ret);

    // 5. 打印输出
    QliV2MetaData result {};
    ret = aclrtMemcpy(&result, sizeof(result), context.metaData.deviceAddr, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d\n", ret);

    for (uint32_t i = 0; i < AIC_CORE_NUM; ++i) {
        printf("AIC Core%u\n", i);
        printf("    Core Enable : %u\n", result.faData[i][QLI_V2_CORE_ENABLE_INDEX]);
        printf("    Start BN2   : %u\n", result.faData[i][QLI_V2_BN2_START_INDEX]);
        printf("    Start M     : %u\n", result.faData[i][QLI_V2_M_START_INDEX]);
        printf("    Start S2    : %u\n", result.faData[i][QLI_V2_S2_START_INDEX]);
        printf("    End BN2     : %u\n", result.faData[i][QLI_V2_BN2_END_INDEX]);
        printf("    End M       : %u\n", result.faData[i][QLI_V2_M_END_INDEX]);
        printf("    End S2      : %u\n", result.faData[i][QLI_V2_S2_END_INDEX]);
        printf("    First Worksapce Index : %u\n", result.faData[i][QLI_V2_FIRST_QLD_V2_DATA_WORKSPACE_IDX_INDEX]);
    }
    for (uint32_t i = 0; i < AIV_CORE_NUM; ++i) {
        printf("AIV Core%u\n", i);
        printf("    Core Enable             : %u\n", result.fdData[i][QLD_V2_CORE_ENABLE_INDEX]);
        printf("    FD Task BN2 Idx         : %u\n", result.fdData[i][QLD_V2_BN2_IDX_INDEX]);
        printf("    FD Task M Idx           : %u\n", result.fdData[i][QLD_V2_M_IDX_INDEX]);
        printf("    FD Task S2 Idx          : %u\n", result.fdData[i][QLD_V2_WORKSPACE_IDX_INDEX]);
        printf("    FD Task Workspace Num   : %u\n", result.fdData[i][QLD_V2_WORKSPACE_NUM_INDEX]);
        printf("    FD Subtask M Start      : %u\n", result.fdData[i][QLD_V2_M_START_INDEX]);
        printf("    FD Subtask M Num        : %u\n", result.fdData[i][QLD_V2_M_NUM_INDEX]);
    }

    return 0;
}
```

## 调用说明

| 调用方式  | 样例代码                                                                | 说明                                                                                          |
| ----------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| aclnn接口 | [test_aclnn_quant_lightning_indexer_v2_metadata.cpp](./examples/test_aclnn_quant_lightning_indexer_v2_metadata.cpp) | 通过[`aclnnQuantLightningIndexerV2Metadata`](./docs/aclnnQuantLightningIndexerV2Metadata.md)接口方式调用 |
