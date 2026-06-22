# aclnnMixedQuantFlashAttnMetadata

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT |      √     |
| Atlas A3训练系列产品/Atlas A3推理系列产品     |    x     |
| Atlas A2训练系列产品/Atlas A2推理系列产品 |    x     |
| Atlas 200I/500 A2推理产品                      |    x     |
| Atlas推理系列产品                              |    x     |
| Atlas训练系列产品                              |    x     |

## 功能说明

- 接口功能：该接口为AI CPU算子接口，是aclnnMixedQuantFlashAttn算子的前置算子接口。根据aclnnMixedQuantFlashAttn算子接口的输入信息，计算并输出负载均衡结果。输出结果可以作为aclnnMixedQuantFlashAttn算子接口的输入，减少aclnnMixedQuantFlashAttn算子接口的执行耗时。

  **该算子不建议单独使用，建议与aclnnMixedQuantFlashAttn算子配合使用，形成完整的工作流。**
    1. 接受aclnnMixedQuantFlashAttn算子接口输入数据shape信息，包含batchSize、maxSeqlenQ、maxSeqlenKv、numHeadsQ、numHeadsKv、headDim、quantMode、maskMode等。通过对输入分块并模拟计算耗时，均匀分配分块到可用核上，以降低aclnnMixedQuantFlashAttn算子的整体计算耗时，并提高硬件利用率。
    2. 分配结果输出后，后续作为输入供aclnnMixedQuantFlashAttn算子使用。
    3. 分配结果包含每个AIC核基本块的起始点和终止点，以及每个AIV核的FD任务信息。详细内容可以参考[调用示例](#调用示例)。

## 函数原型
每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用"aclnnMixedQuantFlashAttnMetadataGetWorkspaceSize"获取workspace大小，再调用"aclnnMixedQuantFlashAttnMetadata"执行计算

``` cpp
aclnnStatus aclnnMixedQuantFlashAttnMetadataGetWorkspaceSize(
    const aclTensor   *cuSeqlensQOptional,
    const aclTensor   *cuSeqlensKvOptional,
    const aclTensor   *sequsedQOptional,
    const aclTensor   *sequsedKvOptional,
    int64_t            batchSize,
    int64_t            maxSeqlenQ,
    int64_t            maxSeqlenKv,
    int64_t            numHeadsQ,
    int64_t            numHeadsKv,
    int64_t            headDim,
    int64_t            quantMode,
    int64_t            maskMode,
    int64_t            winLeft,
    int64_t            winRight,
    const char         *layoutQ,
    const char         *layoutKv,
    const char         *layoutOut,
    const aclTensor   *metaData,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor)
```

``` cpp
aclnnStatus aclnnMixedQuantFlashAttnMetadata(
    void              *workspace,
    uint64_t           workspaceSize,
    aclOpExecutor     *executor,
    aclrtStream        stream)
```

## aclnnMixedQuantFlashAttnMetadataGetWorkspaceSize
- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1600px"><colgroup>
  <col style="width: 150px">
  <col style="width: 100px">
  <col style="width: 350px">
  <col style="width: 200px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 190px">
  <col style="width: 100px">
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
      <td>表示不同Batch中Query的有效Sequence Length</br>TND场景下必传，以该入参的数量作为Batch值</br>第一个值为额外值并固定为0。</td>
      <td>支持空Tensor</br>layoutQ非TND时不传</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(B+1，)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensKvOptional</td>
      <td>输入</td>
      <td>表示不同Batch中Key/Value的有效Sequence Length</br>TND场景下必传，以该入参的数量作为Batch值</br>第一个值为额外值并固定为0。</td>
      <td>支持空Tensor</br>layoutKv非TND时不传</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(B+1，)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedQOptional</td>
      <td>输入</td>
      <td>表示不同Batch中Query实际参与运算的Sequence Length。</td>
      <td>支持空Tensor</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(B，)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedKvOptional</td>
      <td>输入</td>
      <td>表示不同Batch中Key/Value实际参与运算的Sequence Length。</td>
      <td>支持空Tensor</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(B，)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>batchSize</td>
      <td>输入</td>
      <td>表示Batch数量。</td>
      <td>支持非负数</br>建议值为0</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenQ</td>
      <td>输入</td>
      <td>表示Query的最长Sequence Length。</td>
      <td>取值范围[-1，∞)</br>-1表示不传</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenKv</td>
      <td>输入</td>
      <td>表示Key/Value的最长Sequence Length。</td>
      <td>取值范围[-1，∞)</br>-1表示不传</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsQ</td>
      <td>输入</td>
      <td>表示Query的head个数。</td>
      <td>支持非负数</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsKv</td>
      <td>输入</td>
      <td>表示Key/Value的head个数。</td>
      <td>支持非负数</br>为0时默认等于numHeadsQ</br>numHeadsQ必须能被numHeadsKv整除</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim</td>
      <td>输入</td>
      <td>表示每个head的维度大小。</td>
      <td>支持非负数</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode</td>
      <td>输入</td>
      <td>表示KV的量化模式。</td>
      <td>1：INT4量化（4-bit）</br>2：INT8量化（8-bit）</br>3：INT8量化（8-bit）</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskMode</td>
      <td>输入</td>
      <td>表示sparse模式。</td>
      <td>0：No mask</br>3：rightDownCausal模式的mask，对应以右顶点为划分的下三角场景</br>4：其它mask模式</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>winLeft</td>
      <td>输入</td>
      <td>表示滑动窗口左侧token数。</td>
      <td>取值范围[-1，∞)</br>-1表示不限制</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>winRight</td>
      <td>输入</td>
      <td>表示滑动窗口右侧token数。</td>
      <td>取值范围[-1，∞)</br>-1表示不限制</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQ</td>
      <td>输入</td>
      <td>表示Query的排列格式。</td>
      <td>支持BSND、TND、BNSD</td>
      <td>char*</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKv</td>
      <td>输入</td>
      <td>表示Key/Value的排列格式。</td>
      <td>支持BSND、TND、BNSD、PA_BNBD、PA_BBND、PA_NZ</td>
      <td>char*</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutOut</td>
      <td>输入</td>
      <td>表示Output的排列格式。</td>
      <td>支持BSND、TND、BNSD</td>
      <td>char*</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metaData</td>
      <td>输出</td>
      <td>表示负载均衡结果输出。</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape需按4K对齐</td>
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

## aclnnMixedQuantFlashAttnMetadata
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
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>workspace</td>
            <td>输入</td>
            <td>在Device侧申请的workspace内存地址</td>
        </tr>
        <tr>
            <td>workspaceSize</td>
            <td>输入</td>
            <td>在Device侧申请的workspace大小，由第一段接口aclnnMixedQuantFlashAttnMetadataGetWorkspaceSize获取</td>
        </tr>
        <tr>
            <td>executor</td>
            <td>输入</td>
            <td>op执行器，包含了算子计算流程</td>
        </tr>
        <tr>
            <td>stream</td>
            <td>输入</td>
            <td>指定执行任务的Stream</td>
        </tr>
    </tbody>
    </table>

- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
  - aclnnMixedQuantFlashAttnMetadata默认确定性实现

  - quantMode取值约束
    - quantMode仅支持1（INT4量化）、2（INT8量化）、3（INT8量化）
  - maskMode取值约束
    - maskMode仅支持0（No mask）、3（rightDownCausal）、4
    - maskMode为4时，FD任务关闭
  - winLeft取值约束
    - winLeft必须大于等于-1，-1表示不限制
  - winRight取值约束
    - winRight必须大于等于-1，-1表示不限制
  - Batch取值规则
    - 优先获取sequsedQOptional中的Batch信息
    - 如果未传入sequsedQOptional，且layoutQ为TND，则必获取cuSeqlensQOptional中的Batch信息
    - 除上所述，使用batchSize
  - Sequence Length取值规则
    - 优先获取sequsedQOptional中的Sequence Length信息
    - 如果未传入sequsedQOptional，且layoutQ为TND，则必获取cuSeqlensQOptional中的Sequence Length信息
    - 除上所述，使用maxSeqlenQ
    - Key/Value与Query的获取规则一致
  - layout约束
    - layoutQ仅支持BSND、TND、BNSD
    - layoutKv仅支持BSND、TND、BNSD、PA_BNBD、PA_BBND、PA_NZ
    - layoutOut仅支持BSND、TND、BNSD
    - layoutQ为TND时，cuSeqlensQOptional、maxSeqlenQ、sequsedQOptional至少传一个
    - layoutQ非TND时，不应传入cuSeqlensQOptional，maxSeqlenQ或sequsedQOptional必须至少传一个
    - layoutKv为TND时，cuSeqlensKvOptional、maxSeqlenKv、sequsedKvOptional至少传一个
    - layoutKv非TND时，不应传入cuSeqlensKvOptional，maxSeqlenKv或sequsedKvOptional必须至少传一个
  - numHeadsKv约束
    - numHeadsKv为0时，默认等于numHeadsQ
    - numHeadsKv非0时，numHeadsQ必须能被numHeadsKv整除
  - metaData输出约束
    - metaData为INT32类型，1维Tensor
    - metaData的shape需按4K对齐（4096字节对齐）
    - 建议metadata大小计算公式：((36 + 72) × batchSize × numHeadsKv + 1) × 16，再按4096向上取整对齐

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

``` cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_mixed_quant_flash_attn_metadata.h"
#include "../op_kernel_aicpu/mixed_quant_flash_attn_metadata.h"

#define CHECK_LOG_RET(cond, ret_val, fmt, ...)      \
    do {                                            \
        if (!(cond)) {                              \
            printf(fmt "\n", ##__VA_ARGS__);        \
            return (ret_val);                       \
        }                                           \
    } while (0)

constexpr uint32_t AIC_CORE_NUM = 36;
constexpr uint32_t AIV_CORE_NUM = 72;

constexpr uint32_t HEAD_METADATA_SIZE = 16;
constexpr uint32_t FA_METADATA_SIZE = 16;
constexpr uint32_t FD_METADATA_SIZE = 16;

constexpr uint32_t HEAD_SECTION_NUM_INDEX = 0;
constexpr uint32_t HEAD_IS_FD_INDEX = 1;
constexpr uint32_t HEAD_M_BASE_SIZE_INDEX = 2;
constexpr uint32_t HEAD_S2_BASE_SIZE_INDEX = 3;

constexpr uint32_t FA_BN2_START_INDEX = 0;
constexpr uint32_t FA_M_START_INDEX = 1;
constexpr uint32_t FA_S2_START_INDEX = 2;
constexpr uint32_t FA_BN2_END_INDEX = 3;
constexpr uint32_t FA_M_END_INDEX = 4;
constexpr uint32_t FA_S2_END_INDEX = 5;
constexpr uint32_t FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 6;

constexpr uint32_t FD_BN2_IDX_INDEX = 0;
constexpr uint32_t FD_M_IDX_INDEX = 1;
constexpr uint32_t FD_WORKSPACE_IDX_INDEX = 2;
constexpr uint32_t FD_WORKSPACE_NUM_INDEX = 3;
constexpr uint32_t FD_M_START_INDEX = 4;
constexpr uint32_t FD_M_NUM_INDEX = 5;

struct ScopeGuard
{
    explicit ScopeGuard(std::function<void()> onExitScope) : m_exitFunc(std::move(onExitScope)),
        m_isDismissed(false) {}
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

struct ArgContext {
    int64_t numHeadsQ { 32 };
    int64_t numHeadsKv { 32 };
    int64_t headDim { 128 };
    int64_t quantMode { 1 };
    int64_t maskMode { 3 };
    int64_t winLeft { -1 };
    int64_t winRight { -1 };
    int64_t batchSize { 4 };
    int64_t maxSeqlenQ { 1 };
    int64_t maxSeqlenKv { 8192 };
    char *layoutQ { nullptr };
    char *layoutKv { nullptr };
    char *layoutOut { nullptr };
    Tensor cuSeqlensQOptional {};
    Tensor cuSeqlensKvOptional {};
    Tensor sequsedQOptional {};
    Tensor sequsedKvOptional {};
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
    auto ret = aclrtMallocHost(&(tensor.hostAddr), size);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMallocHost failed. ERROR: %d", ret);
    memset(tensor.hostAddr, 0, size);
    ret = aclrtMalloc(&(tensor.deviceAddr), size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMalloc failed. ERROR: %d", ret);
    tensor.data = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), tensor.deviceAddr);
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
    DestroyTensor(context.cuSeqlensKvOptional);
    DestroyTensor(context.sequsedQOptional);
    DestroyTensor(context.sequsedKvOptional);
    if (context.layoutQ != nullptr) {
        free(context.layoutQ);
        context.layoutQ = nullptr;
    }
    if (context.layoutKv != nullptr) {
        free(context.layoutKv);
        context.layoutKv = nullptr;
    }
    if (context.layoutOut != nullptr) {
        free(context.layoutOut);
        context.layoutOut = nullptr;
    }
}

aclnnStatus CreateArgs(ArgContext &context)
{
    aclnnStatus ret;
    int64_t metadataSize = ((36 + 72) * context.batchSize * context.numHeadsKv + 1) * 16;
    int64_t alignedSize = ((metadataSize + 4095) / 4096) * 4096;

    ret = CreateTensor(aclDataType::ACL_INT32, { alignedSize }, context.metaData);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create metaData failed. Error: %d", ret);

    context.layoutQ = (char *)malloc(sizeof(char) * 16);
    context.layoutKv = (char *)malloc(sizeof(char) * 16);
    context.layoutOut = (char *)malloc(sizeof(char) * 16);
    strcpy(context.layoutQ, "BSND");
    strcpy(context.layoutKv, "BSND");
    strcpy(context.layoutOut, "BSND");

    return ACL_SUCCESS;
}

int main() {
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Init acl failed. ERROR: %d", ret);
    ScopeGuard sysGuard([&] { Finalize(deviceId, stream); });

    ArgContext context {};
    ret = CreateArgs(context);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create input arguments failed. ERROR: %d", ret);
    ScopeGuard argsGuard([&] { DestroyArgs(context); });

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
    ret = aclnnMixedQuantFlashAttnMetadataGetWorkspaceSize(
        context.cuSeqlensQOptional.data, context.cuSeqlensKvOptional.data,
        context.sequsedQOptional.data, context.sequsedKvOptional.data,
        context.batchSize, context.maxSeqlenQ, context.maxSeqlenKv,
        context.numHeadsQ, context.numHeadsKv, context.headDim,
        context.quantMode, context.maskMode, context.winLeft, context.winRight,
        context.layoutQ, context.layoutKv, context.layoutOut,
        context.metaData.data, &workspaceSize, &executor);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclnnMixedQuantFlashAttnMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);

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
    
    ret = aclnnMixedQuantFlashAttnMetadata(workspaceAddr, workspaceSize, executor, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclnnMixedQuantFlashAttnMetadata failed. ERROR: %d\n", ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSynchronizeStream failed. ERROR: %d\n", ret);

    int32_t sectionNum = 0;
    ret = aclrtMemcpy(&sectionNum, sizeof(sectionNum), context.metaData.deviceAddr, sizeof(sectionNum), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy sectionNum failed. ERROR: %d\n", ret);

    std::vector<int32_t> metadataHostData(GetShapeSize({ ((36 + 72) * context.batchSize * context.numHeadsKv + 1) * 16 }), 0);
    ret = aclrtMemcpy(metadataHostData.data(), metadataHostData.size() * sizeof(int32_t),
        context.metaData.deviceAddr, metadataHostData.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy metadata failed. ERROR: %d\n", ret);

    optiling::detail::FaMetaData faMetadata(metadataHostData.data(), sectionNum);

    printf("sectionNum : %u\n", faMetadata.GetHeadMedata(optiling::HEAD_SECTION_NUM_INDEX));
    printf("isFd       : %u\n", faMetadata.GetHeadMedata(optiling::HEAD_IS_FD_INDEX));
    printf("mBaseSize  : %u\n", faMetadata.GetHeadMedata(optiling::HEAD_M_BASE_SIZE_INDEX));
    printf("s2BaseSize : %u\n", faMetadata.GetHeadMedata(optiling::HEAD_S2_BASE_SIZE_INDEX));

    for (uint32_t sectionId = 0; sectionId < sectionNum; ++sectionId) {
        printf("Section %u:\n", sectionId);
        for (uint32_t i = 0; i < AIC_CORE_NUM; ++i) {
            printf("  AIC Core%u:\n", i);
            printf("    Start BN2          : %u\n", faMetadata.GetFaMetadata(sectionId, i, optiling::FA_BN2_START_INDEX));
            printf("    Start M            : %u\n", faMetadata.GetFaMetadata(sectionId, i, optiling::FA_M_START_INDEX));
            printf("    Start S2           : %u\n", faMetadata.GetFaMetadata(sectionId, i, optiling::FA_S2_START_INDEX));
            printf("    End BN2            : %u\n", faMetadata.GetFaMetadata(sectionId, i, optiling::FA_BN2_END_INDEX));
            printf("    End M              : %u\n", faMetadata.GetFaMetadata(sectionId, i, optiling::FA_M_END_INDEX));
            printf("    End S2             : %u\n", faMetadata.GetFaMetadata(sectionId, i, optiling::FA_S2_END_INDEX));
            printf("    First FD Ws Index  : %u\n", faMetadata.GetFaMetadata(sectionId, i, optiling::FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX));
        }
        for (uint32_t i = 0; i < AIV_CORE_NUM; ++i) {
            printf("  AIV Core%u:\n", i);
            printf("    FD Task BN2 Idx    : %u\n", faMetadata.GetFdMetadata(sectionId, i, optiling::FD_BN2_IDX_INDEX));
            printf("    FD Task M Idx      : %u\n", faMetadata.GetFdMetadata(sectionId, i, optiling::FD_M_IDX_INDEX));
            printf("    FD Task Ws Idx     : %u\n", faMetadata.GetFdMetadata(sectionId, i, optiling::FD_WORKSPACE_IDX_INDEX));
            printf("    FD Task Ws Num     : %u\n", faMetadata.GetFdMetadata(sectionId, i, optiling::FD_WORKSPACE_NUM_INDEX));
            printf("    FD Subtask M Start : %u\n", faMetadata.GetFdMetadata(sectionId, i, optiling::FD_M_START_INDEX));
            printf("    FD Subtask M Num   : %u\n", faMetadata.GetFdMetadata(sectionId, i, optiling::FD_M_NUM_INDEX));
        }
    }

    return 0;
}
```

