# aclnnSparseFlashMlaMetadata

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

- 接口功能：该接口为AI CPU算子接口，是aclnnSparseFlashMla算子的前置算子接口。根据aclnnSparseFlashMla算子接口的输入信息，计算并输出负载均衡结果。输出结果可以作为aclnnSparseFlashMla算子接口的输入，减少aclnnSparseFlashMla算子接口的执行耗时。

  **该算子不建议单独使用，建议与aclnnSparseFlashMla算子配合使用，形成完整的工作流。**
    1. 接受aclnnSparseFlashMla算子接口输入数据shape信息，包含batchSize、qSeqlen、kSeqlen、mask。通过对输入分块并模拟计算耗时，均匀分配分块到可用核上，以降低aclnnSparseFlashMla算子的整体计算耗时，并提高硬件利用率。
    2. 分配结果输出后，后续作为输入供aclnnSparseFlashMla算子使用。
    3. 分配结果包含每个AIC核基本块的起始点和终止点，已经每个AIV核的FD任务信息。详细内容可以参考[调用示例](#调用示例)。

## 函数原型
每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用"aclnnSparseFlashMlaMetadataGetWorkspaceSize"获取workspace大小，在调用"aclnnSparseFlashMlaMetadata"执行计算

``` cpp
aclnnStatus aclnnSparseFlashMlaMetadataGetWorkspaceSize(
    const aclTensor   *cuSeqlensQOptional,
    const aclTensor   *cuSeqlensOriKvOptional,
    const aclTensor   *cuSeqlensCmpKvOptional,
    const aclTensor   *sequsedQOptional,
    const aclTensor   *sequsedOriKvOptional,
    const aclTensor   *sequsedCmpKvOptional,
    const aclTensor   *cmpResidualKvOptional,
    const aclTensor   *oriTopkLengthOptional,
    const aclTensor   *cmpTopkLengthOptional,
    int64_t            numHeadsQ,
    int64_t            numHeadsKv,
    int64_t            headDim,
    int64_t            batchSize,
    int64_t            maxSeqlenQ,
    int64_t            maxSeqlenOriKv,
    int64_t            maxSeqlenCmpKv,
    int64_t            oriTopk,
    int64_t            cmpTopk,
    int64_t            cmpRatio,
    int64_t            oriMaskMode,
    int64_t            cmpMaskMode,
    int64_t            oriWinLeft,
    int64_t            oriWinRight,
    char              *layoutQOptional,
    char              *layoutKvOptional,
    bool               hasOriKv,
    bool               hasCmpKv,
    const aclTensor   *metaData,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor)
```

``` cpp
aclnnStatus aclnnSparseFlashMlaMetadata(
    void              *workspace,
    uint64_t           workspaceSize,
    aclOpExecutor     *executor,
    aclrtStream        stream)
```

## aclnnSparseFlashMlaMetadataGetWorkspaceSize
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
      <td>表示不同Batch中Query的有效Sequence Length</br>layoutQOptional为TND场景下必传，以该入参的数量作为Batch值</br>第一个值为额外值并固定为0。</td>
      <td>支持空Tensor</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(B+1，)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensOriKvOptional</td>
      <td>输入</td>
      <td>表示不同Batch中ori_Kv的有效Sequence Length</br>layoutKvOptional为TND场景下必传，以该入参的数量作为Batch值</br>第一个值为额外值并固定为0。</td>
      <td>支持空Tensor</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(B+1，)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensCmpKvOptional</td>
      <td>输入</td>
      <td>表示不同Batch中cmp_Kv的有效Sequence Length</br>layoutKvOptional为TND场景下必传，以该入参的数量作为Batch值</br>第一个值为额外值并固定为0。</td>
      <td>支持空Tensor</td>
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
      <td>sequsedOriKvOptional</td>
      <td>输入</td>
      <td>表示不同Batch中ori_kv实际参与运算的Sequence Length。</td>
      <td>支持空Tensor</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(B，)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedCmpKvOptional</td>
      <td>输入</td>
      <td>表示不同Batch中cmp_kv实际参与运算的Sequence Length。</td>
      <td>支持空Tensor</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(B，)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpResidualKvOptional</td>
      <td>输入</td>
      <td>表示不同Batch中cmp_kv压缩后Sequence Length的余数，配合cmpRatio实现cmp_kv部分的mask和负载计算。</br>如果cmpRatio不为1，且mask为3，则必须传入cmpResidualKvOptional。</td>
      <td>支持空Tensor</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(B，)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>oriTopkLengthOptional</td>
      <td>输入</td>
      <td>预留参数，表示不同q token对应的ori_kv部分关键稀疏token的个数。</br>如果oriTopk不为0，且mask为0，则必须传入oriTopkLengthOptional。</td>
      <td>支持空Tensor</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B, S1, N2)或(T1, N2)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpTopkLengthOptional</td>
      <td>输入</td>
      <td>预留参数，表示不同q token对应的cmp_kv部分关键稀疏token的个数。</br>如果cmpTopk不为0，且mask为0，则必须传入cmpTopkLengthOptional。</td>
      <td>支持空Tensor</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B, S1, N2)或(T1, N2)</td>
      <td>√</td>
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
      <td>表示Key的head个数。</td>
      <td>支持非负数</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim</td>
      <td>输入</td>
      <td>注意力头的维度。</td>
      <td>支持非负数</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
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
      <td>支持非负数</br>建议值为0</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenOriKv</td>
      <td>输入</td>
      <td>表示ori_kv的最长Sequence Length。</td>
      <td>支持非负数</br>建议值为0</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenCmpKv</td>
      <td>输入</td>
      <td>表示cmp_kv的最长Sequence Length。</td>
      <td>支持非负数</br>建议值为0</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriTopk</td>
      <td>输入</td>
      <td>表示ori_kv中筛选出的关键稀疏token的个数。0表示非稀疏场景</td>
      <td>支持非负数</br>建议值为0</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpTopk</td>
      <td>输入</td>
      <td>表示cmp_kv中筛选出的关键稀疏token的个数。0表示非稀疏场景</td>
      <td>支持非负数</br>建议值为0</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatio</td>
      <td>输入</td>
      <td>表示对cmp_kv的压缩率。</td>
      <td>取值范围[1，128]</br>建议值0，表示无压缩。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriMaskMode</td>
      <td>输入</td>
      <td>表示q和ori_kv计算的mask模式。</td>
      <td>0: No mask</br>3: rightDownCausal模式的mask，对应以右顶点为划分的下三角场景</br>4: sliding window模式的mask，对应由oriWinLeft和oriWinRight划分的窗口场景</br>建议值为0</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpMaskMode</td>
      <td>输入</td>
      <td>表示q和cmp_kv计算的mask模式。</td>
      <td>0: No mask</br>3: rightDownCausal模式的mask，对应以右顶点为划分的下三角场景</br>建议值为0</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWinLeft</td>
      <td>输入</td>
      <td>表示q和ori_kv计算中q对过去token计算的数量。</td>
      <td>取值范围≥-1，-1表示无穷大</br>建议值为-1</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWinRight</td>
      <td>输入</td>
      <td>表示q和ori_kv计算中q对未来token计算的数量。</td>
      <td>取值范围≥-1，-1表示无穷大</br>建议值为-1</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQOptional</td>
      <td>输入</td>
      <td>表示Query的排列格式。</td>
      <td>支持 BSND、TND</br>建议值为BSND</td>
      <td>char*</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKvOptional</td>
      <td>输入</td>
      <td>表示Key的排列格式。</td>
      <td>支持 BSND、TND、PA_BNBD</br>建议值为BSND</td>
      <td>char*</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>hasOriKv</td>
      <td>输入</td>
      <td>用于标识是否含有ori_kv。</td>
      <td>true: 含有ori_kv</br>false: 不含有ori_kv</br>建议值为true</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>hasCmpKv</td>
      <td>输入</td>
      <td>用于标识是否含有cmp_kv。</td>
      <td>true: 含有cmp_kv</br>false: 不含有cmp_kv</br>建议值为true</td>
      <td>bool</td>
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

## aclnnSparseFlashMlaMetadata
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
            <td>在Device侧申请的workspace大小，由第一段接口aclnnSparseFlashMlaMetadataGetWorkspaceSize获取</td>
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
  - aclnnSparseFlashMlaMetadata默认确定性实现

  - Batch取值规则
    - 优先获取sequsedQOptional中的Batch信息
    - 如果未传入sequsedQOptional，且layoutQOptional为TND和传入了cuSeqlensQOptional，则获取cuSeqlensQOptional中的Batch信息
    - 除上所述，使用batchSize
  - Query Sequence Length取值规则
    - 优先获取sequsedQOptional中的Sequence Length信息
    - 如果未传入sequsedQOptional，且layoutQOptional为TND和传入了cuSeqlensQOptional，则获取cuSeqlensQOptional中的Sequence Length信息
    - 除上所述，使用maxSeqlenQ
    - ori_kv、cmp_kv Sequence Length与Query的获取规则一致
  - BSND场景
    - 当传入的layoutQOptional为"BSND"时，在未传入sequsedQOptional的情况下，必传maxSeqlenQ参数
    - 当传入的layoutKvOptional为"BSND"时，若hasOriKv为true，在未传入sequsedOriKvOptional的情况下，必传maxSeqlenOriKv参数；若hasCmpKv为true，在未传入sequsedCmpKvOptional的情况下，必传maxSeqlenCmpKv参数
  - TND场景
    - 当传入的layoutQOptional为"TND"时，必传cuSeqlensQOptional参数
    - 当传入的layoutKvOptional为"TND"时，若hasOriKv为true，必传cuSeqlensOriKvOptional；若hasCmpKv为true，必传cuSeqlensCmpKvOptional参数

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
#include "aclnnop/aclnn_sparse_flash_mla_metadata.h"

#define CHECK_LOG_RET(cond, ret_val, fmt, ...)      \
    do {                                            \
        if (!(cond)) {                              \
            printf(fmt "\n", ##__VA_ARGS__);        \
            return (ret_val);                       \
        }                                           \
    } while (0)

// 参考 sparse_flash_mla_metadata.h
constexpr uint32_t AIC_CORE_NUM = 36;
constexpr uint32_t AIV_CORE_NUM = 72;
constexpr uint32_t FA_METADATA_SIZE = 9;
constexpr uint32_t FD_METADATA_SIZE = 8;

// FA Metadata Index Definitions
constexpr uint32_t FA_CORE_ENABLE_INDEX = 0;
constexpr uint32_t FA_BN2_START_INDEX = 1;
constexpr uint32_t FA_M_START_INDEX = 2;
constexpr uint32_t FA_S2_START_INDEX = 3;
constexpr uint32_t FA_BN2_END_INDEX = 4;
constexpr uint32_t FA_M_END_INDEX = 5;
constexpr uint32_t FA_S2_END_INDEX = 6;
constexpr uint32_t FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7;
constexpr uint32_t FA_S2_MAX_NUM = 8;

// FD Metadata Index Definitions
constexpr uint32_t FD_CORE_ENABLE_INDEX = 0;
constexpr uint32_t FD_BN2_IDX_INDEX = 1;
constexpr uint32_t FD_M_IDX_INDEX = 2;
constexpr uint32_t FD_WORKSPACE_IDX_INDEX = 3;
constexpr uint32_t FD_WORKSPACE_NUM_INDEX = 4;
constexpr uint32_t FD_M_START_INDEX = 5;
constexpr uint32_t FD_M_NUM_INDEX = 6;

struct QsmlaMetadata {
    uint32_t faMetadata[AIC_CORE_NUM][FA_METADATA_SIZE];
    uint32_t fdMetadata[AIV_CORE_NUM][FD_METADATA_SIZE];
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
    bool hasResidualKv { false };
};

struct ArgContext {
    // required input
    int64_t numHeadsQ { 0 };
    int64_t numHeadsKv { 0 };
    int64_t headDim { 0 };
    // optional input
    Tensor cuSeqlensQOptional {};
    Tensor cuSeqlensOriKvOptional {};
    Tensor cuSeqlensCmpKvOptional {};
    Tensor sequsedQOptional {};
    Tensor sequsedOriKvOptional {};
    Tensor sequsedCmpKvOptional {};
    Tensor cmpResidualKvOptional {};
    Tensor oriTopkLengthOptional {};
    Tensor cmpTopkLengthOptional {};
    int64_t batchSize { 0 };
    int64_t maxSeqlenQ { 0 };
    int64_t maxSeqlenOriKv { 0 };
    int64_t maxSeqlenCmpKv { 0 };
    int64_t oriTopk { 0 };
    int64_t cmpTopk { 0 };
    int64_t cmpRatio { 0 };
    int64_t oriMaskMode { 0 };
    int64_t cmpMaskMode { 0 };
    int64_t oriWinLeft { -1 };
    int64_t oriWinRight { -1 };
    char *layoutQOptional { nullptr };
    char *layoutKvOptional { nullptr };
    bool hasOriKv { true };
    bool hasCmpKv { true };
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
    DestroyTensor(context.cuSeqlensOriKvOptional);
    DestroyTensor(context.cuSeqlensCmpKvOptional);
    DestroyTensor(context.sequsedQOptional);
    DestroyTensor(context.sequsedOriKvOptional);
    DestroyTensor(context.sequsedCmpKvOptional);
    DestroyTensor(context.cmpResidualKvOptional);
    DestroyTensor(context.oriTopkLengthOptional);
    DestroyTensor(context.cmpTopkLengthOptional);

    if (context.layoutQOptional != nullptr) {
        free(context.layoutQOptional);
        context.layoutQOptional = nullptr;
    }
    if (context.layoutKvOptional != nullptr) {
        free(context.layoutKvOptional);
        context.layoutKvOptional = nullptr;
    }
}

aclnnStatus CreateArgs(const ArgScenario &scenario, ArgContext &context)
{
    ScopeGuard argsGuard([&] { DestroyArgs(context); });
    aclnnStatus ret;
    int64_t batchSize = 4;

    context.numHeadsQ = 64;
    context.numHeadsKv = 1;
    context.headDim = 512;
    ret = CreateTensor(aclDataType::ACL_INT32, { 1024 }, context.metaData);     // 1024: Fix size
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create metadata failed. Error: %d", ret);
    context.oriTopk = 0;
    context.cmpTopk = 0;
    context.cmpRatio = 128;
    context.oriMaskMode = 4;
    context.cmpMaskMode = 3;
    context.oriWinLeft = 127;
    context.oriWinRight = 0;
    context.layoutQOptional = (char *)malloc(sizeof(char) * 16);
    context.layoutKvOptional = (char *)malloc(sizeof(char) * 16);
    strcpy(context.layoutQOptional, "BSND");                // BSND,TND
    strcpy(context.layoutKvOptional, "BSND");                // BSND,TND,PA_BNBD

    if (!scenario.hasCuSeq && !scenario.hasSeqused) {
        context.batchSize = batchSize;
        context.maxSeqlenOriKv = 1024;
        context.maxSeqlenCmpKv = 1024;
        context.maxSeqlenQ = 1024;
        return ACL_SUCCESS;
    }

    if (scenario.hasCuSeq) {
        // (B+1,), first element is always 0
        ret = CreateTensor(aclDataType::ACL_INT32, { batchSize + 1 }, context.cuSeqlensQOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensQOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, { batchSize + 1 }, context.cuSeqlensOriKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensOriKvOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, { batchSize + 1 }, context.cuSeqlensCmpKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqlensCmpKvOptional failed. Error: %d", ret);
    }

    if (scenario.hasSeqused) {
        // (B,)
        ret = CreateTensor(aclDataType::ACL_INT32, { batchSize }, context.sequsedQOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedQOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, { batchSize }, context.sequsedOriKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedOriKvOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, { batchSize }, context.sequsedCmpKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create sequsedCmpKvOptional failed. Error: %d", ret);
    }

    if (scenario.hasResidualKv) {
        // (B,)
        ret = CreateTensor(aclDataType::ACL_INT32, { batchSize }, context.cmpResidualKvOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cmpResidualKvOptional failed. Error: %d", ret);
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
    scenario.hasResidualKv = true;
    ArgContext context {};
    ret = CreateArgs(scenario, context);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create input arguments failed. ERROR: %d", ret);
    ScopeGuard argsGuard([&] { DestroyArgs(context); });

    // 3. 调用CANN算子库API，需要修改为具体的API
    // 调用aclnnLightningIndexerV2Metadata第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
    ret = aclnnSparseFlashMlaMetadataGetWorkspaceSize(
        context.cuSeqlensQOptional.data, context.cuSeqlensOriKvOptional.data, context.cuSeqlensCmpKvOptional.data,
        context.sequsedQOptional.data, context.sequsedOriKvOptional.data, context.sequsedCmpKvOptional.data,
        context.cmpResidualKvOptional.data, context.oriTopkLengthOptional.data, context.cmpTopkLengthOptional.data, 
        context.numHeadsQ, context.numHeadsKv, context.headDim, context.batchSize, context.maxSeqlenQ,
        context.maxSeqlenOriKv, context.maxSeqlenCmpKv, context.oriTopk, context.cmpTopk, context.cmpRatio,
        context.oriMaskMode, context.cmpMaskMode, context.oriWinLeft = 127, context.oriWinRight = 0,
        context.layoutQOptional, context.layoutKvOptional, context.hasOriKv, context.hasCmpKv, context.metaData.data,
        &workspaceSize, &executor);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret,
        "aclnnSparseFlashMlaMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);

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
    ret = aclnnSparseFlashMlaMetadata(workspaceAddr, workspaceSize, executor, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclnnSparseFlashMlaMetadata failed. ERROR: %d\n", ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSynchronizeStream failed. ERROR: %d\n", ret);

    // 5. 打印输出
    QliV2MetaData result {};
    ret = aclrtMemcpy(&result, sizeof(result), context.metaData.deviceAddr, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d\n", ret);

    for (uint32_t i = 0; i < AIC_CORE_NUM; ++i) {
        printf("AIC Core%u\n", i);
        printf("    Core Enable : %u\n", result.faData[i][FA_CORE_ENABLE_INDEX]);
        printf("    Start BN2   : %u\n", result.faData[i][FA_BN2_START_INDEX]);
        printf("    Start M     : %u\n", result.faData[i][FA_M_START_INDEX]);
        printf("    Start S2    : %u\n", result.faData[i][FA_S2_START_INDEX]);
        printf("    End BN2     : %u\n", result.faData[i][FA_BN2_END_INDEX]);
        printf("    End M       : %u\n", result.faData[i][FA_M_END_INDEX]);
        printf("    End S2      : %u\n", result.faData[i][FA_S2_END_INDEX]);
        printf("    First Worksapce Index : %u\n", result.faData[i][FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX]);
        printf("    Max S2 Block Num : %u\n", result.faData[i][FA_S2_MAX_NUM]);
    }
    for (uint32_t i = 0; i < AIV_CORE_NUM; ++i) {
        printf("AIV Core%u\n", i);
        printf("    Core Enable             : %u\n", result.fdData[i][FD_CORE_ENABLE_INDEX]);
        printf("    FD Task BN2 Idx         : %u\n", result.fdData[i][FD_BN2_IDX_INDEX]);
        printf("    FD Task M Idx           : %u\n", result.fdData[i][FD_M_IDX_INDEX]);
        printf("    FD Task S2 Idx          : %u\n", result.fdData[i][FD_WORKSPACE_IDX_INDEX]);
        printf("    FD Task Workspace Num   : %u\n", result.fdData[i][FD_WORKSPACE_NUM_INDEX]);
        printf("    FD Subtask M Start      : %u\n", result.fdData[i][FD_M_START_INDEX]);
        printf("    FD Subtask M Num        : %u\n", result.fdData[i][FD_M_NUM_INDEX]);
    }

    return 0;
}
```

## 调用说明

| 调用方式  | 样例代码                                                                | 说明                                                                                          |
| ----------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| aclnn接口 | [test_aclnn_sparse_flash_mla_metadata.cpp](./examples/test_aclnn_sparse_flash_mla_metadata.cpp) | 通过[`aclnnSparseFlashMlaMetadata`](./docs/aclnnSparseFlashMlaMetadata.md)接口方式调用 |

## 附录：

### 归属领域
transformer/attention

### Pytorch AtenIR
```
npu_sparse_flash_mla_metadata(
    int num_heads_q, int num_heads_kv, int head_dim, *, Tensor? cu_seqlens_q=None,
    Tensor? cu_seqlens_ori_kv=None, Tensor? cu_seqlens_cmp_kv=None, Tensor? seqused_q=None, Tensor? seqused_ori_kv=None,
    Tensor? seqused_cmp_kv=None, Tensor? cmp_residual_kv=None, Tensor? ori_topk_length=None,
    Tensor? cmp_topk_length=None, int? batch_size=None, int? max_seqlen_q=None, int? max_seqlen_ori_kv=None,
    int? max_seqlen_cmp_kv=None, int? ori_topk=None, int? cmp_topk=None, int? cmp_ratio=None, int? ori_mask_mode=None,
    int? cmp_mask_mode=None, int? ori_win_left=None, int? ori_win_right=None, str? layout_q=None, str? layout_kv=None,
    bool? has_ori_kv=None, bool? has_cmp_kv=None) -> Tensor

参考：
aclnnStatus aclnnSparseFlashMlaMetadataGetWorkspaceSize(
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensOriKvOptional,
    const aclTensor *cuSeqlensCmpKvOptional, const aclTensor *sequsedQOptional, const aclTensor *sequsedOriKvOptional,
    const aclTensor *sequsedCmpKvOptional, const aclTensor *cmpResidualKvOptional,
    const aclTensor *oriTopkLengthOptional, const aclTensor *cmpTopkLengthOptional, int64_t numHeadsQ,
    int64_t numHeadsKv, int64_t headDim, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenOriKv,
    int64_t maxSeqlenCmpKv, int64_t oriTopk, int64_t cmpTopk, int64_t cmpRatio, int64_t oriMaskMode,
    int64_t cmpMaskMode, int64_t oriWinLeft, int64_t oriWinRight, const char *layoutQOptional,
    const char *layoutKvOptional, bool hasOriKv, bool hasCmpKv, int64_t batchInvariant, const aclTensor *metaData,
    uint64_t *workspaceSize, aclOpExecutor **executor);
```

### AtenIR参数描述
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
        <th>类型</th>
        <th>名称</th>
        <th>描述</th>
        <th>GPU支持的数据类型</th>
        <th>CPU支持的数据类型</th>
        </tr></thead>
    <tbody>
        <tr>
            <td>属性</td>
            <td>num_heads_q</td>
            <td>表示query的head个数</td>
            <td>INT32, INT64</td>
            <td>INT32, INT64</td>
        </tr>
        <tr>
            <td>属性</td>
            <td>num_heads_kv</td>
            <td>key和value对应的多头数</td>
            <td>INT32, INT64</td>
            <td>INT32, INT64</td>
        </tr>
        <tr>
            <td>属性</td>
            <td>head_dim</td>
            <td>表示token数</td>
            <td>INT32, INT64</td>
            <td>INT32, INT64</td>
        </tr>
        <tr>
            <td>输入</td>
            <td>cu_seqlens_q</td>
            <td>表示不同batch中query的有效Sequence Length</td>
            <td>INT32</td>
            <td>INT32</td>
        </tr>
        <tr>
            <td>输入</td>
            <td>cu_seqlens_ori_kv</td>
            <td>表示不同batch中ori_kv的有效Sequence Length</td>
            <td>INT32</td>
            <td>INT32</td>
        </tr>
        <tr>
            <td>输入</td>
            <td>cu_seqlens_cmp_kv</td>
            <td>表示不同batch中cmp_kv的有效Sequence Length</td>
            <td>INT32</td>
            <td>INT32</td>
        </tr>
        <tr>
            <td>输入</td>
            <td>seqused_q</td>
            <td>表示不同batch中query实际参与运算的Sequence Length</td>
            <td>INT32</td>
            <td>INT32</td>
        </tr>
        <tr>
            <td>输入</td>
            <td>seqused_ori_kv</td>
            <td>表示不同batch中ori_kv实际参与运算的Sequence Length</td>
            <td>INT32</td>
            <td>INT32</td>
        </tr>
        <tr>
            <td>输入</td>
            <td>seqused_cmp_kv</td>
            <td>表示不同batch中cmp_kv实际参与运算的Sequence Length</td>
            <td>INT32</td>
            <td>INT32</td>
        </tr>
        <tr>
            <td>输入</td>
            <td>cmp_residual_kv</td>
            <td>表示不同Batch中cmp_kv压缩后Sequence Length的余数</td>
            <td>INT32</td>
            <td>INT32</td>
        </tr>
        <tr>
            <td>输入</td>
            <td>ori_topk_length</td>
            <td>预留参数，表示不同q token对应的ori_kv部分关键稀疏token的个数</td>
            <td>INT32</td>
            <td>INT32</td>
        </tr>
        <tr>
            <td>输入</td>
            <td>cmp_topk_length</td>
            <td>预留参数，表示不同q token对应的cmp_kv部分关键稀疏token的个数</td>
            <td>INT32</td>
            <td>INT32</td>
        </tr>
        <tr>
            <td>属性</td>
            <td>batch_size</td>
            <td>表示batch数量</td>
            <td>INT32, INT64</td>
            <td>INT32, INT64</td>
        </tr>
        <tr>
            <td>属性</td>
            <td>max_seqlen_q</td>
            <td>表示query的最长seqlen</td>
            <td>INT32, INT64</td>
            <td>INT32, INT64</td>
        </tr>
        <tr>
            <td>属性</td>
            <td>max_seqlen_ori_kv</td>
            <td>表示ori_kv的最长seqlen</td>
            <td>INT32, INT64</td>
            <td>INT32, INT64</td>
        </tr>
        <tr>
            <td>属性</td>
            <td>max_seqlen_cmp_kv</td>
            <td>表示cmp_kv的最长seqlen</td>
            <td>INT32, INT64</td>
            <td>INT32, INT64</td>
        </tr>
        <tr>
            <td>属性</td>
            <td>ori_topk</td>
            <td>表示ori_kv中筛选出的关键稀疏token的个数</td>
            <td>INT32, INT64</td>
            <td>INT32, INT64</td>
        </tr>
        <tr>
            <td>属性</td>
            <td>cmp_topk</td>
            <td>表示cmp_kv中筛选出的关键稀疏token的个数</td>
            <td>INT32, INT64</td>
            <td>INT32, INT64</td>
        </tr>
        <tr>
            <td>属性</td>
            <td>cmp_ratio</td>
            <td>表示对cmp_kv的压缩率</td>
            <td>INT32, INT64</td>
            <td>INT32, INT64</td>
        </tr>
        <tr>
            <td>属性</td>
            <td>ori_mask_mode</td>
            <td>表示q和ori_kv计算的mask模式</td>
            <td>INT32, INT64</td>
            <td>INT32, INT64</td>
        </tr>
        <tr>
            <td>属性</td>
            <td>cmp_mask_mode</td>
            <td>表示q和cmp_kv计算的mask模式</td>
            <td>INT32, INT64</td>
            <td>INT32, INT64</td>
        </tr>
        <tr>
            <td>属性</td>
            <td>ori_win_left</td>
            <td>表示q和ori_kv计算中q对过去token计算的数量</td>
            <td>INT32, INT64</td>
            <td>INT32, INT64</td>
        </tr>
        <tr>
            <td>属性</td>
            <td>ori_win_right</td>
            <td>表示q和ori_kv计算中q对未来token计算的数量</td>
            <td>INT32, INT64</td>
            <td>INT32, INT64</td>
        </tr>
        <tr>
            <td>属性</td>
            <td>layout_q</td>
            <td>表示query的排列格式</td>
            <td>STRING</td>
            <td>STRING</td>
        </tr>
        <tr>
            <td>属性</td>
            <td>layout_kv</td>
            <td>表示kv的排列格式</td>
            <td>STRING</td>
            <td>STRING</td>
        </tr>
        <tr>
            <td>属性</td>
            <td>has_ori_kv</td>
            <td>用于标识是否含有ori_kv</td>
            <td>BOOL</td>
            <td>BOOL</td>
        </tr>
        <tr>
            <td>属性</td>
            <td>has_cmp_kv</td>
            <td>用于标识是否含有cmp_kv</td>
            <td>BOOL</td>
            <td>BOOL</td>
        </tr>
        <tr>
            <td>输出</td>
            <td>output</td>
            <td>输出tensor, 表示负载均衡结果输出</td>
            <td>INT32, INT64</td>
            <td>INT32, INT64</td>
        </tr>
    </tbody>
    </table>

### HostAPI接口约束

<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 50px">
    <col style="width: 50px">
    <col style="width: 50px">
    </colgroup>
    <thead>
        <tr>
            <th>功能维度</th>
            <th>以支持</th>
            <th>应支持但未支持</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>数据类型</td>
            <td>INT32, INT64</td>
            <td>NA</td>
        </tr>
        <tr>
            <td>数据格式</td>
            <td>ND</td>
            <td>NA</td>
        </tr>
        <tr>
            <td>空Tensor</td>
            <td>支持空进空出</td>
            <td>NA</td>
        </tr>
        <tr>
            <td>非连续Tensor</td>
            <td>支持输入非连续，不支持输出非连续</td>
            <td>NA</td>
        </tr>
    </tbody>
    </table>

### HostAPI异常处理
以下场景会出现参数校验异常：

1. 传入的Tensor数据类型不在支持的范围内
2. 传入的数据未满足约束条件，参考[约束说明](#约束说明)

### 兼容性说明
1. 功能兼容性：兼容 Pytorch/TensorFlow/MindSpore/Onnx 框架定义；
2. 平台兼容性：已支持的芯片版本，功能无差异
3. 接口兼容性：新增接口
4. 行为兼容性：新增接口
5. 性能兼容性：新增接口
6. 资源兼容性：新增接口
7. 错误处理兼容性：新增接口
