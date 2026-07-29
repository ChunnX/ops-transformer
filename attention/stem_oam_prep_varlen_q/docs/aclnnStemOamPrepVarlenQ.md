# aclnnStemOamPrepVarlenQ

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

> 仅支持 arch35 (DAV_3510) 架构芯片。

## 功能说明

- 接口功能：完成 Stem OAM block-sparse attention 中 Q 侧预处理计算。将变长 Q tensor 从 paged 存储格式转化为按 stem block (128 token) 分组的 flattened qFlat 输出，供后续 OAM score 计算。
- 计算公式：

  阶段1 Scale Fusion:
  $$q\_scale[b, h, pos] = qscale[b, h, pos]$$

  阶段2 De-page Varlen:
  $$Q\_dense[b] = Cast(q[cu\_seqlens\_q[b]:cu\_seqlens\_q[b]+q\_len[b], :, :], \text{FP32})$$

  阶段3 Weighted Group Sum (自然顺序，NO flip):
  $$Q\_group\_sum[b,h,qb,g,:] = \sum_{r=0}^{R-1} Q\_blocks[b,h,qb,r,g,:] \times q\_scale[b,h,position(qb,r,g)]$$

  阶段4 Flatten:
  $$qflat[b, h, qb, g \times D : (g+1) \times D] = Q\_group\_sum[b, h, qb, g, :]$$

  阶段5 Cast输出:
  $$qflat\_out = qflat.to(\text{BF16})$$

- 关键特性：Q 侧 stride 维度为自然顺序（不翻转），与 K 侧的翻转处理不同。

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnStemOamPrepVarlenQGetWorkspaceSize"接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用"aclnnStemOamPrepVarlenQ"接口执行计算。

```cpp
aclnnStatus aclnnStemOamPrepVarlenQGetWorkspaceSize(
  const aclTensor   *q,
  const aclIntArray *qSeqLens,
  const aclIntArray *cuSeqLensQ,
  const aclTensor   *qScale,
  const int64_t      stemBlockSize,
  const int64_t      stemStride,
  aclTensor         *qFlat,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```cpp
aclnnStatus aclnnStemOamPrepVarlenQ(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnStemOamPrepVarlenQGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 350px">
  <col style="width: 250px">
  <col style="width: 100px">
  <col style="width: 100px">
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
      <td>q（aclTensor*）</td>
      <td>输入</td>
      <td>表示变长Q tensor，所有batch的token拼接存储。对应公式中q。</td>
      <td><ul><li>不支持空Tensor。</li><li>最后一维必须等于stemBlockSize（默认128）。</li><li>数据类型为FLOAT8_E4M3FN、BFLOAT16或FLOAT16。</li></ul></td>
      <td>FLOAT8_E4M3FN、BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3 ([total_tokens, H_q, D])</td>
      <td>×</td>
    </tr>
    <tr>
      <td>qScale（aclTensor*）</td>
      <td>输入</td>
      <td>表示Q的per-token scale factor，仅FP8输入时有效。对应公式中qscale。</td>
      <td><ul><li>FP8输入时必填，shape为[total_tokens, H_q]，数据类型必须为FLOAT。</li><li>BF16/FP16输入时传入nullptr，Kernel内部跳过加权，执行无权重Group Sum。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>2 ([total_tokens, H_q])</td>
      <td>×</td>
    </tr>
    <tr>
      <td>qSeqLens（aclIntArray*）</td>
      <td>输入</td>
      <td>表示每个batch的Q序列长度。对应公式中q_seq_lens。</td>
      <td><ul><li>长度等于batch。</li><li>每个值≥0。</li><li>不支持空Array。</li><li>batch最大值为1024。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>1 ([batch])</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cuSeqLensQ（aclIntArray*）</td>
      <td>输入</td>
      <td>表示Q的累积序列长度偏移量，用于varlen索引。对应公式中cu_seqlens_q。</td>
      <td><ul><li>长度等于batch+1。</li><li>cuSeqLensQ[0]必须为0。</li><li>cuSeqLensQ[batch]必须等于total_tokens。</li><li>单调递增。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>1 ([batch+1])</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stemBlockSize（int64_t）</td>
      <td>输入</td>
      <td>表示stem block大小,对应公式中B。控制每个stem block的token数量,决定Q Processing的分组粒度。推荐默认值128。</td>
      <td><ul><li>必须是32的倍数。</li><li>最大值为256。</li><li>推荐默认值为128。</li><li>必须满足stemStride≤stemBlockSize。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stemStride（int64_t）</td>
      <td>输入</td>
      <td>表示stem stride大小,对应公式中S。控制stem block内stride group的token数量,决定qFlat的维度粒度。推荐默认值16。</td>
      <td><ul><li>必须是16的倍数。</li><li>最大值为64。</li><li>推荐默认值为16。</li><li>必须满足stemStride≤stemBlockSize。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>qFlat（aclTensor*）</td>
      <td>输出</td>
      <td>表示flattened Q输出，供OAM score计算使用。对应公式中qflat_out。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型固定为BFLOAT16。</li><li>shape为[batch, H_q, max_Qb, kflat_dim]，其中max_Qb=ceil(max(qSeqLens)/stemBlockSize)，kflat_dim=stemStride×D(默认2048)。</li></ul></td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>4 ([batch, H_q, max_Qb, kflat_dim])</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 550px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>q、qFlat存在空指针。qScale在FP8输入时不允许空指针；在BF16/FP16输入时允许nullptr。</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>q的数据类型不在支持范围之内（仅支持FLOAT8_E4M3FN、BFLOAT16、FLOAT16）。</td>
    </tr>
    <tr>
      <td>qScale的数据类型不为FLOAT。</td>
    </tr>
    <tr>
      <td>q的最后一维不等于stemBlockSize。</td>
    </tr>
    <tr>
      <td>cuSeqLensQ长度不等于batch+1或非单调递增。</td>
    </tr>
    <tr>
      <td>qFlat的shape维度不符合预期。</td>
    </tr>
    <tr>
      <td>stemBlockSize或stemStride不满足约束（stemBlockSize%32≠0或stemBlockSize>256；stemStride%16≠0或stemStride>64或stemStride>stemBlockSize）。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER</td>
      <td>361001</td>
      <td>内部错误，如Tiling计算失败或Kernel查找失败。</td>
    </tr>
  </tbody></table>

## aclnnStemOamPrepVarlenQ

- **参数说明**

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
  </thead>
  <tbody>
    <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
    <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnStemOamPrepVarlenQGetWorkspaceSize获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

- 确定性说明：aclnnStemOamPrepVarlenQ默认确定性实现。
- <term>Ascend 950PR/Ascend 950DT</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：仅支持arch35架构，不支持arch22。
- q的最后一维D必须等于stemBlockSize（推荐默认128）。
- stemBlockSize和stemStride为运行时参数。约束：stemBlockSize % 32 == 0且stemBlockSize ≤ 256；stemStride % 16 == 0且stemStride ≤ 64且stemStride ≤ stemBlockSize。推荐默认值：stemBlockSize=128, stemStride=16。派生值：R = stemBlockSize / stemStride, kflat_dim = stemStride × D。
- qFlat输出维度kflat_dim = stemStride × D（默认16×128=2048）。
- Q侧stride维度为自然顺序（g ∈ [0, S-1]），不翻转（与K侧处理不同）。
- 当qSeqLens中某batch值为0时，该batch对应的qFlat输出填充为零。
- 当q数据类型为BFLOAT16或FLOAT16时，qScale传入nullptr；此时Kernel内部Q Processing执行无权重Group Sum (Σ_r Q_blocks[:, r, :, :])，而非加权Group Sum (Σ_r Q_blocks × qScale)。
- FP8输入时，qScale最后一维需要能覆盖max(qSeqLens)个位置（padding到stemBlockSize对齐）。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

<term>Ascend 950PR/Ascend 950DT</term>、<term>Atlas A3 推理系列产品</term>：

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_stem_oam_prep_varlen_q.h"

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
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor, aclFormat format)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format, shape.data(),
                              shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. 初始化Device和Stream
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 创建输入tensor
    int64_t totalTokens = 256;
    int64_t numHeadQ = 32;
    int64_t dimQk = 128;
    int64_t batch = 2;
    int64_t stemBlockSize = 128;
    int64_t stemStride = 16;
    int64_t kflatDim = stemStride * dimQk;
    int64_t maxQb = 1;

    std::vector<int64_t> qShape = {totalTokens, numHeadQ, dimQk};
    std::vector<int64_t> qScaleShape = {totalTokens, numHeadQ};
    std::vector<int64_t> qFlatShape = {batch, numHeadQ, maxQb, kflatDim};

    void *qDeviceAddr = nullptr;
    void *qScaleDeviceAddr = nullptr;
    void *qFlatDeviceAddr = nullptr;
    aclTensor *q = nullptr;
    aclTensor *qScale = nullptr;
    aclTensor *qFlat = nullptr;

    std::vector<uint8_t> hostQ(GetShapeSize(qShape), 1);
    std::vector<float> hostQScale(GetShapeSize(qScaleShape), 1.0f);
    std::vector<uint16_t> hostQFlat(GetShapeSize(qFlatShape), 0);

    ret = CreateAclTensor(hostQ, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &q,
                          aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostQScale, qScaleShape, &qScaleDeviceAddr, aclDataType::ACL_FLOAT, &qScale,
                          aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostQFlat, qFlatShape, &qFlatDeviceAddr, aclDataType::ACL_BF16, &qFlat,
                          aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int64_t qSeqLensData[] = {128, 128};
    int64_t cuSeqLensQData[] = {0, 128, 256};
    aclIntArray *qSeqLens = aclCreateIntArray(qSeqLensData, 2);
    aclIntArray *cuSeqLensQ = aclCreateIntArray(cuSeqLensQData, 3);

    // 3. 获取workspace大小
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    ret = aclnnStemOamPrepVarlenQGetWorkspaceSize(q, qSeqLens, cuSeqLensQ, qScale, stemBlockSize, stemStride,
                                                  qFlat, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStemOamPrepVarlenQGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 4. 申请workspace并执行计算
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnStemOamPrepVarlenQ(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStemOamPrepVarlenQ failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 释放资源
    aclDestroyTensor(q);
    aclDestroyTensor(qScale);
    aclDestroyTensor(qFlat);
    aclDestroyIntArray(qSeqLens);
    aclDestroyIntArray(cuSeqLensQ);
    aclrtFree(qDeviceAddr);
    aclrtFree(qScaleDeviceAddr);
    aclrtFree(qFlatDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```