# aclnnKeyPool

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

KeyPool 是推理场景下的 Key 压缩算子。算子对每个输入 token 分别执行 K
投影和 Gate 投影，按 `cmpRatio` 个 token 进行 Gate 加权池化，并将未完成
压缩组所需的中间状态写入分页 `stateCache`。

对当前输入的每个 token：

```text
K = hidden_states @ wk.T
G = hidden_states @ gate_weight.T
K_norm = LayerNorm(K, dim=-1)  # 提供 normWeight/normBias 时
```

对第 `q` 个全局压缩组：

```text
logits[q, r] = G[q * cmpRatio + r] + ape[r]
weight = softmax(logits, dim=0)
pooled_key[b, q - startPool[b]] =
    sum(weight[r] * K_norm[q * cmpRatio + r], dim=0)
```

历史 token 通过 `cacheBlockTable` 从 `stateCache` 读取，当前输入 token 的投影
结果优先使用当前调用产生的数据。对于 `startPos` 未按 `cmpRatio` 对齐的场景，
首个压缩组可以由历史 token 和当前输入 token 共同组成。`ape` 只参与池化计算，
不写入 `stateCache`。RoPE 参数入口保留用于后续扩展，但当前版本不实现 RoPE。

LayerNorm 沿每个 token 的最后一维 `D` 独立计算，不跨 token 归约。

## 函数原型

算子采用两段式接口。必须先调用
`aclnnKeyPoolGetWorkspaceSize` 获取 workspace 大小和 executor，再调用
`aclnnKeyPool` 执行计算。

```cpp
aclnnStatus aclnnKeyPoolGetWorkspaceSize(
    const aclTensor *hiddenStates,
    const aclTensor *wk,
    const aclTensor *gateWeight,
    const aclTensor *ape,
    aclTensor       *stateCacheRef,
    const aclTensor *cacheBlockTable,
    const aclTensor *startPos,
    const aclTensor *normWeightOptional,
    const aclTensor *normBiasOptional,
    const aclTensor *cosOptional,
    const aclTensor *sinOptional,
    const aclTensor *cuSeqlensOptional,
    const aclTensor *sequsedOptional,
    int64_t          cmpRatio,
    double           normEps,
    int64_t          rotaryMode,
    int64_t          stateCacheStrideDim0,
    aclTensor       *pooledKeyOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor);
```

```cpp
aclnnStatus aclnnKeyPool(
    void          *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream);
```

`stateCacheRef` 为原地更新输入，`pooledKeyOut` 是唯一输出。

## aclnnKeyPoolGetWorkspaceSize

### 参数说明

| 参数 | 输入/输出 | 数据类型 | Shape | 说明 |
| --- | --- | --- | --- | --- |
| `hiddenStates` | 输入 | BF16/FP16 | BSH `[B,S,H]` 或 TH `[T,H]` | 当前调用的 token |
| `wk` | 输入 | 与 `hiddenStates` 相同 | `[D,H]` | K 投影权重 |
| `gateWeight` | 输入 | 与 `hiddenStates` 相同 | `[D,H]` | Gate 投影权重 |
| `ape` | 输入 | FP32 | `[R,D]` | `R=cmpRatio` 的位置偏置 |
| `stateCacheRef` | 输入/输出 | FP32 | `[N,BS,2D]` | 前 `D` 列保存 LayerNorm 后、未加 APE 的 K，后 `D` 列保存原始 Gate |
| `cacheBlockTable` | 输入 | INT32 | `[B,L]` | 逻辑 block 到物理 block 的映射 |
| `startPos` | 输入 | INT32 | `[B]` | 每个 batch 的逻辑起始位置 |
| `normWeightOptional` | 可选输入 | FP32 | `[D]` | 与 `normBiasOptional` 成对出现 |
| `normBiasOptional` | 可选输入 | FP32 | `[D]` | 与 `normWeightOptional` 成对出现 |
| `cosOptional` | 可选输入 | BF16 | 保留接口 | 当前版本必须为空 |
| `sinOptional` | 可选输入 | BF16 | 保留接口 | 当前版本必须为空 |
| `cuSeqlensOptional` | 可选输入 | INT32 | TH 时 `[B+1]` | BSH 时必须为空 |
| `sequsedOptional` | 可选输入 | INT32 | 保留接口 | 当前版本必须为空 |
| `cmpRatio` | 属性 | INT64 | 标量 | 压缩率 |
| `normEps` | 属性 | `double`（内部按 FLOAT 使用） | 标量 | LayerNorm epsilon |
| `rotaryMode` | 属性 | INT64 | 标量 | RoPE 模式入口，当前仅作扩展占位 |
| `stateCacheStrideDim0` | 属性 | INT64 | 标量 | - |
| `pooledKeyOut` | 输出 | 与 `hiddenStates` 相同 | `[B,Pcap,D]` | 固定容量池化结果 |

其中：

- `R = cmpRatio`；
- `N` 为物理 block 数；
- `BS` 为 block size；
- `L` 为每个 batch 的最大逻辑 block 数；
- `Pcap = ceil(L * BS / cmpRatio)`。

输入 tensor 使用 ND 格式。`stateCacheRef` 是原地输入，用于保存跨算子调用
的未完成压缩组状态。

### 属性约束

| 属性 | 默认值 | 支持范围 |
| --- | --- | --- |
| `cmpRatio` | 4 | `2/4/8/16/32/64/128` |
| `normEps` | `1e-6` | 大于 0 |
| `rotaryMode` | 1 | `0/1`；当前不执行 RoPE |

### 返回值

返回 `aclnnStatus`。第一段接口负责参数、shape、dtype 和 workspace 校验，
常见错误码请参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

常见错误类型包括：

| 返回值 | 含义 |
| --- | --- |
| `ACLNN_ERR_PARAM_NULLPTR` | 必须提供的参数为空 |
| `ACLNN_ERR_PARAM_INVALID` | shape、dtype、属性或布局不支持 |
| `ACLNN_ERR_RUNTIME_ERROR` | Runtime 或内存申请失败 |
| `ACLNN_ERR_INNER_TILING_ERROR` | Tiling 失败 |

## aclnnKeyPool

第二段接口的参数含义与其他 aclnn 两段式接口一致：

| 参数 | 说明 |
| --- | --- |
| `workspace` | Device 侧 workspace 地址 |
| `workspaceSize` | 第一段接口返回的 workspace 大小 |
| `executor` | 第一段接口生成的算子执行器 |
| `stream` | 执行算子的 ACL stream |

## 约束说明

- 支持 BSH 和 TH 两种场景：
  - BSH 场景 `cuSeqlensOptional` 必须为空；
  - TH 场景必须提供合法的 `cuSeqlensOptional` 前缀和数组，首元素为 0，
    末元素为 `T`，且单调不减。
- `cmpRatio` 仅支持 `2/4/8/16/32/64/128`；`normEps` 必须大于 0；
  `rotaryMode` 仅支持 `0/1`。
- `normWeightOptional` 和 `normBiasOptional` 必须同时提供或同时为空。
- `cosOptional` 和 `sinOptional` 必须同时为空，当前不支持 RoPE；
  `sequsedOptional` 必须为空，当前不支持通过 `attention_mask` 选择 token。
- 支持 BSH 场景的 `B=0` 或 `S=0`，以及 TH 场景的 `T=0`。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

下面给出一个可直接编译运行的 BSH 调用示例。示例使用 BF16 的
`hiddenStates`、`wk`、`gateWeight` 和 `pooledKeyOut`，使用 FP32 的
`ape`、`stateCacheRef`、LayerNorm 权重和偏置，并通过空指针关闭当前尚未
实现的 RoPE。示例完整代码如下：

```cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A
 * PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_key_pool.cpp
 * \brief KeyPool aclnn invocation example for A2/A3.
 *
 * The example uses BSH input, BF16 projection/output tensors, FP32 state cache,
 * and enables the optional LayerNorm inputs. RoPE remains disabled by passing
 * null cos/sin tensors, as required by the current operator implementation.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_key_pool.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {

using BFloat16 = uint16_t;

BFloat16 FloatToBFloat16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<BFloat16>(bits >> 16);
}

float BFloat16ToFloat(BFloat16 value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (const auto dim : shape) {
        shapeSize *= dim;
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
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    const auto size = GetShapeSize(shape) * static_cast<int64_t>(sizeof(T));
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
    return ACL_SUCCESS;
}

void PrintBFloat16Tensor(const char *name, const std::vector<int64_t> &shape, void *deviceAddr)
{
    const auto elementCount = GetShapeSize(shape);
    std::vector<BFloat16> hostData(elementCount);
    const auto ret = aclrtMemcpy(hostData.data(), elementCount * sizeof(BFloat16), deviceAddr,
                                 elementCount * sizeof(BFloat16), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy %s from device to host failed. ERROR: %d\n", name, ret); return);
    const auto printCount = std::min<int64_t>(elementCount, 8);
    for (int64_t i = 0; i < printCount; ++i) {
        LOG_PRINT("%s[%ld] = %f\n", name, i, BFloat16ToFloat(hostData[i]));
    }
}

} // namespace

int main()
{
    constexpr int32_t deviceId = 0;
    constexpr int64_t batchSize = 1;
    constexpr int64_t seqLength = 8;
    constexpr int64_t hiddenSize = 4096;
    constexpr int64_t headDim = 128;
    constexpr int64_t cmpRatio = 4;
    constexpr int64_t blockSize = 4;
    constexpr int64_t maxBlockNumPerBatch = (seqLength + blockSize - 1) / blockSize;
    constexpr int64_t blockNum = batchSize * maxBlockNumPerBatch + 1; // block 0 is reserved.
    constexpr int64_t pooledSeqLength = (maxBlockNumPerBatch * blockSize + cmpRatio - 1) / cmpRatio;
    constexpr double normEps = 1e-6;
    constexpr int64_t rotaryMode = 1;
    constexpr int64_t stateCacheStrideDim0 = blockSize * 2 * headDim;

    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    const std::vector<int64_t> hiddenStatesShape = {batchSize, seqLength, hiddenSize};
    const std::vector<int64_t> wkShape = {headDim, hiddenSize};
    const std::vector<int64_t> gateWeightShape = {headDim, hiddenSize};
    const std::vector<int64_t> apeShape = {cmpRatio, headDim};
    const std::vector<int64_t> stateCacheShape = {blockNum, blockSize, 2 * headDim};
    const std::vector<int64_t> cacheBlockTableShape = {batchSize, maxBlockNumPerBatch};
    const std::vector<int64_t> startPosShape = {batchSize};
    const std::vector<int64_t> normShape = {headDim};
    const std::vector<int64_t> pooledKeyShape = {batchSize, pooledSeqLength, headDim};

    const auto hiddenStatesSize = GetShapeSize(hiddenStatesShape);
    const auto wkSize = GetShapeSize(wkShape);
    const auto gateWeightSize = GetShapeSize(gateWeightShape);
    const auto apeSize = GetShapeSize(apeShape);
    const auto stateCacheSize = GetShapeSize(stateCacheShape);
    const auto pooledKeySize = GetShapeSize(pooledKeyShape);

    std::vector<BFloat16> hiddenStatesHost(hiddenStatesSize);
    std::vector<BFloat16> wkHost(wkSize);
    std::vector<BFloat16> gateWeightHost(gateWeightSize);
    std::vector<float> apeHost(apeSize);
    std::vector<float> stateCacheHost(stateCacheSize, 0.0f);
    std::vector<int32_t> cacheBlockTableHost = {1, 2};
    std::vector<int32_t> startPosHost = {0};
    std::vector<float> normWeightHost(headDim, 1.0f);
    std::vector<float> normBiasHost(headDim, 0.0f);
    std::vector<BFloat16> pooledKeyHost(pooledKeySize, FloatToBFloat16(0.0f));

    for (int64_t i = 0; i < hiddenStatesSize; ++i) {
        hiddenStatesHost[i] = FloatToBFloat16(0.01f * std::sin(static_cast<float>(i % 97)));
    }
    for (int64_t i = 0; i < wkSize; ++i) {
        wkHost[i] = FloatToBFloat16(0.02f * std::cos(static_cast<float>(i % 53)));
        gateWeightHost[i] = FloatToBFloat16(0.02f * std::sin(static_cast<float>(i % 47)));
    }
    for (int64_t i = 0; i < apeSize; ++i) {
        apeHost[i] = 0.01f * static_cast<float>((i % cmpRatio) - 1);
    }

    void *hiddenStatesDeviceAddr = nullptr;
    void *wkDeviceAddr = nullptr;
    void *gateWeightDeviceAddr = nullptr;
    void *apeDeviceAddr = nullptr;
    void *stateCacheDeviceAddr = nullptr;
    void *cacheBlockTableDeviceAddr = nullptr;
    void *startPosDeviceAddr = nullptr;
    void *normWeightDeviceAddr = nullptr;
    void *normBiasDeviceAddr = nullptr;
    void *pooledKeyDeviceAddr = nullptr;

    aclTensor *hiddenStates = nullptr;
    aclTensor *wk = nullptr;
    aclTensor *gateWeight = nullptr;
    aclTensor *ape = nullptr;
    aclTensor *stateCacheRef = nullptr;
    aclTensor *cacheBlockTable = nullptr;
    aclTensor *startPos = nullptr;
    aclTensor *normWeight = nullptr;
    aclTensor *normBias = nullptr;
    aclTensor *pooledKeyOut = nullptr;

    ret = CreateAclTensor(hiddenStatesHost, hiddenStatesShape, &hiddenStatesDeviceAddr, ACL_BF16, &hiddenStates);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wkHost, wkShape, &wkDeviceAddr, ACL_BF16, &wk);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gateWeightHost, gateWeightShape, &gateWeightDeviceAddr, ACL_BF16, &gateWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(apeHost, apeShape, &apeDeviceAddr, ACL_FLOAT, &ape);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(stateCacheHost, stateCacheShape, &stateCacheDeviceAddr, ACL_FLOAT, &stateCacheRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cacheBlockTableHost, cacheBlockTableShape, &cacheBlockTableDeviceAddr, ACL_INT32,
                          &cacheBlockTable);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(startPosHost, startPosShape, &startPosDeviceAddr, ACL_INT32, &startPos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normWeightHost, normShape, &normWeightDeviceAddr, ACL_FLOAT, &normWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normBiasHost, normShape, &normBiasDeviceAddr, ACL_FLOAT, &normBias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(pooledKeyHost, pooledKeyShape, &pooledKeyDeviceAddr, ACL_BF16, &pooledKeyOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnKeyPoolGetWorkspaceSize(
        hiddenStates, wk, gateWeight, ape, stateCacheRef, cacheBlockTable, startPos, normWeight, normBias, nullptr,
        nullptr, nullptr, nullptr, cmpRatio, normEps, rotaryMode, stateCacheStrideDim0, pooledKeyOut, &workspaceSize,
        &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnKeyPoolGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnKeyPool(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKeyPool failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("KeyPool execution succeeded. pooled_key shape=[%ld,%ld,%ld]\n", batchSize, pooledSeqLength, headDim);
    PrintBFloat16Tensor("pooled_key", pooledKeyShape, pooledKeyDeviceAddr);

    aclDestroyTensor(hiddenStates);
    aclDestroyTensor(wk);
    aclDestroyTensor(gateWeight);
    aclDestroyTensor(ape);
    aclDestroyTensor(stateCacheRef);
    aclDestroyTensor(cacheBlockTable);
    aclDestroyTensor(startPos);
    aclDestroyTensor(normWeight);
    aclDestroyTensor(normBias);
    aclDestroyTensor(pooledKeyOut);

    aclrtFree(hiddenStatesDeviceAddr);
    aclrtFree(wkDeviceAddr);
    aclrtFree(gateWeightDeviceAddr);
    aclrtFree(apeDeviceAddr);
    aclrtFree(stateCacheDeviceAddr);
    aclrtFree(cacheBlockTableDeviceAddr);
    aclrtFree(startPosDeviceAddr);
    aclrtFree(normWeightDeviceAddr);
    aclrtFree(normBiasDeviceAddr);
    aclrtFree(pooledKeyDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ACL_SUCCESS;
}
```

当前 BSH 示例中 `cosOptional`、`sinOptional`、`cuSeqlensOptional` 和
`sequsedOptional` 均传空指针；TH 场景需要按输入约束创建并传入
`cuSeqlensOptional`。示例执行命令如下：

```bash
bash build.sh --run_example key_pool eager cust \
  --vendor_name=custom --soc=ascend910b --example_name=test_aclnn_key_pool

# A5
bash build.sh --run_example key_pool eager cust \
  --vendor_name=custom --soc=ascend950 --example_name=test_aclnn_key_pool
```

- [A2/A3 完整调用示例](../examples/test_aclnn_key_pool.cpp)
- [A5 完整调用示例](../examples/arch35/test_aclnn_key_pool.cpp)
