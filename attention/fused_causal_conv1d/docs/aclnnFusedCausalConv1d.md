# aclnnFusedCausalConv1d

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/fused_causal_conv1d)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：对序列执行因果一维卷积，沿序列维度使用缓存数据（长度为卷积核宽减1）对各序列头部进行padding，确保输出依赖当前及历史输入；卷积完成后，将当前序列尾部的数据（长度为卷积核宽减1）更新到缓存；在因果一维卷积输出的基础上，将原始输入加到输出上以实现残差连接。<br>

- 支持以下场景：
  - 场景一（prefill场景）：

    ```
    x: [cu_seq_len, dim]
    weight: [K, dim]，其中K=3
    convStates: [-1, K-1, dim]
    queryStartLoc: [batch+1]
    cacheIndices: [batch]
    initialStateMode: [batch]
    bias: [dim]（无作用）
    numAcceptedTokens: [batch]（无作用）
    y: [cu_seq_len, dim]
    runMode: 0
    ```

    其中cu_seq_len为batch内所有变长序列拼接后的总长度。

  - 场景二（decode场景 - 变长序列）：

    ```
    x: [cu_seq_len, dim]
    weight: [K, dim]，其中K=3
    convStates: [-1, state_len, dim]
    queryStartLoc: [batch+1]
    cacheIndices: [batch]
    initialStateMode: [batch]
    bias: [dim]（无作用）
    numAcceptedTokens: [batch]（用于投机解码）
    y: [cu_seq_len, dim]
    runMode: 1
    ```

    其中state_len必须大于所有batch中最大的token个数加1。

  - 场景三（decode场景 - 固定batch）：
  
    ```
    x: [batch, m+1, dim]
    weight: [K, dim]，其中K=3
    convStates: [-1, K-1+m, dim]
    queryStartLoc: [batch+1]（无作用）
    cacheIndices: [batch]
    initialStateMode: [batch]
    bias: [dim]（无作用）
    numAcceptedTokens: [batch]（用于投机解码，m为投机token个数）
    y: [batch, m+1, dim]
    runMode: 1
    ```

- 计算公式：

  K是卷积核宽度（固定为3），L是原始序列长度，dim是特征维度。

  1. 缓存拼接：

    $$
    x'[i, dim] =
    \begin{cases}
    cacheState[i, dim], & 0 \leq i < K-1 \\
    x[i - (K-1), dim], & K-1 \leq i < L + K - 1
    \end{cases}
    $$

  2. 因果1维卷积：

    $$
    y[i, dim] = \sum_{k=0}^{K-1} w[k, dim] \cdot x'[i + k, dim]
    $$

  3. 缓存更新：

    $$
    cacheState[i, dim] = x'[L + i, dim], \quad i = 0, 1, \dots, K-2
    $$

  4. 残差连接（可选）：

    $$
    y[i, dim] += x[i, dim]
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用 `aclnnFusedCausalConv1dGetWorkspaceSize`接口获取入参并计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnFusedCausalConv1d`接口执行计算。

```Cpp
aclnnStatus aclnnFusedCausalConv1dGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *weight,
  aclTensor       *convStates,
  const aclTensor *queryStartLoc,
  const aclTensor *cacheIndices,
  const aclTensor *initialStateMode,
  const aclTensor *bias,
  const aclTensor *numAcceptedTokens,
  int64_t          activationMode,
  int64_t          padSlotId,
  int64_t          runMode,
  int64_t          residualConnection,
  const aclTensor *y,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnFusedCausalConv1d(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnFusedCausalConv1dGetWorkspaceSize

- **参数说明**：

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 240px">
  <col style="width: 100px">
  <col style="width: 333px">
  <col style="width: 400px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 107px">
  <col style="width: 145px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>计算公式中的x，代表输入序列</td>
      <td><ul><li>不支持空tensor。</li><li>prefill场景：shape为[cu_seq_len, dim]。</li><li>decode场景：shape为[cu_seq_len, dim]或[batch, seq_len, dim]。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight（aclTensor*）</td>
      <td>输入</td>
      <td>计算公式中的w，代表因果1维卷积核</td>
      <td><ul><li>不支持空tensor。</li><li>shape为[K, dim]。</li></ul></td>
      <td>数据类型与x一致</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>convStates（aclTensor*）</td>
      <td>输入/输出</td>
      <td>计算公式中的cacheState，代表缓存状态张量，存储各序列的历史token数据，各序列计算完成后原地更新。</td>
      <td><ul><li>不支持空tensor。</li><li>shape为[..., K-1, dim]</li></ul></td>
      <td>数据类型与x一致</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>queryStartLoc（aclTensor*）</td>
      <td>输入</td>
      <td>序列起始位置索引，记录各序列在拼接张量x中的起始位置。</td>
      <td><ul><li>不支持空tensor。</li><li>shape为[batch+1]</li><li>queryStartLoc[i]表示第i个序列的起始偏移</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cacheIndices（aclTensor*）</td>
      <td>输入</td>
      <td>缓存索引，指定每个序列对应的缓存状态在convStates中的索引。</td>
      <td><ul><li>不支持空tensor。</li><li>shape为[batch]。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>initialStateMode（aclTensor*）</td>
      <td>输入</td>
      <td>初始状态标志，表示各序列是否使用缓存数据</td>
      <td><ul><li>不支持空tensor。</li><li>shape为[batch]</li><li>取值为0、1、2<br>0：零填充<br>1：使用缓存<br>2：使用缓存但前K-1个输出置0。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>bias（aclTensor*）</td>
      <td>可选输入</td>
      <td>卷积的偏置</td>
      <td><ul><li>支持空tensor。</li><li>shape为[dim]。</li></ul></td>
      <td>数据类型与x一致</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>numAcceptedTokens（aclTensor*）</td>
      <td>可选输入</td>
      <td>decode场景下的投机token个数。</td>
      <td><ul><li>支持空tensor。<br>shape为[batch]。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>activationMode（int64_t）</td>
      <td>属性</td>
      <td>激活函数类型</td>
      <td><ul><li>取值为0、1、2<br>0：None<br>1：silu<br>2：swish</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padSlotId（int64_t）</td>
      <td>属性</td>
      <td>用于跳过不需要参与计算的batch</td>
      <td>当cacheIndices[i]==padSlotId时跳过该batch</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>runMode（int64_t）</td>
      <td>属性</td>
      <td>用于判断是prefill场景或decode场景</td>
      <td><ul><li>取值为0、1<br>0：prefill场景<br>1：decode场景</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>residualConnection（int64_t）</td>
      <td>属性</td>
      <td>是否做残差连接</td>
      <td><ul><li>取值为0、1<br>0：不做残差连接<br>1：输出y和输入x相加后输出</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y（aclTensor*）</td>
      <td>输出</td>
      <td>计算公式中的y，代表输出序列</td>
      <td>shape与x一致</td>
      <td>数据类型与x一致</td>
      <td>ND</td>
      <td>2-3</td>
      <td>x</td>
    </tr>
    <tr>
      <td>workspaceSize（int64_t*）</td>
      <td>输出</td>
      <td>返回用户需要在Device侧申请的workspace大小</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
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
      <td>传入的x、weight、convStates、y是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>输入和输出的数据类型不在支持的范围内。<br>x、weight、convStates、bias、y的数据类型不一致。<br>queryStartLoc、cacheIndices、initialStateMode、numAcceptedTokens的数据类型不一致。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>
      输入、输出Tensor的shape不在支持的范围内。<br>
      输入的属性不在支持的范围内。<br>
      dim不在指定的取值范围内。<br>
      </td>
    </tr>
  </tbody></table>

## aclnnFusedCausalConv1d

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnFusedCausalConv1dGetWorkspaceSize获取。</td>
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

- 确定性计算：
  - aclnnFusedCausalConv1d默认确定性实现。

- 输入shape限制：
  - prefill场景：
    - x支持2维[cu_seq_len, dim]。
    - weight必须是2维[K, dim]，其中K固定为3。
    - convStates必须是3维[..., K-1, dim]，第0维大小不固定且大于等于batch。
    - cu_seq_len范围[batch, 65536]，dim范围[128, 16384]且是128的倍数，batch范围[1, 256]。
  - decode场景（固定batch）：
    - x支持3维[batch, m+1, dim]。
    - weight必须是2维[K, dim]，其中K固定为3。
    - convStates必须是3维[..., K-1+m, dim]，第0维大小不固定且大于等于batch。
    - m范围[0, 5]，dim范围[128, 16384]且是128的倍数，batch范围[1, 256]。
  - decode场景（变长序列）：
    - x支持2维[cu_seq_len, dim]。
    - weight必须是2维[K, dim]，其中K固定为3。
    - convStates必须是3维[..., state_len, dim]，第0维大小不固定且大于等于batch，state_len必须大于所有batch中最大的token个数加1。
    - cu_seq_len范围[batch, batch*6]，每个batch的token个数范围为[1, 6]。dim范围[128, 16384]且是128的倍数，batch范围[1, 256]。

- 输入值域限制：
  - queryStartLoc是累计偏移量，取值范围[0, cu_seq_len]，长度为batch+1，queryStartLoc[i]表示第i个序列的起始偏移，queryStartLoc[batch+1]表示最后一个序列的结束位置。
  - cacheIndices长度为batch，指定每个序列对应的缓存槽索引。
  - numAcceptedTokens分为None和非None，非None情况下长度为batch，每个元素取值不超过当前batch的token个数且大于0。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file test_aclnn_fused_causal_conv1d.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_fused_causal_conv1d.h"

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

int Init(int32_t deviceId, aclrtStream* stream) {
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor, aclFormat format) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t K = 3;
    int64_t dim = 128;
    int64_t batch = 4;
    int64_t numSlots = 8;
    // prefill场景: seq_lens = [5, 3, 7, 4], cu_seq_len = 19
    int64_t cuSeqLen = 19;
    int64_t stateLen = K - 1; // 2

    std::vector<int64_t> xShape = {cuSeqLen, dim};
    std::vector<int64_t> weightShape = {K, dim};
    std::vector<int64_t> convStatesShape = {numSlots, stateLen, dim};
    std::vector<int64_t> queryStartLocShape = {batch + 1};
    std::vector<int64_t> cacheIndicesShape = {batch};
    std::vector<int64_t> initialStateModeShape = {batch};
    std::vector<int64_t> biasShape = {dim};
    std::vector<int64_t> numAcceptedTokensShape = {batch};
    std::vector<int64_t> yShape = {cuSeqLen, dim};

    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* convStatesDeviceAddr = nullptr;
    void* queryStartLocDeviceAddr = nullptr;
    void* cacheIndicesDeviceAddr = nullptr;
    void* initialStateModeDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* numAcceptedTokensDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* convStates = nullptr;
    aclTensor* queryStartLoc = nullptr;
    aclTensor* cacheIndices = nullptr;
    aclTensor* initialStateMode = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* numAcceptedTokens = nullptr;
    aclTensor* y = nullptr;

    // 初始化host数据
    std::vector<op::fp16_t> hostX(cuSeqLen * dim, 1.0f);
    std::vector<op::fp16_t> hostWeight(K * dim, 1.0f);
    std::vector<op::fp16_t> hostConvStates(numSlots * stateLen * dim, 0.0f);
    // query_start_loc = [0, 5, 8, 15, 19] (累计偏移量)
    std::vector<int32_t> hostQueryStartLoc = {0, 5, 8, 15, 19};
    // cache_indices = [0, 3, 1, 5]
    std::vector<int32_t> hostCacheIndices = {0, 3, 1, 5};
    // initial_state_mode = [1, 0, 2, 1]
    std::vector<int32_t> hostInitialStateMode = {1, 0, 2, 1};
    std::vector<op::fp16_t> hostBias(dim, 0);
    std::vector<int32_t> hostNumAcceptedTokens(batch, 0);
    std::vector<op::fp16_t> hostY(cuSeqLen * dim, 0.0f);

    // 创建x aclTensor
    ret = CreateAclTensor(hostX, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建weight aclTensor
    ret = CreateAclTensor(hostWeight, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建convStates aclTensor
    ret = CreateAclTensor(hostConvStates, convStatesShape, &convStatesDeviceAddr, aclDataType::ACL_FLOAT16, &convStates, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建queryStartLoc aclTensor
    ret = CreateAclTensor(hostQueryStartLoc, queryStartLocShape, &queryStartLocDeviceAddr, aclDataType::ACL_INT32, &queryStartLoc, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建cacheIndices aclTensor
    ret = CreateAclTensor(hostCacheIndices, cacheIndicesShape, &cacheIndicesDeviceAddr, aclDataType::ACL_INT32, &cacheIndices, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建initialStateMode aclTensor
    ret = CreateAclTensor(hostInitialStateMode, initialStateModeShape, &initialStateModeDeviceAddr, aclDataType::ACL_INT32, &initialStateMode, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建bias aclTensor
    ret = CreateAclTensor(hostBias, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建numAcceptedTokens aclTensor
    ret = CreateAclTensor(hostNumAcceptedTokens, numAcceptedTokensShape, &numAcceptedTokensDeviceAddr, aclDataType::ACL_INT32, &numAcceptedTokens, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建y aclTensor
    ret = CreateAclTensor(hostY, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &y, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 参数设置
    int64_t activationMode = 0;     // 0: None
    int64_t padSlotId = -1;         // -1: 不跳过
    int64_t runMode = 0;            // 0: prefill场景
    int64_t residualConnection = 1; // 1: 做残差连接

    // 调用aclnnFusedCausalConv1d第一段接口
    ret = aclnnFusedCausalConv1dGetWorkspaceSize(
        x, weight, convStates, queryStartLoc, cacheIndices, initialStateMode,
        bias, numAcceptedTokens, activationMode, padSlotId, runMode, residualConnection,
        y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedCausalConv1dGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnFusedCausalConv1d第二段接口
    ret = aclnnFusedCausalConv1d(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedCausalConv1d failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(yShape);
    std::vector<op::fp16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("First 10 output values:\n");
    for (int64_t i = 0; i < 10; i++) {
        std::cout << "index: " << i << ": " << static_cast<float>(resultData[i]) << std::endl;
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(convStates);
    aclDestroyTensor(queryStartLoc);
    aclDestroyTensor(cacheIndices);
    aclDestroyTensor(initialStateMode);
    aclDestroyTensor(bias);
    aclDestroyTensor(numAcceptedTokens);
    aclDestroyTensor(y);

    // 7. 释放device资源，需要根据具体API的接口定义参数
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(convStatesDeviceAddr);
    aclrtFree(queryStartLocDeviceAddr);
    aclrtFree(cacheIndicesDeviceAddr);
    aclrtFree(initialStateModeDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(numAcceptedTokensDeviceAddr);
    aclrtFree(yDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("Test completed successfully!\n");
    return 0;
}
```
