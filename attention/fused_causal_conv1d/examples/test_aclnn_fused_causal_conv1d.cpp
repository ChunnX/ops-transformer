/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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