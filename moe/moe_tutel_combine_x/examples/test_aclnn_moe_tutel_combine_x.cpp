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
 * \file test_aclnn_moe_tutel_combine_x.cpp
 * \brief ACLNN calling example of the MoeTutelCombineX operator.
 *
 * No-batch scenario: y_grad [E*C, H], gates/indices/locations [S],
 * x_grad [S, H].  x_grad[t, :] = gates[t] * y_grad[indices[t]*C + locations[t], :].
 */
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_tutel_combine_x.h"

#define CHECK_RET(cond, return_expr)     do {         if (!(cond)) {             return_expr;         }     } while (0)

#define LOG_PRINT(message, ...)     do {         printf(message, ##__VA_ARGS__);     } while (0)

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
                    aclDataType dataType, aclTensor **tensor)
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. device/stream initialization
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. build inputs and output.
    // y_grad: [E*C, H] = [4, 2] with E=2 experts, capacity C=2, hidden H=2
    std::vector<int64_t> yGradShape = {4, 2};
    std::vector<float> yGradHostData = {1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f, 4.0f, 40.0f};
    // gates / indices / locations: [S] = [4]
    std::vector<int64_t> gatesShape = {4};
    std::vector<float> gatesHostData = {0.1f, 0.2f, 0.3f, 0.4f};
    std::vector<int32_t> indicesHostData = {0, 1, 1, 0};
    std::vector<int32_t> locationsHostData = {1, 0, 1, 0};
    // x_grad: [S, H] = [4, 2]
    std::vector<int64_t> xGradShape = {4, 2};
    std::vector<float> xGradHostData(8, 0.0f);

    void *yGradDeviceAddr = nullptr;
    void *gatesDeviceAddr = nullptr;
    void *indicesDeviceAddr = nullptr;
    void *locationsDeviceAddr = nullptr;
    void *xGradDeviceAddr = nullptr;
    aclTensor *yGrad = nullptr;
    aclTensor *gates = nullptr;
    aclTensor *indices = nullptr;
    aclTensor *locations = nullptr;
    aclTensor *xGrad = nullptr;

    ret = CreateAclTensor(yGradHostData, yGradShape, &yGradDeviceAddr, aclDataType::ACL_FLOAT, &yGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gatesHostData, gatesShape, &gatesDeviceAddr, aclDataType::ACL_FLOAT, &gates);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(indicesHostData, gatesShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(locationsHostData, gatesShape, &locationsDeviceAddr, aclDataType::ACL_INT32, &locations);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(xGradHostData, xGradShape, &xGradDeviceAddr, aclDataType::ACL_FLOAT, &xGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. two-phase aclnn call
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnMoeTutelCombineXGetWorkspaceSize(yGrad, gates, indices, locations, xGrad, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTutelCombineXGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnMoeTutelCombineX(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTutelCombineX failed. ERROR: %d\n", ret); return ret);

    // 4. wait for completion
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. copy the result back
    auto size = GetShapeSize(xGradShape);
    std::vector<float> resultData(size, 0.0f);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), xGradDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("xGrad[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. release resources
    aclDestroyTensor(yGrad);
    aclDestroyTensor(gates);
    aclDestroyTensor(indices);
    aclDestroyTensor(locations);
    aclDestroyTensor(xGrad);
    aclrtFree(yGradDeviceAddr);
    aclrtFree(gatesDeviceAddr);
    aclrtFree(indicesDeviceAddr);
    aclrtFree(locationsDeviceAddr);
    aclrtFree(xGradDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
