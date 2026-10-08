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
 * \file test_aclnn_mhc_norm_post_preonly.cpp
 * \brief ACLNN sample for MhcNormPostPreonly.
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_norm_post_preonly.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)
#define LOG_PRINT(message, ...) printf(message, ##__VA_ARGS__)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

struct TensorResource {
    void *deviceAddr = nullptr;
    aclTensor *tensor = nullptr;
};

int CreateTensor(const std::vector<int64_t> &shape, aclDataType dtype, bool initialize, TensorResource &resource)
{
    const int64_t elementCount = GetShapeSize(shape);
    const size_t elementSize = dtype == ACL_FLOAT ? sizeof(float) : sizeof(aclFloat16);
    const size_t byteSize = elementCount * elementSize;
    aclError ret = aclrtMalloc(&resource.deviceAddr, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return -1);

    if (initialize) {
        if (dtype == ACL_FLOAT) {
            std::vector<float> hostData(elementCount, 1.0f);
            ret = aclrtMemcpy(resource.deviceAddr, byteSize, hostData.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
        } else {
            std::vector<aclFloat16> hostData(elementCount, aclFloat16(1.0f));
            ret = aclrtMemcpy(resource.deviceAddr, byteSize, hostData.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
        }
        CHECK_RET(ret == ACL_SUCCESS, return -1);
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    resource.tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                                      shape.size(), resource.deviceAddr);
    CHECK_RET(resource.tensor != nullptr, return -1);
    return 0;
}

void DestroyTensor(TensorResource &resource)
{
    if (resource.tensor != nullptr) {
        aclDestroyTensor(resource.tensor);
    }
    if (resource.deviceAddr != nullptr) {
        aclrtFree(resource.deviceAddr);
    }
}

int main()
{
    constexpr int32_t deviceId = 0;
    constexpr int64_t T = 1;
    constexpr int64_t N = 4;
    constexpr int64_t D = 2560;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    CHECK_RET(aclInit(nullptr) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtSetDevice(deviceId) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtCreateContext(&context, deviceId) == ACL_SUCCESS, return -1);
    CHECK_RET(aclrtCreateStream(&stream) == ACL_SUCCESS, return -1);

    TensorResource hOut, residual, hPost, hResOptional, phi, alpha0, bias, gamma1, gamma0;
    TensorResource hInPrimeOut, x2Out;
    CHECK_RET(CreateTensor({T, D}, ACL_FLOAT16, true, hOut) == 0, return -1);
    CHECK_RET(CreateTensor({T, N, D}, ACL_FLOAT16, true, residual) == 0, return -1);
    CHECK_RET(CreateTensor({T, N}, ACL_FLOAT, true, hPost) == 0, return -1);
    CHECK_RET(CreateTensor({T, N, N}, ACL_FLOAT, true, hResOptional) == 0, return -1);
    CHECK_RET(CreateTensor({N, N * D}, ACL_FLOAT, true, phi) == 0, return -1);
    CHECK_RET(CreateTensor({1}, ACL_FLOAT, true, alpha0) == 0, return -1);
    CHECK_RET(CreateTensor({N}, ACL_FLOAT, true, bias) == 0, return -1);
    CHECK_RET(CreateTensor({D}, ACL_FLOAT, true, gamma1) == 0, return -1);
    CHECK_RET(CreateTensor({D}, ACL_FLOAT, true, gamma0) == 0, return -1);
    CHECK_RET(CreateTensor({T, D}, ACL_FLOAT16, false, hInPrimeOut) == 0, return -1);
    CHECK_RET(CreateTensor({T, N, D}, ACL_FLOAT16, false, x2Out) == 0, return -1);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnMhcNormPostPreonlyGetWorkspaceSize(
        hOut.tensor, residual.tensor, hPost.tensor, hResOptional.tensor, phi.tensor, alpha0.tensor, bias.tensor,
        gamma1.tensor, gamma0.tensor, nullptr, 1e-6, 1e-6, 1e-6, false, hInPrimeOut.tensor, x2Out.tensor, nullptr,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACLNN_SUCCESS, LOG_PRINT("GetWorkspaceSize failed, error: %d\n", ret); return -1);

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_RET(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return -1);
    }
    ret = aclnnMhcNormPostPreonly(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACLNN_SUCCESS, LOG_PRINT("Execute failed, error: %d\n", ret); return -1);
    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, return -1);
    LOG_PRINT("MhcNormPostPreonly compute success!\n");

    for (TensorResource *resource :
         {&hOut, &residual, &hPost, &hResOptional, &phi, &alpha0, &bias, &gamma1, &gamma0, &hInPrimeOut, &x2Out}) {
        DestroyTensor(*resource);
    }
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
