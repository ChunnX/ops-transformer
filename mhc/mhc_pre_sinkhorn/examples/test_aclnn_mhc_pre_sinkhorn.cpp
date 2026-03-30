/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn/aclnn_ext.h"

#define ACL_RETCODE_LOG_ERROR(msg) \
    do { \
        std::cerr << "[ERROR] " << msg << " retCode=" << retCode << std::endl; \
        return 1; \
    } while (0)

#define ACL_CHECK_NULL(ptr, msg) \
    do { \
        if ((ptr) == nullptr) { \
            std::cerr << "[ERROR] " msg " is null" << std::endl; \
            return 1; \
        } \
    } while (0)

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <deviceId>" << std::endl;
        return 1;
    }

    int32_t deviceId = std::stoi(argv[1]);
    
    aclError retCode = aclInit(nullptr);
    if (retCode != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclInit failed, retCode=" << retCode << std::endl;
        return 1;
    }

    retCode = aclrtSetDevice(deviceId);
    if (retCode != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtSetDevice failed, retCode=" << retCode << std::endl;
        aclFinalize();
        return 1;
    }

    aclrtContext context = nullptr;
    retCode = aclrtCreateContext(&context, deviceId);
    if (retCode != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtCreateContext failed, retCode=" << retCode << std::endl;
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    aclrtStream stream = nullptr;
    retCode = aclrtCreateStream(&stream);
    if (retCode != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtCreateStream failed, retCode=" << retCode << std::endl;
        aclrtDestroyContext(context);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    const int64_t T = 8;
    const int64_t n = 4;
    const int64_t numIters = 20;
    const float eps = 1e-6f;
    const int outFlag = 0;

    std::vector<int64_t> hResShape = {T, n, n};
    int64_t hResSize = T * n * n * sizeof(float);

    std::vector<float> hResHost(T * n * n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &val : hResHost) {
        val = dist(gen);
    }

    aclTensor *hResTensor = nullptr;
    aclTensor *hResSinkhornTensor = nullptr;
    aclTensor *normOutTensor = nullptr;
    aclTensor *sumOutTensor = nullptr;

    aclTensorDesc *hResDesc = aclCreateTensorDesc(aclDataType::ACL_FLOAT, hResShape.size(), hResShape.data(),
                                             aclFormat::ACL_FORMAT_ND);
    ACL_CHECK_NULL(hResDesc, "hResDesc");

    aclTensorDesc *hResSinkhornDesc = aclCreateTensorDesc(aclDataType::ACL_FLOAT, hResShape.size(),
                                                       hResShape.data(), aclFormat::ACL_FORMAT_ND);
    ACL_CHECK_NULL(hResSinkhornDesc, "hResSinkhornDesc");

    aclDataBuffer *hResBuffer = aclCreateDataBuffer(hResSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ACL_CHECK_NULL(hResBuffer, "hResBuffer");

    aclDataBuffer *hResSinkhornBuffer = aclCreateDataBuffer(hResSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ACL_CHECK_NULL(hResSinkhornBuffer, "hResSinkhornBuffer");

    retCode = aclrtMemcpy(hResBuffer->data, hResSize, hResHost.data(), hResSize,
                          ACL_MEMCPY_HOST_TO_DEVICE);
    if (retCode != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMemcpy failed, retCode=" << retCode << std::endl;
        aclDestroyDataBuffer(hResSinkhornBuffer);
        aclDestroyDataBuffer(hResBuffer);
        aclDestroyTensorDesc(hResSinkhornDesc);
        aclDestroyTensorDesc(hResDesc);
        aclrtDestroyStream(stream);
        aclrtDestroyContext(context);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    hResTensor = aclCreateTensorFromBuffer(hResDesc, hResBuffer, hResSize);
    ACL_CHECK_NULL(hResTensor, "hResTensor");

    hResSinkhornTensor = aclCreateTensorFromBuffer(hResSinkhornDesc, hResSinkhornBuffer, hResSize);
    ACL_CHECK_NULL(hResSinkhornTensor, "hResSinkhornTensor");

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclnnRet = aclnnMhcPreSinkhornGetWorkspaceSize(
        hResTensor, eps, numIters, outFlag, hResSinkhornTensor, normOutTensor, sumOutTensor,
        &workspaceSize, &executor);
    if (aclnnRet != ACLNN_SUCCESS) {
        std::cerr << "[ERROR] aclnnMhcPreSinkhornGetWorkspaceSize failed, retCode=" << aclnnRet << std::endl;
        aclDestroyTensor(hResSinkhornTensor);
        aclDestroyTensor(hResTensor);
        aclDestroyDataBuffer(hResSinkhornBuffer);
        aclDestroyDataBuffer(hResBuffer);
        aclDestroyTensorDesc(hResSinkhornDesc);
        aclDestroyTensorDesc(hResDesc);
        aclrtDestroyStream(stream);
        aclrtDestroyContext(context);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        retCode = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (retCode != ACL_SUCCESS) {
            std::cerr << "[ERROR] aclrtMalloc failed, retCode=" << retCode << std::endl;
            aclDestroyOpExecutor(executor);
            aclDestroyTensor(hResSinkhornTensor);
            aclDestroyTensor(hResTensor);
            aclDestroyDataBuffer(hResSinkhornBuffer);
            aclDestroyDataBuffer(hResBuffer);
            aclDestroyTensorDesc(hResSinkhornDesc);
            aclDestroyTensorDesc(hResDesc);
            aclrtDestroyStream(stream);
            aclrtDestroyContext(context);
            aclrtResetDevice(deviceId);
            aclFinalize();
            return 1;
        }
    }

    aclnnRet = aclnnMhcPreSinkhorn(workspace, workspaceSize, executor, stream);
    if (aclnnRet != ACLNN_SUCCESS) {
        std::cerr << "[ERROR] aclnnMhcPreSinkhorn failed, retCode=" << aclnnRet << std::endl;
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
        aclDestroyOpExecutor(executor);
        aclDestroyTensor(hResSinkhornTensor);
        aclDestroyTensor(hResTensor);
        aclDestroyDataBuffer(hResSinkhornBuffer);
        aclDestroyDataBuffer(hResBuffer);
        aclDestroyTensorDesc(hResSinkhornDesc);
        aclDestroyTensorDesc(hResDesc);
        aclrtDestroyStream(stream);
        aclrtDestroyContext(context);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    retCode = aclrtSynchronizeStream(stream);
    if (retCode != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtSynchronizeStream failed, retCode=" << retCode << std::endl;
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
        aclDestroyOpExecutor(executor);
        aclDestroyTensor(hResSinkhornTensor);
        aclDestroyTensor(hResTensor);
        aclDestroyDataBuffer(hResSinkhornBuffer);
        aclDestroyDataBuffer(hResBuffer);
        aclDestroyTensorDesc(hResSinkhornDesc);
        aclDestroyTensorDesc(hResDesc);
        aclrtDestroyStream(stream);
        aclrtDestroyContext(context);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    std::vector<float> hResSinkhornHost(T * n * n);
    retCode = aclrtMemcpy(hResSinkhornHost.data(), hResSize, hResSinkhornBuffer->data, hResSize,
                          ACL_MEMCPY_DEVICE_TO_HOST);
    if (retCode != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMemcpy failed, retCode=" << retCode << std::endl;
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
        aclDestroyOpExecutor(executor);
        aclDestroyTensor(hResSinkhornTensor);
        aclDestroyTensor(hResTensor);
        aclDestroyDataBuffer(hResSinkhornBuffer);
        aclDestroyDataBuffer(hResBuffer);
        aclDestroyTensorDesc(hResSinkhornDesc);
        aclDestroyTensorDesc(hResDesc);
        aclrtDestroyStream(stream);
        aclrtDestroyContext(context);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    std::cout << "MhcPreSinkhorn test passed!" << std::endl;
    std::cout << "Input shape: [" << T << ", " << n << ", " << n << "]" << std::endl;
    std::cout << "Output shape: [" << T << ", " << n << ", " << n << "]" << std::endl;
    std::cout << "First 4 output values: ";
    for (int i = 0; i < 4 && i < T * n * n; i++) {
        std::cout << hResSinkhornHost[i] << " ";
    }
    std::cout << std::endl;

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclDestroyOpExecutor(executor);
    aclDestroyTensor(hResSinkhornTensor);
    aclDestroyTensor(hResTensor);
    aclDestroyDataBuffer(hResSinkhornBuffer);
    aclDestroyDataBuffer(hResBuffer);
    aclDestroyTensorDesc(hResSinkhornDesc);
    aclDestroyTensorDesc(hResDesc);
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
