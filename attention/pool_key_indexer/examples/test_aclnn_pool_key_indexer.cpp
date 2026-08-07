/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_pool_key_indexer.cpp
 * \brief PoolKeyIndexer example test (from aclnn doc)
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_pool_key_indexer.h"

using namespace std;

namespace {
#define CHECK_RET(cond) ((cond) ? true : (false))
#define LOG_PRINT(message, ...) do { (void)printf(message, ##__VA_ARGS__); } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) { shapeSize *= i; }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    auto ret = aclInit(nullptr);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }
    ret = aclrtSetDevice(deviceId);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }
    ret = aclrtCreateStream(stream);
    return ret;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) { strides[i] = shape[i + 1] * strides[i + 1]; }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                              ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

struct TensorResources {
    void* queryDeviceAddr = nullptr;
    void* poolKeyDeviceAddr = nullptr;
    void* weightsDeviceAddr = nullptr;
    void* poolTailKDeviceAddr = nullptr;
    void* sparseIndicesDeviceAddr = nullptr;
    void* sparseValuesDeviceAddr = nullptr;
    aclTensor* queryTensor = nullptr;
    aclTensor* poolKeyTensor = nullptr;
    aclTensor* weightsTensor = nullptr;
    aclTensor* poolTailKTensor = nullptr;
    aclTensor* sparseIndicesTensor = nullptr;
    aclTensor* sparseValuesTensor = nullptr;
};

int InitializeTensors(TensorResources& resources) {
    // topk=2048, pool_size=16
    // sparseIndicesOut last dim = topk + pool_size - 1 = 2063
    // sparseValuesOut last dim = topk // pool_size = 128
    std::vector<int64_t> queryShape = {1, 2, 1, 128};
    std::vector<int64_t> poolKeyShape = {1, 2, 1, 128};
    std::vector<int64_t> weightsShape = {1, 2, 1};
    std::vector<int64_t> poolTailKShape = {1};
    std::vector<int64_t> sparseIndicesShape = {1, 2, 2063};
    std::vector<int64_t> sparseValuesShape = {1, 2, 128};

    int64_t querySize = GetShapeSize(queryShape);
    int64_t poolKeySize = GetShapeSize(poolKeyShape);
    int64_t weightsSize = GetShapeSize(weightsShape);
    int64_t poolTailKSize = GetShapeSize(poolTailKShape);
    int64_t sparseIndicesSize = GetShapeSize(sparseIndicesShape);
    int64_t sparseValuesSize = GetShapeSize(sparseValuesShape);

    std::vector<float> queryData(querySize, 1);
    std::vector<float> poolKeyData(poolKeySize, 1);
    std::vector<float> weightsData(weightsSize, 1);
    std::vector<int32_t> poolTailKData(poolTailKSize, 0);
    std::vector<int32_t> sparseIndicesData(sparseIndicesSize, 1);
    std::vector<float> sparseValuesData(sparseValuesSize, 1);

    int ret = CreateAclTensor(queryData, queryShape, &resources.queryDeviceAddr,
                              ACL_FLOAT16, &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;
    ret = CreateAclTensor(poolKeyData, poolKeyShape, &resources.poolKeyDeviceAddr,
                          ACL_FLOAT16, &resources.poolKeyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;
    ret = CreateAclTensor(weightsData, weightsShape, &resources.weightsDeviceAddr,
                          ACL_FLOAT16, &resources.weightsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;
    ret = CreateAclTensor(poolTailKData, poolTailKShape, &resources.poolTailKDeviceAddr,
                          ACL_INT32, &resources.poolTailKTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;
    ret = CreateAclTensor(sparseIndicesData, sparseIndicesShape, &resources.sparseIndicesDeviceAddr,
                          ACL_INT32, &resources.sparseIndicesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;
    ret = CreateAclTensor(sparseValuesData, sparseValuesShape, &resources.sparseValuesDeviceAddr,
                          ACL_FLOAT, &resources.sparseValuesTensor);
    return ret;
}

int ExecutePoolKeyIndexer(TensorResources& resources, aclrtStream stream,
                          void** workspaceAddr, uint64_t* workspaceSize) {
    int64_t topk = 2048;
    int64_t poolSize = 16;
    int64_t maskMode = 3;
    int64_t quantMode = -1;
    bool returnValue = true;
    constexpr const char layoutStr[] = "BSND";
    char layoutQuery[sizeof(layoutStr)];
    char layoutKey[sizeof(layoutStr)];
    errno_t memcpyRet = memcpy_s(layoutQuery, sizeof(layoutQuery), layoutStr, sizeof(layoutStr));
    if (!CHECK_RET(memcpyRet == 0)) return -1;
    memcpyRet = memcpy_s(layoutKey, sizeof(layoutKey), layoutStr, sizeof(layoutStr));
    if (!CHECK_RET(memcpyRet == 0)) return -1;

    aclOpExecutor* executor;
    int ret = aclnnPoolKeyIndexerGetWorkspaceSize(
        resources.queryTensor, resources.poolKeyTensor,
        resources.weightsTensor, resources.poolTailKTensor,
        nullptr, nullptr, nullptr, nullptr, nullptr,
        layoutQuery, layoutKey, topk, poolSize,
        maskMode, quantMode, returnValue,
        resources.sparseIndicesTensor, resources.sparseValuesTensor,
        workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;
    }
    ret = aclnnPoolKeyIndexer(*workspaceAddr, *workspaceSize, executor, stream);
    return ret;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr,
                      aclrtStream stream, int32_t deviceId) {
    if (resources.queryTensor) aclDestroyTensor(resources.queryTensor);
    if (resources.poolKeyTensor) aclDestroyTensor(resources.poolKeyTensor);
    if (resources.weightsTensor) aclDestroyTensor(resources.weightsTensor);
    if (resources.poolTailKTensor) aclDestroyTensor(resources.poolTailKTensor);
    if (resources.sparseIndicesTensor) aclDestroyTensor(resources.sparseIndicesTensor);
    if (resources.sparseValuesTensor) aclDestroyTensor(resources.sparseValuesTensor);
    if (resources.queryDeviceAddr) aclrtFree(resources.queryDeviceAddr);
    if (resources.poolKeyDeviceAddr) aclrtFree(resources.poolKeyDeviceAddr);
    if (resources.weightsDeviceAddr) aclrtFree(resources.weightsDeviceAddr);
    if (resources.poolTailKDeviceAddr) aclrtFree(resources.poolTailKDeviceAddr);
    if (resources.sparseIndicesDeviceAddr) aclrtFree(resources.sparseIndicesDeviceAddr);
    if (resources.sparseValuesDeviceAddr) aclrtFree(resources.sparseValuesDeviceAddr);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    if (stream) aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}
} // namespace

int main() {
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    TensorResources resources = {};
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;

    int ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;
    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { CleanupResources(resources, workspaceAddr, stream, deviceId); return ret; }
    ret = ExecutePoolKeyIndexer(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { CleanupResources(resources, workspaceAddr, stream, deviceId); return ret; }
    ret = aclrtSynchronizeStream(stream);
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return ret;
}
