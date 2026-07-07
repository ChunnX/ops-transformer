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
 * \file test_aclnn_quant_lightning_indexer_v2.cpp
 * \brief
 */
//testci
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_lightning_indexer_v2.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true :(false))

#define LOG_PRINT(message, ...)     \
  do {                              \
    (void)printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
    return ret;
  }
  ret = aclrtSetDevice(deviceId);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
    return ret;
  }
  ret = aclrtCreateStream(stream);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
    return ret;
  }
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
    return ret;
  }

  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
    return ret;
  }

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

struct TensorResources {
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* weightsDeviceAddr = nullptr;
    void* qScaleDeviceAddr = nullptr;
    void* kScaleDeviceAddr = nullptr;
    void* sparseIndicesDeviceAddr = nullptr;
    void* sparseValuesDeviceAddr = nullptr;

    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    aclTensor* weightsTensor = nullptr;
    aclTensor* qScaleTensor = nullptr;
    aclTensor* kScaleTensor = nullptr;
    aclTensor* sparseIndicesTensor = nullptr;
    aclTensor* sparseValuesTensor = nullptr;
};

int InitializeTensors(TensorResources& resources) {
    int64_t b = 2, s1 = 4, n1 = 64, n2 = 1, d = 128;
    int64_t blockSize = 128, blockNum = 4;
    int64_t topk = 512;

    std::vector<int64_t> queryShape = {b, s1, n1, d};
    std::vector<int64_t> keyShape = {blockNum, blockSize, n2, d};
    std::vector<int64_t> weightsShape = {b, s1, n1};
    std::vector<int64_t> qScaleShape = {b, s1, n1};
    std::vector<int64_t> kScaleShape = {blockNum, blockSize, n2};
    std::vector<int64_t> sparseIndicesShape = {b, s1, n2, topk};
    std::vector<int64_t> sparseValuesShape = {b, s1, n2, topk};

    int64_t queryShapeSize = GetShapeSize(queryShape);
    int64_t keyShapeSize = GetShapeSize(keyShape);
    int64_t weightsShapeSize = GetShapeSize(weightsShape);
    int64_t qScaleShapeSize = GetShapeSize(qScaleShape);
    int64_t kScaleShapeSize = GetShapeSize(kScaleShape);
    int64_t sparseIndicesShapeSize = GetShapeSize(sparseIndicesShape);
    int64_t sparseValuesShapeSize = GetShapeSize(sparseValuesShape);

    std::vector<int8_t> queryHostData(queryShapeSize, 1);
    std::vector<int8_t> keyHostData(keyShapeSize, 1);
    std::vector<aclFloat16> weightsHostData(weightsShapeSize, aclFloatToFloat16(0.01f));
    std::vector<aclFloat16> qScaleHostData(qScaleShapeSize, aclFloatToFloat16(1.0f));
    std::vector<aclFloat16> kScaleHostData(kScaleShapeSize, aclFloatToFloat16(1.0f));
    std::vector<int32_t> sparseIndicesHostData(sparseIndicesShapeSize, 0);
    std::vector<aclBFloat16> sparseValuesHostData(sparseValuesShapeSize, aclFloatToBFloat16(0.0f));

    int ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr,
                              aclDataType::ACL_INT8, &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }

    ret = CreateAclTensor(keyHostData, keyShape, &resources.keyDeviceAddr,
                          aclDataType::ACL_INT8, &resources.keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }

    ret = CreateAclTensor(weightsHostData, weightsShape, &resources.weightsDeviceAddr,
                          aclDataType::ACL_FLOAT16, &resources.weightsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }

    ret = CreateAclTensor(qScaleHostData, qScaleShape, &resources.qScaleDeviceAddr,
                          aclDataType::ACL_FLOAT16, &resources.qScaleTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }

    ret = CreateAclTensor(kScaleHostData, kScaleShape, &resources.kScaleDeviceAddr,
                          aclDataType::ACL_FLOAT16, &resources.kScaleTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }

    ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape, &resources.sparseIndicesDeviceAddr,
                          aclDataType::ACL_INT32, &resources.sparseIndicesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }

    ret = CreateAclTensor(sparseValuesHostData, sparseValuesShape, &resources.sparseValuesDeviceAddr,
                          aclDataType::ACL_BF16, &resources.sparseValuesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }

    return ACL_SUCCESS;
}

int ExecuteQuantLightningIndexerV2(TensorResources& resources, aclrtStream stream,
                                   void** workspaceAddr, uint64_t* workspaceSize) {
    int64_t topk = 512;
    int64_t quantMode = 0;
    int64_t maskMode = 0;
    int64_t cmpRatio = 1;
    int64_t returnValue = 1;
    constexpr const char layerOutQStr[] = "BSND";
    constexpr const char layerOutKStr[] = "PA_BSND";
    constexpr size_t layerOutQLen = sizeof(layerOutQStr);
    constexpr size_t layerOutKLen = sizeof(layerOutKStr);
    char layoutQ[layerOutQLen];
    char layoutK[layerOutKLen];
    errno_t memcpyRet = memcpy_s(layoutQ, sizeof(layoutQ), layerOutQStr, layerOutQLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutQ failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    memcpyRet = memcpy_s(layoutK, sizeof(layoutK), layerOutKStr, layerOutKLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutK failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    aclOpExecutor* executor;

    int ret = aclnnQuantLightningIndexerV2GetWorkspaceSize(
        resources.queryTensor, resources.keyTensor, resources.weightsTensor,
        resources.qScaleTensor, resources.kScaleTensor,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        topk, quantMode, -1, layoutQ, layoutK, maskMode, cmpRatio, returnValue,
        resources.sparseIndicesTensor, resources.sparseValuesTensor,
        workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnQuantLightningIndexerV2GetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnQuantLightningIndexerV2(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnQuantLightningIndexerV2 failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr,
                     aclrtStream stream, int32_t deviceId) {
    if (resources.queryTensor) { aclDestroyTensor(resources.queryTensor); }
    if (resources.keyTensor) { aclDestroyTensor(resources.keyTensor); }
    if (resources.weightsTensor) { aclDestroyTensor(resources.weightsTensor); }
    if (resources.qScaleTensor) { aclDestroyTensor(resources.qScaleTensor); }
    if (resources.kScaleTensor) { aclDestroyTensor(resources.kScaleTensor); }
    if (resources.sparseIndicesTensor) { aclDestroyTensor(resources.sparseIndicesTensor); }
    if (resources.sparseValuesTensor) { aclDestroyTensor(resources.sparseValuesTensor); }

    if (resources.queryDeviceAddr) { aclrtFree(resources.queryDeviceAddr); }
    if (resources.keyDeviceAddr) { aclrtFree(resources.keyDeviceAddr); }
    if (resources.weightsDeviceAddr) { aclrtFree(resources.weightsDeviceAddr); }
    if (resources.qScaleDeviceAddr) { aclrtFree(resources.qScaleDeviceAddr); }
    if (resources.kScaleDeviceAddr) { aclrtFree(resources.kScaleDeviceAddr); }
    if (resources.sparseIndicesDeviceAddr) { aclrtFree(resources.sparseIndicesDeviceAddr); }
    if (resources.sparseValuesDeviceAddr) { aclrtFree(resources.sparseValuesDeviceAddr); }

    if (workspaceAddr) { aclrtFree(workspaceAddr); }
    if (stream) { aclrtDestroyStream(stream); }
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
    int ret = ACL_SUCCESS;

    // 1. Initialize device and stream
    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // 2. Initialize tensors
    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 3. Execute the operation
    ret = ExecuteQuantLightningIndexerV2(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 4. Synchronize stream
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 5. Cleanup resources
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
