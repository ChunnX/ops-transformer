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
 * \file test_kv_quant_sparse_flash_attention_pioneer.cpp
 * \brief
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_kv_quant_sparse_flash_attention_pioneer.h"

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
    void* valueDeviceAddr = nullptr;
    void* sparseIndicesDeviceAddr = nullptr;
    void* keyDequantScaleDeviceAddr = nullptr;
    void* valueDequantScaleDeviceAddr = nullptr;
    void* attentionOutDeviceAddr = nullptr;

    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    aclTensor* valueTensor = nullptr;
    aclTensor* sparseIndicesTensor = nullptr;
    aclTensor* keyDequantScaleTensor = nullptr;
    aclTensor* valueDequantScaleTensor = nullptr;
    aclTensor* attentionOutTensor = nullptr;

};

int InitializeTensors(TensorResources& resources) {
    std::vector<int64_t> queryShape = {1, 1, 64, 576};
    std::vector<int64_t> keyShape = {1, 1, 1, 656};
    std::vector<int64_t> valueShape = {1, 1, 1, 656};
    std::vector<int64_t> sparseIndicesShape = {1, 1, 1, 2048 };
    std::vector<int64_t> keyDequantScaleShape = {1, 1, 1, 4};
    std::vector<int64_t> valueDequantScaleShape = {1, 1, 1, 4};
    std::vector<int64_t> attentionOutShape = {1, 1, 64, 512};

    int64_t queryShapeSize = GetShapeSize(queryShape);
    int64_t keyShapeSize = GetShapeSize(keyShape);
    int64_t valueShapeSize = GetShapeSize(valueShape);
    int64_t sparseIndicesShapeSize = GetShapeSize(sparseIndicesShape);
    int64_t keyDequantScaleShapeSize = GetShapeSize(keyDequantScaleShape);
    int64_t valueDequantScaleShapeSize = GetShapeSize(valueDequantScaleShape);
    int64_t attentionOutShapeSize = GetShapeSize(attentionOutShape);

    std::vector<aclFloat16> queryHostData(queryShapeSize, aclFloat16(1.0f));
    std::vector<uint8_t> keyHostData(keyShapeSize, 1);
    std::vector<uint8_t> valueHostData(valueShapeSize, 1);
    std::vector<int32_t> sparseIndicesHostData(sparseIndicesShapeSize, 1);
    std::vector<float> keyDequantScaleHostData(keyDequantScaleShapeSize, 1.0f);
    std::vector<float> valueDequantScaleHostData(valueDequantScaleShapeSize, 1.0f);
    std::vector<aclFloat16> attentionOutHostData(attentionOutShapeSize, aclFloat16(1.0f));

    int ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr,
                              aclDataType::ACL_FLOAT16, &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(keyHostData, keyShape, &resources.keyDeviceAddr,
                          aclDataType::ACL_FLOAT8_E4M3FN, &resources.keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(valueHostData, valueShape, &resources.valueDeviceAddr,
                          aclDataType::ACL_FLOAT8_E4M3FN, &resources.valueTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape, &resources.sparseIndicesDeviceAddr,
                          aclDataType::ACL_INT32, &resources.sparseIndicesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(keyDequantScaleHostData, keyDequantScaleShape, &resources.keyDequantScaleDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.keyDequantScaleTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(valueDequantScaleHostData, valueDequantScaleShape, &resources.valueDequantScaleDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.valueDequantScaleTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &resources.attentionOutDeviceAddr,
                          aclDataType::ACL_FLOAT16, &resources.attentionOutTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    return ACL_SUCCESS;
}

int ExecuteKvQuantSparseFlashAttentionPioneer(TensorResources& resources, aclrtStream stream,
                              void** workspaceAddr, uint64_t* workspaceSize) {
    int64_t keyQuantMode = 2;
    int64_t valueQuantMode = 2;
    int64_t sparseMode = 3;
    int64_t preTokens = 9223372036854775807;
    int64_t nextTokens = 9223372036854775807;
    double scaleValue = 1.0/24.0;
    int64_t sparseBlockSize = 1;
    int64_t attentionMode = 2;
    int64_t quantScaleRepoMode = 1;
    int64_t tileSize = 128;
    int64_t ropeHeadDim = 64;
    constexpr const char layerOutStr[] = "BSND";
    constexpr size_t layerOutLen = sizeof(layerOutStr);
    char layoutQuery[layerOutLen];
    char layoutKey[layerOutLen];
    errno_t memcpyRet = memcpy_s(layoutQuery, sizeof(layoutQuery), layerOutStr, layerOutLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutQuery failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    memcpyRet = memcpy_s(layoutKey, sizeof(layoutKey), layerOutStr, layerOutLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutKey failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    aclOpExecutor* executor;

    int ret = aclnnKvQuantSparseFlashAttentionPioneerGetWorkspaceSize(resources.queryTensor, resources.keyTensor, resources.valueTensor, resources.sparseIndicesTensor,
                        resources.keyDequantScaleTensor, resources.valueDequantScaleTensor, nullptr, nullptr, nullptr, nullptr, nullptr, scaleValue, keyQuantMode,
                        valueQuantMode, sparseBlockSize, layoutQuery, layoutKey, sparseMode, preTokens, nextTokens, attentionMode, quantScaleRepoMode,
                        tileSize, ropeHeadDim, resources.attentionOutTensor, workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnKvQuantSparseFlashAttentionPioneerGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnKvQuantSparseFlashAttentionPioneer(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnKvQuantSparseFlashAttentionPioneer failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

int PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<aclFloat16> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
  }
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
  }
  return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr,
                     aclrtStream stream, int32_t deviceId) {
    if (resources.queryTensor) {
      aclDestroyTensor(resources.queryTensor);
    }
    if (resources.keyTensor) {
      aclDestroyTensor(resources.keyTensor);
    }
    if (resources.valueTensor) {
      aclDestroyTensor(resources.valueTensor);
    }
    if (resources.sparseIndicesTensor) {
      aclDestroyTensor(resources.sparseIndicesTensor);
    }
    if (resources.keyDequantScaleTensor) {
      aclDestroyTensor(resources.keyDequantScaleTensor);
    }
    if (resources.valueDequantScaleTensor) {
      aclDestroyTensor(resources.valueDequantScaleTensor);
    }
    if (resources.attentionOutTensor) {
      aclDestroyTensor(resources.attentionOutTensor);
    }

    if (resources.queryDeviceAddr) {
      aclrtFree(resources.queryDeviceAddr);
    }
    if (resources.keyDeviceAddr) {
      aclrtFree(resources.keyDeviceAddr);
    }
    if (resources.valueDeviceAddr) {
      aclrtFree(resources.valueDeviceAddr);
    }
    if (resources.sparseIndicesDeviceAddr) {
      aclrtFree(resources.sparseIndicesDeviceAddr);
    }
    if (resources.keyDequantScaleDeviceAddr) {
      aclrtFree(resources.keyDequantScaleDeviceAddr);
    }
    if (resources.valueDequantScaleDeviceAddr) {
      aclrtFree(resources.valueDequantScaleDeviceAddr);
    }
    if (resources.attentionOutDeviceAddr) {
      aclrtFree(resources.attentionOutDeviceAddr);
    }

    if (workspaceAddr) {
      aclrtFree(workspaceAddr);
    }
    if (stream) {
      aclrtDestroyStream(stream);
    }
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
    std::vector<int64_t> attentionOutShape = {1, 1, 64, 512};
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
    ret = ExecuteKvQuantSparseFlashAttentionPioneer(resources, stream, &workspaceAddr, &workspaceSize);
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

    // 5. Process results
    PrintOutResult(attentionOutShape, &resources.attentionOutDeviceAddr);

    // 6. Cleanup resources
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}