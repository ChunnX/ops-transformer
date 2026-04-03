/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_sparse_flash_attention_n2_1.cpp
 * \brief White-box test cases for sparse flash attention with N2=1, sparseBlockSize=1, N1=64/128
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_sparse_flash_attention.h"

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

int32_t Init(int32_t deviceId, aclrtStream* stream) {
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
int32_t CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
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
    void* attentionOutDeviceAddr = nullptr;
    void* softmaxMaxDeviceAddr = nullptr;
    void* softmaxSumDeviceAddr = nullptr;
    void* queryRopeDeviceAddr = nullptr;
    void* keyRopeDeviceAddr = nullptr;

    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    aclTensor* valueTensor = nullptr;
    aclTensor* sparseIndicesTensor = nullptr;
    aclTensor* attentionOutTensor = nullptr;
    aclTensor* softmaxMaxTensor = nullptr;
    aclTensor* softmaxSumTensor = nullptr;
    aclTensor* queryRopeTensor = nullptr;
    aclTensor* keyRopeTensor = nullptr;
};

int32_t CreateTensors(TensorResources& resources, int64_t n1, int64_t n2, int64_t s1, int64_t s2, int64_t d) {
    int64_t batchSize = n1;
    int64_t kvHeadNum = n2;
    int64_t qHeadNum = n2;
    int64_t seqLenQ = s1;
    int64_t seqLenKV = s2;
    int64_t headDim = d;

    std::vector<int64_t> queryShape = {batchSize, seqLenQ, qHeadNum, headDim};
    std::vector<int64_t> keyShape = {batchSize, seqLenKV, kvHeadNum, headDim};
    std::vector<int64_t> valueShape = {batchSize, seqLenKV, kvHeadNum, headDim};
    std::vector<int64_t> sparseIndicesShape = {batchSize, seqLenQ, kvHeadNum, (seqLenKV + 1) / 2};
    std::vector<int64_t> attentionOutShape = {batchSize, seqLenQ, qHeadNum, headDim};
    std::vector<int64_t> softmaxMaxShape = {batchSize, seqLenQ, qHeadNum};
    std::vector<int64_t> softmaxSumShape = {batchSize, seqLenQ, qHeadNum};
    std::vector<int64_t> queryRopeShape = {batchSize, seqLenQ, qHeadNum, 2};
    std::vector<int64_t> keyRopeShape = {batchSize, seqLenKV, kvHeadNum, 2};

    int64_t querySize = GetShapeSize(queryShape);
    int64_t keySize = GetShapeSize(keyShape);
    int64_t valueSize = GetShapeSize(valueShape);
    int64_t sparseIndicesSize = GetShapeSize(sparseIndicesShape);
    int64_t attentionOutSize = GetShapeSize(attentionOutShape);
    int64_t softmaxMaxSize = GetShapeSize(softmaxMaxShape);
    int64_t softmaxSumSize = GetShapeSize(softmaxSumShape);
    int64_t queryRopeSize = GetShapeSize(queryRopeShape);
    int64_t keyRopeSize = GetShapeSize(keyRopeShape);

    std::vector<aclFloat16> queryData(querySize, aclFloatToFloat(aclFloat16(0.1f)));
    std::vector<aclFloat16> keyData(keySize, aclFloatToFloat(aclFloat16(0.2f)));
    std::vector<aclFloat16> valueData(valueSize, aclFloatToFloat(aclFloat16(0.3f)));
    std::vector<int32_t> sparseIndicesData(sparseIndicesSize, 0);
    std::vector<aclFloat16> attentionOutData(attentionOutSize, aclFloatToFloat(aclFloat16(0.0f)));
    std::vector<float> softmaxMaxData(softmaxMaxSize, 0.0f);
    std::vector<float> softmaxSumData(softmaxSumSize, 0.0f);
    std::vector<float> queryRopeData(queryRopeSize, 0.0f);
    std::vector<float> keyRopeData(keyRopeSize, 0.0f);

    for (int64_t i = 0; i < sparseIndicesSize; i++) {
        sparseIndicesData[i] = static_cast<int32_t>(i % seqLenKV);
    }

    auto ret = CreateAclTensor(queryData, queryShape, &resources.queryDeviceAddr, ACL_FLOAT16, &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    ret = CreateAclTensor(keyData, keyShape, &resources.keyDeviceAddr, ACL_FLOAT16, &resources.keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    ret = CreateAclTensor(valueData, valueShape, &resources.valueDeviceAddr, ACL_FLOAT16, &resources.valueTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    ret = CreateAclTensor(sparseIndicesData, sparseIndicesShape, &resources.sparseIndicesDeviceAddr, ACL_INT32, &resources.sparseIndicesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    ret = CreateAclTensor(attentionOutData, attentionOutShape, &resources.attentionOutDeviceAddr, ACL_FLOAT16, &resources.attentionOutTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    ret = CreateAclTensor(softmaxMaxData, softmaxMaxShape, &resources.softmaxMaxDeviceAddr, ACL_FLOAT, &resources.softmaxMaxTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    ret = CreateAclTensor(softmaxSumData, softmaxSumShape, &resources.softmaxSumDeviceAddr, ACL_FLOAT, &resources.softmaxSumTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    ret = CreateAclTensor(queryRopeData, queryRopeShape, &resources.queryRopeDeviceAddr, ACL_FLOAT, &resources.queryRopeTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    ret = CreateAclTensor(keyRopeData, keyRopeShape, &resources.keyRopeDeviceAddr, ACL_FLOAT, &resources.keyRopeTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    return ACL_SUCCESS;
}

int32_t ExecuteSparseFlashAttention(TensorResources& resources, aclrtStream stream, 
                              void** workspaceAddr, uint64_t* workspaceSize, int64_t d) {
    double scaleValue = 1 / sqrt(d);
    int64_t sparseBlockSize = 1;
    constexpr const char layerOutStr[] = "BSND";
    constexpr size_t layerOutLen = sizeof(layerOutStr);
    char layoutQuery[layerOutLen];
    char layoutKv[layerOutLen];
    errno_t memcpyRet = memcpy_s(layoutQuery, sizeof(layoutQuery), layerOutStr, layerOutLen);
    if (memcpyRet != 0) {
        LOG_PRINT("memcpy_s layoutQuery failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    memcpyRet = memcpy_s(layoutKv, sizeof(layoutKv), layerOutStr, layerOutLen);
    if (memcpyRet != 0) {
        LOG_PRINT("memcpy_s layoutKv failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    int64_t sparseMode = 3;
    int64_t preTokens = 9223372036854775807;
    int64_t nextTokens = 9223372036854775807;
    int64_t attentionMode = 2;
    bool returnSoftmaxLse = false;
    aclOpExecutor* executor;

    int32_t ret = aclnnSparseFlashAttentionGetWorkspaceSize(resources.queryTensor, resources.keyTensor, resources.valueTensor, resources.sparseIndicesTensor, nullptr, nullptr, 
                                                    resources.queryRopeTensor, resources.keyRopeTensor,
                                                    scaleValue, sparseBlockSize, layoutQuery, layoutKv, sparseMode, preTokens,
                                                    nextTokens, attentionMode, returnSoftmaxLse, resources.attentionOutTensor, resources.softmaxMaxTensor, resources.softmaxSumTensor, workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnSparseFlashAttentionGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnSparseFlashAttention(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnSparseFlashAttention failed. ERROR: %d\n", ret);
        return ret;
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
    if (resources.attentionOutTensor) {
      aclDestroyTensor(resources.attentionOutTensor);
    }
    if (resources.softmaxMaxTensor) {
      aclDestroyTensor(resources.softmaxMaxTensor);
    }
    if (resources.softmaxSumTensor) {
      aclDestroyTensor(resources.softmaxSumTensor);
    }
    if (resources.queryRopeTensor) {
      aclDestroyTensor(resources.queryRopeTensor);
    }
    if (resources.keyRopeTensor) {
      aclDestroyTensor(resources.keyRopeTensor);
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
    if (resources.attentionOutDeviceAddr) {
      aclrtFree(resources.attentionOutDeviceAddr);
    }
    if (resources.softmaxMaxDeviceAddr) {
      aclrtFree(resources.softmaxMaxDeviceAddr);
    }
    if (resources.softmaxSumDeviceAddr) {
      aclrtFree(resources.softmaxSumDeviceAddr);
    }
    if (resources.queryRopeDeviceAddr) {
      aclrtFree(resources.queryRopeDeviceAddr);
    }
    if (resources.keyRopeDeviceAddr) {
      aclrtFree(resources.keyRopeDeviceAddr);
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

int32_t RunTestCase(int64_t n1, int64_t n2, int64_t s1, int64_t s2, int64_t d, int32_t deviceId) {
    LOG_PRINT("Running test case: N1=%ld, N2=%ld, S1=%ld, S2=%ld, D=%ld\n", n1, n2, s1, s2, d);
    
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init failed. ERROR: %d\n", ret);
        return ret;
    }

    TensorResources resources;
    ret = CreateTensors(resources, n1, n2, s1, s2, d);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("CreateTensors failed. ERROR: %d\n", ret);
        CleanupResources(resources, nullptr, stream, deviceId);
        return ret;
    }

    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    ret = ExecuteSparseFlashAttention(resources, stream, &workspaceAddr, &workspaceSize, d);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("ExecuteSparseFlashAttention failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    LOG_PRINT("Test case passed: N1=%ld, N2=%ld, S1=%ld, S2=%ld, D=%ld\n", n1, n2, s1, s2, d);
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return ACL_SUCCESS;
}

} 

int32_t main(int32_t argc, char* argv[]) {
    int32_t deviceId = 0;
    if (argc > 1) {
        deviceId = atoi(argv[1]);
    }

    int32_t ret = ACL_SUCCESS;
    int64_t n2 = 1;
    int64_t sparseBlockSize = 1;
    
    LOG_PRINT("=== Sparse Flash Attention White-box Test Cases ===\n");
    LOG_PRINT("Fixed parameters: N2=%ld, sparseBlockSize=%ld\n", n2, sparseBlockSize);
    
    std::vector<int64_t> n1Values = {64, 128};
    std::vector<int64_t> s1Values = {1, 2, 4, 8};
    std::vector<int64_t> s2Values = {1, 2, 4, 8, 16};
    std::vector<int64_t> dValues = {2, 4, 8};

    int32_t totalTestCases = n1Values.size() * s1Values.size() * s2Values.size() * dValues.size();
    int32_t passedTestCases = 0;

    for (auto n1 : n1Values) {
        for (auto s1 : s1Values) {
            for (auto s2 : s2Values) {
                for (auto d : dValues) {
                    ret = RunTestCase(n1, n2, s1, s2, d, deviceId);
                    if (CHECK_RET(ret == ACL_SUCCESS)) {
                        passedTestCases++;
                    } else {
                        LOG_PRINT("Test case failed: N1=%ld, N2=%ld, S1=%ld, S2=%ld, D=%ld\n", n1, n2, s1, s2, d);
                    }
                }
            }
        }
    }

    LOG_PRINT("=== Test Summary ===\n");
    LOG_PRINT("Total test cases: %d\n", totalTestCases);
    LOG_PRINT("Passed test cases: %d\n", passedTestCases);
    LOG_PRINT("Failed test cases: %d\n", totalTestCases - passedTestCases);

    return (passedTestCases == totalTestCases) ? 0 : 1;
}