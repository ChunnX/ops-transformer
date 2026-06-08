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
 * \file test_aclnn_fused_infer_attention_score_v4_mla_fullquant.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnn/opdev/bfloat16.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v4.h"
#include "securec.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true :(false))

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream) {
    // Fixed writing method, AscendCL initialization.
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to request device side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { 
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); 
        return ret;
    }
    // Call aclrtMemcpy to copy host side data to device side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); 
        return ret;
    }

    // Calculate the strides of continuous tensors.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor interface to create aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

} // namespace

int main() {
    // 1. (Fixed writing method)  device/stream initialization. Refer to AscendCL's list of external interfaces.
    // Fill in the deviceId based on your actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret); 
        return ret;
    }

    // 2. To construct input and output, it is necessary to customize the construction according to the API interface.
    int64_t batchSize = 1;
    int64_t sequenceLengthQ = 1;
    int64_t sequenceLengthKV = 16;
    int64_t numHeads = 2;
    int64_t numKeyValueHeads = 1;
    int64_t headDims = 512;
    int64_t ropeHeadDims = 64;
    int64_t maxSequenceLengthKV = sequenceLengthKV;
    int64_t blockSize= 128;
    int64_t maxBlockNumPerSeqLenKv = (maxSequenceLengthKV + blockSize -1) / blockSize;
    int64_t blockNum = maxBlockNumPerSeqLenKv * batchSize;

    std::vector<int64_t> queryShape = {batchSize, sequenceLengthQ, numHeads, headDims};
    std::vector<int64_t> keyShape = {blockNum, numKeyValueHeads, headDims / 32, blockSize, 32};
    std::vector<int64_t> valueShape = {blockNum, numKeyValueHeads, headDims / 32, blockSize, 32};
    std::vector<int64_t> blockTableShape = {batchSize, maxBlockNumPerSeqLenKv};
    std::vector<int64_t> queryRopeShape = {batchSize, sequenceLengthQ, numHeads, ropeHeadDims};
    std::vector<int64_t> keyRopeShape = {blockNum, numKeyValueHeads, ropeHeadDims / 16, blockSize, 16};
    std::vector<int64_t> dequantScaleQueryShape = {batchSize, sequenceLengthQ, numHeads};
    std::vector<int64_t> keyAntiquantScaleShape = {1};
    std::vector<int64_t> valueAntiquantScaleShape = {1};
    std::vector<int64_t> attentionOutShape = {batchSize, sequenceLengthQ, numHeads, headDims};

    void *queryDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *valueDeviceAddr = nullptr;
    void *blockTableDeviceAddr = nullptr;
    void *queryRopeDeviceAddr = nullptr;
    void *keyRopeDeviceAddr = nullptr;
    void *dequantScaleQueryDeviceAddr = nullptr;
    void *keyAntiquantScaleDeviceAddr = nullptr;
    void *valueAntiquantScaleDeviceAddr = nullptr;
    void *attentionOutDeviceAddr = nullptr;

    aclTensor *queryTensor = nullptr;
    aclTensor *keyTensor = nullptr;
    aclTensor *valueTensor = nullptr;
    aclTensor *blockTableTensor = nullptr;
    aclTensor *queryRopeTensor = nullptr;
    aclTensor *keyRopeTensor = nullptr;
    aclTensor *dequantScaleQueryTensor = nullptr;
    aclTensor *keyAntiquantScaleTensor = nullptr;
    aclTensor *valueAntiquantScaleTensor = nullptr;
    aclTensor *attentionOutTensor = nullptr;

    int64_t queryShapeSize = GetShapeSize(queryShape);
    int64_t keyShapeSize = GetShapeSize(keyShape);
    int64_t valueShapeSize = GetShapeSize(valueShape);
    int64_t blockTableShapeSize = GetShapeSize(blockTableShape);
    int64_t queryRopeShapeSize = GetShapeSize(queryRopeShape);
    int64_t keyRopeShapeSize = GetShapeSize(keyRopeShape);
    int64_t dequantScaleQuerySize = GetShapeSize(dequantScaleQueryShape);
    int64_t keyAntiquantScaleShapeSize = GetShapeSize(keyAntiquantScaleShape);
    int64_t valueAntiquantScaleShapeSize = GetShapeSize(valueAntiquantScaleShape);
    int64_t attentionOutShapeSize = GetShapeSize(attentionOutShape);

    std::vector<int8_t> queryHostData(queryShapeSize, 1);
    std::vector<int8_t> keyHostData(keyShapeSize, 1);
    std::vector<int8_t> valueHostData(valueShapeSize, 1);
    std::vector<int32_t> blockTableHostData(blockTableShapeSize);
    for (int i = 0; i < blockTableShapeSize; i++) {
        blockTableHostData[i] = i;
    }
    std::vector<op::bfloat16> queryRopeHostData(queryRopeShapeSize, 1);
    std::vector<op::bfloat16> keyRopeHostData(keyRopeShapeSize, 1);
    std::vector<float> dequantScaleQueryHostData(dequantScaleQuerySize, 1);
    std::vector<float> keyAntiquantScaleHostData(keyAntiquantScaleShapeSize, 1);
    std::vector<float> valueAntiquantScaleHostData(valueAntiquantScaleShapeSize, 1);
    std::vector<op::bfloat16> attentionOutHostData(attentionOutShapeSize, 1);

    // Create query aclTensor.
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_INT8, &queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    // Create key aclTensor.
    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_INT8, &keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    int kvTensorNum = 1;
    aclTensor *tensorsOfKey[kvTensorNum];
    tensorsOfKey[0] = keyTensor;
    auto tensorKeyList = aclCreateTensorList(tensorsOfKey, kvTensorNum);

    // Create value aclTensor.
    ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_INT8, &valueTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    aclTensor *tensorsOfValue[kvTensorNum];
    tensorsOfValue[0] = valueTensor;
    auto tensorValueList = aclCreateTensorList(tensorsOfValue, kvTensorNum);

    // Create actualSeqLengthsKv.
    std::vector<int64_t> actualSeqLengthsKv = {batchSize, sequenceLengthKV};
    auto actualSeqLengthsKvIntArray = aclCreateIntArray(actualSeqLengthsKv.data(), actualSeqLengthsKv.size());
    if (!CHECK_RET(actualSeqLengthsKvIntArray != nullptr)) {
        return ret;
    }

    // Create blockTable aclTensor.
    ret = CreateAclTensor(blockTableHostData, blockTableShape, &blockTableDeviceAddr, aclDataType::ACL_INT32, &blockTableTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    // Create queryRope aclTensor.
    ret = CreateAclTensor(queryRopeHostData, queryRopeShape, &queryRopeDeviceAddr, aclDataType::ACL_BF16, &queryRopeTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    // Create keyRope aclTensor.
    ret = CreateAclTensor(keyRopeHostData, keyRopeShape, &keyRopeDeviceAddr, aclDataType::ACL_BF16, &keyRopeTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    // Create dequantScaleQuery aclTensor.
    ret = CreateAclTensor(dequantScaleQueryHostData, dequantScaleQueryShape, &dequantScaleQueryDeviceAddr, aclDataType::ACL_FLOAT, &dequantScaleQueryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    // Create keyAntiquantScale aclTensor.
    ret = CreateAclTensor(keyAntiquantScaleHostData, keyAntiquantScaleShape, &keyAntiquantScaleDeviceAddr, aclDataType::ACL_FLOAT, &keyAntiquantScaleTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    // Create valueAntiquantScale aclTensor.
    ret = CreateAclTensor(valueAntiquantScaleHostData, valueAntiquantScaleShape, &valueAntiquantScaleDeviceAddr, aclDataType::ACL_FLOAT, &valueAntiquantScaleTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    // Create attentionOut aclTensor.
    ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &attentionOutDeviceAddr, aclDataType::ACL_BF16, &attentionOutTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    // Create Attrs.
    double scaleValue = 1 / sqrt(headDims);
    int64_t preTokens = 2147483647;
    int64_t nextTokens = 2147483647;
    string sInputLayout = "BSND";
    char inputLayout[sInputLayout.length()+1];
    strcpy(inputLayout, sInputLayout.c_str());
    int64_t sparseMode = 0;
    int64_t innerPrecise = 0;
    int64_t antiquantMode = 0;
    bool softmaxLseFlag = false;
    int keyAntiquantMode = 0;
    int64_t valueAntiquantMode = 0;
    int64_t queryQuantMode = 3;

    // 3. Call CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // Call the first interface.
    ret = aclnnFusedInferAttentionScoreV4GetWorkspaceSize(
        queryTensor, tensorKeyList, tensorValueList, nullptr, nullptr, nullptr, actualSeqLengthsKvIntArray, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, blockTableTensor, nullptr, nullptr, keyAntiquantScaleTensor, nullptr, valueAntiquantScaleTensor, nullptr, nullptr,
        nullptr, nullptr, queryRopeTensor, keyRopeTensor, nullptr, dequantScaleQueryTensor, nullptr, numHeads, scaleValue, preTokens, nextTokens, inputLayout,
        numKeyValueHeads, sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag, keyAntiquantMode,
        valueAntiquantMode, queryQuantMode, attentionOutTensor, nullptr, &workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnFusedInferAttentionScoreV4GetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }
    // Apply for device memory based on the workspaceSize calculated from the first interface paragraph.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0U) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); 
            return ret;
        }
    }
    // Call the second interface.
    ret = aclnnFusedInferAttentionScoreV4(workspaceAddr, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnFusedInferAttentionScoreV4 failed. ERROR: %d\n", ret); 
        return ret;
    }

    // 4. (Fixed writing method) Synchronize and wait for task execution to end.
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); 
        return ret;
    }

    // 5. Retrieve the attentionOutput value, copy the result from the device side memory to the host side, and modify it
    // according to the specific API interface definition.
    auto size = GetShapeSize(attentionOutShape);
    std::vector<op::bfloat16> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), attentionOutDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { 
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); 
        return ret;
    }
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, static_cast<float>(resultData[i]));
    }
    // 6. Release resources.
    aclDestroyTensor(queryTensor);
    aclDestroyTensor(keyTensor);
    aclDestroyTensor(valueTensor);
    aclDestroyIntArray(actualSeqLengthsKvIntArray);
    aclDestroyTensor(blockTableTensor);
    aclDestroyTensor(queryRopeTensor);
    aclDestroyTensor(keyRopeTensor);
    aclDestroyTensor(attentionOutTensor);
    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(blockTableDeviceAddr);
    aclrtFree(queryRopeDeviceAddr);
    aclrtFree(keyRopeDeviceAddr);
    aclrtFree(attentionOutDeviceAddr);
    if (workspaceSize > 0U) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}