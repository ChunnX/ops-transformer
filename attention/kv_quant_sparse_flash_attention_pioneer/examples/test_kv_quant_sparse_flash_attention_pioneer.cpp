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
#include "acl/acl.h"
#include "aclnnop/aclnn_kv_quant_sparse_flash_attention_pioneer.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true : (false))

#define LOG_PRINT(message, ...)               \
    do {                                      \
        (void)printf(message, ##__VA_ARGS__); \
    } while (0)

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
                        aclDataType dataType, aclTensor **tensor)
{
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

// Convert float to BF16 (truncate lower 16 bits of FP32)
uint16_t FloatToBf16(float val)
{
    uint32_t bits;
    memcpy(&bits, &val, sizeof(float));
    return static_cast<uint16_t>(bits >> 16);
}

// Convert BF16 to float
float Bf16ToFloat(uint16_t val)
{
    uint32_t bits = static_cast<uint32_t>(val) << 16;
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

struct TensorResources {
    void *queryDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *valueDeviceAddr = nullptr;
    void *sparseIndicesDeviceAddr = nullptr;
    void *keyDequantScaleDeviceAddr = nullptr;
    void *valueDequantScaleDeviceAddr = nullptr;
    void *blockTableDeviceAddr = nullptr;
    void *actualSeqLenQDeviceAddr = nullptr;
    void *actualSeqLenKvDeviceAddr = nullptr;
    void *keySinkDeviceAddr = nullptr;
    void *valueSinkDeviceAddr = nullptr;
    void *attentionOutDeviceAddr = nullptr;

    aclTensor *queryTensor = nullptr;
    aclTensor *keyTensor = nullptr;
    aclTensor *valueTensor = nullptr;
    aclTensor *sparseIndicesTensor = nullptr;
    aclTensor *keyDequantScaleTensor = nullptr;
    aclTensor *valueDequantScaleTensor = nullptr;
    aclTensor *blockTableTensor = nullptr;
    aclTensor *actualSeqLenQTensor = nullptr;
    aclTensor *actualSeqLenKvTensor = nullptr;
    aclTensor *keySinkTensor = nullptr;
    aclTensor *valueSinkTensor = nullptr;
    aclTensor *attentionOutTensor = nullptr;
};

// Shapes
constexpr int64_t B = 1;
constexpr int64_t S1 = 1;
constexpr int64_t N1 = 64;
constexpr int64_t N2 = 1;
constexpr int64_t D_QUERY = 576;
constexpr int64_t D_KV = 656;
constexpr int64_t D_DEQUANT_SCALE = 4;
constexpr int64_t D_OUT = 512;
constexpr int64_t PA_BLOCK_SIZE = 256;
constexpr int64_t S2 = 512;
constexpr int64_t TOTAL_BLOCKS = S2 / PA_BLOCK_SIZE;
constexpr int64_t MAX_BLOCKS_PER_BATCH = TOTAL_BLOCKS;
constexpr int64_t SPARSE_BLOCK_COUNT = 2048;
constexpr int64_t SINK_TOKEN_NUM = 128;

int InitializeTensors(TensorResources &resources)
{
    std::vector<int64_t> queryShape = {B, S1, N1, D_QUERY};
    std::vector<int64_t> keyShape = {TOTAL_BLOCKS, PA_BLOCK_SIZE, N2, D_KV};
    std::vector<int64_t> valueShape = {TOTAL_BLOCKS, PA_BLOCK_SIZE, N2, D_KV};
    std::vector<int64_t> sparseIndicesShape = {B, S1, N2, SPARSE_BLOCK_COUNT};
    std::vector<int64_t> dequantScaleShape = {TOTAL_BLOCKS, PA_BLOCK_SIZE, N2, D_DEQUANT_SCALE};
    std::vector<int64_t> blockTableShape = {B, MAX_BLOCKS_PER_BATCH};
    std::vector<int64_t> actualSeqLenQShape = {B};
    std::vector<int64_t> actualSeqLenKvShape = {B};
    std::vector<int64_t> keySinkShape = {SINK_TOKEN_NUM, N2, D_QUERY};
    std::vector<int64_t> valueSinkShape = {SINK_TOKEN_NUM, N2, D_OUT};
    std::vector<int64_t> attentionOutShape = {B, S1, N1, D_OUT};

    int64_t querySize = GetShapeSize(queryShape);
    int64_t keySize = GetShapeSize(keyShape);
    int64_t valueSize = GetShapeSize(valueShape);
    int64_t sparseIndicesSize = GetShapeSize(sparseIndicesShape);
    float keyDequantScaleSize = GetShapeSize(dequantScaleShape);
    float valueDequantScaleSize = GetShapeSize(dequantScaleShape);
    int64_t blockTableSize = GetShapeSize(blockTableShape);
    int64_t keySinkSize = GetShapeSize(keySinkShape);
    int64_t valueSinkSize = GetShapeSize(valueSinkShape);
    int64_t attentionOutSize = GetShapeSize(attentionOutShape);

    uint16_t bf16_small = FloatToBf16(0.01f);
    std::vector<uint16_t> queryHostData(querySize, bf16_small);
    std::vector<uint8_t> keyHostData(keySize, 0x38); // FP8 E4M3FN: 0x38 = 1.0
    std::vector<uint8_t> valueHostData(valueSize, 0x38);
    std::vector<int32_t> sparseIndicesHostData(sparseIndicesSize);
    for (int32_t i = 0; i < sparseIndicesSize; i++) {
        sparseIndicesHostData[i] = i;
    }
    std::vector<float> keyDequantScaleHostData = {keyDequantScaleSize, 1.0f};
    std::vector<float> valueDequantScaleHostData = {valueDequantScaleSize, 1.0f};
    std::vector<int32_t> blockTableHostData(blockTableSize);
    for (int32_t i = 0; i < blockTableSize; i++) {
        blockTableHostData[i] = i;
    }
    std::vector<int32_t> actualSeqLenQHostData = {static_cast<int32_t>(S1)};
    std::vector<int32_t> actualSeqLenKvHostData = {static_cast<int32_t>(S2)};
    uint16_t bf16_one = FloatToBf16(1.0f);
    std::vector<uint16_t> keySinkHostData(keySinkSize, bf16_one);
    std::vector<uint16_t> valueSinkHostData(valueSinkSize, bf16_one);
    std::vector<uint16_t> attentionOutHostData(attentionOutSize, 0);

    int ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr,
                                  aclDataType::ACL_BF16, &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(keyHostData, keyShape, &resources.keyDeviceAddr,
                          static_cast<aclDataType>(ACL_FLOAT8_E4M3FN), &resources.keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(valueHostData, valueShape, &resources.valueDeviceAddr,
                          static_cast<aclDataType>(ACL_FLOAT8_E4M3FN), &resources.valueTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape, &resources.sparseIndicesDeviceAddr,
                          aclDataType::ACL_INT32, &resources.sparseIndicesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(keyDequantScaleHostData, dequantScaleShape, &resources.keyDequantScaleDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.keyDequantScaleTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(valueDequantScaleHostData, dequantScaleShape, &resources.valueDequantScaleDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.valueDequantScaleTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(blockTableHostData, blockTableShape, &resources.blockTableDeviceAddr,
                          aclDataType::ACL_INT32, &resources.blockTableTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(actualSeqLenQHostData, actualSeqLenQShape, &resources.actualSeqLenQDeviceAddr,
                          aclDataType::ACL_INT32, &resources.actualSeqLenQTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(actualSeqLenKvHostData, actualSeqLenKvShape, &resources.actualSeqLenKvDeviceAddr,
                          aclDataType::ACL_INT32, &resources.actualSeqLenKvTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(keySinkHostData, keySinkShape, &resources.keySinkDeviceAddr,
                          aclDataType::ACL_BF16, &resources.keySinkTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(valueSinkHostData, valueSinkShape, &resources.valueSinkDeviceAddr,
                          aclDataType::ACL_BF16, &resources.valueSinkTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &resources.attentionOutDeviceAddr,
                          aclDataType::ACL_BF16, &resources.attentionOutTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    return ACL_SUCCESS;
}

int ExecuteKvQuantSparseFlashAttentionPioneer(TensorResources &resources, aclrtStream stream,
                  void **workspaceAddr, uint64_t *workspaceSize)
{
    double scaleValue = 1.0 / sqrt(static_cast<double>(D_QUERY));
    int64_t keyQuantMode = 2;
    int64_t valueQuantMode = 2;
    int64_t sparseBlockSize = 1;
    char layoutQuery[] = "BSND";
    char layoutKv[] = "PA_BSND";
    int64_t sparseMode = 3;
    int64_t preTokens = 9223372036854775807LL;
    int64_t nextTokens = 9223372036854775807LL;
    int64_t attentionMode = 2;
    int64_t quantScaleRepoMode = 1;
    int64_t tileSize = 128;
    int64_t ropeHeadDim = 64;
    aclOpExecutor *executor;

    int ret = aclnnKvQuantSparseFlashAttentionPioneerGetWorkspaceSize(resources.queryTensor, resources.keyTensor, resources.valueTensor,
                            resources.sparseIndicesTensor, resources.keyDequantScaleTensor, resources.valueDequantScaleTensor,
                            resources.blockTableTensor, resources.actualSeqLenQTensor, resources.actualSeqLenKvTensor, resources.keySinkTensor,
                            resources.valueSinkTensor, scaleValue, keyQuantMode, valueQuantMode, sparseBlockSize, layoutQuery, layoutKv,
                            sparseMode, preTokens, nextTokens, attentionMode, quantScaleRepoMode, tileSize, ropeHeadDim,
                            resources.attentionOutTensor, workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnKvQuantSparseFlashAttentionPioneerGetWorkspaceSize failed. ERROR: %d\n", ret);
        const char *errMsg = aclGetRecentErrMsg();
        if (errMsg != nullptr) {
            LOG_PRINT("Detailed error: %s\n", errMsg);
        }
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

int PrintOutResult(const std::vector<int64_t> &shape, void *deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<uint16_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(uint16_t),
                           deviceAddr, size * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }

    // Print first 16 and last 16 elements (BF16)
    int64_t printCount = std::min(size, static_cast<int64_t>(16));
    for (int64_t i = 0; i < printCount; i++) {
        LOG_PRINT("  result[%ld] = %f (raw=0x%04x)\n", i, Bf16ToFloat(resultData[i]), resultData[i]);
    }
    if (size > 16) {
        LOG_PRINT("...\nLast %ld elements:\n", printCount);
        for (int64_t i = size - printCount; i < size; i++) {
            LOG_PRINT("  result[%ld] = %f (raw=0x%04x)\n", i, Bf16ToFloat(resultData[i]), resultData[i]);
        }
    }

    // Check for NaN/Inf
    int nanCount = 0;
    int infCount = 0;
    for (int64_t i = 0; i < size; i++) {
        float val = Bf16ToFloat(resultData[i]);
        if (std::isnan(val)) nanCount++;
        if (std::isinf(val)) infCount++;
    }
    LOG_PRINT("Total elements: %ld, NaN count: %d, Inf count: %d\n", size, nanCount, infCount);

    return ACL_SUCCESS;
}

void CleanupResources(TensorResources &resources, void *workspaceAddr,
                      aclrtStream stream, int32_t deviceId)
{
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
    if (resources.blockTableTensor) {
        aclDestroyTensor(resources.blockTableTensor);
    }
    if (resources.actualSeqLenQTensor) {
        aclDestroyTensor(resources.actualSeqLenQTensor);
    }
    if (resources.actualSeqLenKvTensor) {
        aclDestroyTensor(resources.actualSeqLenKvTensor);
    }
    if (resources.keySinkTensor) {
        aclDestroyTensor(resources.keySinkTensor);
    }
    if (resources.valueSinkTensor) {
        aclDestroyTensor(resources.valueSinkTensor);
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
    if (resources.blockTableDeviceAddr) {
        aclrtFree(resources.blockTableDeviceAddr);
    }
    if (resources.actualSeqLenQDeviceAddr) {
        aclrtFree(resources.actualSeqLenQDeviceAddr);
    }
    if (resources.actualSeqLenKvDeviceAddr) {
        aclrtFree(resources.actualSeqLenKvDeviceAddr);
    }
    if (resources.keySinkDeviceAddr) {
        aclrtFree(resources.keySinkDeviceAddr);
    }
    if (resources.valueSinkDeviceAddr) {
        aclrtFree(resources.valueSinkDeviceAddr);
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

int32_t main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    TensorResources resources = {};
    void *workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    int ret = ACL_SUCCESS;

    // 1. Initialize device and stream
    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // 2. Initialize tensors (with key_sink and value_sink)
    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("InitializeTensors failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 3. Execute the operation
    ret = ExecuteKvQuantSparseFlashAttentionPioneer(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("ExecuteKvQuantSparseFlashAttentionPioneer failed. ERROR: %d\n", ret);
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

    // 5. Print output results
    std::vector<int64_t> attentionOutShape = {B, S1, N1, D_OUT};
    PrintOutResult(attentionOutShape, resources.attentionOutDeviceAddr);

    // 6. Cleanup resources
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
