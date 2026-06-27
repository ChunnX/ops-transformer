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
 * \file test_aclnn_bsa_select_block_mask.cpp
 * \brief
 */
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_bsa_select_block_mask.h"

using namespace std;

#define CHECK_RET(cond, return_expr) \
    do {                               \
        if (!(cond)) {                   \
            return_expr;                   \
        }                                \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do {                              \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int32_t GetEnvInt32(const char* name, int32_t defaultValue) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > INT32_MAX) {
        LOG_PRINT("Invalid %s=%s, use default %d\n", name, value, defaultValue);
        return defaultValue;
    }
    return static_cast<int32_t>(parsed);
}

bool GetEnvBool(const char* name, bool defaultValue) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    if (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 || std::strcmp(value, "FALSE") == 0) {
        return false;
    }
    if (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 || std::strcmp(value, "TRUE") == 0) {
        return true;
    }
    LOG_PRINT("Invalid %s=%s, use default %d\n", name, value, defaultValue ? 1 : 0);
    return defaultValue;
}

float GetEnvFloat(const char* name, float defaultValue) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    char* end = nullptr;
    float parsed = std::strtof(value, &end);
    if (end == value || *end != '\0' || parsed <= 0.0f || parsed >= 1.0f) {
        LOG_PRINT("Invalid %s=%s, use default %.2f\n", name, value, defaultValue);
        return defaultValue;
    }
    return parsed;
}

int Init(int32_t deviceId, aclrtStream* stream) {
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
                    aclDataType dataType, aclTensor** tensor) {
    if (shape.empty()) {
        LOG_PRINT("CreateAclTensor: ERROR - shape is empty\n");
        return -1;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] <= 0) {
            LOG_PRINT("CreateAclTensor: ERROR - shape[%zu]=%ld is invalid\n", i, shape[i]);
            return -1;
        }
    }
    
    auto size = GetShapeSize(shape) * sizeof(T);

    if (hostData.size() != static_cast<size_t>(GetShapeSize(shape))) {
        LOG_PRINT("CreateAclTensor: ERROR - hostData size mismatch: %zu vs %ld\n", 
                  hostData.size(), GetShapeSize(shape));
        return -1;
    }
    
    *deviceAddr = nullptr;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.size() > 1) {
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }

    *tensor = nullptr;
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed - returned nullptr\n"); return -1);
    return 0;
}

void FreeResource(aclTensor *query, aclTensor *key, aclTensor *maskOut,
                  void *queryDeviceAddr, void *keyDeviceAddr, void *maskOutDeviceAddr,
                  void *workspaceAddr, int32_t deviceId, aclrtStream *stream)
{
    if (query) aclDestroyTensor(query);
    if (key) aclDestroyTensor(key);
    if (maskOut) aclDestroyTensor(maskOut);

    if (queryDeviceAddr) aclrtFree(queryDeviceAddr);
    if (keyDeviceAddr) aclrtFree(keyDeviceAddr);
    if (maskOutDeviceAddr) aclrtFree(maskOutDeviceAddr);
    if (workspaceAddr) aclrtFree(workspaceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

void FreeOutputResource(aclTensor *qCompOut, aclTensor *kCompOut, aclTensor *scoreOut, aclTensor *softmaxOut,
                        void *qCompOutDeviceAddr, void *kCompOutDeviceAddr,
                        void *scoreOutDeviceAddr, void *softmaxOutDeviceAddr)
{
    if (qCompOut) aclDestroyTensor(qCompOut);
    if (kCompOut) aclDestroyTensor(kCompOut);
    if (scoreOut) aclDestroyTensor(scoreOut);
    if (softmaxOut) aclDestroyTensor(softmaxOut);

    if (qCompOutDeviceAddr) aclrtFree(qCompOutDeviceAddr);
    if (kCompOutDeviceAddr) aclrtFree(kCompOutDeviceAddr);
    if (scoreOutDeviceAddr) aclrtFree(scoreOutDeviceAddr);
    if (softmaxOutDeviceAddr) aclrtFree(softmaxOutDeviceAddr);
}

int main() {
    LOG_PRINT("=== BSASelectBlockMask ACLNN Test ===\n");
    
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int32_t batch = GetEnvInt32("BSA_BATCH", 1);
    int32_t numHeads = GetEnvInt32("BSA_NUM_HEADS", 1);
    int32_t qSeqlen = GetEnvInt32("BSA_Q_SEQLEN", 65536);
    int32_t kvSeqlen = GetEnvInt32("BSA_KV_SEQLEN", 65536);
    int32_t headDim = 128;
    int32_t blockShapeX = GetEnvInt32("BSA_BLOCK_SHAPE_X", 128);
    int32_t blockShapeY = GetEnvInt32("BSA_BLOCK_SHAPE_Y", 128);
    float sparsity = GetEnvFloat("BSA_SPARSITY", 0.50f);
    
    int32_t xBlocks = (qSeqlen + blockShapeX - 1) / blockShapeX;
    int32_t yBlocks = (kvSeqlen + blockShapeY - 1) / blockShapeY;
    int64_t sortLen = static_cast<int64_t>(xBlocks) * yBlocks;
    float expectedSelectedFloat = std::round(sparsity * static_cast<float>(sortLen));
    int64_t expectedSelectedPerMask = static_cast<int64_t>(expectedSelectedFloat);
    if (expectedSelectedPerMask < 1) {
        expectedSelectedPerMask = 1;
    }
    int64_t expectedSelectedTotal =
        expectedSelectedPerMask * static_cast<int64_t>(batch) * static_cast<int64_t>(numHeads);
    double expectedSparsity = 1.0 - static_cast<double>(expectedSelectedPerMask) / static_cast<double>(sortLen);
    
    LOG_PRINT("Test parameters:\n");
    LOG_PRINT("  batch=%d, numHeads=%d, qSeqlen=%d, kvSeqlen=%d, headDim=%d\n", 
              batch, numHeads, qSeqlen, kvSeqlen, headDim);
    LOG_PRINT("  blockShape=[%d, %d], xBlocks=%d, yBlocks=%d, sparsity=%.2f\n",
              blockShapeX, blockShapeY, xBlocks, yBlocks, sparsity);
    LOG_PRINT("  sortLen=%ld, expectedSelectedPerMask=%ld, expectedSelectedTotal=%ld\n",
              sortLen, expectedSelectedPerMask, expectedSelectedTotal);

    aclTensor *queryTensor = nullptr;
    aclTensor *keyTensor = nullptr;
    aclTensor *maskOutTensor = nullptr;
    aclTensor *qCompOutTensor = nullptr;
    aclTensor *kCompOutTensor = nullptr;
    aclTensor *scoreOutTensor = nullptr;
    aclTensor *softmaxOutTensor = nullptr;
    void *queryDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *maskOutDeviceAddr = nullptr;
    void *qCompOutDeviceAddr = nullptr;
    void *kCompOutDeviceAddr = nullptr;
    void *scoreOutDeviceAddr = nullptr;
    void *softmaxOutDeviceAddr = nullptr;
    void *workspaceAddr = nullptr;

    std::vector<int64_t> queryShape = {batch, numHeads, qSeqlen, headDim};
    std::vector<op::fp16_t> queryHostData(GetShapeSize(queryShape), 1.0f);
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create query tensor\n");
              FreeResource(queryTensor, keyTensor, maskOutTensor, queryDeviceAddr, keyDeviceAddr,
                           maskOutDeviceAddr, workspaceAddr, deviceId, &stream); return ret);

    std::vector<int64_t> keyShape = {batch, numHeads, kvSeqlen, headDim};
    std::vector<op::fp16_t> keyHostData(GetShapeSize(keyShape), 1.0f);
    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create key tensor\n");
              FreeResource(queryTensor, keyTensor, maskOutTensor, queryDeviceAddr, keyDeviceAddr,
                           maskOutDeviceAddr, workspaceAddr, deviceId, &stream); return ret);

    std::vector<int64_t> maskOutShape = {batch, numHeads, xBlocks, yBlocks};
    int64_t maskOutSize = GetShapeSize(maskOutShape);
    std::vector<int8_t> maskOutHostData(maskOutSize, 0);
    ret = CreateAclTensor(maskOutHostData, maskOutShape, &maskOutDeviceAddr, aclDataType::ACL_INT8, &maskOutTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create maskOut tensor\n");
              FreeResource(queryTensor, keyTensor, maskOutTensor, queryDeviceAddr, keyDeviceAddr,
                           maskOutDeviceAddr, workspaceAddr, deviceId, &stream); return ret);

    std::vector<int64_t> qCompOutShape = {batch, numHeads, xBlocks, headDim};
    std::vector<int64_t> kCompOutShape = {batch, numHeads, yBlocks, headDim};
    std::vector<int64_t> scoreOutShape = {batch, numHeads, xBlocks, yBlocks};
    std::vector<int64_t> softmaxOutShape = {batch, numHeads, xBlocks, yBlocks};

    std::vector<float> qCompOutHostData(GetShapeSize(qCompOutShape), 0.0f);
    ret = CreateAclTensor(qCompOutHostData, qCompOutShape, &qCompOutDeviceAddr,
                          aclDataType::ACL_FLOAT, &qCompOutTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create qCompOut tensor\n");
              FreeResource(queryTensor, keyTensor, maskOutTensor, queryDeviceAddr, keyDeviceAddr,
                           maskOutDeviceAddr, workspaceAddr, deviceId, &stream); return ret);

    std::vector<float> kCompOutHostData(GetShapeSize(kCompOutShape), 0.0f);
    ret = CreateAclTensor(kCompOutHostData, kCompOutShape, &kCompOutDeviceAddr,
                          aclDataType::ACL_FLOAT, &kCompOutTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create kCompOut tensor\n");
              FreeResource(queryTensor, keyTensor, maskOutTensor, queryDeviceAddr, keyDeviceAddr,
                           maskOutDeviceAddr, workspaceAddr, deviceId, &stream); return ret);

    std::vector<float> scoreOutHostData(GetShapeSize(scoreOutShape), 0.0f);
    ret = CreateAclTensor(scoreOutHostData, scoreOutShape, &scoreOutDeviceAddr,
                          aclDataType::ACL_FLOAT, &scoreOutTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create scoreOut tensor\n");
              FreeResource(queryTensor, keyTensor, maskOutTensor, queryDeviceAddr, keyDeviceAddr,
                           maskOutDeviceAddr, workspaceAddr, deviceId, &stream); return ret);

    std::vector<op::fp16_t> softmaxOutHostData(GetShapeSize(softmaxOutShape), 0.0f);
    ret = CreateAclTensor(softmaxOutHostData, softmaxOutShape, &softmaxOutDeviceAddr,
                          aclDataType::ACL_FLOAT16, &softmaxOutTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Failed to create softmaxOut tensor\n");
              FreeResource(queryTensor, keyTensor, maskOutTensor, queryDeviceAddr, keyDeviceAddr,
                           maskOutDeviceAddr, workspaceAddr, deviceId, &stream); return ret);

    std::vector<int64_t> blockShapeData = {blockShapeX, blockShapeY};
    aclIntArray *blockShape = aclCreateIntArray(blockShapeData.data(), blockShapeData.size());
    CHECK_RET(blockShape != nullptr, LOG_PRINT("Failed to create blockShape array\n");
              FreeResource(queryTensor, keyTensor, maskOutTensor, queryDeviceAddr, keyDeviceAddr,
                           maskOutDeviceAddr, workspaceAddr, deviceId, &stream); return -1);

    char qLayoutBuffer[16] = "BNSD";
    char kvLayoutBuffer[16] = "BNSD";
    int64_t numKeyValueHeads = numHeads;
    double scaleValue = 1.0 / std::sqrt(static_cast<double>(headDim));

    uint64_t workspaceCapacity = 0;

    std::vector<int8_t> resultData(maskOutSize, 0);
    int64_t selectedCount = 0;
    int64_t selectedFirstHalf = 0;
    int64_t selectedSecondHalf = 0;
    int64_t firstSelectedSecondHalf = -1;
    double actualSparsity = 0.0;
    int32_t kernelRunTimes = GetEnvInt32("BSA_KERNEL_RUN_TIMES", 1);
    bool verifyEachRun = GetEnvBool("BSA_VERIFY_EACH_RUN", true);
    double totalOpExecuteMs = 0.0;
    double maxOpExecuteMs = 0.0;
    double lastOpExecuteMs = 0.0;

    LOG_PRINT("Calling aclnnBSASelectBlockMask %d times with reused tensors and fresh executor...\n", kernelRunTimes);
    LOG_PRINT("Verify result each run: %d\n", verifyEachRun ? 1 : 0);
    for (int32_t runId = 0; runId < kernelRunTimes; ++runId) {
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;

        ret = aclnnBSASelectBlockMaskGetWorkspaceSize(
            queryTensor,
            keyTensor,
            blockShape,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            qLayoutBuffer,
            kvLayoutBuffer,
            numKeyValueHeads,
            scaleValue,
            static_cast<double>(sparsity),
            maskOutTensor,
            qCompOutTensor,
            kCompOutTensor,
            scoreOutTensor,
            softmaxOutTensor,
            &workspaceSize,
            &executor);

        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclnnBSASelectBlockMaskGetWorkspaceSize failed at run %d. ERROR: %d\n",
                            runId + 1, ret);
                  aclDestroyIntArray(blockShape);
                  FreeResource(queryTensor, keyTensor, maskOutTensor, queryDeviceAddr, keyDeviceAddr,
                               maskOutDeviceAddr, workspaceAddr, deviceId, &stream); return ret);
        CHECK_RET(executor != nullptr,
                  LOG_PRINT("executor is null after GetWorkspaceSize at run %d\n", runId + 1);
                  aclDestroyIntArray(blockShape);
                  FreeResource(queryTensor, keyTensor, maskOutTensor, queryDeviceAddr, keyDeviceAddr,
                               maskOutDeviceAddr, workspaceAddr, deviceId, &stream); return -1);

        if (workspaceSize > workspaceCapacity) {
            if (workspaceAddr != nullptr) {
                aclrtFree(workspaceAddr);
                workspaceAddr = nullptr;
            }
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS,
                      LOG_PRINT("allocate workspace failed at run %d. ERROR: %d\n", runId + 1, ret);
                      aclDestroyIntArray(blockShape);
                      FreeResource(queryTensor, keyTensor, maskOutTensor, queryDeviceAddr, keyDeviceAddr,
                                   maskOutDeviceAddr, workspaceAddr, deviceId, &stream); return ret);
            workspaceCapacity = workspaceSize;
            LOG_PRINT("workspaceCapacity = %lu bytes (%.2f MB)\n",
                      workspaceCapacity, workspaceCapacity / 1024.0 / 1024.0);
        }

        auto opExecuteStart = std::chrono::steady_clock::now();
        ret = aclnnBSASelectBlockMask(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclnnBSASelectBlockMask failed at run %d. ERROR: %d\n", runId + 1, ret);
                  aclDestroyIntArray(blockShape);
                  FreeResource(queryTensor, keyTensor, maskOutTensor, queryDeviceAddr, keyDeviceAddr,
                               maskOutDeviceAddr, workspaceAddr, deviceId, &stream); return ret);

        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclrtSynchronizeStream failed at run %d. ERROR: %d\n", runId + 1, ret);
                  aclDestroyIntArray(blockShape);
                  FreeResource(queryTensor, keyTensor, maskOutTensor, queryDeviceAddr, keyDeviceAddr,
                               maskOutDeviceAddr, workspaceAddr, deviceId, &stream); return ret);
        auto opExecuteEnd = std::chrono::steady_clock::now();
        lastOpExecuteMs = std::chrono::duration<double, std::milli>(opExecuteEnd - opExecuteStart).count();
        totalOpExecuteMs += lastOpExecuteMs;
        if (lastOpExecuteMs > maxOpExecuteMs) {
            maxOpExecuteMs = lastOpExecuteMs;
        }
        LOG_PRINT("Run %03d/%03d: Op execute time %.3f ms\n",
                  runId + 1, kernelRunTimes, lastOpExecuteMs);

        bool shouldVerifyResult = verifyEachRun || (runId + 1 == kernelRunTimes);
        if (!shouldVerifyResult) {
            continue;
        }

        ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(int8_t), maskOutDeviceAddr,
                          resultData.size() * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("copy result from device to host failed at run %d. ERROR: %d\n", runId + 1, ret);
                  aclDestroyIntArray(blockShape);
                  FreeResource(queryTensor, keyTensor, maskOutTensor, queryDeviceAddr, keyDeviceAddr,
                               maskOutDeviceAddr, workspaceAddr, deviceId, &stream); return ret);

        selectedCount = 0;
        selectedFirstHalf = 0;
        selectedSecondHalf = 0;
        firstSelectedSecondHalf = -1;
        for (int64_t i = 0; i < maskOutSize; i++) {
            if (resultData[i] != 0) {
                selectedCount++;
                if (i < maskOutSize / 2) {
                    selectedFirstHalf++;
                } else {
                    selectedSecondHalf++;
                    if (firstSelectedSecondHalf < 0) {
                        firstSelectedSecondHalf = i;
                    }
                }
            }
        }

        actualSparsity = 1.0 - static_cast<double>(selectedCount) / static_cast<double>(maskOutSize);
        LOG_PRINT("Run %03d/%03d: Selected=%ld (expected: %ld), Actual sparsity=%.4f (expected: %.4f)\n",
                  runId + 1, kernelRunTimes, selectedCount, expectedSelectedTotal, actualSparsity, expectedSparsity);
    }
    
    LOG_PRINT("\n=== Results ===\n");
    LOG_PRINT("maskOut shape: [%d, %d, %d, %d]\n", batch, numHeads, xBlocks, yBlocks);
    LOG_PRINT("Total elements: %ld\n", maskOutSize);
    LOG_PRINT("Selected (non-zero): %ld (expected: %ld)\n", selectedCount, expectedSelectedTotal);
    LOG_PRINT("Selected first half: %ld, second half: %ld, first second-half index: %ld\n",
              selectedFirstHalf, selectedSecondHalf, firstSelectedSecondHalf);
    LOG_PRINT("Actual sparsity: %.4f (expected: %.4f)\n", actualSparsity, expectedSparsity);
    if (kernelRunTimes > 0) {
        LOG_PRINT("Op execute time: last=%.3f ms, avg=%.3f ms, max=%.3f ms\n",
                  lastOpExecuteMs, totalOpExecuteMs / kernelRunTimes, maxOpExecuteMs);
    }
    
    uint64_t printNum = 20;
    LOG_PRINT("First %lu elements:\n", printNum);
    for (uint64_t i = 0; i < printNum && i < resultData.size(); i++) {
        LOG_PRINT("  index %lu: %d\n", i, static_cast<int>(resultData[i]));
    }

    uint64_t probeTileLen = maskOutSize < 7168 ? static_cast<uint64_t>(maskOutSize) : 7168;
    if (probeTileLen > 0) {
        uint64_t probeTotalTiles = (static_cast<uint64_t>(maskOutSize) + probeTileLen - 1) / probeTileLen;
        uint64_t sampleTiles[3] = {0, probeTotalTiles / 2, probeTotalTiles - 1};
        LOG_PRINT("Robustness probe sample positions:\n");
        for (uint64_t sampleId = 0; sampleId < 3; sampleId++) {
            uint64_t tileId = sampleTiles[sampleId];
            uint64_t tileBase = tileId * probeTileLen;
            uint64_t curTileLen = (tileId + 1 == probeTotalTiles)
                ? static_cast<uint64_t>(maskOutSize) - tileBase : probeTileLen;
            uint64_t oneLen = curTileLen >= 64 ? 8 : (curTileLen >= 16 ? 4 : 1);
            uint64_t oneStart = curTileLen / 2;
            if (oneStart + oneLen > curTileLen) {
                oneStart = curTileLen > oneLen ? curTileLen - oneLen : 0;
            }
            uint64_t oneIndex = tileBase + oneStart;
            LOG_PRINT("  tile %lu one[%lu..%lu]: %d",
                      tileId, oneIndex, oneIndex + oneLen - 1,
                      static_cast<int>(resultData[oneIndex]));

            uint64_t gap = curTileLen >= 64 ? 16 : 4;
            uint64_t setStart = oneStart + oneLen + gap;
            if (setStart < curTileLen) {
                uint64_t stride = curTileLen >= 64 ? 7 : 3;
                uint64_t jitter = tileId % stride;
                for (uint64_t i = 0; i < 4; i++) {
                    uint64_t pos = setStart + jitter + i * stride;
                    if (pos < curTileLen) {
                        uint64_t index = tileBase + pos;
                        LOG_PRINT(", set[%lu]: %d", index, static_cast<int>(resultData[index]));
                    }
                }
            }
            LOG_PRINT("\n");
        }
    }

    aclDestroyIntArray(blockShape);
    FreeOutputResource(qCompOutTensor, kCompOutTensor, scoreOutTensor, softmaxOutTensor,
                       qCompOutDeviceAddr, kCompOutDeviceAddr,
                       scoreOutDeviceAddr, softmaxOutDeviceAddr);
    FreeResource(queryTensor, keyTensor, maskOutTensor, queryDeviceAddr, keyDeviceAddr,
                 maskOutDeviceAddr, workspaceAddr, deviceId, &stream);
    
    LOG_PRINT("\n=== Test completed successfully! ===\n");
    return 0;
}
