/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_compressor.h"

#define CHECK_RET(cond, expr) \
    do { \
        if (!(cond)) { \
            expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        std::printf(message, ##__VA_ARGS__); \
        std::fflush(stdout); \
    } while (0)

namespace {

constexpr int32_t DEVICE_ID = 0;
constexpr int64_t B = 1;
constexpr int64_t S = 1;
constexpr int64_t HIDDEN_SIZE = 1024;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t COFF = 2;
constexpr int64_t CMP_RATIO = 4;
constexpr int64_t ROPE_HEAD_DIM = 64;
constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t OUT_DIM = 2 * COFF * HEAD_DIM;
constexpr int64_t C128_HEAD_DIM = 512;
constexpr int64_t C128_COFF = 1;
constexpr int64_t C128_CMP_RATIO = 128;
constexpr int64_t C128_BLOCK_SIZE = 512;
constexpr int64_t C128_OUT_DIM = 2 * C128_COFF * C128_HEAD_DIM;
constexpr int64_t C128_PREFILL_SEQ = 512;
constexpr int64_t C128_MODEL_BATCH = 16;
constexpr int64_t C128_MODEL_HIDDEN_SIZE = 7168;
constexpr int64_t C128_MODEL_PREFILL_SEQ = 2048;
constexpr int64_t C128_MODEL_PA_MAX_LENGTH = 2560;
constexpr int64_t C128_MODEL_DECODE_START_POS = 1920;
constexpr uint16_t BF16_ZERO = 0x0000;
constexpr uint16_t BF16_ONE = 0x3f80;
constexpr uint16_t BF16_TWO = 0x4000;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

std::vector<int64_t> MakeStrides(const std::vector<int64_t> &shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

float Bf16ToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

struct AclTensorHolder {
    void *deviceAddr = nullptr;
    aclTensor *tensor = nullptr;
    size_t byteSize = 0;

    ~AclTensorHolder()
    {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
        }
        if (deviceAddr != nullptr) {
            aclrtFree(deviceAddr);
        }
    }
};

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, aclDataType dataType,
                    AclTensorHolder &holder)
{
    holder.byteSize = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    auto ret = aclrtMalloc(&holder.deviceAddr, holder.byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed: %d\n", ret); return ret);
    ret = aclrtMemcpy(holder.deviceAddr, holder.byteSize, hostData.data(), holder.byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D failed: %d\n", ret); return ret);

    auto strides = MakeStrides(shape);
    holder.tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND,
                                    shape.data(), shape.size(), holder.deviceAddr);
    CHECK_RET(holder.tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

int CreateAclTensorByShape(const std::vector<int64_t> &shape, aclDataType dataType, size_t elementSize,
                           AclTensorHolder &holder)
{
    holder.byteSize = static_cast<size_t>(GetShapeSize(shape)) * elementSize;
    auto ret = aclrtMalloc(&holder.deviceAddr, holder.byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed: %d\n", ret); return ret);

    auto strides = MakeStrides(shape);
    holder.tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND,
                                    shape.data(), shape.size(), holder.deviceAddr);
    CHECK_RET(holder.tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

template <typename T>
int CopyTensorToHost(const AclTensorHolder &holder, std::vector<T> &hostData)
{
    auto byteSize = hostData.size() * sizeof(T);
    auto ret = aclrtMemcpy(hostData.data(), byteSize, holder.deviceAddr, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy D2H failed: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

int InitAcl(aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed: %d\n", ret); return ret);
    ret = aclrtSetDevice(DEVICE_ID);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

void FillCurrentXForUpdate(std::vector<uint16_t> &xData)
{
    for (int64_t i = 0; i < static_cast<int64_t>(xData.size()); ++i) {
        xData[i] = (i < HIDDEN_SIZE / 2) ? BF16_ONE : BF16_TWO;
    }
}

void FillCachedXForCompress(std::vector<uint16_t> &xCacheData)
{
    std::fill(xCacheData.begin(), xCacheData.end(), BF16_ZERO);
    for (int64_t h = 0; h < HEAD_DIM; ++h) {
        xCacheData[h] = BF16_ONE;
    }
}

void FillIdentityCurrentCoffWkv(std::vector<uint16_t> &wkvData)
{
    std::fill(wkvData.begin(), wkvData.end(), BF16_ZERO);
    for (int64_t d = 0; d < HEAD_DIM; ++d) {
        wkvData[(HEAD_DIM + d) * HIDDEN_SIZE + d] = BF16_ONE;
    }
}

bool CheckUpdateOnlyXCache(const std::vector<uint16_t> &xData, const std::vector<uint16_t> &xCacheResult)
{
    for (int64_t h = 0; h < HIDDEN_SIZE; ++h) {
        if (xCacheResult[h] != xData[h]) {
            LOG_PRINT("update-only mismatch at slot0 h=%ld expected=0x%04x actual=0x%04x\n", h, xData[h],
                      xCacheResult[h]);
            return false;
        }
    }
    for (int64_t i = HIDDEN_SIZE; i < static_cast<int64_t>(xCacheResult.size()); ++i) {
        if (xCacheResult[i] != BF16_ZERO) {
            LOG_PRINT("update-only unexpected nonzero x_cache at index=%ld actual=0x%04x\n", i, xCacheResult[i]);
            return false;
        }
    }
    return true;
}

bool CheckCompressCase(const std::vector<uint16_t> &xCacheResult, const std::vector<uint16_t> &cmpKvResult)
{
    constexpr float expected = 1.41421356f;
    for (int64_t h = 0; h < HIDDEN_SIZE; ++h) {
        if (xCacheResult[3 * HIDDEN_SIZE + h] != BF16_ZERO) {
            LOG_PRINT("compress x_cache slot3 should contain current zero x, h=%ld actual=0x%04x\n", h,
                      xCacheResult[3 * HIDDEN_SIZE + h]);
            return false;
        }
    }
    for (int64_t d = 0; d < HEAD_DIM; ++d) {
        float actual = Bf16ToFloat(cmpKvResult[d]);
        float expectedValue = d < ROPE_HEAD_DIM ? expected : 0.0f;
        if (std::fabs(actual - expectedValue) > 0.04f) {
            LOG_PRINT("compress cmp_kv mismatch d=%ld expected~=%f actual=%f bf16=0x%04x\n", d, expectedValue, actual,
                      cmpKvResult[d]);
            return false;
        }
    }
    return true;
}

int RunCompressorCase(aclrtStream stream, const std::string &caseName, int32_t seqUsed, int32_t xCacheSlot,
                      bool compressCase)
{
    std::vector<int64_t> xShape = {B, S, HIDDEN_SIZE};
    std::vector<int64_t> wShape = {COFF * HEAD_DIM, HIDDEN_SIZE};
    std::vector<int64_t> stateCacheShape = {B, BLOCK_SIZE, OUT_DIM};
    std::vector<int64_t> apeShape = {CMP_RATIO, COFF * HEAD_DIM};
    std::vector<int64_t> normWeightShape = {HEAD_DIM};
    std::vector<int64_t> ropeShape = {B, 1, ROPE_HEAD_DIM};
    std::vector<int64_t> blockTableShape = {B, 1};
    std::vector<int64_t> batchShape = {B};
    std::vector<int64_t> xCacheShape = {B, CMP_RATIO, HIDDEN_SIZE};
    std::vector<int64_t> xCacheSlotMappingShape = {B, 1};
    std::vector<int64_t> cmpKvOutShape = {B, 1, HEAD_DIM};

    std::vector<uint16_t> xData(GetShapeSize(xShape), BF16_ZERO);
    std::vector<uint16_t> wkvData(GetShapeSize(wShape), BF16_ZERO);
    std::vector<uint16_t> wgateData(GetShapeSize(wShape), BF16_ZERO);
    std::vector<float> stateCacheData(GetShapeSize(stateCacheShape), 0.0f);
    std::vector<float> apeData(GetShapeSize(apeShape), 0.0f);
    std::vector<float> normWeightData(GetShapeSize(normWeightShape), 1.0f);
    std::vector<float> ropeSinData(GetShapeSize(ropeShape), 0.0f);
    std::vector<float> ropeCosData(GetShapeSize(ropeShape), 1.0f);
    std::vector<int32_t> stateBlockTableData = {0};
    std::vector<int32_t> sequsedData = {seqUsed};
    std::vector<int32_t> startPosData = {0};
    std::vector<uint16_t> xCacheData(GetShapeSize(xCacheShape), BF16_ZERO);
    std::vector<int32_t> xCacheSlotMappingData = {xCacheSlot};
    std::vector<uint16_t> cmpKvOutData(GetShapeSize(cmpKvOutShape), BF16_TWO);

    if (compressCase) {
        FillCachedXForCompress(xCacheData);
        FillIdentityCurrentCoffWkv(wkvData);
    } else {
        FillCurrentXForUpdate(xData);
    }

    AclTensorHolder x;
    AclTensorHolder wkv;
    AclTensorHolder wgate;
    AclTensorHolder stateCache;
    AclTensorHolder ape;
    AclTensorHolder normWeight;
    AclTensorHolder ropeSin;
    AclTensorHolder ropeCos;
    AclTensorHolder stateBlockTable;
    AclTensorHolder seqused;
    AclTensorHolder startPos;
    AclTensorHolder xCache;
    AclTensorHolder xCacheSlotMapping;
    AclTensorHolder cmpKvOut;

    int ret = CreateAclTensor(xData, xShape, ACL_BF16, x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wkvData, wShape, ACL_BF16, wkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wgateData, wShape, ACL_BF16, wgate);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(stateCacheData, stateCacheShape, ACL_FLOAT, stateCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(apeData, apeShape, ACL_FLOAT, ape);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normWeightData, normWeightShape, ACL_FLOAT, normWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(ropeSinData, ropeShape, ACL_FLOAT, ropeSin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(ropeCosData, ropeShape, ACL_FLOAT, ropeCos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(stateBlockTableData, blockTableShape, ACL_INT32, stateBlockTable);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sequsedData, batchShape, ACL_INT32, seqused);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(startPosData, batchShape, ACL_INT32, startPos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(xCacheData, xCacheShape, ACL_BF16, xCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(xCacheSlotMappingData, xCacheSlotMappingShape, ACL_INT32, xCacheSlotMapping);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cmpKvOutData, cmpKvOutShape, ACL_BF16, cmpKvOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    int64_t stateCacheStrideDim0 = stateCacheShape[1] * stateCacheShape[2];
    ret = aclnnCompressorGetWorkspaceSize(x.tensor, wkv.tensor, wgate.tensor, stateCache.tensor, ape.tensor,
                                          normWeight.tensor, ropeSin.tensor, ropeCos.tensor, stateBlockTable.tensor,
                                          nullptr, seqused.tensor, startPos.tensor, xCache.tensor,
                                          xCacheSlotMapping.tensor, ROPE_HEAD_DIM, CMP_RATIO, COFF, 1e-6, 2, 1,
                                          stateCacheStrideDim0, true, cmpKvOut.tensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s GetWorkspaceSize failed: %d\n", caseName.c_str(), ret); return ret);
    LOG_PRINT("%s workspaceSize=%lu\n", caseName.c_str(), workspaceSize);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s aclrtMalloc workspace failed: %d\n", caseName.c_str(), ret);
                  return ret);
    }

    ret = aclnnCompressor(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s aclnnCompressor failed: %d\n", caseName.c_str(), ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s aclrtSynchronizeStream failed: %d\n", caseName.c_str(), ret);
              return ret);

    std::vector<uint16_t> xCacheResult(GetShapeSize(xCacheShape), BF16_ZERO);
    std::vector<uint16_t> cmpKvResult(GetShapeSize(cmpKvOutShape), BF16_ZERO);
    ret = CopyTensorToHost(xCache, xCacheResult);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CopyTensorToHost(cmpKvOut, cmpKvResult);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }

    bool passed =
        compressCase ? CheckCompressCase(xCacheResult, cmpKvResult) : CheckUpdateOnlyXCache(xData, xCacheResult);
    CHECK_RET(passed, return ACL_ERROR_FAILURE);
    LOG_PRINT("%s success\n", caseName.c_str());
    return ACL_SUCCESS;
}

int RunPrefillNoXCacheWorkspaceCase(const std::string &caseName, int64_t batchSize, int64_t hiddenSize, int64_t seqLen,
                                    int64_t paMaxLength)
{
    int64_t tokenSize = batchSize * seqLen;
    int64_t cmpTokens = std::min(tokenSize, tokenSize / C128_CMP_RATIO + batchSize);
    int64_t stateBlockNumPerBatch = std::max<int64_t>(1, (C128_CMP_RATIO + C128_BLOCK_SIZE - 1) / C128_BLOCK_SIZE);
    int64_t stateBlocks = batchSize * stateBlockNumPerBatch + 1;
    int64_t blockTableLen = (paMaxLength + C128_BLOCK_SIZE - 1) / C128_BLOCK_SIZE;

    std::vector<int64_t> xShape = {tokenSize, hiddenSize};
    std::vector<int64_t> wShape = {C128_COFF * C128_HEAD_DIM, hiddenSize};
    std::vector<int64_t> stateCacheShape = {stateBlocks, C128_BLOCK_SIZE, C128_OUT_DIM};
    std::vector<int64_t> apeShape = {C128_CMP_RATIO, C128_COFF * C128_HEAD_DIM};
    std::vector<int64_t> normWeightShape = {C128_HEAD_DIM};
    std::vector<int64_t> ropeShape = {cmpTokens, ROPE_HEAD_DIM};
    std::vector<int64_t> blockTableShape = {batchSize, blockTableLen};
    std::vector<int64_t> cuSeqlensShape = {batchSize + 1};
    std::vector<int64_t> batchShape = {batchSize};
    std::vector<int64_t> cmpKvOutShape = {cmpTokens, C128_HEAD_DIM};
    std::vector<int32_t> stateBlockTableData(static_cast<size_t>(batchSize * blockTableLen), 0);
    int64_t actualBlockStart = (seqLen - C128_CMP_RATIO - 1) / C128_BLOCK_SIZE;
    int64_t actualBlockEnd = (seqLen - 1) / C128_BLOCK_SIZE;
    for (int64_t batch = 0; batch < batchSize; ++batch) {
        int32_t blockId = static_cast<int32_t>(batch * stateBlockNumPerBatch + 1);
        for (int64_t block = actualBlockStart; block <= actualBlockEnd && block < blockTableLen; ++block) {
            if (block >= 0) {
                stateBlockTableData[static_cast<size_t>(batch * blockTableLen + block)] = blockId;
            }
        }
    }
    std::vector<int32_t> cuSeqlensData(static_cast<size_t>(batchSize + 1), 0);
    for (int64_t batch = 0; batch <= batchSize; ++batch) {
        cuSeqlensData[static_cast<size_t>(batch)] = static_cast<int32_t>(batch * seqLen);
    }
    std::vector<int32_t> sequsedData(static_cast<size_t>(batchSize), static_cast<int32_t>(seqLen));
    std::vector<int32_t> startPosData(static_cast<size_t>(batchSize), 0);

    AclTensorHolder x;
    AclTensorHolder wkv;
    AclTensorHolder wgate;
    AclTensorHolder stateCache;
    AclTensorHolder ape;
    AclTensorHolder normWeight;
    AclTensorHolder ropeSin;
    AclTensorHolder ropeCos;
    AclTensorHolder stateBlockTable;
    AclTensorHolder cuSeqlens;
    AclTensorHolder seqused;
    AclTensorHolder startPos;
    AclTensorHolder cmpKvOut;

    int ret = CreateAclTensorByShape(xShape, ACL_BF16, sizeof(uint16_t), x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorByShape(wShape, ACL_BF16, sizeof(uint16_t), wkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorByShape(wShape, ACL_BF16, sizeof(uint16_t), wgate);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorByShape(stateCacheShape, ACL_FLOAT, sizeof(float), stateCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorByShape(apeShape, ACL_FLOAT, sizeof(float), ape);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorByShape(normWeightShape, ACL_FLOAT, sizeof(float), normWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorByShape(ropeShape, ACL_FLOAT, sizeof(float), ropeSin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorByShape(ropeShape, ACL_FLOAT, sizeof(float), ropeCos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(stateBlockTableData, blockTableShape, ACL_INT32, stateBlockTable);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cuSeqlensData, cuSeqlensShape, ACL_INT32, cuSeqlens);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sequsedData, batchShape, ACL_INT32, seqused);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(startPosData, batchShape, ACL_INT32, startPos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorByShape(cmpKvOutShape, ACL_BF16, sizeof(uint16_t), cmpKvOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    int64_t stateCacheStrideDim0 = stateCacheShape[1] * stateCacheShape[2];
    ret = aclnnCompressorGetWorkspaceSize(x.tensor, wkv.tensor, wgate.tensor, stateCache.tensor, ape.tensor,
                                          normWeight.tensor, ropeSin.tensor, ropeCos.tensor, stateBlockTable.tensor,
                                          cuSeqlens.tensor, seqused.tensor, startPos.tensor, nullptr, nullptr,
                                          ROPE_HEAD_DIM, C128_CMP_RATIO, C128_COFF, 1e-6, 2, 1, stateCacheStrideDim0,
                                          false, cmpKvOut.tensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s GetWorkspaceSize failed: %d\n", caseName.c_str(), ret); return ret);
    LOG_PRINT("%s workspaceSize=%lu B=%ld H=%ld seq=%ld blockTableLen=%ld stateBlocks=%ld\n", caseName.c_str(),
              workspaceSize, batchSize, hiddenSize, seqLen, blockTableLen, stateBlocks);
    LOG_PRINT("%s success\n", caseName.c_str());
    return ACL_SUCCESS;
}

int RunDecodeUpdateWorkspaceCase(const std::string &caseName, int64_t batchSize, int64_t hiddenSize,
                                 int64_t startPosValue, int64_t paMaxLength)
{
    int64_t blockTableLen = (paMaxLength + C128_BLOCK_SIZE - 1) / C128_BLOCK_SIZE;
    int64_t localSlot = startPosValue % C128_CMP_RATIO;

    std::vector<int64_t> xShape = {batchSize, 1, hiddenSize};
    std::vector<int64_t> wShape = {C128_COFF * C128_HEAD_DIM, hiddenSize};
    std::vector<int64_t> stateCacheShape = {batchSize + 1, C128_BLOCK_SIZE, C128_OUT_DIM};
    std::vector<int64_t> apeShape = {C128_CMP_RATIO, C128_COFF * C128_HEAD_DIM};
    std::vector<int64_t> normWeightShape = {C128_HEAD_DIM};
    std::vector<int64_t> ropeShape = {batchSize, 1, ROPE_HEAD_DIM};
    std::vector<int64_t> blockTableShape = {batchSize, blockTableLen};
    std::vector<int64_t> batchShape = {batchSize};
    std::vector<int64_t> xCacheShape = {batchSize, C128_CMP_RATIO, hiddenSize};
    std::vector<int64_t> xCacheSlotMappingShape = {batchSize, 1};
    std::vector<int64_t> cmpKvOutShape = {batchSize, 1, C128_HEAD_DIM};

    std::vector<int32_t> stateBlockTableData(static_cast<size_t>(batchSize * blockTableLen), 0);
    int64_t stateBlockCol = std::min(blockTableLen - 1, startPosValue / C128_BLOCK_SIZE);
    for (int64_t batch = 0; batch < batchSize; ++batch) {
        stateBlockTableData[static_cast<size_t>(batch * blockTableLen + stateBlockCol)] =
            static_cast<int32_t>(batch + 1);
    }
    std::vector<int32_t> sequsedData(static_cast<size_t>(batchSize), 0);
    std::vector<int32_t> startPosData(static_cast<size_t>(batchSize), static_cast<int32_t>(startPosValue));
    std::vector<int32_t> xCacheSlotMappingData(static_cast<size_t>(batchSize), 0);
    for (int64_t batch = 0; batch < batchSize; ++batch) {
        xCacheSlotMappingData[static_cast<size_t>(batch)] = static_cast<int32_t>(batch * C128_CMP_RATIO + localSlot);
    }

    AclTensorHolder x;
    AclTensorHolder wkv;
    AclTensorHolder wgate;
    AclTensorHolder stateCache;
    AclTensorHolder ape;
    AclTensorHolder normWeight;
    AclTensorHolder ropeSin;
    AclTensorHolder ropeCos;
    AclTensorHolder stateBlockTable;
    AclTensorHolder seqused;
    AclTensorHolder startPos;
    AclTensorHolder xCache;
    AclTensorHolder xCacheSlotMapping;
    AclTensorHolder cmpKvOut;

    int ret = CreateAclTensorByShape(xShape, ACL_BF16, sizeof(uint16_t), x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorByShape(wShape, ACL_BF16, sizeof(uint16_t), wkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorByShape(wShape, ACL_BF16, sizeof(uint16_t), wgate);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorByShape(stateCacheShape, ACL_FLOAT, sizeof(float), stateCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorByShape(apeShape, ACL_FLOAT, sizeof(float), ape);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorByShape(normWeightShape, ACL_FLOAT, sizeof(float), normWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorByShape(ropeShape, ACL_FLOAT, sizeof(float), ropeSin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorByShape(ropeShape, ACL_FLOAT, sizeof(float), ropeCos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(stateBlockTableData, blockTableShape, ACL_INT32, stateBlockTable);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sequsedData, batchShape, ACL_INT32, seqused);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(startPosData, batchShape, ACL_INT32, startPos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorByShape(xCacheShape, ACL_BF16, sizeof(uint16_t), xCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(xCacheSlotMappingData, xCacheSlotMappingShape, ACL_INT32, xCacheSlotMapping);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorByShape(cmpKvOutShape, ACL_BF16, sizeof(uint16_t), cmpKvOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    int64_t stateCacheStrideDim0 = stateCacheShape[1] * stateCacheShape[2];
    ret = aclnnCompressorGetWorkspaceSize(x.tensor, wkv.tensor, wgate.tensor, stateCache.tensor, ape.tensor,
                                          normWeight.tensor, ropeSin.tensor, ropeCos.tensor, stateBlockTable.tensor,
                                          nullptr, seqused.tensor, startPos.tensor, xCache.tensor,
                                          xCacheSlotMapping.tensor, ROPE_HEAD_DIM, C128_CMP_RATIO, C128_COFF, 1e-6, 2,
                                          1, stateCacheStrideDim0, true, cmpKvOut.tensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s GetWorkspaceSize failed: %d\n", caseName.c_str(), ret); return ret);
    LOG_PRINT("%s workspaceSize=%lu B=%ld H=%ld startPos=%ld localSlot=%ld blockTableLen=%ld\n", caseName.c_str(),
              workspaceSize, batchSize, hiddenSize, startPosValue, localSlot, blockTableLen);
    LOG_PRINT("%s success\n", caseName.c_str());
    return ACL_SUCCESS;
}

int RunPrefillThenDecodeUpdateWorkspaceCase(const std::string &caseName)
{
    int ret = RunPrefillNoXCacheWorkspaceCase(caseName + ".prefill", C128_MODEL_BATCH, C128_MODEL_HIDDEN_SIZE,
                                              C128_MODEL_PREFILL_SEQ, C128_MODEL_PA_MAX_LENGTH);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = RunDecodeUpdateWorkspaceCase(caseName + ".decode_update", C128_MODEL_BATCH, C128_MODEL_HIDDEN_SIZE,
                                       C128_MODEL_DECODE_START_POS, C128_MODEL_PA_MAX_LENGTH);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    LOG_PRINT("%s success\n", caseName.c_str());
    return ACL_SUCCESS;
}

} // namespace

int main(int argc, char **argv)
{
    std::string mode = argc > 1 ? argv[1] : "all";
    if (mode != "update" && mode != "compress" && mode != "prefill-null-workspace" &&
        mode != "prefill-model-null-workspace" && mode != "prefill-model-null-decode-update-workspace" &&
        mode != "all") {
        LOG_PRINT("usage: %s [update|compress|prefill-null-workspace|prefill-model-null-workspace|"
                  "prefill-model-null-decode-update-workspace|all]\n",
                  argv[0]);
        return 1;
    }

    aclrtStream stream = nullptr;
    auto ret = InitAcl(&stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    if (mode == "update" || mode == "all") {
        ret = RunCompressorCase(stream, "update-only", 0, 0, false);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    if (mode == "compress" || mode == "all") {
        ret = RunCompressorCase(stream, "compress", CMP_RATIO, CMP_RATIO - 1, true);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    if (mode == "prefill-null-workspace") {
        ret = RunPrefillNoXCacheWorkspaceCase("prefill-null-workspace", B, HIDDEN_SIZE, C128_PREFILL_SEQ,
                                              C128_PREFILL_SEQ);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    if (mode == "prefill-model-null-workspace") {
        ret = RunPrefillNoXCacheWorkspaceCase("prefill-model-null-workspace", C128_MODEL_BATCH, C128_MODEL_HIDDEN_SIZE,
                                              C128_MODEL_PREFILL_SEQ, C128_MODEL_PA_MAX_LENGTH);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    if (mode == "prefill-model-null-decode-update-workspace") {
        ret = RunPrefillThenDecodeUpdateWorkspaceCase("prefill-model-null-decode-update-workspace");
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(DEVICE_ID);
    aclFinalize();
    LOG_PRINT("aclnnCompressor grouped decode x-cache examples success: %s\n", mode.c_str());
    return 0;
}
