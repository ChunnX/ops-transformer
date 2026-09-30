/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_generic_lightning_indexer.h"

using namespace std;

namespace {
#define CHECK_RET(cond) ((cond) ? true : (false))
#define LOG_PRINT(message, ...) \
    do { \
        (void)printf(message, ##__VA_ARGS__); \
    } while (0)

uint16_t FloatToFp16(float value)
{
    uint32_t bits = 0;
    (void)memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000U;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFU) - 127 + 15;
    uint32_t man = bits & 0x7FFFFFU;
    if (exp <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (man >> 13));
}

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
    return aclrtCreateStream(stream);
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto nElem = GetShapeSize(shape);
    auto copySize = nElem * sizeof(T);
    auto allocSize = (copySize == 0) ? sizeof(T) : copySize;
    auto ret = aclrtMalloc(deviceAddr, allocSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    if (copySize > 0) {
        ret = aclrtMemcpy(*deviceAddr, allocSize, hostData.data(), copySize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            return ret;
        }
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int RunLiWeightedTopk(aclrtStream stream)
{
    std::vector<int64_t> qShape = {1, 2, 8, 128};
    std::vector<int64_t> kShape = {1, 16, 1, 128};
    std::vector<int64_t> wShape = {1, 2, 8};
    std::vector<int64_t> idxShape = {1, 2, 8};
    const uint16_t fp16One = FloatToFp16(1.0f);
    std::vector<uint16_t> qHost(GetShapeSize(qShape), fp16One);
    std::vector<uint16_t> kHost(GetShapeSize(kShape), fp16One);
    std::vector<uint16_t> wHost(GetShapeSize(wShape), fp16One);
    std::vector<int32_t> idxHost(GetShapeSize(idxShape), 0);
    void *qAddr = nullptr;
    void *kAddr = nullptr;
    void *wAddr = nullptr;
    void *idxAddr = nullptr;
    void *valAddr = nullptr;
    aclTensor *q = nullptr;
    aclTensor *k = nullptr;
    aclTensor *w = nullptr;
    aclTensor *idx = nullptr;
    aclTensor *val = nullptr;
    CreateAclTensor(qHost, qShape, &qAddr, aclDataType::ACL_FLOAT16, &q);
    CreateAclTensor(kHost, kShape, &kAddr, aclDataType::ACL_FLOAT16, &k);
    CreateAclTensor(wHost, wShape, &wAddr, aclDataType::ACL_FLOAT16, &w);
    CreateAclTensor(idxHost, idxShape, &idxAddr, aclDataType::ACL_INT32, &idx);
    std::vector<float> valHost(GetShapeSize(idxShape), 0.0f);
    CreateAclTensor(valHost, idxShape, &valAddr, aclDataType::ACL_FLOAT, &val);

    char layoutQ[] = "BSND";
    char layoutK[] = "BSND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = aclnnGenericLightningIndexerGetWorkspaceSize(
        q, k, w, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 8,
        1, layoutQ, layoutK, 3, 0, true, false, false, -1, true, 0, 0, 1.0f, -1, -1, -1, 0, 0, idx, val, &workspaceSize,
        &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnGenericLightningIndexerGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnGenericLightningIndexer(workspace, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnGenericLightningIndexer failed. ERROR: %d\n", ret);
        return ret;
    }
    aclrtSynchronizeStream(stream);

    const int64_t outNum = GetShapeSize(idxShape);
    std::vector<int32_t> idxOut(outNum, -1);
    std::vector<float> valOut(outNum, 0.0f);
    ret = aclrtMemcpy(idxOut.data(), outNum * sizeof(int32_t), idxAddr, outNum * sizeof(int32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy sparse_indices failed. ERROR: %d\n", ret);
        return ret;
    }
    ret =
        aclrtMemcpy(valOut.data(), outNum * sizeof(float), valAddr, outNum * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy sparse_values failed. ERROR: %d\n", ret);
        return ret;
    }
    int32_t badIdx = 0;
    int32_t nanVal = 0;
    int32_t badVal = 0;
    const float expectVal = 1024.0f;
    const float atol = 8.0f;
    for (int64_t i = 0; i < outNum; ++i) {
        if (idxOut[i] < 0 || idxOut[i] >= kShape[1]) {
            ++badIdx;
        }
        if (valOut[i] != valOut[i]) {
            ++nanVal;
        }
        if (fabsf(valOut[i] - expectVal) > atol) {
            ++badVal;
        }
        if (i < 16) {
            LOG_PRINT("out[%ld] idx=%d val=%f\n", i, idxOut[i], valOut[i]);
        }
    }
    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(w);
    aclDestroyTensor(idx);
    aclDestroyTensor(val);
    aclrtFree(qAddr);
    aclrtFree(kAddr);
    aclrtFree(wAddr);
    aclrtFree(idxAddr);
    aclrtFree(valAddr);
    if (workspace) {
        aclrtFree(workspace);
    }
    if (badIdx > 0 || nanVal > 0 || badVal > 0) {
        LOG_PRINT("LI precision check failed: badIdx=%d nanVal=%d badVal=%d expect=%f\n", badIdx, nanVal, badVal,
                  expectVal);
        return -1;
    }
    LOG_PRINT("LI example passed. indices in [0, %ld), values ~%f.\n", kShape[1], expectVal);
    return 0;
}

int RunQwenFp8Key(aclrtStream stream)
{
    constexpr int64_t kB = 1;
    constexpr int64_t kS1 = 4;
    constexpr int64_t kG = 8;
    constexpr int64_t kD = 128;
    constexpr int64_t kR = 4;
    constexpr int64_t kS2 = 1;
    constexpr int64_t kTopk = 1;
    const int64_t outLen = kTopk * kR + (kR - 1);
    std::vector<int64_t> qShape = {kB, kS1, kG, kD};
    std::vector<int64_t> kShape = {kB, kS2, 1, kD};
    std::vector<int64_t> scaleShape = {kB, kS2};
    std::vector<int64_t> idxShape = {kB, kS1, outLen};
    std::vector<int64_t> valShape = {kB, kS1, kTopk};
    const uint16_t fp16One = FloatToFp16(1.0f);
    const uint8_t fp8One = 0x38;
    std::vector<uint16_t> qHost(GetShapeSize(qShape), fp16One);
    std::vector<uint8_t> kHost(GetShapeSize(kShape), fp8One);
    std::vector<float> scaleHost(GetShapeSize(scaleShape), 1.0f);
    std::vector<int32_t> idxHost(GetShapeSize(idxShape), 0);
    std::vector<float> valHost(GetShapeSize(valShape), 0.0f);
    void *qAddr = nullptr;
    void *kAddr = nullptr;
    void *scaleAddr = nullptr;
    void *idxAddr = nullptr;
    void *valAddr = nullptr;
    aclTensor *q = nullptr;
    aclTensor *k = nullptr;
    aclTensor *scale = nullptr;
    aclTensor *idx = nullptr;
    aclTensor *val = nullptr;
    if (CreateAclTensor(qHost, qShape, &qAddr, aclDataType::ACL_FLOAT16, &q) != 0 ||
        CreateAclTensor(kHost, kShape, &kAddr, aclDataType::ACL_FLOAT8_E4M3FN, &k) != 0 ||
        CreateAclTensor(scaleHost, scaleShape, &scaleAddr, aclDataType::ACL_FLOAT, &scale) != 0 ||
        CreateAclTensor(idxHost, idxShape, &idxAddr, aclDataType::ACL_INT32, &idx) != 0 ||
        CreateAclTensor(valHost, valShape, &valAddr, aclDataType::ACL_FLOAT, &val) != 0) {
        LOG_PRINT("Qwen FP8 CreateAclTensor failed\n");
        return -1;
    }

    char layoutQ[] = "BSND";
    char layoutK[] = "BSND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = aclnnGenericLightningIndexerGetWorkspaceSize(
        q, k, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, scale, nullptr, nullptr, nullptr,
        kTopk, kR, layoutQ, layoutK, 4, 0, false, true, true, -1, true, 0, 0, 1.0f, -1, -1, -1, 0, 0, idx, val,
        &workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Qwen FP8 GetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnGenericLightningIndexer(workspace, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Qwen FP8 execute failed. ERROR: %d\n", ret);
        return ret;
    }
    aclrtSynchronizeStream(stream);

    std::vector<int32_t> idxOut(GetShapeSize(idxShape), -2);
    std::vector<float> valOut(GetShapeSize(valShape), 0.0f);
    ret = aclrtMemcpy(idxOut.data(), idxOut.size() * sizeof(int32_t), idxAddr, idxOut.size() * sizeof(int32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    ret = aclrtMemcpy(valOut.data(), valOut.size() * sizeof(float), valAddr, valOut.size() * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    const int32_t expectIdx[4][7] = {
        {-1, -1, -1, -1, 0, -1, -1},
        {-1, -1, -1, -1, 0, 1, -1},
        {-1, -1, -1, -1, 0, 1, 2},
        {0, 1, 2, 3, -1, -1, -1},
    };
    int32_t badIdx = 0;
    for (int64_t row = 0; row < kS1; ++row) {
        for (int64_t col = 0; col < outLen; ++col) {
            const int32_t got = idxOut[row * outLen + col];
            if (got != expectIdx[row][col]) {
                ++badIdx;
            }
            if (row == kS1 - 1 || col < 8) {
                LOG_PRINT("qwen row %ld col %ld idx=%d expect=%d\n", row, col, got, expectIdx[row][col]);
            }
        }
        LOG_PRINT("qwen row %ld val=%f\n", row, valOut[row]);
    }
    const float expectVal = 1024.0f;
    const bool lastValOk = fabsf(valOut[kS1 - 1] - expectVal) <= 8.0f;
    const bool earlyValOk = !std::isfinite(valOut[0]) && valOut[0] < 0.0f;
    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(scale);
    aclDestroyTensor(idx);
    aclDestroyTensor(val);
    aclrtFree(qAddr);
    aclrtFree(kAddr);
    aclrtFree(scaleAddr);
    aclrtFree(idxAddr);
    aclrtFree(valAddr);
    if (workspace) {
        aclrtFree(workspace);
    }
    if (badIdx > 0 || !lastValOk || !earlyValOk) {
        LOG_PRINT("Qwen FP8 precision check failed: badIdx=%d lastValOk=%d earlyValOk=%d\n", badIdx, lastValOk,
                  earlyValOk);
        return -1;
    }
    LOG_PRINT("Qwen FP8 key example passed.\n");
    return 0;
}

int RunMsaMaxpool(aclrtStream stream)
{
    std::vector<int64_t> qShape = {1, 2, 8, 128};
    std::vector<int64_t> kShape = {1, 32, 1, 128};
    std::vector<int64_t> idxShape = {0};
    std::vector<int64_t> valShape = {1, 2, 8, 16};
    const uint16_t fp16One = FloatToFp16(1.0f);
    std::vector<uint16_t> qHost(GetShapeSize(qShape), fp16One);
    std::vector<uint16_t> kHost(GetShapeSize(kShape), fp16One);
    std::vector<int32_t> idxHost(1, 0);
    std::vector<float> valHost(GetShapeSize(valShape), 0.0f);
    void *qAddr = nullptr;
    void *kAddr = nullptr;
    void *idxAddr = nullptr;
    void *valAddr = nullptr;
    aclTensor *q = nullptr;
    aclTensor *k = nullptr;
    aclTensor *idx = nullptr;
    aclTensor *val = nullptr;
    if (CreateAclTensor(qHost, qShape, &qAddr, aclDataType::ACL_FLOAT16, &q) != 0 ||
        CreateAclTensor(kHost, kShape, &kAddr, aclDataType::ACL_FLOAT16, &k) != 0 ||
        CreateAclTensor(idxHost, idxShape, &idxAddr, aclDataType::ACL_INT32, &idx) != 0 ||
        CreateAclTensor(valHost, valShape, &valAddr, aclDataType::ACL_FLOAT, &val) != 0) {
        LOG_PRINT("MSA CreateAclTensor failed\n");
        return -1;
    }

    char layoutQ[] = "BSND";
    char layoutK[] = "BSND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = aclnnGenericLightningIndexerGetWorkspaceSize(
        q, k, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, 16, 16, layoutQ, layoutK, 0, 1, false, false, false, -1, true, 1, 1, 1.0f, -1, -1, -1, 0, 0, idx, val,
        &workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("MSA GetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnGenericLightningIndexer(workspace, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("MSA execute failed. ERROR: %d\n", ret);
        return ret;
    }
    aclrtSynchronizeStream(stream);

    const int64_t outNum = GetShapeSize(valShape);
    std::vector<float> valOut(outNum, 0.0f);
    ret =
        aclrtMemcpy(valOut.data(), outNum * sizeof(float), valAddr, outNum * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy MSA sparse_values failed. ERROR: %d\n", ret);
        return ret;
    }
    int32_t badVal = 0;
    const float expectBlk = 128.0f;
    const float atol = 4.0f;
    for (int64_t i = 0; i < outNum; ++i) {
        const int64_t blk = i % 16;
        const float v = valOut[i];
        if (i < 32) {
            LOG_PRINT("msa[%ld] val=%f\n", i, v);
        }
        if (blk < 2) {
            if (!(v == v) || fabsf(v - expectBlk) > atol) {
                ++badVal;
            }
        } else if (v > -1.0e30f) {
            ++badVal;
        }
    }
    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(idx);
    aclDestroyTensor(val);
    aclrtFree(qAddr);
    aclrtFree(kAddr);
    aclrtFree(idxAddr);
    aclrtFree(valAddr);
    if (workspace) {
        aclrtFree(workspace);
    }
    if (badVal > 0) {
        LOG_PRINT("MSA precision check failed: badVal=%d expectBlk=%f\n", badVal, expectBlk);
        return -1;
    }
    LOG_PRINT("MSA example passed. valid blocks ~%f, pad -inf.\n", expectBlk);
    return 0;
}

int RunMsaMixedFp16Int8(aclrtStream stream)
{
    std::vector<int64_t> qShape = {1, 2, 8, 128};
    std::vector<int64_t> kShape = {1, 32, 1, 128};
    std::vector<int64_t> scaleShape = {1, 32};
    std::vector<int64_t> idxShape = {0};
    std::vector<int64_t> valShape = {1, 2, 8, 16};
    const uint16_t fp16One = FloatToFp16(1.0f);
    std::vector<uint16_t> qHost(GetShapeSize(qShape), fp16One);
    std::vector<int8_t> kHost(GetShapeSize(kShape), static_cast<int8_t>(1));
    std::vector<float> scaleHost(GetShapeSize(scaleShape), 1.0f);
    std::vector<int32_t> idxHost(1, 0);
    std::vector<float> valHost(GetShapeSize(valShape), 0.0f);
    void *qAddr = nullptr;
    void *kAddr = nullptr;
    void *scaleAddr = nullptr;
    void *idxAddr = nullptr;
    void *valAddr = nullptr;
    aclTensor *q = nullptr;
    aclTensor *k = nullptr;
    aclTensor *kDescale = nullptr;
    aclTensor *idx = nullptr;
    aclTensor *val = nullptr;
    if (CreateAclTensor(qHost, qShape, &qAddr, aclDataType::ACL_FLOAT16, &q) != 0 ||
        CreateAclTensor(kHost, kShape, &kAddr, aclDataType::ACL_INT8, &k) != 0 ||
        CreateAclTensor(scaleHost, scaleShape, &scaleAddr, aclDataType::ACL_FLOAT, &kDescale) != 0 ||
        CreateAclTensor(idxHost, idxShape, &idxAddr, aclDataType::ACL_INT32, &idx) != 0 ||
        CreateAclTensor(valHost, valShape, &valAddr, aclDataType::ACL_FLOAT, &val) != 0) {
        LOG_PRINT("MSA mixed CreateAclTensor failed\n");
        return -1;
    }

    char layoutQ[] = "BSND";
    char layoutK[] = "BSND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = aclnnGenericLightningIndexerGetWorkspaceSize(
        q, k, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, kDescale, nullptr, nullptr,
        nullptr, 16, 16, layoutQ, layoutK, 0, 1, false, false, false, -1, true, 1, 1, 1.0f, -1, -1, -1, 0, 0, idx, val,
        &workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("MSA mixed GetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnGenericLightningIndexer(workspace, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("MSA mixed execute failed. ERROR: %d\n", ret);
        return ret;
    }
    aclrtSynchronizeStream(stream);

    const int64_t outNum = GetShapeSize(valShape);
    std::vector<float> valOut(outNum, 0.0f);
    ret =
        aclrtMemcpy(valOut.data(), outNum * sizeof(float), valAddr, outNum * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy MSA mixed sparse_values failed. ERROR: %d\n", ret);
        return ret;
    }
    int32_t badVal = 0;
    const float expectBlk = 128.0f;
    const float atol = 4.0f;
    for (int64_t i = 0; i < outNum; ++i) {
        const int64_t blk = i % 16;
        const float v = valOut[i];
        if (i < 32) {
            LOG_PRINT("msa_mixed[%ld] val=%f\n", i, v);
        }
        if (blk < 2) {
            if (!(v == v) || fabsf(v - expectBlk) > atol) {
                ++badVal;
            }
        } else if (v > -1.0e30f) {
            ++badVal;
        }
    }
    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(kDescale);
    aclDestroyTensor(idx);
    aclDestroyTensor(val);
    aclrtFree(qAddr);
    aclrtFree(kAddr);
    aclrtFree(scaleAddr);
    aclrtFree(idxAddr);
    aclrtFree(valAddr);
    if (workspace) {
        aclrtFree(workspace);
    }
    if (badVal > 0) {
        LOG_PRINT("MSA mixed precision check failed: badVal=%d expectBlk=%f\n", badVal, expectBlk);
        return -1;
    }
    LOG_PRINT("MSA mixed FP16xINT8 example passed. valid blocks ~%f, pad -inf.\n", expectBlk);
    return 0;
}
} // namespace

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    int ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    ret = RunLiWeightedTopk(stream);
    if (ret == 0) {
        ret = RunMsaMaxpool(stream);
    }
    if (ret == 0) {
        ret = RunMsaMixedFp16Int8(stream);
    }
    if (ret == 0) {
        ret = RunQwenFp8Key(stream);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ret;
}
