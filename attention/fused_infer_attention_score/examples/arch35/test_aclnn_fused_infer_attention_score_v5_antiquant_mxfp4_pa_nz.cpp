/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * Direct ACLNN numerical example for FLOAT4_E2M1 mode-6 PageAttention NZ.
 *
 * The important contracts exercised here are:
 *   - KV cache:       [P, N, D / 16, Bs, 16] (FLOAT4_E2M1 PA_NZ)
 *   - scale / offset: [P, N, D / 32, Bs]
 *   - blockTable:     non-identity mappings through one, two, and three logical pages
 *
 * The example runs FP16/BF16 symmetric and asymmetric cases.
 * Its CPU reference follows the logical sequence through blockTable, so a wrong
 * physical-page or group-plane address changes the checked attention output.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v5.h"

namespace {

constexpr int64_t B = 2;
constexpr int64_t N = 2;
constexpr int64_t S1 = 1;
constexpr int64_t D = 128;
constexpr int64_t G = D / 32;
constexpr int64_t BS = 128;
constexpr int64_t P = 6;
constexpr int64_t D0 = 16;
constexpr float QUERY_VALUE = 0.0625F;

enum class QType : uint8_t { FP16, BF16 };

struct CaseConfig {
    const char *name;
    QType qType;
    bool withOffset;
    std::vector<int64_t> actualKvLens;
    std::vector<int32_t> blockTable;
    int64_t blocksPerBatch;
};

int64_t ShapeSize(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

float Fp16ToFloat(uint16_t value)
{
    const uint32_t sign = (value >> 15U) & 1U;
    uint32_t exponent = (value >> 10U) & 0x1FU;
    uint32_t mantissa = value & 0x3FFU;
    uint32_t bits = 0;
    if (exponent == 0x1FU) {
        bits = (sign << 31U) | (0xFFU << 23U) | (mantissa == 0 ? 0U : 0x7FFFFFU);
    } else if (exponent == 0U) {
        if (mantissa != 0U) {
            exponent = 113U;
            while ((mantissa & 0x400U) == 0U) {
                mantissa <<= 1U;
                --exponent;
            }
            mantissa &= 0x3FFU;
            bits = (sign << 31U) | (exponent << 23U) | (mantissa << 13U);
        } else {
            bits = sign << 31U;
        }
    } else {
        bits = (sign << 31U) | ((exponent + 112U) << 23U) | (mantissa << 13U);
    }
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t FloatToBf16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    // Round to nearest even before discarding the low 16 bits.
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>(bits >> 16U);
}

float Bf16ToFloat(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t EncodeQ(float value, QType type)
{
    return type == QType::FP16 ? static_cast<uint16_t>(aclFloatToFloat16(value)) : FloatToBf16(value);
}

float DecodeQ(uint16_t value, QType type)
{
    return type == QType::FP16 ? Fp16ToFloat(value) : Bf16ToFloat(value);
}

float RoundToBf16(float value)
{
    return Bf16ToFloat(FloatToBf16(value));
}

aclDataType ToAclType(QType type)
{
    return type == QType::FP16 ? aclDataType::ACL_FLOAT16 : aclDataType::ACL_BF16;
}

float E8M0ToFloat(uint8_t exponent)
{
    return std::ldexp(1.0F, static_cast<int>(exponent) - 127);
}

uint8_t ScaleCode(int64_t physicalBlock, int64_t head, int64_t group, int64_t token, int64_t salt)
{
    // 0x7e, 0x7f, 0x80 represent 0.5, 1.0 and 2.0 respectively.
    return static_cast<uint8_t>(126 + ((physicalBlock + 2 * head + 3 * group + token + salt) % 3));
}

float OffsetValue(int64_t physicalBlock, int64_t head, int64_t group, int64_t token, int64_t salt)
{
    // Every operand is a binary fraction, exactly representable in both FP16 and BF16.
    // The negative values make a missing Add or an accidental unsigned reinterpretation observable.
    return 0.0625F * static_cast<float>(-6 + physicalBlock + 2 * head + 3 * group + (token % 7) + salt);
}

float DequantizeReference(float decodedFp4, uint8_t scaleCode, uint16_t offset, bool withOffset, QType qType)
{
    // Match the W4 VF contract: operands and arithmetic are BF16, then the established Q_T store path applies.
    float value = RoundToBf16(decodedFp4);
    if (withOffset) {
        value = RoundToBf16(value + RoundToBf16(DecodeQ(offset, qType)));
    }
    value = RoundToBf16(value * RoundToBf16(E8M0ToFloat(scaleCode)));
    return DecodeQ(EncodeQ(value, qType), qType);
}

int CreateTensor(const std::vector<uint16_t> &host, const std::vector<int64_t> &shape, aclDataType type,
                 void **device, aclTensor **tensor)
{
    const size_t bytes = host.size() * sizeof(uint16_t);
    aclError ret = aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(*device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*device);
        *device = nullptr;
        return ret;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t index = static_cast<int64_t>(shape.size()) - 2; index >= 0; --index) {
        strides[index] = shape[index + 1] * strides[index + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), type, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *device);
    return *tensor == nullptr ? ACL_ERROR_INTERNAL_ERROR : ACL_SUCCESS;
}

int CreateByteTensor(const std::vector<uint8_t> &host, const std::vector<int64_t> &shape, aclDataType type,
                     void **device, aclTensor **tensor)
{
    const size_t bytes = host.size();
    aclError ret = aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(*device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*device);
        *device = nullptr;
        return ret;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t index = static_cast<int64_t>(shape.size()) - 2; index >= 0; --index) {
        strides[index] = shape[index + 1] * strides[index + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), type, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *device);
    return *tensor == nullptr ? ACL_ERROR_INTERNAL_ERROR : ACL_SUCCESS;
}

int CreateBlockTableTensor(const std::vector<int32_t> &host, const std::vector<int64_t> &shape, void **device,
                           aclTensor **tensor)
{
    const size_t bytes = host.size() * sizeof(int32_t);
    aclError ret = aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(*device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*device);
        *device = nullptr;
        return ret;
    }
    const int64_t strides[] = {shape[1], 1};
    *tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_INT32, strides, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *device);
    return *tensor == nullptr ? ACL_ERROR_INTERNAL_ERROR : ACL_SUCCESS;
}

void DestroyTensor(aclTensor *tensor, void *device)
{
    if (tensor != nullptr) {
        aclDestroyTensor(tensor);
    }
    if (device != nullptr) {
        aclrtFree(device);
    }
}

bool RunCase(aclrtStream stream, const CaseConfig &config)
{
    const char *caseName = config.name;
    const QType qType = config.qType;
    const bool withOffset = config.withOffset;
    if (config.actualKvLens.size() != static_cast<size_t>(B) ||
        config.blockTable.size() != static_cast<size_t>(B * config.blocksPerBatch)) {
        std::printf("%s: invalid case configuration\\n", caseName);
        return false;
    }
    for (int64_t length : config.actualKvLens) {
        if (length <= 0 || length > config.blocksPerBatch * BS) {
            std::printf("%s: actual KV length is outside blockTable capacity\\n", caseName);
            return false;
        }
    }
    for (int32_t physicalBlock : config.blockTable) {
        if (physicalBlock < 0 || physicalBlock >= P) {
            std::printf("%s: blockTable contains an invalid physical page\\n", caseName);
            return false;
        }
    }
    const std::vector<int64_t> queryShape = {B, N, S1, D};
    const std::vector<int64_t> kvNzShape = {P, N, D / D0, BS, D0};
    const std::vector<int64_t> antiqParamShape = {P, N, G, BS};
    const std::vector<int64_t> blockTableShape = {B, config.blocksPerBatch};
    const std::vector<int32_t> &blockTable = config.blockTable;
    const std::vector<int64_t> actualQLens = {1, 1};
    const std::vector<int64_t> &actualKvLens = config.actualKvLens;

    std::vector<uint16_t> query(ShapeSize(queryShape), EncodeQ(QUERY_VALUE, qType));
    std::vector<uint16_t> output(ShapeSize(queryShape), EncodeQ(0.0F, qType));
    // FLOAT4 storage is packed: each byte holds E2M1 values 0 (low nibble) and 1 (high nibble).
    // The CPU reference consumes the matching logical D parity below, so both packed nibbles are exercised.
    std::vector<uint8_t> key(ShapeSize(kvNzShape) / 2, 0x20U);
    std::vector<uint8_t> value = key;
    std::vector<uint8_t> keyScale(ShapeSize(antiqParamShape), 0);
    std::vector<uint8_t> valueScale(ShapeSize(antiqParamShape), 0);
    std::vector<uint16_t> keyOffset(ShapeSize(antiqParamShape), EncodeQ(0.0F, qType));
    std::vector<uint16_t> valueOffset(ShapeSize(antiqParamShape), EncodeQ(0.0F, qType));
    for (int64_t p = 0; p < P; ++p) {
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t g = 0; g < G; ++g) {
                for (int64_t token = 0; token < BS; ++token) {
                    const int64_t index = ((p * N + n) * G + g) * BS + token;
                    keyScale[index] = ScaleCode(p, n, g, token, 0);
                    valueScale[index] = ScaleCode(p, n, g, token, 1);
                    keyOffset[index] = EncodeQ(OffsetValue(p, n, g, token, 0), qType);
                    valueOffset[index] = EncodeQ(OffsetValue(p, n, g, token, 1), qType);
                }
            }
        }
    }

    void *queryDevice = nullptr;
    void *keyDevice = nullptr;
    void *valueDevice = nullptr;
    void *keyScaleDevice = nullptr;
    void *valueScaleDevice = nullptr;
    void *keyOffsetDevice = nullptr;
    void *valueOffsetDevice = nullptr;
    void *blockTableDevice = nullptr;
    void *outputDevice = nullptr;
    void *workspace = nullptr;
    aclTensor *queryTensor = nullptr;
    aclTensor *keyTensor = nullptr;
    aclTensor *valueTensor = nullptr;
    aclTensor *keyScaleTensor = nullptr;
    aclTensor *valueScaleTensor = nullptr;
    aclTensor *keyOffsetTensor = nullptr;
    aclTensor *valueOffsetTensor = nullptr;
    aclTensor *blockTableTensor = nullptr;
    aclTensor *outputTensor = nullptr;
    aclTensorList *keyList = nullptr;
    aclTensorList *valueList = nullptr;
    aclIntArray *actualQArray = nullptr;
    aclIntArray *actualKvArray = nullptr;
    aclOpExecutor *executor = nullptr;

    const auto cleanup = [&]() {
        if (keyList != nullptr) {
            aclDestroyTensorList(keyList);
        }
        if (valueList != nullptr) {
            aclDestroyTensorList(valueList);
        }
        if (actualQArray != nullptr) {
            aclDestroyIntArray(actualQArray);
        }
        if (actualKvArray != nullptr) {
            aclDestroyIntArray(actualKvArray);
        }
        DestroyTensor(queryTensor, queryDevice);
        DestroyTensor(keyTensor, keyDevice);
        DestroyTensor(valueTensor, valueDevice);
        DestroyTensor(keyScaleTensor, keyScaleDevice);
        DestroyTensor(valueScaleTensor, valueScaleDevice);
        DestroyTensor(keyOffsetTensor, keyOffsetDevice);
        DestroyTensor(valueOffsetTensor, valueOffsetDevice);
        DestroyTensor(blockTableTensor, blockTableDevice);
        DestroyTensor(outputTensor, outputDevice);
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
    };

    int ret = CreateTensor(query, queryShape, ToAclType(qType), &queryDevice, &queryTensor);
    if (ret != ACL_SUCCESS) {
        std::printf("%s: create query failed: %d\n", caseName, ret);
        cleanup();
        return false;
    }
    ret = CreateByteTensor(key, kvNzShape, aclDataType::ACL_FLOAT4_E2M1, &keyDevice, &keyTensor);
    if (ret == ACL_SUCCESS) {
        ret = CreateByteTensor(value, kvNzShape, aclDataType::ACL_FLOAT4_E2M1, &valueDevice, &valueTensor);
    }
    if (ret == ACL_SUCCESS) {
        ret = CreateByteTensor(keyScale, antiqParamShape, aclDataType::ACL_FLOAT8_E8M0, &keyScaleDevice, &keyScaleTensor);
    }
    if (ret == ACL_SUCCESS) {
        ret = CreateByteTensor(valueScale, antiqParamShape, aclDataType::ACL_FLOAT8_E8M0, &valueScaleDevice, &valueScaleTensor);
    }
    if (ret == ACL_SUCCESS && withOffset) {
        ret = CreateTensor(keyOffset, antiqParamShape, ToAclType(qType), &keyOffsetDevice, &keyOffsetTensor);
    }
    if (ret == ACL_SUCCESS && withOffset) {
        ret = CreateTensor(valueOffset, antiqParamShape, ToAclType(qType), &valueOffsetDevice, &valueOffsetTensor);
    }
    if (ret == ACL_SUCCESS) {
        ret = CreateBlockTableTensor(blockTable, blockTableShape, &blockTableDevice, &blockTableTensor);
    }
    if (ret == ACL_SUCCESS) {
        ret = CreateTensor(output, queryShape, ToAclType(qType), &outputDevice, &outputTensor);
    }
    if (ret != ACL_SUCCESS) {
        std::printf("%s: create input tensor failed: %d\n", caseName, ret);
        cleanup();
        return false;
    }

    keyList = aclCreateTensorList(&keyTensor, 1);
    valueList = aclCreateTensorList(&valueTensor, 1);
    actualQArray = aclCreateIntArray(actualQLens.data(), actualQLens.size());
    actualKvArray = aclCreateIntArray(actualKvLens.data(), actualKvLens.size());
    if (keyList == nullptr || valueList == nullptr || actualQArray == nullptr || actualKvArray == nullptr) {
        std::printf("%s: create tensor list or sequence-length array failed\n", caseName);
        cleanup();
        return false;
    }

    uint64_t workspaceSize = 0;
    char inputLayout[] = "BNSD";
    ret = aclnnFusedInferAttentionScoreV5GetWorkspaceSize(
        queryTensor, keyList, valueList,
        nullptr, nullptr, actualQArray, actualKvArray,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        blockTableTensor, nullptr, nullptr,
        keyScaleTensor, withOffset ? keyOffsetTensor : nullptr,
        valueScaleTensor, withOffset ? valueOffsetTensor : nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        N, 1.0 / std::sqrt(static_cast<double>(D)), 65536, 65536, inputLayout, N, 0, 1, BS, 0, false, 6, 6, 0, 0,
        outputTensor, nullptr, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        std::printf("%s: GetWorkspaceSize failed: %d\n", caseName, ret);
        cleanup();
        return false;
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            std::printf("%s: workspace allocation failed: %d\n", caseName, ret);
            cleanup();
            return false;
        }
    }
    ret = aclnnFusedInferAttentionScoreV5(workspace, workspaceSize, executor, stream);
    if (ret == ACL_SUCCESS) {
        ret = aclrtSynchronizeStream(stream);
    }
    if (ret != ACL_SUCCESS) {
        std::printf("%s: device execution failed: %d\n", caseName, ret);
        cleanup();
        return false;
    }
    ret = aclrtMemcpy(output.data(), output.size() * sizeof(uint16_t), outputDevice,
                      output.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        std::printf("%s: copy output failed: %d\n", caseName, ret);
        cleanup();
        return false;
    }

    float maxError = 0.0F;
    const float queryValue = DecodeQ(EncodeQ(QUERY_VALUE, qType), qType);
    for (int64_t b = 0; b < B; ++b) {
        const int64_t validS = actualKvLens[b];
        for (int64_t n = 0; n < N; ++n) {
            std::vector<float> scores(validS, 0.0F);
            float maxScore = -INFINITY;
            for (int64_t s = 0; s < validS; ++s) {
                const int64_t physicalBlock = blockTable[b * config.blocksPerBatch + s / BS];
                const int64_t token = s % BS;
                float score = 0.0F;
                for (int64_t d = 0; d < D; ++d) {
                    const int64_t group = d / 32;
                    const int64_t index = ((physicalBlock * N + n) * G + group) * BS + token;
                    const float decodedFp4 = (d & 1) == 0 ? 0.0F : 1.0F;
                    const float keyValue = DequantizeReference(decodedFp4, keyScale[index], keyOffset[index], withOffset,
                                                               qType);
                    score += queryValue * keyValue;
                }
                scores[s] = score / std::sqrt(static_cast<float>(D));
                maxScore = std::max(maxScore, scores[s]);
            }
            float sumExp = 0.0F;
            for (float &score : scores) {
                score = std::exp(score - maxScore);
                sumExp += score;
            }
            for (int64_t d = 0; d < D; ++d) {
                const int64_t group = d / 32;
                float expected = 0.0F;
                for (int64_t s = 0; s < validS; ++s) {
                    const int64_t physicalBlock = blockTable[b * config.blocksPerBatch + s / BS];
                    const int64_t token = s % BS;
                    const int64_t index = ((physicalBlock * N + n) * G + group) * BS + token;
                    const float decodedFp4 = (d & 1) == 0 ? 0.0F : 1.0F;
                    const float valueValue = DequantizeReference(decodedFp4, valueScale[index], valueOffset[index],
                                                                 withOffset, qType);
                    expected += (scores[s] / sumExp) * valueValue;
                }
                const int64_t outIndex = ((b * N + n) * S1) * D + d;
                maxError = std::max(maxError, std::fabs(DecodeQ(output[outIndex], qType) - expected));
            }
        }
    }
    const float tolerance = qType == QType::FP16 ? 0.04F : 0.08F;
    std::printf("%s: max_abs_error=%f, tolerance=%f\n", caseName, maxError, tolerance);
    cleanup();
    return maxError <= tolerance;
}

} // namespace

int main()
{
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::printf("aclInit failed: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(0);
    if (ret != ACL_SUCCESS) {
        std::printf("aclrtSetDevice failed: %d\n", ret);
        aclFinalize();
        return ret;
    }
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        std::printf("aclrtCreateStream failed: %d\n", ret);
        aclrtResetDevice(0);
        aclFinalize();
        return ret;
    }

    const std::vector<CaseConfig> cases = {
        // One page, but uses token 126 and a nonzero physical page: catches bad first-page and tail-token addressing.
        {"fp16_symmetric_single_page_tail", QType::FP16, false, {127, 1}, {5, 1}, 1},
        // Two pages with reversed/non-identity mappings: catches the first page transition and blockTable indirection.
        {"fp16_asymmetric_two_page_remap", QType::FP16, true, {130, 129}, {2, 0, 3, 1}, 2},
        {"bf16_symmetric_two_page_remap", QType::BF16, false, {129, 130}, {4, 1, 5, 0}, 2},
        // A third logical page is deliberately only one token long; each batch uses a distinct, non-monotonic mapping.
        {"fp16_symmetric_three_page_tail", QType::FP16, false, {257, 256}, {5, 2, 0, 4, 1, 3}, 3},
        {"bf16_asymmetric_three_page_tail", QType::BF16, true, {257, 256}, {5, 2, 0, 4, 1, 3}, 3},
    };

    bool passed = true;
    for (const CaseConfig &config : cases) {
        // Keep running after a failure so one invocation reports every failing coverage dimension.
        passed = RunCase(stream, config) && passed;
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    if (!passed) {
        std::printf("MXFP4 mode-6 PA_NZ numerical example FAILED\n");
        return 1;
    }
    std::printf("MXFP4 mode-6 PA_NZ numerical example PASSED\n");
    return 0;
}
