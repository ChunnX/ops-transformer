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
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_kv_quant_sparse_flash_attention.h"

namespace {

constexpr int64_t kNopeDim = 512;
constexpr int64_t kRopeDim = 64;
constexpr int64_t kQueryDim = kNopeDim + kRopeDim;
constexpr int64_t kRecordBytes = 400;
constexpr int64_t kFp4Bytes = 256;
constexpr int64_t kRopeOffset = 256;
constexpr int64_t kScaleOffset = 384;
constexpr int64_t kScaleCount = 16;
constexpr int64_t kMxGroupSize = 32;
constexpr int64_t kS2 = 16;

struct ResourceSet {
    std::vector<void *> deviceBuffers;
    std::vector<aclTensor *> tensors;

    ~ResourceSet()
    {
        for (auto *tensor : tensors) {
            aclDestroyTensor(tensor);
        }
        for (auto *buffer : deviceBuffers) {
            aclrtFree(buffer);
        }
    }
};

int64_t ShapeSize(const std::vector<int64_t> &shape)
{
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
}

uint16_t FloatToBf16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>(bits >> 16U);
}

float Bf16ToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float Fp4E2M1Value(uint8_t code)
{
    float sign = (code & 0x8U) == 0U ? 1.0F : -1.0F;
    uint8_t exponent = (code >> 1U) & 0x3U;
    uint8_t mantissa = code & 0x1U;
    if (exponent == 0U) {
        return sign * static_cast<float>(mantissa) * 0.5F;
    }
    return sign * std::ldexp(1.0F + static_cast<float>(mantissa) * 0.5F,
                             static_cast<int>(exponent) - 1);
}

float E8M0Value(uint8_t code)
{
    if (code == 0xFFU) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return std::ldexp(1.0F, static_cast<int>(code) - 127);
}

std::vector<int8_t> MakeRecord(int64_t token)
{
    std::vector<int8_t> record(kRecordBytes, 0);
    for (int64_t d = 0; d < kNopeDim; d += 2) {
        uint8_t low = static_cast<uint8_t>((token + d) % 16);
        uint8_t high = static_cast<uint8_t>((token + d + 1) % 16);
        record[d / 2] = static_cast<int8_t>(low | static_cast<uint8_t>(high << 4U));
    }
    for (int64_t d = 0; d < kRopeDim; ++d) {
        float value = 0.03125F * static_cast<float>(token + 1) * static_cast<float>((d % 9) - 4);
        uint16_t bits = FloatToBf16(value);
        record[kRopeOffset + d * 2] = static_cast<int8_t>(bits & 0xFFU);
        record[kRopeOffset + d * 2 + 1] = static_cast<int8_t>(bits >> 8U);
    }
    for (int64_t group = 0; group < kScaleCount; ++group) {
        record[kScaleOffset + group] = static_cast<int8_t>(126 + ((token + group) % 4));
    }
    return record;
}

std::vector<float> DecodeRecord(const int8_t *record)
{
    std::vector<float> decoded(kQueryDim, 0.0F);
    for (int64_t byte = 0; byte < kFp4Bytes; ++byte) {
        uint8_t packed = static_cast<uint8_t>(record[byte]);
        int64_t group = (byte * 2) / kMxGroupSize;
        float scale = E8M0Value(static_cast<uint8_t>(record[kScaleOffset + group]));
        decoded[byte * 2] = Fp4E2M1Value(packed & 0xFU) * scale;
        decoded[byte * 2 + 1] = Fp4E2M1Value(packed >> 4U) * scale;
    }
    for (int64_t d = 0; d < kRopeDim; ++d) {
        uint16_t bits = static_cast<uint8_t>(record[kRopeOffset + d * 2]);
        bits |= static_cast<uint16_t>(static_cast<uint8_t>(record[kRopeOffset + d * 2 + 1])) << 8U;
        decoded[kNopeDim + d] = Bf16ToFloat(bits);
    }
    return decoded;
}

std::vector<float> CpuReference(const std::vector<uint16_t> &query, const std::vector<int8_t> &records)
{
    std::vector<float> q(kQueryDim);
    std::transform(query.begin(), query.end(), q.begin(), Bf16ToFloat);
    std::vector<std::vector<float>> decoded;
    std::vector<float> scores(kS2, 0.0F);
    decoded.reserve(kS2);
    for (int64_t token = 0; token < kS2; ++token) {
        decoded.push_back(DecodeRecord(records.data() + token * kRecordBytes));
        for (int64_t d = 0; d < kQueryDim; ++d) {
            scores[token] += q[d] * decoded[token][d];
        }
        scores[token] /= std::sqrt(static_cast<float>(kQueryDim));
    }
    float maxScore = *std::max_element(scores.begin(), scores.end());
    float sum = 0.0F;
    for (auto &score : scores) {
        score = std::exp(score - maxScore);
        sum += score;
    }
    for (auto &score : scores) {
        score /= sum;
    }
    std::vector<float> output(kNopeDim, 0.0F);
    for (int64_t token = 0; token < kS2; ++token) {
        for (int64_t d = 0; d < kNopeDim; ++d) {
            output[d] += scores[token] * decoded[token][d];
        }
    }
    return output;
}

template <typename T>
aclTensor *CreateTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, aclDataType dtype,
                        ResourceSet &resources, void **deviceOut = nullptr)
{
    if (static_cast<int64_t>(hostData.size()) != ShapeSize(shape)) {
        return nullptr;
    }
    void *device = nullptr;
    size_t bytes = hostData.size() * sizeof(T);
    if (aclrtMalloc(&device, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return nullptr;
    }
    resources.deviceBuffers.push_back(device);
    if (aclrtMemcpy(device, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        return nullptr;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    aclTensor *tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                                        shape.data(), shape.size(), device);
    if (tensor != nullptr) {
        resources.tensors.push_back(tensor);
    }
    if (deviceOut != nullptr) {
        *deviceOut = device;
    }
    return tensor;
}

int Run()
{
    std::vector<uint16_t> query(kQueryDim);
    for (int64_t d = 0; d < kQueryDim; ++d) {
        query[d] = FloatToBf16(0.015625F * static_cast<float>(((d * 3) % 17) - 8));
    }
    std::vector<int8_t> key;
    key.reserve(kS2 * kRecordBytes);
    for (int64_t token = 0; token < kS2; ++token) {
        auto record = MakeRecord(token);
        key.insert(key.end(), record.begin(), record.end());
    }
    std::vector<int8_t> value = key;
    std::vector<int32_t> sparseIndices(kS2);
    std::iota(sparseIndices.begin(), sparseIndices.end(), 0);
    std::vector<int32_t> actualSeqQ{1};
    std::vector<int32_t> actualSeqKv{static_cast<int32_t>(kS2)};
    std::vector<float> dequantScale{1.0F};
    std::vector<uint16_t> output(kNopeDim, 0);
    auto expected = CpuReference(query, key);

    ResourceSet resources;
    void *outputDevice = nullptr;
    aclTensor *queryTensor = CreateTensor(query, {1, 1, 1, kQueryDim}, ACL_BF16, resources);
    aclTensor *keyTensor = CreateTensor(key, {1, kS2, 1, kRecordBytes}, ACL_INT8, resources);
    aclTensor *valueTensor = CreateTensor(value, {1, kS2, 1, kRecordBytes}, ACL_INT8, resources);
    aclTensor *sparseTensor = CreateTensor(sparseIndices, {1, 1, 1, kS2}, ACL_INT32, resources);
    aclTensor *keyScaleTensor = CreateTensor(dequantScale, {1}, ACL_FLOAT, resources);
    aclTensor *valueScaleTensor = CreateTensor(dequantScale, {1}, ACL_FLOAT, resources);
    aclTensor *actualQTensor = CreateTensor(actualSeqQ, {1}, ACL_INT32, resources);
    aclTensor *actualKvTensor = CreateTensor(actualSeqKv, {1}, ACL_INT32, resources);
    aclTensor *outputTensor = CreateTensor(output, {1, 1, 1, kNopeDim}, ACL_BF16, resources, &outputDevice);
    if (queryTensor == nullptr || keyTensor == nullptr || valueTensor == nullptr || sparseTensor == nullptr ||
        keyScaleTensor == nullptr || valueScaleTensor == nullptr || actualQTensor == nullptr ||
        actualKvTensor == nullptr || outputTensor == nullptr) {
        std::printf("tensor creation failed\n");
        return 1;
    }

    char layoutQuery[] = "BSND";
    char layoutKv[] = "BSND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus status = aclnnKvQuantSparseFlashAttentionGetWorkspaceSize(
        queryTensor, keyTensor, valueTensor, sparseTensor, keyScaleTensor, valueScaleTensor, nullptr, actualQTensor,
        actualKvTensor, 1.0 / std::sqrt(static_cast<double>(kQueryDim)), 2, 2, 1, layoutQuery, layoutKv, 3,
        std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max(), 2, 1, 128, kRopeDim, outputTensor,
        &workspaceSize, &executor);
    if (status != ACL_SUCCESS) {
        std::printf("GetWorkspaceSize failed: %d\n", status);
        return status;
    }

    void *workspace = nullptr;
    if (workspaceSize > 0 && aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        std::printf("workspace allocation failed\n");
        return 1;
    }
    if (workspace != nullptr) {
        resources.deviceBuffers.push_back(workspace);
    }
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        std::printf("stream creation failed\n");
        return 1;
    }
    status = aclnnKvQuantSparseFlashAttention(workspace, workspaceSize, executor, stream);
    if (status == ACL_SUCCESS) {
        status = aclrtSynchronizeStream(stream);
    }
    aclrtDestroyStream(stream);
    if (status != ACL_SUCCESS) {
        std::printf("operator execution failed: %d\n", status);
        return status;
    }

    if (aclrtMemcpy(output.data(), output.size() * sizeof(uint16_t), outputDevice,
                    output.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        std::printf("output copy failed\n");
        return 1;
    }

    int64_t closeCount = 0;
    float maxAbsError = 0.0F;
    for (int64_t i = 0; i < kNopeDim; ++i) {
        float actual = Bf16ToFloat(output[i]);
        float error = std::abs(actual - expected[i]);
        float tolerance = 0.05F + 0.02F * std::abs(expected[i]);
        closeCount += error <= tolerance;
        maxAbsError = std::max(maxAbsError, error);
    }
    double closeRatio = static_cast<double>(closeCount) / static_cast<double>(kNopeDim);
    std::printf("BSND BF16 full16: close_ratio=%.6f, max_abs_error=%.6f\n", closeRatio, maxAbsError);
    return closeRatio >= 0.99 ? 0 : 1;
}

}  // namespace

int main()
{
    constexpr int32_t deviceId = 0;
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        std::printf("ACL initialization failed\n");
        return 1;
    }
    int result = Run();
    aclrtResetDevice(deviceId);
    aclFinalize();
    return result;
}
