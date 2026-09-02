/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_sliding_tile_attention.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

extern "C" int InitHugeMemThreadLocal(void *reserved, bool enableCache);
extern "C" void ReleaseHugeMem(void *reserved, bool enableCache);
extern "C" void UnInitHugeMemThreadLocal(void *reserved, bool enableCache);

namespace {

constexpr int64_t LEGACY_HEAD_COUNT = 8;
constexpr int64_t TILE_TIME = 6;
constexpr int64_t TILE_HEIGHT = 8;
constexpr int64_t TILE_WIDTH = 8;
constexpr int64_t TILE_ELEMENTS = TILE_TIME * TILE_HEIGHT * TILE_WIDTH;
constexpr int64_t FASTVIDEO_TEXT_LENGTH = 128;
constexpr int64_t WINDOW_VALUES_PER_HEAD = 3;
constexpr float Q_VALUE_BASE = 1.0F / 8.0F;
constexpr float Q_VALUE_SCALE = 1.0F / 16.0F;
constexpr float K_VALUE_BASE = 1.0F / 8.0F;
constexpr float K_VALUE_SCALE = 1.0F / 16.0F;
constexpr float TOKEN_VALUE_BASE = 1.0F / 8.0F;
constexpr float TILE_VALUE_SCALE = 1.0F / 64.0F;
constexpr float DIM_VALUE_SCALE = 1.0F / 128.0F;
constexpr float FP16_TOLERANCE = 3.0e-3F;
constexpr float BF16_TOLERANCE = 2.0e-2F;
constexpr int DEFAULT_WARMUP_ITERATIONS = 3;
constexpr int DEFAULT_MEASURE_ITERATIONS = 5;

using WindowValues = std::vector<int64_t>;

const WindowValues SOURCE_WINDOW_PATTERN = {
    1, 1, 1, 3, 3, 3, 3, 5, 5, 5, 5, 5, 1, 3, 3, 3, 3, 5, 5, 5, 7, 7, 7, 7,
};
const WindowValues WIDE_WINDOW_PATTERN = {
    7, 7, 7, 7, 5, 7, 5, 7, 7, 5, 5, 7, 7, 3, 5, 3, 7, 5, 5, 7, 3, 7, 7, 5,
};
const WindowValues UNIFORM_333_WINDOW_PATTERN = {
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
};
const WindowValues WAN_MIXED_WINDOW_PATTERN = {
    3, 3, 5, 3, 1, 9, 1, 3, 5, 5, 5, 5, 3, 5, 7, 5, 3, 7, 1, 1, 1, 7, 7, 7,
};
const WindowValues GENERAL_N3_WINDOW_PATTERN = {
    1, 1, 1, 3, 3, 3, 5, 5, 5,
};
constexpr int64_t WIDE_INT64_WINDOW = (1LL << 32) + 1;
const WindowValues TASK_N3_PER_AXIS_WINDOW_PATTERN = {
    WIDE_INT64_WINDOW, 1, 1, 1, 3, 1, 1, 1, 5,
};
const WindowValues GENERAL_N3_WIDE_AXIS_WINDOW_PATTERN = {
    WIDE_INT64_WINDOW, 3, 5, 5, WIDE_INT64_WINDOW, 7, 7, 9, WIDE_INT64_WINDOW,
};
const WindowValues WIDE_INT64_WINDOW_PATTERN = {
    WIDE_INT64_WINDOW, WIDE_INT64_WINDOW, WIDE_INT64_WINDOW, WIDE_INT64_WINDOW, WIDE_INT64_WINDOW, WIDE_INT64_WINDOW,
    WIDE_INT64_WINDOW, WIDE_INT64_WINDOW, WIDE_INT64_WINDOW, WIDE_INT64_WINDOW, WIDE_INT64_WINDOW, WIDE_INT64_WINDOW,
    WIDE_INT64_WINDOW, WIDE_INT64_WINDOW, WIDE_INT64_WINDOW, WIDE_INT64_WINDOW, WIDE_INT64_WINDOW, WIDE_INT64_WINDOW,
    WIDE_INT64_WINDOW, WIDE_INT64_WINDOW, WIDE_INT64_WINDOW, WIDE_INT64_WINDOW, WIDE_INT64_WINDOW, WIDE_INT64_WINDOW,
};

enum class DType {
    FP16,
    BF16,
};

struct Canvas {
    int64_t time;
    int64_t height;
    int64_t width;
};

struct CaseSpec {
    int id;
    DType dtype;
    int64_t batch;
    int64_t heads;
    int64_t headDim;
    Canvas canvas;
    int64_t textLength;
    int64_t seqShape;
    WindowValues windows;
    bool useBroadcastWindow;
    bool useNonContiguousStorage;
    int64_t inputSequence = 0;
};

struct DeviceTensor {
    void *data = nullptr;
    aclTensor *tensor = nullptr;
    size_t bytes = 0;
    std::vector<int64_t> strides;
};

struct SamplePoint {
    int64_t batch;
    int64_t head;
    int64_t query;
    int64_t dim;
};

struct WindowRange {
    int64_t tStart;
    int64_t tEnd;
    int64_t hStart;
    int64_t hEnd;
    int64_t wStart;
    int64_t wEnd;
};

struct PrecisionStats {
    uint64_t samples = 0;
    double sumAbsoluteError = 0.0;
    double sumSquaredError = 0.0;
    float maxAbsoluteError = 0.0F;
};

struct TimingStats {
    float minUs = 0.0F;
    float medianUs = 0.0F;
    float maxUs = 0.0F;
};

bool BeginHugeMem(bool &initialized)
{
    const int ret = InitHugeMemThreadLocal(nullptr, false);
    if (ret != 0) {
        std::fprintf(stderr, "InitHugeMemThreadLocal failed: %d\n", ret);
        return false;
    }
    initialized = true;
    return true;
}

void EndHugeMem(bool &initialized)
{
    if (!initialized) {
        return;
    }
    ReleaseHugeMem(nullptr, false);
    UnInitHugeMemThreadLocal(nullptr, false);
    initialized = false;
}

uint16_t FloatToHalf(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16U) & 0x8000U;
    int32_t exponent = static_cast<int32_t>((bits >> 23U) & 0xFFU) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFFU;
    if (exponent <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    mantissa += 0x1000U;
    if ((mantissa & 0x800000U) != 0U) {
        mantissa = 0;
        ++exponent;
        if (exponent >= 31) {
            return static_cast<uint16_t>(sign | 0x7C00U);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10U) | (mantissa >> 13U));
}

float HalfToFloat(uint16_t value)
{
    const uint32_t sign = (static_cast<uint32_t>(value) & 0x8000U) << 16U;
    uint32_t exponent = (static_cast<uint32_t>(value) >> 10U) & 0x1FU;
    uint32_t mantissa = static_cast<uint32_t>(value) & 0x3FFU;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int32_t normalizedExponent = 127 - 15 + 1;
            while ((mantissa & 0x400U) == 0U) {
                mantissa <<= 1U;
                --normalizedExponent;
            }
            mantissa &= 0x3FFU;
            bits = sign | (static_cast<uint32_t>(normalizedExponent) << 23U) | (mantissa << 13U);
        }
    } else if (exponent == 0x1FU) {
        bits = sign | 0x7F800000U | (mantissa << 13U);
    } else {
        bits = sign | ((exponent - 15U + 127U) << 23U) | (mantissa << 13U);
    }
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t FloatToBf16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t roundingBias = 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>((bits + roundingBias) >> 16U);
}

float Bf16ToFloat(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t FloatToStorage(float value, DType dtype)
{
    return dtype == DType::FP16 ? FloatToHalf(value) : FloatToBf16(value);
}

float StorageToFloat(uint16_t value, DType dtype)
{
    return dtype == DType::FP16 ? HalfToFloat(value) : Bf16ToFloat(value);
}

const char *DTypeName(DType dtype)
{
    return dtype == DType::FP16 ? "FP16" : "BF16";
}

aclDataType AclDType(DType dtype)
{
    return dtype == DType::FP16 ? ACL_FLOAT16 : ACL_BF16;
}

int64_t ImageSequence(const Canvas &canvas)
{
    return canvas.time * canvas.height * canvas.width;
}

int64_t Sequence(const CaseSpec &spec)
{
    return ImageSequence(spec.canvas) + spec.textLength;
}

int64_t InputSequence(const CaseSpec &spec)
{
    return spec.inputSequence > 0 ? spec.inputSequence : Sequence(spec);
}

std::vector<int64_t> ContiguousStrides(const std::vector<int64_t> &shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t index = static_cast<int64_t>(shape.size()) - 2; index >= 0; --index) {
        strides[index] = strides[index + 1] * shape[index + 1];
    }
    return strides;
}

bool CreateTensor(const std::vector<int64_t> &shape, aclDataType dtype, DeviceTensor &output,
                  int64_t storageHeadDim = 0)
{
    std::vector<int64_t> storageShape = shape;
    if (storageHeadDim > shape.back()) {
        storageShape.back() = storageHeadDim;
    }
    int64_t elementCount = 1;
    for (int64_t dim : storageShape) {
        elementCount *= dim;
    }
    output.bytes = static_cast<size_t>(elementCount) * sizeof(uint16_t);
    if (output.bytes > 0) {
        auto ret = aclrtMalloc(&output.data, output.bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            std::fprintf(stderr, "aclrtMalloc failed: %d, bytes=%zu\n", ret, output.bytes);
            return false;
        }
    }
    output.strides = ContiguousStrides(storageShape);
    output.tensor = aclCreateTensor(shape.data(), shape.size(), dtype, output.strides.data(), 0, ACL_FORMAT_ND,
                                    storageShape.data(), storageShape.size(), output.data);
    if (output.tensor == nullptr) {
        std::fprintf(stderr, "aclCreateTensor failed\n");
        aclrtFree(output.data);
        output.data = nullptr;
        return false;
    }
    return true;
}

void DestroyTensor(DeviceTensor &tensor)
{
    if (tensor.tensor != nullptr) {
        aclDestroyTensor(tensor.tensor);
        tensor.tensor = nullptr;
    }
    if (tensor.data != nullptr) {
        aclrtFree(tensor.data);
        tensor.data = nullptr;
    }
    tensor.strides.clear();
}

float TokenValue(int64_t token)
{
    const int64_t tile = token / TILE_ELEMENTS;
    return TOKEN_VALUE_BASE + static_cast<float>(tile & 15LL) * TILE_VALUE_SCALE;
}

float QueryValue(int64_t token, int64_t dim)
{
    const int64_t tile = token / TILE_ELEMENTS;
    return Q_VALUE_BASE + static_cast<float>((tile + dim) & 3LL) * Q_VALUE_SCALE;
}

float KeyValue(int64_t token, int64_t dim)
{
    const int64_t tile = token / TILE_ELEMENTS;
    return K_VALUE_BASE + static_cast<float>((3 * tile + dim) & 7LL) * K_VALUE_SCALE;
}

float DimValue(int64_t dim)
{
    return static_cast<float>(dim & 3LL) * DIM_VALUE_SCALE;
}

uint16_t VStorageValue(int64_t token, int64_t dim, DType dtype)
{
    return FloatToStorage(TokenValue(token) + DimValue(dim), dtype);
}

float QuantizedValue(float value, DType dtype)
{
    return StorageToFloat(FloatToStorage(value, dtype), dtype);
}

bool InitializeInputs(const CaseSpec &spec, const DeviceTensor &q, const DeviceTensor &k, const DeviceTensor &v,
                      const DeviceTensor &output)
{
    const std::array<std::pair<const char *, const DeviceTensor *>, 4> tensors = {{
        {"q", &q},
        {"k", &k},
        {"v", &v},
        {"output", &output},
    }};
    for (const auto &[name, tensor] : tensors) {
        const auto ret = aclrtMemset(tensor->data, tensor->bytes, 0, tensor->bytes);
        if (ret != ACL_SUCCESS) {
            std::fprintf(stderr, "aclrtMemset(%s) failed: %d\n", name, ret);
            return false;
        }
    }

    const int64_t sequence = Sequence(spec);
    const int64_t storageHeadDim = v.strides[v.strides.size() - 2];
    const size_t rowBytes = static_cast<size_t>(storageHeadDim) * sizeof(uint16_t);
    std::vector<uint16_t> hostQ(static_cast<size_t>(sequence) * storageHeadDim, 0);
    std::vector<uint16_t> hostK(static_cast<size_t>(sequence) * storageHeadDim, 0);
    std::vector<uint16_t> hostV(static_cast<size_t>(sequence) * storageHeadDim, 0);
    for (int64_t token = 0; token < sequence; ++token) {
        for (int64_t dim = 0; dim < spec.headDim; ++dim) {
            const size_t index = static_cast<size_t>(token) * storageHeadDim + dim;
            hostQ[index] = FloatToStorage(QueryValue(token, dim), spec.dtype);
            hostK[index] = FloatToStorage(KeyValue(token, dim), spec.dtype);
            hostV[index] = VStorageValue(token, dim, spec.dtype);
        }
    }
    const size_t headBytes = hostQ.size() * sizeof(uint16_t);
    const size_t deviceHeadStrideBytes = static_cast<size_t>(q.strides[1]) * sizeof(uint16_t);
    for (int64_t batch = 0; batch < spec.batch; ++batch) {
        for (int64_t head = 0; head < spec.heads; ++head) {
            const int64_t headIndex = batch * spec.heads + head;
            const size_t headOffset = static_cast<size_t>(headIndex) * deviceHeadStrideBytes;
            const std::array<std::pair<const char *, const std::vector<uint16_t> *>, 3> inputs = {{
                {"q", &hostQ},
                {"k", &hostK},
                {"v", &hostV},
            }};
            const std::array<const DeviceTensor *, 3> destinations = {{&q, &k, &v}};
            for (size_t inputIndex = 0; inputIndex < inputs.size(); ++inputIndex) {
                auto *headDestination = static_cast<uint8_t *>(destinations[inputIndex]->data) + headOffset;
                const auto ret = aclrtMemcpy(headDestination, headBytes, inputs[inputIndex].second->data(), headBytes,
                                             ACL_MEMCPY_HOST_TO_DEVICE);
                if (ret != ACL_SUCCESS) {
                    std::fprintf(stderr, "aclrtMemcpy(%s) failed: %d, batch=%ld, head=%ld, rowBytes=%zu\n",
                                 inputs[inputIndex].first, ret, static_cast<long>(batch), static_cast<long>(head),
                                 rowBytes);
                    return false;
                }
            }
        }
    }
    return true;
}

int64_t OddWindowStart(int64_t coord, int64_t window, int64_t extent)
{
    if (window >= extent) {
        return 0;
    }
    const int64_t radius = window / 2;
    if (coord <= radius) {
        return 0;
    }
    const int64_t start = coord - radius;
    const int64_t maxStart = extent - window;
    return start > maxStart ? maxStart : start;
}

WindowRange GetWindowRange(const CaseSpec &spec, int64_t head, int64_t query)
{
    const int64_t gridT = spec.canvas.time / TILE_TIME;
    const int64_t gridH = spec.canvas.height / TILE_HEIGHT;
    const int64_t gridW = spec.canvas.width / TILE_WIDTH;
    const int64_t tileId = query / TILE_ELEMENTS;
    const int64_t queryT = tileId / (gridH * gridW);
    const int64_t remain = tileId - queryT * gridH * gridW;
    const int64_t queryH = remain / gridW;
    const int64_t queryW = remain - queryH * gridW;
    const int64_t offset = spec.useBroadcastWindow ? 0 : head * WINDOW_VALUES_PER_HEAD;
    const int64_t windowT = spec.windows[offset];
    const int64_t windowH = spec.windows[offset + 1];
    const int64_t windowW = spec.windows[offset + 2];

    const int64_t tStart = OddWindowStart(queryT, windowT, gridT);
    const int64_t hStart = OddWindowStart(queryH, windowH, gridH);
    const int64_t wStart = OddWindowStart(queryW, windowW, gridW);
    return {
        tStart, tStart + (windowT < gridT ? windowT : gridT), hStart, hStart + (windowH < gridH ? windowH : gridH),
        wStart, wStart + (windowW < gridW ? windowW : gridW),
    };
}

struct WeightedTokenGroup {
    double score;
    double value;
    int64_t count;
};

double AttentionScore(const CaseSpec &spec, int64_t query, int64_t keyToken)
{
    double dot = 0.0;
    for (int64_t dim = 0; dim < spec.headDim; ++dim) {
        const double qValue = query < Sequence(spec) ? QuantizedValue(QueryValue(query, dim), spec.dtype) : 0.0;
        const double kValue = QuantizedValue(KeyValue(keyToken, dim), spec.dtype);
        dot += qValue * kValue;
    }
    return dot / std::sqrt(static_cast<double>(spec.headDim));
}

void AppendTokenRange(const CaseSpec &spec, const SamplePoint &sample, int64_t start, int64_t length,
                      std::vector<WeightedTokenGroup> &groups)
{
    const int64_t end = start + length;
    while (start < end) {
        const int64_t nextTile = (start / TILE_ELEMENTS + 1) * TILE_ELEMENTS;
        const int64_t groupEnd = std::min(end, nextTile);
        groups.push_back({
            AttentionScore(spec, sample.query, start),
            static_cast<double>(StorageToFloat(VStorageValue(start, sample.dim, spec.dtype), spec.dtype)),
            groupEnd - start,
        });
        start = groupEnd;
    }
}

float ExpectedValue(const CaseSpec &spec, const SamplePoint &sample)
{
    const int64_t imageSequence = ImageSequence(spec.canvas);
    const int64_t sequence = Sequence(spec);
    std::vector<WeightedTokenGroup> groups;
    if (sample.query >= imageSequence) {
        AppendTokenRange(spec, sample, 0, sequence, groups);
    } else {
        const WindowRange range = GetWindowRange(spec, sample.head, sample.query);
        const int64_t gridH = spec.canvas.height / TILE_HEIGHT;
        const int64_t gridW = spec.canvas.width / TILE_WIDTH;
        for (int64_t tileT = range.tStart; tileT < range.tEnd; ++tileT) {
            for (int64_t tileH = range.hStart; tileH < range.hEnd; ++tileH) {
                const int64_t tileId = (tileT * gridH + tileH) * gridW + range.wStart;
                const int64_t start = tileId * TILE_ELEMENTS;
                const int64_t length = (range.wEnd - range.wStart) * TILE_ELEMENTS;
                AppendTokenRange(spec, sample, start, length, groups);
            }
        }
        if (spec.textLength > 0) {
            AppendTokenRange(spec, sample, imageSequence, spec.textLength, groups);
        }
    }
    double maxScore = -std::numeric_limits<double>::infinity();
    for (const WeightedTokenGroup &group : groups) {
        maxScore = std::max(maxScore, group.score);
    }
    double weightedValue = 0.0;
    double weightSum = 0.0;
    for (const WeightedTokenGroup &group : groups) {
        const double weight = std::exp(group.score - maxScore) * static_cast<double>(group.count);
        weightedValue += weight * group.value;
        weightSum += weight;
    }
    return static_cast<float>(weightedValue / weightSum);
}

std::vector<int64_t> SampleDims(const CaseSpec &spec)
{
    std::vector<int64_t> dims = {0, 7, 63, 64, 127, 128, spec.headDim - 1};
    dims.erase(
        std::remove_if(dims.begin(), dims.end(), [&spec](int64_t dim) { return dim < 0 || dim >= spec.headDim; }),
        dims.end());
    std::sort(dims.begin(), dims.end());
    dims.erase(std::unique(dims.begin(), dims.end()), dims.end());
    return dims;
}

std::vector<SamplePoint> SamplePoints(const CaseSpec &spec)
{
    const int64_t imageSequence = ImageSequence(spec.canvas);
    const std::vector<int64_t> dims = SampleDims(spec);
    std::vector<SamplePoint> samples;
    samples.reserve((static_cast<size_t>(spec.heads) + (spec.textLength > 0 ? 1U : 0U)) * dims.size());
    for (int64_t head = 0; head < spec.heads; ++head) {
        int64_t query = imageSequence / 2;
        if (head == 0) {
            query = 0;
        } else if (head == spec.heads - 1) {
            query = imageSequence - 1;
        }
        for (int64_t dim : dims) {
            samples.push_back({head == spec.heads - 1 ? spec.batch - 1 : 0, head, query, dim});
        }
    }
    if (spec.textLength > 0) {
        for (int64_t dim : dims) {
            samples.push_back({0, spec.heads > 5 ? 5 : spec.heads - 1, imageSequence, dim});
        }
    }
    if (InputSequence(spec) > Sequence(spec)) {
        for (int64_t dim : dims) {
            samples.push_back({spec.batch - 1, 0, InputSequence(spec) - 1, dim});
        }
    }
    return samples;
}

std::vector<CaseSpec> BuildCases()
{
    constexpr Canvas FASTVIDEO_CANVAS = {30, 48, 80};
    constexpr Canvas STEPVIDEO_CANVAS = {36, 48, 48};
    constexpr Canvas WAN_CANVAS = {18, 48, 80};
    return {
        {1, DType::FP16, 1, 8, 64, FASTVIDEO_CANVAS, FASTVIDEO_TEXT_LENGTH, 1, SOURCE_WINDOW_PATTERN, false, false},
        {2, DType::BF16, 1, 8, 64, FASTVIDEO_CANVAS, FASTVIDEO_TEXT_LENGTH, 1, SOURCE_WINDOW_PATTERN, false, false},
        {3, DType::FP16, 1, 8, 64, FASTVIDEO_CANVAS, FASTVIDEO_TEXT_LENGTH, 1, WIDE_WINDOW_PATTERN, false, false},
        {4, DType::BF16, 1, 8, 64, FASTVIDEO_CANVAS, FASTVIDEO_TEXT_LENGTH, 1, WIDE_WINDOW_PATTERN, false, false},
        {5, DType::FP16, 1, 8, 64, STEPVIDEO_CANVAS, 0, 2, SOURCE_WINDOW_PATTERN, false, false},
        {6, DType::BF16, 1, 8, 64, STEPVIDEO_CANVAS, 0, 2, SOURCE_WINDOW_PATTERN, false, false},
        {7, DType::FP16, 1, 8, 64, STEPVIDEO_CANVAS, 0, 2, WIDE_WINDOW_PATTERN, false, false},
        {8, DType::BF16, 1, 8, 64, STEPVIDEO_CANVAS, 0, 2, WIDE_WINDOW_PATTERN, false, false},
        {9, DType::FP16, 1, 8, 64, STEPVIDEO_CANVAS, 0, 2, UNIFORM_333_WINDOW_PATTERN, false, false},
        {10, DType::BF16, 1, 8, 64, STEPVIDEO_CANVAS, 0, 2, UNIFORM_333_WINDOW_PATTERN, false, false},
        {11, DType::FP16, 1, 8, 64, FASTVIDEO_CANVAS, FASTVIDEO_TEXT_LENGTH, 1, UNIFORM_333_WINDOW_PATTERN, false,
         false},
        {12, DType::BF16, 1, 8, 64, FASTVIDEO_CANVAS, FASTVIDEO_TEXT_LENGTH, 1, UNIFORM_333_WINDOW_PATTERN, false,
         false},
        {13, DType::FP16, 1, 8, 64, WAN_CANVAS, 0, 3, UNIFORM_333_WINDOW_PATTERN, false, false},
        {14, DType::BF16, 1, 8, 64, WAN_CANVAS, 0, 3, UNIFORM_333_WINDOW_PATTERN, false, false},
        {15, DType::FP16, 1, 8, 64, WAN_CANVAS, 0, 3, WAN_MIXED_WINDOW_PATTERN, false, false},
        {16, DType::BF16, 1, 8, 64, WAN_CANVAS, 0, 3, WAN_MIXED_WINDOW_PATTERN, false, false},
        {17, DType::FP16, 2, 8, 64, WAN_CANVAS, 0, 3, UNIFORM_333_WINDOW_PATTERN, false, false},
        {18, DType::BF16, 1, 8, 64, STEPVIDEO_CANVAS, 0, 2, UNIFORM_333_WINDOW_PATTERN, true, false},
        {19, DType::FP16, 3, 3, 72, WAN_CANVAS, 0, 3, TASK_N3_PER_AXIS_WINDOW_PATTERN, false, true},
        {23, DType::FP16, 1, 1, 72, WAN_CANVAS, 0, 3, UNIFORM_333_WINDOW_PATTERN, true, true},
        {24, DType::FP16, 3, 1, 72, WAN_CANVAS, 0, 3, WIDE_INT64_WINDOW_PATTERN, true, false},
        {25, DType::BF16, 2, 3, 72, WAN_CANVAS, 0, 3, GENERAL_N3_WIDE_AXIS_WINDOW_PATTERN, false, true},
    };
}

std::vector<CaseSpec> BuildTaskbookCases()
{
    constexpr Canvas FASTVIDEO_CANVAS = {30, 48, 80};
    constexpr Canvas STEPVIDEO_CANVAS = {36, 48, 48};
    constexpr Canvas WAN_CANVAS = {18, 48, 80};
    return {
        {0, DType::BF16, 4, 1, 32, WAN_CANVAS, 0, 3, {1, 5, 7}, true, false, 69120},
        {1, DType::FP16, 4, 1, 32, STEPVIDEO_CANVAS, 0, 2, {1, 3, 5}, true, false, 82944},
        {2, DType::BF16, 1, 5, 32, FASTVIDEO_CANVAS, 41, 1, {1, 9, 3}, true, false, 115264},
        {3, DType::FP16, 5, 3, 16, STEPVIDEO_CANVAS, 0, 2, {7, 5, 5}, true, false, 82944},
        {4, DType::BF16, 11, 2, 16, WAN_CANVAS, 0, 3, {5, 7, 1}, true, false, 69120},
        {5, DType::BF16, 5, 4, 16, STEPVIDEO_CANVAS, 0, 2, {1, 9, 7}, true, false, 82944},
        {6, DType::FP16, 5, 5, 16, WAN_CANVAS, 0, 3, {9, 7, 5}, true, false, 69120},
        {7, DType::FP16, 4, 4, 16, FASTVIDEO_CANVAS, 87, 1, {1, 7, 7}, true, false, 115456},
        {8, DType::BF16, 14, 1, 32, WAN_CANVAS, 0, 3, {3, 1, 5}, true, false, 69120},
        {9, DType::BF16, 2, 14, 16, WAN_CANVAS, 0, 3, {5, 9, 9}, true, false, 69120},
        {10, DType::BF16, 5, 7, 16, STEPVIDEO_CANVAS, 0, 2, {9, 7, 9}, true, false, 82944},
        {11, DType::BF16, 12, 3, 16, STEPVIDEO_CANVAS, 0, 2, {1, 1, 7}, true, false, 82944},
        {12, DType::FP16, 4, 9, 16, STEPVIDEO_CANVAS, 0, 2, {9, 9, 1}, true, false, 82944},
        {13, DType::BF16, 4, 11, 16, STEPVIDEO_CANVAS, 0, 2, {7, 7, 1}, true, false, 82944},
        {14, DType::BF16, 3, 9, 32, WAN_CANVAS, 0, 3, {1, 9, 1}, true, false, 69120},
        {15, DType::FP16, 9, 2, 32, FASTVIDEO_CANVAS, 9, 1, {7, 7, 1}, true, false, 115328},
        {16, DType::BF16, 2, 10, 32, FASTVIDEO_CANVAS, 27, 1, {3, 3, 5}, true, false, 115264},
        {17, DType::BF16, 10, 4, 16, FASTVIDEO_CANVAS, 18, 1, {7, 5, 3}, true, false, 115328},
        {18, DType::FP16, 9, 9, 16, STEPVIDEO_CANVAS, 0, 2, {5, 5, 1}, true, false, 82944},
        {19, DType::FP16, 15, 4, 16, FASTVIDEO_CANVAS, 41, 1, {3, 5, 7}, true, false, 115456},
        {20, DType::FP16, 4, 8, 32, FASTVIDEO_CANVAS, 216, 1, {7, 5, 9}, true, false, 115456},
        {21, DType::BF16, 3, 9, 64, WAN_CANVAS, 0, 3, {7, 3, 3}, true, false, 69120},
        {22, DType::FP16, 12, 9, 16, WAN_CANVAS, 0, 3, {5, 7, 9}, true, false, 69120},
        {23, DType::BF16, 6, 10, 32, WAN_CANVAS, 0, 3, {1, 3, 9}, true, false, 69120},
        {24, DType::BF16, 12, 3, 32, FASTVIDEO_CANVAS, 112, 1, {1, 1, 3}, true, false, 115328},
        {25, DType::FP16, 3, 13, 32, FASTVIDEO_CANVAS, 34, 1, {5, 7, 3}, true, false, 115264},
        {26, DType::FP16, 10, 16, 16, WAN_CANVAS, 0, 3, {5, 9, 7}, true, false, 69120},
        {27, DType::FP16, 15, 11, 16, WAN_CANVAS, 0, 3, {9, 1, 1}, true, false, 69120},
        {28, DType::BF16, 6, 14, 32, WAN_CANVAS, 0, 3, {1, 1, 9}, true, false, 69120},
        {29, DType::BF16, 3, 12, 64, STEPVIDEO_CANVAS, 0, 2, {7, 5, 3}, true, false, 82944},
        {30, DType::FP16, 7, 16, 16, FASTVIDEO_CANVAS, 63, 1, {5, 9, 1}, true, false, 115264},
        {31, DType::BF16, 16, 7, 16, FASTVIDEO_CANVAS, 239, 1, {1, 1, 7}, true, false, 115456},
        {32, DType::FP16, 8, 10, 32, STEPVIDEO_CANVAS, 0, 2, {3, 7, 5}, true, false, 82944},
        {33, DType::FP16, 15, 4, 32, FASTVIDEO_CANVAS, 11, 1, {9, 3, 5}, true, false, 115264},
        {34, DType::BF16, 14, 3, 64, STEPVIDEO_CANVAS, 0, 2, {5, 7, 1}, true, false, 82944},
        {35, DType::FP16, 11, 11, 16, FASTVIDEO_CANVAS, 35, 1, {3, 3, 3}, true, false, 115328},
        {36, DType::FP16, 3, 15, 64, STEPVIDEO_CANVAS, 0, 2, {7, 7, 5}, true, false, 82944},
        {37, DType::FP16, 7, 8, 64, WAN_CANVAS, 0, 3, {5, 7, 1}, true, false, 69120},
        {38, DType::BF16, 7, 8, 64, WAN_CANVAS, 0, 3, {9, 7, 7}, true, false, 69120},
        {39, DType::FP16, 8, 6, 64, STEPVIDEO_CANVAS, 0, 2, {9, 7, 7}, true, false, 82944},
        {40, DType::BF16, 14, 12, 16, FASTVIDEO_CANVAS, 24, 1, {3, 7, 1}, true, false, 115264},
        {41, DType::FP16, 12, 12, 32, WAN_CANVAS, 0, 3, {1, 1, 3}, true, false, 69120},
        {42, DType::BF16, 8, 16, 32, STEPVIDEO_CANVAS, 0, 2, {1, 3, 3}, true, false, 82944},
        {43, DType::FP16, 10, 13, 32, STEPVIDEO_CANVAS, 0, 2, {3, 5, 1}, true, false, 82944},
        {44, DType::BF16, 6, 14, 64, STEPVIDEO_CANVAS, 0, 2, {7, 7, 3}, true, false, 82944},
        {45, DType::FP16, 13, 16, 32, STEPVIDEO_CANVAS, 0, 2, {1, 7, 7}, true, false, 82944},
        {46, DType::BF16, 14, 6, 64, FASTVIDEO_CANVAS, 118, 1, {3, 1, 9}, true, false, 115328},
        {47, DType::FP16, 9, 13, 64, STEPVIDEO_CANVAS, 0, 2, {3, 1, 3}, true, false, 82944},
        {48, DType::FP16, 16, 9, 64, WAN_CANVAS, 0, 3, {7, 7, 3}, true, false, 69120},
        {49, DType::BF16, 15, 12, 64, STEPVIDEO_CANVAS, 0, 2, {9, 5, 1}, true, false, 82944},
    };
}

bool ReadRawSamples(const CaseSpec &spec, const DeviceTensor &output, std::vector<uint16_t> &values)
{
    values.clear();
    const std::vector<SamplePoint> samples = SamplePoints(spec);
    values.reserve(samples.size());
    for (const SamplePoint &sample : samples) {
        const int64_t flatIndex = sample.batch * output.strides[0] + sample.head * output.strides[1] +
                                  sample.query * output.strides[2] + sample.dim * output.strides[3];
        uint16_t raw = 0;
        const auto *source = static_cast<const uint8_t *>(output.data) + static_cast<size_t>(flatIndex) * sizeof(raw);
        const auto ret = aclrtMemcpy(&raw, sizeof(raw), source, sizeof(raw), ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            std::fprintf(stderr, "case=%d sample copy failed: %d\n", spec.id, ret);
            return false;
        }
        values.push_back(raw);
    }
    return true;
}

bool ReadAndCheckSamples(const CaseSpec &spec, const DeviceTensor &output, PrecisionStats &stats,
                         std::vector<uint16_t> &rawValues)
{
    const float tolerance = spec.dtype == DType::FP16 ? FP16_TOLERANCE : BF16_TOLERANCE;
    const std::vector<SamplePoint> samples = SamplePoints(spec);
    if (!ReadRawSamples(spec, output, rawValues)) {
        return false;
    }
    bool passed = true;
    for (size_t sampleIndex = 0; sampleIndex < samples.size(); ++sampleIndex) {
        const SamplePoint &sample = samples[sampleIndex];
        const uint16_t raw = rawValues[sampleIndex];
        const float actual = StorageToFloat(raw, spec.dtype);
        const float expected = ExpectedValue(spec, sample);
        const float absoluteError = std::fabs(actual - expected);
        ++stats.samples;
        stats.sumAbsoluteError += absoluteError;
        stats.sumSquaredError += static_cast<double>(absoluteError) * absoluteError;
        stats.maxAbsoluteError = std::max(stats.maxAbsoluteError, absoluteError);
        if (!std::isfinite(actual) || absoluteError > tolerance) {
            std::fprintf(stderr,
                         "case=%d dtype=%s sample=[b=%ld,h=%ld,q=%ld,d=%ld] expected=%.8f actual=%.8f "
                         "abs_err=%.8f tol=%.8f\n",
                         spec.id, DTypeName(spec.dtype), static_cast<long>(sample.batch),
                         static_cast<long>(sample.head), static_cast<long>(sample.query), static_cast<long>(sample.dim),
                         expected, actual, absoluteError, tolerance);
            passed = false;
        }
    }
    return passed;
}

bool PrepareExecutor(const CaseSpec &spec, const DeviceTensor &q, const DeviceTensor &k, const DeviceTensor &v,
                     const aclIntArray *windowArray, DeviceTensor &output, void *&workspace,
                     uint64_t &workspaceCapacity, uint64_t &workspaceSize, aclOpExecutor *&executor)
{
    workspaceSize = 0;
    executor = nullptr;
    aclnnStatus ret = aclnnSlidingTileAttentionGetWorkspaceSize(q.tensor, k.tensor, v.tensor, windowArray,
                                                                spec.textLength, spec.textLength > 0, spec.seqShape,
                                                                output.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "case=%d GetWorkspaceSize failed: %d\n", spec.id, ret);
        return false;
    }
    if (workspaceSize <= workspaceCapacity) {
        return true;
    }
    if (workspace != nullptr) {
        ret = aclrtFree(workspace);
        if (ret != ACL_SUCCESS) {
            std::fprintf(stderr, "case=%d old workspace free failed: %d\n", spec.id, ret);
            return false;
        }
        workspace = nullptr;
        workspaceCapacity = 0;
    }
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "case=%d workspace allocation failed: %d, bytes=%lu\n", spec.id, ret,
                     static_cast<unsigned long>(workspaceSize));
        return false;
    }
    workspaceCapacity = workspaceSize;
    return true;
}

bool ExecuteOnce(const CaseSpec &spec, const DeviceTensor &q, const DeviceTensor &k, const DeviceTensor &v,
                 const aclIntArray *windowArray, DeviceTensor &output, void *&workspace, uint64_t &workspaceCapacity,
                 aclrtStream stream)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    if (!PrepareExecutor(spec, q, k, v, windowArray, output, workspace, workspaceCapacity, workspaceSize, executor)) {
        return false;
    }
    const aclnnStatus ret = aclnnSlidingTileAttention(workspace, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "case=%d execute failed: %d\n", spec.id, ret);
        return false;
    }
    return true;
}

bool MeasureCase(const CaseSpec &spec, const DeviceTensor &q, const DeviceTensor &k, const DeviceTensor &v,
                 const aclIntArray *windowArray, DeviceTensor &output, void *&workspace, uint64_t &workspaceCapacity,
                 aclrtStream stream, int warmupIterations, int measureIterations, TimingStats &stats)
{
    for (int iteration = 0; iteration < warmupIterations; ++iteration) {
        if (!ExecuteOnce(spec, q, k, v, windowArray, output, workspace, workspaceCapacity, stream)) {
            std::fprintf(stderr, "case=%d warmup failed: iteration=%d\n", spec.id, iteration);
            return false;
        }
        const aclError syncRet = aclrtSynchronizeStream(stream);
        if (syncRet != ACL_SUCCESS) {
            std::fprintf(stderr, "case=%d warmup stream sync failed: %d iteration=%d\n", spec.id, syncRet, iteration);
            return false;
        }
    }

    aclrtEvent start = nullptr;
    aclrtEvent end = nullptr;
    aclError ret = aclrtCreateEvent(&start);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtCreateEvent(start) failed: %d\n", ret);
        return false;
    }
    ret = aclrtCreateEvent(&end);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtCreateEvent(end) failed: %d\n", ret);
        aclrtDestroyEvent(start);
        return false;
    }

    std::vector<float> samples;
    samples.reserve(static_cast<size_t>(measureIterations));
    for (int iteration = 0; iteration < measureIterations; ++iteration) {
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        if (!PrepareExecutor(spec, q, k, v, windowArray, output, workspace, workspaceCapacity, workspaceSize,
                             executor)) {
            break;
        }
        ret = aclrtRecordEvent(start, stream);
        if (ret != ACL_SUCCESS) {
            std::fprintf(stderr, "aclrtRecordEvent(start) failed: %d\n", ret);
            (void)aclnnSlidingTileAttention(workspace, workspaceSize, executor, stream);
            (void)aclrtSynchronizeStream(stream);
            break;
        }
        const aclnnStatus executeRet = aclnnSlidingTileAttention(workspace, workspaceSize, executor, stream);
        if (executeRet != ACL_SUCCESS) {
            std::fprintf(stderr, "timed execute failed: %d iteration=%d\n", executeRet, iteration);
            break;
        }
        ret = aclrtRecordEvent(end, stream);
        if (ret != ACL_SUCCESS) {
            std::fprintf(stderr, "aclrtRecordEvent(end) failed: %d\n", ret);
            break;
        }
        ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            std::fprintf(stderr, "timed stream sync failed: %d\n", ret);
            break;
        }
        float elapsedMs = 0.0F;
        ret = aclrtEventElapsedTime(&elapsedMs, start, end);
        if (ret != ACL_SUCCESS) {
            std::fprintf(stderr, "aclrtEventElapsedTime failed: %d\n", ret);
            break;
        }
        samples.push_back(elapsedMs * 1000.0F);
    }
    aclrtDestroyEvent(start);
    aclrtDestroyEvent(end);
    if (static_cast<int>(samples.size()) != measureIterations) {
        return false;
    }
    std::sort(samples.begin(), samples.end());
    stats.minUs = samples.front();
    stats.medianUs = samples[samples.size() / 2U];
    stats.maxUs = samples.back();
    return true;
}

bool RunCase(const CaseSpec &spec, aclrtStream stream, int warmupIterations, int measureIterations,
             std::ofstream &report)
{
    const int64_t sequence = InputSequence(spec);
    const std::vector<int64_t> shape = {spec.batch, spec.heads, sequence, spec.headDim};
    const int64_t storageHeadDim = spec.useNonContiguousStorage ? spec.headDim + 8 : spec.headDim;
    DeviceTensor q;
    DeviceTensor k;
    DeviceTensor v;
    DeviceTensor output;
    void *workspace = nullptr;
    aclIntArray *windowArray = nullptr;
    std::array<int64_t, WINDOW_VALUES_PER_HEAD> broadcastWindow = {
        spec.windows[0],
        spec.windows[1],
        spec.windows[2],
    };
    const int64_t *windowData = spec.useBroadcastWindow ? broadcastWindow.data() : spec.windows.data();
    const uint64_t windowCount =
        spec.useBroadcastWindow ? broadcastWindow.size() : static_cast<uint64_t>(spec.heads * WINDOW_VALUES_PER_HEAD);
    uint64_t workspaceCapacity = 0;
    bool passed = false;
    bool hugeMemInitialized = false;
    TimingStats timing;
    PrecisionStats precision;
    std::vector<uint16_t> firstOutputSamples;
    std::vector<uint16_t> repeatedOutputSamples;

    if (!CreateTensor(shape, AclDType(spec.dtype), q, storageHeadDim) ||
        !CreateTensor(shape, AclDType(spec.dtype), k, storageHeadDim) ||
        !CreateTensor(shape, AclDType(spec.dtype), v, storageHeadDim) ||
        !CreateTensor(shape, AclDType(spec.dtype), output, storageHeadDim)) {
        goto cleanup;
    }
    if (!InitializeInputs(spec, q, k, v, output)) {
        goto cleanup;
    }

    windowArray = aclCreateIntArray(windowData, windowCount);
    if (windowArray == nullptr) {
        std::fprintf(stderr, "case=%d aclCreateIntArray failed\n", spec.id);
        goto cleanup;
    }
    if (!BeginHugeMem(hugeMemInitialized)) {
        goto cleanup;
    }

    if (!MeasureCase(spec, q, k, v, windowArray, output, workspace, workspaceCapacity, stream, warmupIterations,
                     measureIterations, timing)) {
        std::fprintf(stderr, "case=%d timing failed\n", spec.id);
        goto cleanup;
    }
    passed = ReadAndCheckSamples(spec, output, precision, firstOutputSamples);
    if (passed) {
        if (!ExecuteOnce(spec, q, k, v, windowArray, output, workspace, workspaceCapacity, stream)) {
            std::fprintf(stderr, "case=%d deterministic repeat execute failed\n", spec.id);
            passed = false;
        } else if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
            std::fprintf(stderr, "case=%d deterministic repeat stream sync failed\n", spec.id);
            passed = false;
        } else if (!ReadRawSamples(spec, output, repeatedOutputSamples)) {
            passed = false;
        } else if (firstOutputSamples != repeatedOutputSamples) {
            std::fprintf(stderr, "case=%d deterministic sample comparison failed\n", spec.id);
            passed = false;
        }
    }
    if (passed) {
        std::printf("CASE_PASS id=%d dtype=%s shape=[%ld,%ld,%ld,%ld] samples=%lu max_abs=%.8g mae=%.8g rmse=%.8g "
                    "tol=%.8g latency_us_median=%.3f latency_us_min=%.3f latency_us_max=%.3f warmup=%d repeat=%d "
                    "text=%ld window_values=%lu layout=%s deterministic=bitwise\n",
                    spec.id, DTypeName(spec.dtype), static_cast<long>(spec.batch), static_cast<long>(spec.heads),
                    static_cast<long>(sequence), static_cast<long>(spec.headDim),
                    static_cast<unsigned long>(precision.samples), precision.maxAbsoluteError,
                    precision.sumAbsoluteError / static_cast<double>(precision.samples),
                    std::sqrt(precision.sumSquaredError / static_cast<double>(precision.samples)),
                    spec.dtype == DType::FP16 ? FP16_TOLERANCE : BF16_TOLERANCE, timing.medianUs, timing.minUs,
                    timing.maxUs, warmupIterations, measureIterations, static_cast<long>(spec.textLength),
                    static_cast<unsigned long>(windowCount),
                    spec.useNonContiguousStorage ? "non_contiguous_d_stride" : "contiguous");
        report << spec.id << ',' << DTypeName(spec.dtype) << ',' << spec.batch << ',' << spec.heads << ',' << sequence
               << ',' << spec.headDim << ',' << precision.samples << ',' << std::setprecision(10)
               << precision.maxAbsoluteError << ','
               << precision.sumAbsoluteError / static_cast<double>(precision.samples) << ','
               << std::sqrt(precision.sumSquaredError / static_cast<double>(precision.samples)) << ','
               << (spec.dtype == DType::FP16 ? FP16_TOLERANCE : BF16_TOLERANCE) << ',' << timing.medianUs << ','
               << timing.minUs << ',' << timing.maxUs << ',' << warmupIterations << ',' << measureIterations << ','
               << (spec.useNonContiguousStorage ? "non_contiguous_d_stride" : "contiguous") << ",bitwise\n";
        report.flush();
        std::fflush(stdout);
    }

cleanup:
    EndHugeMem(hugeMemInitialized);
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    if (windowArray != nullptr) {
        aclDestroyIntArray(windowArray);
    }
    DestroyTensor(q);
    DestroyTensor(k);
    DestroyTensor(v);
    DestroyTensor(output);
    return passed;
}

bool RunEmptyTensorCheck(aclrtStream stream)
{
    const std::vector<int64_t> shape = {0, 1, ImageSequence({18, 48, 80}), 8};
    DeviceTensor q;
    DeviceTensor k;
    DeviceTensor v;
    DeviceTensor output;
    std::array<int64_t, WINDOW_VALUES_PER_HEAD> broadcastWindow = {3, 3, 3};
    aclIntArray *windowArray = nullptr;
    uint64_t workspaceSize = std::numeric_limits<uint64_t>::max();
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ACL_SUCCESS;
    bool passed = false;
    bool hugeMemInitialized = false;

    if (!CreateTensor(shape, ACL_FLOAT16, q) || !CreateTensor(shape, ACL_FLOAT16, k) ||
        !CreateTensor(shape, ACL_FLOAT16, v) || !CreateTensor(shape, ACL_FLOAT16, output)) {
        goto cleanup;
    }
    windowArray = aclCreateIntArray(broadcastWindow.data(), broadcastWindow.size());
    if (windowArray == nullptr) {
        std::fprintf(stderr, "empty tensor aclCreateIntArray failed\n");
        goto cleanup;
    }
    if (!BeginHugeMem(hugeMemInitialized)) {
        goto cleanup;
    }
    ret = aclnnSlidingTileAttentionGetWorkspaceSize(q.tensor, k.tensor, v.tensor, windowArray, 0, false, 3,
                                                    output.tensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS || workspaceSize != 0 || executor == nullptr) {
        std::fprintf(stderr, "empty tensor GetWorkspaceSize failed: ret=%d workspace=%lu executor=%p\n", ret,
                     static_cast<unsigned long>(workspaceSize), static_cast<void *>(executor));
        goto cleanup;
    }
    ret = aclnnSlidingTileAttention(nullptr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "empty tensor execute failed: %d\n", ret);
        goto cleanup;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "empty tensor stream sync failed: %d\n", ret);
        goto cleanup;
    }
    std::printf("API_PASS empty_tensor workspace=0\n");
    passed = true;

cleanup:
    EndHugeMem(hugeMemInitialized);
    if (windowArray != nullptr) {
        aclDestroyIntArray(windowArray);
    }
    DestroyTensor(q);
    DestroyTensor(k);
    DestroyTensor(v);
    DestroyTensor(output);
    return passed;
}

int ParseDeviceId()
{
    const char *value = std::getenv("SLIDING_TILE_ATTENTION_DEVICE_ID");
    if (value == nullptr || *value == '\0') {
        return 0;
    }
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 0 || parsed > std::numeric_limits<int32_t>::max()) {
        std::fprintf(stderr, "invalid SLIDING_TILE_ATTENTION_DEVICE_ID=%s\n", value);
        return -1;
    }
    return static_cast<int>(parsed);
}

int ParseSelectedCaseId()
{
    const char *value = std::getenv("SLIDING_TILE_ATTENTION_CASE_ID");
    if (value == nullptr || *value == '\0') {
        return 0;
    }
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int32_t>::max()) {
        std::fprintf(stderr, "invalid SLIDING_TILE_ATTENTION_CASE_ID=%s\n", value);
        return -1;
    }
    return static_cast<int>(parsed);
}

int ParseSelectedCaseRepeat()
{
    const char *value = std::getenv("SLIDING_TILE_ATTENTION_CASE_REPEAT");
    if (value == nullptr || *value == '\0') {
        return 1;
    }
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int32_t>::max()) {
        std::fprintf(stderr, "invalid SLIDING_TILE_ATTENTION_CASE_REPEAT=%s\n", value);
        return -1;
    }
    return static_cast<int>(parsed);
}

int ParsePositiveEnvironment(const char *name, int defaultValue)
{
    const char *value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return defaultValue;
    }
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int32_t>::max()) {
        std::fprintf(stderr, "invalid %s=%s\n", name, value);
        return -1;
    }
    return static_cast<int>(parsed);
}

int ParseNonNegativeEnvironment(const char *name, int defaultValue)
{
    const char *value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return defaultValue;
    }
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 0 || parsed > std::numeric_limits<int32_t>::max()) {
        std::fprintf(stderr, "invalid %s=%s\n", name, value);
        return -1;
    }
    return static_cast<int>(parsed);
}

std::vector<int> ParseCaseSequence()
{
    const char *value = std::getenv("SLIDING_TILE_ATTENTION_CASE_SEQUENCE");
    if (value == nullptr || *value == '\0') {
        return {};
    }
    std::vector<int> result;
    const std::string text(value);
    size_t start = 0;
    while (start < text.size()) {
        const size_t end = text.find(',', start);
        const std::string item = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
        char *parseEnd = nullptr;
        const long parsed = std::strtol(item.c_str(), &parseEnd, 10);
        if (item.empty() || parseEnd == item.c_str() || *parseEnd != '\0' || parsed <= 0 ||
            parsed > std::numeric_limits<int32_t>::max()) {
            std::fprintf(stderr, "invalid SLIDING_TILE_ATTENTION_CASE_SEQUENCE=%s\n", value);
            return {-1};
        }
        result.push_back(static_cast<int>(parsed));
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return result;
}

} // namespace

int main()
{
    const int deviceId = ParseDeviceId();
    const int selectedCaseId = ParseSelectedCaseId();
    const int selectedCaseRepeat = ParseSelectedCaseRepeat();
    const int warmupIterations =
        ParseNonNegativeEnvironment("SLIDING_TILE_ATTENTION_WARMUP", DEFAULT_WARMUP_ITERATIONS);
    const int measureIterations =
        ParsePositiveEnvironment("SLIDING_TILE_ATTENTION_MEASURE_REPEAT", DEFAULT_MEASURE_ITERATIONS);
    const std::vector<int> caseSequence = ParseCaseSequence();
    if (deviceId < 0 || selectedCaseId < 0 || selectedCaseRepeat < 0 || warmupIterations < 0 || measureIterations < 0 ||
        (!caseSequence.empty() && caseSequence.front() < 0)) {
        return EXIT_FAILURE;
    }
    const char *reportPath = std::getenv("SLIDING_TILE_ATTENTION_REPORT_PATH");
    const std::string resolvedReportPath =
        reportPath == nullptr || *reportPath == '\0' ? "sliding_tile_attention_self_validation.csv" : reportPath;
    std::ofstream report(resolvedReportPath, std::ios::out | std::ios::trunc);
    if (!report.is_open()) {
        std::fprintf(stderr, "failed to open report path: %s\n", resolvedReportPath.c_str());
        return EXIT_FAILURE;
    }
    report << "case_id,dtype,batch,heads,sequence,head_dim,samples,max_abs_error,mean_abs_error,rmse,"
              "abs_tolerance,latency_us_median,latency_us_min,latency_us_max,warmup,measure_repeat,layout,"
              "deterministic\n";
    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "aclInit failed: %d\n", ret);
        return EXIT_FAILURE;
    }
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtSetDevice failed: %d, device=%d\n", ret, deviceId);
        aclFinalize();
        return EXIT_FAILURE;
    }
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtCreateStream failed: %d\n", ret);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return EXIT_FAILURE;
    }

    int passedCases = 0;
    const char *suite = std::getenv("SLIDING_TILE_ATTENTION_SUITE");
    const bool taskbookSuite = suite != nullptr && std::strcmp(suite, "taskbook") == 0;
    const std::vector<CaseSpec> cases = taskbookSuite ? BuildTaskbookCases() : BuildCases();
    std::vector<const CaseSpec *> casesToRun;
    if (!caseSequence.empty()) {
        for (int caseId : caseSequence) {
            const CaseSpec *matched = nullptr;
            for (const CaseSpec &spec : cases) {
                if (spec.id == caseId) {
                    matched = &spec;
                    break;
                }
            }
            if (matched == nullptr) {
                std::fprintf(stderr, "case id not found: %d\n", caseId);
                aclrtDestroyStream(stream);
                aclrtResetDevice(deviceId);
                aclFinalize();
                return EXIT_FAILURE;
            }
            casesToRun.push_back(matched);
        }
    } else {
        for (const CaseSpec &spec : cases) {
            if (selectedCaseId == 0 || spec.id == selectedCaseId) {
                casesToRun.push_back(&spec);
            }
        }
    }
    for (const CaseSpec *specPointer : casesToRun) {
        const CaseSpec &spec = *specPointer;
        const int repeat = selectedCaseId == 0 || !caseSequence.empty() ? 1 : selectedCaseRepeat;
        for (int iteration = 0; iteration < repeat; ++iteration) {
            if (!RunCase(spec, stream, warmupIterations, measureIterations, report)) {
                std::fprintf(stderr, "CASE_FAIL id=%d dtype=%s repeat=%d/%d\n", spec.id, DTypeName(spec.dtype),
                             iteration + 1, repeat);
                aclrtDestroyStream(stream);
                aclrtResetDevice(deviceId);
                aclFinalize();
                return EXIT_FAILURE;
            }
            ++passedCases;
        }
    }
    if (casesToRun.empty()) {
        std::fprintf(stderr, "case id not found: %d\n", selectedCaseId);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return EXIT_FAILURE;
    }
    if (!RunEmptyTensorCheck(stream)) {
        std::fprintf(stderr, "API_FAIL empty_tensor\n");
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return EXIT_FAILURE;
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    std::printf("ALL_LOCAL_SLIDING_TILE_ATTENTION_CASES_PASSED suite=%s cases=%d report=%s warmup=%d repeat=%d\n",
                taskbookSuite ? "taskbook" : "reviewer22", passedCases, resolvedReportPath.c_str(), warmupIterations,
                measureIterations);
    return EXIT_SUCCESS;
}
