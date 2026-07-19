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

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr int64_t HEAD_COUNT = 8;
constexpr int64_t HEAD_DIM = 64;
constexpr int64_t TILE_TIME = 6;
constexpr int64_t TILE_HEIGHT = 8;
constexpr int64_t TILE_WIDTH = 8;
constexpr int64_t TILE_ELEMENTS = TILE_TIME * TILE_HEIGHT * TILE_WIDTH;
constexpr int64_t FASTVIDEO_TEXT_LENGTH = 128;
constexpr int64_t WINDOW_VALUES_PER_HEAD = 3;
constexpr int64_t FULL_WINDOW_VALUE_COUNT = HEAD_COUNT * WINDOW_VALUES_PER_HEAD;
constexpr float TOKEN_VALUE_BASE = 1.0F / 8.0F;
constexpr float TILE_VALUE_SCALE = 1.0F / 64.0F;
constexpr float DIM_VALUE_SCALE = 1.0F / 128.0F;
constexpr float FP16_TOLERANCE = 3.0e-3F;
constexpr float BF16_TOLERANCE = 2.0e-2F;

using WindowArray = std::array<int64_t, FULL_WINDOW_VALUE_COUNT>;

constexpr WindowArray SOURCE_WINDOW_PATTERN = {
    1, 1, 1, 3, 3, 3, 3, 5, 5, 5, 5, 5, 1, 3, 3, 3, 3, 5, 5, 5, 7, 7, 7, 7,
};
constexpr WindowArray WIDE_WINDOW_PATTERN = {
    7, 7, 7, 7, 5, 7, 5, 7, 7, 5, 5, 7, 7, 3, 5, 3, 7, 5, 5, 7, 3, 7, 7, 5,
};
constexpr WindowArray UNIFORM_333_WINDOW_PATTERN = {
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
};
constexpr WindowArray WAN_MIXED_WINDOW_PATTERN = {
    3, 3, 5, 3, 1, 9, 1, 3, 5, 5, 5, 5, 3, 5, 7, 5, 3, 7, 1, 1, 1, 7, 7, 7,
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
    Canvas canvas;
    int64_t textLength;
    int64_t seqShape;
    WindowArray windows;
    bool useBroadcastWindow;
};

struct DeviceTensor {
    void *data = nullptr;
    aclTensor *tensor = nullptr;
    size_t bytes = 0;
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

std::vector<int64_t> ContiguousStrides(const std::vector<int64_t> &shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t index = static_cast<int64_t>(shape.size()) - 2; index >= 0; --index) {
        strides[index] = strides[index + 1] * shape[index + 1];
    }
    return strides;
}

bool CreateTensor(const std::vector<int64_t> &shape, aclDataType dtype, DeviceTensor &output)
{
    int64_t elementCount = 1;
    for (int64_t dim : shape) {
        elementCount *= dim;
    }
    output.bytes = static_cast<size_t>(elementCount) * sizeof(uint16_t);
    auto ret = aclrtMalloc(&output.data, output.bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtMalloc failed: %d, bytes=%zu\n", ret, output.bytes);
        return false;
    }
    const auto strides = ContiguousStrides(shape);
    output.tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                                    shape.size(), output.data);
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
}

float TokenValue(int64_t token)
{
    const int64_t tile = token / TILE_ELEMENTS;
    return TOKEN_VALUE_BASE + static_cast<float>(tile & 15LL) * TILE_VALUE_SCALE;
}

float DimValue(int64_t dim)
{
    return static_cast<float>(dim & 3LL) * DIM_VALUE_SCALE;
}

uint16_t VStorageValue(int64_t token, int64_t dim, DType dtype)
{
    return FloatToStorage(TokenValue(token) + DimValue(dim), dtype);
}

bool InitializeInputs(const CaseSpec &spec, const DeviceTensor &q, const DeviceTensor &k, const DeviceTensor &v,
                      const DeviceTensor &output)
{
    auto ret = aclrtMemset(q.data, q.bytes, 0, q.bytes);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtMemset(q) failed: %d\n", ret);
        return false;
    }
    ret = aclrtMemset(k.data, k.bytes, 0, k.bytes);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtMemset(k) failed: %d\n", ret);
        return false;
    }
    ret = aclrtMemset(output.data, output.bytes, 0, output.bytes);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtMemset(output) failed: %d\n", ret);
        return false;
    }

    const int64_t sequence = Sequence(spec);
    const size_t rowBytes = static_cast<size_t>(HEAD_DIM) * sizeof(uint16_t);
    std::vector<uint16_t> hostV(static_cast<size_t>(sequence) * HEAD_DIM);
    for (int64_t token = 0; token < sequence; ++token) {
        for (int64_t dim = 0; dim < HEAD_DIM; ++dim) {
            hostV[static_cast<size_t>(token) * HEAD_DIM + dim] = VStorageValue(token, dim, spec.dtype);
        }
    }
    const size_t headBytes = hostV.size() * sizeof(uint16_t);
    for (int64_t batch = 0; batch < spec.batch; ++batch) {
        for (int64_t head = 0; head < HEAD_COUNT; ++head) {
            const int64_t headIndex = batch * HEAD_COUNT + head;
            auto *headDestination = static_cast<uint8_t *>(v.data) + static_cast<size_t>(headIndex) * headBytes;
            ret = aclrtMemcpy(headDestination, headBytes, hostV.data(), headBytes, ACL_MEMCPY_HOST_TO_DEVICE);
            if (ret != ACL_SUCCESS) {
                std::fprintf(stderr, "aclrtMemcpy(v) failed: %d, batch=%ld, head=%ld, rowBytes=%zu\n", ret,
                             static_cast<long>(batch), static_cast<long>(head), rowBytes);
                return false;
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
    const int64_t offset = head * WINDOW_VALUES_PER_HEAD;
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

void AccumulateTokenPattern(int64_t start, int64_t length, double &tokenSum, int64_t &tokenCount)
{
    for (int64_t token = start; token < start + length; ++token) {
        tokenSum += TokenValue(token);
    }
    tokenCount += length;
}

float ExpectedValue(const CaseSpec &spec, const SamplePoint &sample)
{
    const int64_t imageSequence = ImageSequence(spec.canvas);
    const int64_t sequence = Sequence(spec);
    double tokenSum = 0.0;
    int64_t tokenCount = 0;
    if (sample.query >= imageSequence) {
        AccumulateTokenPattern(0, sequence, tokenSum, tokenCount);
    } else {
        const WindowRange range = GetWindowRange(spec, sample.head, sample.query);
        const int64_t gridH = spec.canvas.height / TILE_HEIGHT;
        const int64_t gridW = spec.canvas.width / TILE_WIDTH;
        for (int64_t tileT = range.tStart; tileT < range.tEnd; ++tileT) {
            for (int64_t tileH = range.hStart; tileH < range.hEnd; ++tileH) {
                const int64_t tileId = (tileT * gridH + tileH) * gridW + range.wStart;
                const int64_t start = tileId * TILE_ELEMENTS;
                const int64_t length = (range.wEnd - range.wStart) * TILE_ELEMENTS;
                AccumulateTokenPattern(start, length, tokenSum, tokenCount);
            }
        }
        if (spec.textLength > 0) {
            AccumulateTokenPattern(imageSequence, spec.textLength, tokenSum, tokenCount);
        }
    }
    return static_cast<float>(tokenSum / static_cast<double>(tokenCount)) + DimValue(sample.dim);
}

std::vector<SamplePoint> SamplePoints(const CaseSpec &spec)
{
    const int64_t imageSequence = ImageSequence(spec.canvas);
    std::vector<SamplePoint> samples;
    samples.reserve(static_cast<size_t>(HEAD_COUNT) + (spec.textLength > 0 ? 1U : 0U));
    for (int64_t head = 0; head < HEAD_COUNT; ++head) {
        int64_t query = imageSequence / 2;
        if (head == 0) {
            query = 0;
        } else if (head == HEAD_COUNT - 1) {
            query = imageSequence - 1;
        }
        samples.push_back({head == HEAD_COUNT - 1 ? spec.batch - 1 : 0, head, query, (head * 9) % HEAD_DIM});
    }
    if (spec.textLength > 0) {
        samples.push_back({0, 5, imageSequence, 31});
    }
    return samples;
}

std::vector<CaseSpec> BuildCases()
{
    constexpr Canvas FASTVIDEO_CANVAS = {30, 48, 80};
    constexpr Canvas STEPVIDEO_CANVAS = {36, 48, 48};
    constexpr Canvas WAN_CANVAS = {18, 48, 80};
    return {
        {1, DType::FP16, 1, FASTVIDEO_CANVAS, FASTVIDEO_TEXT_LENGTH, 1, SOURCE_WINDOW_PATTERN, false},
        {2, DType::BF16, 1, FASTVIDEO_CANVAS, FASTVIDEO_TEXT_LENGTH, 1, SOURCE_WINDOW_PATTERN, false},
        {3, DType::FP16, 1, FASTVIDEO_CANVAS, FASTVIDEO_TEXT_LENGTH, 1, WIDE_WINDOW_PATTERN, false},
        {4, DType::BF16, 1, FASTVIDEO_CANVAS, FASTVIDEO_TEXT_LENGTH, 1, WIDE_WINDOW_PATTERN, false},
        {5, DType::FP16, 1, STEPVIDEO_CANVAS, 0, 2, SOURCE_WINDOW_PATTERN, false},
        {6, DType::BF16, 1, STEPVIDEO_CANVAS, 0, 2, SOURCE_WINDOW_PATTERN, false},
        {7, DType::FP16, 1, STEPVIDEO_CANVAS, 0, 2, WIDE_WINDOW_PATTERN, false},
        {8, DType::BF16, 1, STEPVIDEO_CANVAS, 0, 2, WIDE_WINDOW_PATTERN, false},
        {9, DType::FP16, 1, STEPVIDEO_CANVAS, 0, 2, UNIFORM_333_WINDOW_PATTERN, false},
        {10, DType::BF16, 1, STEPVIDEO_CANVAS, 0, 2, UNIFORM_333_WINDOW_PATTERN, false},
        {11, DType::FP16, 1, FASTVIDEO_CANVAS, FASTVIDEO_TEXT_LENGTH, 1, UNIFORM_333_WINDOW_PATTERN, false},
        {12, DType::BF16, 1, FASTVIDEO_CANVAS, FASTVIDEO_TEXT_LENGTH, 1, UNIFORM_333_WINDOW_PATTERN, false},
        {13, DType::FP16, 1, WAN_CANVAS, 0, 3, UNIFORM_333_WINDOW_PATTERN, false},
        {14, DType::BF16, 1, WAN_CANVAS, 0, 3, UNIFORM_333_WINDOW_PATTERN, false},
        {15, DType::FP16, 1, WAN_CANVAS, 0, 3, WAN_MIXED_WINDOW_PATTERN, false},
        {16, DType::BF16, 1, WAN_CANVAS, 0, 3, WAN_MIXED_WINDOW_PATTERN, false},
        {17, DType::FP16, 2, WAN_CANVAS, 0, 3, UNIFORM_333_WINDOW_PATTERN, false},
        {18, DType::BF16, 1, STEPVIDEO_CANVAS, 0, 2, UNIFORM_333_WINDOW_PATTERN, true},
    };
}

bool ReadAndCheckSamples(const CaseSpec &spec, const DeviceTensor &output)
{
    const int64_t sequence = Sequence(spec);
    const float tolerance = spec.dtype == DType::FP16 ? FP16_TOLERANCE : BF16_TOLERANCE;
    bool passed = true;
    for (const SamplePoint &sample : SamplePoints(spec)) {
        const int64_t flatIndex =
            ((sample.batch * HEAD_COUNT + sample.head) * sequence + sample.query) * HEAD_DIM + sample.dim;
        uint16_t raw = 0;
        const auto *source = static_cast<const uint8_t *>(output.data) + static_cast<size_t>(flatIndex) * sizeof(raw);
        const auto ret = aclrtMemcpy(&raw, sizeof(raw), source, sizeof(raw), ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            std::fprintf(stderr, "case=%d sample copy failed: %d\n", spec.id, ret);
            return false;
        }
        const float actual = StorageToFloat(raw, spec.dtype);
        const float expected = ExpectedValue(spec, sample);
        const float absoluteError = std::fabs(actual - expected);
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

bool RunCase(const CaseSpec &spec, aclrtStream stream)
{
    const int64_t sequence = Sequence(spec);
    const std::vector<int64_t> shape = {spec.batch, HEAD_COUNT, sequence, HEAD_DIM};
    DeviceTensor q;
    DeviceTensor k;
    DeviceTensor v;
    DeviceTensor output;
    void *workspace = nullptr;
    aclIntArray *windowArray = nullptr;
    std::array<int64_t, WINDOW_VALUES_PER_HEAD> broadcastWindow = {3, 3, 3};
    const int64_t *windowData = spec.useBroadcastWindow ? broadcastWindow.data() : spec.windows.data();
    const uint64_t windowCount = spec.useBroadcastWindow ? broadcastWindow.size() : spec.windows.size();
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = ACL_SUCCESS;
    bool passed = false;

    if (!CreateTensor(shape, AclDType(spec.dtype), q) || !CreateTensor(shape, AclDType(spec.dtype), k) ||
        !CreateTensor(shape, AclDType(spec.dtype), v) || !CreateTensor(shape, AclDType(spec.dtype), output)) {
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

    ret = aclnnSlidingTileAttentionGetWorkspaceSize(q.tensor, k.tensor, v.tensor, windowArray, spec.textLength,
                                                    spec.textLength > 0, spec.seqShape, output.tensor, &workspaceSize,
                                                    &executor);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "case=%d GetWorkspaceSize failed: %d\n", spec.id, ret);
        goto cleanup;
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            std::fprintf(stderr, "case=%d workspace allocation failed: %d, bytes=%lu\n", spec.id, ret,
                         static_cast<unsigned long>(workspaceSize));
            goto cleanup;
        }
    }
    ret = aclnnSlidingTileAttention(workspace, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "case=%d execute failed: %d\n", spec.id, ret);
        goto cleanup;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "case=%d stream sync failed: %d\n", spec.id, ret);
        goto cleanup;
    }
    passed = ReadAndCheckSamples(spec, output);
    if (passed) {
        std::printf(
            "CASE_PASS id=%d dtype=%s shape=[%ld,8,%ld,64] text=%ld window_values=%lu\n", spec.id,
            DTypeName(spec.dtype), static_cast<long>(spec.batch), static_cast<long>(sequence),
            static_cast<long>(spec.textLength),
            static_cast<unsigned long>(spec.useBroadcastWindow ? WINDOW_VALUES_PER_HEAD : FULL_WINDOW_VALUE_COUNT));
        std::fflush(stdout);
    }

cleanup:
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

} // namespace

int main()
{
    const int deviceId = ParseDeviceId();
    if (deviceId < 0) {
        return EXIT_FAILURE;
    }
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
    for (const CaseSpec &spec : BuildCases()) {
        if (!RunCase(spec, stream)) {
            std::fprintf(stderr, "CASE_FAIL id=%d dtype=%s\n", spec.id, DTypeName(spec.dtype));
            aclrtDestroyStream(stream);
            aclrtResetDevice(deviceId);
            aclFinalize();
            return EXIT_FAILURE;
        }
        ++passedCases;
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    std::printf("ALL_LOCAL_SLIDING_TILE_ATTENTION_CASES_PASSED cases=%d\n", passedCases);
    return EXIT_SUCCESS;
}
