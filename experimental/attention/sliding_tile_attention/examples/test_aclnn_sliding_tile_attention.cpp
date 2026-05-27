/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sliding_tile_attention.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {
constexpr int64_t BATCH = 1;
constexpr int64_t HEAD_NUM = 2;
constexpr int64_t SEQ_T = 6;
constexpr int64_t SEQ_H = 16;
constexpr int64_t SEQ_W = 8;
constexpr int64_t TILE_T = 6;
constexpr int64_t TILE_H = 8;
constexpr int64_t TILE_W = 8;
constexpr int64_t TEXT_LENGTH = 2;
constexpr int64_t HEAD_DIM = 8;
constexpr float ATOL = 5e-2F;
constexpr float RTOL = 5e-2F;

enum class TensorDtype {
    FP16,
    BF16,
};

struct TestCase {
    std::string name;
    TensorDtype dtype;
    int64_t batch;
    int64_t headNum;
    int64_t seqT;
    int64_t seqH;
    int64_t seqW;
    int64_t textLength;
    bool hasText;
    int64_t headDim;
    std::vector<int64_t> windowSize;
    bool checkGolden;
    std::string targetPerformance;
};

template <TensorDtype DTYPE>
struct TensorDtypeTraits;

template <>
struct TensorDtypeTraits<TensorDtype::FP16> {
    using Storage = aclFloat16;
    static constexpr aclDataType ACL_DTYPE = aclDataType::ACL_FLOAT16;
    static constexpr const char *NAME = "fp16";

    static Storage FromFloat(float value)
    {
        return aclFloatToFloat16(value);
    }

    static float ToFloat(Storage value)
    {
        return aclFloat16ToFloat(value);
    }
};

template <>
struct TensorDtypeTraits<TensorDtype::BF16> {
    using Storage = uint16_t;
    static constexpr aclDataType ACL_DTYPE = aclDataType::ACL_BF16;
    static constexpr const char *NAME = "bf16";

    static Storage FromFloat(float value)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        const uint32_t roundingBias = 0x7FFFU + ((bits >> 16U) & 1U);
        return static_cast<Storage>((bits + roundingBias) >> 16U);
    }

    static float ToFloat(Storage value)
    {
        const uint32_t bits = static_cast<uint32_t>(value) << 16U;
        float result = 0.0F;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }
};

const char *DtypeName(TensorDtype dtype)
{
    return dtype == TensorDtype::BF16 ? TensorDtypeTraits<TensorDtype::BF16>::NAME
                                      : TensorDtypeTraits<TensorDtype::FP16>::NAME;
}

int64_t ImageSeqLen(const TestCase &testCase)
{
    return testCase.seqT * testCase.seqH * testCase.seqW;
}

int64_t RealTextLength(const TestCase &testCase)
{
    return testCase.hasText ? testCase.textLength : 0;
}

int64_t SeqLen(const TestCase &testCase)
{
    return ImageSeqLen(testCase) + RealTextLength(testCase);
}

std::string SeqShapeString(const TestCase &testCase)
{
    return std::to_string(testCase.seqT) + "x" + std::to_string(testCase.seqH) + "x" +
           std::to_string(testCase.seqW);
}

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    const auto size = GetShapeSize(shape) * static_cast<int64_t>(sizeof(T));
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

int64_t TensorOffset(const TestCase &testCase, int64_t batch, int64_t head, int64_t seq, int64_t dim)
{
    return (((batch * testCase.headNum + head) * SeqLen(testCase) + seq) * testCase.headDim + dim);
}

int64_t ClampStart(int64_t center, int64_t window, int64_t limit)
{
    if (limit <= window) {
        return 0;
    }
    int64_t start = center - (window / 2);
    start = std::max<int64_t>(start, 0);
    return std::min<int64_t>(start, limit - window);
}

int64_t HeadWindow(const std::vector<int64_t> &windowSize, int64_t head, int64_t axis)
{
    if (windowSize.size() == 3U) {
        return windowSize[axis];
    }
    return windowSize[head * 3 + axis];
}

template <typename Func>
void VisitKvSeq(const TestCase &testCase, int64_t head, int64_t querySeq, const std::vector<int64_t> &windowSize,
                Func func)
{
    const int64_t imageSeqLen = ImageSeqLen(testCase);
    const int64_t seqLen = SeqLen(testCase);
    const bool isTextQuery = testCase.hasText && querySeq >= imageSeqLen && querySeq < seqLen;
    if (isTextQuery) {
        for (int64_t kvSeq = 0; kvSeq < seqLen; ++kvSeq) {
            func(kvSeq);
        }
        return;
    }

    const int64_t tileTokenNum = TILE_T * TILE_H * TILE_W;
    const int64_t tileNumT = testCase.seqT / TILE_T;
    const int64_t tileNumH = testCase.seqH / TILE_H;
    const int64_t tileNumW = testCase.seqW / TILE_W;
    const int64_t tileId = querySeq / tileTokenNum;
    const int64_t tileT = tileId / (tileNumH * tileNumW);
    const int64_t rem = tileId - tileT * tileNumH * tileNumW;
    const int64_t tileH = rem / tileNumW;
    const int64_t tileW = rem - tileH * tileNumW;

    const int64_t winT = HeadWindow(windowSize, head, 0);
    const int64_t winH = HeadWindow(windowSize, head, 1);
    const int64_t winW = HeadWindow(windowSize, head, 2);
    const int64_t tBegin = ClampStart(tileT, winT, tileNumT);
    const int64_t hBegin = ClampStart(tileH, winH, tileNumH);
    const int64_t wBegin = ClampStart(tileW, winW, tileNumW);
    const int64_t tEnd = std::min<int64_t>(tBegin + winT, tileNumT);
    const int64_t hEnd = std::min<int64_t>(hBegin + winH, tileNumH);
    const int64_t wEnd = std::min<int64_t>(wBegin + winW, tileNumW);

    for (int64_t tt = tBegin; tt < tEnd; ++tt) {
        for (int64_t hh = hBegin; hh < hEnd; ++hh) {
            for (int64_t ww = wBegin; ww < wEnd; ++ww) {
                const int64_t base = ((tt * tileNumH + hh) * tileNumW + ww) * tileTokenNum;
                for (int64_t offset = 0; offset < tileTokenNum; ++offset) {
                    const int64_t kvSeq = base + offset;
                    if (kvSeq < imageSeqLen) {
                        func(kvSeq);
                    }
                }
            }
        }
    }
    for (int64_t i = 0; i < RealTextLength(testCase); ++i) {
        func(imageSeqLen + i);
    }
}

template <TensorDtype DTYPE>
float DotScore(const TestCase &testCase, const std::vector<typename TensorDtypeTraits<DTYPE>::Storage> &q,
               const std::vector<typename TensorDtypeTraits<DTYPE>::Storage> &k, int64_t qBase, int64_t kvBase)
{
    float score = 0.0F;
    for (int64_t dim = 0; dim < testCase.headDim; ++dim) {
        score += TensorDtypeTraits<DTYPE>::ToFloat(q[qBase + dim]) * TensorDtypeTraits<DTYPE>::ToFloat(k[kvBase + dim]);
    }
    return score / std::sqrt(static_cast<float>(testCase.headDim));
}

template <TensorDtype DTYPE>
std::vector<float> ComputeCpuGolden(const TestCase &testCase,
                                    const std::vector<typename TensorDtypeTraits<DTYPE>::Storage> &q,
                                    const std::vector<typename TensorDtypeTraits<DTYPE>::Storage> &k,
                                    const std::vector<typename TensorDtypeTraits<DTYPE>::Storage> &v,
                                    const std::vector<int64_t> &windowSize)
{
    std::vector<float> y(testCase.batch * testCase.headNum * SeqLen(testCase) * testCase.headDim, 0.0F);
    for (int64_t b = 0; b < testCase.batch; ++b) {
        for (int64_t h = 0; h < testCase.headNum; ++h) {
            for (int64_t s = 0; s < SeqLen(testCase); ++s) {
                const int64_t qBase = TensorOffset(testCase, b, h, s, 0);
                float maxScore = -std::numeric_limits<float>::infinity();
                VisitKvSeq(testCase, h, s, windowSize, [&](int64_t kvSeq) {
                    const int64_t kvBase = TensorOffset(testCase, b, h, kvSeq, 0);
                    maxScore = std::max(maxScore, DotScore<DTYPE>(testCase, q, k, qBase, kvBase));
                });

                float denom = 0.0F;
                VisitKvSeq(testCase, h, s, windowSize, [&](int64_t kvSeq) {
                    const int64_t kvBase = TensorOffset(testCase, b, h, kvSeq, 0);
                    denom += std::exp(DotScore<DTYPE>(testCase, q, k, qBase, kvBase) - maxScore);
                });

                for (int64_t dim = 0; dim < testCase.headDim; ++dim) {
                    float acc = 0.0F;
                    VisitKvSeq(testCase, h, s, windowSize, [&](int64_t kvSeq) {
                        const int64_t kvBase = TensorOffset(testCase, b, h, kvSeq, 0);
                        const float weight = std::exp(DotScore<DTYPE>(testCase, q, k, qBase, kvBase) - maxScore);
                        acc += weight * TensorDtypeTraits<DTYPE>::ToFloat(v[kvBase + dim]);
                    });
                    y[qBase + dim] = denom > 0.0F ? acc / denom : 0.0F;
                }
            }
        }
    }
    return y;
}

template <TensorDtype DTYPE>
void FillInput(std::vector<typename TensorDtypeTraits<DTYPE>::Storage> &data, float phase)
{
    for (size_t i = 0; i < data.size(); ++i) {
        const float value = 0.2F * std::sin(static_cast<float>(i) * 0.013F + phase) +
                            0.1F * std::cos(static_cast<float>(i) * 0.007F + phase * 0.5F);
        data[i] = TensorDtypeTraits<DTYPE>::FromFloat(value);
    }
}

template <TensorDtype DTYPE>
void FillPerfInput(std::vector<typename TensorDtypeTraits<DTYPE>::Storage> &data, float phase)
{
    for (size_t i = 0; i < data.size(); ++i) {
        const float value = (static_cast<float>(i % 251U) - 125.0F) * 0.001F + phase * 0.01F;
        data[i] = TensorDtypeTraits<DTYPE>::FromFloat(value);
    }
}

template <TensorDtype DTYPE>
int CompareResult(const std::vector<float> &expected,
                  const std::vector<typename TensorDtypeTraits<DTYPE>::Storage> &actual)
{
    float maxAbsDiff = 0.0F;
    int64_t mismatchCount = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const float actualVal = TensorDtypeTraits<DTYPE>::ToFloat(actual[i]);
        const float diff = std::fabs(expected[i] - actualVal);
        const float tolerance = ATOL + RTOL * std::fabs(expected[i]);
        maxAbsDiff = std::max(maxAbsDiff, diff);
        if (diff > tolerance) {
            if (mismatchCount < 5) {
                LOG_PRINT("mismatch[%zu]: expect=%f actual=%f diff=%f tolerance=%f\n", i, expected[i], actualVal,
                          diff, tolerance);
            }
            ++mismatchCount;
        }
    }
    LOG_PRINT("SlidingTileAttention max_abs_diff=%f mismatch_count=%ld\n", maxAbsDiff, mismatchCount);
    return mismatchCount == 0 ? ACL_SUCCESS : 1;
}

int RunSlidingTileAttentionOnce(aclTensor *q, aclTensor *k, aclTensor *v, aclTensor *y, aclIntArray *windowSize,
                                const TestCase &testCase, const std::string &seqShape, aclrtStream stream,
                                const char *stage)
{
    uint64_t workspaceSize = 0;
    void *workspaceAddr = nullptr;
    aclOpExecutor *executor = nullptr;
    auto ret = aclnnSlidingTileAttentionGetWorkspaceSize(q, k, v, windowSize, testCase.textLength, testCase.hasText,
                                                         const_cast<char *>(seqShape.c_str()), y, &workspaceSize,
                                                         &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("%s aclnnSlidingTileAttentionGetWorkspaceSize failed. ERROR: %d\n", stage, ret); return ret);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s allocate workspace failed. ERROR: %d\n", stage, ret); return ret);
    }

    ret = aclnnSlidingTileAttention(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s aclnnSlidingTileAttention failed. ERROR: %d\n", stage, ret);
              if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("%s aclrtSynchronizeStream failed. ERROR: %d\n[ERROR msg]%s\n", stage, ret,
                        aclGetRecentErrMsg());
              if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } return ret);

    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    return ACL_SUCCESS;
}

std::vector<TestCase> BuildTestCases()
{
    return {
        // {"golden_text_and_per_head_windows",
        //  TensorDtype::FP16,
        //  BATCH,
        //  HEAD_NUM,
        //  SEQ_T,
        //  SEQ_H,
        //  SEQ_W,
        //  TEXT_LENGTH,
        //  true,
        //  HEAD_DIM,
        //  {1, 1, 1, 1, 2, 1},
        //  true,
        //  ""},
        // {"golden_bf16_text_and_per_head_windows",
        //  TensorDtype::BF16,
        //  BATCH,
        //  HEAD_NUM,
        //  SEQ_T,
        //  SEQ_H,
        //  SEQ_W,
        //  TEXT_LENGTH,
        //  true,
        //  HEAD_DIM,
        //  {1, 1, 1, 1, 2, 1},
        //  true,
        //  ""},
        {"perf_a100_b1_h8_d64_img_30x48x80_text_128_mixed_windows",
         TensorDtype::FP16,
         1,
         8,
         30,
         48,
         80,
         128,
         true,
         64,
         {1, 1, 1, 3, 3, 3, 3, 5, 5, 5, 5, 5, 1, 3, 3, 3, 3, 5, 5, 5, 7, 7, 7, 7},
         false,
         "GPU A100: 12510 calls, total 6.565 s, avg 524.79 us"},
        {"perf_a100_b1_h8_d64_img_36x48x48_no_text_window_3x3x3",
         TensorDtype::FP16,
         1,
         8,
         36,
         48,
         48,
         0,
         false,
         64,
         {3, 3, 3},
         false,
         "GPU A100: 2204 calls, total 10.177 s, avg 4618 us"},
    };
}

template <TensorDtype DTYPE>
int RunTestCaseWithDtype(const TestCase &testCase, aclrtStream stream)
{
    using Traits = TensorDtypeTraits<DTYPE>;
    using Storage = typename Traits::Storage;

    LOG_PRINT("Running %s: Dtype=%s B=%ld H=%ld D=%ld ImgSeq=%ld TextLen=%ld WindowSize=%zu SeqShape=%s\n",
              testCase.name.c_str(), Traits::NAME, testCase.batch, testCase.headNum, testCase.headDim,
              ImageSeqLen(testCase), RealTextLength(testCase), testCase.windowSize.size(),
              SeqShapeString(testCase).c_str());
    if (!testCase.targetPerformance.empty()) {
        LOG_PRINT("Target performance reference: %s\n", testCase.targetPerformance.c_str());
    }

    const std::vector<int64_t> qkvShape = {testCase.batch, testCase.headNum, SeqLen(testCase), testCase.headDim};
    const int64_t tensorSize = GetShapeSize(qkvShape);
    std::vector<Storage> qHostData(tensorSize);
    std::vector<Storage> kHostData(tensorSize);
    std::vector<Storage> vHostData(tensorSize);
    std::vector<Storage> yHostData(tensorSize, Traits::FromFloat(0.0F));
    if (testCase.checkGolden) {
        FillInput<DTYPE>(qHostData, 0.1F);
        FillInput<DTYPE>(kHostData, 0.7F);
        FillInput<DTYPE>(vHostData, 1.3F);
    } else {
        FillPerfInput<DTYPE>(qHostData, 0.1F);
        FillPerfInput<DTYPE>(kHostData, 0.7F);
        FillPerfInput<DTYPE>(vHostData, 1.3F);
    }

    std::vector<float> expected;
    if (testCase.checkGolden) {
        expected = ComputeCpuGolden<DTYPE>(testCase, qHostData, kHostData, vHostData, testCase.windowSize);
    }

    void *qDeviceAddr = nullptr;
    void *kDeviceAddr = nullptr;
    void *vDeviceAddr = nullptr;
    void *yDeviceAddr = nullptr;
    aclTensor *q = nullptr;
    aclTensor *k = nullptr;
    aclTensor *v = nullptr;
    aclTensor *y = nullptr;
    aclIntArray *windowSize = nullptr;

    int ret = CreateAclTensor(qHostData, qkvShape, &qDeviceAddr, Traits::ACL_DTYPE, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHostData, qkvShape, &kDeviceAddr, Traits::ACL_DTYPE, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHostData, qkvShape, &vDeviceAddr, Traits::ACL_DTYPE, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, qkvShape, &yDeviceAddr, Traits::ACL_DTYPE, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    windowSize = aclCreateIntArray(testCase.windowSize.data(), testCase.windowSize.size());
    std::string seqShape = SeqShapeString(testCase);

    LOG_PRINT("Warm up %s...\n", testCase.name.c_str());
    ret = RunSlidingTileAttentionOnce(q, k, v, y, windowSize, testCase, seqShape, stream, "warm up");
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = RunSlidingTileAttentionOnce(q, k, v, y, windowSize, testCase, seqShape, stream, "run");
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    if (testCase.checkGolden) {
        std::vector<Storage> actual(tensorSize, Traits::FromFloat(0.0F));
        ret = aclrtMemcpy(actual.data(), tensorSize * static_cast<int64_t>(sizeof(actual[0])), yDeviceAddr,
                          tensorSize * static_cast<int64_t>(sizeof(actual[0])), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

        ret = CompareResult<DTYPE>(expected, actual);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s golden compare failed.\n", testCase.name.c_str()); return ret);
    }
    LOG_PRINT("%s executed successfully.\n", testCase.name.c_str());

    aclDestroyIntArray(windowSize);
    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(v);
    aclDestroyTensor(y);
    aclrtFree(qDeviceAddr);
    aclrtFree(kDeviceAddr);
    aclrtFree(vDeviceAddr);
    aclrtFree(yDeviceAddr);
    return ACL_SUCCESS;
}

int RunTestCase(const TestCase &testCase, aclrtStream stream)
{
    switch (testCase.dtype) {
        case TensorDtype::FP16:
            return RunTestCaseWithDtype<TensorDtype::FP16>(testCase, stream);
        case TensorDtype::BF16:
            return RunTestCaseWithDtype<TensorDtype::BF16>(testCase, stream);
        default:
            LOG_PRINT("Unsupported dtype %s.\n", DtypeName(testCase.dtype));
            return 1;
    }
}
} // namespace

int main()
{
    int32_t deviceId = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    for (const auto &testCase : BuildTestCases()) {
        ret = RunTestCase(testCase, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("SlidingTileAttention case %s failed.\n", testCase.name.c_str());
                  return ret);
    }
    LOG_PRINT("SlidingTileAttention aclnn tests executed successfully.\n");

    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ACL_SUCCESS;
}
