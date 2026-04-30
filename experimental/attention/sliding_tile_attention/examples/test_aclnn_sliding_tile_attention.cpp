/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
constexpr int64_t IMAGE_SEQ_LEN = SEQ_T * SEQ_H * SEQ_W;
constexpr int64_t TEXT_LENGTH = 2;
constexpr int64_t SEQ_LEN = IMAGE_SEQ_LEN + TEXT_LENGTH;
constexpr int64_t HEAD_DIM = 8;
constexpr float ATOL = 5e-2F;
constexpr float RTOL = 5e-2F;

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

int64_t TensorOffset(int64_t batch, int64_t head, int64_t seq, int64_t dim)
{
    return (((batch * HEAD_NUM + head) * SEQ_LEN + seq) * HEAD_DIM + dim);
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
void VisitKvSeq(int64_t head, int64_t querySeq, const std::vector<int64_t> &windowSize, Func func)
{
    const bool isTextQuery = querySeq >= IMAGE_SEQ_LEN && querySeq < SEQ_LEN;
    if (isTextQuery) {
        for (int64_t kvSeq = 0; kvSeq < SEQ_LEN; ++kvSeq) {
            func(kvSeq);
        }
        return;
    }

    const int64_t tileTokenNum = TILE_T * TILE_H * TILE_W;
    const int64_t tileNumT = SEQ_T / TILE_T;
    const int64_t tileNumH = SEQ_H / TILE_H;
    const int64_t tileNumW = SEQ_W / TILE_W;
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
                    if (kvSeq < IMAGE_SEQ_LEN) {
                        func(kvSeq);
                    }
                }
            }
        }
    }
    for (int64_t i = 0; i < TEXT_LENGTH; ++i) {
        func(IMAGE_SEQ_LEN + i);
    }
}

float DotScore(const std::vector<aclFloat16> &q, const std::vector<aclFloat16> &k, int64_t qBase, int64_t kvBase)
{
    float score = 0.0F;
    for (int64_t dim = 0; dim < HEAD_DIM; ++dim) {
        score += aclFloat16ToFloat(q[qBase + dim]) * aclFloat16ToFloat(k[kvBase + dim]);
    }
    return score / std::sqrt(static_cast<float>(HEAD_DIM));
}

std::vector<float> ComputeCpuGolden(const std::vector<aclFloat16> &q, const std::vector<aclFloat16> &k,
                                    const std::vector<aclFloat16> &v, const std::vector<int64_t> &windowSize)
{
    std::vector<float> y(BATCH * HEAD_NUM * SEQ_LEN * HEAD_DIM, 0.0F);
    for (int64_t b = 0; b < BATCH; ++b) {
        for (int64_t h = 0; h < HEAD_NUM; ++h) {
            for (int64_t s = 0; s < SEQ_LEN; ++s) {
                const int64_t qBase = TensorOffset(b, h, s, 0);
                float maxScore = -std::numeric_limits<float>::infinity();
                VisitKvSeq(h, s, windowSize, [&](int64_t kvSeq) {
                    const int64_t kvBase = TensorOffset(b, h, kvSeq, 0);
                    maxScore = std::max(maxScore, DotScore(q, k, qBase, kvBase));
                });

                float denom = 0.0F;
                VisitKvSeq(h, s, windowSize, [&](int64_t kvSeq) {
                    const int64_t kvBase = TensorOffset(b, h, kvSeq, 0);
                    denom += std::exp(DotScore(q, k, qBase, kvBase) - maxScore);
                });

                for (int64_t dim = 0; dim < HEAD_DIM; ++dim) {
                    float acc = 0.0F;
                    VisitKvSeq(h, s, windowSize, [&](int64_t kvSeq) {
                        const int64_t kvBase = TensorOffset(b, h, kvSeq, 0);
                        const float weight = std::exp(DotScore(q, k, qBase, kvBase) - maxScore);
                        acc += weight * aclFloat16ToFloat(v[kvBase + dim]);
                    });
                    y[qBase + dim] = denom > 0.0F ? acc / denom : 0.0F;
                }
            }
        }
    }
    return y;
}

void FillInput(std::vector<aclFloat16> &data, float phase)
{
    for (size_t i = 0; i < data.size(); ++i) {
        const float value = 0.2F * std::sin(static_cast<float>(i) * 0.013F + phase) +
                            0.1F * std::cos(static_cast<float>(i) * 0.007F + phase * 0.5F);
        data[i] = aclFloatToFloat16(value);
    }
}

int CompareResult(const std::vector<float> &expected, const std::vector<aclFloat16> &actual)
{
    float maxAbsDiff = 0.0F;
    int64_t mismatchCount = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const float actualVal = aclFloat16ToFloat(actual[i]);
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
} // namespace

int main()
{
    int32_t deviceId = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    const std::vector<int64_t> qkvShape = {BATCH, HEAD_NUM, SEQ_LEN, HEAD_DIM};
    const int64_t tensorSize = GetShapeSize(qkvShape);
    std::vector<aclFloat16> qHostData(tensorSize);
    std::vector<aclFloat16> kHostData(tensorSize);
    std::vector<aclFloat16> vHostData(tensorSize);
    std::vector<aclFloat16> yHostData(tensorSize, aclFloatToFloat16(0.0F));
    FillInput(qHostData, 0.1F);
    FillInput(kHostData, 0.7F);
    FillInput(vHostData, 1.3F);

    const std::vector<int64_t> windowHostData = {1, 1, 1, 1, 2, 1};
    std::vector<float> expected = ComputeCpuGolden(qHostData, kHostData, vHostData, windowHostData);

    void *qDeviceAddr = nullptr;
    void *kDeviceAddr = nullptr;
    void *vDeviceAddr = nullptr;
    void *yDeviceAddr = nullptr;
    aclTensor *q = nullptr;
    aclTensor *k = nullptr;
    aclTensor *v = nullptr;
    aclTensor *y = nullptr;

    ret = CreateAclTensor(qHostData, qkvShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHostData, qkvShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHostData, qkvShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, qkvShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclIntArray *windowSize = aclCreateIntArray(windowHostData.data(), windowHostData.size());
    char seqShape[] = "6x16x8";
    bool hasText = true;

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnSlidingTileAttentionGetWorkspaceSize(q, k, v, windowSize, TEXT_LENGTH, hasText, seqShape, y,
                                                    &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnSlidingTileAttentionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnSlidingTileAttention(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSlidingTileAttention failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    std::vector<aclFloat16> actual(tensorSize, aclFloatToFloat16(0.0F));
    ret = aclrtMemcpy(actual.data(), tensorSize * static_cast<int64_t>(sizeof(actual[0])), yDeviceAddr,
                      tensorSize * static_cast<int64_t>(sizeof(actual[0])), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    ret = CompareResult(expected, actual);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("SlidingTileAttention golden compare failed.\n"); return ret);
    LOG_PRINT("SlidingTileAttention aclnn execute and golden compare successfully.\n");

    aclDestroyIntArray(windowSize);
    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(v);
    aclDestroyTensor(y);
    aclrtFree(qDeviceAddr);
    aclrtFree(kDeviceAddr);
    aclrtFree(vDeviceAddr);
    aclrtFree(yDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ACL_SUCCESS;
}
