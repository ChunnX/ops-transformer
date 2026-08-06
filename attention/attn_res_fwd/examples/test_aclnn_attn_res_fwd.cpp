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
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_attn_res_fwd.h"

namespace {

constexpr double kMereThreshold = 1.0 / 128.0; // 2^-7
constexpr double kNormEps = 1e-5;

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

float Bf16BitsToFloat(uint16_t bits)
{
    uint32_t sign = (bits >> 15) & 0x1;
    uint32_t exp = (bits >> 7) & 0xFF;
    uint32_t mant = bits & 0x7F;
    uint32_t fbits;
    if (exp == 0) {
        if (mant == 0) {
            fbits = sign << 31;
        } else {
            while ((mant & 0x80) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x7F;
            fbits = (sign << 31) | ((exp + 127 - 127) << 23) | (mant << 16);
        }
    } else if (exp == 0xFF) {
        fbits = (sign << 31) | (0xFFu << 23) | (mant << 16);
    } else {
        fbits = (sign << 31) | ((exp + 127 - 127) << 23) | (mant << 16);
    }
    float value;
    std::memcpy(&value, &fbits, sizeof(value));
    return value;
}

uint16_t FloatToBf16Bits(float value)
{
    uint32_t fbits;
    std::memcpy(&fbits, &value, sizeof(fbits));
    uint32_t sign = (fbits >> 31) & 0x1;
    int32_t exp = static_cast<int32_t>((fbits >> 23) & 0xFF) - 127 + 127;
    uint32_t mant = (fbits >> 16) & 0x7F;
    if (exp <= 0) {
        return static_cast<uint16_t>(sign << 15);
    }
    if (exp >= 0xFF) {
        return static_cast<uint16_t>((sign << 15) | (0xFFu << 7));
    }
    return static_cast<uint16_t>((sign << 15) | (static_cast<uint32_t>(exp) << 7) | mant);
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
    auto size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

void ApplyAttnResCustGolden(int64_t numTokens, int64_t numBlocks, int64_t hiddenSize,
                            const std::vector<uint16_t> &prefixHost, const std::vector<uint16_t> &blockHost,
                            const std::vector<uint16_t> &projHost, const std::vector<uint16_t> &normHost,
                            std::vector<uint16_t> &outHost)
{
    const int64_t blockCount = numBlocks + 1;
    std::vector<float> scoreWeight(hiddenSize);
    for (int64_t h = 0; h < hiddenSize; ++h) {
        scoreWeight[h] = Bf16BitsToFloat(normHost[h]) * Bf16BitsToFloat(projHost[h]);
    }

    outHost.assign(static_cast<size_t>(numTokens * hiddenSize), 0);
    std::vector<float> vBlock(blockCount * hiddenSize);
    std::vector<float> scores(blockCount);
    std::vector<float> probs(blockCount);

    for (int64_t t = 0; t < numTokens; ++t) {
        for (int64_t n = 0; n < numBlocks; ++n) {
            for (int64_t h = 0; h < hiddenSize; ++h) {
                const size_t idx = static_cast<size_t>((t * numBlocks + n) * hiddenSize + h);
                vBlock[static_cast<size_t>(n * hiddenSize + h)] = Bf16BitsToFloat(blockHost[idx]);
            }
        }
        for (int64_t h = 0; h < hiddenSize; ++h) {
            const size_t idx = static_cast<size_t>(t * hiddenSize + h);
            vBlock[static_cast<size_t>(numBlocks * hiddenSize + h)] = Bf16BitsToFloat(prefixHost[idx]);
        }

        for (int64_t n = 0; n < blockCount; ++n) {
            double sumSq = 0.0;
            for (int64_t h = 0; h < hiddenSize; ++h) {
                const float val = vBlock[static_cast<size_t>(n * hiddenSize + h)];
                sumSq += static_cast<double>(val) * val;
            }
            const float invStd = static_cast<float>(1.0 / std::sqrt(sumSq / static_cast<double>(hiddenSize) + kNormEps));
            float score = 0.0f;
            for (int64_t h = 0; h < hiddenSize; ++h) {
                const float kVal = vBlock[static_cast<size_t>(n * hiddenSize + h)] * invStd;
                score += kVal * scoreWeight[h];
            }
            scores[static_cast<size_t>(n)] = score;
        }

        float maxScore = scores[0];
        for (int64_t n = 1; n < blockCount; ++n) {
            maxScore = std::max(maxScore, scores[static_cast<size_t>(n)]);
        }
        double expSum = 0.0;
        for (int64_t n = 0; n < blockCount; ++n) {
            probs[static_cast<size_t>(n)] = std::exp(static_cast<double>(scores[static_cast<size_t>(n)] - maxScore));
            expSum += probs[static_cast<size_t>(n)];
        }
        for (int64_t n = 0; n < blockCount; ++n) {
            probs[static_cast<size_t>(n)] = static_cast<float>(probs[static_cast<size_t>(n)] / expSum);
        }

        for (int64_t h = 0; h < hiddenSize; ++h) {
            float outVal = 0.0f;
            for (int64_t n = 0; n < blockCount; ++n) {
                outVal += probs[static_cast<size_t>(n)] * vBlock[static_cast<size_t>(n * hiddenSize + h)];
            }
            outHost[static_cast<size_t>(t * hiddenSize + h)] = FloatToBf16Bits(outVal);
        }
    }
}

struct PrecisionStats {
    double mere{0.0};
    double mare{0.0};
    bool pass{false};
};

PrecisionStats CompareOutputs(const std::vector<uint16_t> &golden, const std::vector<uint16_t> &actual)
{
    PrecisionStats stats;
    const size_t count = golden.size();
    double maxRel = 0.0;
    double maxAbs = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const float g = Bf16BitsToFloat(golden[i]);
        const float a = Bf16BitsToFloat(actual[i]);
        const double absErr = std::abs(static_cast<double>(a - g));
        const double denom = std::max(std::abs(static_cast<double>(g)), 1e-10);
        const double relErr = absErr / denom;
        maxRel = std::max(maxRel, relErr);
        maxAbs = std::max(maxAbs, absErr);
    }
    stats.mere = maxRel;
    stats.mare = maxAbs;
    stats.pass = maxRel < kMereThreshold;
    return stats;
}

void FillRandomBf16(std::vector<uint16_t> &data, std::mt19937 &rng)
{
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto &item : data) {
        item = FloatToBf16Bits(dist(rng));
    }
}

} // namespace

int main(int argc, char **argv)
{
    int64_t numTokens = 544;
    int64_t numBlocks = 12;
    int64_t hiddenSize = 7168;
    int32_t deviceId = 0;
    uint32_t seed = 0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--T") == 0 && i + 1 < argc) {
            numTokens = std::atoll(argv[++i]);
        } else if (std::strcmp(argv[i], "--N") == 0 && i + 1 < argc) {
            numBlocks = std::atoll(argv[++i]);
        } else if (std::strcmp(argv[i], "--H") == 0 && i + 1 < argc) {
            hiddenSize = std::atoll(argv[++i]);
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            deviceId = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = static_cast<uint32_t>(std::atoi(argv[++i]));
        }
    }

    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::mt19937 rng(seed);
    std::vector<int64_t> prefixShape = {numTokens, hiddenSize};
    std::vector<int64_t> blockShape = {numTokens, numBlocks, hiddenSize};
    std::vector<int64_t> projShape = {1, hiddenSize};
    std::vector<int64_t> normShape = {hiddenSize};
    std::vector<int64_t> outShape = {numTokens, hiddenSize};

    std::vector<uint16_t> prefixHost(static_cast<size_t>(GetShapeSize(prefixShape)));
    std::vector<uint16_t> blockHost(static_cast<size_t>(GetShapeSize(blockShape)));
    std::vector<uint16_t> projHost(static_cast<size_t>(GetShapeSize(projShape)));
    std::vector<uint16_t> normHost(static_cast<size_t>(GetShapeSize(normShape)));
    std::vector<uint16_t> outHost(static_cast<size_t>(GetShapeSize(outShape)));
    std::vector<uint16_t> goldenHost;

    FillRandomBf16(prefixHost, rng);
    FillRandomBf16(blockHost, rng);
    FillRandomBf16(projHost, rng);
    FillRandomBf16(normHost, rng);
    ApplyAttnResCustGolden(numTokens, numBlocks, hiddenSize, prefixHost, blockHost, projHost, normHost, goldenHost);

    void *prefixDev = nullptr;
    void *blockDev = nullptr;
    void *projDev = nullptr;
    void *normDev = nullptr;
    void *outDev = nullptr;
    aclTensor *prefixTensor = nullptr;
    aclTensor *blockTensor = nullptr;
    aclTensor *projTensor = nullptr;
    aclTensor *normTensor = nullptr;
    aclTensor *outTensor = nullptr;

    ret = CreateAclTensor(prefixHost, prefixShape, &prefixDev, ACL_BF16, &prefixTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(blockHost, blockShape, &blockDev, ACL_BF16, &blockTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(projHost, projShape, &projDev, ACL_BF16, &projTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normHost, normShape, &normDev, ACL_BF16, &normTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHost, outShape, &outDev, ACL_BF16, &outTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnAttnResFwdGetWorkspaceSize(prefixTensor, blockTensor, projTensor, normTensor, kNormEps, false,
                                          outTensor, nullptr, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAttnResFwdGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("workspace aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnAttnResFwd(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAttnResFwd failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(outHost.data(), outHost.size() * sizeof(uint16_t), outDev, outHost.size() * sizeof(uint16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy output failed. ERROR: %d\n", ret); return ret);

    const PrecisionStats stats = CompareOutputs(goldenHost, outHost);
    LOG_PRINT("AttnResFwd precision: T=%ld N=%ld H=%ld seed=%u workspaceSize=%lu\n", numTokens, numBlocks,
              hiddenSize, seed, static_cast<unsigned long>(workspaceSize));
    LOG_PRINT("sample golden[0..2]: %.6f %.6f %.6f\n", Bf16BitsToFloat(goldenHost[0]), Bf16BitsToFloat(goldenHost[1]),
              Bf16BitsToFloat(goldenHost[2]));
    LOG_PRINT("sample actual[0..2]: %.6f %.6f %.6f\n", Bf16BitsToFloat(outHost[0]), Bf16BitsToFloat(outHost[1]),
              Bf16BitsToFloat(outHost[2]));
    LOG_PRINT("MERE=%.8f MARE=%.8f threshold=%.8f result=%s\n", stats.mere, stats.mare, kMereThreshold,
              stats.pass ? "pass" : "fail");

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclDestroyTensor(prefixTensor);
    aclDestroyTensor(blockTensor);
    aclDestroyTensor(projTensor);
    aclDestroyTensor(normTensor);
    aclDestroyTensor(outTensor);
    aclrtFree(prefixDev);
    aclrtFree(blockDev);
    aclrtFree(projDev);
    aclrtFree(normDev);
    aclrtFree(outDev);
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return stats.pass ? 0 : 1;
}
