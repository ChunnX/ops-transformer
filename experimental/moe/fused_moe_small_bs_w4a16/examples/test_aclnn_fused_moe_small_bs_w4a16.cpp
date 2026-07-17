/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_fused_moe_small_bs_w4a16.cpp
 * \brief aclnnFusedMoeSmallBsW4a16 功能验证用例
 *
 * 测试流程：
 * 1. Host 侧生成确定性伪随机输入（x, w13/w2 打包 int4 权重, scales, expert_ids, topk_weights）
 * 2. Host 侧用 FP32 参考实现计算 golden（模拟 kernel 的量化/精度路径）
 * 3. 调用 aclnnFusedMoeSmallBsW4a16 两段式接口在 NPU 上执行
 * 4. 对比 NPU 输出与 golden，打印最大误差并判定 PASS/FAIL
 *
 * 覆盖 fp16/bf16 两种数据类型与多组 shape（含 K 方向多 block 原子加场景）。
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_fused_moe_small_bs_w4a16.h"

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

// -----------------------------------------------------------------------------
// 测试配置（hiddenSize/interSize 必须为 128 的倍数）
// -----------------------------------------------------------------------------
struct TestShape {
    int64_t batchSize;
    int64_t topK;
    int64_t numExperts;
    int64_t hiddenSize;
    int64_t interSize;
};

constexpr int64_t GROUP_SIZE = 32;
constexpr int64_t PACK_RATIO = 8;

constexpr float ATOL = 3e-2f;
constexpr float RTOL = 3e-2f;

// -----------------------------------------------------------------------------
// 精度转换辅助：fp16/bf16 与 fp32 互转（host 侧）
// -----------------------------------------------------------------------------
static uint16_t FloatToFp16Bits(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) {
        return static_cast<uint16_t>(sign); // 简化：下溢直接归零（测试数据不会触发）
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u); // 上溢转 inf（测试数据不会触发）
    }
    // round to nearest even
    uint32_t rounded = mant + 0x1000u;
    uint32_t outMant = rounded >> 13;
    if (outMant & 0x400u) {
        exp += 1;
        outMant = 0;
        if (exp >= 31) {
            return static_cast<uint16_t>(sign | 0x7C00u);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (outMant & 0x3FFu));
}

static float Fp16BitsToFloat(uint16_t h)
{
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        f = sign; // 简化：非规格化数按 0 处理
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

static uint16_t FloatToBf16Bits(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t lsb = (x >> 16) & 1u;
    x += 0x7FFFu + lsb; // round to nearest even
    return static_cast<uint16_t>(x >> 16);
}

static float Bf16BitsToFloat(uint16_t b)
{
    uint32_t f = static_cast<uint32_t>(b) << 16;
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

static std::vector<float> BitsToFloat(const std::vector<uint16_t> &bits, bool isBf16)
{
    std::vector<float> out(bits.size());
    for (size_t i = 0; i < bits.size(); ++i) {
        out[i] = isBf16 ? Bf16BitsToFloat(bits[i]) : Fp16BitsToFloat(bits[i]);
    }
    return out;
}

// 模拟 kernel 计算路径：T 输入会先转 fp32 再转 fp16 参与计算
static float QuantForCompute(float v, bool isBf16)
{
    float qt = isBf16 ? Bf16BitsToFloat(FloatToBf16Bits(v)) : Fp16BitsToFloat(FloatToFp16Bits(v));
    return Fp16BitsToFloat(FloatToFp16Bits(qt));
}

// 模拟 cast fp32 -> T (CAST_ROUND)
static float QuantForStore(float v, bool isBf16)
{
    return isBf16 ? Bf16BitsToFloat(FloatToBf16Bits(v)) : Fp16BitsToFloat(FloatToFp16Bits(v));
}

// -----------------------------------------------------------------------------
// 确定性伪随机数（LCG）
// -----------------------------------------------------------------------------
class Lcg {
public:
    explicit Lcg(uint64_t seed) : state_(seed) {}
    float Uniform(float lo, float hi)
    {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        float u = static_cast<float>((state_ >> 33) & 0xFFFFFFu) / static_cast<float>(0x1000000u);
        return lo + (hi - lo) * u;
    }
    int32_t RandInt4() { return static_cast<int32_t>(Uniform(-8.0f, 8.0f)); } // (-8, 8) 截断 -> [-7, 7]

private:
    uint64_t state_;
};

// 将 8 个连续 N 方向的 int4 值打包为一个 int32（低位在前）
static int32_t PackInt4x8(const int8_t *v)
{
    uint32_t packed = 0;
    for (int j = 0; j < PACK_RATIO; ++j) {
        packed |= (static_cast<uint32_t>(v[j]) & 0xFu) << (4 * j);
    }
    return static_cast<int32_t>(packed);
}

// -----------------------------------------------------------------------------
// Host 参考实现（FP32，模拟 kernel 精度路径）
// weight: [E, K, N/8] int32(int4打包)  scales: [E, K/32, N]
// 返回每个 token 的 GEMV 结果 [N]（FP32）
// -----------------------------------------------------------------------------
static void RefGemvW4A16(const std::vector<float> &xUse, // [K]，已按 compute 精度量化
                         const std::vector<int32_t> &weight, const std::vector<float> &scales, int64_t expertId,
                         int64_t inDim, int64_t outDim, std::vector<float> &out)
{
    int64_t outPacked = outDim / PACK_RATIO;
    int64_t numGroups = inDim / GROUP_SIZE;
    out.assign(outDim, 0.0f);
    for (int64_t n = 0; n < outDim; ++n) {
        double acc = 0.0;
        for (int64_t g = 0; g < numGroups; ++g) {
            double groupAcc = 0.0;
            for (int64_t j = 0; j < GROUP_SIZE; ++j) {
                int64_t k = g * GROUP_SIZE + j;
                int32_t w4 = (static_cast<uint32_t>(weight[(expertId * inDim + k) * outPacked + n / PACK_RATIO]) >>
                              (4 * (n % PACK_RATIO))) &
                             0xFu;
                int32_t wSext = (w4 >= 8) ? (w4 - 16) : w4; // int4 符号扩展
                groupAcc += static_cast<double>(xUse[k]) * wSext;
            }
            acc += static_cast<double>(scales[(expertId * numGroups + g) * outDim + n]) * groupAcc;
        }
        out[n] = static_cast<float>(acc);
    }
}

// 完整 MoE 参考：W13 -> SwiGLU -> W2 -> 加权求和
static std::vector<float> RefMoe(const TestShape &shape, const std::vector<float> &x,
                                 const std::vector<int32_t> &w13Weight, const std::vector<float> &w13Scales,
                                 const std::vector<int32_t> &w2Weight, const std::vector<float> &w2Scales,
                                 const std::vector<int32_t> &expertIds, const std::vector<float> &topkWeights,
                                 bool isBf16)
{
    int64_t totalTokens = shape.batchSize * shape.topK;
    std::vector<float> y(shape.batchSize * shape.hiddenSize, 0.0f);
    for (int64_t t = 0; t < totalTokens; ++t) {
        int64_t b = t / shape.topK;
        int64_t e = expertIds[t];
        // x 按 compute 精度量化（bf16 路径会先转 fp16）
        std::vector<float> xUse(shape.hiddenSize);
        for (int64_t k = 0; k < shape.hiddenSize; ++k) {
            xUse[k] = QuantForCompute(x[b * shape.hiddenSize + k], isBf16);
        }
        // W13 GEMV -> [2 * InterSize]
        std::vector<float> w13Out;
        RefGemvW4A16(xUse, w13Weight, w13Scales, e, shape.hiddenSize, 2 * shape.interSize, w13Out);
        // SwiGLU: silu(gate) * value，再 cast 到 T，并模拟 T->fp16 的 compute 精度
        std::vector<float> actUse(shape.interSize);
        for (int64_t i = 0; i < shape.interSize; ++i) {
            double gate = w13Out[i];
            double act = gate / (1.0 + std::exp(-gate)) * static_cast<double>(w13Out[shape.interSize + i]);
            actUse[i] = QuantForCompute(static_cast<float>(act), isBf16);
        }
        // W2 GEMV -> [HiddenSize]
        std::vector<float> w2Out;
        RefGemvW4A16(actUse, w2Weight, w2Scales, e, shape.interSize, shape.hiddenSize, w2Out);
        // topk 加权累加
        for (int64_t n = 0; n < shape.hiddenSize; ++n) {
            y[b * shape.hiddenSize + n] += topkWeights[t] * w2Out[n];
        }
    }
    // 最终 cast 到 T
    for (auto &v : y) {
        v = QuantForStore(v, isBf16);
    }
    return y;
}

// -----------------------------------------------------------------------------
// ACL 基础设施
// -----------------------------------------------------------------------------
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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

// -----------------------------------------------------------------------------
// 单个 dtype x shape 的测试用例
// -----------------------------------------------------------------------------
int RunTestCase(bool isBf16, const TestShape &shape, uint64_t seed, aclrtStream stream)
{
    const char *tag = isBf16 ? "bf16" : "fp16";
    aclDataType tDtype = isBf16 ? aclDataType::ACL_BF16 : aclDataType::ACL_FLOAT16;
    int64_t totalTokens = shape.batchSize * shape.topK;
    LOG_PRINT("[%s] shape: B=%ld K=%ld E=%ld H=%ld I=%ld\n", tag, shape.batchSize, shape.topK, shape.numExperts,
              shape.hiddenSize, shape.interSize);

    // 1. 生成输入数据
    Lcg rng(seed);
    std::vector<float> xF(shape.batchSize * shape.hiddenSize);
    for (auto &v : xF) {
        v = rng.Uniform(-1.0f, 1.0f);
    }
    std::vector<uint16_t> xBits(xF.size());
    for (size_t i = 0; i < xF.size(); ++i) {
        xBits[i] = isBf16 ? FloatToBf16Bits(xF[i]) : FloatToFp16Bits(xF[i]);
        xF[i] = isBf16 ? Bf16BitsToFloat(xBits[i]) : Fp16BitsToFloat(xBits[i]); // x 的实际值
    }

    std::vector<int8_t> w13Int4(shape.numExperts * shape.hiddenSize * 2 * shape.interSize);
    for (auto &v : w13Int4) {
        v = static_cast<int8_t>(rng.RandInt4());
    }
    std::vector<int32_t> w13Weight(shape.numExperts * shape.hiddenSize * (2 * shape.interSize / PACK_RATIO));
    for (int64_t idx = 0; idx < static_cast<int64_t>(w13Weight.size()); ++idx) {
        w13Weight[idx] = PackInt4x8(&w13Int4[idx * PACK_RATIO]);
    }
    std::vector<float> w13ScalesF(shape.numExperts * (shape.hiddenSize / GROUP_SIZE) * 2 * shape.interSize);
    std::vector<uint16_t> w13ScalesBits(w13ScalesF.size());
    for (size_t i = 0; i < w13ScalesF.size(); ++i) {
        float v = rng.Uniform(0.005f, 0.03f);
        w13ScalesBits[i] = isBf16 ? FloatToBf16Bits(v) : FloatToFp16Bits(v);
        w13ScalesF[i] = isBf16 ? Bf16BitsToFloat(w13ScalesBits[i]) : Fp16BitsToFloat(w13ScalesBits[i]);
    }

    std::vector<int8_t> w2Int4(shape.numExperts * shape.interSize * shape.hiddenSize);
    for (auto &v : w2Int4) {
        v = static_cast<int8_t>(rng.RandInt4());
    }
    std::vector<int32_t> w2Weight(shape.numExperts * shape.interSize * (shape.hiddenSize / PACK_RATIO));
    for (int64_t idx = 0; idx < static_cast<int64_t>(w2Weight.size()); ++idx) {
        w2Weight[idx] = PackInt4x8(&w2Int4[idx * PACK_RATIO]);
    }
    std::vector<float> w2ScalesF(shape.numExperts * (shape.interSize / GROUP_SIZE) * shape.hiddenSize);
    std::vector<uint16_t> w2ScalesBits(w2ScalesF.size());
    for (size_t i = 0; i < w2ScalesF.size(); ++i) {
        float v = rng.Uniform(0.005f, 0.03f);
        w2ScalesBits[i] = isBf16 ? FloatToBf16Bits(v) : FloatToFp16Bits(v);
        w2ScalesF[i] = isBf16 ? Bf16BitsToFloat(w2ScalesBits[i]) : Fp16BitsToFloat(w2ScalesBits[i]);
    }

    std::vector<int32_t> expertIds(totalTokens);
    for (int64_t t = 0; t < totalTokens; ++t) {
        expertIds[t] = static_cast<int32_t>(t % shape.numExperts); // 覆盖全部 expert
    }
    std::vector<float> topkWeights(totalTokens);
    for (auto &v : topkWeights) {
        v = rng.Uniform(0.2f, 1.0f);
    }

    // 2. Host 参考计算
    std::vector<float> yRef =
        RefMoe(shape, xF, w13Weight, w13ScalesF, w2Weight, w2ScalesF, expertIds, topkWeights, isBf16);

    // 3. 构造 device tensor
    aclTensor *x = nullptr;
    void *xDevice = nullptr;
    std::vector<int64_t> xShape = {shape.batchSize, shape.hiddenSize};
    int ret = CreateAclTensor(xBits, xShape, &xDevice, tDtype, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor *w13WeightT = nullptr;
    void *w13WeightDevice = nullptr;
    std::vector<int64_t> w13WeightShape = {shape.numExperts, shape.hiddenSize, 2 * shape.interSize / PACK_RATIO};
    ret = CreateAclTensor(w13Weight, w13WeightShape, &w13WeightDevice, aclDataType::ACL_INT32, &w13WeightT);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor *w13ScalesT = nullptr;
    void *w13ScalesDevice = nullptr;
    std::vector<int64_t> w13ScalesShape = {shape.numExperts, shape.hiddenSize / GROUP_SIZE, 2 * shape.interSize};
    ret = CreateAclTensor(w13ScalesBits, w13ScalesShape, &w13ScalesDevice, tDtype, &w13ScalesT);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor *w2WeightT = nullptr;
    void *w2WeightDevice = nullptr;
    std::vector<int64_t> w2WeightShape = {shape.numExperts, shape.interSize, shape.hiddenSize / PACK_RATIO};
    ret = CreateAclTensor(w2Weight, w2WeightShape, &w2WeightDevice, aclDataType::ACL_INT32, &w2WeightT);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor *w2ScalesT = nullptr;
    void *w2ScalesDevice = nullptr;
    std::vector<int64_t> w2ScalesShape = {shape.numExperts, shape.interSize / GROUP_SIZE, shape.hiddenSize};
    ret = CreateAclTensor(w2ScalesBits, w2ScalesShape, &w2ScalesDevice, tDtype, &w2ScalesT);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor *expertIdsT = nullptr;
    void *expertIdsDevice = nullptr;
    std::vector<int64_t> expertIdsShape = {shape.batchSize, shape.topK};
    ret = CreateAclTensor(expertIds, expertIdsShape, &expertIdsDevice, aclDataType::ACL_INT32, &expertIdsT);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor *topkWeightsT = nullptr;
    void *topkWeightsDevice = nullptr;
    std::vector<int64_t> topkWeightsShape = {shape.batchSize, shape.topK};
    ret = CreateAclTensor(topkWeights, topkWeightsShape, &topkWeightsDevice, aclDataType::ACL_FLOAT, &topkWeightsT);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor *y = nullptr;
    void *yDevice = nullptr;
    std::vector<int64_t> yShape = {shape.batchSize, shape.hiddenSize};
    std::vector<uint16_t> yBits(shape.batchSize * shape.hiddenSize, 0);
    ret = CreateAclTensor(yBits, yShape, &yDevice, tDtype, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 4. 调用 aclnn 两段式接口
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnFusedMoeSmallBsW4a16GetWorkspaceSize(x, w13WeightT, w13ScalesT, w2WeightT, w2ScalesT, expertIdsT,
                                                    topkWeightsT, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[%s] aclnnFusedMoeSmallBsW4a16GetWorkspaceSize failed. ERROR: %d\n", tag, ret);
              return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] allocate workspace failed. ERROR: %d\n", tag, ret); return ret);
    }

    ret = aclnnFusedMoeSmallBsW4a16(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] aclnnFusedMoeSmallBsW4a16 failed. ERROR: %d\n", tag, ret);
              return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] aclrtSynchronizeStream failed. ERROR: %d\n", tag, ret); return ret);

    // 5. 拷贝结果并对比
    auto outSize = yBits.size() * sizeof(uint16_t);
    ret = aclrtMemcpy(yBits.data(), outSize, yDevice, outSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] copy result from device to host failed. ERROR: %d\n", tag, ret);
              return ret);
    std::vector<float> yOut = BitsToFloat(yBits, isBf16);

    double maxAbsErr = 0.0;
    double maxRelErr = 0.0;
    int64_t failCnt = 0;
    for (size_t i = 0; i < yOut.size(); ++i) {
        double absErr = std::fabs(static_cast<double>(yOut[i]) - yRef[i]);
        double relErr = absErr / (std::fabs(static_cast<double>(yRef[i])) + 1e-6);
        maxAbsErr = std::max(maxAbsErr, absErr);
        maxRelErr = std::max(maxRelErr, relErr);
        if (absErr > ATOL + RTOL * std::fabs(static_cast<double>(yRef[i]))) {
            if (failCnt < 5) {
                LOG_PRINT("[%s] mismatch at [%zu]: device=%f ref=%f\n", tag, i, yOut[i], yRef[i]);
            }
            failCnt++;
        }
    }
    LOG_PRINT("[%s] maxAbsErr=%.6f maxRelErr=%.6f mismatchCount=%ld/%zu\n", tag, maxAbsErr, maxRelErr, failCnt,
              yOut.size());

    // 6. 释放资源
    aclDestroyTensor(x);
    aclDestroyTensor(w13WeightT);
    aclDestroyTensor(w13ScalesT);
    aclDestroyTensor(w2WeightT);
    aclDestroyTensor(w2ScalesT);
    aclDestroyTensor(expertIdsT);
    aclDestroyTensor(topkWeightsT);
    aclDestroyTensor(y);
    aclrtFree(xDevice);
    aclrtFree(w13WeightDevice);
    aclrtFree(w13ScalesDevice);
    aclrtFree(w2WeightDevice);
    aclrtFree(w2ScalesDevice);
    aclrtFree(expertIdsDevice);
    aclrtFree(topkWeightsDevice);
    aclrtFree(yDevice);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }

    if (failCnt > 0) {
        LOG_PRINT("[%s] TEST FAILED\n", tag);
        return 1;
    }
    LOG_PRINT("[%s] TEST PASSED\n", tag);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // {batchSize, topK, numExperts, hiddenSize, interSize}
    // case1: 基础小 shape，K 方向单 block
    TestShape shape1 = {2, 2, 4, 128, 128};
    // case2: 更大 batch/expert，hiddenSize=256 覆盖 K 方向多 block 原子加
    TestShape shape2 = {4, 2, 8, 256, 128};
    // case3: interSize=256，W2 的 K 方向多 block；单 token 单 expert
    TestShape shape3 = {1, 1, 4, 128, 256};

    int fails = 0;
    fails += RunTestCase(false, shape1, 20260716ULL, stream);
    fails += RunTestCase(true, shape1, 20260717ULL, stream);
    fails += RunTestCase(false, shape2, 20260718ULL, stream);
    fails += RunTestCase(true, shape2, 20260719ULL, stream);
    fails += RunTestCase(false, shape3, 20260720ULL, stream);
    fails += RunTestCase(true, shape3, 20260721ULL, stream);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (fails == 0) {
        LOG_PRINT("fused_moe_small_bs_w4a16 all test cases PASSED\n");
        return 0;
    }
    LOG_PRINT("fused_moe_small_bs_w4a16 test FAILED, failed cases: %d\n", fails);
    return 1;
}
