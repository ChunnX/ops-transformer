/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_cloud_causal_conv1d_fn.cpp
 * \brief W=4 correctness smoke test for CloudCausalConv1dFn.
 */

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cloud_causal_conv1d_fn.h"

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

float Fp16ToFloat(uint16_t raw)
{
    uint32_t sign = (raw >> 15) & 0x1;
    uint32_t exp = (raw >> 10) & 0x1F;
    uint32_t mant = raw & 0x3FF;
    if (exp == 0) {
        return (sign ? -1.0f : 1.0f) * (static_cast<float>(mant) / 1024.0f) * (1.0f / 16384.0f);
    }
    if (exp == 31) {
        return (mant == 0) ? (sign ? -HUGE_VALF : HUGE_VALF) : NAN;
    }
    return (sign ? -1.0f : 1.0f) * (1.0f + static_cast<float>(mant) / 1024.0f) *
           ldexpf(1.0f, static_cast<int>(exp) - 15);
}

uint16_t FloatToFp16(float value)
{
    if (value == 0.0f) {
        return 0;
    }
    uint16_t sign = value < 0.0f ? 0x8000 : 0;
    float absValue = value < 0.0f ? -value : value;
    int exp = 0;
    float mant = std::frexp(absValue, &exp);
    exp -= 1;
    int fp16Exp = exp + 15;
    if (fp16Exp <= 0) {
        return sign;
    }
    if (fp16Exp >= 31) {
        return sign | 0x7C00;
    }
    float normalized = mant * 2.0f - 1.0f;
    uint16_t fp16Mant = static_cast<uint16_t>(std::round(normalized * 1024.0f));
    if (fp16Mant == 1024) {
        fp16Mant = 0;
        fp16Exp += 1;
    }
    return sign | (static_cast<uint16_t>(fp16Exp) << 10) | (fp16Mant & 0x03FF);
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor, aclFormat format)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), static_cast<int64_t>(shape.size()), dataType, strides.data(), 0, format,
                              shape.data(), static_cast<int64_t>(shape.size()), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return -1);
    return 0;
}

void FreeTensorAndBuffer(aclTensor *tensor, void *deviceAddr)
{
    if (tensor != nullptr) {
        aclDestroyTensor(tensor);
    }
    if (deviceAddr != nullptr) {
        aclrtFree(deviceAddr);
    }
}

bool CheckValue(const char *name, int64_t index, uint16_t actualRaw, float expected)
{
    float actual = Fp16ToFloat(actualRaw);
    float diff = std::fabs(actual - expected);
    if (diff > 1e-2f) {
        LOG_PRINT("%s[%lld] mismatch: actual=%f expected=%f diff=%f raw=0x%04x\n", name, static_cast<long long>(index),
                  static_cast<double>(actual), static_cast<double>(expected), static_cast<double>(diff), actualRaw);
        return false;
    }
    return true;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    constexpr int64_t K = 4;
    constexpr int64_t batch = 1;
    constexpr int64_t seqLen = 5;
    constexpr int64_t dim = 128;
    constexpr int64_t stateLen = K - 1;

    std::vector<int64_t> xShape = {seqLen, dim};
    std::vector<int64_t> weightShape = {K, dim};
    std::vector<int64_t> convStatesShape = {batch, stateLen, dim};
    std::vector<int64_t> queryStartLocShape = {batch + 1};
    std::vector<int64_t> yShape = {seqLen, dim};

    std::vector<float> stateRows = {10.0f, 20.0f, 30.0f};
    std::vector<float> xRows = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> weightRows = {1.0f, 2.0f, 4.0f, 8.0f};

    std::vector<uint16_t> hostX(batch * seqLen * dim);
    for (int64_t s = 0; s < seqLen; ++s) {
        for (int64_t d = 0; d < dim; ++d) {
            hostX[s * dim + d] = FloatToFp16(xRows[s]);
        }
    }

    std::vector<uint16_t> hostWeight(K * dim);
    for (int64_t k = 0; k < K; ++k) {
        for (int64_t d = 0; d < dim; ++d) {
            hostWeight[k * dim + d] = FloatToFp16(weightRows[k]);
        }
    }

    std::vector<uint16_t> hostConvStates(batch * stateLen * dim);
    for (int64_t h = 0; h < stateLen; ++h) {
        for (int64_t d = 0; d < dim; ++d) {
            hostConvStates[h * dim + d] = FloatToFp16(stateRows[h]);
        }
    }

    std::vector<int32_t> hostQueryStartLoc = {0, static_cast<int32_t>(seqLen)};
    std::vector<uint16_t> hostY(seqLen * dim, 0);

    void *xDev = nullptr;
    void *weightDev = nullptr;
    void *convStatesDev = nullptr;
    void *queryStartLocDev = nullptr;
    void *yDev = nullptr;
    aclTensor *xTensor = nullptr;
    aclTensor *weightTensor = nullptr;
    aclTensor *convStatesTensor = nullptr;
    aclTensor *queryStartLocTensor = nullptr;
    aclTensor *yTensor = nullptr;

    ret = CreateAclTensor(hostX, xShape, &xDev, ACL_FLOAT16, &xTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostWeight, weightShape, &weightDev, ACL_FLOAT16, &weightTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret =
        CreateAclTensor(hostConvStates, convStatesShape, &convStatesDev, ACL_FLOAT16, &convStatesTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostQueryStartLoc, queryStartLocShape, &queryStartLocDev, ACL_INT32, &queryStartLocTensor,
                          ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostY, yShape, &yDev, ACL_FLOAT16, &yTensor, ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int64_t activationMode = 0;
    int64_t padSlotId = -1;

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnCloudCausalConv1dFnGetWorkspaceSize(
        xTensor, weightTensor,
        nullptr, // bias
        convStatesTensor,
        queryStartLocTensor, // 2D decode path: one batch with rows [0, seqLen)
        nullptr, // cache_indices: slot defaults to batch index
        nullptr, // num_accepted_tokens: offset = stateLen - (K - 1) = 0
        activationMode, padSlotId, yTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCloudCausalConv1dFnGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnCloudCausalConv1dFn(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCloudCausalConv1dFn failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    std::vector<uint16_t> yResult(hostY.size(), 0);
    ret = aclrtMemcpy(yResult.data(), yResult.size() * sizeof(uint16_t), yDev, yResult.size() * sizeof(uint16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y from device to host failed. ERROR: %d\n", ret); return ret);

    std::vector<uint16_t> stateResult(hostConvStates.size(), 0);
    ret = aclrtMemcpy(stateResult.data(), stateResult.size() * sizeof(uint16_t), convStatesDev,
                      stateResult.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy convStates from device to host failed. ERROR: %d\n", ret);
              return ret);

    std::vector<float> expectY(seqLen, 0.0f);
    expectY[0] = 10.0f * 1.0f + 20.0f * 2.0f + 30.0f * 4.0f + 1.0f * 8.0f;
    expectY[1] = 20.0f * 1.0f + 30.0f * 2.0f + 1.0f * 4.0f + 2.0f * 8.0f;
    expectY[2] = 30.0f * 1.0f + 1.0f * 2.0f + 2.0f * 4.0f + 3.0f * 8.0f;
    expectY[3] = 1.0f * 1.0f + 2.0f * 2.0f + 3.0f * 4.0f + 4.0f * 8.0f;
    expectY[4] = 2.0f * 1.0f + 3.0f * 2.0f + 4.0f * 4.0f + 5.0f * 8.0f;

    bool ok = true;
    for (int64_t s = 0; s < seqLen; ++s) {
        for (int64_t d = 0; d < dim; ++d) {
            ok = CheckValue("y", s * dim + d, yResult[s * dim + d], expectY[s]) && ok;
        }
    }

    std::vector<float> expectState = {3.0f, 4.0f, 5.0f};
    for (int64_t h = 0; h < stateLen; ++h) {
        for (int64_t d = 0; d < dim; ++d) {
            ok = CheckValue("convStates", h * dim + d, stateResult[h * dim + d], expectState[h]) && ok;
        }
    }

    LOG_PRINT("Expected W4 y rows: [%f, %f, %f, %f, %f]\n", static_cast<double>(expectY[0]),
              static_cast<double>(expectY[1]), static_cast<double>(expectY[2]), static_cast<double>(expectY[3]),
              static_cast<double>(expectY[4]));
    LOG_PRINT("Expected updated state rows: [%f, %f, %f]\n", static_cast<double>(expectState[0]),
              static_cast<double>(expectState[1]), static_cast<double>(expectState[2]));

    FreeTensorAndBuffer(xTensor, xDev);
    FreeTensorAndBuffer(weightTensor, weightDev);
    FreeTensorAndBuffer(convStatesTensor, convStatesDev);
    FreeTensorAndBuffer(queryStartLocTensor, queryStartLocDev);
    FreeTensorAndBuffer(yTensor, yDev);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    CHECK_RET(ok, LOG_PRINT("CloudCausalConv1dFn W=4 correctness test failed.\n"); return 1);
    LOG_PRINT("CloudCausalConv1dFn W=4 correctness test passed.\n");
    return 0;
}
