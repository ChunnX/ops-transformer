/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "acl/acl.h"
#include "aclnnop/aclnn_aclnn_fused_qkv_projection.h"

using DataType = float;

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr, const char* name)
{
    auto size = GetShapeSize(shape);
    std::vector<DataType> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n%s", ret, aclGetRecentErrMsg()); return);
    int64_t printCount = size < 10 ? size : 10;
    for (int64_t i = 0; i < printCount; i++) {
        LOG_PRINT("%s[%ld] is: %f\n", name, i, resultData[i]);
    }
    if (size > 10) LOG_PRINT("%s... (%ld total) all good\n", name, size);
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n %s", ret, aclGetRecentErrMsg()); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n %s", ret, aclGetRecentErrMsg()); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n %s", ret, aclGetRecentErrMsg()); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n%s", ret, aclGetRecentErrMsg()); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n%s", ret, aclGetRecentErrMsg()); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int RunTest(int64_t batchSize, int64_t seqLen, int64_t hiddenSize,
            int64_t qDim, int64_t kDim, int64_t vDim,
            bool useBias, int testIdx, aclrtStream stream)
{
    int64_t fusedDim = qDim + kDim + vDim;
    int64_t m = batchSize * seqLen;
    int ret;

    std::vector<int64_t> hsShape = {batchSize, seqLen, hiddenSize};
    std::vector<int64_t> wtShape = {hiddenSize, fusedDim};
    std::vector<int64_t> biasShape = {fusedDim};
    std::vector<int64_t> qShape = {batchSize, seqLen, qDim};
    std::vector<int64_t> kShape = {batchSize, seqLen, kDim};
    std::vector<int64_t> vShape = {batchSize, seqLen, vDim};

    // 用多样化的值填充，并在 CPU 上精确计算期望输出
    std::vector<DataType> hsData(GetShapeSize(hsShape));
    std::vector<DataType> wtData(GetShapeSize(wtShape));
    std::vector<DataType> biasData(GetShapeSize(biasShape));

    // X[b,s,h]: 用伪随机填充，值范围 [-1.5, 1.5]
    for (int64_t i = 0; i < (int64_t)hsData.size(); i++)
        hsData[i] = std::sin(static_cast<float>(i) * 0.73f) * 1.5f;
    // W[h,fd]: 用伪随机填充
    for (int64_t i = 0; i < (int64_t)wtData.size(); i++)
        wtData[i] = std::cos(static_cast<float>(i) * 1.19f);
    // bias[fd]: 用伪随机填充
    for (int64_t i = 0; i < (int64_t)biasData.size(); i++)
        biasData[i] = std::sin(static_cast<float>(i) * 2.37f) * 0.5f;

    // CPU 计算期望输出：X @ W + bias
    std::vector<DataType> expQ(GetShapeSize(qShape));
    std::vector<DataType> expK(GetShapeSize(kShape));
    std::vector<DataType> expV(GetShapeSize(vShape));
    for (int64_t r = 0; r < m; r++) {
        for (int64_t fd = 0; fd < fusedDim; fd++) {
            float sum = 0.0f;
            for (int64_t hh = 0; hh < hiddenSize; hh++)
                sum += hsData[r * hiddenSize + hh] * wtData[hh * fusedDim + fd];
            if (useBias) sum += biasData[fd];
            if (fd < qDim)           expQ[r * qDim + fd] = sum;
            else if (fd < qDim + kDim) expK[r * kDim + (fd - qDim)] = sum;
            else                      expV[r * vDim + (fd - qDim - kDim)] = sum;
        }
    }

    // 打印输入数据前几个值
    int printN = 5;
    LOG_PRINT("[Test %d] Input X[0..%d]: ", testIdx, std::min(printN, (int)hsData.size())-1);
    for (int i = 0; i < printN && i < (int)hsData.size(); i++) LOG_PRINT("%.4f ", hsData[i]);
    LOG_PRINT("\n  W[0..%d]: ", std::min(printN, (int)wtData.size())-1);
    for (int i = 0; i < printN && i < (int)wtData.size(); i++) LOG_PRINT("%.4f ", wtData[i]);
    if (useBias) {
        LOG_PRINT("\n  Bias[0..%d]: ", std::min(printN, (int)biasData.size())-1);
        for (int i = 0; i < printN && i < (int)biasData.size(); i++) LOG_PRINT("%.4f ", biasData[i]);
    }
    LOG_PRINT("\n");

    aclTensor* hs = nullptr; void* hsDev = nullptr;
    ret = CreateAclTensor(hsData, hsShape, &hsDev, aclDataType::ACL_FLOAT, &hs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* wt = nullptr; void* wtDev = nullptr;
    ret = CreateAclTensor(wtData, wtShape, &wtDev, aclDataType::ACL_FLOAT, &wt);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* bias = nullptr; void* biasDev = nullptr;
    if (useBias) {
        ret = CreateAclTensor(biasData, biasShape, &biasDev, aclDataType::ACL_FLOAT, &bias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    aclTensor *q = nullptr, *k = nullptr, *v = nullptr;
    void *qDev = nullptr, *kDev = nullptr, *vDev = nullptr;
    int64_t qs = GetShapeSize(qShape), ks = GetShapeSize(kShape), vs = GetShapeSize(vShape);
    int64_t maxOutSize = qs > ks ? (qs > vs ? qs : vs) : (ks > vs ? ks : vs);
    std::vector<DataType> zeros(maxOutSize, 0);
    ret = CreateAclTensor(zeros, qShape, &qDev, aclDataType::ACL_FLOAT, &q); CHECK_RET(ret==ACL_SUCCESS, return ret);
    ret = CreateAclTensor(zeros, kShape, &kDev, aclDataType::ACL_FLOAT, &k); CHECK_RET(ret==ACL_SUCCESS, return ret);
    ret = CreateAclTensor(zeros, vShape, &vDev, aclDataType::ACL_FLOAT, &v); CHECK_RET(ret==ACL_SUCCESS, return ret);

    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    ret = aclnnAclnnFusedQkvProjectionGetWorkspaceSize(
        hs, wt, bias, qDim, kDim, vDim, q, k, v, &wsSize, &exec);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[Test %d] GetWorkspaceSize failed. %d\n", testIdx, ret); return ret);

    void* ws = nullptr;
    if (wsSize > 0) { aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST); }

    ret = aclnnAclnnFusedQkvProjection(ws, wsSize, exec, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[Test %d] Execute failed. %d\n", testIdx, ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[Test %d] Sync failed. %d\n%s", testIdx, ret, aclGetRecentErrMsg()); return ret);

    // 验证 Q/K/V 全部元素（容忍浮点误差）
    std::vector<DataType> qD(GetShapeSize(qShape)), kD(GetShapeSize(kShape)), vD(GetShapeSize(vShape));
    aclrtMemcpy(qD.data(), qD.size()*sizeof(DataType), qDev, qD.size()*sizeof(DataType), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(kD.data(), kD.size()*sizeof(DataType), kDev, kD.size()*sizeof(DataType), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(vD.data(), vD.size()*sizeof(DataType), vDev, vD.size()*sizeof(DataType), ACL_MEMCPY_DEVICE_TO_HOST);

    const float eps = 1e-4f;
    for (int64_t i = 0; i < (int64_t)qD.size(); i++) {
        if (std::fabs(qD[i] - expQ[i]) > eps) {
            LOG_PRINT("[Test %d] Q[%ld] mismatch: exp %.6f got %.6f\n", testIdx, i, expQ[i], qD[i]);
            return -1;
        }
    }
    for (int64_t i = 0; i < (int64_t)kD.size(); i++) {
        if (std::fabs(kD[i] - expK[i]) > eps) {
            LOG_PRINT("[Test %d] K[%ld] mismatch: exp %.6f got %.6f\n", testIdx, i, expK[i], kD[i]);
            return -1;
        }
    }
    for (int64_t i = 0; i < (int64_t)vD.size(); i++) {
        if (std::fabs(vD[i] - expV[i]) > eps) {
            LOG_PRINT("[Test %d] V[%ld] mismatch: exp %.6f got %.6f\n", testIdx, i, expV[i], vD[i]);
            return -1;
        }
    }

    LOG_PRINT("[Test %d] PASSED (batch=%ld seq=%ld h=%ld q=%ld k=%ld v=%ld bias=%d)\n",
              testIdx, batchSize, seqLen, hiddenSize, qDim, kDim, vDim, useBias);

    // 打印前几个 Q/K/V 结果用于人工核对
    LOG_PRINT("  Q[0..%d]: ", std::min(printN, (int)qD.size())-1);
    for (int i = 0; i < printN && i < (int)qD.size(); i++) LOG_PRINT("%.4f ", qD[i]);
    LOG_PRINT("\n  K[0..%d]: ", std::min(printN, (int)kD.size())-1);
    for (int i = 0; i < printN && i < (int)kD.size(); i++) LOG_PRINT("%.4f ", kD[i]);
    LOG_PRINT("\n  V[0..%d]: ", std::min(printN, (int)vD.size())-1);
    for (int i = 0; i < printN && i < (int)vD.size(); i++) LOG_PRINT("%.4f ", vD[i]);
    LOG_PRINT("\n");

    aclDestroyTensor(hs); aclrtFree(hsDev);
    aclDestroyTensor(wt); aclrtFree(wtDev);
    if (bias) { aclDestroyTensor(bias); aclrtFree(biasDev); }
    aclDestroyTensor(q); aclDestroyTensor(k); aclDestroyTensor(v);
    aclrtFree(qDev); aclrtFree(kDev); aclrtFree(vDev);
    if (wsSize > 0) aclrtFree(ws);
    return 0;
}

int main()
{
    int32_t deviceId = 0; aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. %d\n", ret); return ret);

    // Note: M = batch * seqLen must be >= 16 for Cube minimum granularity
    //       q/k/v 维度建议为 8 的倍数 (32B 对齐)
    struct Case { int64_t b, s, h, q, k, v; bool bias; const char* desc; };
    Case tests[] = {
        // === 基础对齐样例 ===
        {2, 8,  16,  16, 8,  8,  true,  "M=16 最小Cube"},
        {1, 32, 16,  16, 8,  8,  true,  "M=32"},
        {1, 16, 64,  32, 16, 16, false, "M=16 无偏置"},
        {1, 64, 64,  32, 16, 16, true,  "M=64"},
        // === 中等复杂度 ===
        {4, 32, 64,  64, 32, 32, true,  "M=128 q≠k≠v"},
        {8, 16, 32,  16, 8,  8,  false, "M=128 大batch无偏置"},
        {1, 256,64,  32, 16, 16, true,  "M=256 长序列"},
        {1, 128,64,  32, 16, 16, false, "M=128 无偏置"},
        {2, 64, 32,  16, 8,  8,  true,  "M=128 中等batch"},
        // === 极端样例 ===
        {1, 16, 8,   8,  8,  8,  true,  "M=16 h=8 极小hidden"},
        {1, 1024,64, 32, 16, 16, true,  "M=1024 超长序列"},
        {2, 8,  256, 128,64, 64, false, "M=16 h=256 大hidden"},
        {4, 4,  128, 64, 32, 32, true,  "M=16 h=128 大hidden"},
        {1, 17, 32,  16, 8,  8,  true,  "M=17 非16倍数"},
        {1, 31, 32,  16, 8,  8,  false, "M=31 非16倍数"},
        // === q=k=v 等分 ===
        {2, 8,  48,  16, 16, 16, true,  "q=k=v=16"},
        {1, 32, 96,  32, 32, 32, false, "q=k=v=32"},
        {2, 16, 24,  8,  8,  8,  true,  "q=k=v=8"},
        // === 非对齐维度 (可能失败，用于诊断) ===
        {3, 7,  32,  17, 7,  8,  false, "q=17 非8B对齐"},
        {2, 8,  32,  9,  9,  14, true,  "q=9 k=9 非对齐"},
        {1, 16, 32,  10, 10, 12, false, "q=10 非对齐"},
        {2, 10, 32,  20, 6,  6,  true,  "k=6 非对齐"},
    };
    int numTests = sizeof(tests) / sizeof(tests[0]);
    int passed = 0, failed = 0;
    for (int i = 0; i < numTests; i++) {
        LOG_PRINT("\n[Test %d/%d] %s (b=%ld s=%ld h=%ld q=%ld k=%ld v=%ld bias=%d)\n",
                  i+1, numTests, tests[i].desc, tests[i].b, tests[i].s, tests[i].h,
                  tests[i].q, tests[i].k, tests[i].v, tests[i].bias);
        ret = RunTest(tests[i].b, tests[i].s, tests[i].h, tests[i].q, tests[i].k, tests[i].v,
                      tests[i].bias, i+1, stream);
        if (ret) { LOG_PRINT("=== Test %d FAILED ===\n", i+1); failed++; }
        else { passed++; }
    }
    LOG_PRINT("\n=== %d/%d tests PASSED, %d FAILED ===\n", passed, numTests, failed);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
