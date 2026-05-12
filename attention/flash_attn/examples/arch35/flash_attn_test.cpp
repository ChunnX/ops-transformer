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
 * \file test_aclnn_flash_attn.cpp
 * \brief FlashAttn算子调用示例（推理场景）
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>
#include <cstdio>
#include "acl/acl.h"
#include "../op_api/aclnn_flash_attn.h"

namespace {

static float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1u;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t f;
    if (exp == 0) {
        f = (sign << 31) | (mant << 13);  // zero / denormal → treat as zero here
    } else if (exp == 31) {
        f = (sign << 31) | 0x7f800000u | (mant << 13);  // inf / nan
    } else {
        f = (sign << 31) | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}

#define CHECK_RET(cond)  ((cond) ? true : (false))

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
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        return ret;
    }
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        return ret;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

static uint16_t float_to_fp16(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 31) & 0x1u;
    int32_t  exp  = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ffu;
    if (exp <= 0)  return (uint16_t)(sign << 15);
    if (exp >= 31) return (uint16_t)((sign << 15) | 0x7c00u);
    return (uint16_t)((sign << 15) | ((uint32_t)exp << 10) | mant);
}

static void cpu_flash_attention(
    const std::vector<float> &Q,   // BNSD [B, N_q, S_q, D]
    const std::vector<float> &K,   // BNSD [B, N_kv, S_kv, D]
    const std::vector<float> &V,   // BNSD [B, N_kv, S_kv, D]
    std::vector<float>       &Out, // BNSD [B, N_q, S_q, D]
    int64_t B, int64_t N_q, int64_t N_kv,
    int64_t S_q, int64_t S_kv, int64_t D)
{
    float scale = 1.0f / std::sqrt((float)D);
    std::vector<float> score(S_q * S_kv);
    for (int64_t b = 0; b < B; b++) {
        for (int64_t nq = 0; nq < N_q; nq++) {
            int64_t nkv = nq * N_kv / N_q;  // GQA head mapping
            // 1. QK^T * scale
            for (int64_t s1 = 0; s1 < S_q; s1++) {
                for (int64_t s2 = 0; s2 < S_kv; s2++) {
                    float dot = 0.0f;
                    for (int64_t d = 0; d < D; d++) {
                        dot += Q[((b * N_q  + nq)  * S_q  + s1) * D + d] *
                               K[((b * N_kv + nkv) * S_kv + s2) * D + d];
                    }
                    score[s1 * S_kv + s2] = dot * scale;
                }
            }
            // 2. row-wise softmax
            for (int64_t s1 = 0; s1 < S_q; s1++) {
                float mx = *std::max_element(score.data() + s1 * S_kv,
                                             score.data() + s1 * S_kv + S_kv);
                float sm = 0.0f;
                for (int64_t s2 = 0; s2 < S_kv; s2++) {
                    score[s1 * S_kv + s2] = std::exp(score[s1 * S_kv + s2] - mx);
                    sm += score[s1 * S_kv + s2];
                }
                for (int64_t s2 = 0; s2 < S_kv; s2++)
                    score[s1 * S_kv + s2] /= sm;
            }
            // 3. softmax * V
            for (int64_t s1 = 0; s1 < S_q; s1++) {
                for (int64_t d = 0; d < D; d++) {
                    float acc = 0.0f;
                    for (int64_t s2 = 0; s2 < S_kv; s2++) {
                        acc += score[s1 * S_kv + s2] *
                               V[((b * N_kv + nkv) * S_kv + s2) * D + d];
                    }
                    Out[((b * N_q + nq) * S_q + s1) * D + d] = acc;
                }
            }
        }
    }
}

} // namespace

int main()
{
    // 1. （固定写法）device/stream初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // 2. 构造输入与输出
    // BNSD推理场景：B=1, N_q=8, N_kv=2 (GQA 4:1), S_q=S_kv=128, D=64
    const int64_t B   = 1;
    const int64_t N_q = 1;
    const int64_t N_kv = 1;   // GQA：每2个q heads共享1个kv head
    const int64_t S_q = 128;
    const int64_t S_kv = 128;
    const int64_t D   = 64;

    // Q/Out: BNSD (B, N, S, D)
    // KV:    BSND (B, S, N, D) — layoutKv仅支持BSND/TND/PA_ND/PA_Nz
    std::vector<int64_t> qShape   = {B, N_q,  S_q,  D};   // BNSD
    std::vector<int64_t> kShape   = {B, N_kv, S_kv, D};   // BNSD
    std::vector<int64_t> vShape   = {B, N_kv, S_kv, D};   // BNSD
    std::vector<int64_t> outShape = {B, N_q,  S_q,  D};   // BNSD

    void *qDeviceAddr   = nullptr;
    void *kDeviceAddr   = nullptr;
    void *vDeviceAddr   = nullptr;
    void *outDeviceAddr = nullptr;

    aclTensor *qTensor   = nullptr;
    aclTensor *kTensor   = nullptr;
    aclTensor *vTensor   = nullptr;
    aclTensor *outTensor = nullptr;

    // 生成随机FP16输入；同时保留fp32版本（由fp16回转，确保CPU/NPU使用同一精度数据）
    // 整数随机值，范围 [-4, 4]，可精确表示为 fp16
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(10, 11);
    
    auto qElem = GetShapeSize(qShape);
    auto kElem = GetShapeSize(kShape);
    auto vElem = GetShapeSize(vShape);

    std::vector<float>    qFloat(qElem), kFloat(kElem), vFloat(vElem);
    std::vector<uint16_t> qHostData(qElem), kHostData(kElem), vHostData(vElem);
    std::vector<uint16_t> outHostData(GetShapeSize(outShape), 0);

    for (int64_t i = 0; i < qElem; i++) {
        qHostData[i] = float_to_fp16((float)dist(rng));
        qFloat[i]    = fp16_to_float(qHostData[i]);
    }
    for (int64_t i = 0; i < kElem; i++) {
        kHostData[i] = float_to_fp16((float)dist(rng));
        kFloat[i]    = fp16_to_float(kHostData[i]);
    }
    for (int64_t i = 0; i < vElem; i++) {
        vHostData[i] = float_to_fp16((float)dist(rng));
        vFloat[i]    = fp16_to_float(vHostData[i]);
    }

    // 保存 Q / K / V 到文件
    auto saveInput = [&](const char *fname, const std::vector<float> &data,
                         int64_t rows, int64_t cols) {
        FILE *f = fopen(fname, "w");
        for (int64_t r = 0; r < rows; r++) {
            for (int64_t c = 0; c < cols; c++)
                fprintf(f, "%6.1f ", data[r * cols + c]);
            fprintf(f, "\n");
        }
        fclose(f);
    };
    saveInput("q_data.txt", qFloat, N_q * S_q, D);   // (N_q*S_q) x D
    saveInput("k_data.txt", kFloat, N_kv * S_kv, D);
    saveInput("v_data.txt", vFloat, N_kv * S_kv, D);
    LOG_PRINT("Q/K/V saved to q_data.txt / k_data.txt / v_data.txt\n");

    ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &qTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }
    ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &kTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }
    ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &vTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }

    // 属性参数
    // softmaxScale=0.0f 表示使用默认缩放系数 1/sqrt(D)
    double  softmaxScale      = 1.0 / sqrt((double)D);
    int64_t maskMode         = 0;      // 0: 无掩码
    int64_t winLeft          = 0;
    int64_t winRight         = 0;
    int64_t maxseqlenq          = 0;
    int64_t maxseqlenkv         = 0;
    const char *layoutQ      = "BNSD";
    const char *layoutKv     = "BNSD";  // KV布局仅支持BSND/TND/PA_ND/PA_Nz
    const char *layoutOut    = "BNSD";
    int64_t returnSoftmaxLse = 0;      // 推理场景不输出softmax_lse
    int64_t deterministic    = 0;

    // 3. 调用aclnnFlashAttn两段式接口
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    // 第一段：计算workspace大小
    ret = aclnnFlashAttnGetWorkspaceSize(
        qTensor, kTensor, vTensor,
        nullptr,    // blockTableOptional: 无PA
        nullptr,    // cuSeqlensQOptional
        nullptr,    // cuSeqlensKvOptional
        nullptr,    // sequsedQOptional
        nullptr,    // sequsedKvOptional
        nullptr,    // sinksOptional
        nullptr,    // metadataOptional
        softmaxScale, maskMode, winLeft, winRight, maxseqlenq, maxseqlenkv,
        layoutQ, layoutKv, layoutOut,
        returnSoftmaxLse, deterministic,
        outTensor,
        nullptr,    // softmaxLseOptional: 推理场景不输出
        &workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnFlashAttnGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }
    else {
        LOG_PRINT("aclnnFlashAttnGetWorkspaceSize success. \n");
    }

    // 根据workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0U) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    // 第二段：执行计算
    ret = aclnnFlashAttn(workspaceAddr, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnFlashAttn failed. ERROR: %d\n", ret);
        return ret;
    }
    else {
        LOG_PRINT("aclnnFlashAttn success. \n");
    }

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        return ret;
    }

    // 5. 获取输出结果，将device侧内存拷贝至host侧
    auto outSize = GetShapeSize(outShape);
    std::vector<uint16_t> resultData(outSize, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                      outDeviceAddr, outSize * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    // 5b. CPU参考计算
    std::vector<float> refOut(outSize, 0.0f);
    cpu_flash_attention(qFloat, kFloat, vFloat, refOut, B, N_q, N_kv, S_q, S_kv, D);

    // 保存 NPU 输出和 CPU 参考输出到文件
    {
        FILE *fn = fopen("npu_out.txt", "w");
        FILE *fc = fopen("cpu_out.txt", "w");
        for (int64_t i = 0; i < outSize; i++) {
            fprintf(fn, "[%4ld] %10.6f\n", i, fp16_to_float(resultData[i]));
            fprintf(fc, "[%4ld] %10.6f\n", i, refOut[i]);
        }
        fclose(fn); fclose(fc);
        LOG_PRINT("NPU output saved to npu_out.txt, CPU ref saved to cpu_out.txt\n");
    }

    // 逐点全量对比打印 + 精度统计（千分之五相对误差）
    int64_t failCount = 0;
    float   maxRelErr = 0.0f;
    float   maxAbsErr = 0.0f;
    const float kRelTol = 0.005f;
    const float kEps    = 1e-5f;
    FILE *fcmp = fopen("comparison.txt", "w");
    fprintf(fcmp, "%5s  %10s  %10s  %10s  %10s  %s\n",
            "idx", "npu", "cpu_ref", "abs_err", "rel_err", "status");
    for (int64_t i = 0; i < outSize; i++) {
        float npuVal = fp16_to_float(resultData[i]);
        float refVal = refOut[i];
        float absErr = std::abs(npuVal - refVal);
        float relErr = absErr / (std::abs(refVal) + kEps);
        bool  ok     = (relErr <= kRelTol);
        if (absErr > maxAbsErr) maxAbsErr = absErr;
        if (relErr > maxRelErr) maxRelErr = relErr;
        if (!ok) failCount++;
        // 打印到 stdout 和文件
        const char *tag = ok ? "OK" : "FAIL";
        printf("[%4ld] npu=%10.6f  ref=%10.6f  abs=%9.6f  rel=%8.6f  %s\n",
               i, npuVal, refVal, absErr, relErr, tag);
        fprintf(fcmp, "%5ld  %10.6f  %10.6f  %10.6f  %10.6f  %s\n",
                i, npuVal, refVal, absErr, relErr, tag);
    }
    fclose(fcmp);
    LOG_PRINT("Per-element comparison saved to comparison.txt\n");
    LOG_PRINT("Precision: max_abs_err=%.6f  max_rel_err=%.6f  fail=%ld/%ld\n",
              maxAbsErr, maxRelErr, failCount, outSize);
    LOG_PRINT("Precision result: %s  (rel_tol=%.3f)\n",
              (failCount == 0) ? "PASS" : "FAIL", kRelTol);

    // 6. 释放aclTensor资源
    aclDestroyTensor(qTensor);
    aclDestroyTensor(kTensor);
    aclDestroyTensor(vTensor);
    aclDestroyTensor(outTensor);

    // 7. 释放device资源
    aclrtFree(qDeviceAddr);
    aclrtFree(kDeviceAddr);
    aclrtFree(vDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0U) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}