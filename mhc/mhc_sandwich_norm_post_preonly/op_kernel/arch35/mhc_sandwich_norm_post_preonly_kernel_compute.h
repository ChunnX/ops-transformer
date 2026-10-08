/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ========================================================================
// mhc_sandwich_norm_post_preonly_kernel_compute.h
// 共享计算模块：数据加载、RMSNorm、MHC Post、门控
// ========================================================================

// 数据加载

/// 加载一个token的h_post[N]和h_res[N*N]。
__aicore__ inline void LoadTokenWeights(uint32_t token, float (&postVals)[N_MAX], float (&combVals)[N_MAX * N_MAX])
{
    // Host已固定校验N为4，使用编译期常量便于展开标量循环。
    constexpr uint32_t N = N_MAX;
    LocalTensor<float> postCombUb = bufPostComb_.Get<float>();
    LocalTensor<float> hPostUb = postCombUb;
    LocalTensor<float> hResUb = postCombUb[ALIGN8]; // h_post[N] aligned to 8 elements
    {
        DataCopyExtParams pp = {1, 0, 0, 0, 0};
        pp.blockLen = N * sizeof(float);
        DataCopyPadExtParams<float> pad = {true, 0, 0, 0};
        DataCopyPad(hPostUb, hPostGm_[(uint64_t)token * N], pp, pad);

        DataCopyExtParams cp = {1, 0, 0, 0, 0};
        cp.blockLen = N * N * sizeof(float);
        DataCopyPadExtParams<float> cpad = {true, 0, 0, 0};
        DataCopyPad(hResUb, hResGm_[(uint64_t)token * N * N], cp, cpad);
    }
    PipeBarrier<PIPE_ALL>();
    for (uint32_t i = 0; i < N; ++i)
        postVals[i] = hPostUb.GetValue(i);
    for (uint32_t i = 0; i < N * N; ++i)
        combVals[i] = hResUb.GetValue(i);
}

/// 将h_out[token, tOff:tOff+tD]从低精度加载并转换为FP32。
__aicore__ inline void LoadHOutAndCast(uint64_t gmOffset, uint32_t tD)
{
    LocalTensor<T> hBf16 = bufHOutBf16_.AllocTensor<T>();
    DataCopy(hBf16, hOutGm_[gmOffset], tD);
    bufHOutBf16_.EnQue(hBf16);
    hBf16 = bufHOutBf16_.DeQue<T>();
    Cast(bufHOutFp32_.Get<float>(), hBf16, RoundMode::CAST_NONE, tD);
    PipeBarrier<PIPE_V>();
    bufHOutBf16_.FreeTensor(hBf16);
}

/// 将residual[token, 0:N, tOff:tOff+tD]从低精度加载并转换为FP32。
__aicore__ inline void LoadResidualAndCast(uint64_t gmBaseOffset, uint32_t tD)
{
    // Host已固定校验N为4，使用编译期常量便于展开标量循环。
    constexpr uint32_t N = N_MAX;
    LocalTensor<T> resBf16 = bufResidualBf16_.AllocTensor<T>();
    for (uint32_t n = 0; n < N; ++n)
        DataCopy(resBf16[n * dimTile_], residualGm_[gmBaseOffset + (uint64_t)n * D_], tD);
    bufResidualBf16_.EnQue(resBf16);
    resBf16 = bufResidualBf16_.DeQue<T>();
    LocalTensor<float> resFp32 = bufResFp32_.Get<float>();
    for (uint32_t n = 0; n < N; ++n)
        Cast(resFp32[n * dimTile_], resBf16[n * dimTile_], RoundMode::CAST_NONE, tD);
    PipeBarrier<PIPE_V>();
    bufResidualBf16_.FreeTensor(resBf16);
}

/// 将gamma的[tOff, tOff+tD)分块加载到指定缓存。
__aicore__ inline void LoadGammaInto(LocalTensor<float> dst, GlobalTensor<float> &src, uint32_t tOff, uint32_t tD)
{
    DataCopyExtParams gP = {1, 0, 0, 0, 0};
    gP.blockLen = tD * sizeof(float);
    DataCopyPadExtParams<float> gPad = {false, 0, 0, 0};
    DataCopyPad(dst, src[tOff], gP, gPad);
}

// RMSNorm基础计算

/// 使用二叉树累加模式执行高精度ReduceSum。
__aicore__ inline float ReduceSumPrecise(LocalTensor<float> src, uint32_t tD)
{
    LocalTensor<float> scBuf = bufScalar_.Get<float>();
    constexpr uint32_t REDUCE_ALIGN = 16; // ReduceSum要求16个元素对齐
    uint32_t tDAligned = (tD + REDUCE_ALIGN - 1) / REDUCE_ALIGN * REDUCE_ALIGN;
    uint32_t srcShape[2] = {1, tDAligned};
    // 输入在归约后不再使用，允许归约过程复用输入缓冲，减少内部UB与标量调度开销。
    ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(scBuf, src, srcShape, true);
    PipeBarrier<PIPE_V>();
    return scBuf.GetValue(0);
}

/// 使用ReduceSumPrecise计算平方和。
__aicore__ inline float ReduceSumSq(LocalTensor<float> src, uint32_t tD)
{
    LocalTensor<float> sqBuf = bufSqFp32_.Get<float>();
    Mul(sqBuf, src, src, tD);
    PipeBarrier<PIPE_V>();
    return ReduceSumPrecise(sqBuf, tD);
}

/// 使用标量精度计算1/sqrt(sumSq * invCount + eps)。
__aicore__ inline __attribute__((always_inline)) float ComputeInvRms(float sumSq, float invCount)
{
    return 1.0f / sqrt(sumSq * invCount + normEps_);
}

/// 原地依次执行标量缩放和gamma缩放。
__aicore__ inline void ApplyScaleAndGamma(LocalTensor<float> buf, float scale, LocalTensor<float> gamma, uint32_t tD)
{
    Muls(buf, buf, scale, tD);
    PipeBarrier<PIPE_V>();
    Mul(buf, buf, gamma, tD);
    PipeBarrier<PIPE_V>();
}

// MHC_Post计算

/// 单核批量计算全部N个head的x2，并写入bufX2Fp32_。
__aicore__ inline void ComputeX2AllHeadsBatched(const float (&postVals)[N_MAX], const float (&combVals)[N_MAX * N_MAX],
                                                uint32_t tD)
{
    // Host已固定校验N为4，使用编译期常量便于展开标量循环。
    constexpr uint32_t N = N_MAX;
    LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
    LocalTensor<float> hOut = bufHOutFp32_.Get<float>();
    LocalTensor<float> res = bufResFp32_.Get<float>();

    for (uint32_t h = 0; h < N; ++h)
        Muls(x2[h * dimTile_], hOut, postVals[h], tD);
    PipeBarrier<PIPE_V>();
    for (uint32_t j = 0; j < N; ++j) {
        for (uint32_t h = 0; h < N; ++h)
            Axpy(x2[h * dimTile_], res[j * dimTile_], combVals[j * N + h], tD);
        PipeBarrier<PIPE_V>();
    }
}

/// 单核多分块路径按head串行计算全部x2。
__aicore__ inline void ComputeX2AllHeadsSerial(const float (&postVals)[N_MAX], const float (&combVals)[N_MAX * N_MAX],
                                               uint32_t tD)
{
    // Host已固定校验N为4，使用编译期常量便于展开标量循环。
    constexpr uint32_t N = N_MAX;
    LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
    LocalTensor<float> hOut = bufHOutFp32_.Get<float>();
    LocalTensor<float> res = bufResFp32_.Get<float>();

    for (uint32_t h = 0; h < N; ++h) {
        Muls(x2[h * dimTile_], hOut, postVals[h], tD);
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < N; ++j) {
            Axpy(x2[h * dimTile_], res[j * dimTile_], combVals[j * N + h], tD);
            PipeBarrier<PIPE_V>();
        }
    }
}

// 门控计算

/// 由xHatVals计算gate[0..N-1]，其中包含invRmsPre缩放：
///   gate[i] = sigmoid(alpha0 * xHat[i] + bias[i]) + hcEps
__aicore__ inline void ComputeGateValues(float totalSumSqPre, float (&xHatVals)[N_MAX], float (&gate)[N_MAX])
{
    // Host已固定校验N为4，使用编译期常量便于展开标量循环。
    constexpr uint32_t N = N_MAX;
    float invRmsPre = 1.0f / sqrt(totalSumSqPre * invK_ + mhcPreNormEps_);
    for (uint32_t o = 0; o < N; ++o)
        xHatVals[o] *= invRmsPre;

    LocalTensor<float> scBuf = bufScalar_.Get<float>();
    for (uint32_t i = 0; i < N; ++i)
        scBuf.SetValue(i, -(alpha0_ * xHatVals[i] + biasCache_[i]));
    PipeBarrier<PIPE_V>();
    // 三个缓冲仍按32B错开，只计算真实存在的四个head。
    Exp(scBuf, scBuf, N);
    PipeBarrier<PIPE_V>();
    Adds(scBuf, scBuf, 1.0f, N); // 1 + exp(-z)
    PipeBarrier<PIPE_V>();
    LocalTensor<float> onesBuf = scBuf[EXP_DIV_ELEM * 2];
    Duplicate(onesBuf, 1.0f, N);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> gateBuf = scBuf[EXP_DIV_ELEM];
    Div(gateBuf, onesBuf, scBuf, N); // 1 / (1+exp(-z))
    PipeBarrier<PIPE_V>();
    Adds(gateBuf, gateBuf, hcEps_, N); // + hcEps
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < N; ++i)
        gate[i] = gateBuf.GetValue(i);
}
