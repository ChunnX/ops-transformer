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
__aicore__ inline void LoadTokenWeights(uint32_t token, float (&postVals)[N_MAX])
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
}

/// 把一个h_out分块预取到输入队列；双槽允许下一分块MTE2与当前分块Vector重叠。
__aicore__ inline void PrefetchHOutTile(uint64_t gmOffset, uint32_t tD)
{
    LocalTensor<T> hBf16 = bufHOutBf16_.AllocTensor<T>();
    DataCopy(hBf16, hOutGm_[gmOffset], tD);
    bufHOutBf16_.EnQue(hBf16);
}

/// 按队列顺序取出已经预取的h_out分块并转换为FP32。
__aicore__ inline void ConsumeHOutTile(uint32_t tD)
{
    LocalTensor<T> hBf16 = bufHOutBf16_.DeQue<T>();
    LocalTensor<float> hOut = bufResFp32_.Get<float>()[N_MAX * dimTile_];
    MhcSandwichRegBase::CastB16ToFp32(hOut, hBf16, tD);
    PipeBarrier<PIPE_V>();
    bufHOutBf16_.FreeTensor(hBf16);
}

/// 非流水调用点仍使用成对的预取与消费。
__aicore__ inline void LoadHOutAndCast(uint64_t gmOffset, uint32_t tD)
{
    PrefetchHOutTile(gmOffset, tD);
    ConsumeHOutTile(tD);
}

/// 提前搬运完整h_out；后续权重加载的全流水等待同时保证该数据对Vector可见。
__aicore__ inline void CacheHOut(uint64_t gmOffset)
{
    LocalTensor<T> hOutCache = bufX2Cache_.Get<T>();
    DataCopy(hOutCache, hOutGm_[gmOffset], D_);
}

/// 每个token一次搬运完整gamma0，与h_out和权重共用初始全流水等待。
__aicore__ inline void CacheGamma0()
{
    LocalTensor<float> gamma0 = bufGamma0Fp32_.Get<float>();
    DataCopyExtParams params = {1, 0, 0, 0, 0};
    params.blockLen = D_ * sizeof(float);
    DataCopyPadExtParams<float> pad = {false, 0, 0, 0};
    DataCopyPad(gamma0, gamma0Gm_, params, pad);
}

/// R6归约完成后发起完整gamma1搬运，与invRms1标量计算重叠。
__aicore__ inline void StartCacheGamma1()
{
    if (D_ > GAMMA1_CACHE_MAX_D) {
        return;
    }
    LocalTensor<float> gamma1 = bufGamma2Fp32_.Get<float>();
    DataCopyExtParams params = {1, 0, 0, 0, 0};
    params.blockLen = D_ * sizeof(float);
    DataCopyPadExtParams<float> pad = {false, 0, 0, 0};
    DataCopyPad(gamma1, gamma1Gm_, params, pad);
    SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
}

/// R7读取完整gamma1前回收MTE2到Vector事件。
__aicore__ inline void WaitCacheGamma1()
{
    if (D_ > GAMMA1_CACHE_MAX_D) {
        return;
    }
    WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);
}

/// 第二遍R1/R2直接从x2缓存恢复h_out，避免再次从GM读取。
__aicore__ inline void LoadCachedHOut(uint32_t tOff, uint32_t tD)
{
    LocalTensor<T> hOutCache = bufX2Cache_.Get<T>()[tOff];
    LocalTensor<float> hOut = bufResFp32_.Get<float>()[N_MAX * dimTile_];
    MhcSandwichRegBase::CastB16ToFp32(hOut, hOutCache, tD);
    PipeBarrier<PIPE_V>();
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
    // gamma2缓存与平方归约不同时存活，复用同一块UB作为平方临时区。
    LocalTensor<float> sqBuf = bufGamma2Fp32_.Get<float>();
    MhcSandwichRegBase::Mul(sqBuf, src, src, tD);
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
    // TODO：完成多VF融合和双缓冲后，执行Membase回退消融，复测改回高层接口的组合收益。
    MhcSandwichRegBase::ScaleAndMulGamma(buf, buf, gamma, scale, tD);
    PipeBarrier<PIPE_V>();
}

// MHC_Post计算

/// 批量加载低精度残差，在R2的VF内完成转换并计算全部N个head的x2。
__aicore__ inline void ComputeX2AllHeadsBatched(const float (&postVals)[N_MAX], uint64_t gmBaseOffset, uint32_t tOff,
                                                uint32_t tD)
{
    // Host已固定校验N为4，使用编译期常量便于展开标量循环。
    constexpr uint32_t N = N_MAX;
    LocalTensor<T> residual = bufResidualBf16_.AllocTensor<T>();
    // 四行在GM和UB中都使用固定行跨度，一次二维搬运减少MTE2指令发射。
    DataCopyExtParams copyParams = {N, tD * sizeof(T), (D_ - tD) * sizeof(T),
                                    (dimTile_ - tD) * sizeof(T), 0};
    DataCopyPadExtParams<T> copyPad = {false, 0, 0, 0};
    DataCopyPad(residual, residualGm_[gmBaseOffset], copyParams, copyPad);
    bufResidualBf16_.EnQue(residual);
    residual = bufResidualBf16_.DeQue<T>();
    LocalTensor<T> x2Cache = bufX2Cache_.Get<T>();
    LocalTensor<float> hOut = bufResFp32_.Get<float>()[N_MAX * dimTile_];
    LocalTensor<float> combWeights = bufPostComb_.Get<float>()[ALIGN8];

    for (uint32_t h = 0; h < N; h += 2U) {
        MhcSandwichRegBase::ComputePostHeadPair<T>(x2Cache[h * D_ + tOff], x2Cache[(h + 1U) * D_ + tOff], hOut,
                                                   residual, combWeights, dimTile_, h, postVals[h], postVals[h + 1U],
                                                   tD);
    }
    PipeBarrier<PIPE_V>();
    bufResidualBf16_.FreeTensor(residual);
}

// 门控计算

/// 由xHatVals计算gate[0..N-1]，其中包含invRmsPre缩放：
///   gate[i] = sigmoid(alpha0 * xHat[i] + bias[i]) + hcEps
__aicore__ inline void ComputeGateValues(float totalSumSqPre, float (&xHatVals)[N_MAX])
{
    // Host已固定校验N为4，使用编译期常量便于展开标量循环。
    constexpr uint32_t N = N_MAX;
    float invRmsPre = 1.0f / sqrt(totalSumSqPre * invK_ + mhcPreNormEps_);
    for (uint32_t o = 0; o < N; ++o)
        xHatVals[o] *= invRmsPre;

    LocalTensor<float> scBuf = bufScalar_.Get<float>();
    for (uint32_t i = 0; i < N; ++i)
        scBuf.SetValue(i, -(alpha0_ * xHatVals[i] + biasCache_[i]));
    LocalTensor<float> gateBuf = scBuf[EXP_DIV_ELEM];
    // TODO：后续单独评估固定四元素循环展开，本轮保持通用VF循环。
    MhcSandwichRegBase::SigmoidAddEps(gateBuf, scBuf, hcEps_, N);
    PipeBarrier<PIPE_V>();
}
