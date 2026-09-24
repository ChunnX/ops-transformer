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
// mhc_norm_post_preonly_kernel_compute.h
// 共享计算模块：数据加载、RMSNorm、MHC Post、门控
// ========================================================================

// 数据加载

/// 加载一个token的h_post[N]和h_res[N*N]。
__aicore__ inline void LoadTokenWeights(uint32_t token, float (&postVals)[N_MAX], uint32_t headBegin = 0U,
                                        uint32_t headEnd = N_MAX)
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
    // 双AIV每条lane只消费自己负责的两个post标量，避免无效UB到Scalar回读。
    for (uint32_t i = headBegin; i < headEnd; ++i)
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
    MhcNormRegBase::CastB16ToFp32(hOut, hBf16, tD);
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

/// 提前发起完整gamma1搬运到bufGamma0Fp32_，使MTE2与R6的Vector计算重叠。
/// bufGamma0Fp32_在R3完成后即空闲，R6的平方缓存bufGamma2Fp32_不受影响。
__aicore__ inline void StartCacheGamma1()
{
    if (D_ > GAMMA1_CACHE_MAX_D) {
        return;
    }
    LocalTensor<float> gamma1 = bufGamma0Fp32_.Get<float>();
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
    MhcNormRegBase::CastB16ToFp32(hOut, hOutCache, tD);
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
    MhcNormRegBase::Mul(sqBuf, src, src, tD);
    PipeBarrier<PIPE_V>();
    return ReduceSumPrecise(sqBuf, tD);
}

/// R1直接从低精度h_out缓存计算平方和，省去先恢复完整FP32 h_out再平方的中间UB写读。
__aicore__ inline float ReduceSumSqFromCache(uint32_t tOff, uint32_t tD)
{
    LocalTensor<T> hOutCache = bufX2Cache_.Get<T>()[tOff];
    LocalTensor<float> sqBuf = bufGamma2Fp32_.Get<float>();
    MhcNormRegBase::CastB16ToFp32Square(sqBuf, hOutCache, tD);
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
    MhcNormRegBase::ScaleAndMulGamma(buf, buf, gamma, scale, tD);
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
    for (uint32_t n = 0; n < N; ++n)
        DataCopy(residual[n * dimTile_], residualGm_[gmBaseOffset + (uint64_t)n * D_], tD);
    bufResidualBf16_.EnQue(residual);
    residual = bufResidualBf16_.DeQue<T>();
    LocalTensor<T> x2Cache = bufX2Cache_.Get<T>();
    LocalTensor<float> hOut = bufResFp32_.Get<float>()[N_MAX * dimTile_];
    LocalTensor<float> combWeights = bufPostComb_.Get<float>()[ALIGN8];

    for (uint32_t h = 0; h < N; h += 2U) {
        MhcNormRegBase::ComputePostHeadPair<T>(x2Cache[h * D_ + tOff], x2Cache[(h + 1U) * D_ + tOff], hOut, residual,
                                               combWeights, dimTile_, h, postVals[h], postVals[h + 1U], tD);
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
    MhcNormRegBase::SigmoidAddEps(gateBuf, scBuf, hcEps_, N);
    PipeBarrier<PIPE_V>();
}

// 输入输出与本核流水同步辅助函数。

__aicore__ inline void QuantizeRoundTrip(LocalTensor<float> buf, uint32_t count)
{
    LocalTensor<T> tmp = bufHInPrimeBf16_.AllocTensor<T>();
    CastToIO(tmp, buf, count);
    PipeBarrier<PIPE_V>();
    MhcNormRegBase::CastB16ToFp32(buf, tmp, count);
    PipeBarrier<PIPE_V>();
    bufHInPrimeBf16_.FreeTensor(tmp);
}

__aicore__ inline void CastToIO(LocalTensor<T> dst, LocalTensor<float> src, uint32_t count)
{
    MhcNormRegBase::CastFp32ToB16(dst, src, count);
}

__aicore__ inline void FenceMte3ToV()
{
    SetFlag<HardEvent::MTE3_V>(evtMte3toV_);
    WaitFlag<HardEvent::MTE3_V>(evtMte3toV_);
}

__aicore__ inline void FenceVToMte2()
{
    SetFlag<HardEvent::V_MTE2>(evtVtoMte2_);
    WaitFlag<HardEvent::V_MTE2>(evtVtoMte2_);
}

__aicore__ inline void WriteLowPrecisionTile(GlobalTensor<T> &dst, uint64_t gmOffset, LocalTensor<float> src,
                                             uint32_t count)
{
    LocalTensor<T> tmp = bufHInPrimeBf16_.AllocTensor<T>();
    CastToIO(tmp, src, count);
    bufHInPrimeBf16_.EnQue(tmp);
    tmp = bufHInPrimeBf16_.DeQue<T>();
    DataCopy(dst[gmOffset], tmp, count);
    bufHInPrimeBf16_.FreeTensor(tmp);
}

__aicore__ inline void WriteHInPrimeTile(LocalTensor<float> src, uint64_t gmOffset, uint32_t count)
{
    WriteLowPrecisionTile(hInPrimeGm_, gmOffset, src, count);
    if constexpr (RETURN_H_IN_F32) {
        SetFlag<HardEvent::V_MTE3>(evtVtoMte3_);
        WaitFlag<HardEvent::V_MTE3>(evtVtoMte3_);
        DataCopy(hInF32Gm_[gmOffset], src, count);
        FenceMte3ToV();
    }
}

__aicore__ inline uint32_t GetTileD(uint32_t tile) const
{
    if (lastDimTile_ != 0 && tile == dimLoop_ - 1) {
        return lastDimTile_;
    }
    return dimTile_;
}

// 单AIV和双sibling AIV路径共用的分块计算。

// 构造一个低精度x2_post分块。R1使用normEps_，R2保持h_res的[j,n]索引顺序，
// 并显式保留设计规定的两处低精度往返边界。
__aicore__ inline void ComputeX2PostTile(uint32_t token, uint32_t tOff, uint32_t tD, float invRms0,
                                         const float (&postVals)[N_MAX])
{
    LoadCachedHOut(tOff, tD);
    LocalTensor<float> hOut = bufResFp32_.Get<float>()[N_MAX * dimTile_];
    ApplyScaleAndGamma(hOut, invRms0, bufGamma0Fp32_.Get<float>()[tOff], tD);
    QuantizeRoundTrip(hOut, tD);

    ComputeX2AllHeadsBatched(postVals, (uint64_t)token * N_ * D_ + tOff, tOff, tD);
}

// 双AIV仍需读取全部残差行，但每个lane只计算自己负责的两个输出head。
__aicore__ inline void ComputeX2PostTileForLane(uint32_t token, uint32_t tOff, uint32_t tD, float invRms0,
                                                const float (&postVals)[N_MAX])
{
    LoadCachedHOut(tOff, tD);
    LocalTensor<float> hOut = bufResFp32_.Get<float>()[N_MAX * dimTile_];

    // 提前发起residual搬运（MTE2），与下方h_out的scale/gamma/quantize（Vector）重叠。
    const uint32_t headBegin = laneId_ * 2U;
    const uint64_t gmBaseOffset = (uint64_t)token * N_ * D_ + tOff;
    LocalTensor<T> residual = bufResidualBf16_.AllocTensor<T>();
    for (uint32_t n = 0; n < N_MAX; ++n) {
        DataCopy(residual[n * dimTile_], residualGm_[gmBaseOffset + (uint64_t)n * D_], tD);
    }
    bufResidualBf16_.EnQue(residual);

    ApplyScaleAndGamma(hOut, invRms0, bufGamma0Fp32_.Get<float>()[tOff], tD);
    QuantizeRoundTrip(hOut, tD);

    residual = bufResidualBf16_.DeQue<T>();
    LocalTensor<T> x2Cache = bufX2Cache_.Get<T>();
    LocalTensor<float> combWeights = bufPostComb_.Get<float>()[ALIGN8];
    MhcNormRegBase::ComputePostHeadPair<T>(x2Cache[headBegin * D_ + tOff], x2Cache[(headBegin + 1U) * D_ + tOff], hOut,
                                           residual, combWeights, dimTile_, headBegin, postVals[headBegin],
                                           postVals[headBegin + 1U], tD);
    PipeBarrier<PIPE_V>();
    bufResidualBf16_.FreeTensor(residual);
}

// 将完整x2以输出dtype常驻UB，同时完成设计要求的低精度往返；写GM仅用于保留x2_out输出语义。
__aicore__ inline void CacheX2Tile(uint32_t token, uint32_t tOff, uint32_t tD, bool writeOutput,
                                   bool quantizeInput = true)
{
    LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
    LocalTensor<T> x2Cache = bufX2Cache_.Get<T>();
    const uint64_t outBase = (uint64_t)token * N_ * D_ + tOff;

    // R2已在成对VF尾部完成首次量化；R3后的第二次量化仍使用该路径。
    if (quantizeInput) {
        for (uint32_t n = 0; n < N_MAX; n += 2U) {
            MhcNormRegBase::CastFp32PairToB16(x2Cache[n * D_ + tOff], x2Cache[(n + 1U) * D_ + tOff], x2[n * dimTile_],
                                              x2[(n + 1U) * dimTile_], tD);
        }
        PipeBarrier<PIPE_V>();
    }
    // RMS_mid第一遍直接从低精度缓存计算平方和；R3后必须恢复，
    // 保证后续R4/R5仍使用低精度往返后的x2。
    if (!USE_RMS_MID || writeOutput) {
        for (uint32_t n = 0; n < N_MAX; n += 2U) {
            MhcNormRegBase::CastB16PairToFp32(x2[n * dimTile_], x2[(n + 1U) * dimTile_], x2Cache[n * D_ + tOff],
                                              x2Cache[(n + 1U) * D_ + tOff], tD);
        }
        PipeBarrier<PIPE_V>();
    }

    if (writeOutput) {
        // cacheTile已经是x2_out所需的最终低精度位值，直接写GM，
        // 避免先恢复FP32再重复量化和反复申请输出队列。
        SetFlag<HardEvent::V_MTE3>(evtVtoMte3_);
        WaitFlag<HardEvent::V_MTE3>(evtVtoMte3_);
        for (uint32_t n = 0; n < N_MAX; ++n) {
            DataCopy(x2OutGm_[outBase + (uint64_t)n * D_], x2Cache[n * D_ + tOff], tD);
        }
    }
}

// 双sibling路径仍在每个lane的私有UB中保留完整x2，便于两个lane按同一顺序执行R1/R2/R3；
// 写回x2_out时每个lane只写自己负责的两个head，避免任何输出竞争。
__aicore__ inline void CacheX2TileForLane(uint32_t token, uint32_t tOff, uint32_t tD, bool writeOutput,
                                          bool quantizeInput = true)
{
    LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
    LocalTensor<T> x2Cache = bufX2Cache_.Get<T>();
    const uint64_t outBase = (uint64_t)token * N_ * D_ + tOff;
    const uint32_t headBegin = laneId_ * 2U;

    if (quantizeInput) {
        for (uint32_t n = 0; n < N_MAX; n += 2U) {
            MhcNormRegBase::CastFp32PairToB16(x2Cache[n * D_ + tOff], x2Cache[(n + 1U) * D_ + tOff], x2[n * dimTile_],
                                              x2[(n + 1U) * dimTile_], tD);
        }
        PipeBarrier<PIPE_V>();
    }
    if (!USE_RMS_MID || writeOutput) {
        // 双路R4/R5只读取本lane负责的相邻两行，避免恢复另一lane的无效副本。
        MhcNormRegBase::CastB16PairToFp32(x2, x2[dimTile_], x2Cache[headBegin * D_ + tOff],
                                          x2Cache[(headBegin + 1U) * D_ + tOff], tD);
        PipeBarrier<PIPE_V>();
    }

    if (writeOutput) {
        SetFlag<HardEvent::V_MTE3>(evtVtoMte3_);
        WaitFlag<HardEvent::V_MTE3>(evtVtoMte3_);
        for (uint32_t n = headBegin; n < headBegin + 2U; ++n) {
            DataCopy(x2OutGm_[outBase + (uint64_t)n * D_], x2Cache[n * D_ + tOff], tD);
        }
    }
}

// R6读取完当前x2分块后，复用首个head缓存保留低精度h_in。
// 恢复FP32与平方在同一VF内完成，保留量化往返边界并减少UB中间写读。
__aicore__ inline float CacheHInAndReduceSumSq(uint32_t tOff, uint32_t tD, LocalTensor<float> hIn)
{
    LocalTensor<T> hInCache = bufX2Cache_.Get<T>()[tOff];
    CastToIO(hInCache, hIn, tD);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> sqBuf = bufGamma2Fp32_.Get<float>();
    MhcNormRegBase::CastB16ToFp32Square(sqBuf, hInCache, tD);
    PipeBarrier<PIPE_V>();
    return ReduceSumPrecise(sqBuf, tD);
}

__aicore__ inline void LoadHInCacheTile(uint32_t tOff, uint32_t tD, LocalTensor<float> hIn)
{
    LocalTensor<T> hInCache = bufX2Cache_.Get<T>()[tOff];
    MhcNormRegBase::CastB16ToFp32(hIn, hInCache, tD);
    PipeBarrier<PIPE_V>();
}

// 固定的R4/R5归约树由单AIV和双sibling AIV路径共用。half0固定覆盖展平区间[0,2D)，
// half1固定覆盖[2D,4D)，调用方只执行一次顶层合并partial0+partial1。
__aicore__ inline void AccumulateHalfPartials(uint32_t tOff, uint32_t tD, float (&sumSqHalf)[2],
                                              float (&dotHalf)[2][N_MAX], uint32_t halfBegin = 0, uint32_t halfEnd = 2)
{
    LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
    LocalTensor<float> reduceRows = bufResFp32_.Get<float>();
    LocalTensor<float> tmp = bufGamma2Fp32_.Get<float>();

    DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
    copyParams.blockLen = tD * sizeof(float);
    DataCopyPadExtParams<float> copyPad = {false, 0, 0, 0};

    // 尾块的物理行跨度仍为dimTile_，不能作为紧凑二维张量归约；
    // 尾块保持逐行归约，完整分块才使用五行批量归约。
    if (tD != dimTile_) {
        LocalTensor<float> out = reduceRows[N_MAX * dimTile_];
        for (uint32_t half = halfBegin; half < halfEnd; ++half) {
            const uint32_t h0 = half * 2;
            const uint32_t h1 = h0 + 1;
            const uint32_t x2h0 = DUAL_SIBLING ? 0U : h0;
            const uint32_t x2h1 = DUAL_SIBLING ? 1U : h1;

            MhcNormRegBase::Mul(out, x2[x2h0 * dimTile_], x2[x2h0 * dimTile_], tD);
            PipeBarrier<PIPE_V>();
            MhcNormRegBase::Mul(tmp, x2[x2h1 * dimTile_], x2[x2h1 * dimTile_], tD);
            PipeBarrier<PIPE_V>();
            MhcNormRegBase::Add(out, out, tmp, tD);
            PipeBarrier<PIPE_V>();
            sumSqHalf[half] += ReduceSumPrecise(out, tD);

            for (uint32_t o = 0; o < N_MAX; ++o) {
                LocalTensor<float> phi = bufPhiFp32_.AllocTensor<float>();
                uint64_t phiOffset = (uint64_t)o * phiCols_ + (uint64_t)h0 * D_ + tOff;
                DataCopyPad(phi, phiGm_[phiOffset], copyParams, copyPad);
                bufPhiFp32_.EnQue(phi);
                phi = bufPhiFp32_.DeQue<float>();
                MhcNormRegBase::Mul(out, x2[x2h0 * dimTile_], phi, tD);
                PipeBarrier<PIPE_V>();
                bufPhiFp32_.FreeTensor(phi);

                phi = bufPhiFp32_.AllocTensor<float>();
                phiOffset = (uint64_t)o * phiCols_ + (uint64_t)h1 * D_ + tOff;
                DataCopyPad(phi, phiGm_[phiOffset], copyParams, copyPad);
                bufPhiFp32_.EnQue(phi);
                phi = bufPhiFp32_.DeQue<float>();
                MhcNormRegBase::Mul(tmp, x2[x2h1 * dimTile_], phi, tD);
                PipeBarrier<PIPE_V>();
                bufPhiFp32_.FreeTensor(phi);

                MhcNormRegBase::Add(out, out, tmp, tD);
                PipeBarrier<PIPE_V>();
                dotHalf[half][o] += ReduceSumPrecise(out, tD);
            }
        }
        return;
    }

    for (uint32_t half = halfBegin; half < halfEnd; ++half) {
        const uint32_t h0 = half * 2;
        const uint32_t h1 = h0 + 1;
        const uint32_t x2h0 = DUAL_SIBLING ? 0U : h0;
        const uint32_t x2h1 = DUAL_SIBLING ? 1U : h1;

        // 五行依次保存{sumSqPre,dot0,dot1,dot2,dot3}，一次按行归约，
        // 避免每个标量结果都单独触发一次Vector到Scalar的等待。
        MhcNormRegBase::Mul(reduceRows, x2[x2h0 * dimTile_], x2[x2h0 * dimTile_], tD);
        PipeBarrier<PIPE_V>();
        MhcNormRegBase::Mul(tmp, x2[x2h1 * dimTile_], x2[x2h1 * dimTile_], tD);
        PipeBarrier<PIPE_V>();
        MhcNormRegBase::Add(reduceRows, reduceRows, tmp, tD);

        // phi ping-pong：prologue预取o=0的两路，后续算o时预取o+1，4槽使MTE2与Vector重叠。
        LocalTensor<float> curPhi0 = bufPhiFp32_.AllocTensor<float>();
        LocalTensor<float> curPhi1 = bufPhiFp32_.AllocTensor<float>();
        DataCopyPad(curPhi0, phiGm_[(uint64_t)h0 * D_ + tOff], copyParams, copyPad);
        DataCopyPad(curPhi1, phiGm_[(uint64_t)h1 * D_ + tOff], copyParams, copyPad);
        bufPhiFp32_.EnQue(curPhi0);
        bufPhiFp32_.EnQue(curPhi1);

        for (uint32_t o = 0; o < N_MAX; ++o) {
            // 预取下一轮o+1的两路phi，与当前dot计算重叠。
            LocalTensor<float> nextPhi0;
            LocalTensor<float> nextPhi1;
            if (o + 1U < N_MAX) {
                nextPhi0 = bufPhiFp32_.AllocTensor<float>();
                nextPhi1 = bufPhiFp32_.AllocTensor<float>();
                DataCopyPad(nextPhi0, phiGm_[(uint64_t)(o + 1U) * phiCols_ + (uint64_t)h0 * D_ + tOff], copyParams,
                            copyPad);
                DataCopyPad(nextPhi1, phiGm_[(uint64_t)(o + 1U) * phiCols_ + (uint64_t)h1 * D_ + tOff], copyParams,
                            copyPad);
                bufPhiFp32_.EnQue(nextPhi0);
                bufPhiFp32_.EnQue(nextPhi1);
            }

            LocalTensor<float> dotRow = reduceRows[(o + 1) * dimTile_];
            curPhi0 = bufPhiFp32_.DeQue<float>();
            curPhi1 = bufPhiFp32_.DeQue<float>();
            MhcNormRegBase::Mul(dotRow, x2[x2h0 * dimTile_], curPhi0, tD);
            PipeBarrier<PIPE_V>();
            MhcNormRegBase::Mul(tmp, x2[x2h1 * dimTile_], curPhi1, tD);
            PipeBarrier<PIPE_V>();
            MhcNormRegBase::Add(dotRow, dotRow, tmp, tD);
            bufPhiFp32_.FreeTensor(curPhi0);
            bufPhiFp32_.FreeTensor(curPhi1);

            if (o + 1U < N_MAX) {
                curPhi0 = nextPhi0;
                curPhi1 = nextPhi1;
            }
        }

        PipeBarrier<PIPE_V>();
        LocalTensor<float> scBuf = bufScalar_.Get<float>();
        uint32_t reduceShape[2] = {N_MAX + 1, tD};
        // 五行输入在归约后不再使用，允许接口复用源缓冲，避免额外申请UB。
        ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(scBuf, reduceRows, reduceShape, true);
        PipeBarrier<PIPE_V>();
        sumSqHalf[half] += scBuf.GetValue(0);
        for (uint32_t o = 0; o < N_MAX; ++o) {
            dotHalf[half][o] += scBuf.GetValue(o + 1);
        }
    }
}
