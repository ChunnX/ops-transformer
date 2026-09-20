/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MHC_SANDWICH_NORM_POST_PREONLY_SINGLE_AIV_H_
#define MHC_SANDWICH_NORM_POST_PREONLY_SINGLE_AIV_H_

// 构造一个低精度x2_post分块。R1使用normEps_，R2保持h_res的[j,n]索引顺序，
// 并显式保留设计规定的两处低精度往返边界。
template<typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcSandwichNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ComputeX2PostTile(
    uint32_t token, uint32_t tOff, uint32_t tD, float invRms0,
    const float (&postVals)[N_MAX], const float (&combVals)[N_MAX * N_MAX])
{
    LoadHOutAndCast((uint64_t)token * D_ + tOff, tD);
    LoadGammaInto(bufGamma0Fp32_.Get<float>(), gamma0Gm_, tOff, tD);
    SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
    WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);
    ApplyScaleAndGamma(bufHOutFp32_.Get<float>(), invRms0, bufGamma0Fp32_.Get<float>(), tD);
    FenceVToMte2();
    QuantizeRoundTrip(bufHOutFp32_.Get<float>(), tD);

    LoadResidualAndCast((uint64_t)token * N_ * D_ + tOff, tD);
    ComputeX2AllHeadsBatched(postVals, combVals, tD);
}

// 将完整x2以输出dtype常驻UB，同时完成设计要求的低精度往返；写GM仅用于保留x2_out输出语义。
template<typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcSandwichNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::CacheX2Tile(
    uint32_t token, uint32_t tOff, uint32_t tD, bool writeOutput)
{
    LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
    LocalTensor<T> x2Cache = bufX2Cache_.Get<T>();
    const uint64_t outBase = (uint64_t)token * N_ * D_ + tOff;

    // 四个head写入互不重叠的低精度缓存，统一等待后再恢复FP32，
    // 保留逐元素量化边界并减少重复的Vector流水屏障。
    for (uint32_t n = 0; n < N_MAX; ++n) {
        LocalTensor<T> cacheTile = x2Cache[n * D_ + tOff];
        CastToIO(cacheTile, x2[n * dimTile_], tD);
    }
    PipeBarrier<PIPE_V>();
    for (uint32_t n = 0; n < N_MAX; ++n) {
        LocalTensor<T> cacheTile = x2Cache[n * D_ + tOff];
        Cast(x2[n * dimTile_], cacheTile, RoundMode::CAST_NONE, tD);
    }
    PipeBarrier<PIPE_V>();

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
template<typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcSandwichNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::CacheX2TileForLane(
    uint32_t token, uint32_t tOff, uint32_t tD, bool writeOutput)
{
    LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
    LocalTensor<T> x2Cache = bufX2Cache_.Get<T>();
    const uint64_t outBase = (uint64_t)token * N_ * D_ + tOff;

    for (uint32_t n = 0; n < N_MAX; ++n) {
        CastToIO(x2Cache[n * D_ + tOff], x2[n * dimTile_], tD);
    }
    PipeBarrier<PIPE_V>();
    for (uint32_t n = 0; n < N_MAX; ++n) {
        Cast(x2[n * dimTile_], x2Cache[n * D_ + tOff], RoundMode::CAST_NONE, tD);
    }
    PipeBarrier<PIPE_V>();

    if (writeOutput) {
        SetFlag<HardEvent::V_MTE3>(evtVtoMte3_);
        WaitFlag<HardEvent::V_MTE3>(evtVtoMte3_);
        const uint32_t headBegin = laneId_ * 2U;
        for (uint32_t n = headBegin; n < headBegin + 2U; ++n) {
            DataCopy(x2OutGm_[outBase + (uint64_t)n * D_], x2Cache[n * D_ + tOff], tD);
        }
    }
}

template<typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcSandwichNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::LoadX2CacheTile(
    uint32_t tOff, uint32_t tD)
{
    LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
    LocalTensor<T> x2Cache = bufX2Cache_.Get<T>();
    for (uint32_t n = 0; n < N_MAX; ++n) {
        Cast(x2[n * dimTile_], x2Cache[n * D_ + tOff], RoundMode::CAST_NONE, tD);
    }
    PipeBarrier<PIPE_V>();
}

// R6读取完当前x2分块后，首个head的对应缓存区已经不会再被使用，
// 将其复用为低精度h_in缓存，保留量化往返边界并避免GM临时写回。
template<typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcSandwichNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::CacheHInTile(
    uint32_t tOff, uint32_t tD, LocalTensor<float> hIn)
{
    LocalTensor<T> hInCache = bufX2Cache_.Get<T>()[tOff];
    CastToIO(hInCache, hIn, tD);
    PipeBarrier<PIPE_V>();
    Cast(hIn, hInCache, RoundMode::CAST_NONE, tD);
    PipeBarrier<PIPE_V>();
}

template<typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcSandwichNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::LoadHInCacheTile(
    uint32_t tOff, uint32_t tD, LocalTensor<float> hIn)
{
    LocalTensor<T> hInCache = bufX2Cache_.Get<T>()[tOff];
    Cast(hIn, hInCache, RoundMode::CAST_NONE, tD);
    PipeBarrier<PIPE_V>();
}

// 固定的R4/R5归约树将与双sibling AIV路径共用。half0固定覆盖展平区间[0,2D)，
// half1固定覆盖[2D,4D)，调用方只执行一次顶层合并partial0+partial1。
template<typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcSandwichNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::AccumulateHalfPartials(
    uint32_t tOff, uint32_t tD, float (&sumSqHalf)[2], float (&dotHalf)[2][N_MAX],
    uint32_t halfBegin, uint32_t halfEnd)
{
    LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
    LocalTensor<float> reduceRows = bufResFp32_.Get<float>();
    LocalTensor<float> tmp = bufSqFp32_.Get<float>();

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

            Mul(out, x2[h0 * dimTile_], x2[h0 * dimTile_], tD);
            PipeBarrier<PIPE_V>();
            Mul(tmp, x2[h1 * dimTile_], x2[h1 * dimTile_], tD);
            PipeBarrier<PIPE_V>();
            Add(out, out, tmp, tD);
            PipeBarrier<PIPE_V>();
            sumSqHalf[half] += ReduceSumPrecise(out, tD);

            for (uint32_t o = 0; o < N_MAX; ++o) {
                LocalTensor<float> phi = bufPhiFp32_.AllocTensor<float>();
                uint64_t phiOffset = (uint64_t)o * phiCols_ + (uint64_t)h0 * D_ + tOff;
                DataCopyPad(phi, phiGm_[phiOffset], copyParams, copyPad);
                bufPhiFp32_.EnQue(phi);
                phi = bufPhiFp32_.DeQue<float>();
                Mul(out, x2[h0 * dimTile_], phi, tD);
                PipeBarrier<PIPE_V>();
                bufPhiFp32_.FreeTensor(phi);

                phi = bufPhiFp32_.AllocTensor<float>();
                phiOffset = (uint64_t)o * phiCols_ + (uint64_t)h1 * D_ + tOff;
                DataCopyPad(phi, phiGm_[phiOffset], copyParams, copyPad);
                bufPhiFp32_.EnQue(phi);
                phi = bufPhiFp32_.DeQue<float>();
                Mul(tmp, x2[h1 * dimTile_], phi, tD);
                PipeBarrier<PIPE_V>();
                bufPhiFp32_.FreeTensor(phi);

                Add(out, out, tmp, tD);
                PipeBarrier<PIPE_V>();
                dotHalf[half][o] += ReduceSumPrecise(out, tD);
            }
        }
        return;
    }

    for (uint32_t half = halfBegin; half < halfEnd; ++half) {
        const uint32_t h0 = half * 2;
        const uint32_t h1 = h0 + 1;

        // 五行依次保存{sumSqPre,dot0,dot1,dot2,dot3}，一次按行归约，
        // 避免每个标量结果都单独触发一次Vector到Scalar的等待。
        Mul(reduceRows, x2[h0 * dimTile_], x2[h0 * dimTile_], tD);
        PipeBarrier<PIPE_V>();
        Mul(tmp, x2[h1 * dimTile_], x2[h1 * dimTile_], tD);
        PipeBarrier<PIPE_V>();
        Add(reduceRows, reduceRows, tmp, tD);

        for (uint32_t o = 0; o < N_MAX; ++o) {
            LocalTensor<float> dotRow = reduceRows[(o + 1) * dimTile_];
            LocalTensor<float> phi = bufPhiFp32_.AllocTensor<float>();
            uint64_t phiOffset = (uint64_t)o * phiCols_ + (uint64_t)h0 * D_ + tOff;
            DataCopyPad(phi, phiGm_[phiOffset], copyParams, copyPad);
            bufPhiFp32_.EnQue(phi);
            phi = bufPhiFp32_.DeQue<float>();
            Mul(dotRow, x2[h0 * dimTile_], phi, tD);
            PipeBarrier<PIPE_V>();
            bufPhiFp32_.FreeTensor(phi);

            phi = bufPhiFp32_.AllocTensor<float>();
            phiOffset = (uint64_t)o * phiCols_ + (uint64_t)h1 * D_ + tOff;
            DataCopyPad(phi, phiGm_[phiOffset], copyParams, copyPad);
            bufPhiFp32_.EnQue(phi);
            phi = bufPhiFp32_.DeQue<float>();
            Mul(tmp, x2[h1 * dimTile_], phi, tD);
            PipeBarrier<PIPE_V>();
            bufPhiFp32_.FreeTensor(phi);

            Add(dotRow, dotRow, tmp, tD);
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

template<typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcSandwichNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ProcessTokenSingleAiv(
    uint32_t token)
{
    float postVals[N_MAX];
    float combVals[N_MAX * N_MAX];
    float sumSqHalf[2] = {0.0f, 0.0f};
    float dotHalf[2][N_MAX] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    float midSqHalf[2] = {0.0f, 0.0f};
    LoadTokenWeights(token, postVals, combVals);

    // 第一遍：沿完整D轴计算R1。
    float sumSq0 = 0.0f;
    for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dimTile_;
        LoadHOutAndCast((uint64_t)token * D_ + tOff, tD);
        sumSq0 += ReduceSumSq(bufHOutFp32_.Get<float>(), tD);
    }
    const float invRms0 = ComputeInvRms(sumSq0, invD_);

    // 第二遍：计算R1/R2和第一处x2低精度边界。完整x2以输出dtype常驻UB，
    // 避免后续两遍从GM回读，同时不分配完整FP32 x2工作空间。
    for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dimTile_;
        ComputeX2PostTile(token, tOff, tD, invRms0, postVals, combVals);
        CacheX2Tile(token, tOff, tD, !USE_RMS_MID);

        LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
        if constexpr (USE_RMS_MID) {
            LocalTensor<float> out = bufResFp32_.Get<float>()[N_MAX * dimTile_];
            LocalTensor<float> tmp = bufSqFp32_.Get<float>();
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t h0 = half * 2;
                const uint32_t h1 = h0 + 1;
                Mul(out, x2[h0 * dimTile_], x2[h0 * dimTile_], tD);
                PipeBarrier<PIPE_V>();
                Mul(tmp, x2[h1 * dimTile_], x2[h1 * dimTile_], tD);
                PipeBarrier<PIPE_V>();
                Add(out, out, tmp, tD);
                PipeBarrier<PIPE_V>();
                midSqHalf[half] += ReduceSumPrecise(out, tD);
            }
        } else {
            AccumulateHalfPartials(tOff, tD, sumSqHalf, dotHalf);
        }

    }

    // 可选R3沿完整N*D向量计算，不按head分别归一化。
    if constexpr (USE_RMS_MID) {
        const float totalMidSq = midSqHalf[0] + midSqHalf[1];
        const float invRmsMid = 1.0f / sqrt(totalMidSq * invK_ + normEps_);
        for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
            const uint32_t tD = GetTileD(tile);
            const uint32_t tOff = tile * dimTile_;
            // R3阶段gamma0缓存已经空闲，与gamma2缓存组成双缓冲；
            // 首个gamma2搬运和x2缓存恢复重叠，后续搬运与前一head的Vector计算重叠。
            LocalTensor<float> gammaEven = bufGamma2Fp32_.Get<float>();
            LocalTensor<float> gammaOdd = bufGamma0Fp32_.Get<float>();
            LoadGammaInto(gammaEven, gamma2Gm_, tOff, tD);
            LoadX2CacheTile(tOff, tD);
            LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
            for (uint32_t n = 0; n < N_MAX; ++n) {
                LocalTensor<float> gammaCurrent = gammaEven;
                LocalTensor<float> gammaNext = gammaOdd;
                if ((n & 1U) != 0) {
                    gammaCurrent = gammaOdd;
                    gammaNext = gammaEven;
                }
                SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
                WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);
                if (n + 1 < N_MAX) {
                    LoadGammaInto(gammaNext, gamma2Gm_, (n + 1) * D_ + tOff, tD);
                }
                ApplyScaleAndGamma(x2[n * dimTile_], invRmsMid, gammaCurrent, tD);
                // 当前缓存会在两轮后再次由MTE2写入，前两轮显式建立V到MTE2依赖。
                if (n + 2 < N_MAX) {
                    FenceVToMte2();
                }
            }
            CacheX2Tile(token, tOff, tD, true);
            AccumulateHalfPartials(tOff, tD, sumSqHalf, dotHalf);
        }
    }
    FenceMte3ToV();

    // 固定执行顶层合并，与双sibling方案保持一致。
    const float totalSumSqPre = sumSqHalf[0] + sumSqHalf[1];
    float xHatVals[N_MAX];
    for (uint32_t o = 0; o < N_MAX; ++o) {
        xHatVals[o] = dotHalf[0][o] + dotHalf[1][o];
    }
    float gate[N_MAX];
    ComputeGateValues(totalSumSqPre, xHatVals, gate);

    // 第三遍：计算R6和h_in低精度边界。在完整D轴的R7标量确定前，
    // 复用已经消费完的x2缓存首段保存低精度h_in。
    float sumSq1 = 0.0f;
    for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dimTile_;
        LoadX2CacheTile(tOff, tD);
        LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
        LocalTensor<float> hIn = bufResFp32_.Get<float>()[N_MAX * dimTile_];
        Muls(hIn, x2, gate[0], tD);
        PipeBarrier<PIPE_V>();
        for (uint32_t n = 1; n < N_MAX; ++n) {
            Axpy(hIn, x2[n * dimTile_], gate[n], tD);
            PipeBarrier<PIPE_V>();
        }
        CacheHInTile(tOff, tD, hIn);
        sumSq1 += ReduceSumSq(hIn, tD);
    }

    // 第四遍：沿完整D轴计算R7并写最终输出。
    const float invRms1 = ComputeInvRms(sumSq1, invD_);
    for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dimTile_;
        LocalTensor<float> hIn = bufResFp32_.Get<float>()[N_MAX * dimTile_];
        LoadHInCacheTile(tOff, tD, hIn);
        LoadGammaInto(bufGamma1Fp32_.Get<float>(), gamma1Gm_, tOff, tD);
        SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
        WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);
        ApplyScaleAndGamma(hIn, invRms1, bufGamma1Fp32_.Get<float>(), tD);
        FenceVToMte2();
        WriteHInPrimeTile(hIn, (uint64_t)token * D_ + tOff, tD);
    }
}

#endif // MHC_SANDWICH_NORM_POST_PREONLY_SINGLE_AIV_H_
