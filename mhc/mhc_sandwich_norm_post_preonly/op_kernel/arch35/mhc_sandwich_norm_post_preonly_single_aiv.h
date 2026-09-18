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
template<typename T>
__aicore__ inline void KernelMhcSandwichNormPostPreonly<T>::ComputeX2PostTile(
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
    LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
    for (uint32_t n = 0; n < N_; ++n) {
        QuantizeRoundTrip(x2[n * dimTile_], tD);
    }
}

template<typename T>
__aicore__ inline void KernelMhcSandwichNormPostPreonly<T>::LoadX2OutputTile(
    uint32_t token, uint32_t tOff, uint32_t tD)
{
    LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
    const uint64_t base = (uint64_t)token * N_ * D_ + tOff;
    for (uint32_t n = 0; n < N_; ++n) {
        LoadLowPrecisionTile(x2OutGm_, base + (uint64_t)n * D_, x2[n * dimTile_], tD);
    }
}

// 固定的R4/R5归约树将与双sibling AIV路径共用。half0固定覆盖展平区间[0,2D)，
// half1固定覆盖[2D,4D)，调用方只执行一次顶层合并partial0+partial1。
template<typename T>
__aicore__ inline void KernelMhcSandwichNormPostPreonly<T>::AccumulateHalfPartials(
    uint32_t tOff, uint32_t tD, float (&sumSqHalf)[2], float (&dotHalf)[2][N_MAX])
{
    LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
    LocalTensor<float> out = bufOutFp32_.Get<float>();
    LocalTensor<float> tmp = bufSqFp32_.Get<float>();

    DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
    copyParams.blockLen = tD * sizeof(float);
    DataCopyPadExtParams<float> copyPad = {false, 0, 0, 0};

    for (uint32_t half = 0; half < 2; ++half) {
        const uint32_t h0 = half * 2;
        const uint32_t h1 = h0 + 1;

        Mul(out, x2[h0 * dimTile_], x2[h0 * dimTile_], tD);
        PipeBarrier<PIPE_V>();
        Mul(tmp, x2[h1 * dimTile_], x2[h1 * dimTile_], tD);
        PipeBarrier<PIPE_V>();
        Add(out, out, tmp, tD);
        PipeBarrier<PIPE_V>();
        sumSqHalf[half] += ReduceSumPrecise(out, tD);

        for (uint32_t o = 0; o < N_; ++o) {
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
}

template<typename T>
__aicore__ inline void KernelMhcSandwichNormPostPreonly<T>::ProcessTokenSingleAiv(uint32_t token)
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

    // 第二遍：计算R1/R2和第一处x2低精度边界。x2_out是唯一的跨遍暂存，
    // 不分配完整FP32 x2工作空间。
    for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dimTile_;
        ComputeX2PostTile(token, tOff, tD, invRms0, postVals, combVals);

        LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
        if (useRmsMid_) {
            LocalTensor<float> out = bufOutFp32_.Get<float>();
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

        const uint64_t outBase = (uint64_t)token * N_ * D_ + tOff;
        for (uint32_t n = 0; n < N_; ++n) {
            WriteX2OutTile(x2[n * dimTile_], outBase + (uint64_t)n * D_, tD);
        }
    }
    FenceMte3ToMte2();

    // 可选R3沿完整N*D向量计算，不按head分别归一化。
    if (useRmsMid_) {
        const float totalMidSq = midSqHalf[0] + midSqHalf[1];
        const float invRmsMid = 1.0f / sqrt(totalMidSq * invK_ + normEps_);
        for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
            const uint32_t tD = GetTileD(tile);
            const uint32_t tOff = tile * dimTile_;
            LoadX2OutputTile(token, tOff, tD);
            LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
            LocalTensor<float> gamma2 = bufGamma2Fp32_.Get<float>();
            for (uint32_t n = 0; n < N_; ++n) {
                LoadGammaInto(gamma2, gamma2Gm_, n * D_ + tOff, tD);
                SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
                WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);
                ApplyScaleAndGamma(x2[n * dimTile_], invRmsMid, gamma2, tD);
                FenceVToMte2();
                QuantizeRoundTrip(x2[n * dimTile_], tD);
            }
            AccumulateHalfPartials(tOff, tD, sumSqHalf, dotHalf);

            const uint64_t outBase = (uint64_t)token * N_ * D_ + tOff;
            for (uint32_t n = 0; n < N_; ++n) {
                WriteX2OutTile(x2[n * dimTile_], outBase + (uint64_t)n * D_, tD);
            }
        }
        FenceMte3ToMte2();
    }

    // 固定执行顶层合并，与双sibling方案保持一致。
    const float totalSumSqPre = sumSqHalf[0] + sumSqHalf[1];
    float xHatVals[N_MAX];
    for (uint32_t o = 0; o < N_; ++o) {
        xHatVals[o] = dotHalf[0][o] + dotHalf[1][o];
    }
    float gate[N_MAX];
    ComputeGateValues(totalSumSqPre, xHatVals, gate);

    // 第三遍：计算R6和h_in低精度边界。在完整D轴的R7标量确定前，
    // h_in_prime临时保存h_in。
    float sumSq1 = 0.0f;
    for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dimTile_;
        LoadX2OutputTile(token, tOff, tD);
        LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
        LocalTensor<float> hIn = bufOutFp32_.Get<float>();
        Muls(hIn, x2, gate[0], tD);
        PipeBarrier<PIPE_V>();
        for (uint32_t n = 1; n < N_; ++n) {
            Axpy(hIn, x2[n * dimTile_], gate[n], tD);
            PipeBarrier<PIPE_V>();
        }
        QuantizeRoundTrip(hIn, tD);
        sumSq1 += ReduceSumSq(hIn, tD);
        WriteLowPrecisionTile(hInPrimeGm_, (uint64_t)token * D_ + tOff, hIn, tD);
    }
    FenceMte3ToMte2();

    // 第四遍：沿完整D轴计算R7并写最终输出。
    const float invRms1 = ComputeInvRms(sumSq1, invD_);
    for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dimTile_;
        LocalTensor<float> hIn = bufOutFp32_.Get<float>();
        LoadLowPrecisionTile(hInPrimeGm_, (uint64_t)token * D_ + tOff, hIn, tD);
        LoadGammaInto(bufGamma1Fp32_.Get<float>(), gamma1Gm_, tOff, tD);
        SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
        WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);
        ApplyScaleAndGamma(hIn, invRms1, bufGamma1Fp32_.Get<float>(), tD);
        FenceVToMte2();
        WriteHInPrimeTile(hIn, (uint64_t)token * D_ + tOff, tD);
    }
}

#endif // MHC_SANDWICH_NORM_POST_PREONLY_SINGLE_AIV_H_
