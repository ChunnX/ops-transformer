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
// mhc_sandwich_norm_post_preonly_singlecore.h — single-core fallback (dimLoop>2 or N<2)
// v2: QuantizeRoundTrip at stage boundaries, N*D RMSNorm_mid, ReduceSumPrecise
// ========================================================================
#ifndef MHC_SANDWICH_NORM_POST_PREONLY_SINGLECORE_H_
#define MHC_SANDWICH_NORM_POST_PREONLY_SINGLECORE_H_

// =================================================================
//  ProcessToken: single-core entry point
// =================================================================
template<typename T>
__aicore__ __attribute__((noinline)) void
KernelMhcSandwichNormPostPreonly<T>::Apply2BatchResidualAxpyTail(
    uint64_t rBaseTile, uint32_t tD,
    const float (&combVals)[N_MAX * N_MAX])
{
    constexpr uint32_t RB = 2;
    const uint32_t N = N_, D = D_;
    LocalTensor<float> x2 = bufX2Fp32_.Get<float>();

    // Batch 0: DeQue + Cast (V)
    {
        LocalTensor<T> resBf16 = bufResidualBf16_.DeQue<T>();
        LocalTensor<float> resFp32 = bufResFp32_.Get<float>();
        for (uint32_t n = 0; n < RB; ++n)
            Cast(resFp32[n * dimTile_], resBf16[n * dimTile_], RoundMode::CAST_NONE, tD);
        PipeBarrier<PIPE_V>();
        bufResidualBf16_.FreeTensor(resBf16);
    }

    // Batch 1: start MTE2 for heads [RB, 2*RB) (overlaps with axpy below)
    {
        LocalTensor<T> resBf16 = bufResidualBf16_.AllocTensor<T>();
        for (uint32_t n = 0; n < RB; ++n)
            DataCopy(resBf16[n * dimTile_],
                     residualGm_[rBaseTile + (uint64_t)(n + RB) * D], tD);
        bufResidualBf16_.EnQue(resBf16);
    }

    // x2 += combVals[0..RB-1] * res[0..RB-1] (V — MTE2 for batch 1 in flight)
    {
        LocalTensor<float> resFp32 = bufResFp32_.Get<float>();
        for (uint32_t j = 0; j < RB; ++j)
            for (uint32_t h = 0; h < N; ++h)
                Axpy(x2[h * dimTile_], resFp32[j * dimTile_],
                     combVals[j * N + h], tD);
    }

    // Batch 1: DeQue + Cast (V)
    {
        LocalTensor<T> resBf16 = bufResidualBf16_.DeQue<T>();
        LocalTensor<float> resFp32 = bufResFp32_.Get<float>();
        for (uint32_t n = 0; n < RB; ++n)
            Cast(resFp32[n * dimTile_], resBf16[n * dimTile_], RoundMode::CAST_NONE, tD);
        PipeBarrier<PIPE_V>();
        bufResidualBf16_.FreeTensor(resBf16);
    }

    // x2 += combVals[RB..2*RB-1] * res[0..RB-1] (V)
    {
        LocalTensor<float> resFp32 = bufResFp32_.Get<float>();
        for (uint32_t j = 0; j < RB; ++j)
            for (uint32_t h = 0; h < N; ++h)
                Axpy(x2[h * dimTile_], resFp32[j * dimTile_],
                     combVals[(j + RB) * N + h], tD);
    }
    PipeBarrier<PIPE_V>();
}

template<typename T>
__aicore__ inline void KernelMhcSandwichNormPostPreonly<T>::ProcessToken(uint32_t token)
{
    const uint32_t D = D_, N = N_;
    float postVals[N_MAX], combVals[N_MAX * N_MAX];
    float xHatVals[N_MAX], gate[N_MAX];
    float sumSqPre = 0.0f, sumSq1 = 0.0f;
    for (uint32_t i = 0; i < N; ++i) { xHatVals[i] = 0.0f; }

    LoadTokenWeights(token, postVals, combVals);

    if (dimLoop_ > 1) {
        uint32_t tD0 = GetTileD(0);
        LoadGammaInto(bufGamma0Fp32_.Get<float>(), gamma0Gm_, 0, tD0);
        LoadGammaInto(bufGamma1Fp32_.Get<float>(), gamma1Gm_, 0, tD0);

        SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
        WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);
    }

    if (dimLoop_ == 1)
        PhaseAB_SingleTile(token, postVals, combVals, xHatVals, sumSqPre);
    else
        PhaseAB_MultiTile(token, postVals, combVals, xHatVals, sumSqPre);

    ComputeGateValues(sumSqPre, xHatVals, gate);
    PhaseC_HInAndX2Output(token, gate, sumSq1);

    float invRms1 = ComputeInvRms(sumSq1, invD_);
    PhaseD_RmsNorm1Output(token, invRms1);
}

// =================================================================
//  PhaseAB_SingleTile
// =================================================================
template<typename T>
__aicore__ inline void KernelMhcSandwichNormPostPreonly<T>::PhaseAB_SingleTile(
    uint32_t token,
    const float (&postVals)[N_MAX], const float (&combVals)[N_MAX * N_MAX],
    float (&xHatVals)[N_MAX], float &sumSqPre)
{
    const uint32_t N = N_, D = D_;
    uint32_t tD = dimTile_;
    const uint64_t rBase = (uint64_t)token * N * D;
    LoadHOutAndCast((uint64_t)token * D, tD);

    {
        LocalTensor<T> resBf16 = bufResidualBf16_.AllocTensor<T>();
        constexpr uint32_t RB = 2;  // residual batch size: 2 heads/batch
        for (uint32_t n = 0; n < RB; ++n)
            DataCopy(resBf16[n * dimTile_], residualGm_[rBase + (uint64_t)n * D], tD);
        bufResidualBf16_.EnQue(resBf16);
    }

    LocalTensor<float> hOutFp32 = bufHOutFp32_.Get<float>();
    float invRms0 = ComputeInvRms(ReduceSumSq(hOutFp32, tD), invD_);
    ApplyScaleAndGamma(hOutFp32, invRms0, bufGamma0Fp32_.Get<float>(), tD);
    // Stage 1 boundary: x1 → dtype → fp32
    QuantizeRoundTrip(hOutFp32, tD);

    // x2 init: x2[h] = postVals[h] * hOutNorm  (V)
    {
        LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
        for (uint32_t h = 0; h < N; ++h)
            Muls(x2[h * dimTile_], hOutFp32, postVals[h], tD);
    }

    // [perf] 2-batch residual axpy tail (shared with PhaseAB_MultiTile via
    // noinline helper).  See class declaration for full pre-/post-conditions.
    Apply2BatchResidualAxpyTail(rBase, tD, combVals);

    // Stage 2 boundary: x2 → dtype → fp32
    {
        LocalTensor<float> x2Tmp = bufX2Fp32_.Get<float>();
        for (uint32_t n = 0; n < N; ++n)
            QuantizeRoundTrip(x2Tmp[n * dimTile_], tD);
    }

    if (useRmsMid_) {
        ApplyRmsNormMidSingleTile(tD);
        // Stage 2.5 boundary: x2 → dtype → fp32
        LocalTensor<float> x2Tmp = bufX2Fp32_.Get<float>();
        for (uint32_t n = 0; n < N; ++n)
            QuantizeRoundTrip(x2Tmp[n * dimTile_], tD);
    }

    // sumSqPre
    LocalTensor<float> x2Fp32 = bufX2Fp32_.Get<float>();
    LocalTensor<float> outFp32 = bufOutFp32_.Get<float>();
    LocalTensor<float> sqBuf = bufSqFp32_.Get<float>();

    Mul(outFp32, x2Fp32, x2Fp32, tD); PipeBarrier<PIPE_V>();
    for (uint32_t n = 1; n < N; ++n) {
        Mul(sqBuf, x2Fp32[n*dimTile_], x2Fp32[n*dimTile_], tD); PipeBarrier<PIPE_V>();
        Add(outFp32, outFp32, sqBuf, tD); PipeBarrier<PIPE_V>();
    }
    sumSqPre += ReduceSumPrecise(outFp32, tD);

    // phi matmul
    DataCopyExtParams cpP = {1, 0, 0, 0, 0};
    cpP.blockLen = tD * sizeof(float);
    DataCopyPadExtParams<float> cpPad = {true, 0, 0, 0};

    for (uint32_t o = 0; o < phiRows_; ++o) {
        LocalTensor<float> phiFp32 = bufPhiFp32_.AllocTensor<float>();
        DataCopyPad(phiFp32, phiGm_[(uint64_t)o * phiCols_], cpP, cpPad);
        bufPhiFp32_.EnQue(phiFp32);
        phiFp32 = bufPhiFp32_.DeQue<float>();
        Mul(outFp32, x2Fp32, phiFp32, tD); PipeBarrier<PIPE_V>();
        bufPhiFp32_.FreeTensor(phiFp32);

        for (uint32_t n = 1; n < N; ++n) {
            uint64_t phiOff = (uint64_t)o * phiCols_ + (uint64_t)n * D;
            phiFp32 = bufPhiFp32_.AllocTensor<float>();
            DataCopyPad(phiFp32, phiGm_[phiOff], cpP, cpPad);
            bufPhiFp32_.EnQue(phiFp32);
            phiFp32 = bufPhiFp32_.DeQue<float>();
            Mul(sqBuf, x2Fp32[n*dimTile_], phiFp32, tD); PipeBarrier<PIPE_V>();
            Add(outFp32, outFp32, sqBuf, tD); PipeBarrier<PIPE_V>();
            bufPhiFp32_.FreeTensor(phiFp32);
        }
        xHatVals[o] += ReduceSumPrecise(outFp32, tD);
    }
}

// =================================================================
//  PhaseAB_MultiTile (single-core, dimLoop>2 fallback)
// =================================================================
template<typename T>
__aicore__ inline void KernelMhcSandwichNormPostPreonly<T>::PhaseAB_MultiTile(
    uint32_t token,
    const float (&postVals)[N_MAX], const float (&combVals)[N_MAX * N_MAX],
    float (&xHatVals)[N_MAX], float &sumSqPre)
{
    const uint32_t N = N_, D = D_;

    // Pass 1: RMSNorm_0 sum-of-squares
    float sumSq0 = 0.0f;
    for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
        uint32_t tD = GetTileD(tile);
        LoadHOutAndCast((uint64_t)token * D + tile * dimTile_, tD);
        sumSq0 += ReduceSumSq(bufHOutFp32_.Get<float>(), tD);
    }
    float invRms0 = ComputeInvRms(sumSq0, invD_);
    LocalTensor<float> gamma0Fp32 = bufGamma0Fp32_.Get<float>();

    // Pass 2: normalize + MHC_Post + accumulate sumSqMid (total across all heads)
    float totalSumSqMid = 0.0f;
    for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
        uint32_t tD = GetTileD(tile), tOff = tile * dimTile_;

        LoadHOutAndCast((uint64_t)token * D + tOff, tD);
        if (tile > 0) {
            LoadGammaInto(gamma0Fp32, gamma0Gm_, tOff, tD);
            SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
            WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);
        }
        ApplyScaleAndGamma(bufHOutFp32_.Get<float>(), invRms0, gamma0Fp32, tD);
        // Stage 1 boundary: x1 → dtype → fp32
        QuantizeRoundTrip(bufHOutFp32_.Get<float>(), tD);

        // [perf] 2-batch residual loading per tile (matches PhaseAB_SingleTile):
        // bufResFp32_ holds 2 heads in single-core mode → load + apply in 2
        // batches, overlapping MTE2 (next batch) with V (axpy of current batch).
        const uint64_t rBaseTile = (uint64_t)token * N * D + tOff;

        // Batch 0: kick off MTE2 for heads [0, RB) (parallel with x2 init below).
        {
            constexpr uint32_t RB = 2;
            LocalTensor<T> resBf16 = bufResidualBf16_.AllocTensor<T>();
            for (uint32_t n = 0; n < RB; ++n)
                DataCopy(resBf16[n * dimTile_],
                         residualGm_[rBaseTile + (uint64_t)n * D], tD);
            bufResidualBf16_.EnQue(resBf16);
        }

        // x2 init: x2[h] = postVals[h] * hOutNorm  (V, overlaps with batch 0 MTE2)
        {
            LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
            LocalTensor<float> hOut = bufHOutFp32_.Get<float>();
            for (uint32_t h = 0; h < N; ++h)
                Muls(x2[h * dimTile_], hOut, postVals[h], tD);
        }

        // Shared 2-batch residual axpy tail (noinline helper, see class decl).
        Apply2BatchResidualAxpyTail(rBaseTile, tD, combVals);

        // Stage 2 boundary: x2 → dtype → fp32
        LocalTensor<float> x2Fp32 = bufX2Fp32_.Get<float>();
        for (uint32_t n = 0; n < N; ++n)
            QuantizeRoundTrip(x2Fp32[n * dimTile_], tD);

        if (useRmsMid_) {
            // Accumulate total sumSq across ALL heads for N*D RMSNorm_mid
            for (uint32_t n = 0; n < N; ++n)
                totalSumSqMid += ReduceSumSq(x2Fp32[n * dimTile_], tD);
            PipeBarrier<PIPE_V>();
        } else {
            sumSqPre += ComputeSumSqAllHeads(tD);
            ComputePhiMatmulTile(tD, tOff, xHatVals);
            PipeBarrier<PIPE_V>();
        }
        // Write x2 to workspace for later phases
        for (uint32_t n = 0; n < N; ++n) {
            DataCopyExtParams wp = {1, 0, 0, 0, 0};
            wp.blockLen = tD * sizeof(float);
            DataCopyPad(workspaceGm_[pairWsBase_ + (uint64_t)n * D + tOff],
                        x2Fp32[n * dimTile_], wp);
        }
        PipeBarrier<PIPE_ALL>();
    }

    // RMSNorm_mid second pass (if enabled): single invRms over N*D
    if (useRmsMid_) {
        float invRmsMid = 1.0f / sqrt(totalSumSqMid * invK_ + normEps_);

        for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
            uint32_t tD = GetTileD(tile), tOff = tile * dimTile_;
            LocalTensor<float> x2Fp32 = bufX2Fp32_.Get<float>();
            for (uint32_t n = 0; n < N; ++n) {
                DataCopyExtParams rp = {1, 0, 0, 0, 0};
                rp.blockLen = tD * sizeof(float);
                DataCopyPadExtParams<float> rpad = {false, 0, 0, 0};
                DataCopyPad(x2Fp32[n * dimTile_],
                            workspaceGm_[pairWsBase_ + (uint64_t)n * D + tOff], rp, rpad);
            }
            LocalTensor<float> gamma2 = bufGamma2Fp32_.Get<float>();

            // gamma_2 is [N*D]: load gamma_2[n*D+tOff : n*D+tOff+tD] for each head
            for (uint32_t n = 0; n < N; ++n) {
                LoadGammaInto(gamma2, gamma2Gm_, n * D + tOff, tD);
                SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
                WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);
                ApplyScaleAndGamma(x2Fp32[n * dimTile_], invRmsMid, gamma2, tD);
                PipeBarrier<PIPE_ALL>();
            }

            // Stage 2.5 boundary: x2 → dtype → fp32
            for (uint32_t n = 0; n < N; ++n)
                QuantizeRoundTrip(x2Fp32[n * dimTile_], tD);

            sumSqPre += ComputeSumSqAllHeads(tD);
            ComputePhiMatmulTile(tD, tOff, xHatVals);

            PipeBarrier<PIPE_V>();
            for (uint32_t n = 0; n < N; ++n) {
                DataCopyExtParams wp = {1, 0, 0, 0, 0};
                wp.blockLen = tD * sizeof(float);
                DataCopyPad(workspaceGm_[pairWsBase_ + (uint64_t)n * D + tOff],
                            x2Fp32[n * dimTile_], wp);
            }
            PipeBarrier<PIPE_ALL>();
        }
    }
}

// =================================================================
//  PhaseC_HInAndX2Output (single-core)
// =================================================================
template<typename T>
__aicore__ inline void KernelMhcSandwichNormPostPreonly<T>::PhaseC_HInAndX2Output(
    uint32_t token, const float (&gate)[N_MAX], float &sumSq1)
{
    const uint32_t N = N_, D = D_;
    bool x2InWs = (dimLoop_ > 1);

    for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
        uint32_t tD = GetTileD(tile), tOff = tile * dimTile_;
        LocalTensor<float> x2Fp32 = bufX2Fp32_.Get<float>();

        if (x2InWs) {
            for (uint32_t n = 0; n < N; ++n) {
                DataCopyExtParams rp = {1, 0, 0, 0, 0};
                rp.blockLen = tD * sizeof(float);
                DataCopyPadExtParams<float> rpad = {false, 0, 0, 0};
                DataCopyPad(x2Fp32[n * dimTile_],
                            workspaceGm_[pairWsBase_ + (uint64_t)n * D + tOff], rp, rpad);
            }
            PipeBarrier<PIPE_ALL>();
        }

        // Write x2_out
        uint64_t x2OutBase = (uint64_t)token * N * D + tOff;
        for (uint32_t n = 0; n < N; ++n)
            WriteX2OutTile(x2Fp32[n * dimTile_], x2OutBase + (uint64_t)n * D, tD);

        // Compute h_in
        LocalTensor<float> outFp32 = bufOutFp32_.Get<float>();
        Muls(outFp32, x2Fp32, gate[0], tD); PipeBarrier<PIPE_V>();
        for (uint32_t n = 1; n < N; ++n) {
            Axpy(outFp32, x2Fp32[n * dimTile_], gate[n], tD); PipeBarrier<PIPE_V>();
        }

        // Stage 3 boundary: h_in → dtype → fp32
        QuantizeRoundTrip(outFp32, tD);

        sumSq1 += ReduceSumSq(outFp32, tD);

        if (dimLoop_ > 1) {
            PipeBarrier<PIPE_V>();
            DataCopyExtParams wp = {1, 0, 0, 0, 0};
            wp.blockLen = tD * sizeof(float);
            DataCopyPad(workspaceGm_[pairWsBase_ + (uint64_t)N * D + tOff], outFp32, wp);
            PipeBarrier<PIPE_ALL>();
        }
    }
}

// =================================================================
//  PhaseD_RmsNorm1Output (single-core)
// =================================================================
template<typename T>
__aicore__ inline void KernelMhcSandwichNormPostPreonly<T>::PhaseD_RmsNorm1Output(
    uint32_t token, float invRms1)
{
    const uint32_t N = N_, D = D_;
    LocalTensor<float> gamma1Fp32 = bufGamma1Fp32_.Get<float>();

    for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
        uint32_t tD = GetTileD(tile), tOff = tile * dimTile_;
        LocalTensor<float> outFp32 = bufOutFp32_.Get<float>();

        if (dimLoop_ > 1) {
            if (tile > 0 && returnHInF32_) {
                PipeBarrier<PIPE_ALL>();
            }
            DataCopyExtParams rp = {1, 0, 0, 0, 0};
            rp.blockLen = tD * sizeof(float);
            DataCopyPadExtParams<float> rpad = {false, 0, 0, 0};
            DataCopyPad(outFp32, workspaceGm_[pairWsBase_ + (uint64_t)N * D + tOff], rp, rpad);
            PipeBarrier<PIPE_ALL>();
        }
        if (tile > 0) {
            LoadGammaInto(gamma1Fp32, gamma1Gm_, tOff, tD);
            SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
            WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);
        }
        ApplyScaleAndGamma(outFp32, invRms1, gamma1Fp32, tD);
        WriteHInPrimeTile(outFp32, (uint64_t)token * D + tOff, tD);
    }
}

// =================================================================
//  Single-core building blocks
// =================================================================
template<typename T>
__aicore__ inline float KernelMhcSandwichNormPostPreonly<T>::ComputeSumSqAllHeads(uint32_t tD)
{
    LocalTensor<float> x2  = bufX2Fp32_.Get<float>();
    LocalTensor<float> out = bufOutFp32_.Get<float>();
    LocalTensor<float> sq  = bufSqFp32_.Get<float>();
    const uint32_t N = N_;

    Mul(out, x2, x2, tD); PipeBarrier<PIPE_V>();
    for (uint32_t n = 1; n < N; ++n) {
        Mul(sq, x2[n*dimTile_], x2[n*dimTile_], tD); PipeBarrier<PIPE_V>();
        Add(out, out, sq, tD); PipeBarrier<PIPE_V>();
    }
    return ReduceSumPrecise(out, tD);
}

template<typename T>
__aicore__ inline void KernelMhcSandwichNormPostPreonly<T>::ComputePhiMatmulTile(
    uint32_t tD, uint32_t tOff, float (&xHatVals)[N_MAX])
{
    LocalTensor<float> x2  = bufX2Fp32_.Get<float>();
    LocalTensor<float> out = bufOutFp32_.Get<float>();
    LocalTensor<float> sq  = bufSqFp32_.Get<float>();
    const uint32_t N = N_, D = D_;

    DataCopyExtParams cpP = {1, 0, 0, 0, 0};
    cpP.blockLen = tD * sizeof(float);
    DataCopyPadExtParams<float> cpPad = {true, 0, 0, 0};

    for (uint32_t o = 0; o < phiRows_; ++o) {
        {
            LocalTensor<float> phi = bufPhiFp32_.AllocTensor<float>();
            DataCopyPad(phi, phiGm_[(uint64_t)o * phiCols_ + tOff], cpP, cpPad);
            bufPhiFp32_.EnQue(phi); phi = bufPhiFp32_.DeQue<float>();
            Mul(out, x2, phi, tD); PipeBarrier<PIPE_V>();
            bufPhiFp32_.FreeTensor(phi);
        }
        for (uint32_t n = 1; n < N; ++n) {
            uint64_t phiOff = (uint64_t)o * phiCols_ + (uint64_t)n * D + tOff;
            LocalTensor<float> phi = bufPhiFp32_.AllocTensor<float>();
            DataCopyPad(phi, phiGm_[phiOff], cpP, cpPad);
            bufPhiFp32_.EnQue(phi); phi = bufPhiFp32_.DeQue<float>();
            Mul(sq, x2[n*dimTile_], phi, tD); PipeBarrier<PIPE_V>();
            Add(out, out, sq, tD); PipeBarrier<PIPE_V>();
            bufPhiFp32_.FreeTensor(phi);
        }
        xHatVals[o] += ReduceSumPrecise(out, tD);
    }
}

template<typename T>
__aicore__ inline void KernelMhcSandwichNormPostPreonly<T>::ApplyRmsNormMidSingleTile(
    uint32_t tD)
{
    LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
    const uint32_t N = N_, D = D_;

    // Compute total sumSq across ALL N heads (N*D RMSNorm)
    float totalSumSqMid = 0.0f;
    for (uint32_t n = 0; n < N; ++n)
        totalSumSqMid += ReduceSumSq(x2[n * dimTile_], tD);

    // Single invRms using invK_ = 1/(N*D)
    float invRmsMid = 1.0f / sqrt(totalSumSqMid * invK_ + normEps_);
    PipeBarrier<PIPE_ALL>();

    // gamma_2 is [N*D]: load gamma_2[n*D : (n+1)*D] for each head
    LocalTensor<float> gamma2 = bufGamma2Fp32_.Get<float>();
    for (uint32_t n = 0; n < N; ++n) {
        LoadGammaInto(gamma2, gamma2Gm_, n * D, tD);
        SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
        WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);
        ApplyScaleAndGamma(x2[n * dimTile_], invRmsMid, gamma2, tD);
        PipeBarrier<PIPE_ALL>();
    }
}

#endif // MHC_SANDWICH_NORM_POST_PREONLY_SINGLECORE_H_

