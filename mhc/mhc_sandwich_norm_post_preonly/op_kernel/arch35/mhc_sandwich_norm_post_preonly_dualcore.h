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
// mhc_sandwich_norm_post_preonly_dualcore.h — single-tile dual-core path (D<=2560)
// v2: QuantizeRoundTrip at stage boundaries, N*D RMSNorm_mid, ReduceSumPrecise
// ========================================================================
#ifndef MHC_SANDWICH_NORM_POST_PREONLY_DUALCORE_H_
#define MHC_SANDWICH_NORM_POST_PREONLY_DUALCORE_H_

template<typename T>
__aicore__ inline void KernelMhcSandwichNormPostPreonly<T>::ProcessTokenDualCore(uint32_t token)
{
    const uint32_t D = D_, N = N_, myN = myHeadCount_, hStart = headStart_;
    const uint32_t tD = dimTile_;
    float postVals[N_MAX], combVals[N_MAX * N_MAX], xHatVals[N_MAX], gate[N_MAX];
    for (uint32_t i = 0; i < N; ++i) xHatVals[i] = 0.0f;

    LoadTokenWeights(token, postVals, combVals);

    // ---- Phase A: RMSNorm_0 (redundant on both cores) ----
    LoadHOutAndCast((uint64_t)token * D, tD);

    // Start residual DMA in parallel with RMSNorm_0 compute
    {
        LocalTensor<T> resBf16 = bufResidualBf16_.AllocTensor<T>();
        uint64_t rBase = (uint64_t)token * N * D;
        for (uint32_t n = 0; n < N; ++n)
            DataCopy(resBf16[n * dimTile_], residualGm_[rBase + (uint64_t)n * D], tD);
        bufResidualBf16_.EnQue(resBf16);
    }

    LocalTensor<float> hOutFp32 = bufHOutFp32_.Get<float>();
    float invRms0 = ComputeInvRms(ReduceSumSq(hOutFp32, tD), invD_);
    ApplyScaleAndGamma(hOutFp32, invRms0, bufGamma0Fp32_.Get<float>(), tD);
    // Stage 1 boundary: x1 → dtype → fp32
    QuantizeRoundTrip(hOutFp32, tD);

    // Finish residual cast
    {
        LocalTensor<T> resBf16 = bufResidualBf16_.DeQue<T>();
        LocalTensor<float> resFp32 = bufResFp32_.Get<float>();
        for (uint32_t n = 0; n < N; ++n)
            Cast(resFp32[n * dimTile_], resBf16[n * dimTile_], RoundMode::CAST_NONE, tD);
        PipeBarrier<PIPE_V>();
        bufResidualBf16_.FreeTensor(resBf16);
    }

    // ---- Phase B: MHC_Post (my N/2 heads) ----
    ComputeX2MyHeads(postVals, combVals, tD);

    // Stage 2 boundary: x2 → dtype → fp32
    {
        LocalTensor<float> x2Tmp = bufX2Fp32_.Get<float>();
        for (uint32_t hi = 0; hi < myN; ++hi)
            QuantizeRoundTrip(x2Tmp[hi * dimTile_], tD);
    }

    // ---- Optional RMSNorm_mid (over N*D: redundant computation of partner heads' sumSq) ----
    if (useRmsMid_) {
        LocalTensor<float> x2Fp32 = bufX2Fp32_.Get<float>();
        LocalTensor<float> hOutFp32 = bufHOutFp32_.Get<float>();  // still contains x1
        LocalTensor<float> resFp32 = bufResFp32_.Get<float>();    // still contains residual

        // Compute sumSq for my N/2 heads
        float totalSumSqMid = 0.0f;
        for (uint32_t hi = 0; hi < myN; ++hi)
            totalSumSqMid += ReduceSumSq(x2Fp32[hi * dimTile_], tD);

        // Compute sumSq for partner's N/2 heads (redundant computation, no sync needed)
        // Partner x2 also needs to be computed from quantized x1
        uint32_t partHStart = (coreRole_ == 0) ? myN : 0;
        LocalTensor<float> tmpBuf = bufOutFp32_.Get<float>();
        for (uint32_t ph = 0; ph < myN; ++ph) {
            uint32_t h = partHStart + ph;
            Muls(tmpBuf, hOutFp32, postVals[h], tD); PipeBarrier<PIPE_V>();
            for (uint32_t j = 0; j < N; ++j) {
                Axpy(tmpBuf, resFp32[j * dimTile_], combVals[j * N + h], tD);
                PipeBarrier<PIPE_V>();
            }
            // Quantize partner x2 before sumSq (same as my x2 was quantized)
            QuantizeRoundTrip(tmpBuf, tD);
            totalSumSqMid += ReduceSumSq(tmpBuf, tD);
        }

        float invRmsMid = 1.0f / sqrt(totalSumSqMid * invK_ + normEps_);
        PipeBarrier<PIPE_ALL>();

        // Apply: gamma_2 is [N*D], load gamma_2[hStart*D : (hStart+hi+1)*D] per head
        LocalTensor<float> gamma2 = bufGamma2Fp32_.Get<float>();
        for (uint32_t hi = 0; hi < myN; ++hi) {
            LoadGammaInto(gamma2, gamma2Gm_, (hStart + hi) * D, tD);

            SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
            WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);

            ApplyScaleAndGamma(x2Fp32[hi * dimTile_], invRmsMid, gamma2, tD);
            PipeBarrier<PIPE_ALL>();  // WAR: V-pipe must finish reading gamma2
                                      // before next iteration's MTE2 overwrites it
        }

        // Stage 2.5 boundary: x2 → dtype → fp32
        for (uint32_t hi = 0; hi < myN; ++hi)
            QuantizeRoundTrip(x2Fp32[hi * dimTile_], tD);
    }

    // ---- Phase C: sumSqPre + phi matmul ----
    LocalTensor<float> x2Fp32  = bufX2Fp32_.Get<float>();
    LocalTensor<float> outFp32 = bufOutFp32_.Get<float>();
    LocalTensor<float> sqBuf   = bufSqFp32_.Get<float>();

    Mul(outFp32, x2Fp32, x2Fp32, tD);
    PipeBarrier<PIPE_V>();
    for (uint32_t hi = 1; hi < myN; ++hi) {
        Mul(sqBuf, x2Fp32[hi * dimTile_], x2Fp32[hi * dimTile_], tD);
        PipeBarrier<PIPE_V>();
        Add(outFp32, outFp32, sqBuf, tD);
        PipeBarrier<PIPE_V>();
    }
    float mySumSqPre = ReduceSumPrecise(outFp32, tD);

    // phi matmul — double-buffer prefetch
    float myXHat[N_MAX];
    DataCopyExtParams cpP = {1, 0, 0, 0, 0};
    cpP.blockLen = tD * sizeof(float);
    DataCopyPadExtParams<float> cpPad = {true, 0, 0, 0};

    const uint32_t totalPhiIters = phiRows_ * myN;
    {
        uint64_t phiOff = (uint64_t)0 * phiCols_ + (uint64_t)hStart * D;
        LocalTensor<float> phiBuf = bufPhiFp32_.AllocTensor<float>();
        DataCopyPad(phiBuf, phiGm_[phiOff], cpP, cpPad);
        bufPhiFp32_.EnQue(phiBuf);
    }
    for (uint32_t iter = 0; iter < totalPhiIters; ++iter) {
        uint32_t o = iter / myN, hi = iter % myN;
        if (iter + 1 < totalPhiIters) {
            uint32_t nO = (iter+1)/myN, nHi = (iter+1)%myN;
            uint64_t nOff = (uint64_t)nO * phiCols_ + (uint64_t)(hStart + nHi) * D;
            LocalTensor<float> phiNext = bufPhiFp32_.AllocTensor<float>();
            DataCopyPad(phiNext, phiGm_[nOff], cpP, cpPad);
            bufPhiFp32_.EnQue(phiNext);
        }
        LocalTensor<float> phiCur = bufPhiFp32_.DeQue<float>();
        if (hi == 0) { Mul(outFp32, x2Fp32, phiCur, tD); PipeBarrier<PIPE_V>(); }
        else { Mul(sqBuf, x2Fp32[hi*dimTile_], phiCur, tD); PipeBarrier<PIPE_V>();
               Add(outFp32, outFp32, sqBuf, tD); PipeBarrier<PIPE_V>(); }
        bufPhiFp32_.FreeTensor(phiCur);
        if (hi == myN - 1) {
            myXHat[o] = ReduceSumPrecise(outFp32, tD);
        }
    }

    // ---- SYNC: scalars + x2 exchange ----
    uint32_t myX2WsSlot = WS_HIN_BASE + coreRole_ * myN * D;
    DualCoreSyncExchange(mySumSqPre, myXHat, mySumSqPre, xHatVals);
    WriteX2ToWorkspace(tD, 0, x2Fp32, myN, myX2WsSlot);
    float totalSumSqPre;
    SyncAndReadPartner(mySumSqPre, myXHat, totalSumSqPre, xHatVals);

    // ---- ComputeGate ----
    ComputeGateValues(totalSumSqPre, xHatVals, gate);

    // ---- Phase D: h_in (my + partner heads) ----
    Muls(outFp32, x2Fp32, gate[hStart], tD);
    PipeBarrier<PIPE_V>();
    for (uint32_t hi = 1; hi < myN; ++hi) {
        Axpy(outFp32, x2Fp32[hi * dimTile_], gate[hStart + hi], tD);
        PipeBarrier<PIPE_V>();
    }

    // Partner x2 reads + my x2_out writes (MTE2/MTE3 overlap)
    uint32_t partnerHStart = (coreRole_ == 0) ? myN : 0;
    uint64_t x2OutBase = (uint64_t)token * N * D;
    for (uint32_t hi = 0; hi < myN; ++hi) {
        ReadPartnerX2Tile(sqBuf, hi, 0, tD);
        WriteX2OutTile(x2Fp32[hi * dimTile_], x2OutBase + (uint64_t)(hStart + hi) * D, tD);
        Axpy(outFp32, sqBuf, gate[partnerHStart + hi], tD);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(evtVtoMte2_);
        WaitFlag<HardEvent::V_MTE2>(evtVtoMte2_);
    }

    // ---- Phase E: RMSNorm_1 ----
    // Flush MTE3 from WriteX2OutTile — WAR with QuantizeRoundTrip
    // Stage 3 boundary: h_in → dtype → fp32
    QuantizeRoundTrip(outFp32, tD);
    float sumSq1 = ReduceSumSq(outFp32, tD);
    float invRms1 = ComputeInvRms(sumSq1, invD_);

    if (coreRole_ == 0) {
        ApplyScaleAndGamma(outFp32, invRms1, bufGamma1Fp32_.Get<float>(), tD);
        WriteHInPrimeTile(outFp32, (uint64_t)token * D, tD);
    }
}

#endif // MHC_SANDWICH_NORM_POST_PREONLY_DUALCORE_H_

