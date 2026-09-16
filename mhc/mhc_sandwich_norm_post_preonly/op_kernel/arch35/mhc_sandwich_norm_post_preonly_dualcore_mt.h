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
// mhc_sandwich_norm_post_preonly_dualcore_mt.h -- multi-tile dual-core path (D=5120)
// v2: QuantizeRoundTrip at stage boundaries, N*D RMSNorm_mid, ReduceSumPrecise
//
// x2 storage in UB (no GM workspace for tile spill):
//   tile 0: head0 -> bufGamma1Fp32_, head1 -> bufGamma2Fp32_
//   tile 1: head0..1 -> bufX2Fp32_
//
// h_in storage in UB:
//   tile 0 -> bufHOutFp32_, tile 1 -> bufOutFp32_
//
// RMSNorm_mid uses redundant computation: each core computes ALL N heads'
// sumSq independently using already-loaded x1 and residual data, avoiding
// the need for cross-core synchronization.
// ========================================================================
#ifndef MHC_SANDWICH_NORM_POST_PREONLY_DUALCORE_MT_H_
#define MHC_SANDWICH_NORM_POST_PREONLY_DUALCORE_MT_H_

template<typename T>
__aicore__ inline void KernelMhcSandwichNormPostPreonly<T>::ProcessTokenDualCoreMultiTile(uint32_t token)
{
    const uint32_t D = D_, N = N_, myN = myHeadCount_, hStart = headStart_;
    float postVals[N_MAX], combVals[N_MAX * N_MAX];
    float xHatVals[N_MAX], gate[N_MAX];
    float totalSumSqMid = 0.0f;
    for (uint32_t i = 0; i < N; ++i) { xHatVals[i] = 0.0f; }

    LoadTokenWeights(token, postVals, combVals);

    // ===== Pass 1: RMSNorm_0 sum-of-squares =====
    float sumSq0 = 0.0f;
    for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
        uint32_t tD = GetTileD(tile);
        LoadHOutAndCast((uint64_t)token * D + tile * dimTile_, tD);
        sumSq0 += ReduceSumSq(bufHOutFp32_.Get<float>(), tD);
    }
    float invRms0 = ComputeInvRms(sumSq0, invD_);

    // ===== Pass 2: Normalize + MHC_Post + store x2 in UB =====
    // Also accumulate totalSumSqMid for RMSNorm_mid (redundant computation of
    // partner's heads to avoid cross-core sync)
    for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
        uint32_t tD = GetTileD(tile), tOff = tile * dimTile_;

        LoadHOutAndCast((uint64_t)token * D + tOff, tD);

        // Load gamma_0 for this tile
        LocalTensor<float> gammaFp32 = bufGamma0Fp32_.Get<float>();
        LoadGammaInto(gammaFp32, gamma0Gm_, tOff, tD);

        SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
        WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);

        ApplyScaleAndGamma(bufHOutFp32_.Get<float>(), invRms0, gammaFp32, tD);
        // Stage 1 boundary: x1 → dtype → fp32
        QuantizeRoundTrip(bufHOutFp32_.Get<float>(), tD);

        LoadResidualAndCast((uint64_t)token * N * D + tOff, tD);
        ComputeX2MyHeads(postVals, combVals, tD);

        // Stage 2 boundary: quantize x2 per head
        {
            LocalTensor<float> x2Fp32 = bufX2Fp32_.Get<float>();
            for (uint32_t hi = 0; hi < myN; ++hi)
                QuantizeRoundTrip(x2Fp32[hi * dimTile_], tD);
        }

        // Accumulate sumSqMid for RMSNorm_mid (my heads + partner heads redundant)
        if (useRmsMid_) {
            LocalTensor<float> x2Fp32 = bufX2Fp32_.Get<float>();
            // My N/2 heads
            for (uint32_t hi = 0; hi < myN; ++hi) {
                totalSumSqMid += ReduceSumSq(x2Fp32[hi * dimTile_], tD);
            } 

            // Partner's N/2 heads (redundant computation, no sync needed)
            // x1 is in bufHOutFp32_, residual is in bufResFp32_
            uint32_t partHStart = (coreRole_ == 0) ? myN : 0;
            LocalTensor<float> tmpBuf = bufOutFp32_.Get<float>();
            LocalTensor<float> hOutFp32 = bufHOutFp32_.Get<float>();
            LocalTensor<float> resFp32 = bufResFp32_.Get<float>();
            for (uint32_t ph = 0; ph < myN; ++ph) {
                uint32_t h = partHStart + ph;
                Muls(tmpBuf, hOutFp32, postVals[h], tD); PipeBarrier<PIPE_V>();
                for (uint32_t j = 0; j < N; ++j) {
                    Axpy(tmpBuf, resFp32[j * dimTile_], combVals[j * N + h], tD);
                    PipeBarrier<PIPE_V>();
                }
                // Quantize partner x2 (matches partner's Stage 2 boundary)
                QuantizeRoundTrip(tmpBuf, tD);
                totalSumSqMid += ReduceSumSq(tmpBuf, tD);
            }
        }

        // Store x2 into UB slots
        if (tile == 0) {
            LocalTensor<float> x2Fp32 = bufX2Fp32_.Get<float>();
            Adds(bufGamma1Fp32_.Get<float>(), x2Fp32[0], 0.0f, tD);
            PipeBarrier<PIPE_V>();
            Adds(bufGamma2Fp32_.Get<float>(), x2Fp32[dimTile_], 0.0f, tD);
            PipeBarrier<PIPE_V>();
        }
        // tile 1: x2 stays in bufX2Fp32_
    }

    // ===== [Optional] RMSNorm_mid (apply normalization) =====
    if (useRmsMid_) {
        float invRmsMid = 1.0f / sqrt(totalSumSqMid * invK_ + normEps_);

        for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
            uint32_t tD = GetTileD(tile), tOff = tile * dimTile_;
            for (uint32_t hi = 0; hi < myN; ++hi) {
                // gamma_2 is [N*D]: load gamma_2[(hStart+hi)*D + tOff]
                LocalTensor<float> gammaFp32 = bufGamma0Fp32_.Get<float>();
                LoadGammaInto(gammaFp32, gamma2Gm_, (hStart + hi) * D + tOff, tD);

                SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
                WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);

                ApplyScaleAndGamma(GetX2HeadTile(tile, hi), invRmsMid, gammaFp32, tD);
                PipeBarrier<PIPE_ALL>();
            }
        }

        // Stage 2.5 boundary: quantize x2 per head per tile
        for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
            uint32_t tD = GetTileD(tile);
            for (uint32_t hi = 0; hi < myN; ++hi)
                QuantizeRoundTrip(GetX2HeadTile(tile, hi), tD);
        }
    }

    // ===== Phase C: sumSqPre + phi matmul =====
    float mySumSqPre = 0.0f;
    for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
        uint32_t tD = GetTileD(tile), tOff = tile * dimTile_;
        LocalTensor<float> outFp32 = bufOutFp32_.Get<float>();
        LocalTensor<float> sqBuf   = bufSqFp32_.Get<float>();

        // sumSqPre partial
        { LocalTensor<float> x2h0 = GetX2HeadTile(tile, 0);
          Mul(outFp32, x2h0, x2h0, tD); PipeBarrier<PIPE_V>(); }
        for (uint32_t hi = 1; hi < myN; ++hi) {
            LocalTensor<float> x2hi = GetX2HeadTile(tile, hi);
            Mul(sqBuf, x2hi, x2hi, tD); PipeBarrier<PIPE_V>();
            Add(outFp32, outFp32, sqBuf, tD); PipeBarrier<PIPE_V>();
        }
        mySumSqPre += ReduceSumPrecise(outFp32, tD);

        // phi matmul partial
        DataCopyExtParams cpP = {1, 0, 0, 0, 0};
        cpP.blockLen = tD * sizeof(float);
        DataCopyPadExtParams<float> cpPad = {true, 0, 0, 0};
        for (uint32_t o = 0; o < phiRows_; ++o) {
            for (uint32_t hi = 0; hi < myN; ++hi) {
                uint64_t phiOff = (uint64_t)o * phiCols_ + (uint64_t)(hStart + hi) * D + tOff;
                LocalTensor<float> phiBuf = bufPhiFp32_.AllocTensor<float>();
                DataCopyPad(phiBuf, phiGm_[phiOff], cpP, cpPad);
                bufPhiFp32_.EnQue(phiBuf);
                phiBuf = bufPhiFp32_.DeQue<float>();
                LocalTensor<float> x2hi = GetX2HeadTile(tile, hi);
                if (hi == 0) { Mul(outFp32, x2hi, phiBuf, tD); PipeBarrier<PIPE_V>(); }
                else { Mul(sqBuf, x2hi, phiBuf, tD); PipeBarrier<PIPE_V>();
                       Add(outFp32, outFp32, sqBuf, tD); PipeBarrier<PIPE_V>(); }
                bufPhiFp32_.FreeTensor(phiBuf);
            }
            xHatVals[o] += ReduceSumPrecise(outFp32, tD);
        }
    }

    // ===== SYNC + Gate =====
    float myXHat[N_MAX];
    for (uint32_t o = 0; o < phiRows_; ++o) myXHat[o] = xHatVals[o];

    uint32_t myX2WsSlot = WS_HIN_BASE + coreRole_ * myN * D;
    DualCoreSyncExchange(mySumSqPre, myXHat, mySumSqPre, xHatVals);

    // Write my x2 to workspace per tile
    for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
        uint32_t tD = GetTileD(tile), tOff = tile * dimTile_;
        for (uint32_t hi = 0; hi < myN; ++hi) {
            LocalTensor<float> x2hi = GetX2HeadTile(tile, hi);
            DataCopyExtParams wp = {1, 0, 0, 0, 0};
            wp.blockLen = tD * sizeof(float);
            DataCopyPad(workspaceGm_[pairWsBase_ + myX2WsSlot + (uint64_t)hi * D + tOff],
                        x2hi, wp);
        }
    }

    float totalSumSqPre;
    SyncAndReadPartner(mySumSqPre, myXHat, totalSumSqPre, xHatVals);
    ComputeGateValues(totalSumSqPre, xHatVals, gate);

    // ===== Phase D: h_in + x2_out + RMSNorm_1 (2-pass) =====
    uint32_t partnerHStart = (coreRole_ == 0) ? myN : 0;
    float sumSq1 = 0.0f;

    // Pass 1: compute h_in per tile, accumulate sumSq1
    for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
        uint32_t tD = GetTileD(tile), tOff = tile * dimTile_;
        // h_in dest: tile0->hOutFp32, tile1->outFp32
        LocalTensor<float> hInDst = (tile == 0) ?
            bufHOutFp32_.Get<float>() : bufOutFp32_.Get<float>();

        { LocalTensor<float> x2h0 = GetX2HeadTile(tile, 0);
          Muls(hInDst, x2h0, gate[hStart], tD); PipeBarrier<PIPE_V>(); }
        for (uint32_t hi = 1; hi < myN; ++hi) {
            LocalTensor<float> x2hi = GetX2HeadTile(tile, hi);
            Axpy(hInDst, x2hi, gate[hStart + hi], tD); PipeBarrier<PIPE_V>();
        }

        // Write x2_out for my heads
        uint64_t x2OutBase = (uint64_t)token * N * D + tOff;
        for (uint32_t hi = 0; hi < myN; ++hi)
            WriteX2OutTile(GetX2HeadTile(tile, hi),
                           x2OutBase + (uint64_t)(hStart + hi) * D, tD);

        // Add partner x2 contribution
        LocalTensor<float> sqBuf = bufSqFp32_.Get<float>();
        for (uint32_t hi = 0; hi < myN; ++hi) {
            ReadPartnerX2Tile(sqBuf, hi, tOff, tD);
            Axpy(hInDst, sqBuf, gate[partnerHStart + hi], tD);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(evtVtoMte2_);
            WaitFlag<HardEvent::V_MTE2>(evtVtoMte2_);
        }

        // Stage 3 boundary: h_in → dtype → fp32
        QuantizeRoundTrip(hInDst, tD);

        sumSq1 += ReduceSumSq(hInDst, tD);
    }

    // Pass 2: RMSNorm_1 + output (core0 only)
    float invRms1 = ComputeInvRms(sumSq1, invD_);
    if (coreRole_ == 0) {
        for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
            uint32_t tD = GetTileD(tile), tOff = tile * dimTile_;
            LocalTensor<float> hInSrc = (tile == 0) ?
                bufHOutFp32_.Get<float>() : bufOutFp32_.Get<float>();
            LocalTensor<float> gammaFp32 = bufGamma0Fp32_.Get<float>();
            LoadGammaInto(gammaFp32, gamma1Gm_, tOff, tD);
            PipeBarrier<PIPE_ALL>();
            ApplyScaleAndGamma(hInSrc, invRms1, gammaFp32, tD);
            WriteHInPrimeTile(hInSrc, (uint64_t)token * D + tOff, tD);
        }
    }
}

#endif // MHC_SANDWICH_NORM_POST_PREONLY_DUALCORE_MT_H_

