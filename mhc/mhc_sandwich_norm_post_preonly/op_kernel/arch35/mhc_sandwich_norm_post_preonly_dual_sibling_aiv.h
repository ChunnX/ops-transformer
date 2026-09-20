/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the LICENSE.
 */

#ifndef MHC_SANDWICH_NORM_POST_PREONLY_DUAL_SIBLING_AIV_H_
#define MHC_SANDWICH_NORM_POST_PREONLY_DUAL_SIBLING_AIV_H_

template<typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcSandwichNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32,
    DUAL_SIBLING>::ProcessTokenDualSiblingAiv(uint32_t token)
{
    float postVals[N_MAX];
    float combVals[N_MAX * N_MAX];
    float sumSqHalf[2] = {0.0f, 0.0f};
    float dotHalf[2][N_MAX] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    float midSqHalf[2] = {0.0f, 0.0f};
    LoadTokenWeights(token, postVals, combVals);

    // 两个lane按完全相同的分块和顺序冗余执行R1。
    float sumSq0 = 0.0f;
    for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dimTile_;
        LoadHOutAndCast((uint64_t)token * D_ + tOff, tD);
        sumSq0 += ReduceSumSq(bufHOutFp32_.Get<float>(), tD);
    }
    const float invRms0 = ComputeInvRms(sumSq0, invD_);

    // 两个lane冗余构造完整x2，但只写各自负责的展平半片。
    for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dimTile_;
        ComputeX2PostTile(token, tOff, tD, invRms0, postVals, combVals);
        CacheX2TileForLane(token, tOff, tD, !USE_RMS_MID);

        LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
        if constexpr (USE_RMS_MID) {
            // R3的归一化范围是完整N*D，因此该标量也在两个lane上按相同树冗余计算。
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
            AccumulateHalfPartials(tOff, tD, sumSqHalf, dotHalf, laneId_, laneId_ + 1U);
        }
    }

    if constexpr (USE_RMS_MID) {
        const float totalMidSq = midSqHalf[0] + midSqHalf[1];
        const float invRmsMid = 1.0f / sqrt(totalMidSq * invK_ + normEps_);
        for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
            const uint32_t tD = GetTileD(tile);
            const uint32_t tOff = tile * dimTile_;
            LocalTensor<float> gammaEven = bufGamma2Fp32_.Get<float>();
            LocalTensor<float> gammaOdd = bufGamma0Fp32_.Get<float>();
            LoadGammaInto(gammaEven, gamma2Gm_, tOff, tD);
            LoadX2CacheTile(tOff, tD);
            LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
            for (uint32_t n = 0; n < N_MAX; ++n) {
                LocalTensor<float> gammaCurrent = ((n & 1U) == 0) ? gammaEven : gammaOdd;
                LocalTensor<float> gammaNext = ((n & 1U) == 0) ? gammaOdd : gammaEven;
                SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
                WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);
                if (n + 1 < N_MAX) {
                    LoadGammaInto(gammaNext, gamma2Gm_, (n + 1) * D_ + tOff, tD);
                }
                ApplyScaleAndGamma(x2[n * dimTile_], invRmsMid, gammaCurrent, tD);
                if (n + 2 < N_MAX) {
                    FenceVToMte2();
                }
            }
            CacheX2TileForLane(token, tOff, tD, true);
            AccumulateHalfPartials(tOff, tD, sumSqHalf, dotHalf, laneId_, laneId_ + 1U);
        }
    }

    // follower把五个有效partial写入独占的32B槽；leader的partial0始终留在本核。
    LocalTensor<float> partialBuf = bufScalar_.Get<float>();
    if (laneId_ == 1U) {
        partialBuf.SetValue(0, sumSqHalf[1]);
        for (uint32_t o = 0; o < N_MAX; ++o) {
            partialBuf.SetValue(o + 1U, dotHalf[1][o]);
        }
        for (uint32_t i = N_MAX + 1U; i < PARTIAL_SLOT_ELEMS; ++i) {
            partialBuf.SetValue(i, 0.0f);
        }
        SetFlag<HardEvent::S_MTE3>(evtStoMte3_);
        WaitFlag<HardEvent::S_MTE3>(evtStoMte3_);
        DataCopy(partial1Gm_[(uint64_t)token * PARTIAL_SLOT_ELEMS], partialBuf, PARTIAL_SLOT_ELEMS);
    }

    // 两个lane均无条件执行一次mode1 Set/Wait；该等待不替代本核流水事件。
    MhcPairReadySync();
    if (laneId_ == 1U) {
        return;
    }

    DataCopy(partialBuf, partial1Gm_[(uint64_t)token * PARTIAL_SLOT_ELEMS], PARTIAL_SLOT_ELEMS);
    SetFlag<HardEvent::MTE2_S>(evtMte2toS_);
    WaitFlag<HardEvent::MTE2_S>(evtMte2toS_);
    sumSqHalf[1] = partialBuf.GetValue(0);
    for (uint32_t o = 0; o < N_MAX; ++o) {
        dotHalf[1][o] = partialBuf.GetValue(o + 1U);
    }

    const float totalSumSqPre = sumSqHalf[0] + sumSqHalf[1];
    float xHatVals[N_MAX];
    for (uint32_t o = 0; o < N_MAX; ++o) {
        xHatVals[o] = dotHalf[0][o] + dotHalf[1][o];
    }
    float gate[N_MAX];
    ComputeGateValues(totalSumSqPre, xHatVals, gate);

    // 会合后只有leader继续执行R6、R7和最终输出。
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

#endif // MHC_SANDWICH_NORM_POST_PREONLY_DUAL_SIBLING_AIV_H_
