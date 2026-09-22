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

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING, bool D_SPLIT>
__aicore__ inline void KernelMhcSandwichNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32,
                                                        DUAL_SIBLING, D_SPLIT>::ProcessTokenDualSiblingAiv(uint32_t token)
{
    float postVals[N_MAX];
    float sumSqHalf[2] = {0.0f, 0.0f};
    float dotHalf[2][N_MAX] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    float midSqHalf[2] = {0.0f, 0.0f};
    CacheHOut((uint64_t)token * D_);
    CacheGamma0();
    const uint32_t headBegin = laneId_ * 2U;
    LoadTokenWeights(token, postVals, headBegin, headBegin + 2U);

    // 两个lane按完全相同的分块和顺序冗余执行R1。
    float sumSq0 = 0.0f;
    for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dimTile_;
        sumSq0 += ReduceSumSqFromCache(tOff, tD);
    }
    const float invRms0 = ComputeInvRms(sumSq0, invD_);

    // 两个lane分别构造自己负责的两个x2 head。
    for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dimTile_;
        ComputeX2PostTileForLane(token, tOff, tD, invRms0, postVals);
        CacheX2TileForLane(token, tOff, tD, !USE_RMS_MID, false);

        if constexpr (USE_RMS_MID) {
            // 每个lane只归约自己负责的两个head，循环结束后交换半和。
            LocalTensor<float> out = bufResFp32_.Get<float>()[N_MAX * dimTile_];
            LocalTensor<T> x2Cache = bufX2Cache_.Get<T>();
            for (uint32_t half = laneId_; half < laneId_ + 1U; ++half) {
                const uint32_t h0 = half * 2U;
                const uint32_t h1 = h0 + 1;
                MhcSandwichRegBase::PairSquareB16ToFp32(out, x2Cache[h0 * D_ + tOff],
                                                       x2Cache[h1 * D_ + tOff], tD);
                PipeBarrier<PIPE_V>();
                midSqHalf[half] += ReduceSumPrecise(out, tD);
            }
        } else {
            AccumulateHalfPartials(tOff, tD, sumSqHalf, dotHalf, laneId_, laneId_ + 1U);
        }
    }

    if constexpr (USE_RMS_MID) {
        LocalTensor<float> midBuf = bufScalar_.Get<float>();
        midBuf.SetValue(0, midSqHalf[laneId_]);
        for (uint32_t i = 1U; i < PARTIAL_SLOT_ELEMS; ++i) {
            midBuf.SetValue(i, 0.0f);
        }
        SetFlag<HardEvent::S_MTE3>(evtStoMte3_);
        WaitFlag<HardEvent::S_MTE3>(evtStoMte3_);
        const uint64_t ownSlot = ((uint64_t)token * 2U + laneId_) * PARTIAL_SLOT_ELEMS;
        DataCopy(partial1Gm_[ownSlot], midBuf, PARTIAL_SLOT_ELEMS);
        MhcPairMidReadySync();
        const uint64_t siblingSlot = ((uint64_t)token * 2U + (1U - laneId_)) * PARTIAL_SLOT_ELEMS;
        DataCopy(midBuf, partial1Gm_[siblingSlot], PARTIAL_SLOT_ELEMS);
        SetFlag<HardEvent::MTE2_S>(evtMte2toS_);
        WaitFlag<HardEvent::MTE2_S>(evtMte2toS_);
        midSqHalf[1U - laneId_] = midBuf.GetValue(0);
        const float totalMidSq = midSqHalf[0] + midSqHalf[1];
        const float invRmsMid = 1.0f / sqrt(totalMidSq * invK_ + normEps_);
        for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
            const uint32_t tD = GetTileD(tile);
            const uint32_t tOff = tile * dimTile_;
            LocalTensor<float> gammaEven = bufGamma2Fp32_.Get<float>();
            LocalTensor<float> gammaOdd = bufGamma0Fp32_.Get<float>();
            LoadGammaInto(gammaEven, gamma2Gm_, headBegin * D_ + tOff, tD);
            LocalTensor<T> x2Cache = bufX2Cache_.Get<T>();
            // 每个lane只执行自己负责的两个head，另一半由sibling并行完成。
            for (uint32_t n = headBegin; n < headBegin + 2U; ++n) {
                const uint32_t laneHead = n - headBegin;
                LocalTensor<float> gammaCurrent = ((laneHead & 1U) == 0) ? gammaEven : gammaOdd;
                LocalTensor<float> gammaNext = ((laneHead & 1U) == 0) ? gammaOdd : gammaEven;
                SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
                WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);
                if (n + 1U < headBegin + 2U) {
                    LoadGammaInto(gammaNext, gamma2Gm_, (n + 1) * D_ + tOff, tD);
                }
                MhcSandwichRegBase::CastB16ScaleAndMulGammaToB16(
                    x2Cache[n * D_ + tOff], x2Cache[n * D_ + tOff], gammaCurrent, invRmsMid, tD);
                PipeBarrier<PIPE_V>();
                if (n + 2U < headBegin + 2U) {
                    FenceVToMte2();
                }
            }
            CacheX2TileForLane(token, tOff, tD, true, false);
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
        const uint64_t followerSlot = ((uint64_t)token * 2U + 1U) * PARTIAL_SLOT_ELEMS;
        DataCopy(partial1Gm_[followerSlot], partialBuf, PARTIAL_SLOT_ELEMS);
    }

    // 两个lane均无条件执行一次mode1 Set/Wait；该等待不替代本核流水事件。
    MhcPairReadySync();
    if (laneId_ == 1U) {
        return;
    }

    // 发出本核MTE3完成事件，同时读取follower的partial1，用32B搬运隐藏事件等待。
    SetFlag<HardEvent::MTE3_V>(evtMte3toV_);
    const uint64_t followerSlot = ((uint64_t)token * 2U + 1U) * PARTIAL_SLOT_ELEMS;
    DataCopy(partialBuf, partial1Gm_[followerSlot], PARTIAL_SLOT_ELEMS);
    SetFlag<HardEvent::MTE2_S>(evtMte2toS_);
    WaitFlag<HardEvent::MTE3_V>(evtMte3toV_);
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
    ComputeGateValues(totalSumSqPre, xHatVals);

    // 会合后只有leader继续执行R6、R7和最终输出。leader的私有缓存仅包含
    // 自己完成R3的前两行，从已经同步写出的x2_out恢复follower负责的后两行。
    // follower负责的两个head在GM和UB中均连续，一次恢复完整2D区间，
    // 避免按tile、按head重复发出MTE2搬运指令。
    LocalTensor<T> x2Cache = bufX2Cache_.Get<T>();
    const uint64_t followerHeadOffset = (uint64_t)token * N_ * D_ + 2U * D_;
    DataCopy(x2Cache[2U * D_], x2OutGm_[followerHeadOffset], 2U * D_);
    SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
    WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);

    float sumSq1 = 0.0f;
    for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dimTile_;
        LocalTensor<float> hIn = bufResFp32_.Get<float>()[N_MAX * dimTile_];
        LocalTensor<T> x2Cache = bufX2Cache_.Get<T>()[tOff];
        LocalTensor<float> gateBuf = bufScalar_.Get<float>()[EXP_DIV_ELEM];
        MhcSandwichRegBase::MulsB16WithUbScalar(hIn, x2Cache, gateBuf, 0U, tD);
        PipeBarrier<PIPE_V>();
        for (uint32_t n = 1; n < N_MAX; ++n) {
            MhcSandwichRegBase::AxpyB16WithUbScalar(hIn, x2Cache[n * D_], gateBuf, n, tD);
            PipeBarrier<PIPE_V>();
        }
        sumSq1 += CacheHInAndReduceSumSq(tOff, tD, hIn);
    }

    StartCacheGamma1();
    const float invRms1 = ComputeInvRms(sumSq1, invD_);
    WaitCacheGamma1();
    for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dimTile_;
        LocalTensor<float> hIn = bufResFp32_.Get<float>()[N_MAX * dimTile_];
        LoadHInCacheTile(tOff, tD, hIn);
        LocalTensor<float> gamma1 = bufGamma2Fp32_.Get<float>();
        if (D_ > GAMMA1_CACHE_MAX_D) {
            LoadGammaInto(gamma1, gamma1Gm_, tOff, tD);
            SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
            WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);
        } else {
            gamma1 = gamma1[tOff];
        }
        ApplyScaleAndGamma(hIn, invRms1, gamma1, tD);
        if (D_ > GAMMA1_CACHE_MAX_D) {
            FenceVToMte2();
        }
        WriteHInPrimeTile(hIn, (uint64_t)token * D_ + tOff, tD);
    }
}

// D轴切分路径：每条lane只处理D的一半（dimLoop_/2个tile），但负责全部四个head。
// 四个全局归约阶段（R1/R3/R4R5/R7）经mode1握手与32B GM槽交换合并。相比head切分，
// 消除R2的residual冗余读取、leader的x2回读，并把R6/R7并行到两条lane。
template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING, bool D_SPLIT>
__aicore__ inline void KernelMhcSandwichNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32,
                                                        DUAL_SIBLING, D_SPLIT>::ProcessTokenDSplitAiv(uint32_t token)
{
    const uint32_t halfLoop = dimLoop_ / 2U;
    const uint32_t tileBegin = laneId_ * halfLoop;
    const uint32_t dSeg = tileBegin * dimTile_; // 本lane的D偏移（元素数）
    float postVals[N_MAX];
    float sumSqHalf[2] = {0.0f, 0.0f};
    float dotHalf[2][N_MAX] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    float midSqHalf[2] = {0.0f, 0.0f};

    CacheHOutSegment((uint64_t)token * D_ + dSeg, dSeg, D_ / 2U);
    CacheGamma0();
    LoadTokenWeights(token, postVals);

    // R1：归约本lane的D/2段平方和，交换合并得到完整D的sumSq0。
    float sumSq0 = 0.0f;
    for (uint32_t tile = tileBegin; tile < tileBegin + halfLoop; ++tile) {
        const uint32_t tOff = tile * dimTile_;
        sumSq0 += ReduceSumSqFromCache(tOff, dimTile_);
    }
    {
        LocalTensor<float> buf = bufScalar_.Get<float>();
        buf.SetValue(0, sumSq0);
        for (uint32_t i = 1U; i < PARTIAL_SLOT_ELEMS; ++i) {
            buf.SetValue(i, 0.0f);
        }
        SetFlag<HardEvent::S_MTE3>(evtStoMte3_);
        WaitFlag<HardEvent::S_MTE3>(evtStoMte3_);
        DataCopy(partial1Gm_[((uint64_t)token * 2U + laneId_) * PARTIAL_SLOT_ELEMS], buf, PARTIAL_SLOT_ELEMS);
        MhcDSplitSync(DSPLIT_R1_FLAG);
        DataCopy(buf, partial1Gm_[((uint64_t)token * 2U + (1U - laneId_)) * PARTIAL_SLOT_ELEMS], PARTIAL_SLOT_ELEMS);
        SetFlag<HardEvent::MTE2_S>(evtMte2toS_);
        WaitFlag<HardEvent::MTE2_S>(evtMte2toS_);
        sumSq0 += buf.GetValue(0);
    }
    const float invRms0 = ComputeInvRms(sumSq0, invD_);

    // R2/R3：计算本lane D/2段的四head x2，归约mid平方和。
    for (uint32_t tile = tileBegin; tile < tileBegin + halfLoop; ++tile) {
        const uint32_t tOff = tile * dimTile_;
        ComputeX2PostTile(token, tOff, dimTile_, invRms0, postVals);
        CacheX2Tile(token, tOff, dimTile_, !USE_RMS_MID, false);

        if constexpr (USE_RMS_MID) {
            LocalTensor<float> out = bufResFp32_.Get<float>()[N_MAX * dimTile_];
            LocalTensor<T> x2Cache = bufX2Cache_.Get<T>();
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t h0 = half * 2;
                const uint32_t h1 = h0 + 1;
                MhcSandwichRegBase::PairSquareB16ToFp32(out, x2Cache[h0 * D_ + tOff],
                                                       x2Cache[h1 * D_ + tOff], dimTile_);
                PipeBarrier<PIPE_V>();
                midSqHalf[half] += ReduceSumPrecise(out, dimTile_);
            }
        } else {
            AccumulateHalfPartials(tOff, dimTile_, sumSqHalf, dotHalf, 0U, 2U, false);
        }
    }

    // R3：交换合并mid平方和，得到完整N*D的invRmsMid。
    if constexpr (USE_RMS_MID) {
        float totalMidSq = midSqHalf[0] + midSqHalf[1];
        {
            LocalTensor<float> buf = bufScalar_.Get<float>();
            buf.SetValue(0, totalMidSq);
            for (uint32_t i = 1U; i < PARTIAL_SLOT_ELEMS; ++i) {
                buf.SetValue(i, 0.0f);
            }
            SetFlag<HardEvent::S_MTE3>(evtStoMte3_);
            WaitFlag<HardEvent::S_MTE3>(evtStoMte3_);
            DataCopy(partial1Gm_[((uint64_t)token * 2U + laneId_) * PARTIAL_SLOT_ELEMS], buf, PARTIAL_SLOT_ELEMS);
            MhcDSplitSync(DSPLIT_R3_FLAG);
            DataCopy(buf, partial1Gm_[((uint64_t)token * 2U + (1U - laneId_)) * PARTIAL_SLOT_ELEMS],
                     PARTIAL_SLOT_ELEMS);
            SetFlag<HardEvent::MTE2_S>(evtMte2toS_);
            WaitFlag<HardEvent::MTE2_S>(evtMte2toS_);
            totalMidSq += buf.GetValue(0);
        }
        const float invRmsMid = 1.0f / sqrt(totalMidSq * invK_ + normEps_);
        for (uint32_t tile = tileBegin; tile < tileBegin + halfLoop; ++tile) {
            const uint32_t tOff = tile * dimTile_;
            LocalTensor<float> gammaEven = bufGamma2Fp32_.Get<float>();
            LocalTensor<float> gammaOdd = bufGamma0Fp32_.Get<float>();
            LoadGammaInto(gammaEven, gamma2Gm_, tOff, dimTile_);
            LocalTensor<T> x2Cache = bufX2Cache_.Get<T>();
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
                    LoadGammaInto(gammaNext, gamma2Gm_, (n + 1) * D_ + tOff, dimTile_);
                }
                MhcSandwichRegBase::CastB16ScaleAndMulGammaToB16(
                    x2Cache[n * D_ + tOff], x2Cache[n * D_ + tOff], gammaCurrent, invRmsMid, dimTile_);
                PipeBarrier<PIPE_V>();
                if (n + 2 < N_MAX) {
                    FenceVToMte2();
                }
            }
            CacheX2Tile(token, tOff, dimTile_, true, false);
            AccumulateHalfPartials(tOff, dimTile_, sumSqHalf, dotHalf, 0U, 2U, false);
        }
    }
    FenceMte3ToV();

    // R4/R5：交换合并五统计量（sumSqPre + 4 dot），得到完整D的gate。
    {
        float localSumSq = sumSqHalf[0] + sumSqHalf[1];
        LocalTensor<float> buf = bufScalar_.Get<float>();
        buf.SetValue(0, localSumSq);
        for (uint32_t o = 0; o < N_MAX; ++o) {
            buf.SetValue(o + 1U, dotHalf[0][o] + dotHalf[1][o]);
        }
        for (uint32_t i = N_MAX + 1U; i < PARTIAL_SLOT_ELEMS; ++i) {
            buf.SetValue(i, 0.0f);
        }
        SetFlag<HardEvent::S_MTE3>(evtStoMte3_);
        WaitFlag<HardEvent::S_MTE3>(evtStoMte3_);
        DataCopy(partial1Gm_[((uint64_t)token * 2U + laneId_) * PARTIAL_SLOT_ELEMS], buf, PARTIAL_SLOT_ELEMS);
        MhcDSplitSync(DSPLIT_R45_FLAG);
        DataCopy(buf, partial1Gm_[((uint64_t)token * 2U + (1U - laneId_)) * PARTIAL_SLOT_ELEMS], PARTIAL_SLOT_ELEMS);
        SetFlag<HardEvent::MTE2_S>(evtMte2toS_);
        WaitFlag<HardEvent::MTE2_S>(evtMte2toS_);
        sumSqHalf[0] = localSumSq + buf.GetValue(0);
        sumSqHalf[1] = 0.0f;
        for (uint32_t o = 0; o < N_MAX; ++o) {
            dotHalf[0][o] = (dotHalf[0][o] + dotHalf[1][o]) + buf.GetValue(o + 1U);
            dotHalf[1][o] = 0.0f;
        }
    }
    const float totalSumSqPre = sumSqHalf[0] + sumSqHalf[1];
    float xHatVals[N_MAX];
    for (uint32_t o = 0; o < N_MAX; ++o) {
        xHatVals[o] = dotHalf[0][o] + dotHalf[1][o];
    }
    ComputeGateValues(totalSumSqPre, xHatVals);

    // R6：用完整gate对本lane的D/2段做四head加权，得到本lane的h_in并归约平方和。
    float sumSq1 = 0.0f;
    for (uint32_t tile = tileBegin; tile < tileBegin + halfLoop; ++tile) {
        const uint32_t tOff = tile * dimTile_;
        LocalTensor<float> hIn = bufResFp32_.Get<float>()[N_MAX * dimTile_];
        LocalTensor<T> x2Cache = bufX2Cache_.Get<T>()[tOff];
        LocalTensor<float> gateBuf = bufScalar_.Get<float>()[EXP_DIV_ELEM];
        MhcSandwichRegBase::MulsB16WithUbScalar(hIn, x2Cache, gateBuf, 0U, dimTile_);
        PipeBarrier<PIPE_V>();
        for (uint32_t n = 1; n < N_MAX; ++n) {
            MhcSandwichRegBase::AxpyB16WithUbScalar(hIn, x2Cache[n * D_], gateBuf, n, dimTile_);
            PipeBarrier<PIPE_V>();
        }
        sumSq1 += CacheHInAndReduceSumSq(tOff, dimTile_, hIn);
    }

    // R7：交换合并sumSq1，得到完整D的invRms1。
    {
        LocalTensor<float> buf = bufScalar_.Get<float>();
        buf.SetValue(0, sumSq1);
        for (uint32_t i = 1U; i < PARTIAL_SLOT_ELEMS; ++i) {
            buf.SetValue(i, 0.0f);
        }
        SetFlag<HardEvent::S_MTE3>(evtStoMte3_);
        WaitFlag<HardEvent::S_MTE3>(evtStoMte3_);
        DataCopy(partial1Gm_[((uint64_t)token * 2U + laneId_) * PARTIAL_SLOT_ELEMS], buf, PARTIAL_SLOT_ELEMS);
        MhcDSplitSync(DSPLIT_R7_FLAG);
        DataCopy(buf, partial1Gm_[((uint64_t)token * 2U + (1U - laneId_)) * PARTIAL_SLOT_ELEMS], PARTIAL_SLOT_ELEMS);
        SetFlag<HardEvent::MTE2_S>(evtMte2toS_);
        WaitFlag<HardEvent::MTE2_S>(evtMte2toS_);
        sumSq1 += buf.GetValue(0);
    }

    StartCacheGamma1();
    const float invRms1 = ComputeInvRms(sumSq1, invD_);
    WaitCacheGamma1();
    for (uint32_t tile = tileBegin; tile < tileBegin + halfLoop; ++tile) {
        const uint32_t tOff = tile * dimTile_;
        LocalTensor<float> hIn = bufResFp32_.Get<float>()[N_MAX * dimTile_];
        LoadHInCacheTile(tOff, dimTile_, hIn);
        LocalTensor<float> gamma1 = bufGamma2Fp32_.Get<float>();
        if (D_ > GAMMA1_CACHE_MAX_D) {
            LoadGammaInto(gamma1, gamma1Gm_, tOff, dimTile_);
            SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
            WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);
        } else {
            gamma1 = gamma1[tOff];
        }
        ApplyScaleAndGamma(hIn, invRms1, gamma1, dimTile_);
        if (D_ > GAMMA1_CACHE_MAX_D) {
            FenceVToMte2();
        }
        WriteHInPrimeTile(hIn, (uint64_t)token * D_ + tOff, dimTile_);
    }
}

#endif // MHC_SANDWICH_NORM_POST_PREONLY_DUAL_SIBLING_AIV_H_
