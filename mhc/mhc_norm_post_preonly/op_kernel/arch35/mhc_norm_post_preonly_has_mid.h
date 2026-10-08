/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MHC_NORM_POST_PREONLY_HAS_MID_H_
#define MHC_NORM_POST_PREONLY_HAS_MID_H_

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void
KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ProcessTokenDualHasMid(uint32_t token)
{
    float postVals[N_MAX];
    float sumSqHalf[2] = {0.0f, 0.0f};
    float dotHalf[2][N_MAX] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    float midSqHalf[2] = {0.0f, 0.0f};
    CacheHOut((uint64_t)token * D_);
    // 单分块gamma0通常由Process预载；5op会在R3复用该缓冲缓存gamma2，
    // 因此同一核处理后续token时需恢复gamma0。4op不增加重复搬运。
    if (dimLoop_ != 1U || (true && token != tokenStart_)) {
        CacheGamma0();
    }
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

    {
        // 5op第一遍R2后只使用bufResFp32_的第五行做midSq，前四行暂时空闲。
        // tile0先进入队列，后续tile在队列与前四行scratch之间交替预取，
        // 不增加UB占用即可让下一tile residual MTE2跨越当前tile的R2/RMS_mid计算。
        LocalTensor<T> firstResidual = bufResidualBf16_.AllocTensor<T>();
        const uint32_t firstTD = GetTileD(0U);
        const uint64_t firstBaseOffset = (uint64_t)token * N_ * D_;
        for (uint32_t n = 0; n < N_MAX; ++n) {
            DataCopy(firstResidual[n * dimTile_], residualGm_[firstBaseOffset + (uint64_t)n * D_], firstTD);
        }
        bufResidualBf16_.EnQue(firstResidual);
    }

    // 两个lane分别构造自己负责的两个x2 head。
    for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dimTile_;
        {
            LocalTensor<float> hOut = bufResFp32_.Get<float>()[N_MAX * dimTile_];
            LocalTensor<T> hOutCache = bufX2Cache_.Get<T>()[tOff];
            MhcNormRegBase::VFCastB16ScaleGammaQuantizeToFp32(
                hOut, hOutCache, bufGamma0Fp32_.Get<float>()[tOff], invRms0, tD);
            PipeBarrier<PIPE_V>();

            const bool residualFromQueue = (tile & 1U) == 0U;
            LocalTensor<T> residual;
            if (residualFromQueue) {
                residual = bufResidualBf16_.DeQue<T>();
            } else {
                WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);
                residual = bufResFp32_.Get<T>();
            }

            if (tile + 1U < dimLoop_) {
                const uint32_t nextTile = tile + 1U;
                const uint32_t nextTOff = nextTile * dimTile_;
                const uint32_t nextTD = GetTileD(nextTile);
                const uint64_t nextBaseOffset = (uint64_t)token * N_ * D_ + nextTOff;
                if (residualFromQueue) {
                    LocalTensor<T> nextResidual = bufResFp32_.Get<T>();
                    for (uint32_t n = 0; n < N_MAX; ++n) {
                        DataCopy(nextResidual[n * dimTile_], residualGm_[nextBaseOffset + (uint64_t)n * D_], nextTD);
                    }
                    SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
                } else {
                    LocalTensor<T> nextResidual = bufResidualBf16_.AllocTensor<T>();
                    for (uint32_t n = 0; n < N_MAX; ++n) {
                        DataCopy(nextResidual[n * dimTile_], residualGm_[nextBaseOffset + (uint64_t)n * D_], nextTD);
                    }
                    bufResidualBf16_.EnQue(nextResidual);
                }
            }

            LocalTensor<T> x2Cache = bufX2Cache_.Get<T>();
            LocalTensor<float> combWeights = bufPostComb_.Get<float>()[ALIGN8];
            MhcNormRegBase::VFComputePostHeadPair<T>(
                x2Cache[headBegin * D_ + tOff], x2Cache[(headBegin + 1U) * D_ + tOff], hOut, residual, combWeights,
                dimTile_, headBegin, postVals[headBegin], postVals[headBegin + 1U], tD);
            PipeBarrier<PIPE_V>();
            if (residualFromQueue) {
                bufResidualBf16_.FreeTensor(residual);
            }
        }
        CacheX2TileForLane(token, tOff, tD, !true);

        {
            // 每个lane只归约自己负责的两个head，循环结束后交换半和。
            LocalTensor<float> out = bufResFp32_.Get<float>()[N_MAX * dimTile_];
            LocalTensor<T> x2Cache = bufX2Cache_.Get<T>();
            for (uint32_t half = laneId_; half < laneId_ + 1U; ++half) {
                const uint32_t h0 = half * 2U;
                const uint32_t h1 = h0 + 1;
                MhcNormRegBase::VFPairSquareB16ToFp32(out, x2Cache[h0 * D_ + tOff], x2Cache[h1 * D_ + tOff], tD);
                PipeBarrier<PIPE_V>();
                midSqHalf[half] += ReduceSumPrecise(out, tD);
            }
        }
    }

    {
        // R2结束后gamma0不再使用：整段预取本lane第一个gamma2 head，
        // 用下方RMS_mid跨核交换的Scalar/MTE3/MTE2时间隐藏这次搬运。
        LocalTensor<float> gammaEvenCache = bufGamma0Fp32_.Get<float>();
        FenceVToMte2();
        LoadGammaInto(gammaEvenCache, gamma2Gm_, headBegin * D_, D_);
        SetFlag<HardEvent::MTE2_V>(evtMte2toV_);

        LocalTensor<float> midBuf = bufScalar_.Get<float>();
        midBuf.SetValue(0, midSqHalf[laneId_]);
        // 32B槽的其余元素不会被sibling读取，无需在关键同步路径上做Scalar清零。
        SetFlag<HardEvent::S_MTE3>(evtStoMte3_);
        WaitFlag<HardEvent::S_MTE3>(evtStoMte3_);
        const uint64_t ownSlot = ((uint64_t)token * 2U + laneId_) * PARTIAL_SLOT_ELEMS;
        DataCopy(partial1Gm_[ownSlot], midBuf, PARTIAL_SLOT_ELEMS);
        CrossCoreSetFlag<0x1, PIPE_MTE3>(MID_READY_FLAG);
        CrossCoreWaitFlag<0x1, PIPE_MTE2>(MID_READY_FLAG);
        const uint64_t siblingSlot = ((uint64_t)token * 2U + (1U - laneId_)) * PARTIAL_SLOT_ELEMS;
        DataCopy(midBuf, partial1Gm_[siblingSlot], PARTIAL_SLOT_ELEMS);
        SetFlag<HardEvent::MTE2_S>(evtMte2toS_);
        WaitFlag<HardEvent::MTE2_S>(evtMte2toS_);
        midSqHalf[1U - laneId_] = midBuf.GetValue(0);
        const float totalMidSq = midSqHalf[0] + midSqHalf[1];
        const float invRmsMid = 1.0f / sqrt(totalMidSq * invK_ + normEps_);
        WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);
        for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
            const uint32_t tD = GetTileD(tile);
            const uint32_t tOff = tile * dimTile_;
            LocalTensor<float> gammaEven = gammaEvenCache[tOff];
            LocalTensor<float> gammaOdd = bufGamma2Fp32_.Get<float>();
            LocalTensor<T> x2Cache = bufX2Cache_.Get<T>();
            // 每个lane只执行自己负责的两个head，另一半由sibling并行完成。
            // 第一个head已整段缓存；第二个head的分块MTE2与第一个head的Vector重叠。
            LoadGammaInto(gammaOdd, gamma2Gm_, (headBegin + 1U) * D_ + tOff, tD);
            MhcNormRegBase::VFCastB16ScaleAndMulGammaToB16(x2Cache[headBegin * D_ + tOff],
                                                           x2Cache[headBegin * D_ + tOff], gammaEven, invRmsMid, tD);
            PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
            WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);
            MhcNormRegBase::VFCastB16ScaleAndMulGammaToB16(
                x2Cache[(headBegin + 1U) * D_ + tOff], x2Cache[(headBegin + 1U) * D_ + tOff], gammaOdd, invRmsMid, tD);
            PipeBarrier<PIPE_V>();
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
        // leader只读取前五个有效partial，槽尾三个元素无需清零。
        SetFlag<HardEvent::S_MTE3>(evtStoMte3_);
        WaitFlag<HardEvent::S_MTE3>(evtStoMte3_);
        const uint64_t followerSlot = ((uint64_t)token * 2U + 1U) * PARTIAL_SLOT_ELEMS;
        DataCopy(partial1Gm_[followerSlot], partialBuf, PARTIAL_SLOT_ELEMS);
    }

    // 两个lane均无条件执行一次mode1 Set/Wait；该等待不替代本核流水事件。
    CrossCoreSetFlag<0x1, PIPE_MTE3>(PAIR_READY_FLAG);
    CrossCoreWaitFlag<0x1, PIPE_MTE2>(PAIR_READY_FLAG);
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

    // follower负责的两个head在GM和UB中均连续。partial到达后立即发起2D回读，
    // 用下方Gate的Scalar/Vector计算隐藏这段MTE2；R6消费x2前再等待完成。
    LocalTensor<T> x2Cache = bufX2Cache_.Get<T>();
    const uint64_t followerHeadOffset = (uint64_t)token * N_ * D_ + 2U * D_;
    {
        DataCopy(x2Cache[2U * D_], x2OutGm_[followerHeadOffset], 2U * D_);
        SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
    }

    const float totalSumSqPre = sumSqHalf[0] + sumSqHalf[1];
    float xHatVals[N_MAX];
    for (uint32_t o = 0; o < N_MAX; ++o) {
        xHatVals[o] = dotHalf[0][o] + dotHalf[1][o];
    }
    ComputeGateValues(totalSumSqPre, xHatVals);

    // 会合后只有leader继续执行R6、R7和最终输出。
    if constexpr (!true) {
        DataCopy(x2Cache[2U * D_], x2OutGm_[followerHeadOffset], 2U * D_);
        SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
    }
    WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);

    float sumSq1 = 0.0f;
    // 提前发起gamma1搬运（到bufGamma0Fp32_），与下方R6的Vector计算重叠。
    StartCacheGamma1();
    for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dimTile_;
        LocalTensor<T> x2Cache = bufX2Cache_.Get<T>()[tOff];
        LocalTensor<float> gateBuf = bufScalar_.Get<float>()[EXP_DIV_ELEM];
        sumSq1 += ComputeHInCacheAndReduceSumSq(tOff, tD, x2Cache, gateBuf);
    }

    const float invRms1 = ComputeInvRms(sumSq1, invD_);
    WaitCacheGamma1();
    for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dimTile_;
        LocalTensor<float> hIn = bufResFp32_.Get<float>()[N_MAX * dimTile_];
        LocalTensor<T> hInCache = bufX2Cache_.Get<T>()[tOff];
        LocalTensor<float> gamma1 = bufGamma0Fp32_.Get<float>();
        if (D_ > GAMMA1_CACHE_MAX_D) {
            LoadGammaInto(gamma1, gamma1Gm_, tOff, tD);
            SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
            WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);
        } else {
            gamma1 = gamma1[tOff];
        }
        MhcNormRegBase::VFCastB16ScaleAndMulGamma(hIn, hInCache, gamma1, invRms1, tD);
        PipeBarrier<PIPE_V>();
        if (D_ > GAMMA1_CACHE_MAX_D) {
            FenceVToMte2();
        }
        WriteHInPrimeTile(hIn, (uint64_t)token * D_ + tOff, tD);
    }
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ProcessTokenSingleHasMid(
    uint32_t token)
{
    float postVals[N_MAX];
    float sumSqHalf[2] = {0.0f, 0.0f};
    float dotHalf[2][N_MAX] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    float midSqHalf[2] = {0.0f, 0.0f};
    CacheHOut((uint64_t)token * D_);
    if (dimLoop_ != 1U) {
        CacheGamma0();
    }
    LoadTokenWeights(token, postVals);

    // 第一遍：沿完整D轴计算R1。
    float sumSq0 = 0.0f;
    for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dimTile_;
        sumSq0 += ReduceSumSqFromCache(tOff, tD);
    }
    const float invRms0 = ComputeInvRms(sumSq0, invD_);

    // 第二遍：计算R1/R2和第一处x2低精度边界。完整x2以输出dtype常驻UB，
    // 避免后续两遍从GM回读，同时不分配完整FP32 x2工作空间。
    for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dimTile_;
        ComputeX2PostTile(token, tOff, tD, invRms0, postVals);
        CacheX2Tile(token, tOff, tD, !true);

        {
            LocalTensor<float> out = bufResFp32_.Get<float>()[N_MAX * dimTile_];
            LocalTensor<T> x2Cache = bufX2Cache_.Get<T>();
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t h0 = half * 2;
                const uint32_t h1 = h0 + 1;
                MhcNormRegBase::VFPairSquareB16ToFp32(out, x2Cache[h0 * D_ + tOff], x2Cache[h1 * D_ + tOff], tD);
                PipeBarrier<PIPE_V>();
                midSqHalf[half] += ReduceSumPrecise(out, tD);
            }
        }
    }

    // 可选R3沿完整N*D向量计算，不按head分别归一化。
    {
        const float totalMidSq = midSqHalf[0] + midSqHalf[1];
        const float invRmsMid = 1.0f / sqrt(totalMidSq * invK_ + normEps_);
        for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
            const uint32_t tD = GetTileD(tile);
            const uint32_t tOff = tile * dimTile_;
            // 单分块时复用R2结束后已空闲的residual scratch作为第二个gamma2槽，
            // 保留Process预载的gamma0供后续token复用；多分块仍复用gamma0缓冲。
            // 首个gamma2搬运和x2缓存恢复重叠，后续搬运与前一head的Vector计算重叠。
            LocalTensor<float> gammaEven = bufGamma2Fp32_.Get<float>();
            LocalTensor<float> gammaOdd =
                dimLoop_ == 1U ? bufResFp32_.Get<float>() : bufGamma0Fp32_.Get<float>();
            LoadGammaInto(gammaEven, gamma2Gm_, tOff, tD);
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
                    LoadGammaInto(gammaNext, gamma2Gm_, (n + 1) * D_ + tOff, tD);
                }
                MhcNormRegBase::VFCastB16ScaleAndMulGammaToB16(x2Cache[n * D_ + tOff], x2Cache[n * D_ + tOff],
                                                               gammaCurrent, invRmsMid, tD);
                PipeBarrier<PIPE_V>();
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
    ComputeGateValues(totalSumSqPre, xHatVals);

    // 第三遍：计算R6和h_in低精度边界。在完整D轴的R7标量确定前，
    // 复用已经消费完的x2缓存首段保存低精度h_in。
    float sumSq1 = 0.0f;
    // 提前发起gamma1搬运（到bufGamma0Fp32_），与下方R6的Vector计算重叠。
    StartCacheGamma1();
    for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dimTile_;
        LocalTensor<T> x2Cache = bufX2Cache_.Get<T>()[tOff];
        LocalTensor<float> gateBuf = bufScalar_.Get<float>()[EXP_DIV_ELEM];
        sumSq1 += ComputeHInCacheAndReduceSumSq(tOff, tD, x2Cache, gateBuf);
    }

    // 第四遍：沿完整D轴计算R7并写最终输出。
    const float invRms1 = ComputeInvRms(sumSq1, invD_);
    WaitCacheGamma1();
    for (uint32_t tile = 0; tile < dimLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dimTile_;
        LocalTensor<float> hIn = bufResFp32_.Get<float>()[N_MAX * dimTile_];
        LocalTensor<T> hInCache = bufX2Cache_.Get<T>()[tOff];
        LocalTensor<float> gamma1 = bufGamma0Fp32_.Get<float>();
        if (D_ > GAMMA1_CACHE_MAX_D) {
            LoadGammaInto(gamma1, gamma1Gm_, tOff, tD);
            SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
            WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);
        } else {
            gamma1 = gamma1[tOff];
        }
        MhcNormRegBase::VFCastB16ScaleAndMulGamma(hIn, hInCache, gamma1, invRms1, tD);
        PipeBarrier<PIPE_V>();
        if (D_ > GAMMA1_CACHE_MAX_D) {
            FenceVToMte2();
        }
        WriteHInPrimeTile(hIn, (uint64_t)token * D_ + tOff, tD);
    }
}

#endif // MHC_NORM_POST_PREONLY_HAS_MID_H_
