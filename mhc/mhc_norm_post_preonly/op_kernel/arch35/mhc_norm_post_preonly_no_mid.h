/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MHC_NORM_POST_PREONLY_NO_MID_H_
#define MHC_NORM_POST_PREONLY_NO_MID_H_

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void
KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ProcessTokenDualNoMid(uint32_t token)
{
    float postVals[N_MAX];
    float sumSqHalf[2] = {0.0f, 0.0f};
    float dotHalf[2][N_MAX] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    float midSqHalf[2] = {0.0f, 0.0f};
    CacheHOut((uint64_t)token * D_);
    if (dLoop_ != 1U || (false && token != tStart_)) {
        CacheGamma0();
    }
    const uint32_t headBegin = siblingAivIdx_ * 2U;
    LoadTokenWeights(token, postVals, headBegin, headBegin + 2U);

    {}

    // 两个lane按完全相同的分块和顺序冗余执行输入归一化。
    float sumSq0 = 0.0f;
    {
        for (uint32_t tile = 0; tile < dLoop_; ++tile) {
            const uint32_t tD = GetTileD(tile);
            const uint32_t tOff = tile * dTile_;
            sumSq0 += ReduceSumSqFromCache(tOff, tD);
        }
    }
    const float invRms0 = ComputeInvRms(sumSq0, invD_);

    // 两个lane分别构造自己负责的两个x2 head。
    for (uint32_t tile = 0; tile < dLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dTile_;
        bool firstPhiPrefetched = false;
        if constexpr (!false) {
            firstPhiPrefetched = tD == dTile_;
            if (firstPhiPrefetched) {
                LocalTensor<float> firstPhi = bufPhiFp32_.AllocTensor<float>();
                DataCopyExtParams phiCopy = {2, 0, 0, 0, 0};
                phiCopy.blockLen = tD * sizeof(float);
                phiCopy.srcStride = (D_ - tD) * sizeof(float);
                DataCopyPadExtParams<float> phiPad = {false, 0, 0, 0};
                DataCopyPad(firstPhi, phiGm_[(uint64_t)headBegin * D_ + tOff], phiCopy, phiPad);
                bufPhiFp32_.EnQue(firstPhi);
            }
        }
        {
            ComputeX2PostTileForLane(token, tOff, tD, invRms0, postVals);
        }
        CacheX2TileForLane(token, tOff, tD, !false);

        {
            AccumulateHalfPartials(tOff, tD, sumSqHalf, dotHalf, siblingAivIdx_, siblingAivIdx_ + 1U,
                                   firstPhiPrefetched);
        }
    }

    {}

    // follower把五个有效partial写入独占的32B槽；leader的partial0始终留在本核。
    LocalTensor<float> partialBuf = bufScalar_.Get<float>();
    if (siblingAivIdx_ == 1U) {
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
    if (siblingAivIdx_ == 1U) {
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
    // 用下方Gate的Scalar/Vector计算隐藏这段MTE2；门控融合消费x2前再等待完成。
    LocalTensor<T> x2Cache = bufX2Cache_.Get<T>();
    const uint64_t followerHeadOffset = (uint64_t)token * N_ * D_ + 2U * D_;
    {}

    const float totalSumSqPre = sumSqHalf[0] + sumSqHalf[1];
    float xHatVals[N_MAX];
    for (uint32_t o = 0; o < N_MAX; ++o) {
        xHatVals[o] = dotHalf[0][o] + dotHalf[1][o];
    }
    ComputeGateValues(totalSumSqPre, xHatVals);

    // 会合后只有leader继续执行门控融合、输出归一化和最终输出。
    if constexpr (!false) {
        DataCopy(x2Cache[2U * D_], x2OutGm_[followerHeadOffset], 2U * D_);
        SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
    }
    WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);

    float sumSq1 = 0.0f;
    // 提前发起gamma1搬运（到bufGamma0Fp32_），与下方门控融合的Vector计算重叠。
    StartCacheGamma1();
    for (uint32_t tile = 0; tile < dLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dTile_;
        LocalTensor<T> x2Cache = bufX2Cache_.Get<T>()[tOff];
        LocalTensor<float> gateBuf = bufScalar_.Get<float>()[EXP_DIV_ELEM];
        {
            sumSq1 += ComputeHInCacheAndReduceSumSq(tOff, tD, x2Cache, gateBuf);
        }
    }

    {}
    const float invRms1 = ComputeInvRms(sumSq1, invD_);
    WaitCacheGamma1();
    for (uint32_t tile = 0; tile < dLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dTile_;
        LocalTensor<float> hIn = bufResFp32_.Get<float>()[N_MAX * dTile_];
        LocalTensor<T> hInCache = bufX2Cache_.Get<T>()[tOff];
        LocalTensor<float> gamma1 = bufGamma0Fp32_.Get<float>()[tOff];
        if constexpr (RETURN_H_IN_F32) {
            MhcNormRegBase::VFCastB16ScaleAndMulGamma(hIn, hInCache, gamma1, invRms1, tD);
            PipeBarrier<PIPE_V>();
            WriteHInPrimeTile(hIn, (uint64_t)token * D_ + tOff, tD);
        } else {
            WriteHInPrimeTileFromCache(hInCache, gamma1, invRms1,
                                       (uint64_t)token * D_ + tOff, tD);
        }
    }
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ProcessTokenSingleNoMid(
    uint32_t token)
{
    float postVals[N_MAX];
    float sumSqHalf[2] = {0.0f, 0.0f};
    float dotHalf[2][N_MAX] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    float midSqHalf[2] = {0.0f, 0.0f};
    CacheHOut((uint64_t)token * D_);
    if (dLoop_ != 1U) {
        CacheGamma0();
    }
    LoadTokenWeights(token, postVals);

    // 第一遍：沿完整D轴计算输入归一化。
    float sumSq0 = 0.0f;
    for (uint32_t tile = 0; tile < dLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dTile_;
        sumSq0 += ReduceSumSqFromCache(tOff, tD);
    }
    const float invRms0 = ComputeInvRms(sumSq0, invD_);

    // 第二遍：计算输入归一化、残差融合及第一处x2低精度边界。完整x2以输出dtype常驻UB，
    // 避免后续两遍从GM回读，同时不分配完整FP32 x2工作空间。
    for (uint32_t tile = 0; tile < dLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dTile_;
        ComputeX2PostTile(token, tOff, tD, invRms0, postVals);
        CacheX2Tile(token, tOff, tD, !false);

        {
            AccumulateHalfPartials(tOff, tD, sumSqHalf, dotHalf);
        }
    }

    // 可选中间归一化沿完整N*D向量计算，不按head分别归一化。
    {}
    FenceMte3ToV();

    // 固定执行顶层合并，与双sibling方案保持一致。
    const float totalSumSqPre = sumSqHalf[0] + sumSqHalf[1];
    float xHatVals[N_MAX];
    for (uint32_t o = 0; o < N_MAX; ++o) {
        xHatVals[o] = dotHalf[0][o] + dotHalf[1][o];
    }
    ComputeGateValues(totalSumSqPre, xHatVals);

    // 第三遍：计算门控融合和h_in低精度边界。在完整D轴的输出归一化标量确定前，
    // 复用已经消费完的x2缓存首段保存低精度h_in。
    float sumSq1 = 0.0f;
    // 提前发起gamma1搬运（到bufGamma0Fp32_），与下方门控融合的Vector计算重叠。
    StartCacheGamma1();
    for (uint32_t tile = 0; tile < dLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dTile_;
        LocalTensor<T> x2Cache = bufX2Cache_.Get<T>()[tOff];
        LocalTensor<float> gateBuf = bufScalar_.Get<float>()[EXP_DIV_ELEM];
        sumSq1 += ComputeHInCacheAndReduceSumSq(tOff, tD, x2Cache, gateBuf);
    }

    // 第四遍：沿完整D轴计算输出归一化并写最终输出。
    const float invRms1 = ComputeInvRms(sumSq1, invD_);
    WaitCacheGamma1();
    for (uint32_t tile = 0; tile < dLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dTile_;
        LocalTensor<float> hIn = bufResFp32_.Get<float>()[N_MAX * dTile_];
        LocalTensor<T> hInCache = bufX2Cache_.Get<T>()[tOff];
        LocalTensor<float> gamma1 = bufGamma0Fp32_.Get<float>()[tOff];
        if constexpr (RETURN_H_IN_F32) {
            MhcNormRegBase::VFCastB16ScaleAndMulGamma(hIn, hInCache, gamma1, invRms1, tD);
            PipeBarrier<PIPE_V>();
            WriteHInPrimeTile(hIn, (uint64_t)token * D_ + tOff, tD);
        } else {
            WriteHInPrimeTileFromCache(hInCache, gamma1, invRms1,
                                       (uint64_t)token * D_ + tOff, tD);
        }
    }
}

#endif // MHC_NORM_POST_PREONLY_NO_MID_H_
