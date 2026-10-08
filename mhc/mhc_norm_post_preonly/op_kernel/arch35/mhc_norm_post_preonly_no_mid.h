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
    float postVals[N];
    float sumSqHalf[2] = {0.0f, 0.0f};
    float dotHalf[2][N] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    CopyInHOut((uint64_t)token * D_);
    if (dLoop_ != 1U) {
        CopyInGamma0();
    }
    const uint32_t nStart = siblingAivIdx_ * HEADS_PER_SIBLING;
    CopyInTokenWeights(token, postVals, nStart, nStart + HEADS_PER_SIBLING);

    // 两个sibling AIV均按tile归约完整D轴的输入平方和。
    float sumSq0 = 0.0f;
    for (uint32_t tile = 0; tile < dLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dTile_;
        sumSq0 += ReduceSumSqFromCache(tOff, tD);
    }
    const float invRms0 = ComputeInvRms(sumSq0, invD_);

    for (uint32_t tile = 0; tile < dLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dTile_;
        const bool firstPhiPrefetched = tD == dTile_;
        if (firstPhiPrefetched) {
            LocalTensor<float> firstPhi = bufPhiFp32_.AllocTensor<float>();
            DataCopyExtParams phiCopy = {2, 0, 0, 0, 0};
            phiCopy.blockLen = tD * sizeof(float);
            phiCopy.srcStride = (D_ - tD) * sizeof(float);
            DataCopyPadExtParams<float> phiPad = {false, 0, 0, 0};
            DataCopyPad(firstPhi, phiGm_[(uint64_t)nStart * D_ + tOff], phiCopy, phiPad);
            bufPhiFp32_.EnQue(firstPhi);
        }
        ComputeX2PostTileForSiblingAiv(token, tOff, tD, invRms0, postVals);
        RestoreAndCopyOutX2TileForSiblingAiv(token, tOff, tD);
        AccumulateHalfPartials(tOff, tD, sumSqHalf, dotHalf, siblingAivIdx_, siblingAivIdx_ + 1U,
                               firstPhiPrefetched);
    }

    // follower写partial；两个sibling AIV均完成会合，follower立即退出。
    LocalTensor<float> partialBuf = bufScalar_.Get<float>();
    if (siblingAivIdx_ == 1U) {
        partialBuf.SetValue(0, sumSqHalf[1]);
        for (uint32_t o = 0; o < N; ++o) {
            partialBuf.SetValue(o + 1U, dotHalf[1][o]);
        }
        SetFlag<HardEvent::S_MTE3>(evtStoMte3_);
        WaitFlag<HardEvent::S_MTE3>(evtStoMte3_);
        const uint64_t followerSlot = ((uint64_t)token * SIBLING_AIV_COUNT + 1U) * PARTIAL_SLOT_ELEMS;
        DataCopy(partial1Gm_[followerSlot], partialBuf, PARTIAL_SLOT_ELEMS);
    }
    CrossCoreSetFlag<0x1, PIPE_MTE3>(PAIR_READY_FLAG);
    CrossCoreWaitFlag<0x1, PIPE_MTE2>(PAIR_READY_FLAG);
    if (siblingAivIdx_ == 1U) {
        return;
    }

    SetFlag<HardEvent::MTE3_V>(evtMte3toV_);
    const uint64_t followerSlot = ((uint64_t)token * SIBLING_AIV_COUNT + 1U) * PARTIAL_SLOT_ELEMS;
    DataCopy(partialBuf, partial1Gm_[followerSlot], PARTIAL_SLOT_ELEMS);
    SetFlag<HardEvent::MTE2_S>(evtMte2toS_);
    WaitFlag<HardEvent::MTE3_V>(evtMte3toV_);
    WaitFlag<HardEvent::MTE2_S>(evtMte2toS_);
    sumSqHalf[1] = partialBuf.GetValue(0);
    for (uint32_t o = 0; o < N; ++o) {
        dotHalf[1][o] = partialBuf.GetValue(o + 1U);
    }

    // 4op保持Gate之后回读follower的两个x2 head。
    LocalTensor<T> x2Cache = bufActivationLowp_.Get<T>();
    const uint64_t followerHeadOffset = (uint64_t)token * N_ * D_ + HEADS_PER_SIBLING * D_;
    const float totalSumSqPre = sumSqHalf[0] + sumSqHalf[1];
    float xHatVals[N];
    for (uint32_t o = 0; o < N; ++o) {
        xHatVals[o] = dotHalf[0][o] + dotHalf[1][o];
    }
    ComputeGateValues(totalSumSqPre, xHatVals);
    DataCopy(x2Cache[HEADS_PER_SIBLING * D_], x2OutGm_[followerHeadOffset], HEADS_PER_SIBLING * D_);
    SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
    WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);

    float sumSq1 = 0.0f;
    StartCopyInGamma1();
    for (uint32_t tile = 0; tile < dLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dTile_;
        LocalTensor<T> x2Tile = bufActivationLowp_.Get<T>()[tOff];
        LocalTensor<float> gateBuf = bufScalar_.Get<float>()[EXP_DIV_ELEM];
        sumSq1 += ComputeHInCacheAndReduceSumSq(tOff, tD, x2Tile, gateBuf);
    }
    FinalizeHInPrime(token, sumSq1);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ProcessTokenSingleNoMid(
    uint32_t token)
{
    float postVals[N];
    float sumSqHalf[2] = {0.0f, 0.0f};
    float dotHalf[2][N] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    CopyInHOut((uint64_t)token * D_);
    if (dLoop_ != 1U) {
        CopyInGamma0();
    }
    CopyInTokenWeights(token, postVals);

    // 第一遍：沿完整D轴计算输入归一化。
    float sumSq0 = 0.0f;
    for (uint32_t tile = 0; tile < dLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dTile_;
        sumSq0 += ReduceSumSqFromCache(tOff, tD);
    }
    const float invRms0 = ComputeInvRms(sumSq0, invD_);

    // 每个tile输出低精度x2后立即计算Phi partial。
    for (uint32_t tile = 0; tile < dLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dTile_;
        ComputeX2PostTile(token, tOff, tD, invRms0, postVals);
        RestoreAndCopyOutX2Tile(token, tOff, tD);
        AccumulateHalfPartials(tOff, tD, sumSqHalf, dotHalf);
    }
    FenceMte3ToV();

    // 固定执行顶层合并，与双sibling方案保持一致。
    const float totalSumSqPre = sumSqHalf[0] + sumSqHalf[1];
    float xHatVals[N];
    for (uint32_t o = 0; o < N; ++o) {
        xHatVals[o] = dotHalf[0][o] + dotHalf[1][o];
    }
    ComputeGateValues(totalSumSqPre, xHatVals);

    // 第三遍：计算门控融合和h_in低精度边界。在完整D轴的输出归一化标量确定前，
    // 复用已经消费完的x2缓存首段保存低精度h_in。
    float sumSq1 = 0.0f;
    // 提前发起gamma1搬运（到bufGammaFp32_），与下方门控融合的Vector计算重叠。
    StartCopyInGamma1();
    for (uint32_t tile = 0; tile < dLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dTile_;
        LocalTensor<T> x2Cache = bufActivationLowp_.Get<T>()[tOff];
        LocalTensor<float> gateBuf = bufScalar_.Get<float>()[EXP_DIV_ELEM];
        sumSq1 += ComputeHInCacheAndReduceSumSq(tOff, tD, x2Cache, gateBuf);
    }

    // 第四遍：沿完整D轴计算输出归一化并写最终输出。
    FinalizeHInPrime(token, sumSq1);
}

#endif // MHC_NORM_POST_PREONLY_NO_MID_H_
