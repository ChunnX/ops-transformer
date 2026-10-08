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
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32,
                                                        DUAL_SIBLING>::ProcessTokenDualNoMid(uint32_t token)
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
        CacheX2TileForLane(token, tOff, tD, !false);

        LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
        {
            AccumulateHalfPartials(tOff, tD, sumSqHalf, dotHalf, laneId_, laneId_ + 1U);
        }
    }

    {}

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
        LocalTensor<float> hIn = bufResFp32_.Get<float>()[N_MAX * dimTile_];
        LocalTensor<T> x2Cache = bufX2Cache_.Get<T>()[tOff];
        MhcNormRegBase::GateWeightedSum(hIn, x2Cache, D_, gate, tD);
        PipeBarrier<PIPE_V>();
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

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void
KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ProcessTokenSingleNoMid(uint32_t token)
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
        CacheX2Tile(token, tOff, tD, !false);

        LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
        {
            AccumulateHalfPartials(tOff, tD, sumSqHalf, dotHalf);
        }
    }

    // 可选R3沿完整N*D向量计算，不按head分别归一化。
    {}
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
        LocalTensor<float> hIn = bufResFp32_.Get<float>()[N_MAX * dimTile_];
        LocalTensor<T> x2Cache = bufX2Cache_.Get<T>()[tOff];
        MhcNormRegBase::GateWeightedSum(hIn, x2Cache, D_, gate, tD);
        PipeBarrier<PIPE_V>();
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

#endif // MHC_NORM_POST_PREONLY_NO_MID_H_
