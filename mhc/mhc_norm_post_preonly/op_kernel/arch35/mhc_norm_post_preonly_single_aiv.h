/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MHC_NORM_POST_PREONLY_SINGLE_AIV_H_
#define MHC_NORM_POST_PREONLY_SINGLE_AIV_H_

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ProcessTokenSingleAiv(
    uint32_t token)
{
    float postVals[N];
    float sumSqHalf[2] = {0.0f, 0.0f};
    float dotHalf[2][N] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    float midSqHalf[2] = {0.0f, 0.0f};
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

    if constexpr (USE_RMS_MID) {
        BuildX2AndPartialsSingle5Op(token, invRms0, postVals, sumSqHalf, dotHalf, midSqHalf);
    } else {
        BuildX2AndPartialsSingle4Op(token, invRms0, postVals, sumSqHalf, dotHalf);
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
    const float invRms1 = ComputeInvRms(sumSq1, invD_);
    WaitCopyInGamma1();
    for (uint32_t tile = 0; tile < dLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dTile_;
        LocalTensor<float> hIn = bufResFp32_.Get<float>()[N * dTile_];
        LocalTensor<T> hInCache = bufActivationLowp_.Get<T>()[tOff];
        LocalTensor<float> gamma1 = bufGammaFp32_.Get<float>()[tOff];
        if constexpr (RETURN_H_IN_F32) {
            MhcNormRegBase::VFCastB16ScaleAndMulGamma(hIn, hInCache, gamma1, invRms1, tD);
            PipeBarrier<PIPE_V>();
            CopyOutHInPrimeTile(hIn, (uint64_t)token * D_ + tOff, tD);
        } else {
            ComputeAndCopyOutHInPrimeTile(hInCache, gamma1, invRms1,
                                       (uint64_t)token * D_ + tOff, tD);
        }
    }
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void
KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::BuildX2AndPartialsSingle4Op(
    uint32_t token, float invRms0, const float (&postVals)[N], float (&sumSqHalf)[2], float (&dotHalf)[2][N])
{
    // 每个tile输出低精度x2后立即计算Phi partial。
    for (uint32_t tile = 0; tile < dLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dTile_;
        ComputeX2PostTile(token, tOff, tD, invRms0, postVals);
        PrepareX2Tile(token, tOff, tD, true);
        AccumulateHalfPartials(tOff, tD, sumSqHalf, dotHalf);
    }
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void
KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::BuildX2AndPartialsSingle5Op(
    uint32_t token, float invRms0, const float (&postVals)[N], float (&sumSqHalf)[2], float (&dotHalf)[2][N],
    float (&midSqHalf)[2])
{
    // 第一遍计算x2低精度边界和中间归一化平方和。
    for (uint32_t tile = 0; tile < dLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dTile_;
        ComputeX2PostTile(token, tOff, tD, invRms0, postVals);
        PrepareX2Tile(token, tOff, tD, false);

        LocalTensor<float> out = bufResFp32_.Get<float>()[N * dTile_];
        LocalTensor<T> x2Cache = bufActivationLowp_.Get<T>();
        for (uint32_t half = 0; half < 2; ++half) {
            const uint32_t h0 = half * 2;
            const uint32_t h1 = h0 + 1;
            MhcNormRegBase::VFPairSquareB16ToFp32(out, x2Cache[h0 * D_ + tOff], x2Cache[h1 * D_ + tOff], tD);
            PipeBarrier<PIPE_V>();
            midSqHalf[half] += ReduceSumPrecise(out, tD);
        }
    }

    // 中间归一化沿完整N*D向量计算，不按head分别归一化。
    const float totalMidSq = midSqHalf[0] + midSqHalf[1];
    const float invRmsMid = 1.0f / sqrt(totalMidSq * invK_ + normEps_);
    for (uint32_t tile = 0; tile < dLoop_; ++tile) {
        const uint32_t tD = GetTileD(tile);
        const uint32_t tOff = tile * dTile_;
        // 单分块时复用残差融合结束后已空闲的residual scratch作为第二个gamma2槽，
        // 保留Process预载的gamma0供后续token复用；多分块仍复用gamma0缓冲。
        // 首个gamma2搬运和x2缓存恢复重叠，后续搬运与前一head的Vector计算重叠。
        LocalTensor<float> gammaEven = bufScratchFp32_.Get<float>();
        LocalTensor<float> gammaOdd =
            dLoop_ == 1U ? bufResFp32_.Get<float>() : bufGammaFp32_.Get<float>();
        CopyInGammaTile(gammaEven, gamma2Gm_, tOff, tD);
        LocalTensor<T> x2Cache = bufActivationLowp_.Get<T>();
        for (uint32_t n = 0; n < N; ++n) {
            LocalTensor<float> gammaCurrent = gammaEven;
            LocalTensor<float> gammaNext = gammaOdd;
            if ((n & 1U) != 0) {
                gammaCurrent = gammaOdd;
                gammaNext = gammaEven;
            }
            SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
            WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);
            if (n + 1 < N) {
                CopyInGammaTile(gammaNext, gamma2Gm_, (n + 1) * D_ + tOff, tD);
            }
            MhcNormRegBase::VFCastB16ScaleAndMulGammaToB16(x2Cache[n * D_ + tOff], x2Cache[n * D_ + tOff],
                                                           gammaCurrent, invRmsMid, tD);
            PipeBarrier<PIPE_V>();
            // 当前缓存会在两轮后再次由MTE2写入，前两轮显式建立V到MTE2依赖。
            if (n + 2 < N) {
                FenceVToMte2();
            }
        }
        PrepareX2Tile(token, tOff, tD, true);
        AccumulateHalfPartials(tOff, tD, sumSqHalf, dotHalf);
    }
}

#endif // MHC_NORM_POST_PREONLY_SINGLE_AIV_H_
