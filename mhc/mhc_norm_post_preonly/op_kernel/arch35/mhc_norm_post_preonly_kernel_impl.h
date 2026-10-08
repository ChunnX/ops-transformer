/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MHC_NORM_POST_PREONLY_KERNEL_IMPL_H_
#define MHC_NORM_POST_PREONLY_KERNEL_IMPL_H_

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::AccumulateHalfPartials(uint32_t tOff, uint32_t tD, float (&sumSqHalf)[2],
                                                  float (&dotHalf)[2][N], uint32_t halfBegin,
                                                  uint32_t halfEnd, bool firstPhiPrefetched)
{
        LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
        LocalTensor<float> reduceRows = bufResFp32_.Get<float>();
        // 仅dual has-mid且D无尾块时延迟归约；single和no-mid在当前tile归约。
        const bool deferReduce = DUAL_SIBLING && USE_RMS_MID && lastDTile_ == 0U;
        const bool accumulateRows = deferReduce && tOff != 0U;
        const bool compactRows = DUAL_SIBLING && tD == dTile_ &&
                                 tD % MhcNormRegBase::VL_FP32 == 0U;
        const uint32_t reduceWidth = compactRows ? MhcNormRegBase::VL_FP32 : tD;

        DataCopyExtParams copyParams = {2, 0, 0, 0, 0};
        copyParams.blockLen = tD * sizeof(float);
        copyParams.srcStride = (D_ - tD) * sizeof(float);
        copyParams.dstStride = (dTile_ - tD) * sizeof(float) / 32U;
        DataCopyPadExtParams<float> copyPad = {false, 0, 0, 0};

        // 当前支持的尾块仅来自D=6144/7168，使用紧凑五行结果做二维归约。
        if (__builtin_expect(tD != dTile_, 0)) {
            for (uint32_t half = halfBegin; half < halfEnd; ++half) {
                const uint32_t h0 = half * 2;
                const uint32_t h1 = h0 + 1;
                const uint32_t x2h0 = DUAL_SIBLING ? 0U : h0;
                const uint32_t x2h1 = DUAL_SIBLING ? 1U : h1;

                MhcNormRegBase::VFPairSquareAdd(reduceRows, x2[x2h0 * dTile_], x2[x2h1 * dTile_], tD);
                for (uint32_t o = 0; o < N; ++o) {
                    LocalTensor<float> phi0 = bufPhiFp32_.AllocTensor<float>();
                    const uint64_t phiBaseOffset = (uint64_t)o * phiCols_ + tOff;
                    DataCopyPad(phi0, phiGm_[phiBaseOffset + (uint64_t)h0 * D_], copyParams, copyPad);
                    bufPhiFp32_.EnQue(phi0);
                    phi0 = bufPhiFp32_.DeQue<float>();
                    LocalTensor<float> dotRow = reduceRows[(o + 1U) * tD];
                    MhcNormRegBase::VFPairMulAdd(dotRow, x2[x2h0 * dTile_], phi0, x2[x2h1 * dTile_],
                                                 phi0[dTile_], tD);
                    PipeBarrier<PIPE_V>();
                    bufPhiFp32_.FreeTensor(phi0);
                }

                PipeBarrier<PIPE_V>();
                LocalTensor<float> scBuf = bufScalar_.Get<float>();
                uint32_t reduceShape[2] = {N + 1U, tD};
                ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(scBuf, reduceRows, reduceShape, true);
                PipeBarrier<PIPE_V>();
                sumSqHalf[half] += scBuf.GetValue(0);
                for (uint32_t o = 0; o < N; ++o) {
                    dotHalf[half][o] += scBuf.GetValue(o + 1U);
                }
            }
            return;
        }

        // 完整tile：复用五行平方/dot归约树；sibling只处理自己负责的half。
        for (uint32_t half = halfBegin; half < halfEnd; ++half) {
            const uint32_t h0 = half * 2;
            const uint32_t h1 = h0 + 1;
            const uint32_t x2h0 = DUAL_SIBLING ? 0U : h0;
            const uint32_t x2h1 = DUAL_SIBLING ? 1U : h1;

            // 五行依次保存{sumSqPre,dot0,dot1,dot2,dot3}，一次按行归约，
            // 避免每个标量结果都单独触发一次Vector到Scalar的等待。
            if (compactRows) {
                if (accumulateRows) {
                    if constexpr (USE_RMS_MID) {
                        MhcNormRegBase::VFPairSquareAddCompact<true>(reduceRows, x2[x2h0 * dTile_],
                                                                      x2[x2h1 * dTile_], tD);
                    } else {
                        MhcNormRegBase::VFPairMulAddCompact<true>(reduceRows, x2[x2h0 * dTile_], x2[x2h0 * dTile_],
                                                                   x2[x2h1 * dTile_], x2[x2h1 * dTile_], tD);
                    }
                } else {
                    if constexpr (USE_RMS_MID) {
                        MhcNormRegBase::VFPairSquareAddCompact(reduceRows, x2[x2h0 * dTile_],
                                                               x2[x2h1 * dTile_], tD);
                    } else {
                        MhcNormRegBase::VFPairMulAddCompact(reduceRows, x2[x2h0 * dTile_], x2[x2h0 * dTile_],
                                                            x2[x2h1 * dTile_], x2[x2h1 * dTile_], tD);
                    }
                }
            } else if (accumulateRows) {
                MhcNormRegBase::VFPairSquareAdd<true>(reduceRows, x2[x2h0 * dTile_], x2[x2h1 * dTile_], tD);
            } else {
                MhcNormRegBase::VFPairSquareAdd(reduceRows, x2[x2h0 * dTile_], x2[x2h1 * dTile_], tD);
            }

            LocalTensor<float> curPhi0;
            if (!firstPhiPrefetched) {
                curPhi0 = bufPhiFp32_.AllocTensor<float>();
                DataCopyPad(curPhi0, phiGm_[(uint64_t)h0 * D_ + tOff], copyParams, copyPad);
                bufPhiFp32_.EnQue(curPhi0);
            }

            for (uint32_t o = 0; o < N; ++o) {
                // 预取下一轮o+1的两路phi，与当前dot计算重叠。
                LocalTensor<float> nextPhi0;
                if (o + 1U < N) {
                    nextPhi0 = bufPhiFp32_.AllocTensor<float>();
                    DataCopyPad(nextPhi0, phiGm_[(uint64_t)(o + 1U) * phiCols_ + (uint64_t)h0 * D_ + tOff], copyParams,
                                copyPad);
                    bufPhiFp32_.EnQue(nextPhi0);
                }

                LocalTensor<float> dotRow = reduceRows[(o + 1U) * reduceWidth];
                curPhi0 = bufPhiFp32_.DeQue<float>();
                if (compactRows) {
                    if (accumulateRows) {
                        MhcNormRegBase::VFPairMulAddCompact<true>(dotRow, x2[x2h0 * dTile_], curPhi0,
                                                                   x2[x2h1 * dTile_], curPhi0[dTile_], tD);
                    } else {
                        MhcNormRegBase::VFPairMulAddCompact(dotRow, x2[x2h0 * dTile_], curPhi0,
                                                            x2[x2h1 * dTile_], curPhi0[dTile_], tD);
                    }
                } else if (accumulateRows) {
                    MhcNormRegBase::VFPairMulAdd<true>(dotRow, x2[x2h0 * dTile_], curPhi0, x2[x2h1 * dTile_],
                                                       curPhi0[dTile_], tD);
                } else {
                    MhcNormRegBase::VFPairMulAdd(dotRow, x2[x2h0 * dTile_], curPhi0, x2[x2h1 * dTile_],
                                                 curPhi0[dTile_], tD);
                }
                bufPhiFp32_.FreeTensor(curPhi0);

                if (o + 1U < N) {
                    curPhi0 = nextPhi0;
                }
            }

            PipeBarrier<PIPE_V>();
            if (deferReduce && tOff + tD < D_) {
                continue;
            }
            LocalTensor<float> scBuf = bufScalar_.Get<float>();
            uint32_t reduceShape[2] = {N + 1U, reduceWidth};
            // 五行输入在归约后不再使用，允许接口复用源缓冲，避免额外申请UB。
            ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(scBuf, reduceRows, reduceShape, true);
            PipeBarrier<PIPE_V>();
            sumSqHalf[half] += scBuf.GetValue(0);
            for (uint32_t o = 0; o < N; ++o) {
                dotHalf[half][o] += scBuf.GetValue(o + 1);
            }
        }
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::CastToIO(LocalTensor<T> dst, LocalTensor<float> src, uint32_t count)
{
        MhcNormRegBase::VFCastFp32ToB16(dst, src, count);
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ComputeAndCopyOutHInPrimeTile(LocalTensor<T> srcDst, LocalTensor<float> gamma, float scale,
                                                         uint64_t gmOffset, uint32_t count)
{
        MhcNormRegBase::VFCastB16ScaleAndMulGammaToB16(srcDst, srcDst, gamma, scale, count);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(evtVtoMte3_);
        WaitFlag<HardEvent::V_MTE3>(evtVtoMte3_);
        DataCopy(hInPrimeGm_[gmOffset], srcDst, count);
        // 复用x2缓存前，确保MTE3已读取本轮输出归一化的低精度结果。
        FenceMte3ToV();
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ComputeGateValues(float totalSumSqPre, float (&xHatVals)[N])
{
        // Host已固定校验N为4，使用编译期常量便于展开标量循环。
        float invRmsPre = 1.0f / sqrt(totalSumSqPre * invK_ + mhcPreNormEps_);
        for (uint32_t o = 0; o < N; ++o)
            xHatVals[o] *= invRmsPre;

        LocalTensor<float> scBuf = bufScalar_.Get<float>();
        for (uint32_t i = 0; i < N; ++i)
            scBuf.SetValue(i, -(alpha0_ * xHatVals[i] + biasCache_[i]));
        LocalTensor<float> gateBuf = scBuf[EXP_DIV_ELEM];
        // TODO：后续单独评估固定四元素循环展开，本轮保持通用VF循环。
        MhcNormRegBase::VFSigmoidAddEps(gateBuf, scBuf, hcEps_, N);
        PipeBarrier<PIPE_V>();
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline float KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ComputeHInCacheAndReduceSumSq(uint32_t tOff, uint32_t tD, LocalTensor<T> x2Cache,
                                                          LocalTensor<float> gateBuf)
{
        LocalTensor<T> hInCache = bufActivationLowp_.Get<T>()[tOff];
        LocalTensor<float> sqBuf = bufScratchFp32_.Get<float>();
        MhcNormRegBase::VFGateWeightedSumQuantizeSquare<T>(hInCache, sqBuf, x2Cache, D_, gateBuf, tD);
        PipeBarrier<PIPE_V>();
        return ReduceSumPrecise(sqBuf, tD);
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline __attribute__((always_inline)) float KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ComputeInvRms(float sumSq, float invCount)
{
        return 1.0f / sqrt(sumSq * invCount + normEps_);
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ComputeX2AllHeadsBatched(const float (&postVals)[N], uint64_t gmBaseOffset,
                                                    uint32_t tOff, uint32_t tD)
{
        // Host已固定校验N为4，使用编译期常量便于展开标量循环。
        LocalTensor<T> residual = queResidualLowp_.AllocTensor<T>();
        for (uint32_t n = 0; n < N; ++n)
            DataCopy(residual[n * dTile_], residualGm_[gmBaseOffset + (uint64_t)n * D_], tD);
        queResidualLowp_.EnQue(residual);
        residual = queResidualLowp_.DeQue<T>();
        LocalTensor<T> x2Cache = bufActivationLowp_.Get<T>();
        LocalTensor<float> hOut = bufResFp32_.Get<float>()[N * dTile_];
        LocalTensor<float> combWeights = bufPostComb_.Get<float>()[ALIGN8];

        for (uint32_t h = 0; h < N; h += 2U) {
            MhcNormRegBase::VFComputePostHeadPair<T>(x2Cache[h * D_ + tOff], x2Cache[(h + 1U) * D_ + tOff], hOut,
                                                     residual, combWeights, dTile_, h, postVals[h], postVals[h + 1U],
                                                     tD);
        }
        PipeBarrier<PIPE_V>();
        queResidualLowp_.FreeTensor(residual);
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ComputeX2PostTile(uint32_t token, uint32_t tOff, uint32_t tD, float invRms0,
                                             const float (&postVals)[N])
{
        LocalTensor<float> hOut = bufResFp32_.Get<float>()[N * dTile_];
        LocalTensor<T> hOutCache = bufActivationLowp_.Get<T>()[tOff];
        MhcNormRegBase::VFCastB16ScaleGammaQuantizeToFp32(hOut, hOutCache, bufGammaFp32_.Get<float>()[tOff], invRms0,
                                                          tD);
        PipeBarrier<PIPE_V>();

        ComputeX2AllHeadsBatched(postVals, (uint64_t)token * N_ * D_ + tOff, tOff, tD);
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ComputeX2PostTileForSiblingAiv(uint32_t token, uint32_t tOff, uint32_t tD, float invRms0,
                                                           const float (&postVals)[N])
{
        LocalTensor<float> hOut = bufResFp32_.Get<float>()[N * dTile_];
        LocalTensor<T> hOutCache = bufActivationLowp_.Get<T>()[tOff];

        // 提前发起residual搬运（MTE2），与下方h_out的scale/gamma/quantize（Vector）重叠。
        const uint32_t nStart = siblingAivIdx_ * HEADS_PER_SIBLING;
        const uint64_t gmBaseOffset = (uint64_t)token * N_ * D_ + tOff;
        LocalTensor<T> residual = queResidualLowp_.AllocTensor<T>();
        for (uint32_t n = 0; n < N; ++n) {
            DataCopy(residual[n * dTile_], residualGm_[gmBaseOffset + (uint64_t)n * D_], tD);
        }
        queResidualLowp_.EnQue(residual);

        MhcNormRegBase::VFCastB16ScaleGammaQuantizeToFp32(hOut, hOutCache, bufGammaFp32_.Get<float>()[tOff], invRms0,
                                                          tD);
        PipeBarrier<PIPE_V>();

        residual = queResidualLowp_.DeQue<T>();
        LocalTensor<T> x2Cache = bufActivationLowp_.Get<T>();
        LocalTensor<float> combWeights = bufPostComb_.Get<float>()[ALIGN8];
        MhcNormRegBase::VFComputePostHeadPair<T>(x2Cache[nStart * D_ + tOff], x2Cache[(nStart + 1U) * D_ + tOff],
                                                 hOut, residual, combWeights, dTile_, nStart, postVals[nStart],
                                                 postVals[nStart + 1U], tD);
        PipeBarrier<PIPE_V>();
        queResidualLowp_.FreeTensor(residual);
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::CopyInGamma0()
{
        LocalTensor<float> gamma0 = bufGammaFp32_.Get<float>();
        DataCopyExtParams params = {1, 0, 0, 0, 0};
        params.blockLen = D_ * sizeof(float);
        DataCopyPadExtParams<float> pad = {false, 0, 0, 0};
        DataCopyPad(gamma0, gamma0Gm_, params, pad);
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::CopyInGammaTile(LocalTensor<float> dst, GlobalTensor<float> &src, uint32_t tOff, uint32_t tD)
{
        DataCopyExtParams gP = {1, 0, 0, 0, 0};
        gP.blockLen = tD * sizeof(float);
        DataCopyPadExtParams<float> gPad = {false, 0, 0, 0};
        DataCopyPad(dst, src[tOff], gP, gPad);
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::CopyInHOut(uint64_t gmOffset)
{
        LocalTensor<T> hOutCache = bufActivationLowp_.Get<T>();
        DataCopy(hOutCache, hOutGm_[gmOffset], D_);
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::CopyInTokenWeights(uint32_t token, float (&postVals)[N], uint32_t nStart,
                                              uint32_t nEnd)
{
        // Host已固定校验N为4，使用编译期常量便于展开标量循环。
        LocalTensor<float> postCombUb = bufPostComb_.Get<float>();
        LocalTensor<float> hPostUb = postCombUb;
        LocalTensor<float> hResUb = postCombUb[ALIGN8]; // h_post[N] aligned to 8 elements
        {
            DataCopyExtParams pp = {1, 0, 0, 0, 0};
            pp.blockLen = N * sizeof(float);
            DataCopyPadExtParams<float> pad = {true, 0, 0, 0};
            DataCopyPad(hPostUb, hPostGm_[(uint64_t)token * N], pp, pad);

            DataCopyExtParams cp = {1, 0, 0, 0, 0};
            cp.blockLen = N * N * sizeof(float);
            DataCopyPadExtParams<float> cpad = {true, 0, 0, 0};
            DataCopyPad(hResUb, hResGm_[(uint64_t)token * N * N], cp, cpad);
        }
        PipeBarrier<PIPE_ALL>();
        // 每个sibling AIV只消费自己负责的两个post标量，避免无效UB到Scalar回读。
        for (uint32_t i = nStart; i < nEnd; ++i)
            postVals[i] = hPostUb.GetValue(i);
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::CopyOutHInPrimeTile(LocalTensor<float> src, uint64_t gmOffset, uint32_t count)
{
        CopyOutLowPrecisionTile(hInPrimeGm_, gmOffset, src, count);
        if constexpr (RETURN_H_IN_F32) {
            SetFlag<HardEvent::V_MTE3>(evtVtoMte3_);
            WaitFlag<HardEvent::V_MTE3>(evtVtoMte3_);
            DataCopy(hInF32Gm_[gmOffset], src, count);
            FenceMte3ToV();
        }
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::CopyOutLowPrecisionTile(GlobalTensor<T> &dst, uint64_t gmOffset, LocalTensor<float> src,
                                                    uint32_t count)
{
        LocalTensor<T> tmp = bufHInPrimeBf16_.AllocTensor<T>();
        CastToIO(tmp, src, count);
        bufHInPrimeBf16_.EnQue(tmp);
        tmp = bufHInPrimeBf16_.DeQue<T>();
        DataCopy(dst[gmOffset], tmp, count);
        bufHInPrimeBf16_.FreeTensor(tmp);
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::FenceMte3ToV()
{
        SetFlag<HardEvent::MTE3_V>(evtMte3toV_);
        WaitFlag<HardEvent::MTE3_V>(evtMte3toV_);
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::FenceVToMte2()
{
        SetFlag<HardEvent::V_MTE2>(evtVtoMte2_);
        WaitFlag<HardEvent::V_MTE2>(evtVtoMte2_);
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::FinalizeHInPrime(
    uint32_t token, float sumSq1)
{
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
__aicore__ inline uint32_t KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::GetTileD(uint32_t tile) const
{
        if (lastDTile_ != 0 && tile == dLoop_ - 1) {
            return lastDTile_;
        }
        return dTile_;
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::Init(GM_ADDR h_out_gm, GM_ADDR residual_gm, GM_ADDR h_post_gm, GM_ADDR h_res_gm,
                                GM_ADDR phi_gm, GM_ADDR alpha_gm, GM_ADDR bias_gm, GM_ADDR pre_mix_gm,
                                GM_ADDR gamma_0_gm, GM_ADDR gamma_1_gm, GM_ADDR gamma_2_gm, GM_ADDR h_in_prime_gm,
                                GM_ADDR x_2_out_gm, GM_ADDR h_in_f32_gm, GM_ADDR h_pre_gm, GM_ADDR workspace,
                                GM_ADDR tilingDataGm)
{
        GET_TILING_DATA(tilingData, tilingDataGm);

        // pre_mix 为接口预留的可选输入，h_pre 为接口预留的可选输出，当前均不参与计算。
        (void)pre_mix_gm;
        (void)h_pre_gm;

        T_ = tilingData.T;
        N_ = tilingData.N;
        D_ = tilingData.D;
        usedCoreNum_ = tilingData.usedCoreNum;
        baseTPerCore_ = tilingData.baseTPerCore;
        remainderT_ = tilingData.remainderT;
        dTile_ = tilingData.dTile;
        dLoop_ = tilingData.dLoop;
        lastDTile_ = tilingData.lastDTile;
        phiCols_ = tilingData.phiCols;
        normEps_ = tilingData.normEps;
        mhcPreNormEps_ = tilingData.mhcPreNormEps;
        hcEps_ = tilingData.hcEps;
        K_ = N_ * D_;
        invD_ = tilingData.invD;
        invK_ = tilingData.invK;

        const uint32_t aivIdx = GetBlockIdx();
        // MIX_AIC_1_2按每个物理AI Core展开两个连续AIV。
        siblingAivIdx_ = DUAL_SIBLING ? (aivIdx & 1U) : 0U;
        const uint32_t coreIdx = DUAL_SIBLING ? (aivIdx >> 1U) : aivIdx;
        if (coreIdx >= usedCoreNum_) {
            tStart_ = 0;
            tEnd_ = 0;
            return;
        }

        // 单AIV按token分区；双sibling通过aivIdx奇偶映射相邻sibling AIV，并在后半段收敛到偶数leader。
        if (coreIdx < remainderT_) {
            tStart_ = coreIdx * (baseTPerCore_ + 1U);
            tEnd_ = tStart_ + baseTPerCore_ + 1U;
        } else {
            tStart_ = remainderT_ * (baseTPerCore_ + 1U) + (coreIdx - remainderT_) * baseTPerCore_;
            tEnd_ = tStart_ + baseTPerCore_;
        }
        if (tEnd_ > T_)
            tEnd_ = T_;

        hOutGm_.SetGlobalBuffer((__gm__ T *)h_out_gm, (uint64_t)T_ * D_);
        residualGm_.SetGlobalBuffer((__gm__ T *)residual_gm, (uint64_t)T_ * N_ * D_);
        hPostGm_.SetGlobalBuffer((__gm__ float *)h_post_gm, (uint64_t)T_ * N_);
        hResGm_.SetGlobalBuffer((__gm__ float *)h_res_gm, (uint64_t)T_ * N_ * N_);
        phiGm_.SetGlobalBuffer((__gm__ float *)phi_gm, (uint64_t)N_ * phiCols_);
        alpha0Gm_.SetGlobalBuffer((__gm__ float *)alpha_gm, ALIGN8);
        biasGm_.SetGlobalBuffer((__gm__ float *)bias_gm, ALIGN8);
        gamma0Gm_.SetGlobalBuffer((__gm__ float *)gamma_0_gm, D_);
        gamma1Gm_.SetGlobalBuffer((__gm__ float *)gamma_1_gm, D_);
        if constexpr (USE_RMS_MID) {
            gamma2Gm_.SetGlobalBuffer((__gm__ float *)gamma_2_gm, K_);
        }
        hInPrimeGm_.SetGlobalBuffer((__gm__ T *)h_in_prime_gm, (uint64_t)T_ * D_);
        x2OutGm_.SetGlobalBuffer((__gm__ T *)x_2_out_gm, (uint64_t)T_ * N_ * D_);
        if (h_in_f32_gm != nullptr) {
            hInF32Gm_.SetGlobalBuffer((__gm__ float *)h_in_f32_gm, (uint64_t)T_ * D_);
        }
        if constexpr (DUAL_SIBLING) {
            constexpr uint32_t slotsPerToken = USE_RMS_MID ? HAS_MID_SLOTS_PER_TOKEN : SIBLING_AIV_COUNT;
            partial1Gm_.SetGlobalBuffer((__gm__ float *)workspace,
                                        (uint64_t)T_ * slotsPerToken * PARTIAL_SLOT_ELEMS);
        }

        pipe_.InitBuffer(bufGammaFp32_, D_ * sizeof(float));
        const uint32_t x2Fp32Rows = DUAL_SIBLING ? HEADS_PER_SIBLING : N;
        pipe_.InitBuffer(bufX2Fp32_, x2Fp32Rows * dTile_ * sizeof(float));
        pipe_.InitBuffer(bufActivationLowp_, K_ * sizeof(T));
        pipe_.InitBuffer(queResidualLowp_, 1, N_ * dTile_ * sizeof(T));
        pipe_.InitBuffer(bufResFp32_, (N_ + 1) * dTile_ * sizeof(float));
        pipe_.InitBuffer(bufPostComb_, POST_COMB_SIZE * sizeof(float));
        pipe_.InitBuffer(bufPhiFp32_, 2, 2U * dTile_ * sizeof(float));
        pipe_.InitBuffer(bufHInPrimeBf16_, BUFFER_NUM_ONE, dTile_ * sizeof(T));
        pipe_.InitBuffer(bufScalar_, SCALAR_BUF_SIZE);
        const uint32_t gamma1CacheElems = (DUAL_SIBLING && USE_RMS_MID) || D_ <= GAMMA1_CACHE_MAX_D ? D_ : dTile_;
        pipe_.InitBuffer(bufScratchFp32_, gamma1CacheElems * sizeof(float));

        evtMte2toV_ = static_cast<int32_t>(pipe_.AllocEventID<HardEvent::MTE2_V>());
        evtVtoMte2_ = static_cast<int32_t>(pipe_.AllocEventID<HardEvent::V_MTE2>());
        evtMte3toV_ = static_cast<int32_t>(pipe_.AllocEventID<HardEvent::MTE3_V>());
        evtVtoMte3_ = static_cast<int32_t>(pipe_.AllocEventID<HardEvent::V_MTE3>());
        if constexpr (DUAL_SIBLING) {
            evtMte2toS_ = static_cast<int32_t>(pipe_.AllocEventID<HardEvent::MTE2_S>());
            evtStoMte3_ = static_cast<int32_t>(pipe_.AllocEventID<HardEvent::S_MTE3>());
        }
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::Process()
{
        if (tStart_ >= tEnd_) {
            return;
        }
        // 双AIV的follower在会合后退出，不参与gate计算，省去五次无效标量GM读取。
        if constexpr (!DUAL_SIBLING) {
            alpha0_ = alpha0Gm_.GetValue(0);
            for (uint32_t i = 0; i < N; ++i) {
                biasCache_[i] = biasGm_.GetValue(i);
            }
        } else if (siblingAivIdx_ == 0U) {
            alpha0_ = alpha0Gm_.GetValue(0);
            for (uint32_t i = 0; i < N; ++i) {
                biasCache_[i] = biasGm_.GetValue(i);
            }
        }

        // 仅单分块路径提前加载gamma
        if (dLoop_ == 1) {
            DataCopyExtParams gP = {1, 0, 0, 0, 0};
            gP.blockLen = dTile_ * sizeof(float);
            DataCopyPadExtParams<float> gPad = {false, 0, 0, 0};
            DataCopyPad(bufGammaFp32_.Get<float>(), gamma0Gm_[0], gP, gPad);
            // gamma_2的形状为[N*D]，在RMSNorm_mid中按head加载
        }

        SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
        WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);

        for (uint32_t t = tStart_; t < tEnd_; ++t) {
            ProcessToken(t);
        }

        pipe_.ReleaseEventID<HardEvent::MTE2_V>(evtMte2toV_);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(evtVtoMte2_);
        pipe_.ReleaseEventID<HardEvent::MTE3_V>(evtMte3toV_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(evtVtoMte3_);
        if constexpr (DUAL_SIBLING) {
            pipe_.ReleaseEventID<HardEvent::MTE2_S>(evtMte2toS_);
            pipe_.ReleaseEventID<HardEvent::S_MTE3>(evtStoMte3_);
        }
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline float KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ReduceSumPrecise(LocalTensor<float> src, uint32_t tD)
{
        LocalTensor<float> scBuf = bufScalar_.Get<float>();
        constexpr uint32_t REDUCE_ALIGN = 16; // ReduceSum要求16个元素对齐
        uint32_t tDAligned = (tD + REDUCE_ALIGN - 1) / REDUCE_ALIGN * REDUCE_ALIGN;
        uint32_t srcShape[2] = {1, tDAligned};
        ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(scBuf, src, srcShape, true);
        PipeBarrier<PIPE_V>();
        return scBuf.GetValue(0);
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline float KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ReduceSumSq(LocalTensor<float> src, uint32_t tD)
{
        // gamma2缓存与平方归约不同时存活，复用同一块UB作为平方临时区。
        LocalTensor<float> sqBuf = bufScratchFp32_.Get<float>();
        MhcNormRegBase::VFMul(sqBuf, src, src, tD);
        PipeBarrier<PIPE_V>();
        return ReduceSumPrecise(sqBuf, tD);
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline float KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ReduceSumSqFromCache(uint32_t tOff, uint32_t tD)
{
        LocalTensor<T> hOutCache = bufActivationLowp_.Get<T>()[tOff];
        LocalTensor<float> sqBuf = bufScratchFp32_.Get<float>();
        MhcNormRegBase::VFCastB16ToFp32Square(sqBuf, hOutCache, tD);
        PipeBarrier<PIPE_V>();
        return ReduceSumPrecise(sqBuf, tD);
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::RestoreAndCopyOutX2Tile(uint32_t token, uint32_t tOff, uint32_t tD)
{
        LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
        LocalTensor<T> x2Cache = bufActivationLowp_.Get<T>();
        const uint64_t outBase = (uint64_t)token * N_ * D_ + tOff;

        for (uint32_t n = 0; n < N; n += 2U) {
            MhcNormRegBase::VFCastB16PairToFp32(x2[n * dTile_], x2[(n + 1U) * dTile_], x2Cache[n * D_ + tOff],
                                                x2Cache[(n + 1U) * D_ + tOff], tD);
        }
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(evtVtoMte3_);
        WaitFlag<HardEvent::V_MTE3>(evtVtoMte3_);
        for (uint32_t n = 0; n < N; ++n) {
            DataCopy(x2OutGm_[outBase + (uint64_t)n * D_], x2Cache[n * D_ + tOff], tD);
        }
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::RestoreAndCopyOutX2TileForSiblingAiv(uint32_t token, uint32_t tOff, uint32_t tD)
{
        LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
        LocalTensor<T> x2Cache = bufActivationLowp_.Get<T>();
        const uint64_t outBase = (uint64_t)token * N_ * D_ + tOff;
        const uint32_t nStart = siblingAivIdx_ * HEADS_PER_SIBLING;

        MhcNormRegBase::VFCastB16PairToFp32(x2, x2[dTile_], x2Cache[nStart * D_ + tOff],
                                            x2Cache[(nStart + 1U) * D_ + tOff], tD);
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(evtVtoMte3_);
        WaitFlag<HardEvent::V_MTE3>(evtVtoMte3_);
        for (uint32_t n = nStart; n < nStart + HEADS_PER_SIBLING; ++n) {
            DataCopy(x2OutGm_[outBase + (uint64_t)n * D_], x2Cache[n * D_ + tOff], tD);
        }
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::StartCopyInGamma1()
{
        LocalTensor<float> gamma1 = bufGammaFp32_.Get<float>();
        DataCopyExtParams params = {1, 0, 0, 0, 0};
        params.blockLen = D_ * sizeof(float);
        DataCopyPadExtParams<float> pad = {false, 0, 0, 0};
        DataCopyPad(gamma1, gamma1Gm_, params, pad);
        SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::WaitCopyInGamma1()
{
        WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);
    }

#endif // MHC_NORM_POST_PREONLY_KERNEL_IMPL_H_
