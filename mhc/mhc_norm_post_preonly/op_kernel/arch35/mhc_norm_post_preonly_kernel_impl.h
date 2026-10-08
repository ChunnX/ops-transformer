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
__aicore__ inline void
KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::AccumulateHalfPartials(
    uint32_t tOff, uint32_t tD, float (&sumSqHalf)[2], float (&dotHalf)[2][N_MAX], uint32_t halfBegin, uint32_t halfEnd)
{
    LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
    LocalTensor<float> reduceRows = bufResFp32_.Get<float>();
    LocalTensor<float> tmp = bufSqFp32_.Get<float>();

    DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
    copyParams.blockLen = tD * sizeof(float);
    DataCopyPadExtParams<float> copyPad = {false, 0, 0, 0};

    // 尾块的物理行跨度仍为dimTile_，不能作为紧凑二维张量归约；
    // 尾块保持逐行归约，完整分块才使用五行批量归约。
    if (tD != dimTile_) {
        LocalTensor<float> out = reduceRows[N_MAX * dimTile_];
        for (uint32_t half = halfBegin; half < halfEnd; ++half) {
            const uint32_t h0 = half * 2;
            const uint32_t h1 = h0 + 1;

            MhcSandwichRegBase::Mul(out, x2[h0 * dimTile_], x2[h0 * dimTile_], tD);
            PipeBarrier<PIPE_V>();
            MhcSandwichRegBase::Mul(tmp, x2[h1 * dimTile_], x2[h1 * dimTile_], tD);
            PipeBarrier<PIPE_V>();
            MhcSandwichRegBase::Add(out, out, tmp, tD);
            PipeBarrier<PIPE_V>();
            sumSqHalf[half] += ReduceSumPrecise(out, tD);

            for (uint32_t o = 0; o < N_MAX; ++o) {
                LocalTensor<float> phi = bufPhiFp32_.AllocTensor<float>();
                uint64_t phiOffset = (uint64_t)o * phiCols_ + (uint64_t)h0 * D_ + tOff;
                DataCopyPad(phi, phiGm_[phiOffset], copyParams, copyPad);
                bufPhiFp32_.EnQue(phi);
                phi = bufPhiFp32_.DeQue<float>();
                MhcSandwichRegBase::Mul(out, x2[h0 * dimTile_], phi, tD);
                PipeBarrier<PIPE_V>();
                bufPhiFp32_.FreeTensor(phi);

                phi = bufPhiFp32_.AllocTensor<float>();
                phiOffset = (uint64_t)o * phiCols_ + (uint64_t)h1 * D_ + tOff;
                DataCopyPad(phi, phiGm_[phiOffset], copyParams, copyPad);
                bufPhiFp32_.EnQue(phi);
                phi = bufPhiFp32_.DeQue<float>();
                MhcSandwichRegBase::Mul(tmp, x2[h1 * dimTile_], phi, tD);
                PipeBarrier<PIPE_V>();
                bufPhiFp32_.FreeTensor(phi);

                MhcSandwichRegBase::Add(out, out, tmp, tD);
                PipeBarrier<PIPE_V>();
                dotHalf[half][o] += ReduceSumPrecise(out, tD);
            }
        }
        return;
    }

    for (uint32_t half = halfBegin; half < halfEnd; ++half) {
        const uint32_t h0 = half * 2;
        const uint32_t h1 = h0 + 1;

        // 五行依次保存{sumSqPre,dot0,dot1,dot2,dot3}，一次按行归约，
        // 避免每个标量结果都单独触发一次Vector到Scalar的等待。
        MhcSandwichRegBase::Mul(reduceRows, x2[h0 * dimTile_], x2[h0 * dimTile_], tD);
        PipeBarrier<PIPE_V>();
        MhcSandwichRegBase::Mul(tmp, x2[h1 * dimTile_], x2[h1 * dimTile_], tD);
        PipeBarrier<PIPE_V>();
        MhcSandwichRegBase::Add(reduceRows, reduceRows, tmp, tD);

        for (uint32_t o = 0; o < N_MAX; ++o) {
            LocalTensor<float> dotRow = reduceRows[(o + 1) * dimTile_];
            LocalTensor<float> phi = bufPhiFp32_.AllocTensor<float>();
            uint64_t phiOffset = (uint64_t)o * phiCols_ + (uint64_t)h0 * D_ + tOff;
            DataCopyPad(phi, phiGm_[phiOffset], copyParams, copyPad);
            bufPhiFp32_.EnQue(phi);
            phi = bufPhiFp32_.DeQue<float>();
            MhcSandwichRegBase::Mul(dotRow, x2[h0 * dimTile_], phi, tD);
            PipeBarrier<PIPE_V>();
            bufPhiFp32_.FreeTensor(phi);

            phi = bufPhiFp32_.AllocTensor<float>();
            phiOffset = (uint64_t)o * phiCols_ + (uint64_t)h1 * D_ + tOff;
            DataCopyPad(phi, phiGm_[phiOffset], copyParams, copyPad);
            bufPhiFp32_.EnQue(phi);
            phi = bufPhiFp32_.DeQue<float>();
            MhcSandwichRegBase::Mul(tmp, x2[h1 * dimTile_], phi, tD);
            PipeBarrier<PIPE_V>();
            bufPhiFp32_.FreeTensor(phi);

            MhcSandwichRegBase::Add(dotRow, dotRow, tmp, tD);
        }

        PipeBarrier<PIPE_V>();
        LocalTensor<float> scBuf = bufScalar_.Get<float>();
        uint32_t reduceShape[2] = {N_MAX + 1, tD};
        // 五行输入在归约后不再使用，允许接口复用源缓冲，避免额外申请UB。
        ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(scBuf, reduceRows, reduceShape, true);
        PipeBarrier<PIPE_V>();
        sumSqHalf[half] += scBuf.GetValue(0);
        for (uint32_t o = 0; o < N_MAX; ++o) {
            dotHalf[half][o] += scBuf.GetValue(o + 1);
        }
    }
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ApplyScaleAndGamma(LocalTensor<float> buf, float scale, LocalTensor<float> gamma, uint32_t tD)
{
    // TODO：完成多VF融合和双缓冲后，执行Membase回退消融，复测改回高层接口的组合收益。
    MhcSandwichRegBase::ScaleAndMulGamma(buf, buf, gamma, scale, tD);
    PipeBarrier<PIPE_V>();
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::CacheHInTile(
    uint32_t tOff, uint32_t tD, LocalTensor<float> hIn)
{
    LocalTensor<T> hInCache = bufX2Cache_.Get<T>()[tOff];
    CastToIO(hInCache, hIn, tD);
    PipeBarrier<PIPE_V>();
    MhcSandwichRegBase::CastB16ToFp32(hIn, hInCache, tD);
    PipeBarrier<PIPE_V>();
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::CacheX2Tile(
    uint32_t token, uint32_t tOff, uint32_t tD, bool writeOutput)
{
    LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
    LocalTensor<T> x2Cache = bufX2Cache_.Get<T>();
    const uint64_t outBase = (uint64_t)token * N_ * D_ + tOff;

    // 四个head写入互不重叠的低精度缓存，统一等待后再恢复FP32，
    // 保留逐元素量化边界并减少重复的Vector流水屏障。
    for (uint32_t n = 0; n < N_MAX; ++n) {
        LocalTensor<T> cacheTile = x2Cache[n * D_ + tOff];
        CastToIO(cacheTile, x2[n * dimTile_], tD);
    }
    PipeBarrier<PIPE_V>();
    for (uint32_t n = 0; n < N_MAX; ++n) {
        LocalTensor<T> cacheTile = x2Cache[n * D_ + tOff];
        MhcSandwichRegBase::CastB16ToFp32(x2[n * dimTile_], cacheTile, tD);
    }
    PipeBarrier<PIPE_V>();

    if (writeOutput) {
        // cacheTile已经是x2_out所需的最终低精度位值，直接写GM，
        // 避免先恢复FP32再重复量化和反复申请输出队列。
        SetFlag<HardEvent::V_MTE3>(evtVtoMte3_);
        WaitFlag<HardEvent::V_MTE3>(evtVtoMte3_);
        for (uint32_t n = 0; n < N_MAX; ++n) {
            DataCopy(x2OutGm_[outBase + (uint64_t)n * D_], x2Cache[n * D_ + tOff], tD);
        }
    }
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32,
                                                        DUAL_SIBLING>::CacheX2TileForLane(uint32_t token, uint32_t tOff,
                                                                                          uint32_t tD, bool writeOutput)
{
    LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
    LocalTensor<T> x2Cache = bufX2Cache_.Get<T>();
    const uint64_t outBase = (uint64_t)token * N_ * D_ + tOff;

    for (uint32_t n = 0; n < N_MAX; ++n) {
        CastToIO(x2Cache[n * D_ + tOff], x2[n * dimTile_], tD);
    }
    PipeBarrier<PIPE_V>();
    for (uint32_t n = 0; n < N_MAX; ++n) {
        MhcSandwichRegBase::CastB16ToFp32(x2[n * dimTile_], x2Cache[n * D_ + tOff], tD);
    }
    PipeBarrier<PIPE_V>();

    if (writeOutput) {
        SetFlag<HardEvent::V_MTE3>(evtVtoMte3_);
        WaitFlag<HardEvent::V_MTE3>(evtVtoMte3_);
        const uint32_t headBegin = laneId_ * 2U;
        for (uint32_t n = headBegin; n < headBegin + 2U; ++n) {
            DataCopy(x2OutGm_[outBase + (uint64_t)n * D_], x2Cache[n * D_ + tOff], tD);
        }
    }
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::CastToIO(LocalTensor<T> dst, LocalTensor<float> src, uint32_t count)
{
    MhcSandwichRegBase::CastFp32ToB16(dst, src, count);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ComputeGateValues(float totalSumSqPre, float (&xHatVals)[N_MAX], float (&gate)[N_MAX])
{
    // Host已固定校验N为4，使用编译期常量便于展开标量循环。
    constexpr uint32_t N = N_MAX;
    float invRmsPre = 1.0f / sqrt(totalSumSqPre * invK_ + mhcPreNormEps_);
    for (uint32_t o = 0; o < N; ++o)
        xHatVals[o] *= invRmsPre;

    LocalTensor<float> scBuf = bufScalar_.Get<float>();
    for (uint32_t i = 0; i < N; ++i)
        scBuf.SetValue(i, -(alpha0_ * xHatVals[i] + biasCache_[i]));
    LocalTensor<float> gateBuf = scBuf[EXP_DIV_ELEM];
    // TODO：后续单独评估固定四元素循环展开，本轮保持通用VF循环。
    MhcSandwichRegBase::SigmoidAddEps(gateBuf, scBuf, hcEps_, N);
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < N; ++i)
        gate[i] = gateBuf.GetValue(i);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline __attribute__((always_inline)) float KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ComputeInvRms(float sumSq, float invCount)
{
    return 1.0f / sqrt(sumSq * invCount + normEps_);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ComputeX2AllHeadsBatched(const float (&postVals)[N_MAX], const float (&combVals)[N_MAX * N_MAX],
                                                uint32_t tD)
{
    // Host已固定校验N为4，使用编译期常量便于展开标量循环。
    constexpr uint32_t N = N_MAX;
    LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
    LocalTensor<float> hOut = bufHOutFp32_.Get<float>();
    LocalTensor<float> res = bufResFp32_.Get<float>();

    for (uint32_t h = 0; h < N; ++h) {
        float headComb[N_MAX];
        for (uint32_t j = 0; j < N; ++j) {
            headComb[j] = combVals[j * N + h];
        }
        MhcSandwichRegBase::ComputePostHead(x2[h * dimTile_], hOut, res, dimTile_, postVals[h], headComb, tD);
    }
    PipeBarrier<PIPE_V>();
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ComputeX2AllHeadsSerial(const float (&postVals)[N_MAX], const float (&combVals)[N_MAX * N_MAX],
                                               uint32_t tD)
{
    // Host已固定校验N为4，使用编译期常量便于展开标量循环。
    constexpr uint32_t N = N_MAX;
    LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
    LocalTensor<float> hOut = bufHOutFp32_.Get<float>();
    LocalTensor<float> res = bufResFp32_.Get<float>();

    for (uint32_t h = 0; h < N; ++h) {
        float headComb[N_MAX];
        for (uint32_t j = 0; j < N; ++j) {
            headComb[j] = combVals[j * N + h];
        }
        MhcSandwichRegBase::ComputePostHead(x2[h * dimTile_], hOut, res, dimTile_, postVals[h], headComb, tD);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void
KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ComputeX2PostTile(
    uint32_t token, uint32_t tOff, uint32_t tD, float invRms0, const float (&postVals)[N_MAX],
    const float (&combVals)[N_MAX * N_MAX])
{
    LoadHOutAndCast((uint64_t)token * D_ + tOff, tD);
    LoadGammaInto(bufGamma0Fp32_.Get<float>(), gamma0Gm_, tOff, tD);
    SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
    WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);
    ApplyScaleAndGamma(bufHOutFp32_.Get<float>(), invRms0, bufGamma0Fp32_.Get<float>(), tD);
    FenceVToMte2();
    QuantizeRoundTrip(bufHOutFp32_.Get<float>(), tD);

    LoadResidualAndCast((uint64_t)token * N_ * D_ + tOff, tD);
    ComputeX2AllHeadsBatched(postVals, combVals, tD);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::FenceMte3ToMte2()
{
    SetFlag<HardEvent::MTE3_MTE2>(evtMte3toMte2_);
    WaitFlag<HardEvent::MTE3_MTE2>(evtMte3toMte2_);
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
__aicore__ inline uint32_t KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::GetTileD(uint32_t tile) const
{
    if (lastDimTile_ != 0 && tile == dimLoop_ - 1) {
        return lastDimTile_;
    }
    return dimTile_;
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::Init(
        GM_ADDR h_out_gm, GM_ADDR residual_gm,
        GM_ADDR h_post_gm, GM_ADDR h_res_gm,
        GM_ADDR phi_gm, GM_ADDR alpha_0_gm, GM_ADDR bias_gm,
        GM_ADDR gamma_0_gm, GM_ADDR gamma_1_gm, GM_ADDR gamma_2_gm,
        GM_ADDR h_in_prime_gm, GM_ADDR x_2_out_gm, GM_ADDR h_in_f32_gm,
        GM_ADDR workspace, GM_ADDR tiling)
{
        GET_TILING_DATA(td, tiling);

        totalTokens_    = td.totalTokens;
        N_              = td.headCount;
        D_              = td.embedDim;
        usedCoreNum_    = td.usedCoreNum;
        rowsPerCore_    = td.rowsPerCore;
        rowsPerCoreSp_  = td.rowsPerCoreSp;
        blockPivot_     = td.blockPivot;
        dimTile_        = td.dimTile;
        dimLoop_        = td.dimLoop;
        lastDimTile_    = td.lastDimTile;
        phiRows_        = td.phiRows;
        phiCols_        = td.phiCols;
        normEps_        = td.normEps;
        mhcPreNormEps_  = td.mhcPreNormEps;
        hcEps_          = td.hcEps;
        K_              = N_ * D_;
        invD_           = td.invD;
        invK_           = td.invK;
        coresPerToken_  = td.coresPerToken;

        const uint32_t aivIdx = GetBlockIdx();
        // MIX_AIC_1_2在A5上按每个物理AI Core展开两个连续AIV。
        // 该映射与同平台算子保持一致，只在双sibling模板中使用。
        laneId_ = DUAL_SIBLING ? (aivIdx & 1U) : 0U;
        const uint32_t blockIdx = DUAL_SIBLING ? (aivIdx >> 1U) : aivIdx;
        if (blockIdx >= usedCoreNum_) {
            tokenStart_ = 0;
            tokenEnd_ = 0;
            return;
        }

        // 第一阶段按token分区启动相互独立的AIV，不使用block奇偶配对、相邻block配对或N轴切分。
        if (blockIdx < blockPivot_) {
            tokenStart_ = blockIdx * rowsPerCoreSp_;
            tokenEnd_ = tokenStart_ + rowsPerCoreSp_;
        } else {
            tokenStart_ = blockPivot_ * rowsPerCoreSp_ + (blockIdx - blockPivot_) * rowsPerCore_;
            tokenEnd_ = tokenStart_ + rowsPerCore_;
        }
        if (tokenEnd_ > totalTokens_)
            tokenEnd_ = totalTokens_;

        // GM张量
        hOutGm_.SetGlobalBuffer((__gm__ T *)h_out_gm,         (uint64_t)totalTokens_ * D_);
        residualGm_.SetGlobalBuffer((__gm__ T *)residual_gm,  (uint64_t)totalTokens_ * N_ * D_);
        hPostGm_.SetGlobalBuffer((__gm__ float *)h_post_gm, (uint64_t)totalTokens_ * N_);
        hResGm_.SetGlobalBuffer((__gm__ float *)h_res_gm, (uint64_t)totalTokens_ * N_ * N_);
        phiGm_.SetGlobalBuffer((__gm__ float *)phi_gm,        (uint64_t)phiRows_ * phiCols_);
        alpha0Gm_.SetGlobalBuffer((__gm__ float *)alpha_0_gm, ALIGN8);
        biasGm_.SetGlobalBuffer((__gm__ float *)bias_gm,      ALIGN8);
        gamma0Gm_.SetGlobalBuffer((__gm__ float *)gamma_0_gm, D_);
        gamma1Gm_.SetGlobalBuffer((__gm__ float *)gamma_1_gm, D_);
        if constexpr (USE_RMS_MID) {
            gamma2Gm_.SetGlobalBuffer((__gm__ float *)gamma_2_gm, K_);
        }
        hInPrimeGm_.SetGlobalBuffer((__gm__ T *)h_in_prime_gm, (uint64_t)totalTokens_ * D_);
        x2OutGm_.SetGlobalBuffer((__gm__ T *)x_2_out_gm,      (uint64_t)totalTokens_ * N_ * D_);
        if (h_in_f32_gm != nullptr) {
            hInF32Gm_.SetGlobalBuffer((__gm__ float *)h_in_f32_gm, (uint64_t)totalTokens_ * D_);
        }
        if constexpr (DUAL_SIBLING) {
            partial1Gm_.SetGlobalBuffer((__gm__ float *)workspace, (uint64_t)totalTokens_ * PARTIAL_SLOT_ELEMS);
        }

        // UB空间分配
        pipe_.InitBuffer(bufHOutBf16_,       1, dimTile_ * sizeof(T));
        pipe_.InitBuffer(bufHOutFp32_,          dimTile_ * sizeof(float));
        pipe_.InitBuffer(bufGamma0Fp32_,        dimTile_ * sizeof(float));
        pipe_.InitBuffer(bufGamma1Fp32_,        dimTile_ * sizeof(float));
        pipe_.InitBuffer(bufX2Fp32_,            N_ * dimTile_ * sizeof(float));
        // dimTile=2560时其余固定缓存共194912B；D=7168的完整低精度x2为57344B，
        // 合计252256B，小于A5的256KiB UB上限。
        pipe_.InitBuffer(bufX2Cache_,            K_ * sizeof(T));
        pipe_.InitBuffer(bufResidualBf16_,   1, N_ * dimTile_ * sizeof(T));
        // 前四行先保存residual，之后复用为R4/R5的五行归约输入；第五行兼作通用输出暂存。
        pipe_.InitBuffer(bufResFp32_,     (N_ + 1) * dimTile_ * sizeof(float));
        pipe_.InitBuffer(bufSqFp32_,            dimTile_ * sizeof(float));
        pipe_.InitBuffer(bufPostComb_,          POST_COMB_SIZE * sizeof(float));
        pipe_.InitBuffer(bufPhiFp32_,  BUFFER_NUM_TWO, dimTile_ * sizeof(float));
        pipe_.InitBuffer(bufHInPrimeBf16_,   1, dimTile_ * sizeof(T));
        pipe_.InitBuffer(bufScalar_,            SCALAR_BUF_SIZE);
        pipe_.InitBuffer(bufGamma2Fp32_,        dimTile_ * sizeof(float));

        evtMte2toV_ = static_cast<int32_t>(pipe_.AllocEventID<HardEvent::MTE2_V>());
        evtVtoMte2_ = static_cast<int32_t>(pipe_.AllocEventID<HardEvent::V_MTE2>());
        evtMte3toV_ = static_cast<int32_t>(pipe_.AllocEventID<HardEvent::MTE3_V>());
        evtVtoMte3_ = static_cast<int32_t>(pipe_.AllocEventID<HardEvent::V_MTE3>());
        evtMte3toMte2_ = static_cast<int32_t>(pipe_.AllocEventID<HardEvent::MTE3_MTE2>());
        if constexpr (DUAL_SIBLING) {
            evtMte2toS_ = static_cast<int32_t>(pipe_.AllocEventID<HardEvent::MTE2_S>());
            evtStoMte3_ = static_cast<int32_t>(pipe_.AllocEventID<HardEvent::S_MTE3>());
        }
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::LoadGammaInto(LocalTensor<float> dst, GlobalTensor<float> &src, uint32_t tOff, uint32_t tD)
{
    DataCopyExtParams gP = {1, 0, 0, 0, 0};
    gP.blockLen = tD * sizeof(float);
    DataCopyPadExtParams<float> gPad = {false, 0, 0, 0};
    DataCopyPad(dst, src[tOff], gP, gPad);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32,
                                                        DUAL_SIBLING>::LoadHInCacheTile(uint32_t tOff, uint32_t tD,
                                                                                        LocalTensor<float> hIn)
{
    LocalTensor<T> hInCache = bufX2Cache_.Get<T>()[tOff];
    MhcSandwichRegBase::CastB16ToFp32(hIn, hInCache, tD);
    PipeBarrier<PIPE_V>();
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::LoadHOutAndCast(uint64_t gmOffset, uint32_t tD)
{
    LocalTensor<T> hBf16 = bufHOutBf16_.AllocTensor<T>();
    DataCopy(hBf16, hOutGm_[gmOffset], tD);
    bufHOutBf16_.EnQue(hBf16);
    hBf16 = bufHOutBf16_.DeQue<T>();
    MhcSandwichRegBase::CastB16ToFp32(bufHOutFp32_.Get<float>(), hBf16, tD);
    PipeBarrier<PIPE_V>();
    bufHOutBf16_.FreeTensor(hBf16);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::LoadLowPrecisionTile(
    GlobalTensor<T> &src, uint64_t gmOffset, LocalTensor<float> dst, uint32_t count)
{
    LocalTensor<T> tmp = bufHOutBf16_.AllocTensor<T>();
    DataCopy(tmp, src[gmOffset], count);
    bufHOutBf16_.EnQue(tmp);
    tmp = bufHOutBf16_.DeQue<T>();
    MhcSandwichRegBase::CastB16ToFp32(dst, tmp, count);
    PipeBarrier<PIPE_V>();
    bufHOutBf16_.FreeTensor(tmp);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::LoadResidualAndCast(uint64_t gmBaseOffset, uint32_t tD)
{
    // Host已固定校验N为4，使用编译期常量便于展开标量循环。
    constexpr uint32_t N = N_MAX;
    LocalTensor<T> resBf16 = bufResidualBf16_.AllocTensor<T>();
    for (uint32_t n = 0; n < N; ++n)
        DataCopy(resBf16[n * dimTile_], residualGm_[gmBaseOffset + (uint64_t)n * D_], tD);
    bufResidualBf16_.EnQue(resBf16);
    resBf16 = bufResidualBf16_.DeQue<T>();
    LocalTensor<float> resFp32 = bufResFp32_.Get<float>();
    for (uint32_t n = 0; n < N; ++n)
        MhcSandwichRegBase::CastB16ToFp32(resFp32[n * dimTile_], resBf16[n * dimTile_], tD);
    PipeBarrier<PIPE_V>();
    bufResidualBf16_.FreeTensor(resBf16);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::LoadTokenWeights(uint32_t token, float (&postVals)[N_MAX], float (&combVals)[N_MAX * N_MAX])
{
    // Host已固定校验N为4，使用编译期常量便于展开标量循环。
    constexpr uint32_t N = N_MAX;
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
    for (uint32_t i = 0; i < N; ++i)
        postVals[i] = hPostUb.GetValue(i);
    for (uint32_t i = 0; i < N * N; ++i)
        combVals[i] = hResUb.GetValue(i);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::LoadX2CacheTile(
    uint32_t tOff, uint32_t tD)
{
    LocalTensor<float> x2 = bufX2Fp32_.Get<float>();
    LocalTensor<T> x2Cache = bufX2Cache_.Get<T>();
    for (uint32_t n = 0; n < N_MAX; ++n) {
        MhcSandwichRegBase::CastB16ToFp32(x2[n * dimTile_], x2Cache[n * D_ + tOff], tD);
    }
    PipeBarrier<PIPE_V>();
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::Process()
{
        if (tokenStart_ >= tokenEnd_) {
            return;
        }
        // 缓存标量
        alpha0_ = alpha0Gm_.GetValue(0);
        for (uint32_t i = 0; i < N_MAX; ++i) {
            biasCache_[i] = biasGm_.GetValue(i);
        }

        // 仅单分块路径提前加载gamma
        if (dimLoop_ == 1) {
            DataCopyExtParams gP = {1, 0, 0, 0, 0};
            gP.blockLen = dimTile_ * sizeof(float);
            DataCopyPadExtParams<float> gPad = {false, 0, 0, 0};
            DataCopyPad(bufGamma0Fp32_.Get<float>(), gamma0Gm_[0], gP, gPad);
            DataCopyPad(bufGamma1Fp32_.Get<float>(), gamma1Gm_[0], gP, gPad);
            // gamma_2的形状为[N*D]，在RMSNorm_mid中按head加载
        }

        SetFlag<HardEvent::MTE2_V>(evtMte2toV_);
        WaitFlag<HardEvent::MTE2_V>(evtMte2toV_);

        for (uint32_t t = tokenStart_; t < tokenEnd_; ++t) {
            if constexpr (DUAL_SIBLING) {
                ProcessTokenDualSiblingAiv(t);
            } else {
                ProcessTokenSingleAiv(t);
            }
        }

        pipe_.ReleaseEventID<HardEvent::MTE2_V>(evtMte2toV_);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(evtVtoMte2_);
        pipe_.ReleaseEventID<HardEvent::MTE3_V>(evtMte3toV_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(evtVtoMte3_);
        pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(evtMte3toMte2_);
        if constexpr (DUAL_SIBLING) {
            pipe_.ReleaseEventID<HardEvent::MTE2_S>(evtMte2toS_);
            pipe_.ReleaseEventID<HardEvent::S_MTE3>(evtStoMte3_);
        }
    }

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::QuantizeRoundTrip(LocalTensor<float> buf, uint32_t count)
{
    LocalTensor<T> tmp = bufHInPrimeBf16_.AllocTensor<T>();
    CastToIO(tmp, buf, count);
    PipeBarrier<PIPE_V>();
    MhcSandwichRegBase::CastB16ToFp32(buf, tmp, count);
    PipeBarrier<PIPE_V>();
    bufHInPrimeBf16_.FreeTensor(tmp);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline float KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ReduceSumPrecise(LocalTensor<float> src, uint32_t tD)
{
    LocalTensor<float> scBuf = bufScalar_.Get<float>();
    constexpr uint32_t REDUCE_ALIGN = 16; // ReduceSum要求16个元素对齐
    uint32_t tDAligned = (tD + REDUCE_ALIGN - 1) / REDUCE_ALIGN * REDUCE_ALIGN;
    uint32_t srcShape[2] = {1, tDAligned};
    // 输入在归约后不再使用，允许归约过程复用输入缓冲，减少内部UB与标量调度开销。
    ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(scBuf, src, srcShape, true);
    PipeBarrier<PIPE_V>();
    return scBuf.GetValue(0);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline float KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::ReduceSumSq(LocalTensor<float> src, uint32_t tD)
{
    LocalTensor<float> sqBuf = bufSqFp32_.Get<float>();
    MhcSandwichRegBase::Mul(sqBuf, src, src, tD);
    PipeBarrier<PIPE_V>();
    return ReduceSumPrecise(sqBuf, tD);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::WriteHInPrimeTile(LocalTensor<float> src, uint64_t gmOffset, uint32_t count)
{
    WriteLowPrecisionTile(hInPrimeGm_, gmOffset, src, count);
    if constexpr (RETURN_H_IN_F32) {
        SetFlag<HardEvent::V_MTE3>(evtVtoMte3_);
        WaitFlag<HardEvent::V_MTE3>(evtVtoMte3_);
        DataCopy(hInF32Gm_[gmOffset], src, count);
        FenceMte3ToV();
    }
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::WriteLowPrecisionTile(
    GlobalTensor<T> &dst, uint64_t gmOffset, LocalTensor<float> src, uint32_t count)
{
    LocalTensor<T> tmp = bufHInPrimeBf16_.AllocTensor<T>();
    CastToIO(tmp, src, count);
    bufHInPrimeBf16_.EnQue(tmp);
    tmp = bufHInPrimeBf16_.DeQue<T>();
    DataCopy(dst[gmOffset], tmp, count);
    bufHInPrimeBf16_.FreeTensor(tmp);
}

template <typename T, bool USE_RMS_MID, bool RETURN_H_IN_F32, bool DUAL_SIBLING>
__aicore__ inline void KernelMhcNormPostPreonly<T, USE_RMS_MID, RETURN_H_IN_F32, DUAL_SIBLING>::WriteX2OutTile(LocalTensor<float> src, uint64_t gmOffset, uint32_t count)
{
    WriteLowPrecisionTile(x2OutGm_, gmOffset, src, count);
}

#endif // MHC_NORM_POST_PREONLY_KERNEL_IMPL_H_
