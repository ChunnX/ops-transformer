/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file mhc_pre_sinkhorn_backward_one_stage.h
 * \brief mhc_pre_sinkhorn_backward
 */

#ifndef MHC_PRE_SINKHORN_BACKWARD_ONE_STAGE_H
#define MHC_PRE_SINKHORN_BACKWARD_ONE_STAGE_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "mhc_pre_sinkhorn_backward_simd_vf.h"
#include "mhc_pre_sinkhorn_backward_simt_vf.h"
#include "mhc_pre_sinkhorn_backward_data_arch35.h"

using namespace AscendC;
using namespace MicroAPI;

namespace {
constexpr int32_t DOUBLE_BUF = 2;
constexpr int32_t BYTE_SIZE_PER_BLOCK = 32;
constexpr int32_t ELEMENTS_SIZE_PER_BLOCK = BYTE_SIZE_PER_BLOCK / sizeof(float);
constexpr int32_t FP32_PER_REPEAT = 64;
// Deterministic per-core partial layout (floats): [0..2] grad_alpha(3), [8..31] grad_bias(24).
// PAD=32 keeps each slot 128B aligned. grad_bias starts at index 8 (32B aligned) so the fold's
// DataCopyPad UB source (acc[8]) is aligned, avoiding the "VEC/MTE UB not aligned" aicore fault.
// Must match GetWorkspaceSize (aivNum * 32 floats).
constexpr int32_t DET_PARTIAL_PAD = 32;
constexpr int32_t DET_ALPHA_LEN = 3;        // grad_alpha element count
constexpr int32_t DET_BIAS_OFFSET = 8;      // grad_bias starts at slot index 8 (32B aligned)

constexpr MatmulConfig MHC_PRE_GRAD_MM1_CFG = GetMDLConfig(false, false, 0, true, false, false, true);
constexpr MatmulConfig MHC_PRE_GRAD_MM2_CFG = GetMDLConfig(false, false, 0, true, false, false, true);

// grad_x fp32-accumulate: AIC->AIV completion sync. After the cube finishes ALL mm1 atomic-adds of
// grad_x_from_matmul into the fp32 grad_x workspace, it raises this flag; each AIV then reads the
// fp32 grad_x, casts it once to half, and writes the final gradX output.
// Uses the proven 1C2V MODE4 sub-block sync (mirrors mhc_pre_sinkhorn / the deprecated kernel's
// AIC->AIV handshake): in 1C2V the cube must signal BOTH companion sub-blocks, so it sets both
// GRADX_C2V_FLAG (sub-block 0) and GRADX_C2V_FLAG + FLAG_ID_MAX (sub-block 1); each AIV waits on
// GRADX_C2V_FLAG. Flag id 10 avoids the V->C per-tile ids (which cycle 0..9).
constexpr uint8_t MPSB_SYNC_MODE4 = 4;
constexpr uint16_t MPSB_FLAG_ID_MAX = 16;
constexpr uint16_t MPSB_SYNC_GRADX_C2V_FLAG = 10;
}

template<typename T, typename U>
class MhcPreSinkhornBackwardOneStage {
public:
    
    using A0Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using A1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    // grad_x fp32-accumulate: mm1 output is now fp32 (T), written to gradXFp32Global_ and
    // atomic-added onto the vector-written fp32 grad_x_from_hin/rms. Final combine casts once to half.
    using C1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using C2Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;

    matmul::MatmulImpl<A0Type, BType, C1Type, C1Type, MHC_PRE_GRAD_MM1_CFG> mm1_;
    matmul::MatmulImpl<A1Type, BType, C2Type, C2Type, MHC_PRE_GRAD_MM2_CFG> mm2_;

    // Deterministic mode flag, set by apt.cpp before Process(). When true the cross-core
    // grad_alpha/grad_bias reduction uses a fixed-order per-core partial sum instead of
    // floating-point atomic-add, making grad_alpha/grad_bias bit-reproducible.
    bool deterministic_ = false;

    __aicore__ inline MhcPreSinkhornBackwardOneStage() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR phi, GM_ADDR hPre, GM_ADDR gradHin, GM_ADDR gradHPost,
        GM_ADDR gradHRes, GM_ADDR alpha, GM_ADDR bias, GM_ADDR hcBeforeNorm, GM_ADDR invRms, GM_ADDR sumOut,
        GM_ADDR normOut, GM_ADDR gradX, GM_ADDR gradPhi, GM_ADDR gradAlpha, GM_ADDR gradBias,
        GM_ADDR workspace, const MhcPreSinkhornBackwardArch35TilingData* tilingData, TPipe* pipe)
    {
        pipe_ = pipe;
        blkIdx_ = GetBlockIdx();

        InitTiling(tilingData);
        InitGM(x, phi, hPre, gradHin, gradHPost, gradHRes, alpha,
            bias, hcBeforeNorm, invRms, sumOut, normOut, gradX, gradPhi,
            gradAlpha, gradBias, workspace);
        InitBuffer();
    }
    __aicore__ inline void Process();

protected:
    int8_t ping1_ = 0;
    int64_t blkIdx_, aivNum_, aicNum_;
    TPipe* pipe_;
    int64_t batchSize_, seqLength_, totalTasks_, totalTasksAligned_;
    int64_t c_, n_, c0_, c1_;
    int64_t tile_, tileAligned_;
    int64_t skIterCount_;
    int64_t mm1K_, mm1M_, mm1N_;
    int64_t mm2K_, mm2M_, mm2N_;
    T alphaPre_, alphaPost_, alphaRes_;

    TBuf<TPosition::VECCALC> gradHinBuf_, hPreBuf_, hResBuf_, xBuf_;
    TBuf<TPosition::VECCALC> xFp32Buf_, gradXBuf_;
    // grad_x fp32-accumulate: fp32 scratch for grad_x_from_hin (and reused for the rms term) so the
    // hin+rms contribution is computed entirely in fp32 (no intermediate half rounding).
    TBuf<TPosition::VECCALC> gradHinFp32Buf_;
    TBuf<TPosition::VECCALC> fusedGradHPreAndGradHPostBuf_, gradHResBuf_;
    TBuf<TPosition::VECCALC> alphaBuf_, biasBuf_, invRmsBuf_, gradInvRmsBuf_;
    TBuf<TPosition::VECCALC> fusedHPre2AndHPost2Buf_;
    TBuf<TPosition::VECCALC> sigmoidGradBuf1_, sigmoidGradBuf2_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> gradRMSNormBuf_;
    TBuf<TPosition::VECOUT> gradH2Buf_;
    TBuf<TPosition::VECOUT> gradAlphaBuf_, gradBiasBuf_;
    // Deterministic fold scratch (core0 only): one-core staging + accumulator, PAD floats each.
    TBuf<TPosition::VECCALC> detStageBuf_, detAccBuf_;

    LocalTensor<U> xLocal_;
    LocalTensor<T> xFp32Local_;
    LocalTensor<U> gradHinLocal_;
    // grad_x fp32-accumulate: grad_x_from_hin/rms staging is now fp32 (T).
    LocalTensor<T> gradXLocal_;
    LocalTensor<T> gradHinFp32Local_;
    LocalTensor<T> hPreLocal_;
    LocalTensor<T> fusedGradHPreAndGradHPostLocal_, gradHResLocal_;
    LocalTensor<T> biasLocal_, alphaLocal_;
    LocalTensor<T> gradBiasLocal_, gradAlphaLocal_;
    
    LocalTensor<T> invRmsLocal_, gradInvRmsLocal_;
    LocalTensor<T> gradH2Local_;
    LocalTensor<T> fusedHPre2AndHPost2Local_, hRes2Local_;
    LocalTensor<T> tmpLocal_;
    LocalTensor<T> gradSigmoidLocal1_;
    LocalTensor<T> gradSigmoidLocal2_;
    LocalTensor<T> gradRMSNormLocal_;

    GlobalTensor<U> xGlobal_;
    GlobalTensor<T> xFp32Global_;
    GlobalTensor<U> gradHinGlobal_;
    GlobalTensor<T> hPreGlobal_;
    GlobalTensor<U> gradXGlobal_;
    GlobalTensor<T> gradHPostGlobal_;
    GlobalTensor<T> alphaGlobal_, biasGlobal_;
    GlobalTensor<T> invRmsGlobal_;
    GlobalTensor<T> gradAlphaGlobal_, gradBiasGlobal_;
    GlobalTensor<T> h2Global_;
    GlobalTensor<T> skNormGlobal_, skSumGlobal_;
    GlobalTensor<T> gradHResGlobal_;
    GlobalTensor<T> gradH2Global_;
    GlobalTensor<T> gradPhiGlobal_;
    GlobalTensor<T> phiGlobal_;
    // Deterministic per-core partial region in user workspace (aivNum_ * DET_PARTIAL_PAD floats).
    GlobalTensor<T> detPartialWs_;
    // grad_x fp32-accumulate regions in user workspace (each B*S*N*C floats):
    //  - gradXFp32Global_ : mm1 grad_x_from_matmul (cube single-writer; AIV reads cross-core)
    //  - gradXHinRmsWs_   : vector grad_x_from_hin + grad_x_from_rms (AIV writes & reads, intra-core)
    // The final combine adds the two fp32 sources and casts ONCE to the half gradX output (golden).
    GlobalTensor<T> gradXFp32Global_;
    GlobalTensor<T> gradXHinRmsWs_;

private:
    __aicore__ inline void InitTiling(const MhcPreSinkhornBackwardArch35TilingData* tilingData)
    {
        batchSize_ = tilingData->batchSize;
        seqLength_ = tilingData->seqLength;
        aivNum_ = tilingData->aivNum;
        aicNum_ = tilingData->aivNum / 2;
        c_ = tilingData->c;
        n_ = tilingData->n;
        c0_ = tilingData->c0;
        c1_ = tilingData->c1;
        skIterCount_ = tilingData->skIterCount;
        tile_ = tilingData->tileSize;
        tileAligned_ = AlignUp(tile_, BYTE_SIZE_PER_BLOCK / sizeof(T));
        totalTasks_ = batchSize_ * seqLength_;
        totalTasksAligned_ = AlignUp(totalTasks_, aivNum_ * tile_);
        
        if ASCEND_IS_AIC {
            mm1K_ = n_ * n_ + 2 * n_;
            mm1M_ = tile_ * 2;
            mm1N_ = n_ * c_;

            mm2K_ = batchSize_ * seqLength_;
            mm2M_ = n_ * n_ + 2 * n_;
            mm2N_ = n_ * c_;
        }
    }

    __aicore__ inline void InitGM(GM_ADDR x, GM_ADDR phi, GM_ADDR hPre, GM_ADDR gradHin, GM_ADDR gradHPost,
        GM_ADDR gradHRes, GM_ADDR alpha, GM_ADDR bias, GM_ADDR hcBeforeNorm, GM_ADDR invRms,
        GM_ADDR sumOut, GM_ADDR normOut, GM_ADDR gradX, GM_ADDR gradPhi,
        GM_ADDR gradAlpha, GM_ADDR gradBias, GM_ADDR workspace)
    {
        // cube 和 vector 共用
        xGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(x));               // fp32
        xFp32Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace));               // fp32
        gradPhiGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradPhi));  // fp32
        phiGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(phi));  // fp32
        gradXGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(gradX));   // fp32
        gradH2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace) + batchSize_ * seqLength_ * n_ * c_);
        // Deterministic per-core partial region: right after xFp32 (B*S*N*C) and gradH2 (B*S*(N^2+2N)).
        int64_t detPartialOffset =
            batchSize_ * seqLength_ * n_ * c_ + batchSize_ * seqLength_ * (n_ * n_ + 2 * n_);
        detPartialWs_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace) + detPartialOffset);
        // grad_x fp32-accumulate regions: two B*S*N*C-float blocks right after the deterministic
        // per-core partial region. Must match GetWorkspaceSize (gradHat2 + aivNum_*DET_PARTIAL_PAD +
        // 2 * B*S*N*C floats).
        int64_t gradXFp32Offset = detPartialOffset + aivNum_ * DET_PARTIAL_PAD;
        int64_t bsnc = batchSize_ * seqLength_ * n_ * c_;
        gradXFp32Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace) + gradXFp32Offset);
        gradXHinRmsWs_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace) + gradXFp32Offset + bsnc);

        if ASCEND_IS_AIV {
            hPreGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(hPre));        // fp32
            gradHinGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(gradHin));   // fp32
            
            gradHPostGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradHPost));   // fp32

            alphaGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(alpha));   // fp32
            biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias));   // fp32

            invRmsGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(invRms));   // fp32
            gradAlphaGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradAlpha));
            gradBiasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradBias));

            h2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(hcBeforeNorm));   // fp32

            skNormGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(normOut));   // fp32
            skSumGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(sumOut));   // fp32

            gradHResGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradHRes));   // fp32
        }
    }

    __aicore__ inline void InitBuffer()
    {
        if ASCEND_IS_AIV {
            pipe_->InitBuffer(xBuf_, n_ * c_ * sizeof(U));
            pipe_->InitBuffer(xFp32Buf_, DOUBLE_BUF * c_ * sizeof(T));
            pipe_->InitBuffer(gradHinBuf_, DOUBLE_BUF * c_ * sizeof(U));
            // grad_x fp32-accumulate: the per-tile grad_x_from_hin/rms is now kept in fp32 (T),
            // so this double-buffered grad_x staging buffer must hold fp32, not half.
            pipe_->InitBuffer(gradXBuf_, DOUBLE_BUF * c_ * sizeof(T));
            pipe_->InitBuffer(gradHinFp32Buf_, DOUBLE_BUF * c_ * sizeof(T));
            pipe_->InitBuffer(hPreBuf_, DOUBLE_BUF * 2 * n_ * sizeof(T));
            pipe_->InitBuffer(hResBuf_, tile_ * n_ * n_ * sizeof(T));
            pipe_->InitBuffer(invRmsBuf_, tile_ * sizeof(T));
            pipe_->InitBuffer(gradH2Buf_, tile_ * (n_ * n_ + 2 * n_) * sizeof(T));

            int32_t sigmoidGradBufSize = AlignUp(tile_ * 2 * n_ * sizeof(T), VEC_LENGTH);

            pipe_->InitBuffer(sigmoidGradBuf1_, sigmoidGradBufSize);
            pipe_->InitBuffer(sigmoidGradBuf2_, sigmoidGradBufSize);

            pipe_->InitBuffer(biasBuf_, (n_ * n_ + 2 * n_) * sizeof(T));
            pipe_->InitBuffer(alphaBuf_, (n_ * n_ + 2 * n_) * sizeof(T));

            pipe_->InitBuffer(tmpBuf_, tile_ * (n_ * n_ + 2 * n_) * sizeof(T));
            pipe_->InitBuffer(gradInvRmsBuf_, tile_ * sizeof(T));
            pipe_->InitBuffer(gradHResBuf_, tile_ * n_ * n_ * sizeof(T));
            pipe_->InitBuffer(fusedGradHPreAndGradHPostBuf_, tile_ * 2 * n_ * sizeof(T));
            pipe_->InitBuffer(fusedHPre2AndHPost2Buf_, tile_ * 2 * n_ * sizeof(T));
            pipe_->InitBuffer(gradAlphaBuf_, 8 * sizeof(T));
            pipe_->InitBuffer(gradBiasBuf_, 2 * n_ * sizeof(T));
            pipe_->InitBuffer(detStageBuf_, DET_PARTIAL_PAD * sizeof(T));
            pipe_->InitBuffer(detAccBuf_, DET_PARTIAL_PAD * sizeof(T));
            pipe_->InitBuffer(gradRMSNormBuf_, tile_* sizeof(T));

            xLocal_ = xBuf_.Get<U>();
            xFp32Local_ = xFp32Buf_.Get<T>();
            gradHinLocal_ = gradHinBuf_.Get<U>();

            fusedGradHPreAndGradHPostLocal_ = fusedGradHPreAndGradHPostBuf_.Get<T>();
            fusedHPre2AndHPost2Local_ = fusedHPre2AndHPost2Buf_.Get<T>();

            biasLocal_ = biasBuf_.Get<T>();
            alphaLocal_ = alphaBuf_.Get<T>();

            gradH2Local_ = gradH2Buf_.Get<T>();
            tmpLocal_ = tmpBuf_.Get<T>();
            
            gradHResLocal_ = gradHResBuf_.Get<T>();
            
            invRmsLocal_ = invRmsBuf_.Get<T>();
            gradInvRmsLocal_ = gradInvRmsBuf_.Get<T>();

            gradSigmoidLocal1_ = sigmoidGradBuf1_.Get<T>();
            gradSigmoidLocal2_ = sigmoidGradBuf2_.Get<T>();

            gradBiasLocal_ = gradBiasBuf_.Get<T>();
            gradAlphaLocal_ = gradAlphaBuf_.Get<T>();

            gradRMSNormLocal_ = gradRMSNormBuf_.Get<T>();
            gradXLocal_ = gradXBuf_.Get<T>();
            gradHinFp32Local_ = gradHinFp32Buf_.Get<T>();

            hPreLocal_ = hPreBuf_.Get<T>();

            hRes2Local_ = hResBuf_.Get<T>();
        }
    }

    __aicore__ inline void GetAlphaAndBias();
    __aicore__ inline void ComputeGradResHHat2(const int32_t taskOffset, const int32_t tileTaskCount);
    __aicore__ inline void ComputeGradSigmoid(const int32_t taskOffset, const int32_t tileTaskCount);
    __aicore__ inline T ComputeGradInvRms(const int32_t i);
    __aicore__ inline void ComputeGradRMSNorm(const int32_t tileTaskCount);
    __aicore__ inline void ComputeGradXVector(
        const T gradRsqrtVal, const T gradRMSNormVal, const int32_t taskOffset, bool copyNext);
    __aicore__ inline void ProcessMatmul1(const int32_t taskOffset, const int32_t mm1M);
    __aicore__ inline void ProcessMatmul2(const int32_t taskOffset, const int32_t mm2K);
    __aicore__ inline void CombineGradX();
};

template<typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardOneStage<T, U>::GetAlphaAndBias()
{
    // HcBase 常驻 UB
    alphaPre_ = alphaGlobal_.GetValue(0);
    alphaPost_ = alphaGlobal_.GetValue(1);
    alphaRes_ = alphaGlobal_.GetValue(2);
    DataCopyPad(biasLocal_, biasGlobal_, {static_cast<uint16_t>(1),
        static_cast<uint32_t>((n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    
    for (int32_t i = 0; i < n_; i++) {
        alphaLocal_.SetValue(i, alphaPre_);
        alphaLocal_.SetValue(i + n_, alphaPost_);
    }
}

template<typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardOneStage<T, U>::ComputeGradResHHat2(
    const int32_t taskOffset, const int32_t tileTaskCount)
{
    // Compute sigmoidGrad
    DataCopyPad(hRes2Local_, h2Global_[taskOffset * (n_ * n_ + 2 * n_) + 2 * n_],
        {static_cast<uint16_t>(tileTaskCount), static_cast<uint32_t>(n_ * n_ * sizeof(T)),
        static_cast<int64_t>(2 * n_ * sizeof(T)), 0, 0}, {false, 0, 0, 0});
    DataCopyPad(gradHResLocal_, gradHResGlobal_[taskOffset * n_ * n_], {static_cast<uint16_t>(tileTaskCount),
        static_cast<uint32_t>(n_ * n_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    DataCopyPad(invRmsLocal_, invRmsGlobal_[taskOffset], {static_cast<uint16_t>(1),
        static_cast<uint32_t>(tileTaskCount * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});

    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);

    uint32_t thread_num = min(static_cast<uint32_t>(THREAD_NUM), static_cast<uint32_t>(tileTaskCount * n_ * n_));
    uint32_t xThreadCount = n_ * n_;
    uint32_t yThreadCount = thread_num / xThreadCount;

    Duplicate(gradInvRmsLocal_, 0.f, tileTaskCount);
    // In deterministic mode the SIMT kernel must NOT atomic-add its grad_alpha[2] (alphaRes) and
    // grad_bias[2N..] (the N*N part) cross-core into the shared output. Redirect those two GM
    // pointers into this core's private partial slot so the SIMT atomic-adds only accumulate this
    // core's tiles (sequential -> deterministic); core0 folds all core partials in fixed order.
    // Partial slot layout: [0..2]=alpha, [3..26]=bias, so alpha[2]->slot+2, bias[2N]->slot+3+2N.
    __gm__ T* simtGradAlpha = deterministic_
        ? (__gm__ T*) detPartialWs_[blkIdx_ * DET_PARTIAL_PAD + 2].GetPhyAddr()
        : (__gm__ T*) gradAlphaGlobal_[2].GetPhyAddr();
    __gm__ T* simtGradBias = deterministic_
        ? (__gm__ T*) detPartialWs_[blkIdx_ * DET_PARTIAL_PAD + DET_BIAS_OFFSET + n_ * 2].GetPhyAddr()
        : (__gm__ T*) gradBiasGlobal_[n_ * 2].GetPhyAddr();
    Simt::VF_CALL<SinkhornKnoppSimt<T>>(Simt::Dim3{xThreadCount, yThreadCount},
        (__ubuf__ T*) gradH2Local_.GetPhyAddr(),
        (__ubuf__ T*) gradInvRmsLocal_.GetPhyAddr(),
        simtGradAlpha,
        simtGradBias,
        (__gm__ T*) skSumGlobal_.GetPhyAddr(),
        (__gm__ T*) skNormGlobal_.GetPhyAddr(),
        (__ubuf__ T*) gradHResLocal_.GetPhyAddr(),
        (__ubuf__ T*) hRes2Local_.GetPhyAddr(),
        (__ubuf__ T*) invRmsLocal_.GetPhyAddr(),
        (__ubuf__ T*) biasLocal_[n_ * 2].GetPhyAddr(),
        (__ubuf__ T*) tmpLocal_.GetPhyAddr(),
        alphaRes_, totalTasks_, taskOffset, tileTaskCount, skIterCount_, n_);
}

template<typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardOneStage<T, U>::ComputeGradSigmoid(
    const int32_t taskOffset, const int32_t tileTaskCount)
{
    DataCopyPad(fusedHPre2AndHPost2Local_, h2Global_[taskOffset * (n_ * n_ + 2 * n_)],
        {static_cast<uint16_t>(tileTaskCount), static_cast<uint32_t>(2 * n_ * sizeof(T)),
        static_cast<int64_t>(n_ * n_ * sizeof(T)), 0, 0}, {false, 0, 0, 0});

    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    
    uint16_t repeatTimes = Ceil(tileTaskCount * n_ * 2, FP32_PER_VL);
    ComputeGradSigmoidVf<T>(gradSigmoidLocal1_, gradSigmoidLocal2_, fusedHPre2AndHPost2Local_, invRmsLocal_,
        alphaLocal_, biasLocal_, repeatTimes, tileTaskCount * n_ * 2);
}

template<typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardOneStage<T, U>::ComputeGradRMSNorm(
    const int32_t tileTaskCount)
{
    Mul(gradRMSNormLocal_, invRmsLocal_, invRmsLocal_, tileTaskCount);
    Mul(gradRMSNormLocal_, gradRMSNormLocal_, invRmsLocal_, tileTaskCount);
    Muls(gradRMSNormLocal_, gradRMSNormLocal_, -1.f / static_cast<T>(n_ * c_), tileTaskCount);
}

template<typename T, typename U>
__aicore__ inline T MhcPreSinkhornBackwardOneStage<T, U>::ComputeGradInvRms(const int32_t i)
{
    Mul(gradSigmoidLocal1_[i * 2 * n_], fusedGradHPreAndGradHPostLocal_[i * 2 * n_],
        gradSigmoidLocal1_[i * 2 * n_], 2 * n_);
    Mul(gradSigmoidLocal2_[i * 2 * n_], fusedGradHPreAndGradHPostLocal_[i * 2 * n_],
        gradSigmoidLocal2_[i * 2 * n_], 2 * n_);
    Mul(tmpLocal_[i * 2 * n_], gradSigmoidLocal1_[i * 2 * n_], alphaLocal_, 2 * n_);
    Muls(gradH2Local_[i * (2 * n_ + n_ * n_)], tmpLocal_[i * 2 * n_], invRmsLocal_.GetValue(i), 2 * n_);
    Mul(tmpLocal_[i * 2 * n_], tmpLocal_[i * 2 * n_], fusedHPre2AndHPost2Local_[i * 2 * n_], 2 * n_);
    ReduceSum<T>(tmpLocal_[i * 2 * n_], tmpLocal_[i * 2 * n_], tmpLocal_[i * 2 * n_], 2 * n_);
    T gradRsqrtVal = tmpLocal_.GetValue(i * 2 * n_) + gradInvRmsLocal_.GetValue(i);
    return gradRsqrtVal;
}

template<typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardOneStage<T, U>::ComputeGradXVector(
    const T gradRsqrtVal, const T gradRMSNormVal, const int32_t taskOffset, bool copyNext)
{
    int8_t ping = 0;

    SetFlag<HardEvent::MTE3_V>(0);
    SetFlag<HardEvent::MTE3_V>(1);
    for (int32_t i = 0; i < n_; i++) {
        WaitFlag<HardEvent::MTE3_V>(ping);

        // grad_x fp32-accumulate: compute grad_x_from_hin + grad_x_from_rms ENTIRELY in fp32 (no
        // intermediate half rounding), so it matches the golden's fp32 accumulation. The math is
        //   grad_x_hinrms = gradHin*hPre + x*(gradRsqrt*gradRMSNorm)
        // with hPre and (gradRsqrt*gradRMSNorm) per-(i) scalars. Cast the half inputs up to fp32,
        // then Muls/Add in fp32. xFp32Local_ keeps the PURE x-cast for mm2's grad_phi; the rms term
        // is built in gradHinFp32Local_ so xFp32Local_ is not disturbed. The cube's fp32
        // grad_x_from_matmul is added in the final combine, which casts the sum ONCE to half.
        const T gradXRmsCoeff = gradRsqrtVal * gradRMSNormVal;
        const T hPreScalar = hPreLocal_.GetValue(ping1_ * 8 + i);
        Cast(xFp32Local_[ping * c_], xLocal_[i * c_], RoundMode::CAST_NONE, c_);
        Cast(gradHinFp32Local_[ping * c_], gradHinLocal_[ping1_ * c_], RoundMode::CAST_NONE, c_);
        Muls(gradXLocal_[ping * c_], gradHinFp32Local_[ping * c_], hPreScalar, c_);            // hin term
        Muls(gradHinFp32Local_[ping * c_], xFp32Local_[ping * c_], gradXRmsCoeff, c_);         // rms term
        Add(gradXLocal_[ping * c_], gradXLocal_[ping * c_], gradHinFp32Local_[ping * c_], c_);  // hin+rms (fp32)

        SetFlag<HardEvent::V_MTE3>(0);
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE3>(0);

        DataCopyPad(gradXHinRmsWs_[taskOffset * n_ * c_ + i * c_], gradXLocal_[ping * c_],
            {static_cast<uint16_t>(1), static_cast<uint32_t>(c_ * sizeof(T)), 0, 0, 0});
        DataCopyPad(xFp32Global_[taskOffset * n_ * c_ + i * c_], xFp32Local_[ping * c_], {static_cast<uint16_t>(1),
            static_cast<uint32_t>(c_ * sizeof(T)), 0, 0, 0});

        SetFlag<HardEvent::MTE3_V>(ping);
        WaitFlag<HardEvent::V_MTE2>(0);

        SetFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(0);

        if (copyNext) {
            DataCopyPad(xLocal_[i * c_], xGlobal_[(taskOffset + 1) * n_ * c_ + i * c_], {static_cast<uint16_t>(1),
                        static_cast<uint32_t>(c_ * sizeof(U)), 0, 0, 0}, {false, 0, 0, 0});
        }
        
        ping = 1 - ping;
    }
    WaitFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(1);
}

/**
    约束：
    c: c >= 64 && c % 64 == 0
    n: n == 4
*/
template<typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardOneStage<T, U>::Process()
{
    if ASCEND_IS_AIV {
        if (blkIdx_ == 0) {
            InitOutput<T>(gradPhiGlobal_, (n_ * n_ + 2 * n_) * n_ * c_, 0);
            InitOutput<T>(gradAlphaGlobal_, 3, 0);
            InitOutput<T>(gradBiasGlobal_, n_ * n_ + 2 * n_, 0);
            if (deterministic_) {
                // core0 zeroes the ENTIRE deterministic per-core partial region, mirroring the
                // proven InitOutput(gradAlphaGlobal_)+SyncAll pattern above. Each core then atomic-
                // adds only into its OWN slot (no cross-core contention -> per-core deterministic),
                // and the SyncAll below guarantees this zero is globally visible first. Per-core
                // self-zeroing of a sub-slice proved racy; whole-region zero by core0 is reliable.
                InitOutput<T>(detPartialWs_, aivNum_ * DET_PARTIAL_PAD, 0);
            }
        }
    }
    SyncAll<false>();

    if ASCEND_IS_AIV {
        int8_t ping = 0;
        GetAlphaAndBias();

        for (int32_t taskOffset = blkIdx_ * tile_; taskOffset < totalTasksAligned_; taskOffset += aivNum_ * tile_) {
            int32_t tileTaskCount = min(tile_, totalTasks_ - taskOffset);

            if (tileTaskCount > 0) {
                ComputeGradResHHat2(taskOffset, tileTaskCount);
                ComputeGradSigmoid(taskOffset, tileTaskCount);
                ComputeGradRMSNorm(tileTaskCount);

                DataCopyPad(fusedGradHPreAndGradHPostLocal_, gradHPostGlobal_[taskOffset * n_],
                    {static_cast<uint16_t>(tileTaskCount),
                    static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0},
                    {true, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0, 0});
                DataCopyPad(xLocal_, xGlobal_[taskOffset * n_ * c_], {static_cast<uint16_t>(n_),
                        static_cast<uint32_t>(c_ * sizeof(U)), 0, 0, 0}, {false, 0, 0, 0});

                SetFlag<HardEvent::V_MTE2>(0);
                SetFlag<HardEvent::V_MTE2>(1);
                for (int32_t i = 0; i < tileTaskCount; i++) {
                    WaitFlag<HardEvent::V_MTE2>(ping1_);
                    DataCopyPad(hPreLocal_[ping1_ * 8], hPreGlobal_[(taskOffset + i) * n_], {static_cast<uint16_t>(1),
                        static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
                    DataCopyPad(gradHinLocal_[ping1_ * c_], gradHinGlobal_[(taskOffset + i) * c_],
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(c_ * sizeof(U)), 0, 0, 0}, {false, 0, 0, 0});
                        
                    SetFlag<HardEvent::MTE2_V>(0);
                    WaitFlag<HardEvent::MTE2_V>(0);

                    ComputeGradPreVf<T>(fusedGradHPreAndGradHPostLocal_[i * 2 * n_], xLocal_,
                        gradHinLocal_[ping1_ * c_], n_, c_ / FP32_PER_VL);
                    T gradRsqrtVal = ComputeGradInvRms(i);
                    ComputeGradXVector(
                        gradRsqrtVal, gradRMSNormLocal_.GetValue(i), taskOffset + i, i != tileTaskCount - 1);
                    SetFlag<HardEvent::V_MTE2>(ping1_);
                    ping1_ = 1 - ping1_;
                }
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);

                DataCopyPad(gradH2Global_[taskOffset * (n_ * n_ + 2 * n_)], gradH2Local_,
                    {static_cast<uint16_t>(tileTaskCount),
                    static_cast<uint32_t>((n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0});
                
                WaitFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::V_MTE2>(1);
            }

            CrossCoreSetFlag<0x2, PIPE_MTE3>(ping);
            ping = (ping + 1) % 10;

            if (tileTaskCount > 0) {
                int32_t dupSize = AlignUp(tile_ * n_ * 2, FP32_PER_REPEAT) - tileTaskCount * 2 * n_;
                Duplicate(gradSigmoidLocal1_[tileTaskCount * 2 * n_], 0.f, dupSize);
                Duplicate(gradSigmoidLocal2_[tileTaskCount * 2 * n_], 0.f, dupSize);
                // compute gradHcScale
                int32_t repeatTimes = Ceil(tileTaskCount * n_ * 2, FP32_PER_REPEAT);
                uint64_t preMask[] = {0X0F0F0F0F0F0F0F0F};
                uint64_t postMask[] = {0XF0F0F0F0F0F0F0F0};
                ReduceSum<T>(gradAlphaLocal_, gradSigmoidLocal2_, tmpLocal_, preMask, repeatTimes, 8);
                ReduceSum<T>(gradAlphaLocal_[1], gradSigmoidLocal2_, tmpLocal_, postMask, repeatTimes, 8);
                uint32_t shape[] = { static_cast<uint32_t>(tileTaskCount), static_cast<uint32_t>(2 * n_) };
                ReduceSum<T, Pattern::Reduce::RA, true>(
                    gradBiasLocal_, gradSigmoidLocal1_, tmpLocal_.template ReinterpretCast<uint8_t>(), shape, true);

                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                SetAtomicAdd<T>();
                if (deterministic_) {
                    // Accumulate this tile's grad_alpha[0..1] / grad_bias[0..2N-1] into THIS core's
                    // private partial slot. Atomic-add here only serializes tiles on the same core
                    // (sequential -> deterministic); there is no cross-core contention.
                    DataCopyPad(detPartialWs_[blkIdx_ * DET_PARTIAL_PAD], gradAlphaLocal_,
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(2 * sizeof(T)), 0, 0, 0});
                    DataCopyPad(detPartialWs_[blkIdx_ * DET_PARTIAL_PAD + DET_BIAS_OFFSET], gradBiasLocal_,
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(2 * n_ * sizeof(T)), 0, 0, 0});
                } else {
                    DataCopyPad(gradAlphaGlobal_, gradAlphaLocal_,
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(2 * sizeof(T)), 0, 0, 0});
                    DataCopyPad(gradBiasGlobal_, gradBiasLocal_,
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(2 * n_ * sizeof(T)), 0, 0, 0});
                }
                SetAtomicNone();
            }
        }
        if (deterministic_) {
            // Writer-side coherency: drain this core's MTE3, then FLUSH this core's partial slot from
            // the local data cache out to GM (dcci CACHELINE_OUT). SyncAll<true> is only a cross-core
            // execution barrier (ffts_cross_core_sync) — it does NOT make GM written by a peer core
            // visible to core0's reads. Without this writer flush, core0's fold DMA can read stale
            // slot data (observed as intermittent grad_bias/grad_alpha % drops). The flush only
            // touches this core's own DET_PARTIAL_PAD-float slot, so it cannot disturb the AIC
            // grad_phi accumulation (different addresses). It is safe here because the slot writes
            // above are complete for this core.
            SetFlag<HardEvent::MTE3_MTE2>(6);
            WaitFlag<HardEvent::MTE3_MTE2>(6);
            // Flush every 64B cache line of this core's DET_PARTIAL_PAD-float (128B) slot.
            // CACHE_LINE_SIZE = 64B = 16 floats; the slot spans 2 lines. SINGLE_CACHE_LINE flushes
            // one line per call, so iterate. CACHELINE_OUT writes the line back to GM (no invalidate
            // of other cores), exactly the writer half of the cross-core GM exchange protocol.
            constexpr int32_t FLOATS_PER_CACHE_LINE = 64 / sizeof(T);
            for (int32_t off = 0; off < DET_PARTIAL_PAD; off += FLOATS_PER_CACHE_LINE) {
                DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                    detPartialWs_[blkIdx_ * DET_PARTIAL_PAD + off]);
            }
        }
    }

    if ASCEND_IS_AIC {
        int8_t ping = 0;
        for (int32_t taskOffset = blkIdx_ * 2 * tile_;
            taskOffset < totalTasksAligned_; taskOffset += aicNum_ * 2 * tile_) {

            int32_t tileTaskCount = min(2 * tile_, totalTasks_ - taskOffset);
            CrossCoreWaitFlag<0x2>(ping);
            ping = (ping + 1) % 10;

            if (tileTaskCount > 0) {
                ProcessMatmul1(taskOffset, tileTaskCount);
                ProcessMatmul2(taskOffset, tileTaskCount);
            }
        }
        // grad_x fp32-accumulate: all mm1 atomic-adds of grad_x_from_matmul into the fp32 grad_x
        // workspace are now done. Signal BOTH companion AIV sub-blocks (1C2V MODE4) so they can run
        // the final fp32->half combine. PIPE_FIX orders the cube's matmul GM writes before the flags.
        CrossCoreSetFlag<MPSB_SYNC_MODE4, PIPE_FIX>(MPSB_SYNC_GRADX_C2V_FLAG);
        CrossCoreSetFlag<MPSB_SYNC_MODE4, PIPE_FIX>(MPSB_SYNC_GRADX_C2V_FLAG + MPSB_FLAG_ID_MAX);
    }

    if (deterministic_) {
      if ASCEND_IS_AIV {
        // AIV-ONLY barrier: the per-core partial slots are written only by AIV cores, so the fold
        // needs all AIV cores synced, NOT the AIC cores. Using SyncAll<true> (isAIVOnly=true) avoids
        // making the fast AIV cores wait on the (potentially very slow, e.g. c=4096) AIC matmul,
        // which with an all-core SyncAll<false> tripped the AIV watchdog -> aicore timeout/trap and
        // NO_OUTPUT at large c. AIC cores must NOT enter this block (they don't call this SyncAll).
        SyncAll<true>();

        {
            if (blkIdx_ == 0) {
                // Reader-side: now that the AIV barrier has passed (all AIV partial writes done), it
                // is safe to invalidate core0's whole data cache so the fold DMA reads fresh GM
                // values for every core's slot.
                DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE>(detPartialWs_);
                // core0 folds all aivNum_ per-core partials in fixed blkIdx order into the final
                // grad_alpha (3) + grad_bias (24). Each core's slot is brought in via DMA
                // (DataCopy GM->UB, coherent after SyncAll; scalar GM reads would hit stale cache)
                // and accumulated into a UB accumulator with vector Add (fixed core order ->
                // bit-reproducible). The accumulator layout is [0..2]=alpha, [3..26]=bias, exactly
                // the per-core slot layout, so one contiguous Add covers both.
                int32_t biasLen = n_ * n_ + 2 * n_;  // 24
                int32_t copyLen = DET_BIAS_OFFSET + biasLen;  // 27 valid floats per slot

                LocalTensor<T> stage = detStageBuf_.Get<T>();
                LocalTensor<T> acc = detAccBuf_.Get<T>();
                Duplicate(acc, 0.f, DET_PARTIAL_PAD);
                SetFlag<HardEvent::V_MTE2>(2);
                WaitFlag<HardEvent::V_MTE2>(2);

                for (int32_t core = 0; core < aivNum_; core++) {
                    DataCopyPad(stage, detPartialWs_[core * DET_PARTIAL_PAD],
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(copyLen * sizeof(T)), 0, 0, 0},
                        {false, 0, 0, 0});
                    SetFlag<HardEvent::MTE2_V>(2);
                    WaitFlag<HardEvent::MTE2_V>(2);
                    Add(acc, acc, stage, copyLen);
                    SetFlag<HardEvent::V_MTE2>(2);
                    WaitFlag<HardEvent::V_MTE2>(2);
                }

                SetFlag<HardEvent::V_MTE3>(2);
                WaitFlag<HardEvent::V_MTE3>(2);
                DataCopyPad(gradAlphaGlobal_, acc,
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(DET_ALPHA_LEN * sizeof(T)), 0, 0, 0});
                DataCopyPad(gradBiasGlobal_, acc[DET_BIAS_OFFSET],
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(biasLen * sizeof(T)), 0, 0, 0});
            }
        }
      }
    }

    // grad_x fp32-accumulate final combine (both TilingKey 0 and 1): each AIV waits for the cube's
    // C2V flag (all mm1 atomic-adds done), then reads the fp32 grad_x workspace and casts ONCE to the
    // half gradX output. This is the only place gradX (half) is written, so there is exactly one
    // fp16 rounding of the (from_hin + from_rms + from_matmul) sum — matching the golden.
    if ASCEND_IS_AIV {
        // Proven 1C2V MODE4 AIC->AIV handshake: the cube set both sub-block flags; each AIV waits on
        // its GRADX_C2V_FLAG before reading the cube's fp32 grad_x_from_matmul. The combine reads two
        // SEPARATE workspaces: gradXHinRmsWs_ (this core's OWN hin+rms write, intra-core coherent) and
        // gradXFp32Global_ (cube single-writer matmul, cross-core coherent after this flag, like the
        // mhc_pre AIV<-cube GM exchange). Keeping them separate avoids the stale-self-cache hazard of
        // atomic-adding the cube result onto a region the AIV had already written.
        CrossCoreWaitFlag<MPSB_SYNC_MODE4, PIPE_MTE3>(MPSB_SYNC_GRADX_C2V_FLAG);
        CombineGradX();
    }
}

// mm1K_ = n_ * n_ + 2 * n_;
// mm1M_ = tile_ * 2;
// mm1N_ = n_ * c_;

// mm2K_ = tile_ * 2;
// mm2M_ = n_ * n_ + 2 * n_;
// mm2N_ = n_ * c_;
template<typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardOneStage<T, U>::ProcessMatmul1(
    const int32_t taskOffset, const int32_t mm1M)
{
    if (mm1M <= 0)
        return;

    mm1_.SetTensorA(gradH2Global_[taskOffset * mm1K_]);
    mm1_.SetTensorB(phiGlobal_);
    mm1_.SetHF32(true, 1);   // HF32 matmul; golden HF32-ifies operands, phi input HF32-conditioned
    mm1_.SetSingleShape(mm1M, mm1N_, mm1K_);
    // grad_x fp32-accumulate: write grad_x_from_matmul as fp32 into its OWN workspace (cube is the
    // single writer here -> enAtomic=0). The combine adds it to the vector's fp32 hin+rms and casts
    // once to half. Cube being the sole writer of this region makes the AIV cross-core read coherent
    // (the AIV never cached it), avoiding the stale-self-cache hazard of an atomic-add onto an
    // AIV-written region.
    mm1_.template IterateAll<false> (gradXFp32Global_[taskOffset * (n_ * c_)], 0);
    mm1_.End();
}

template<typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardOneStage<T, U>::CombineGradX()
{
    // grad_x fp32-accumulate final stage (AIV): add the two fp32 grad_x sources --
    //   matmulIn = gradXFp32Global_  (cube grad_x_from_matmul, cross-core read, coherent after flag)
    //   hinRmsIn = gradXHinRmsWs_    (this core's grad_x_from_hin + grad_x_from_rms, its own write)
    // -- in fp32, then cast the sum ONCE to the half gradX output (CAST_RINT, matching the golden's
    // single .astype). Each AIV core covers exactly the tasks it processed in the main loop, so the
    // whole B*S*N*C grad_x is covered with no overlap.
    LocalTensor<T> matmulIn = gradXLocal_;     // fp32 staging (DOUBLE_BUF * c_ floats)
    LocalTensor<T> hinRmsIn = xFp32Local_;     // fp32 staging (DOUBLE_BUF * c_ floats)
    LocalTensor<U> halfOut = xLocal_;          // half staging (reuse x buffer: n_*c_ halves)
    int8_t ping = 0;
    // Drain this core's prior MTE3 writes (the main-loop grad_x_from_hin/rms store to gradXHinRmsWs_)
    // so the combine's MTE2 reads of that same region see committed GM data (intra-core coherency).
    SetFlag<HardEvent::MTE3_MTE2>(7);
    WaitFlag<HardEvent::MTE3_MTE2>(7);
    SetFlag<HardEvent::MTE3_MTE2>(0);
    SetFlag<HardEvent::MTE3_MTE2>(1);
    for (int32_t taskOffset = blkIdx_ * tile_; taskOffset < totalTasksAligned_; taskOffset += aivNum_ * tile_) {
        int32_t tileTaskCount = min(tile_, totalTasks_ - taskOffset);
        if (tileTaskCount <= 0) {
            continue;
        }
        for (int32_t t = 0; t < tileTaskCount; t++) {
            for (int32_t i = 0; i < n_; i++) {
                int64_t elemOffset = (int64_t)(taskOffset + t) * n_ * c_ + (int64_t)i * c_;
                WaitFlag<HardEvent::MTE3_MTE2>(ping);
                DataCopyPad(matmulIn[ping * c_], gradXFp32Global_[elemOffset],
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(c_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
                DataCopyPad(hinRmsIn[ping * c_], gradXHinRmsWs_[elemOffset],
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(c_ * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_V>(ping);
                WaitFlag<HardEvent::MTE2_V>(ping);
                Add(matmulIn[ping * c_], matmulIn[ping * c_], hinRmsIn[ping * c_], c_);
                Cast(halfOut[ping * c_], matmulIn[ping * c_], RoundMode::CAST_RINT, c_);
                SetFlag<HardEvent::V_MTE3>(ping);
                WaitFlag<HardEvent::V_MTE3>(ping);
                DataCopyPad(gradXGlobal_[elemOffset], halfOut[ping * c_],
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(c_ * sizeof(U)), 0, 0, 0});
                SetFlag<HardEvent::MTE3_MTE2>(ping);
                ping = 1 - ping;
            }
        }
    }
    WaitFlag<HardEvent::MTE3_MTE2>(0);
    WaitFlag<HardEvent::MTE3_MTE2>(1);
}

template<typename T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardOneStage<T, U>::ProcessMatmul2(
    const int32_t taskOffset, const int32_t mm2K)
{
    if (mm2K <= 0)
        return;

    mm2_.SetTensorA(gradH2Global_[taskOffset * mm2M_], true);
    mm2_.SetTensorB(xFp32Global_[taskOffset * mm2N_]);
    mm2_.SetHF32(true, 1);   // HF32 matmul; golden HF32-ifies operands (grad_hc_before_norm & x)
    mm2_.SetSingleShape(mm2M_, mm2N_, mm2K);
    mm2_.template IterateAll<false> (gradPhiGlobal_, 1);
    mm2_.End();
}

#endif // MHC_PRE_SINKHORN_BACKWARD_ONE_STAGE_H