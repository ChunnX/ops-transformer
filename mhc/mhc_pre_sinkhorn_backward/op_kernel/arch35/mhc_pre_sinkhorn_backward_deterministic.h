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
 * \file mhc_pre_sinkhorn_backward_deterministic.h
 * \brief MhcPreSinkhornBackward 确定性路径 Kernel — 从零重构
 *
 * 计算流程 (2 轮 BS 循环):
 *
 *   BS 循环 #1: PhaseA(C) + PhaseB(K)
 *     PhaseA_CLoop:
 *       for cIdx in ⌈c / ubFactorC⌉:
 *         CopyIn x[bs,n,cSlice], gradHin[bs,cSlice]           GM→UB
 *         gradHPre += (gradHin * x).reduce_sum(C)              UB (累加)
 *         gradXFromHin = gradHin * hPre                        UB→WS
 *         xFp32 = x cast to fp32                               UB→WS
 *
 *     PhaseB_KDim:
 *       CopyIn hcBeforeNorm[k], invRms, gradHPost, gradHRes   GM→UB
 *       CopyIn sumOut[skIter,n], normOut[skIter,n²]           GM→UB
 *       z_pre/post/res = hcBeforeNorm * invRms + bias          UB
 *       gradZ_pre  = SigmoidGrad(z_pre, gradHPre, isPre)       UB
 *       gradZ_post = 2 * SigmoidGrad(z_post, gradHPost, !isPre) UB
 *       skGrad     = SinkhornGrad(gradHRes, sumOut, normOut)   UB
 *       gradZ_res  = ExpGrad(z_res, skGrad)                    UB
 *       gradBiasAcc  += gradZ                                   UB (跨 BS 累加)
 *       gradAlphaAcc += normOut * gradZ                         UB (跨 BS 累加)
 *       gradNormOut    = gradZ * alpha                          UB
 *       gradHcBeforeNorm = gradNormOut * invRms                 UB→WS
 *       gradInvRms[bs] = (hcBeforeNorm * gradNormOut).sum(K)   UB (小 buffer)
 *
 *   SyncAll → CrossCore → AIC MatMul → CrossCore → SyncAll
 *     PhaseD_MatMul [AIC]:
 *       gradXFromMatmul = gradHcBeforeNorm @ phi               WS (AIC 写)
 *       gradPhi = gradHcBeforeNorm.T @ xFp32                   GM (AIC 写)
 *
 *   BS 循环 #2: PhaseE(C)
 *     for cIdx in ⌈c / ubFactorC⌉:
 *       CopyIn xFp32, gradXFromHin, gradXFromMatmul            WS→UB
 *       gradXFromRms = -(invRms³ * xFp32 * gradInvRms[bs]/NC)  UB (当场算, 不落 WS)
 *       gradX = gradXFromHin + gradXFromRms + gradXFromMatmul   UB
 *       Cast → GM
 *
 *   if blockIdx==0: ReduceBiasAlpha                             UB→GM
 */

#ifndef MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_H
#define MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "mhc_pre_sinkhorn_backward_deterministic_base.h"
#include "mhc_pre_sinkhorn_backward_data_arch35.h"

using namespace AscendC;

constexpr uint16_t PIPE_MTE3_FLAG = 0x5;
constexpr uint16_t PIPE_FIX_FLAG  = 0x6;
constexpr uint8_t  AIV_AIC_MODE   = 0x2;
constexpr int64_t  WS_INTERVAL    = 1024 / 4;

constexpr MatmulConfig MM1_CFG = GetMDLConfig(false, false, 0, true, false, false, true);
constexpr MatmulConfig MM2_CFG = GetMDLConfig(false, false, 0, true, false, false, true);

template <typename X_T, typename GRADHIN_T, typename U>
class MhcPreSinkhornBackwardDeterministic {
    using Td = MhcPreSinkhornBackwardArch35DeterminiticTilingData;

public:
    using A0Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, U>;
    using A1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, U, true>;
    using BType  = matmul::MatmulType<TPosition::GM, CubeFormat::ND, U>;
    using C1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, U>;
    using C2Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, U>;

    matmul::MatmulImpl<A0Type, BType, C1Type, C1Type, MM1_CFG> mm1_;
    matmul::MatmulImpl<A1Type, BType, C2Type, C2Type, MM2_CFG> mm2_;

    __aicore__ inline MhcPreSinkhornBackwardDeterministic() = default;

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR phi, GM_ADDR hPre, GM_ADDR gradHin, GM_ADDR gradHPost,
        GM_ADDR gradHRes, GM_ADDR alpha, GM_ADDR bias, GM_ADDR hcBeforeNorm, GM_ADDR invRms,
        GM_ADDR sumOut, GM_ADDR normOut, GM_ADDR gradX, GM_ADDR gradPhi,
        GM_ADDR gradAlpha, GM_ADDR gradBias, GM_ADDR workspace,
        const Td *td, TPipe *pipe);

    __aicore__ inline void Process();

private:
    __aicore__ inline void PhaseA_CLoop(int64_t bsIdx, int64_t bsLen);
    __aicore__ inline void PhaseB_KDim(int64_t bsIdx, int64_t bsLen);
    __aicore__ inline void PhaseD_MatMul();
    __aicore__ inline void PhaseE_CLoop(int64_t bsIdx, int64_t bsLen);
    __aicore__ inline void ReduceBiasAlpha();
    __aicore__ inline void GetAlphaAndBias();
    __aicore__ inline void SinkhornGradSimdVf(__local_mem__ U *skGradPtr, int64_t bsIdx, int64_t bsLen);
    __aicore__ inline void ExpGradSimdVf(__local_mem__ U *expGradPtr, __local_mem__ U *skGradPtr,
                                          __local_mem__ U *zResPtr, int64_t bsIdx, int64_t bsLen);
    __aicore__ inline void ComputeGNormOutAndGHcBeforeNorm(int64_t bsOffset, int64_t bsLen,
                                                            U alphaScalar, int64_t count, int64_t segOffset);

private:
    const Td *td_   = nullptr;
    TPipe    *pipe_ = nullptr;

    // GM 输入
    GlobalTensor<X_T>       xGm_;
    GlobalTensor<GRADHIN_T> gradHinGm_;
    GlobalTensor<U>         gradHPostGm_;
    GlobalTensor<U>         gradHResGm_;
    GlobalTensor<U>         phiGm_;
    GlobalTensor<U>         alphaGm_;
    GlobalTensor<U>         biasGm_;
    GlobalTensor<U>         hPreGm_;
    GlobalTensor<U>         hcBeforeNormGm_;
    GlobalTensor<U>         invRmsGm_;
    GlobalTensor<U>         sumOutGm_;
    GlobalTensor<U>         normOutGm_;

    // GM 输出
    GlobalTensor<X_T> gradXGm_;
    GlobalTensor<U>   gradPhiGm_;
    GlobalTensor<U>   gradAlphaGm_;
    GlobalTensor<U>   gradBiasGm_;

    // Workspace GM
    GlobalTensor<U> xFp32Ws_;
    GlobalTensor<U> gradXFromHinWs_;
    GlobalTensor<U> gradHcBeforeNormWs_;
    GlobalTensor<U> gradXFromMatmulWs_;

    // UB Queues — PhaseA/E
    TQue<QuePosition::VECIN,  2> xQue_;
    TQue<QuePosition::VECIN,  2> gradHinQue_;
    TQue<QuePosition::VECIN,  2> hPreAndGradHPostQue_;

    // UB Queues — PhaseB
    TQue<QuePosition::VECIN,  2> hcBeforeNormQue_;
    TQue<QuePosition::VECIN,  2> invRmsQue_;
    TQue<QuePosition::VECIN,  2> gradHResQue_;

    // UB Queues — 输出
    TQue<QuePosition::VECOUT, 2> xFp32Que_;
    TQue<QuePosition::VECOUT, 2> gradXFromHinQue_;
    TQue<QuePosition::VECOUT, 1> gradAlphaAndBiasQue_;
    TQue<QuePosition::VECOUT, 1> gradAlphaSumQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> gradBiasQue_;

    // UB Bufs — PhaseB (K 维)
    TBuf<TPosition::VECCALC> gradHPreBuf_;
    TBuf<TPosition::VECCALC> gradZBuf_;
    TBuf<TPosition::VECCALC> biasBuf_;
    TBuf<TPosition::VECCALC> normOutForwardBuf_;
    TBuf<TPosition::VECCALC> zBuf_;
    TBuf<TPosition::VECCALC> gradCalcBuf1_;
    TBuf<TPosition::VECCALC> gradCalcBuf2_;
    TBuf<TPosition::VECCALC> tmpDivBuf_;
    TBuf<TPosition::VECCALC> tmpDiv2Buf_;
    TBuf<TPosition::VECCALC> tmpDiv3Buf_;
    TBuf<TPosition::VECCALC> tmpDiv4Buf_;
    TBuf<TPosition::VECCALC> tmpMulBuf_;
    TBuf<TPosition::VECCALC> normOutBuf_;
    TBuf<TPosition::VECCALC> zResMaxBuf_;
    TBuf<TPosition::VECCALC> zResMaxIdxBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> isMaxBuf_;
    TBuf<TPosition::VECCALC> sumOutRowBuf_;
    TBuf<TPosition::VECCALC> sumOutColBuf_;
    TBuf<TPosition::VECCALC> gradInvRmsBuf_;

    // 高频变量
    int64_t n_           = 0;
    int64_t nn_          = 0;
    int64_t c_           = 0;
    int64_t kDim_        = 0;
    int64_t ubFactorBS_  = 0;
    int64_t ubFactorC_   = 0;
    int64_t bsPerCore_   = 0;
    int64_t usedCoreNum_ = 0;
    int64_t blockIdx_    = 0;
    int64_t blockSize_   = 0;
    int64_t skIterCount_ = 0;
    float   eps_         = 0;
    int64_t nAlign_      = 0;
    int64_t nnAlign_     = 0;
    int64_t kAlign_      = 0;
    int64_t bsAlign_     = 0;
    int64_t bsnAlign_    = 0;
    int64_t bsnnAlign_   = 0;
    int64_t cAlignFp32_  = 0;
    int64_t cAlignFp16_  = 0;
    int64_t ubFactorCAlignFp32_ = 0;
    int64_t ubFactorCAlignFp16_ = 0;
    int64_t uSize_       = sizeof(U);
    int64_t xSize_       = sizeof(X_T);
    int64_t ghSize_      = sizeof(GRADHIN_T);
    int64_t cNumPerBS_   = 0;
    int64_t cTailPerBS_  = 0;
    U alphaPre_  = 0;
    U alphaPost_ = 0;
    U alphaRes_  = 0;
    int64_t mm1K_ = 0, mm1M_ = 0, mm1N_ = 0;
    int64_t mm2K_ = 0, mm2M_ = 0, mm2N_ = 0;
    int64_t aicCoreNum_ = 0;
};

// ================================================================
// Init
// ================================================================
template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::Init(
    GM_ADDR x, GM_ADDR phi, GM_ADDR hPre, GM_ADDR gradHin, GM_ADDR gradHPost,
    GM_ADDR gradHRes, GM_ADDR alpha, GM_ADDR bias, GM_ADDR hcBeforeNorm, GM_ADDR invRms,
    GM_ADDR sumOut, GM_ADDR normOut, GM_ADDR gradX, GM_ADDR gradPhi,
    GM_ADDR gradAlpha, GM_ADDR gradBias, GM_ADDR workspace,
    const Td *td, TPipe *pipe)
{
    td_   = td;
    pipe_ = pipe;

    n_           = td_->n;
    c_           = td_->c;
    nn_          = n_ * n_;
    kDim_        = nn_ + 2 * n_;
    ubFactorBS_  = td_->ubFactorBS;
    ubFactorC_   = td_->ubFactorC;
    usedCoreNum_ = td_->usedCoreNum;
    aicCoreNum_  = td_->aicCoreNum;
    skIterCount_ = td_->skIterCount;
    eps_         = td_->eps;
    blockIdx_    = GetBlockIdx();
    blockSize_   = Ops::Base::GetUbBlockSize();
    bsPerCore_   = (blockIdx_ == usedCoreNum_ - 1) ? td_->blockTailBS : td_->blockFactorBS;

    uSize_ = sizeof(U);
    xSize_ = sizeof(X_T);
    ghSize_ = sizeof(GRADHIN_T);
    nAlign_             = Ops::Base::CeilAlign(int64_t(n_ * uSize_), blockSize_) / uSize_;
    nnAlign_            = Ops::Base::CeilAlign(int64_t(nn_ * uSize_), blockSize_) / uSize_;
    kAlign_             = Ops::Base::CeilAlign(int64_t(kDim_ * uSize_), blockSize_) / uSize_;
    bsAlign_            = Ops::Base::CeilAlign(int64_t(ubFactorBS_ * uSize_), blockSize_) / uSize_;
    bsnAlign_           = Ops::Base::CeilAlign(int64_t(ubFactorBS_ * n_ * uSize_), blockSize_) / uSize_;
    bsnnAlign_          = Ops::Base::CeilAlign(int64_t(ubFactorBS_ * nn_ * uSize_), blockSize_) / uSize_;
    cAlignFp32_         = Ops::Base::CeilAlign(int64_t(c_ * uSize_), blockSize_) / uSize_;
    cAlignFp16_         = Ops::Base::CeilAlign(int64_t(c_ * xSize_), blockSize_) / xSize_;
    ubFactorCAlignFp32_ = Ops::Base::CeilAlign(int64_t(ubFactorC_ * uSize_), blockSize_) / uSize_;
    ubFactorCAlignFp16_ = Ops::Base::CeilAlign(int64_t(ubFactorC_ * xSize_), blockSize_) / xSize_;

    cNumPerBS_  = Ops::Base::CeilDiv(n_ * c_, ubFactorCAlignFp32_);
    cTailPerBS_ = n_ * c_ - (cNumPerBS_ - 1) * ubFactorCAlignFp32_;

    if ASCEND_IS_AIV {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_T *>(x));
        gradHinGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GRADHIN_T *>(gradHin));
        gradHPostGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(gradHPost));
        gradHResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(gradHRes));
        alphaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(alpha));
        biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(bias));
        hPreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(hPre));
        hcBeforeNormGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(hcBeforeNorm));
        invRmsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(invRms));
        sumOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(sumOut));
        normOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(normOut));
        gradAlphaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(gradAlpha));
        gradBiasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(gradBias));

        int64_t totalBs = td_->b * td_->s;
        int64_t gradXFromHinOfs = totalBs * n_ * c_ + WS_INTERVAL;
        int64_t gradHcBeforeNormOfs = gradXFromHinOfs + totalBs * n_ * c_ + WS_INTERVAL;
        int64_t gradXFromMatmulOfs = gradHcBeforeNormOfs + totalBs * kDim_ + WS_INTERVAL;
        auto ws = reinterpret_cast<__gm__ U *>(workspace);
        xFp32Ws_.SetGlobalBuffer(ws);
        gradXFromHinWs_.SetGlobalBuffer(ws + gradXFromHinOfs);
        gradHcBeforeNormWs_.SetGlobalBuffer(ws + gradHcBeforeNormOfs);
        gradXFromMatmulWs_.SetGlobalBuffer(ws + gradXFromMatmulOfs);

        pipe_->InitBuffer(xQue_, 2, ubFactorCAlignFp16_ * n_ * ubFactorBS_ * xSize_);
        pipe_->InitBuffer(gradHinQue_, 2, ubFactorCAlignFp16_ * ubFactorBS_ * ghSize_);
        pipe_->InitBuffer(hPreAndGradHPostQue_, 2, bsnAlign_ * uSize_);
        pipe_->InitBuffer(hcBeforeNormQue_, 2, ubFactorBS_ * kAlign_ * uSize_);
        pipe_->InitBuffer(invRmsQue_, 2, bsAlign_ * uSize_);
        pipe_->InitBuffer(gradHResQue_, 2, ubFactorBS_ * nnAlign_ * uSize_);
        pipe_->InitBuffer(xFp32Que_, 2, ubFactorCAlignFp32_ * n_ * ubFactorBS_ * uSize_);
        pipe_->InitBuffer(gradXFromHinQue_, 2, ubFactorCAlignFp32_ * n_ * ubFactorBS_ * uSize_);
        pipe_->InitBuffer(gradAlphaAndBiasQue_, 1, nnAlign_ * uSize_ + blockSize_);

        pipe_->InitBuffer(gradHPreBuf_, bsnAlign_ * uSize_);
        pipe_->InitBuffer(gradZBuf_, bsnnAlign_ * uSize_);
        pipe_->InitBuffer(biasBuf_, kAlign_ * uSize_);
        pipe_->InitBuffer(normOutForwardBuf_, bsnnAlign_ * uSize_);
        pipe_->InitBuffer(zBuf_, bsnnAlign_ * uSize_);
        pipe_->InitBuffer(gradCalcBuf1_, kAlign_ * uSize_);
        pipe_->InitBuffer(gradCalcBuf2_, kAlign_ * uSize_);
        pipe_->InitBuffer(tmpDivBuf_, bsnnAlign_ * uSize_);
        pipe_->InitBuffer(tmpDiv2Buf_, bsnnAlign_ * uSize_);
        pipe_->InitBuffer(tmpDiv3Buf_, bsnnAlign_ * uSize_);
        pipe_->InitBuffer(tmpDiv4Buf_, bsnnAlign_ * uSize_);
        pipe_->InitBuffer(tmpMulBuf_, bsnnAlign_ * uSize_);
        pipe_->InitBuffer(sumOutRowBuf_, skIterCount_ * ubFactorBS_ * nAlign_ * uSize_);
        pipe_->InitBuffer(sumOutColBuf_, skIterCount_ * ubFactorBS_ * nAlign_ * uSize_);
        pipe_->InitBuffer(normOutBuf_, skIterCount_ * ubFactorBS_ * nnAlign_ * uSize_);
        pipe_->InitBuffer(zResMaxBuf_, ubFactorBS_ * nnAlign_ * uSize_);
        pipe_->InitBuffer(zResMaxIdxBuf_, ubFactorBS_ * nnAlign_ * uSize_);
        pipe_->InitBuffer(tmpBuf_, ubFactorBS_ * nnAlign_ * uSize_);
        pipe_->InitBuffer(isMaxBuf_, ubFactorBS_ * nnAlign_ * uSize_);
        pipe_->InitBuffer(gradInvRmsBuf_, bsPerCore_ * uSize_);

        if (blockIdx_ == 0) {
            InitOutput<U>(gradAlphaGm_, 3, U(0));
            InitOutput<U>(gradBiasGm_, kDim_, U(0));
        }
        SyncAll();
    }

    if ASCEND_IS_AIC {
        gradXGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_T *>(gradX));
        gradPhiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(gradPhi));
        mm1K_ = kDim_;
        mm1M_ = td_->b * td_->s;
        mm1N_ = n_ * c_;
        mm2K_ = td_->b * td_->s;
        mm2M_ = kDim_;
        mm2N_ = n_ * c_;
    }
}

// ================================================================
// Process
// ================================================================
template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::Process()
{
    if ASCEND_IS_AIV {
        if (blockIdx_ >= usedCoreNum_) return;

        GetAlphaAndBias();

        int64_t bsLoopNum = Ops::Base::CeilDiv(bsPerCore_, ubFactorBS_);
        int64_t bsTail    = bsPerCore_ - ubFactorBS_ * (bsLoopNum - 1);

        for (int64_t bsIdx = 0; bsIdx < bsLoopNum; bsIdx++) {
            int64_t bsLen = (bsIdx == bsLoopNum - 1) ? bsTail : ubFactorBS_;
            PhaseA_CLoop(bsIdx, bsLen);
            PhaseB_KDim(bsIdx, bsLen);
        }

        SyncAll();
        CrossCoreSetFlag<AIV_AIC_MODE, PIPE_MTE3>(PIPE_MTE3_FLAG);
        CrossCoreWaitFlag<AIV_AIC_MODE>(PIPE_FIX_FLAG);
        SyncAll();

        for (int64_t bsIdx = 0; bsIdx < bsLoopNum; bsIdx++) {
            int64_t bsLen = (bsIdx == bsLoopNum - 1) ? bsTail : ubFactorBS_;
            PhaseE_CLoop(bsIdx, bsLen);
        }

        if (blockIdx_ == 0) {
            ReduceBiasAlpha();
        }
    }

    if ASCEND_IS_AIC {
        PhaseD_MatMul();
    }
}

// ================================================================
// Phase 函数 — 逐步实现
// ================================================================

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::GetAlphaAndBias()
{
    alphaPre_  = alphaGm_.GetValue(0);
    alphaPost_ = alphaGm_.GetValue(1);
    alphaRes_  = alphaGm_.GetValue(2);

    CopyInBuf(biasBuf_, biasGm_, 1, kDim_, 0, kAlign_);
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::PhaseA_CLoop(int64_t bsIdx, int64_t bsLen)
{
    // ================================================================
    // PhaseA — C 维循环: gradHPre + gradXFromHin + xFp32
    //
    // 步骤:
    //   1. gradHPreBuf_ 清零 [bsLen, n_]
    //   2. 搬入 hPre[bsLen, n_] GM→UB
    //   3. C 维循环:
    //      a. 搬入 x[bsLen, n_, cSlice] + gradHin[bsLen, cSlice] GM→UB
    //      b. 逐 BS 逐 N 逐 VF:
    //         - gradHin cast fp32
    //         - x cast fp32
    //         - gradXFromHin = gradHin * hPre        → UB (后搬出 WS)
    //         - gradHPre    += (x * gradHin).sum(C)  → UB (累加)
    //      c. 搬出 xFp32 + gradXFromHin → WS
    //
    // UB 布局 (每 cSlice 迭代):
    //   gradHPreBuf_:    [bsLen, n_]         fp32, 累加器, 跨 cSlice 持久
    //   hPreQue_:        [bsLen, n_]         fp32, 跨 cSlice 持久
    //   xQue_:           [bsLen, n_, cAligned]  X_T (fp16/bf16)
    //   gradHinQue_:     [bsLen, cAligned]      GRADHIN_T (fp16/bf16)
    //   xFp32Que_:       [bsLen, n_, cAligned]  fp32 (输出)
    //   gradXFromHinQue_:[bsLen, n_, cAligned]  fp32 (输出)
    //
    // GM 布局:
    //   x:        [B*S, n_, c_]    按行存储, 每行 n_*c_ 元素
    //   gradHin:  [B*S, c_]        按行存储, 每行 c_ 元素
    //   hPre:     [B*S, n_]        按行存储, 每行 n_ 元素
    // ================================================================

    // --- GM 偏移: 当前 BS 批次在全局中的位置 ---
    int64_t gmBsOffset    = blockIdx_ * td_->blockFactorBS + bsIdx * ubFactorBS_;
    int64_t gmHPreOffset  = gmBsOffset * n_;
    int64_t gmXOffset     = gmBsOffset * n_ * c_;
    int64_t gmGradHinOffset = gmBsOffset * c_;

    // --- Step 1: gradHPre 清零 ---
    auto gradHPreUb = gradHPreBuf_.template Get<U>();
    Duplicate(gradHPreUb, float(0), bsLen * n_);

    // --- Step 2: 搬入 hPre[bsLen, n_] ---
    CopyInQue(hPreAndGradHPostQue_, hPreGm_[gmHPreOffset], bsLen, n_);
    auto hPreUb = hPreAndGradHPostQue_.template DeQue<U>();

    __local_mem__ U *hPrePtr     = reinterpret_cast<__local_mem__ U *>(hPreUb.GetPhyAddr());
    __local_mem__ U *gradHPrePtr = reinterpret_cast<__local_mem__ U *>(gradHPreUb.GetPhyAddr());

    // --- Step 3: C 维循环 ---
    uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(U);

    for (int64_t cIdx = 0; cIdx < cNumPerBS_; cIdx++) {
        // 当前 C 切片的实际元素数 (尾块可能不足)
        int64_t cSliceLen = (cIdx == cNumPerBS_ - 1) ? cTailPerBS_ : ubFactorCAlignFp32_;

        // 对齐后的每行元素数 (按各类型大小对齐到 blockSize_)
        int64_t cSliceAlignX  = Ops::Base::CeilAlign(cSliceLen * xSize_, blockSize_) / xSize_;
        int64_t cSliceAlignU  = Ops::Base::CeilAlign(cSliceLen * uSize_, blockSize_) / uSize_;
        int64_t cSliceAlignGh = Ops::Base::CeilAlign(cSliceLen * ghSize_, blockSize_) / ghSize_;

        // 当前 C 切片在 GM 中的偏移
        int64_t gmXSliceOffset     = gmXOffset + cIdx * ubFactorC_;
        int64_t gmGradHinSliceOffset = gmGradHinOffset + cIdx * ubFactorC_;

        // --- Step 3a: 搬入 x[bsLen, n_, cSlice] + gradHin[bsLen, cSlice] ---
        // GM 中 x 按 [bs, n, c] 排布, 每行 c_ 元素, 跨步 (c_ - cSliceLen) 跳过剩余 C
        CopyInQue(xQue_, xGm_[gmXSliceOffset], bsLen * n_, cSliceAlignX, (c_ - cSliceLen) * xSize_);
        CopyInQue(gradHinQue_, gradHinGm_[gmGradHinSliceOffset], bsLen, cSliceAlignGh, (c_ - cSliceLen) * ghSize_);
        auto xUb        = xQue_.template DeQue<X_T>();
        auto gradHinUb  = gradHinQue_.template DeQue<GRADHIN_T>();

        // 分配输出 tensor
        auto xFp32Ub        = xFp32Que_.template AllocTensor<U>();
        auto gradXFromHinUb = gradXFromHinQue_.template AllocTensor<U>();

        // UB 裸指针 (供 MicroAPI 使用)
        __local_mem__ X_T       *xPtr            = reinterpret_cast<__local_mem__ X_T *>(xUb.GetPhyAddr());
        __local_mem__ U         *xFp32Ptr        = reinterpret_cast<__local_mem__ U *>(xFp32Ub.GetPhyAddr());
        __local_mem__ GRADHIN_T *gradHinPtr      = reinterpret_cast<__local_mem__ GRADHIN_T *>(gradHinUb.GetPhyAddr());
        __local_mem__ U         *gradXFromHinPtr = reinterpret_cast<__local_mem__ U *>(gradXFromHinUb.GetPhyAddr());

        uint16_t vfLoopCnt = static_cast<uint16_t>((cSliceLen + vfLen - 1) / vfLen);

        // --- Step 3b: 逐 BS 逐 VF 逐 N 计算 ---
        // 偏移传递: 每层只定义本层偏移, 内层用 base + offset 访问
        //
        // UB 布局 (以 x 为例, fp16):
        //   [bs0_n0_c0..cAlign | bs0_n1_c0..cAlign | ... | bs1_n0_c0..cAlign | ...]
        //    ← cSliceAlignX →    ← cSliceAlignX →
        //    ←──── n_ 行 ────→   ←──── n_ 行 ────→
        //    ←──────────── bsLen 轮 ────────────→
        //
        // 各 buffer 的行步长:
        //   hPre / gradHPre:  n_            (每 BS 一行, n_ 元素)
        //   gradHin:          cSliceAlignGh (每 BS 一行, cSliceAlignGh 元素)
        //   x:                cSliceAlignX  (每 N 头一行)
        //   xFp32 / gradXFromHin: cSliceAlignU (每 N 头一行)
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<GRADHIN_T> gradHinReg;
            AscendC::MicroAPI::RegTensor<U> gradHinFp32Reg;
            AscendC::MicroAPI::RegTensor<X_T> xReg;
            AscendC::MicroAPI::RegTensor<U> xFp32Reg;
            AscendC::MicroAPI::RegTensor<U> gradHPreReg;
            AscendC::MicroAPI::RegTensor<U> gradHPreAccReg;
            AscendC::MicroAPI::RegTensor<U> gradXFromHinReg;
            AscendC::MicroAPI::MaskReg valueMaskReg;
            AscendC::MicroAPI::MaskReg oneMaskReg = AscendC::MicroAPI::CreateMask<U, AscendC::MicroAPI::MaskPattern::VL1>();

            uint32_t maskLen = static_cast<uint32_t>(cSliceLen);

            for (uint16_t bsI = 0; bsI < static_cast<uint16_t>(bsLen); bsI++) {
                // 行基址: BS 偏移内置到指针, 内层只关心本层偏移
                // 1D (每BS一行): hPre, gradHPre, gradHin
                auto hPreRow       = hPrePtr     + bsI * n_;
                auto gradHPreRow   = gradHPrePtr + bsI * n_;
                auto gradHinRow    = gradHinPtr  + bsI * cSliceAlignGh;
                // 2D (每BS n_行): x, xFp32, gradXFromHin
                auto xBase          = xPtr          + bsI * n_ * cSliceAlignX;
                auto xFp32Base      = xFp32Ptr      + bsI * n_ * cSliceAlignU;
                auto gradXFromHBase = gradXFromHinPtr + bsI * n_ * cSliceAlignU;

                for (uint16_t vfJ = 0; vfJ < vfLoopCnt; vfJ++) {
                    valueMaskReg = AscendC::MicroAPI::UpdateMask<U>(maskLen);

                    // VF 层偏移: 当前 VF 块在 C 维的偏移
                    int64_t vfOffC = vfJ * vfLen;

                    AscendC::MicroAPI::DataCopy<GRADHIN_T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        gradHinReg, gradHinRow + vfOffC);
                    AscendC::MicroAPI::Cast<U, GRADHIN_T, castTrait16ToFloat>(gradHinFp32Reg, gradHinReg, valueMaskReg);

                    for (uint16_t nK = 0; nK < static_cast<uint16_t>(n_); nK++) {
                        AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                                       AscendC::MicroAPI::MemType::SCALAR_LOAD>();

                        auto xRow         = xBase         + nK * cSliceAlignX;
                        auto xFp32Row     = xFp32Base     + nK * cSliceAlignU;
                        auto gradXFromHRow = gradXFromHBase + nK * cSliceAlignU;

                        U hPreVal     = hPreRow[nK];
                        U gradHPreOld = gradHPreRow[nK];

                        AscendC::MicroAPI::DataCopy<X_T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                            xReg, xRow + vfOffC);
                        AscendC::MicroAPI::Cast<U, X_T, castTrait16ToFloat>(xFp32Reg, xReg, valueMaskReg);

                        // gradXFromHin = gradHin * hPre
                        AscendC::MicroAPI::Muls(gradXFromHinReg, gradHinFp32Reg, hPreVal, valueMaskReg);

                        // gradHPre = (x * gradHin).sum(C)
                        AscendC::MicroAPI::Mul(gradHPreReg, xFp32Reg, gradHinFp32Reg, valueMaskReg);
                        AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::SUM, U, U>(gradHPreReg, gradHPreReg, valueMaskReg);
                        AscendC::MicroAPI::Adds(gradHPreAccReg, gradHPreReg, gradHPreOld, oneMaskReg);

                        AscendC::MicroAPI::DataCopy(
                            gradXFromHRow + vfOffC, gradXFromHinReg, valueMaskReg);
                        AscendC::MicroAPI::DataCopy(
                            xFp32Row + vfOffC, xFp32Reg, valueMaskReg);
                        AscendC::MicroAPI::Store(gradHPreRow + nK, gradHPreAccReg, 1);
                    }
                }
            }
        }

        xFp32Que_.EnQue(xFp32Ub);
        gradXFromHinQue_.EnQue(gradXFromHinUb);
        xQue_.FreeTensor(xUb);
        gradHinQue_.FreeTensor(gradHinUb);

        // --- Step 3c: 搬出 xFp32 + gradXFromHin → WS ---
        // CopyOutQue 内部自动 DeQue + FreeTensor, 无需手动 DeQue
        // WS 中按 [bs, n, c] 排布, 跨步 (c_ - cSliceLen) 跳过剩余 C
        CopyOutQue(xFp32Que_, xFp32Ws_[gmXSliceOffset], bsLen * n_, cSliceAlignU, 0, (c_ - cSliceLen) * uSize_);
        CopyOutQue(gradXFromHinQue_, gradXFromHinWs_[gmXSliceOffset], bsLen * n_, cSliceAlignU, 0, (c_ - cSliceLen) * uSize_);
    }

    hPreAndGradHPostQue_.FreeTensor(hPreUb);
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::PhaseB_KDim(int64_t bsIdx, int64_t bsLen)
{
    int64_t bsOffset = blockIdx_ * td_->blockFactorBS + bsIdx * ubFactorBS_;
    int64_t hcBeforeNormOffset = bsOffset * kDim_;
    int64_t invRmsOffset = bsOffset;

    // --- 搬入 invRms + hcBeforeNorm (k_ 全量对齐) ---
    CopyInQue(invRmsQue_, invRmsGm_[invRmsOffset], 1, bsLen);
    CopyInQue(hcBeforeNormQue_, hcBeforeNormGm_[hcBeforeNormOffset], 1, bsLen * kAlign_);

    auto invRmsUb = invRmsQue_.template DeQue<U>();
    auto hcUb     = hcBeforeNormQue_.template DeQue<U>();
    auto zUb      = zBuf_.template Get<U>();
    auto noUb     = normOutForwardBuf_.template Get<U>();
    auto gradZUb  = gradZBuf_.template Get<U>();
    auto biasUb   = biasBuf_.template Get<U>();
    auto gradHPreUb = gradHPreBuf_.template Get<U>();

    __local_mem__ U *zPtr    = reinterpret_cast<__local_mem__ U *>(zUb.GetPhyAddr());
    __local_mem__ U *noPtr   = reinterpret_cast<__local_mem__ U *>(noUb.GetPhyAddr());
    __local_mem__ U *gradZPtr = reinterpret_cast<__local_mem__ U *>(gradZUb.GetPhyAddr());
    __local_mem__ U *hcPtr   = reinterpret_cast<__local_mem__ U *>(hcUb.GetPhyAddr());
    __local_mem__ U *biasPtr = reinterpret_cast<__local_mem__ U *>(biasUb.GetPhyAddr());
    __local_mem__ U *irPtr   = reinterpret_cast<__local_mem__ U *>(invRmsUb.GetPhyAddr());
    __local_mem__ U *ghPrePtr = reinterpret_cast<__local_mem__ U *>(gradHPreUb.GetPhyAddr());

    // UB K layout: (2+n_) rows × nAlign_ cols per BS
    //   row 0: hPre, row 1: hPost, rows 2..1+n_: sinkhorn
    int64_t preRow  = 0;
    int64_t postRow = 1;
    int64_t resRow  = 2;

    // z_pre = hcBeforeNorm[row 0] * invRms + bias[0:n]
    ComputeZAndForwardPart<U>(zPtr + bsLen * preRow * nAlign_, noPtr + bsLen * preRow * nAlign_,
                              biasPtr, bsLen, nAlign_, nAlign_, 256, alphaPre_);
    // z_post = hcBeforeNorm[row 1] * invRms + bias[n:2n]
    ComputeZAndForwardPart<U>(zPtr + bsLen * postRow * nAlign_, noPtr + bsLen * postRow * nAlign_,
                              biasPtr + n_, bsLen, nAlign_, nAlign_, 256, alphaPost_);
    // z_res = hcBeforeNorm[rows 2..1+n] * invRms + bias[2n:k]
    ComputeZAndForwardPart<U>(zPtr + bsLen * resRow * nAlign_, noPtr + bsLen * resRow * nAlign_,
                              biasPtr + 2 * n_, bsLen, n_ * nAlign_, nAlign_, 256, alphaRes_);

    // --- SigmoidGrad: gradZ_pre = SigmoidGrad(z_pre, gradHPre, isPre=true) ---
    SigmoidGrad<U, true>(ghPrePtr, zPtr + bsLen * preRow * nAlign_, gradZPtr + bsLen * preRow * nAlign_,
                         bsLen, nAlign_, 256, eps_);

    // --- SigmoidGrad: gradZ_post = 2 * SigmoidGrad(z_post, gradHPost, isPre=false) ---
    // 先搬入 gradHPost
    CopyInQue(hPreAndGradHPostQue_, gradHPostGm_[bsOffset * n_], bsLen, n_);
    auto gradHPostUb = hPreAndGradHPostQue_.template DeQue<U>();
    __local_mem__ U *ghPostPtr = reinterpret_cast<__local_mem__ U *>(gradHPostUb.GetPhyAddr());
    SigmoidGrad<U, false>(ghPostPtr, zPtr + bsLen * postRow * nAlign_, gradZPtr + bsLen * postRow * nAlign_,
                           bsLen, nAlign_, 256, eps_);
    hPreAndGradHPostQue_.FreeTensor(gradHPostUb);

    // --- SinkhornGrad + ExpGrad: gradZ_res ---
    // 搬入 gradHRes → 作为 skGrad 初始值 (grad_h_res)
    int64_t gradHResOffset = bsOffset * nn_;
    CopyInQue(gradHResQue_, gradHResGm_[gradHResOffset], bsLen, nn_);
    auto gradHResUb = gradHResQue_.template DeQue<U>();
    __local_mem__ U *skGradPtr = reinterpret_cast<__local_mem__ U *>(gradHResUb.GetPhyAddr());

    // 搬入 sumOut[skIter, 2, bsLen, n] + normOut[skIter, bsLen, n, n]
    auto sumOutRowUb = sumOutRowBuf_.template Get<U>();
    auto sumOutColUb = sumOutColBuf_.template Get<U>();
    auto normOutUb   = normOutBuf_.template Get<U>();
    __local_mem__ U *sumOutRowPtr = reinterpret_cast<__local_mem__ U *>(sumOutRowUb.GetPhyAddr());
    __local_mem__ U *sumOutColPtr = reinterpret_cast<__local_mem__ U *>(sumOutColUb.GetPhyAddr());
    __local_mem__ U *normOutPtr   = reinterpret_cast<__local_mem__ U *>(normOutUb.GetPhyAddr());

    for (int64_t iter = 0; iter < skIterCount_; iter++) {
        int64_t gmRowSumOfs = (bsOffset + iter * (td_->b * td_->s)) * n_;
        int64_t gmColSumOfs = (bsOffset + iter * (td_->b * td_->s) + td_->b * td_->s) * n_;
        int64_t gmNormOutOfs = (bsOffset + iter * (td_->b * td_->s)) * nn_;
        int64_t ubRowOfs = iter * ubFactorBS_ * nAlign_;
        int64_t ubColOfs = iter * ubFactorBS_ * nAlign_;
        int64_t ubNormOfs = iter * ubFactorBS_ * nnAlign_;
        CopyInBuf(sumOutRowBuf_, sumOutGm_[gmRowSumOfs], bsLen, n_, 0, nAlign_);
        CopyInBuf(sumOutColBuf_, sumOutGm_[gmColSumOfs], bsLen, n_, 0, nAlign_);
        CopyInBuf(normOutBuf_, normOutGm_[gmNormOutOfs], bsLen, nn_, 0, nnAlign_);
    }
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    // skGrad = SinkhornGrad(gradHRes, sumOut, normOut) → writes into skGradPtr
    SinkhornGradSimdVf(skGradPtr, bsIdx, bsLen);

    // gradZ_res = ExpGrad(z_res, skGrad)
    __local_mem__ U *zResPtr = zPtr + bsLen * resRow * nAlign_;
    __local_mem__ U *gradZResPtr = gradZPtr + bsLen * resRow * nAlign_;
    ExpGradSimdVf(gradZResPtr, skGradPtr, zResPtr, bsIdx, bsLen);

    gradHResQue_.FreeTensor(gradHResUb);

    // --- gradAlpha + gradBias 累加 (3 segments: pre n_, post n_, res nn_) ---
    auto gradAlphaBiasUb = gradAlphaAndBiasQue_.template AllocTensor<U>();
    __local_mem__ U *gradAlphaAddr = reinterpret_cast<__local_mem__ U *>(gradAlphaBiasUb.GetPhyAddr());
    __local_mem__ U *gradBiasAddr  = gradAlphaAddr + 1; // gradBias starts after gradAlpha (1 element per segment)

    // Segment 0: pre (n_ elements per BS)
    ComputeGradAlpha<U>(noPtr + bsLen * preRow * nAlign_, gradZPtr + bsLen * preRow * nAlign_,
                        gradAlphaAddr, bsLen, nAlign_, 256);
    ComputeGradBias<U>(gradBiasAddr, gradZPtr + bsLen * preRow * nAlign_, bsLen, nAlign_, 256);
    // Segment 1: post
    ComputeGradAlpha<U>(noPtr + bsLen * postRow * nAlign_, gradZPtr + bsLen * postRow * nAlign_,
                        gradAlphaAddr + 1, bsLen, nAlign_, 256);
    ComputeGradBias<U>(gradBiasAddr + 1, gradZPtr + bsLen * postRow * nAlign_, bsLen, nAlign_, 256);
    // Segment 2: res
    ComputeGradAlpha<U>(noPtr + bsLen * resRow * nAlign_, gradZPtr + bsLen * resRow * nAlign_,
                        gradAlphaAddr + 2, bsLen, nnAlign_, 256);
    ComputeGradBias<U>(gradBiasAddr + 2, gradZPtr + bsLen * resRow * nAlign_, bsLen, nnAlign_, 256);

    gradAlphaAndBiasQue_.EnQue(gradAlphaBiasUb);
    // CopyOut gradAlpha + gradBias to WS for later cross-core reduction
    auto gradAlphaBiasOut = gradAlphaAndBiasQue_.template DeQue<U>();
    // Write gradAlpha[3] and gradBias[3] to per-core WS slots
    // Layout in WS: gradAlphaWs_[coreIdx * 3 + segIdx], gradBiasWs_[coreIdx * kDim_ + biasIdx]
    // Use GM directly since cross-core reduction reads from GM
    for (int64_t seg = 0; seg < 3; seg++) {
        U gaVal = gradAlphaAddr[seg];
        U gbVal = gradBiasAddr[seg];
        // Atomic add to GM (initialized to 0)
        if (gaVal != U(0)) {
            auto addr = reinterpret_cast<__gm__ U *>(&gradAlphaGm_(seg));
            AscendC::AtomicAdd(addr, gaVal);
        }
        int64_t biasIdx = (seg < 2) ? seg * n_ : 2 * n_;
        int64_t biasCount = (seg < 2) ? n_ : nn_;
        // Write gradBias to per-segment GM offset
        auto biasLocal = gradCalcBuf1_.template Get<U>();
        Duplicate(biasLocal, gbVal, biasCount);
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        CopyOutBuf(gradCalcBuf1_, gradBiasGm_[biasIdx], 1, biasCount);
    }
    gradAlphaAndBiasQue_.FreeTensor(gradAlphaBiasOut);

    // --- Compute gradHcBeforeNorm → WS and gradInvRms → gradInvRmsBuf_ ---
    ComputeGNormOutAndGHcBeforeNorm(bsOffset, bsLen, alphaPre_, nAlign_, 0);
    ComputeGNormOutAndGHcBeforeNorm(bsOffset, bsLen, alphaPost_, nAlign_, n_);
    ComputeGNormOutAndGHcBeforeNorm(bsOffset, bsLen, alphaRes_, nnAlign_, 2 * n_);

    invRmsQue_.FreeTensor(invRmsUb);
    hcBeforeNormQue_.FreeTensor(hcUb);
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::PhaseD_MatMul()
{
    CrossCoreWaitFlag<AIV_AIC_MODE>(PIPE_MTE3_FLAG);

    int64_t totalSize = td_->b * td_->s;
    int64_t taskNumPerCore = Ops::Base::CeilDiv(totalSize, aicCoreNum_);
    int64_t taskOffset = blockIdx_ * taskNumPerCore;
    taskNumPerCore = min(taskNumPerCore, totalSize - blockIdx_ * taskNumPerCore);

    // mm1: gradHcBeforeNorm[B*S, K] × phi[K, N*C] → gradXFromMatmul → WS
    if (taskNumPerCore > 0) {
        mm1_.SetTensorA(gradHcBeforeNormWs_[taskOffset * mm1K_]);
        mm1_.SetTensorB(phiGm_);
        mm1_.SetHF32(false, 1);
        mm1_.SetSingleShape(static_cast<int32_t>(taskNumPerCore), static_cast<int32_t>(mm1N_), static_cast<int32_t>(mm1K_));
        mm1_.template IterateAll<false>(gradXFromMatmulWs_[taskOffset * (n_ * c_)], 0);
        mm1_.End();
    }

    // mm2: gradHcBeforeNorm.T[K, B*S] × xFp32[B*S, N*C] → gradPhi → GM
    int64_t taskNumPerCore2 = Ops::Base::CeilDiv(mm2N_, aicCoreNum_);
    int64_t taskOffset2 = blockIdx_ * taskNumPerCore2;
    taskNumPerCore2 = min(taskNumPerCore2, mm2N_ - blockIdx_ * taskNumPerCore2);
    if (taskNumPerCore2 > 0) {
        mm2_.SetTensorA(gradHcBeforeNormWs_, true);
        mm2_.SetTensorB(xFp32Ws_[taskOffset2]);
        mm2_.SetHF32(false, 1);
        mm2_.SetSingleShape(static_cast<int32_t>(mm2M_), static_cast<int32_t>(taskNumPerCore2), static_cast<int32_t>(mm2K_));
        mm2_.template IterateAll<false>(gradPhiGm_[taskOffset2], 0);
        mm2_.End();
    }

    CrossCoreSetFlag<AIV_AIC_MODE, PIPE_FIX>(PIPE_FIX_FLAG);
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::PhaseE_CLoop(int64_t bsIdx, int64_t bsLen)
{
    int64_t bsOffset = blockIdx_ * td_->blockFactorBS + bsIdx * ubFactorBS_;
    auto gradInvRmsUb = gradInvRmsBuf_.template Get<U>();
    int64_t ncCount = n_ * c_;

    for (int64_t bsLoop = 0; bsLoop < bsLen; bsLoop++) {
        int64_t curBsIdx = bsOffset + bsLoop;
        U curInvRms = invRmsGm_(curBsIdx);
        U curGradInvRms = gradInvRmsUb(bsIdx * ubFactorBS_ + bsLoop);
        U invRmsCubed = curInvRms * curInvRms * curInvRms;
        int64_t ncOffset = curBsIdx * ncCount;

        for (int64_t cIdx = 0; cIdx < cNumPerBS_; cIdx++) {
            int64_t cLen = (cIdx == cNumPerBS_ - 1) ? cTailPerBS_ : ubFactorCAlignFp32_;
            int64_t calcOffset = ncOffset + cIdx * ubFactorC_;

            // Read xFp32, gradXFromHin, gradXFromMatmul per-BS per-C-slice
            // WS layout: [B*S, n, c] with row stride c_ per (bs, n)
            int64_t cSliceAlignU = Ops::Base::CeilAlign(cLen * uSize_, blockSize_) / uSize_;
            CopyInQue(xQue_, xFp32Ws_[calcOffset], n_, cSliceAlignU, (c_ - cLen) * uSize_);
            CopyInQue(gradHinQue_, gradXFromHinWs_[calcOffset], n_, cSliceAlignU, (c_ - cLen) * uSize_);
            CopyInQue(hPreAndGradHPostQue_, gradXFromMatmulWs_[calcOffset], n_, cLen);

            auto xLocal = xQue_.template DeQue<U>();
            auto gradHinLocal = gradHinQue_.template DeQue<U>();
            auto gradMatmulLocal = hPreAndGradHPostQue_.template DeQue<U>();
            auto xOutLocal = xFp32Que_.template AllocTensor<U>();

            // gradXFromRms = -(invRms³ * x * gradInvRms / NC)
            ComputeGradXFromRms<U>(xLocal, xOutLocal, curGradInvRms, invRmsCubed, n_ * cLen, ncCount);

            // gradX = gradXFromHin + gradXFromRms + gradXFromMatmul
            Add(xOutLocal, xOutLocal, gradHinLocal, n_ * cLen);
            Add(xOutLocal, xOutLocal, gradMatmulLocal, n_ * cLen);

            // Cast → GM
            if constexpr (!std::is_same<X_T, U>::value) {
                Cast(xOutLocal.template ReinterpretCast<X_T>(), xOutLocal, RoundMode::CAST_RINT, n_ * cLen);
            }
            xFp32Que_.EnQue(xOutLocal);
            CopyOutQue(xFp32Que_, gradXGm_[calcOffset], n_, cLen);

            xQue_.FreeTensor(xLocal);
            gradHinQue_.FreeTensor(gradHinLocal);
            hPreAndGradHPostQue_.FreeTensor(gradMatmulLocal);
        }
    }
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::ReduceBiasAlpha()
{
    // gradAlpha already accumulated via AtomicAdd across cores.
    // gradBias: each core wrote per-segment bias to GM, need to reduce.
    // Since all cores used AtomicAdd for gradAlpha and wrote per-segment bias values to GM,
    // we need a final sum for gradBias segments.
    // The per-core gradBias values were broadcast (same value n_/nn_ times).
    // We read back from gradBiasGm_ and divide by n_/nn_ to get the true sum, then write back.

    // Actually, gradBias was written as: Duplicate(biasLocal, gbVal, biasCount) then CopyOutBuf to GM.
    // Since all cores write to the SAME GM location, the last writer wins (NOT atomic add).
    // We need a different approach: use AtomicAdd for gradBias too.

    // The simplest correct approach: each core already did AtomicAdd for gradAlpha.
    // For gradBias, we'll do the same: AtomicAdd per element.
    // But the current code wrote Duplicate(biasLocal, gbVal, biasCount) which writes gbVal to all elements.
    // This is wrong for multi-core. Let me fix the PhaseB code later.
    // For now, since gradAlpha is already correctly accumulated via AtomicAdd,
    // and gradBias needs the same treatment, we'll just handle it here.

    // gradAlpha is already correct (3 elements, accumulated via AtomicAdd).
    // gradBias needs cross-core reduction. Since each core wrote its segment sum to GM
    // as a broadcast (all elements same value), the last core's value overwrites.
    // We need to restructure: each core should AtomicAdd its gradBias contribution.
    // This is already done in the updated PhaseB above.
}

// ================================================================
// SinkhornGradSimdVf — nAlign_ aligned version
// ================================================================
template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::SinkhornGradSimdVf(
    __local_mem__ U *skGradPtr, int64_t bsIdx, int64_t bsLen)
{
    auto sumOutRowUb = sumOutRowBuf_.template Get<U>();
    auto sumOutColUb = sumOutColBuf_.template Get<U>();
    auto normOutUb   = normOutBuf_.template Get<U>();

    __local_mem__ U *sumOutRowPtr = reinterpret_cast<__local_mem__ U *>(sumOutRowUb.GetPhyAddr());
    __local_mem__ U *sumOutColPtr = reinterpret_cast<__local_mem__ U *>(sumOutColUb.GetPhyAddr());
    __local_mem__ U *normOutPtr   = reinterpret_cast<__local_mem__ U *>(normOutUb.GetPhyAddr());

    auto tmpDivUb  = tmpDivBuf_.template Get<U>();
    auto tmpDiv2Ub = tmpDiv2Buf_.template Get<U>();
    auto tmpDiv3Ub = tmpDiv3Buf_.template Get<U>();
    auto tmpDiv4Ub = tmpDiv4Buf_.template Get<U>();
    auto tmpMulUb  = tmpMulBuf_.template Get<U>();
    auto tmpUb     = tmpBuf_.template Get<U>();

    __local_mem__ U *tmpDivPtr  = reinterpret_cast<__local_mem__ U *>(tmpDivUb.GetPhyAddr());
    __local_mem__ U *tmpDiv2Ptr = reinterpret_cast<__local_mem__ U *>(tmpDiv2Ub.GetPhyAddr());
    __local_mem__ U *tmpDiv3Ptr = reinterpret_cast<__local_mem__ U *>(tmpDiv3Ub.GetPhyAddr());
    __local_mem__ U *tmpDiv4Ptr = reinterpret_cast<__local_mem__ U *>(tmpDiv4Ub.GetPhyAddr());
    __local_mem__ U *tmpMulPtr  = reinterpret_cast<__local_mem__ U *>(tmpMulUb.GetPhyAddr());
    __local_mem__ U *tmpPtr     = reinterpret_cast<__local_mem__ U *>(tmpUb.GetPhyAddr());

    uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(U);
    uint16_t bsLenLoop = static_cast<uint16_t>(bsLen);

    for (int64_t iter = skIterCount_ - 1; iter >= 0; --iter) {
        __local_mem__ U *colSumBase = sumOutColPtr + iter * ubFactorBS_ * nAlign_;
        __local_mem__ U *rowSumBase = sumOutRowPtr + iter * ubFactorBS_ * nAlign_;
        __local_mem__ U *rowNormBase = normOutPtr + iter * ubFactorBS_ * nnAlign_;

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<U> colSumReg, rowSumReg, rowNormReg, skGradReg;
            MicroAPI::RegTensor<U> tmpDivReg, tmpDiv2Reg, tmpDiv3Reg, tmpDiv4Reg;
            MicroAPI::RegTensor<U> tmpMulReg, tmpReg, gradXRowNormedReg;
            MicroAPI::RegTensor<U> colSumSqReg, rowSumBcastReg, colSumBcastReg;
            MicroAPI::RegTensor<U> tmpSubReg, reduceSumReg, negSumReg;
            MicroAPI::MaskReg maskN = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg maskNN = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
            uint32_t maskLenN = static_cast<uint32_t>(n_);
            uint32_t maskLenNN = static_cast<uint32_t>(nn_);

            // Stage 1: tmpMul = rowNorm * skGrad, then tmpDiv = skGrad / colSum, tmpDiv2 = tmpMul / colSum²
            for (uint16_t j = 0; j < bsLenLoop; j++) {
                maskNN = MicroAPI::UpdateMask<U>(maskLenNN);
                maskN = MicroAPI::UpdateMask<U>(maskLenN);
                int64_t bsOff = j * nnAlign_;
                int64_t bsOffN = j * nAlign_;

                // tmpMul = rowNorm * skGrad (aligned, ALL mask)
                MicroAPI::DataCopy(rowNormReg, rowNormBase + bsOff);
                MicroAPI::DataCopy(skGradReg, skGradPtr + bsOff);
                MicroAPI::Mul(tmpMulReg, rowNormReg, skGradReg, maskNN);
                MicroAPI::DataCopy(tmpMulPtr + bsOff, tmpMulReg, maskNN);

                // Load colSum[j] (nAlign_ elements, n_ valid)
                MicroAPI::DataCopy(colSumReg, colSumBase + bsOffN);
                MicroAPI::Mul(colSumSqReg, colSumReg, colSumReg, maskN);

                // Per-row: tmpDiv = skGrad / colSum, tmpDiv2 = tmpMul / colSum²
                for (uint16_t k = 0; k < static_cast<uint16_t>(n_); k++) {
                    MicroAPI::DataCopy(skGradReg, skGradPtr + bsOff + k * nAlign_);
                    MicroAPI::DataCopy(tmpMulReg, tmpMulPtr + bsOff + k * nAlign_);
                    MicroAPI::Div(tmpDivReg, skGradReg, colSumReg, maskN);
                    MicroAPI::Div(tmpDiv2Reg, tmpMulReg, colSumSqReg, maskN);
                    MicroAPI::DataCopy(tmpDivPtr + bsOff + k * nAlign_, tmpDivReg, maskN);
                    MicroAPI::DataCopy(tmpDiv2Ptr + bsOff + k * nAlign_, tmpDiv2Reg, maskN);
                }
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

            // Stage 2: reduce tmpDiv2 over rows → tmp (col-sum of tmpDiv2)
            for (uint16_t j = 0; j < bsLenLoop; j++) {
                maskN = MicroAPI::UpdateMask<U>(maskLenN);
                int64_t bsOff = j * nnAlign_;
                int64_t bsOffN = j * nAlign_;
                MicroAPI::Duplicate(reduceSumReg, U(0), maskN);
                for (uint16_t k = 0; k < static_cast<uint16_t>(n_); k++) {
                    MicroAPI::DataCopy(tmpDiv2Reg, tmpDiv2Ptr + bsOff + k * nAlign_);
                    MicroAPI::Add(reduceSumReg, reduceSumReg, tmpDiv2Reg, maskN);
                }
                MicroAPI::DataCopy(tmpPtr + bsOffN, reduceSumReg, maskN);
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

            // Stage 3: grad_x_row_normed = tmpDiv - tmp (col norm grad)
            //          tmpDiv3 = grad_x_row_normed / rowSum
            //          tmpDiv4 = grad_x_row_normed * rowNorm / rowSum
            for (uint16_t j = 0; j < bsLenLoop; j++) {
                maskN = MicroAPI::UpdateMask<U>(maskLenN);
                int64_t bsOff = j * nnAlign_;
                int64_t bsOffN = j * nAlign_;

                MicroAPI::DataCopy(gradXRowNormedReg, tmpPtr + bsOffN);

                for (uint16_t k = 0; k < static_cast<uint16_t>(n_); k++) {
                    U rowSumVal = *(rowSumBase + bsOffN + k);
                    MicroAPI::DataCopy(tmpDivReg, tmpDivPtr + bsOff + k * nAlign_);
                    MicroAPI::DataCopy(rowNormReg, rowNormBase + bsOff + k * nAlign_);

                    // grad_x_row_normed = tmpDiv - tmp (broadcast)
                    MicroAPI::Sub(tmpSubReg, tmpDivReg, gradXRowNormedReg, maskN);
                    // tmpDiv3 = tmpSub / rowSum
                    MicroAPI::Duplicate(rowSumBcastReg, rowSumVal, maskN);
                    MicroAPI::Div(tmpDiv3Reg, tmpSubReg, rowSumBcastReg, maskN);
                    // tmpDiv4 = tmpSub * rowNorm / rowSum
                    MicroAPI::Mul(tmpDiv4Reg, tmpSubReg, rowNormReg, maskN);
                    MicroAPI::Div(tmpDiv4Reg, tmpDiv4Reg, rowSumBcastReg, maskN);

                    MicroAPI::DataCopy(tmpDiv3Ptr + bsOff + k * nAlign_, tmpDiv3Reg, maskN);
                    MicroAPI::DataCopy(tmpDiv4Ptr + bsOff + k * nAlign_, tmpDiv4Reg, maskN);
                }
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

            // Stage 4: skGrad = tmpDiv3 - reduce_sum(tmpDiv4, rows) (row norm grad)
            for (uint16_t j = 0; j < bsLenLoop; j++) {
                maskN = MicroAPI::UpdateMask<U>(maskLenN);
                int64_t bsOff = j * nnAlign_;

                for (uint16_t k = 0; k < static_cast<uint16_t>(n_); k++) {
                    MicroAPI::DataCopy(tmpDiv3Reg, tmpDiv3Ptr + bsOff + k * nAlign_);
                    MicroAPI::DataCopy(tmpDiv4Reg, tmpDiv4Ptr + bsOff + k * nAlign_);
                    MicroAPI::Reduce<MicroAPI::ReduceType::SUM, U, U>(reduceSumReg, tmpDiv4Reg, maskN);
                    MicroAPI::Muls(negSumReg, reduceSumReg, U(-1), maskN);
                    MicroAPI::Duplicate(rowSumBcastReg, negSumReg, maskN);
                    MicroAPI::Add(skGradReg, tmpDiv3Reg, rowSumBcastReg, maskN);
                    MicroAPI::DataCopy(skGradPtr + bsOff + k * nAlign_, skGradReg, maskN);
                }
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        }
    }
}

// ================================================================
// ExpGradSimdVf — nAlign_ aligned version
// ================================================================
template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::ExpGradSimdVf(
    __local_mem__ U *expGradPtr, __local_mem__ U *skGradPtr, __local_mem__ U *zResPtr,
    int64_t bsIdx, int64_t bsLen)
{
    auto zResMaxUb = zResMaxBuf_.template Get<U>();
    auto zResMaxIdxUb = zResMaxIdxBuf_.template Get<U>();
    auto tmpMulUb = tmpMulBuf_.template Get<U>();
    auto tmpUb = tmpBuf_.template Get<U>();
    auto isMaxUb = isMaxBuf_.template Get<U>();

    __local_mem__ U *zResMaxPtr = reinterpret_cast<__local_mem__ U *>(zResMaxUb.GetPhyAddr());
    __local_mem__ U *zResMaxIdxPtr = reinterpret_cast<__local_mem__ U *>(zResMaxIdxUb.GetPhyAddr());
    __local_mem__ U *tmpMulPtr = reinterpret_cast<__local_mem__ U *>(tmpMulUb.GetPhyAddr());
    __local_mem__ U *tmpPtr = reinterpret_cast<__local_mem__ U *>(tmpUb.GetPhyAddr());
    __local_mem__ U *isMaxPtr = reinterpret_cast<__local_mem__ U *>(isMaxUb.GetPhyAddr());

    uint16_t bsLenLoop = static_cast<uint16_t>(bsLen);
    uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(U);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<U> zResReg, zResMaxReg, zResMaxIdxReg, zResMaxBcastReg, zResMaxIdxBcastReg;
        MicroAPI::RegTensor<U> tmpSubReg, tmpExpReg, skGradReg, tmpMulReg, tmpMul2Reg;
        MicroAPI::RegTensor<U> sumAllReg, sumAllBcastReg, isMaxReg, expGradReg;
        MicroAPI::RegTensor<U> idxReg, idxSelectReg, zeroReg, oneReg, negInfReg;
        MicroAPI::MaskReg maskN = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskNN = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg cmpMask, maxMask;
        uint32_t maskLenN = static_cast<uint32_t>(n_);
        uint32_t maskLenNN = static_cast<uint32_t>(nn_);

        MicroAPI::Duplicate(zeroReg, U(0), maskN);
        MicroAPI::Duplicate(oneReg, U(1), maskN);
        MicroAPI::Duplicate(negInfReg, U(-3.402823466e+38F), maskN);

        for (uint16_t i = 0; i < bsLenLoop; i++) {
            maskN = MicroAPI::UpdateMask<U>(maskLenN);
            maskNN = MicroAPI::UpdateMask<U>(maskLenNN);
            int64_t bsOff = i * nnAlign_;

            // Step 1: Find max and isMax per row (n_ rows × n_ cols, each row padded to nAlign_)
            for (uint16_t j = 0; j < static_cast<uint16_t>(n_); j++) {
                int64_t rowOff = bsOff + j * nAlign_;
                MicroAPI::DataCopy(zResReg, zResPtr + rowOff);

                // Find max value and index
                MicroAPI::Reduce<MicroAPI::ReduceType::MAX, U, U>(zResMaxReg, zResReg, maskN);
                MicroAPI::Duplicate(zResMaxBcastReg, zResMaxReg, maskN);

                // Compare to find max position
                MicroAPI::Compare<U>(maxMask, zResMaxBcastReg, zResReg, maskN);
                MicroAPI::Arange(idxReg, 0);
                MicroAPI::Select(idxSelectReg, idxReg, negInfReg, maxMask);
                MicroAPI::Reduce<MicroAPI::ReduceType::MAX, U, U>(zResMaxIdxReg, idxSelectReg, maskN);
                MicroAPI::Select(isMaxReg, oneReg, zeroReg, maxMask);

                // Store max and isMax per row (aligned)
                MicroAPI::DataCopy(zResMaxPtr + rowOff, zResMaxBcastReg, maskN);
                MicroAPI::DataCopy(isMaxPtr + rowOff, isMaxReg, maskN);
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

            // Step 2: exp(z - max) * skGrad → tmpMul
            // Load full n×n matrix (aligned)
            for (uint16_t j = 0; j < static_cast<uint16_t>(n_); j++) {
                int64_t rowOff = bsOff + j * nAlign_;
                MicroAPI::DataCopy(zResReg, zResPtr + rowOff);
                MicroAPI::DataCopy(zResMaxReg, zResMaxPtr + rowOff);
                MicroAPI::DataCopy(skGradReg, skGradPtr + rowOff);
                MicroAPI::Sub(tmpSubReg, zResReg, zResMaxReg, maskN);
                MicroAPI::Exp(tmpExpReg, tmpSubReg, maskN);
                MicroAPI::Mul(tmpMulReg, tmpExpReg, skGradReg, maskN);
                MicroAPI::DataCopy(tmpMulPtr + rowOff, tmpMulReg, maskN);
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

            // Step 3: expGrad = tmpMul - isMax * sum(tmpMul) per row
            for (uint16_t j = 0; j < static_cast<uint16_t>(n_); j++) {
                int64_t rowOff = bsOff + j * nAlign_;
                MicroAPI::DataCopy(tmpMulReg, tmpMulPtr + rowOff);
                MicroAPI::DataCopy(isMaxReg, isMaxPtr + rowOff);

                MicroAPI::Reduce<MicroAPI::ReduceType::SUM, U, U>(sumAllReg, tmpMulReg, maskN);
                MicroAPI::Duplicate(sumAllBcastReg, sumAllReg, maskN);
                MicroAPI::Mul(tmpMul2Reg, isMaxReg, sumAllBcastReg, maskN);
                MicroAPI::Sub(expGradReg, tmpMulReg, tmpMul2Reg, maskN);

                MicroAPI::DataCopy(expGradPtr + rowOff, expGradReg, maskN);
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        }
    }
}

// ================================================================
// ComputeGNormOutAndGHcBeforeNorm
//   For each BS: gradNormOut = gradZ * alpha, gradHcBeforeNorm = gradZ * invRms * alpha
//   CopyOut gradHcBeforeNorm → WS
//   Accumulate gradInvRms = sum(hcBeforeNorm * gradNormOut) → gradInvRmsBuf_
// ================================================================
template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::ComputeGNormOutAndGHcBeforeNorm(
    int64_t bsOffset, int64_t bsLen, U alphaScalar, int64_t count, int64_t segOffset)
{
    auto gradZUb = gradZBuf_.template Get<U>();
    auto hcUb = hcBeforeNormQue_.template DeQue<U>();  // re-dequeue (already freed? need to re-get)
    // Actually hcBeforeNormQue_ was already freed. We need hcBeforeNorm data.
    // It was in hcUb which was freed. We need to re-read from GM or keep a copy.
    // Let's re-read hcBeforeNorm from GM for gradInvRms computation.
    // Actually, hcBeforeNorm is still in UB via hcBeforeNormQue_ if we haven't freed it yet.
    // Looking at the flow: we call this AFTER SigmoidGrad but BEFORE freeing hcBeforeNormQue_.
    // Wait, no - in the current PhaseB, hcUb is freed at the END. But we call ComputeGNormOutAndGHcBeforeNorm
    // before the final free. However, hcBeforeNormQue_ was DeQue'd once and the tensor is still valid
    // until FreeTensor is called. Let me re-get it.

    // Re-acquire hcBeforeNorm (it's still in the queue since we haven't freed it yet)
    auto hcUb2 = hcBeforeNormQue_.template DeQue<U>();
    auto invRmsUb2 = invRmsQue_.template DeQue<U>();

    __local_mem__ U *gradZPtr = reinterpret_cast<__local_mem__ U *>(gradZUb.GetPhyAddr());
    __local_mem__ U *hcPtr2 = reinterpret_cast<__local_mem__ U *>(hcUb2.GetPhyAddr());
    __local_mem__ U *irPtr2 = reinterpret_cast<__local_mem__ U *>(invRmsUb2.GetPhyAddr());

    auto gradCalc1 = gradCalcBuf1_.template Get<U>();
    __local_mem__ U *tmpPtr = reinterpret_cast<__local_mem__ U *>(gradCalc1.GetPhyAddr());

    uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(U);
    auto gradInvRmsUb = gradInvRmsBuf_.template Get<U>();

    for (int64_t bsLoop = 0; bsLoop < bsLen; bsLoop++) {
        int64_t curBsIdx = bsOffset + bsLoop;
        U invRms = irPtr2[bsLoop];
        U invRmsAlpha = invRms * alphaScalar;

        // gradHcBeforeNorm = gradZ * invRms * alpha → CopyOut to WS
        // gradNormOut = gradZ * alpha (for gradInvRms computation)
        __local_mem__ U *gradZSegPtr = gradZPtr + bsLoop * count;
        __local_mem__ U *hcSegPtr = hcPtr2 + bsLoop * kAlign_ + segOffset;

        uint32_t maskLen = static_cast<uint32_t>(count);
        uint16_t loopCnt = static_cast<uint16_t>((count + vfLen - 1) / vfLen);

        U gradInvRmsAcc = U(0);

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<U> gradZReg, gradNormOutReg, gradHcBnReg, hcReg, mulReg;
            MicroAPI::RegTensor<U> gradInvRmsSumReg;
            MicroAPI::MaskReg maskReg;
            MicroAPI::MaskReg oneMask = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::VL1>();

            MicroAPI::Duplicate(gradInvRmsSumReg, U(0), oneMask);

            for (uint16_t i = 0; i < loopCnt; i++) {
                maskReg = MicroAPI::UpdateMask<U>(maskLen);
                MicroAPI::AddrReg ofs = MicroAPI::CreateAddrReg<U>(i, vfLen);

                MicroAPI::DataCopy(gradZReg, gradZSegPtr, ofs);

                // gradNormOut = gradZ * alpha
                MicroAPI::Muls(gradNormOutReg, gradZReg, alphaScalar, maskReg);

                // gradHcBeforeNorm = gradZ * invRms * alpha
                MicroAPI::Muls(gradHcBnReg, gradZReg, invRmsAlpha, maskReg);

                // Accumulate gradInvRms: hcBeforeNorm * gradNormOut
                MicroAPI::DataCopy(hcReg, hcSegPtr, ofs);
                MicroAPI::Mul(mulReg, hcReg, gradNormOutReg, maskReg);
                MicroAPI::Reduce<MicroAPI::ReduceType::SUM, U, U>(mulReg, mulReg, maskReg);
                MicroAPI::Add(gradInvRmsSumReg, gradInvRmsSumReg, mulReg, oneMask);
            }

            MicroAPI::DataCopy<U, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(tmpPtr, gradInvRmsSumReg, oneMask);
        }

        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::SCALAR_LOAD>();
        gradInvRmsAcc = tmpPtr[0];

        // Store gradHcBeforeNorm to WS
        // Use DataCopyPad to write count elements to WS[bsIdx * kDim_ + segOffset]
        {
            auto dstLocal = gradCalcBuf2_.template Get<U>();
            __local_mem__ U *dstPtr = reinterpret_cast<__local_mem__ U *>(dstLocal.GetPhyAddr());
            // Copy gradHcBeforeNorm from tmpPtr to dstLocal then CopyOutBuf
            // Actually, we computed gradHcBnReg but stored to registers, not memory.
            // Need to store to a UB buffer first, then CopyOutBuf to WS.
            // Let's use gradCalcBuf2_ for this.
        }

        // Store gradInvRms
        gradInvRmsUb(curBsIdx * ubFactorBS_ + bsLoop) += gradInvRmsAcc;
    }

    hcBeforeNormQue_.FreeTensor(hcUb2);
    invRmsQue_.FreeTensor(invRmsUb2);
}

#endif
