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

    n_           = td->n;
    c_           = td->c;
    nn_          = n_ * n_;
    kDim_        = nn_ + 2 * n_;
    ubFactorBS_  = td->ubFactorBS;
    ubFactorC_   = td->ubFactorC;
    usedCoreNum_ = td->usedCoreNum;
    aicCoreNum_  = td->aicCoreNum;
    skIterCount_ = td->skIterCount;
    eps_         = td->eps;
    blockIdx_    = GetBlockIdx();
    blockSize_   = Ops::Base::GetUbBlockSize();
    bsPerCore_   = (blockIdx_ == usedCoreNum_ - 1) ? td->blockTailBS : td->blockFactorBS;

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

        int64_t totalBs = td->b * td->s;
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
        mm1M_ = td->b * td->s;
        mm1N_ = n_ * c_;
        mm2K_ = td->b * td->s;
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
    int64_t bsOffset = blockIdx_ * td_->blockFactorBS + bsIdx * ubFactorBS_;
    int64_t hPreOffset = bsOffset * n_;
    int64_t xOffset = bsOffset * n_ * c_;
    int64_t gradHinOffset = bsOffset * c_;

    // gradHPre 清零
    auto gradHPreLocal = gradHPreBuf_.template Get<U>();
    Duplicate(gradHPreLocal, float(0), bsLen * n_);

    // 搬入 hPre
    CopyInQue(hPreAndGradHPostQue_, hPreGm_[hPreOffset], bsLen, n_);
    auto hPreLocal = hPreAndGradHPostQue_.template DeQue<U>();

    __local_mem__ U *hPreAddr    = reinterpret_cast<__local_mem__ U *>(hPreLocal.GetPhyAddr());
    __local_mem__ U *gradHPreAddr = reinterpret_cast<__local_mem__ U *>(gradHPreLocal.GetPhyAddr());

    // C 维循环
    for (int64_t cIdx = 0; cIdx < cNumPerBS_; cIdx++) {
        int64_t cLen = (cIdx == cNumPerBS_ - 1) ? cTailPerBS_ : ubFactorCAlignFp32_;
        int64_t cLenXAlign = Ops::Base::CeilAlign(cLen * xSize_, blockSize_) / xSize_;
        int64_t cLenUAlign = Ops::Base::CeilAlign(cLen * uSize_, blockSize_) / uSize_;
        int64_t cLenGhAlign = Ops::Base::CeilAlign(cLen * ghSize_, blockSize_) / ghSize_;

        int64_t xStartOft = xOffset + cIdx * ubFactorC_;
        int64_t gradHinStartOft = gradHinOffset + cIdx * ubFactorC_;

        // 搬入 x[bs, n, cSlice] 和 gradHin[bs, cSlice]
        CopyInQue(xQue_, xGm_[xStartOft], bsLen * n_, cLenXAlign, (c_ - cLen) * xSize_);
        CopyInQue(gradHinQue_, gradHinGm_[gradHinStartOft], bsLen, cLenGhAlign, (c_ - cLen) * ghSize_);
        auto xLocal = xQue_.template DeQue<X_T>();
        auto gradHinLocal = gradHinQue_.template DeQue<GRADHIN_T>();

        auto xFp32Local = xFp32Que_.template AllocTensor<U>();
        auto gradXFromHinLocal = gradXFromHinQue_.template AllocTensor<U>();

        __local_mem__ X_T *xAddr = reinterpret_cast<__local_mem__ X_T *>(xLocal.GetPhyAddr());
        __local_mem__ U *xCastAddr = reinterpret_cast<__local_mem__ U *>(xFp32Local.GetPhyAddr());
        __local_mem__ GRADHIN_T *gradHinAddr = reinterpret_cast<__local_mem__ GRADHIN_T *>(gradHinLocal.GetPhyAddr());
        __local_mem__ U *gradXFromHinAddr = reinterpret_cast<__local_mem__ U *>(gradXFromHinLocal.GetPhyAddr());

        uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(U);
        uint16_t loopCnt = (cLen + vfLen - 1) / vfLen;

        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<GRADHIN_T> gradHinReg;
            AscendC::MicroAPI::RegTensor<U> gradHinCastReg;
            AscendC::MicroAPI::RegTensor<X_T> xReg;
            AscendC::MicroAPI::RegTensor<U> xCastReg;
            AscendC::MicroAPI::RegTensor<U> gradHPreReg;
            AscendC::MicroAPI::RegTensor<U> gradHPreRegSumReg;
            AscendC::MicroAPI::RegTensor<U> gradXFromHinReg;
            AscendC::MicroAPI::MaskReg valueMaskReg;
            AscendC::MicroAPI::MaskReg OneMaskReg = AscendC::MicroAPI::CreateMask<U, AscendC::MicroAPI::MaskPattern::VL1>();

            for (uint16_t i = 0; i < static_cast<uint16_t>(bsLen); i++) {
                auto gradHPreStart = gradHPreAddr + i * n_;
                uint32_t maskLen = static_cast<uint32_t>(cLen);
                for (uint16_t j = 0; j < loopCnt; j++) {
                    auto gradHinStart = gradHinAddr + i * cLenGhAlign + j * vfLen;
                    valueMaskReg = AscendC::MicroAPI::UpdateMask<U>(maskLen);
                    AscendC::MicroAPI::DataCopy<GRADHIN_T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(gradHinReg, gradHinStart);
                    AscendC::MicroAPI::Cast<U, GRADHIN_T, castTrait16ToFloat>(gradHinCastReg, gradHinReg, valueMaskReg);

                    for (uint16_t k = 0; k < static_cast<uint16_t>(n_); k++) {
                        AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                                       AscendC::MicroAPI::MemType::SCALAR_LOAD>();
                        U hPre = *(hPreAddr + i * n_ + k);
                        U gradHPre = *(gradHPreStart + k);
                        auto xCastStart = xCastAddr + (i * n_ + k) * cLenUAlign + j * vfLen;
                        auto xAddrStart = xAddr + (i * n_ + k) * cLenXAlign + j * vfLen;
                        auto gradXFromHinStart = gradXFromHinAddr + (i * n_ + k) * cLenUAlign + j * vfLen;

                        AscendC::MicroAPI::DataCopy<X_T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(xReg, xAddrStart);
                        AscendC::MicroAPI::Cast<U, X_T, castTrait16ToFloat>(xCastReg, xReg, valueMaskReg);
                        AscendC::MicroAPI::Muls(gradXFromHinReg, gradHinCastReg, hPre, valueMaskReg);
                        AscendC::MicroAPI::Mul(gradHPreReg, xCastReg, gradHinCastReg, valueMaskReg);
                        AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::SUM, U, U>(gradHPreReg, gradHPreReg, valueMaskReg);
                        AscendC::MicroAPI::Adds(gradHPreRegSumReg, gradHPreReg, gradHPre, OneMaskReg);

                        AscendC::MicroAPI::DataCopy(gradXFromHinStart, gradXFromHinReg, valueMaskReg);
                        AscendC::MicroAPI::DataCopy(xCastStart, xCastReg, valueMaskReg);
                        AscendC::MicroAPI::Store(gradHPreStart + k, gradHPreRegSumReg, 1);
                    }
                }
            }
        }

        xFp32Que_.EnQue(xFp32Local);
        gradXFromHinQue_.EnQue(gradXFromHinLocal);
        xQue_.FreeTensor(xLocal);
        gradHinQue_.FreeTensor(gradHinLocal);

        // 搬出 → WS
        xFp32Local = xFp32Que_.template DeQue<U>();
        gradXFromHinLocal = gradXFromHinQue_.template DeQue<U>();
        CopyOutQue(xFp32Que_, xFp32Ws_[xStartOft], bsLen * n_, cLenUAlign, 0, (c_ - cLen) * uSize_);
        CopyOutQue(gradXFromHinQue_, gradXFromHinWs_[xStartOft], bsLen * n_, cLenUAlign, 0, (c_ - cLen) * uSize_);
    }

    hPreAndGradHPostQue_.FreeTensor(hPreLocal);
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

    // --- z_pre/post/res = hcBeforeNorm * invRms + bias, normOut = sigmoid(z) * alpha ---
    ComputeZAndForwardPart<U>(zPtr, noPtr, biasPtr, bsLen, n_, cAlignFp32_, 256, alphaPre_);
    ComputeZAndForwardPart<U>(zPtr + bsLen * n_, noPtr + bsLen * n_, biasPtr + n_, bsLen, n_, cAlignFp32_, 256, alphaPost_);
    ComputeZAndForwardPart<U>(zPtr + bsLen * 2 * n_, noPtr + bsLen * 2 * n_, biasPtr + 2 * n_, bsLen, nn_, cAlignFp32_, 256, alphaRes_);

    // --- SigmoidGrad: gradZ_pre = SigmoidGrad(z_pre, gradHPre, isPre=true) ---
    SigmoidGrad<U, true>(ghPrePtr, zPtr, gradZPtr, bsLen, n_, 256, eps_);

    // --- SigmoidGrad: gradZ_post = 2 * SigmoidGrad(z_post, gradHPost, isPre=false) ---
    // 先搬入 gradHPost
    CopyInQue(hPreAndGradHPostQue_, gradHPostGm_[bsOffset * n_], bsLen, n_);
    auto gradHPostUb = hPreAndGradHPostQue_.template DeQue<U>();
    __local_mem__ U *ghPostPtr = reinterpret_cast<__local_mem__ U *>(gradHPostUb.GetPhyAddr());
    SigmoidGrad<U, false>(ghPostPtr, zPtr + bsLen * n_, gradZPtr + bsLen * n_, bsLen, n_, 256, eps_);
    hPreAndGradHPostQue_.FreeTensor(gradHPostUb);

    // --- SinkhornGrad + ExpGrad: gradZ_res ---
    // TODO: 搬入 gradHRes + sumOut + normOut, 调用 SinkhornGradSimdVf + ExpGradSimdVf
    // 参考 base.h SinkhornGradSimdVf + ExpGradSimdVf 实现

    // --- gradAlpha + gradBias 累加 ---
    // TODO: ComputeGradAlpha + ComputeGradBias, 累加到 gradAlphaAndBiasQue

    // --- gradNormOut = gradZ * alpha → gradHcBeforeNorm = gradNormOut * invRms → WS ---
    // TODO: ComputeGradNormOutOrGradHcBeforeNorm + CopyOut → WS

    // --- gradInvRms = (hcBeforeNorm * gradNormOut).reduce_sum(K) → gradInvRmsBuf_ ---
    // TODO: ComputeGradInvRms → gradInvRmsBuf_[bsIdx]

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
    int64_t ncOffset = bsOffset * n_ * c_;

    auto gradInvRmsUb = gradInvRmsBuf_.template Get<U>();

    for (int64_t cIdx = 0; cIdx < cNumPerBS_; cIdx++) {
        int64_t cLen = (cIdx == cNumPerBS_ - 1) ? cTailPerBS_ : ubFactorCAlignFp32_;
        int64_t calcOffset = ncOffset + cIdx * ubFactorC_;

        // 搬入 xFp32, gradXFromHin, gradXFromMatmul from WS
        CopyInQue(xQue_, xFp32Ws_[calcOffset], 1, cLen * n_);
        CopyInQue(gradHinQue_, gradXFromHinWs_[calcOffset], 1, cLen * n_);
        CopyInQue(hPreAndGradHPostQue_, gradXFromMatmulWs_[calcOffset], 1, cLen * n_);

        auto xLocal = xQue_.template DeQue<U>();
        auto gradHinLocal = gradHinQue_.template DeQue<U>();
        auto gradMatmulLocal = hPreAndGradHPostQue_.template DeQue<U>();
        auto xOutLocal = xFp32Que_.template AllocTensor<U>();

        // gradXFromRms = -(invRms³ * x * gradInvRms / NC)
        U curInvRms = invRmsGm_(bsOffset);
        U curGradInvRms = gradInvRmsUb(bsOffset - blockIdx_ * td_->blockFactorBS);
        U invRmsCubed = curInvRms * curInvRms * curInvRms;
        int64_t ncCount = n_ * c_;
        ComputeGradXFromRms<U>(xLocal, xOutLocal, curGradInvRms, invRmsCubed, cLen * n_, ncCount);

        // gradX = gradXFromHin + gradXFromRms + gradXFromMatmul
        Add(xOutLocal, xOutLocal, gradHinLocal, cLen * n_);
        Add(xOutLocal, xOutLocal, gradMatmulLocal, cLen * n_);

        // Cast → GM
        if constexpr (!std::is_same<X_T, U>::value) {
            Cast(xOutLocal.template ReinterpretCast<X_T>(), xOutLocal, RoundMode::CAST_RINT, cLen * n_);
        }
        xFp32Que_.EnQue(xOutLocal);
        auto xOutCast = xFp32Que_.template DeQue<X_T>();
        CopyOutQue(xFp32Que_, gradXGm_[calcOffset], 1, cLen * n_);

        xQue_.FreeTensor(xLocal);
        gradHinQue_.FreeTensor(gradHinLocal);
        hPreAndGradHPostQue_.FreeTensor(gradMatmulLocal);
    }
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::ReduceBiasAlpha()
{
    // TODO: 跨核归约 gradBias + gradAlpha → GM
    // 参考 base.h AddAlphaAndBias 实现
    // 需要从各核的 WS 中读取累加结果，求和后写 GM
}

#endif
