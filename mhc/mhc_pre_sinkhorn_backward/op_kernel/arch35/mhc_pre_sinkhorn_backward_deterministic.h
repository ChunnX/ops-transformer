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
 *       for idxC in ⌈c / ubFactorC⌉:
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
 *     for idxC in ⌈c / ubFactorC⌉:
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
    __aicore__ inline void PhaseA_CLoop(int64_t idxBs, int64_t lenBs);
    __aicore__ inline void PhaseB_KDim(int64_t idxBs, int64_t lenBs);
    __aicore__ inline void PhaseD_MatMul();
    __aicore__ inline void PhaseE_CLoop(int64_t idxBs, int64_t lenBs);
    __aicore__ inline void ReduceBiasAlpha();
    __aicore__ inline void GetAlphaAndBias();
    __aicore__ inline void SinkhornGradSimdVf(__local_mem__ U *skGradPtr, int64_t idxBs, int64_t lenBs);
    __aicore__ inline void ExpGradSimdVf(__local_mem__ U *expGradPtr, __local_mem__ U *skGradPtr,
                                          __local_mem__ U *zResPtr, int64_t idxBs, int64_t lenBs);
    __aicore__ inline void ComputeGNormOutAndGHcBeforeNorm(int64_t ofsBs, int64_t lenBs,
                                                            U alphaScalar, int64_t nRows,
                                                            int64_t ubSegOffset, int64_t wsSegOffset,
                                                            __local_mem__ U *hcPtr, __local_mem__ U *irPtr);
    // Sub-functions for method splitting
    __aicore__ inline void SinkhornGradStage1_2(__local_mem__ U *colSumBase, __local_mem__ U *rowNormBase,
                                                 __local_mem__ U *skGradPtr, __local_mem__ U *tmpDivPtr,
                                                 __local_mem__ U *tmpPtr, int64_t lenBs);
    __aicore__ inline void SinkhornGradStage3_4(__local_mem__ U *rowSumBase, __local_mem__ U *rowNormBase,
                                                 __local_mem__ U *skGradPtr, __local_mem__ U *tmpDivPtr,
                                                 __local_mem__ U *tmpDiv3Ptr, __local_mem__ U *tmpPtr,
                                                 int64_t lenBs);
    __aicore__ inline void PhaseA_ComputeVfLoop(__local_mem__ X_T *xPtr, __local_mem__ U *xFp32Ptr,
                                                  __local_mem__ GRADHIN_T *gradHinPtr,
                                                  __local_mem__ U *gradXFromHinPtr,
                                                  __local_mem__ U *hPrePtr, __local_mem__ U *gradHPrePtr,
                                                  int64_t lenBs, int64_t lenCSlice,
                                                  int64_t alignXCSlice, int64_t alignUCSlice,
                                                  int64_t alignGhCSlice);
    __aicore__ inline void PhaseB_ForwardCompute(__local_mem__ U *zPtr, __local_mem__ U *noPtr,
                                                   __local_mem__ U *hcPtr, __local_mem__ U *biasPtr,
                                                   __local_mem__ U *irPtr, __local_mem__ U *ghPrePtr,
                                                   __local_mem__ U *gradZPtr, int64_t ofsBs, int64_t lenBs);
    __aicore__ inline void PhaseB_GradAlphaBiasAndNorm(int64_t ofsBs, int64_t lenBs,
                                                        __local_mem__ U *noPtr, __local_mem__ U *gradZPtr,
                                                        __local_mem__ U *hcPtr, __local_mem__ U *irPtr);
    __aicore__ inline void CopyOutGradHcRows(int64_t wsIdx, int64_t nRows);

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
    GlobalTensor<U> gradAlphaBiasWs_;

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
    nnAlign_            = n_ * nAlign_;
    kAlign_             = (2 + n_) * nAlign_;
    bsAlign_            = Ops::Base::CeilAlign(int64_t(ubFactorBS_ * uSize_), blockSize_) / uSize_;
    bsnAlign_           = ubFactorBS_ * nAlign_;
    bsnnAlign_          = ubFactorBS_ * nnAlign_;
    cAlignFp32_         = Ops::Base::CeilAlign(int64_t(c_ * uSize_), blockSize_) / uSize_;
    cAlignFp16_         = Ops::Base::CeilAlign(int64_t(c_ * xSize_), blockSize_) / xSize_;
    ubFactorCAlignFp32_ = Ops::Base::CeilAlign(int64_t(ubFactorC_ * uSize_), blockSize_) / uSize_;
    ubFactorCAlignFp16_ = Ops::Base::CeilAlign(int64_t(ubFactorC_ * xSize_), blockSize_) / xSize_;

    cNumPerBS_  = Ops::Base::CeilDiv(c_, ubFactorCAlignFp32_);
    cTailPerBS_ = c_ - (cNumPerBS_ - 1) * ubFactorCAlignFp32_;

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
        int64_t biasBaseElem = Ops::Base::CeilAlign(int64_t(3 * uSize_), blockSize_) / uSize_;
        int64_t perCoreElem = biasBaseElem + 2 * nAlign_ + nnAlign_;
        int64_t gradAlphaBiasOfs = gradXFromMatmulOfs + totalBs * n_ * c_ + WS_INTERVAL;
        auto ws = reinterpret_cast<__gm__ U *>(workspace);
        xFp32Ws_.SetGlobalBuffer(ws);
        gradXFromHinWs_.SetGlobalBuffer(ws + gradXFromHinOfs);
        gradHcBeforeNormWs_.SetGlobalBuffer(ws + gradHcBeforeNormOfs);
        gradXFromMatmulWs_.SetGlobalBuffer(ws + gradXFromMatmulOfs);
        gradAlphaBiasWs_.SetGlobalBuffer(ws + gradAlphaBiasOfs);

        pipe_->InitBuffer(xQue_, 2, ubFactorCAlignFp32_ * n_ * ubFactorBS_ * uSize_);
        pipe_->InitBuffer(gradHinQue_, 2, ubFactorCAlignFp32_ * n_ * ubFactorBS_ * uSize_);
        pipe_->InitBuffer(hPreAndGradHPostQue_, 2, bsnAlign_ * uSize_);
        pipe_->InitBuffer(hcBeforeNormQue_, 2, ubFactorBS_ * kAlign_ * uSize_);
        pipe_->InitBuffer(invRmsQue_, 2, bsAlign_ * uSize_);
        pipe_->InitBuffer(gradHResQue_, 2, ubFactorBS_ * nnAlign_ * uSize_);
        pipe_->InitBuffer(xFp32Que_, 2, ubFactorCAlignFp32_ * n_ * ubFactorBS_ * uSize_);
        pipe_->InitBuffer(gradXFromHinQue_, 2, ubFactorCAlignFp32_ * n_ * ubFactorBS_ * uSize_);
        pipe_->InitBuffer(gradAlphaAndBiasQue_, 1, (2 * nAlign_ + nnAlign_ + nAlign_) * uSize_ + blockSize_);

        pipe_->InitBuffer(gradHPreBuf_, bsnAlign_ * uSize_);
        pipe_->InitBuffer(gradZBuf_, ubFactorBS_ * kAlign_ * uSize_);
        pipe_->InitBuffer(biasBuf_, kAlign_ * uSize_);
        pipe_->InitBuffer(normOutForwardBuf_, ubFactorBS_ * kAlign_ * uSize_);
        pipe_->InitBuffer(zBuf_, ubFactorBS_ * kAlign_ * uSize_);
        pipe_->InitBuffer(gradCalcBuf1_, (kAlign_ + 2 * nAlign_) * uSize_);
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
        if (blockIdx_ < usedCoreNum_) {
            GetAlphaAndBias();

            int64_t numLoopBs = Ops::Base::CeilDiv(bsPerCore_, ubFactorBS_);
            int64_t tailBs    = bsPerCore_ - ubFactorBS_ * (numLoopBs - 1);

            for (int64_t idxBs = 0; idxBs < numLoopBs; idxBs++) {
                int64_t lenBs = (idxBs == numLoopBs - 1) ? tailBs : ubFactorBS_;
                PhaseA_CLoop(idxBs, lenBs);
                PhaseB_KDim(idxBs, lenBs);
            }
        }
        SyncAll();
    }


    // CrossCoreSetFlag<AIV_AIC_MODE, PIPE_MTE3>(PIPE_MTE3_FLAG);
    // CrossCoreWaitFlag<AIV_AIC_MODE>(PIPE_FIX_FLAG);

    if ASCEND_IS_AIC {
        if (blockIdx_ < aicCoreNum_) {
            // PhaseD_MatMul();
        }
    }

    // SyncAll();

    // CrossCoreSetFlag<AIV_AIC_MODE, PIPE_FIX>(PIPE_FIX_FLAG);

    if ASCEND_IS_AIV {
        if (blockIdx_ < usedCoreNum_) {
            int64_t numLoopBs = Ops::Base::CeilDiv(bsPerCore_, ubFactorBS_);
            int64_t tailBs    = bsPerCore_ - ubFactorBS_ * (numLoopBs - 1);
            for (int64_t idxBs = 0; idxBs < numLoopBs; idxBs++) {
                int64_t lenBs = (idxBs == numLoopBs - 1) ? tailBs : ubFactorBS_;
                PhaseE_CLoop(idxBs, lenBs);
            }

            if (blockIdx_ == 0) {
                // ReduceBiasAlpha();
            }
        }
    }

    SyncAll();
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

    CopyInBuf(biasBuf_, biasGm_, (2 + n_), n_, 0, nAlign_);
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::PhaseA_CLoop(int64_t idxBs, int64_t lenBs)
{
    // ================================================================
    // PhaseA — C 维循环: gradHPre + gradXFromHin + xFp32
    //
    // 步骤:
    //   1. gradHPreBuf_ 清零 [lenBs, n_]
    //   2. 搬入 hPre[lenBs, n_] GM→UB
    //   3. C 维循环:
    //      a. 搬入 x[lenBs, n_, cSlice] + gradHin[lenBs, cSlice] GM→UB
    //      b. 逐 BS 逐 N 逐 VF:
    //         - gradHin cast fp32
    //         - x cast fp32
    //         - gradXFromHin = gradHin * hPre        → UB (后搬出 WS)
    //         - gradHPre    += (x * gradHin).sum(C)  → UB (累加)
    //      c. 搬出 xFp32 + gradXFromHin → WS
    //
    // UB 布局 (每 cSlice 迭代):
    //   gradHPreBuf_:    [lenBs, n_]         fp32, 累加器, 跨 cSlice 持久
    //   hPreQue_:        [lenBs, n_]         fp32, 跨 cSlice 持久
    //   xQue_:           [lenBs, n_, cAligned]  X_T (fp16/bf16)
    //   gradHinQue_:     [lenBs, cAligned]      GRADHIN_T (fp16/bf16)
    //   xFp32Que_:       [lenBs, n_, cAligned]  fp32 (输出)
    //   gradXFromHinQue_:[lenBs, n_, cAligned]  fp32 (输出)
    //
    // GM 布局:
    //   x:        [B*S, n_, c_]    按行存储, 每行 n_*c_ 元素
    //   gradHin:  [B*S, c_]        按行存储, 每行 c_ 元素
    //   hPre:     [B*S, n_]        按行存储, 每行 n_ 元素
    // ================================================================

    // --- GM 偏移: 当前 BS 批次在全局中的位置 ---
    int64_t ofsGmBs    = blockIdx_ * td_->blockFactorBS + idxBs * ubFactorBS_;
    int64_t ofsGmHPre  = ofsGmBs * n_;
    int64_t ofsGmX     = ofsGmBs * n_ * c_;
    int64_t ofsGmGh = ofsGmBs * c_;

    // --- Step 1: gradHPre 清零 (仅首块全量清零, 避免尾块残留) ---
    auto gradHPreUb = gradHPreBuf_.template Get<U>();
    if (idxBs == 0) {
        Duplicate(gradHPreUb, float(0), ubFactorBS_ * nAlign_);
    }

    // --- Step 2: 搬入 hPre[lenBs, n_] ---
    CopyInQue(hPreAndGradHPostQue_, hPreGm_[ofsGmHPre], lenBs, n_, 0, nAlign_);
    auto hPreUb = hPreAndGradHPostQue_.template DeQue<U>();

    __local_mem__ U *hPrePtr     = reinterpret_cast<__local_mem__ U *>(hPreUb.GetPhyAddr());
    __local_mem__ U *gradHPrePtr = reinterpret_cast<__local_mem__ U *>(gradHPreUb.GetPhyAddr());

    // --- Step 3: C 维循环 ---
    uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(U);

    for (int64_t idxC = 0; idxC < cNumPerBS_; idxC++) {
        // 当前 C 切片的实际元素数 (尾块可能不足)
        int64_t lenCSlice = (idxC == cNumPerBS_ - 1) ? cTailPerBS_ : ubFactorCAlignFp32_;

        // 对齐后的每行元素数 (按各类型大小对齐到 blockSize_)
        int64_t alignXCSlice  = Ops::Base::CeilAlign(lenCSlice * xSize_, blockSize_) / xSize_;
        int64_t alignUCSlice  = Ops::Base::CeilAlign(lenCSlice * uSize_, blockSize_) / uSize_;
        int64_t alignGhCSlice = Ops::Base::CeilAlign(lenCSlice * ghSize_, blockSize_) / ghSize_;

        // 当前 C 切片在 GM 中的偏移
        int64_t ofsGmXSlice     = ofsGmX + idxC * ubFactorC_;
        int64_t ofsGmGhSlice = ofsGmGh + idxC * ubFactorC_;

        // --- Step 3a: 搬入 x[lenBs, n_, cSlice] + gradHin[lenBs, cSlice] ---
        // GM 中 x 按 [bs, n, c] 排布, 每行 c_ 元素, 跨步 (c_ - lenCSlice) 跳过剩余 C
        CopyInQue(xQue_, xGm_[ofsGmXSlice], lenBs * n_, lenCSlice, (c_ - lenCSlice) * xSize_, alignXCSlice);
        CopyInQue(gradHinQue_, gradHinGm_[ofsGmGhSlice], lenBs, lenCSlice, (c_ - lenCSlice) * ghSize_, alignGhCSlice);
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

        PhaseA_ComputeVfLoop(xPtr, xFp32Ptr, gradHinPtr, gradXFromHinPtr,
                             hPrePtr, gradHPrePtr, lenBs, lenCSlice,
                             alignXCSlice, alignUCSlice, alignGhCSlice);

        xFp32Que_.EnQue(xFp32Ub);
        gradXFromHinQue_.EnQue(gradXFromHinUb);
        xQue_.FreeTensor(xUb);
        gradHinQue_.FreeTensor(gradHinUb);

        // --- Step 3c: 搬出 xFp32 + gradXFromHin → WS ---
        // CopyOutQue 内部自动 DeQue + FreeTensor, 无需手动 DeQue
        // WS 中按 [bs, n, c] 排布, 跨步 (c_ - lenCSlice) 跳过剩余 C
        CopyOutQue(xFp32Que_, xFp32Ws_[ofsGmXSlice], lenBs * n_, lenCSlice, alignUCSlice, (c_ - lenCSlice) * uSize_);
        CopyOutQue(gradXFromHinQue_, gradXFromHinWs_[ofsGmXSlice], lenBs * n_, lenCSlice, alignUCSlice, (c_ - lenCSlice) * uSize_);
    }

    AscendC::printf("[DUMP] 200 grad_h_pre lenBs=%d\n", lenBs);
    AscendC::DumpTensor(gradHPreBuf_.template Get<U>(), 200, lenBs * nAlign_);
    AscendC::printf("[DUMP] 215 grad_x_from_hin lenBs=%d\n", lenBs);
    AscendC::DumpTensor(gradXFromHinWs_[ofsGmBs * n_ * c_], 215, lenBs * n_ * c_);

    hPreAndGradHPostQue_.FreeTensor(hPreUb);
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::PhaseA_ComputeVfLoop(
    __local_mem__ X_T *xPtr, __local_mem__ U *xFp32Ptr,
    __local_mem__ GRADHIN_T *gradHinPtr, __local_mem__ U *gradXFromHinPtr,
    __local_mem__ U *hPrePtr, __local_mem__ U *gradHPrePtr,
    int64_t lenBs, int64_t lenCSlice,
    int64_t alignXCSlice, int64_t alignUCSlice, int64_t alignGhCSlice)
{
    uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(U);
    uint16_t cntVfLoop = static_cast<uint16_t>((lenCSlice + vfLen - 1) / vfLen);

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

        uint32_t maskLen = static_cast<uint32_t>(lenCSlice);

        for (uint16_t idxBsInner = 0; idxBsInner < static_cast<uint16_t>(lenBs); idxBsInner++) {
            auto hPreRow       = hPrePtr     + idxBsInner * nAlign_;
            auto gradHPreRow   = gradHPrePtr + idxBsInner * nAlign_;
            auto gradHinRow    = gradHinPtr  + idxBsInner * alignGhCSlice;
            auto xBase          = xPtr          + idxBsInner * n_ * alignXCSlice;
            auto xFp32Base      = xFp32Ptr      + idxBsInner * n_ * alignUCSlice;
            auto gradXFromHBase = gradXFromHinPtr + idxBsInner * n_ * alignUCSlice;

            for (uint16_t idxVf = 0; idxVf < cntVfLoop; idxVf++) {
                valueMaskReg = AscendC::MicroAPI::UpdateMask<U>(maskLen);
                int64_t ofsVfC = idxVf * vfLen;

                AscendC::MicroAPI::LoadAlign<GRADHIN_T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    gradHinReg, gradHinRow + ofsVfC);
                AscendC::MicroAPI::Cast<U, GRADHIN_T, castTrait16ToFloat>(gradHinFp32Reg, gradHinReg, valueMaskReg);

                for (uint16_t idxN = 0; idxN < static_cast<uint16_t>(n_); idxN++) {
                    AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                                   AscendC::MicroAPI::MemType::SCALAR_LOAD>();

                    auto xRow         = xBase         + idxN * alignXCSlice;
                    auto xFp32Row     = xFp32Base     + idxN * alignUCSlice;
                    auto gradXFromHRow = gradXFromHBase + idxN * alignUCSlice;

                    U hPreVal     = hPreRow[idxN];
                    U gradHPreOld = gradHPreRow[idxN];

                    AscendC::MicroAPI::LoadAlign<X_T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        xReg, xRow + ofsVfC);
                    AscendC::MicroAPI::Cast<U, X_T, castTrait16ToFloat>(xFp32Reg, xReg, valueMaskReg);

                    AscendC::MicroAPI::Muls(gradXFromHinReg, gradHinFp32Reg, hPreVal, valueMaskReg);

                    AscendC::MicroAPI::Mul(gradHPreReg, xFp32Reg, gradHinFp32Reg, valueMaskReg);
                    AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::SUM, U, U>(gradHPreReg, gradHPreReg, valueMaskReg);
                    AscendC::MicroAPI::Adds(gradHPreAccReg, gradHPreReg, gradHPreOld, oneMaskReg);

                    AscendC::MicroAPI::StoreAlign(
                        gradXFromHRow + ofsVfC, gradXFromHinReg, valueMaskReg);
                    AscendC::MicroAPI::StoreAlign(
                        xFp32Row + ofsVfC, xFp32Reg, valueMaskReg);
                    AscendC::MicroAPI::Store(gradHPreRow + idxN, gradHPreAccReg, 1);
                }
            }
        }
    }
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::PhaseB_KDim(int64_t idxBs, int64_t lenBs)
{
    int64_t ofsBs = blockIdx_ * td_->blockFactorBS + idxBs * ubFactorBS_;
    int64_t ofsHcBN = ofsBs * kDim_;
    int64_t ofsInvRms = ofsBs;

    if (idxBs == 0) {
        auto gradInvRmsUb = gradInvRmsBuf_.template Get<U>();
        Duplicate(gradInvRmsUb, U(0), bsPerCore_);
    }

    // --- 搬入 invRms + hcBeforeNorm (k_ 全量对齐) ---
    CopyInQue(invRmsQue_, invRmsGm_[ofsInvRms], 1, lenBs);
    CopyInQue(hcBeforeNormQue_, hcBeforeNormGm_[ofsHcBN], lenBs * (2 + n_), n_, 0, nAlign_);

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

    PhaseB_ForwardCompute(zPtr, noPtr, hcPtr, biasPtr, irPtr, ghPrePtr, gradZPtr, ofsBs, lenBs);

    // --- SinkhornGrad + ExpGrad: gradZ_res ---
    int64_t ofsGradHRes = ofsBs * nn_;
    CopyInQue(gradHResQue_, gradHResGm_[ofsGradHRes], lenBs * n_, n_, 0, nAlign_);
    auto gradHResUb = gradHResQue_.template DeQue<U>();
    __local_mem__ U *skGradPtr = reinterpret_cast<__local_mem__ U *>(gradHResUb.GetPhyAddr());

    // 搬入 sumOut[skIter, 2, lenBs, n] + normOut[skIter, lenBs, n, n]
    auto sumOutRowUb = sumOutRowBuf_.template Get<U>();
    auto sumOutColUb = sumOutColBuf_.template Get<U>();
    auto normOutUb   = normOutBuf_.template Get<U>();
    __local_mem__ U *sumOutRowPtr = reinterpret_cast<__local_mem__ U *>(sumOutRowUb.GetPhyAddr());
    __local_mem__ U *sumOutColPtr = reinterpret_cast<__local_mem__ U *>(sumOutColUb.GetPhyAddr());
    __local_mem__ U *normOutPtr   = reinterpret_cast<__local_mem__ U *>(normOutUb.GetPhyAddr());

    for (int64_t iter = 0; iter < skIterCount_; iter++) {
        int64_t totalBs = td_->b * td_->s;
        int64_t ofsGmRowSum = (iter * 2 * totalBs + ofsBs) * n_;
        int64_t ofsGmColSum = (iter * 2 * totalBs + totalBs + ofsBs) * n_;
        int64_t ofsGmNormOut = (iter * 2 * totalBs + ofsBs) * nn_;
        int64_t ofsUbRowSum = iter * ubFactorBS_ * nAlign_;
        int64_t ofsUbColSum = iter * ubFactorBS_ * nAlign_;
        int64_t ofsUbNormOut = iter * ubFactorBS_ * nnAlign_;

        auto rowBufLocal = sumOutRowBuf_.template Get<U>();
        auto colBufLocal = sumOutColBuf_.template Get<U>();
        auto normBufLocal = normOutBuf_.template Get<U>();

        DataCopyExtParams rowParams;
        rowParams.blockCount = static_cast<uint16_t>(lenBs);
        rowParams.blockLen = static_cast<uint32_t>(n_ * uSize_);
        rowParams.srcStride = 0;
        rowParams.dstStride = 0;
        DataCopyPadExtParams<U> rowPad{nAlign_ > n_, 0, static_cast<uint8_t>(nAlign_ - n_), U(0)};
        DataCopyPad(rowBufLocal[ofsUbRowSum], sumOutGm_[ofsGmRowSum], rowParams, rowPad);
        DataCopyPad(colBufLocal[ofsUbColSum], sumOutGm_[ofsGmColSum], rowParams, rowPad);

        DataCopyExtParams normParams;
        normParams.blockCount = static_cast<uint16_t>(lenBs * n_);
        normParams.blockLen = static_cast<uint32_t>(n_ * uSize_);
        normParams.srcStride = 0;
        normParams.dstStride = 0;
        DataCopyPadExtParams<U> normPad{nAlign_ > n_, 0, static_cast<uint8_t>(nAlign_ - n_), U(0)};
        DataCopyPad(normBufLocal[ofsUbNormOut], normOutGm_[ofsGmNormOut], normParams, normPad);
    }
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    SinkhornGradSimdVf(skGradPtr, idxBs, lenBs);
    AscendC::printf("[DUMP] 218 sk_grad(sinkhorn) lenBs=%d\n", lenBs);
    AscendC::DumpTensor(gradHResUb, 218, lenBs * nnAlign_);

    __local_mem__ U *zResPtr = zPtr + 2 * nAlign_;
    __local_mem__ U *gradZResPtr = gradZPtr + 2 * nAlign_;
    ExpGradSimdVf(gradZResPtr, skGradPtr, zResPtr, idxBs, lenBs);

    gradHResQue_.FreeTensor(gradHResUb);

    PhaseB_GradAlphaBiasAndNorm(ofsBs, lenBs, noPtr, gradZPtr, hcPtr, irPtr);

    AscendC::printf("[DUMP] 211 hc_before_norm lenBs=%d\n", lenBs);
    AscendC::DumpTensor(hcUb, 211, lenBs * kAlign_);
    AscendC::printf("[DUMP] 212 inv_rms lenBs=%d\n", lenBs);
    AscendC::DumpTensor(invRmsUb, 212, lenBs);

    invRmsQue_.FreeTensor(invRmsUb);
    hcBeforeNormQue_.FreeTensor(hcUb);
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::PhaseB_ForwardCompute(
    __local_mem__ U *zPtr, __local_mem__ U *noPtr,
    __local_mem__ U *hcPtr, __local_mem__ U *biasPtr, __local_mem__ U *irPtr,
    __local_mem__ U *ghPrePtr, __local_mem__ U *gradZPtr, int64_t ofsBs, int64_t lenBs)
{
    ComputeZAndForwardPart<U>(zPtr, noPtr, hcPtr, biasPtr, irPtr, lenBs, nAlign_, kAlign_, 256, alphaPre_);
    ComputeZAndForwardPart<U>(zPtr + nAlign_, noPtr + nAlign_, hcPtr + nAlign_, biasPtr + nAlign_, irPtr,
                              lenBs, nAlign_, kAlign_, 256, alphaPost_);
    ComputeZAndForwardPart<U>(zPtr + 2 * nAlign_, noPtr + 2 * nAlign_, hcPtr + 2 * nAlign_, biasPtr + 2 * nAlign_,
                              irPtr, lenBs, nnAlign_, kAlign_, 256, alphaRes_);

    SigmoidGrad<U, true>(ghPrePtr, zPtr, gradZPtr, lenBs, nAlign_, kAlign_, 256, eps_);

    CopyInQue(hPreAndGradHPostQue_, gradHPostGm_[ofsBs * n_], lenBs, n_, 0, nAlign_);
    auto gradHPostUb = hPreAndGradHPostQue_.template DeQue<U>();
    __local_mem__ U *ghPostPtr = reinterpret_cast<__local_mem__ U *>(gradHPostUb.GetPhyAddr());
    SigmoidGrad<U, false>(ghPostPtr, zPtr + nAlign_, gradZPtr + nAlign_, lenBs, nAlign_, kAlign_, 256, eps_);
    hPreAndGradHPostQue_.FreeTensor(gradHPostUb);
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::PhaseB_GradAlphaBiasAndNorm(
    int64_t ofsBs, int64_t lenBs,
    __local_mem__ U *noPtr, __local_mem__ U *gradZPtr,
    __local_mem__ U *hcPtr, __local_mem__ U *irPtr)
{
    auto gradAlphaBiasUb = gradAlphaAndBiasQue_.template AllocTensor<U>();
    __local_mem__ U *baseAddr = reinterpret_cast<__local_mem__ U *>(gradAlphaBiasUb.GetPhyAddr());
    __local_mem__ U *gradAlphaAddr = baseAddr;
    int64_t biasBase = 3;
    biasBase = Ops::Base::CeilAlign(biasBase * uSize_, blockSize_) / uSize_;
    __local_mem__ U *gradBiasPre  = baseAddr + biasBase;
    __local_mem__ U *gradBiasPost = baseAddr + biasBase + nAlign_;
    __local_mem__ U *gradBiasRes  = baseAddr + biasBase + 2 * nAlign_;

    ComputeGradAlphaAll<U>(noPtr, gradZPtr, gradAlphaAddr, lenBs, n_, kAlign_, 256);
    ComputeGradBiasAll<U>(gradZPtr, lenBs, n_, kAlign_, 256, gradBiasPre);

    gradAlphaAndBiasQue_.EnQue(gradAlphaBiasUb);
    auto gradAlphaBiasOut = gradAlphaAndBiasQue_.template DeQue<U>();

    int64_t perCoreElem = biasBase + 2 * nAlign_ + nnAlign_;
    int64_t wsOffset = blockIdx_ * perCoreElem;
    CopyOutQue(gradAlphaAndBiasQue_, gradAlphaBiasWs_[wsOffset], 1, perCoreElem);

    gradAlphaAndBiasQue_.FreeTensor(gradAlphaBiasOut);

    AscendC::printf("[DUMP] 220 grad_alpha+bias(ws) perCoreElem=%d\n", perCoreElem);
    AscendC::DumpTensor(gradAlphaBiasWs_[wsOffset], 220, perCoreElem);
    AscendC::printf("[DUMP] 210 grad_z(pre+post+res) lenBs=%d\n", lenBs);
    AscendC::DumpTensor(gradZBuf_.template Get<U>(), 210, lenBs * kAlign_);

    ComputeGNormOutAndGHcBeforeNorm(ofsBs, lenBs, alphaPre_, 1, 0, 0, hcPtr, irPtr);
    ComputeGNormOutAndGHcBeforeNorm(ofsBs, lenBs, alphaPost_, 1, nAlign_, n_, hcPtr, irPtr);
    ComputeGNormOutAndGHcBeforeNorm(ofsBs, lenBs, alphaRes_, n_, 2 * nAlign_, 2 * n_, hcPtr, irPtr);
    AscendC::printf("[DUMP] 213 grad_inv_rms bsPerCore=%d\n", bsPerCore_);
    AscendC::DumpTensor(gradInvRmsBuf_.template Get<U>(), 213, bsPerCore_);
    AscendC::printf("[DUMP] 214 grad_hc_before_norm(ws) lenBs=%d kDim=%d\n", lenBs, kDim_);
    AscendC::DumpTensor(gradHcBeforeNormWs_[ofsBs * kDim_], 214, lenBs * kDim_);
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
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::PhaseE_CLoop(int64_t idxBs, int64_t lenBs)
{
    AscendC::printf("[DBG FN] >> PhaseE_CLoop blk=%d idxBs=%d lenBs=%d\n", GetBlockIdx(), idxBs, lenBs);
    int64_t ofsBs = blockIdx_ * td_->blockFactorBS + idxBs * ubFactorBS_;
    auto gradInvRmsUb = gradInvRmsBuf_.template Get<U>();
    int64_t cntNC = n_ * c_;

    for (int64_t loopBs = 0; loopBs < lenBs; loopBs++) {
        int64_t idxCurBs = ofsBs + loopBs;
        U curInvRms = invRmsGm_(idxCurBs);
        U curGradInvRms = gradInvRmsUb.GetValue(loopBs);
        U invRmsCubed = curInvRms * curInvRms * curInvRms;
        int64_t ofsNC = idxCurBs * cntNC;

            AscendC::printf("[DUMP] 250 phaseE bs0: invRms=%f gradInvRms=%f invRmsCubed=%f NC=%d\n",
                            curInvRms, curGradInvRms, invRmsCubed, cntNC);
            AscendC::printf("[DUMP] 213 grad_inv_rms(verify) bsPerCore=%d\n", bsPerCore_);
            AscendC::DumpTensor(gradInvRmsBuf_.template Get<U>(), 213, bsPerCore_);

        for (int64_t idxC = 0; idxC < cNumPerBS_; idxC++) {
            int64_t cLen = (idxC == cNumPerBS_ - 1) ? cTailPerBS_ : ubFactorCAlignFp32_;
            int64_t ofsCalc = ofsNC + idxC * ubFactorC_;

            int64_t alignUCSlice = Ops::Base::CeilAlign(cLen * uSize_, blockSize_) / uSize_;

            CopyInQue(xQue_, xFp32Ws_[ofsCalc], n_, cLen, (c_ - cLen) * uSize_, alignUCSlice);
            CopyInQue(gradHinQue_, gradXFromHinWs_[ofsCalc], n_, cLen, (c_ - cLen) * uSize_, alignUCSlice);
            CopyInQue(hPreAndGradHPostQue_, gradXFromMatmulWs_[ofsCalc], n_, cLen, (c_ - cLen) * uSize_);

            auto xLocal = xQue_.template DeQue<U>();
            auto gradHinLocal = gradHinQue_.template DeQue<U>();
            auto gradMatmulLocal = hPreAndGradHPostQue_.template DeQue<U>();
            auto xOutLocal = xFp32Que_.template AllocTensor<U>();

            if (loopBs == 0 && idxC == 0) {
                AscendC::printf("[DUMP] 250a xFp32_input(bs0,c0) n*cLen=%d\n", n_ * cLen);
                AscendC::DumpTensor(xLocal, 250, n_ * cLen);
            }

            ComputeGradXFromRms<U>(xLocal, xOutLocal, curGradInvRms, invRmsCubed, n_ * cLen, cntNC);

                AscendC::printf("[DUMP] 251 grad_x_from_rms(bs0,c0) n*cLen=%d\n", n_ * cLen);
                AscendC::DumpTensor(xOutLocal, 251, n_ * cLen);

            Add(xOutLocal, xOutLocal, gradHinLocal, n_ * cLen);
            Add(xOutLocal, xOutLocal, gradMatmulLocal, n_ * cLen);

                AscendC::printf("[DUMP] 252 grad_x_final(bs0,c0) n*cLen=%d\n", n_ * cLen);
                AscendC::DumpTensor(xOutLocal, 252, n_ * cLen);

            if constexpr (!std::is_same<X_T, U>::value) {
                Cast(xOutLocal.template ReinterpretCast<X_T>(), xOutLocal, RoundMode::CAST_RINT, n_ * cLen);
            }
            xQue_.FreeTensor(xLocal);
            gradHinQue_.FreeTensor(gradHinLocal);
            hPreAndGradHPostQue_.FreeTensor(gradMatmulLocal);
            xFp32Que_.EnQue(xOutLocal);
            CopyOutQue(xFp32Que_, gradXGm_[ofsCalc], n_, cLen, alignUCSlice, 0);
        }
    }
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::ReduceBiasAlpha()
{
    if (blockIdx_ != 0) return;

    int64_t biasBase = Ops::Base::CeilAlign(int64_t(3 * uSize_), blockSize_) / uSize_;
    int64_t perCoreElem = biasBase + 2 * nAlign_ + nnAlign_;

    auto alphaBiasUb = gradAlphaAndBiasQue_.template AllocTensor<U>();
    __local_mem__ U *ubPtr = reinterpret_cast<__local_mem__ U *>(alphaBiasUb.GetPhyAddr());
    auto tmpUb = gradCalcBuf1_.template Get<U>();
    __local_mem__ U *tmpPtr = reinterpret_cast<__local_mem__ U *>(tmpUb.GetPhyAddr());

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(perCoreElem * uSize_);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    for (int64_t c = 0; c < usedCoreNum_; c++) {
        int64_t wsOffset = c * perCoreElem;
        DataCopy(tmpUb, gradAlphaBiasWs_[wsOffset], perCoreElem);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        if (c == 0) {
            DataCopy(alphaBiasUb, tmpUb, perCoreElem);
        } else {
            __VEC_SCOPE__
            {
                MicroAPI::RegTensor<U> accReg, inReg;
                uint16_t vfLen = static_cast<uint16_t>(Ops::Base::GetVRegSize() / sizeof(U));
                uint16_t totalElems = static_cast<uint16_t>(perCoreElem);
                uint16_t loopCnt = (totalElems + vfLen - 1) / vfLen;
                for (uint16_t i = 0; i < loopCnt; i++) {
                    uint16_t offset = i * vfLen;
                    uint32_t remain = static_cast<uint32_t>(totalElems - offset);
                    MicroAPI::MaskReg mask = MicroAPI::UpdateMask<U>(remain);
                    MicroAPI::LoadAlign(accReg, ubPtr + offset);
                    MicroAPI::LoadAlign(inReg, tmpPtr + offset);
                    MicroAPI::Add(accReg, accReg, inReg, mask);
                    MicroAPI::StoreAlign(ubPtr + offset, accReg, mask);
                }
            }
        }
    }

    gradAlphaAndBiasQue_.EnQue(alphaBiasUb);
    auto outUb = gradAlphaAndBiasQue_.template DeQue<U>();
    __local_mem__ U *outPtr = reinterpret_cast<__local_mem__ U *>(outUb.GetPhyAddr());

    // gradAlpha[3]
    for (int64_t i = 0; i < 3; i++) {
        gradAlphaGm_.SetValue(i, outPtr[i]);
    }

    // gradBias: [pre(n) post(n) res(nn)] = kDim 连续
    int64_t biasGmIdx = 0;
    // pre
    for (int64_t i = 0; i < n_; i++) {
        gradBiasGm_.SetValue(biasGmIdx++, outPtr[biasBase + i]);
    }
    // post
    for (int64_t i = 0; i < n_; i++) {
        gradBiasGm_.SetValue(biasGmIdx++, outPtr[biasBase + nAlign_ + i]);
    }
    // res
    for (int64_t i = 0; i < nn_; i++) {
        int64_t row = i / n_;
        int64_t col = i % n_;
        gradBiasGm_.SetValue(biasGmIdx++, outPtr[biasBase + 2 * nAlign_ + row * nAlign_ + col]);
    }

    AscendC::printf("[DUMP] 260 grad_alpha(final) [%f,%f,%f]\n", outPtr[0], outPtr[1], outPtr[2]);
    AscendC::printf("[DUMP] 261 grad_bias(final) pre=[%f,%f,%f,%f] post=[%f,%f,%f,%f]\n",
        outPtr[biasBase], outPtr[biasBase+1], outPtr[biasBase+2], outPtr[biasBase+3],
        outPtr[biasBase+nAlign_], outPtr[biasBase+nAlign_+1],
        outPtr[biasBase+nAlign_+2], outPtr[biasBase+nAlign_+3]);

    gradAlphaAndBiasQue_.FreeTensor(outUb);
}

// ================================================================
// SinkhornGradSimdVf — nAlign_ aligned version
// ================================================================
template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::SinkhornGradSimdVf(
    __local_mem__ U *skGradPtr, int64_t idxBs, int64_t lenBs)
{
    auto sumOutRowUb = sumOutRowBuf_.template Get<U>();
    auto sumOutColUb = sumOutColBuf_.template Get<U>();
    auto normOutUb   = normOutBuf_.template Get<U>();

    __local_mem__ U *sumOutRowPtr = reinterpret_cast<__local_mem__ U *>(sumOutRowUb.GetPhyAddr());
    __local_mem__ U *sumOutColPtr = reinterpret_cast<__local_mem__ U *>(sumOutColUb.GetPhyAddr());
    __local_mem__ U *normOutPtr   = reinterpret_cast<__local_mem__ U *>(normOutUb.GetPhyAddr());

    auto tmpDivUb  = tmpDivBuf_.template Get<U>();
    auto tmpDiv3Ub = tmpDiv3Buf_.template Get<U>();
    auto tmpMulUb  = tmpMulBuf_.template Get<U>();
    auto tmpUb     = tmpBuf_.template Get<U>();

    __local_mem__ U *tmpDivPtr  = reinterpret_cast<__local_mem__ U *>(tmpDivUb.GetPhyAddr());
    __local_mem__ U *tmpDiv3Ptr = reinterpret_cast<__local_mem__ U *>(tmpDiv3Ub.GetPhyAddr());
    __local_mem__ U *tmpMulPtr  = reinterpret_cast<__local_mem__ U *>(tmpMulUb.GetPhyAddr());
    __local_mem__ U *tmpPtr     = reinterpret_cast<__local_mem__ U *>(tmpUb.GetPhyAddr());

    for (int64_t iter = skIterCount_ - 1; iter >= 0; --iter) {
        __local_mem__ U *colSumBase = sumOutColPtr + iter * ubFactorBS_ * nAlign_;
        __local_mem__ U *rowSumBase = sumOutRowPtr + iter * ubFactorBS_ * nAlign_;
        __local_mem__ U *rowNormBase = normOutPtr + iter * ubFactorBS_ * nnAlign_;

        SinkhornGradStage1_2(colSumBase, rowNormBase, skGradPtr, tmpDivPtr, tmpPtr, lenBs);
        SinkhornGradStage3_4(rowSumBase, rowNormBase, skGradPtr, tmpDivPtr, tmpDiv3Ptr, tmpPtr, lenBs);
    }
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::SinkhornGradStage1_2(
    __local_mem__ U *colSumBase, __local_mem__ U *rowNormBase,
    __local_mem__ U *skGradPtr, __local_mem__ U *tmpDivPtr, __local_mem__ U *tmpPtr,
    int64_t lenBs)
{
    uint16_t bsLenLoop = static_cast<uint16_t>(lenBs);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<U> colSumReg, rowNormReg, skGradReg;
        MicroAPI::RegTensor<U> tmpDivReg, tmpMulReg, colSumSqReg, tmpDiv2Reg, reduceSumReg;
        MicroAPI::MaskReg maskN = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
        uint32_t maskLenN = static_cast<uint32_t>(n_);
        maskN = MicroAPI::UpdateMask<U>(maskLenN);

        for (uint16_t j = 0; j < bsLenLoop; j++) {
            int64_t bsOff = j * nnAlign_;
            int64_t bsOffN = j * nAlign_;

            MicroAPI::LoadAlign(colSumReg, colSumBase + bsOffN);
            MicroAPI::Mul(colSumSqReg, colSumReg, colSumReg, maskN);
            MicroAPI::Duplicate(reduceSumReg, U(0), maskN);

            for (uint16_t k = 0; k < static_cast<uint16_t>(n_); k++) {
                MicroAPI::LoadAlign(rowNormReg, rowNormBase + bsOff + k * nAlign_);
                MicroAPI::LoadAlign(skGradReg, skGradPtr + bsOff + k * nAlign_);

                MicroAPI::Div(tmpDivReg, skGradReg, colSumReg, maskN);
                MicroAPI::StoreAlign(tmpDivPtr + bsOff + k * nAlign_, tmpDivReg, maskN);

                MicroAPI::Mul(tmpMulReg, rowNormReg, skGradReg, maskN);
                MicroAPI::Div(tmpDiv2Reg, tmpMulReg, colSumSqReg, maskN);
                MicroAPI::Add(reduceSumReg, reduceSumReg, tmpDiv2Reg, maskN);
            }
            MicroAPI::StoreAlign(tmpPtr + bsOffN, reduceSumReg, maskN);
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    }
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::SinkhornGradStage3_4(
    __local_mem__ U *rowSumBase, __local_mem__ U *rowNormBase,
    __local_mem__ U *skGradPtr, __local_mem__ U *tmpDivPtr, __local_mem__ U *tmpDiv3Ptr,
    __local_mem__ U *tmpPtr, int64_t lenBs)
{
    uint16_t bsLenLoop = static_cast<uint16_t>(lenBs);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<U> tmpDivReg, rowNormReg, tmpSubReg, gradXRowNormedReg;
        MicroAPI::RegTensor<U> tmpDiv3Reg, tmpDiv4Reg, rowSumBcastReg;
        MicroAPI::RegTensor<U> colReduceReg, colReduceBcastReg, skGradReg, zeroReg;
        MicroAPI::MaskReg maskN = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg allMask = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
        uint32_t maskLenN = static_cast<uint32_t>(n_);

        MicroAPI::Duplicate(zeroReg, U(0), allMask);
        maskN = MicroAPI::UpdateMask<U>(maskLenN);

        for (uint16_t j = 0; j < bsLenLoop; j++) {
            int64_t bsOff = j * nnAlign_;
            int64_t bsOffN = j * nAlign_;

            MicroAPI::LoadAlign(gradXRowNormedReg, tmpPtr + bsOffN);

            for (uint16_t k = 0; k < static_cast<uint16_t>(n_); k++) {
                U rowSumVal = *(rowSumBase + bsOffN + k);
                MicroAPI::LoadAlign(tmpDivReg, tmpDivPtr + bsOff + k * nAlign_);
                MicroAPI::LoadAlign(rowNormReg, rowNormBase + bsOff + k * nAlign_);

                MicroAPI::Sub(tmpSubReg, tmpDivReg, gradXRowNormedReg, maskN);
                MicroAPI::Duplicate(rowSumBcastReg, rowSumVal, maskN);
                MicroAPI::Div(tmpDiv3Reg, tmpSubReg, rowSumBcastReg, maskN);
                MicroAPI::StoreAlign(tmpDiv3Ptr + bsOff + k * nAlign_, tmpDiv3Reg, maskN);

                MicroAPI::Mul(tmpDiv4Reg, tmpSubReg, rowNormReg, maskN);
                MicroAPI::Div(tmpDiv4Reg, tmpDiv4Reg, rowSumBcastReg, maskN);
                MicroAPI::Select(tmpDiv4Reg, tmpDiv4Reg, zeroReg, maskN);
                MicroAPI::Reduce<MicroAPI::ReduceType::SUM, U, U>(colReduceReg, tmpDiv4Reg, allMask);
                MicroAPI::Duplicate(colReduceBcastReg, colReduceReg, maskN);

                MicroAPI::Sub(skGradReg, tmpDiv3Reg, colReduceBcastReg, maskN);
                MicroAPI::StoreAlign(skGradPtr + bsOff + k * nAlign_, skGradReg, maskN);
            }
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    }
}

// ================================================================
// ExpGradSimdVf — nAlign_ aligned version
// ================================================================
template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::ExpGradSimdVf(
    __local_mem__ U *expGradPtr, __local_mem__ U *skGradPtr, __local_mem__ U *zResPtr,
    int64_t idxBs, int64_t lenBs)
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

    uint16_t bsLenLoop = static_cast<uint16_t>(lenBs);
    uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(U);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<U> zResReg, zResMaxReg, zResMaxIdxReg, zResMaxBcastReg, zResMaxIdxBcastReg;
        MicroAPI::RegTensor<U> tmpSubReg, tmpExpReg, skGradReg, tmpMulReg, tmpMul2Reg;
        MicroAPI::RegTensor<U> sumAllReg, sumAllBcastReg, isMaxReg, expGradReg;
        MicroAPI::RegTensor<U> idxReg, idxSelectReg, zeroReg, oneReg, negInfReg;
        MicroAPI::MaskReg maskN = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg allMask = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg cmpMask, maxMask;
        uint32_t maskLenN = static_cast<uint32_t>(n_);

        MicroAPI::Duplicate(zeroReg, U(0), allMask);
        MicroAPI::Duplicate(oneReg, U(1), allMask);
        MicroAPI::Duplicate(negInfReg, U(-3.402823466e+38F), allMask);
        maskN = MicroAPI::UpdateMask<U>(maskLenN);

        for (uint16_t i = 0; i < bsLenLoop; i++) {
            int64_t bsOffZ = i * kAlign_;
            int64_t bsOffSk = i * nnAlign_;
            int64_t bsOut = i * kAlign_;

            // Step 1: Find max and isMax per row
            for (uint16_t j = 0; j < static_cast<uint16_t>(n_); j++) {
                int64_t rowOffZ = bsOffZ + j * nAlign_;
                int64_t rowOffSk = bsOffSk + j * nAlign_;
                int64_t rowOut = bsOut + j * nAlign_;
                MicroAPI::LoadAlign(zResReg, zResPtr + rowOffZ);
                MicroAPI::Select(zResReg, zResReg, negInfReg, maskN);

                MicroAPI::Reduce<MicroAPI::ReduceType::MAX, U, U>(zResMaxReg, zResReg, allMask);
                MicroAPI::Duplicate(zResMaxBcastReg, zResMaxReg, maskN);

                MicroAPI::Compare<U>(maxMask, zResMaxBcastReg, zResReg, maskN);
                MicroAPI::Arange(idxReg, 0);
                MicroAPI::Select(idxSelectReg, idxReg, negInfReg, maxMask);
                MicroAPI::Reduce<MicroAPI::ReduceType::MAX, U, U>(zResMaxIdxReg, idxSelectReg, maskN);
                MicroAPI::Select(isMaxReg, oneReg, zeroReg, maxMask);

                MicroAPI::StoreAlign(zResMaxPtr + rowOffSk, zResMaxBcastReg, maskN);
                MicroAPI::StoreAlign(isMaxPtr + rowOffSk, isMaxReg, maskN);
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

            // Step 2: exp(z - max) * skGrad → tmpMul
            for (uint16_t j = 0; j < static_cast<uint16_t>(n_); j++) {
                int64_t rowOffZ = bsOffZ + j * nAlign_;
                int64_t rowOffSk = bsOffSk + j * nAlign_;
                MicroAPI::LoadAlign(zResReg, zResPtr + rowOffZ);
                MicroAPI::LoadAlign(zResMaxReg, zResMaxPtr + rowOffSk);
                MicroAPI::LoadAlign(skGradReg, skGradPtr + rowOffSk);
                MicroAPI::Sub(tmpSubReg, zResReg, zResMaxReg, maskN);
                MicroAPI::Exp(tmpExpReg, tmpSubReg, maskN);
                MicroAPI::Mul(tmpMulReg, tmpExpReg, skGradReg, maskN);
                MicroAPI::StoreAlign(tmpMulPtr + rowOffSk, tmpMulReg, maskN);
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

            // Step 3: expGrad = tmpMul - isMax * sum(tmpMul)
            for (uint16_t j = 0; j < static_cast<uint16_t>(n_); j++) {
                int64_t rowOffSk = bsOffSk + j * nAlign_;
                int64_t rowOut = bsOut + j * nAlign_;
                MicroAPI::LoadAlign(tmpMulReg, tmpMulPtr + rowOffSk);
                MicroAPI::Select(tmpMulReg, tmpMulReg, zeroReg, maskN);
                MicroAPI::Reduce<MicroAPI::ReduceType::SUM, U, U>(sumAllReg, tmpMulReg, allMask);
                MicroAPI::Duplicate(sumAllBcastReg, sumAllReg, maskN);

                MicroAPI::LoadAlign(isMaxReg, isMaxPtr + rowOffSk);
                MicroAPI::Mul(tmpMul2Reg, isMaxReg, sumAllBcastReg, maskN);
                MicroAPI::Sub(expGradReg, tmpMulReg, tmpMul2Reg, maskN);

                MicroAPI::StoreAlign(expGradPtr + rowOut, expGradReg, maskN);
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
    int64_t ofsBs, int64_t lenBs, U alphaScalar, int64_t nRows,
    int64_t ubSegOffset, int64_t wsSegOffset,
    __local_mem__ U *hcPtr, __local_mem__ U *irPtr)
{
    auto gradZUb = gradZBuf_.template Get<U>();
    auto gradCalc2 = gradCalcBuf2_.template Get<U>();
    auto gradInvRmsUb = gradInvRmsBuf_.template Get<U>();

    __local_mem__ U *gradZPtr = reinterpret_cast<__local_mem__ U *>(gradZUb.GetPhyAddr());
    __local_mem__ U *tmpPtr = reinterpret_cast<__local_mem__ U *>(gradCalc2.GetPhyAddr());
    __local_mem__ U *gradInvRmsPtr = reinterpret_cast<__local_mem__ U *>(gradInvRmsUb.GetPhyAddr());

    uint64_t nAlign = nAlign_;
    uint64_t rowStride = kAlign_;

    for (int64_t loopBs = 0; loopBs < lenBs; loopBs++) {
        U invRms = irPtr[loopBs];
        U invRmsAlpha = invRms * alphaScalar;

        __local_mem__ U *gradZBs = gradZPtr + loopBs * rowStride + ubSegOffset;
        __local_mem__ U *hcBs = hcPtr + loopBs * rowStride + ubSegOffset;

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<U> gradZReg, gradNormOutReg, gradHcBnReg, hcReg, mulReg;
            MicroAPI::RegTensor<U> gradInvRmsSumReg, reduceReg;
            MicroAPI::MaskReg maskN = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg oneMask = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::VL1>();
            uint32_t maskLenN = static_cast<uint32_t>(n_);

            MicroAPI::Duplicate(gradInvRmsSumReg, U(0), oneMask);
            maskN = MicroAPI::UpdateMask<U>(maskLenN);

            for (uint16_t row = 0; row < static_cast<uint16_t>(nRows); row++) {
                __local_mem__ U *gzRow = gradZBs + row * nAlign;
                __local_mem__ U *hcRow = hcBs + row * nAlign;
                __local_mem__ U *dstRow = tmpPtr + row * nAlign;

                MicroAPI::LoadAlign(gradZReg, gzRow);
                MicroAPI::Muls(gradNormOutReg, gradZReg, alphaScalar, maskN);
                MicroAPI::Muls(gradHcBnReg, gradZReg, invRmsAlpha, maskN);
                MicroAPI::Store(dstRow, gradHcBnReg, static_cast<uint32_t>(n_));

                MicroAPI::LoadAlign(hcReg, hcRow);
                MicroAPI::Mul(mulReg, hcReg, gradNormOutReg, maskN);
                MicroAPI::Reduce<MicroAPI::ReduceType::SUM, U, U>(reduceReg, mulReg, maskN);
                MicroAPI::Add(gradInvRmsSumReg, gradInvRmsSumReg, reduceReg, oneMask);
            }

            MicroAPI::RegTensor<U> oldValReg;
            MicroAPI::Load(oldValReg, gradInvRmsPtr + loopBs);
            MicroAPI::Add(gradInvRmsSumReg, gradInvRmsSumReg, oldValReg, oneMask);
            MicroAPI::Store(gradInvRmsPtr + loopBs, gradInvRmsSumReg, 1);
        }

        int64_t wsIdx = (ofsBs + loopBs) * kDim_ + wsSegOffset;
        if (wsSegOffset == 2 * n_ && loopBs == lenBs - 1) {
            AscendC::printf("[DUMP] 232 gradHcBn(res, last bs) nRows=%d\n", nRows);
            AscendC::DumpTensor(gradCalc2, 232, nRows * nAlign + 1);
        }
        CopyOutGradHcRows(wsIdx, nRows);
    }
}

template <typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::CopyOutGradHcRows(
    int64_t wsIdx, int64_t nRows)
{
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    for (uint16_t row = 0; row < static_cast<uint16_t>(nRows); row++) {
        CopyOutBuf(gradCalcBuf2_, gradHcBeforeNormWs_[wsIdx + row * n_], 1, n_, nAlign_, 0, row * nAlign_);
    }
    event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
}

#endif
