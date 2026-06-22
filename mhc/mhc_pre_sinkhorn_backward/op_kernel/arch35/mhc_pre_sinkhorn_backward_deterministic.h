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
* \brief
*/
#ifndef MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_H
#define MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_H


#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "mhc_pre_sinkhorn_backward_deterministic_simd_vf.h"

using namespace AscendC;

constexpr uint8_t NUMONE = 1;
constexpr uint8_t NUMTWO = 2;
constexpr int64_t AIV_AIC_RATE = 2;

template<typename X_T, typename GRADHIN_T, typename U>
class MhcPreSinkhornBackwardDeterministic {
public:
    __aicore__ inline MhcPreSinkhornBackwardDeterministic() = default;
    __aicore__ inline void Process(void);
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR phi, GM_ADDR hPre, GM_ADDR gradHin, GM_ADDR gradHPost,
        GM_ADDR gradHRes, GM_ADDR alpha, GM_ADDR bias, GM_ADDR hcBeforeNorm, GM_ADDR invRms, GM_ADDR sumOut,
        GM_ADDR normOut, GM_ADDR gradX, GM_ADDR gradPhi, GM_ADDR gradAlpha, GM_ADDR gradBias,
        GM_ADDR workspace, const MhcPreSinkhornBackwardArch35DeterminiticTilingData* tilingData, TPipe* pipe);

private:
    __aicore__ inline void GetAlphaAndBias(void);
    __aicore__ inline void ComputeFirst(uint64_t bsIdx, uint64_t bsLen);
    __aicore__ inline void CopyInHpreAndGradHPost(const GlobalTensor<U>& gmTensor, uint64_t bsLen);
    __aicore__ inline void CopyInXandGradHin(uint64_t xGmOffset, uint64_t gradHinGmOffset, uint64_t bsLen, uint64_t cLen);
    __aicore__ inline void ComputeGradHpreAndXFromIn(LocalTensor<U>& hPreLocal, LocalTensor<U>& gradHPreLocal, const Process1Info& info);
    __aicore__ inline void ComputeSigmoidGrad(uint64_t bsLen, uint64_t colLen, uint8_t num);
    __aicore__ inline void ComputeGradAlphaAndBias(uint64_t bsLen, uint64_t colLen);
    __aicore__ inline void ComputeZAndNormOutForward(const LocalTensor<U>& invRmsLocal, uint64_t bsLen, 
                                        uint64_t colLen, uint64_t oft, const U& alpha, uint64_t biasOft);
    __aicore__ inline void CopyInHcBeforeNorm(uint64_t bsLen, uint64_t colLen, uint64_t gmOft);
    __aicore__ inline void CopyInInvRms(uint64_t bsLen, uint64_t gmOft);
    __aicore__ inline void CopyOutAlphaAndBias(uint64_t gmAlphaOft, uint64_t gmBiasOft);
    __aicore__ inline void AddAlphaAndBias(void);
    __aicore__ inline void CopyInAlphaAndBias(int64_t wsBiasOft, int64_t wsAlphaOft, int64_t biasLen);
    __aicore__ inline void CopyOutAlphaAndBiasSum(int64_t gmBiasOft, int64_t gmAlphaOft, int64_t biasLen);
    __aicore__ inline void ProcessGradXFromRms(int64_t bsIdx);
    __aicore__ inline void ComputeFinal(void);
    __aicore__ inline void ComputeGradXMatmul(const int32_t taskOffset, const int32_t mm1M);
    __aicore__ inline void ComputeGradPhiMatmul(const int32_t taskOffset, const int32_t mm2M);
    __aicore__ inline void SinkhornGradSimdVf(LocalTensor<float> skGradLocal, uint64_t bsIdx, uint64_t bsLen);
    __aicore__ inline void ExpGradSimdVf(LocalTensor<float> skGrad, LocalTensor<float> expGrad, LocalTensor<float> zResLocal, uint64_t bsIdx, uint64_t bsLen);

    
private:


    using A0Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, U>;
    using A1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, U, true>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, U>;
    using C1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, X_T>;
    using C2Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, U>;

    matmul::MatmulImpl<A0Type, BType, C1Type, C1Type, MHC_PRE_GRAD_MM1_CFG> mm1_;
    matmul::MatmulImpl<A1Type, BType, C2Type, C2Type, MHC_PRE_GRAD_MM2_CFG> mm2_;

    GlobalTensor<X_T>  xGm_;                //输入
    GlobalTensor<GRADHIN_T>  gradHinGm_;
    GlobalTensor<U>  gradHPostGm_;
    GlobalTensor<U>  gradHResGm_;
    GlobalTensor<U>  phiGm_;
    GlobalTensor<U>  alphaGm_;
    GlobalTensor<U>  biasGm_;
    GlobalTensor<U>  hPreGm_;
    GlobalTensor<U>  hcBeforeNormGm_;
    GlobalTensor<U>  invRmsGm_;
    GlobalTensor<U>  sumOutGm_;
    GlobalTensor<U>  normOutGm_;

    GlobalTensor<U>  xFp32Ws_;
    GlobalTensor<U>  gradAlphaWs_;
    GlobalTensor<U>  gradBiasWs_;
    GlobalTensor<U>  gradXFromHinWs_;

    // GlobalTensor<U>  xFp32Ws_;                  //中间结果
    GlobalTensor<U>  gradHcBeforeNormWs_;
    // GlobalTensor<U>  gradXWs_;
    GlobalTensor<U>  gradNormOutWs_;
    // GlobalTensor<U>  gradXFromHinWs_;
    // GlobalTensor<U>  gradAlphaWs_;
    // GlobalTensor<U>  gradBiasWs_;

    GlobalTensor<X_T>  gradXGm_;            //输出
    GlobalTensor<U>  gradPhiGm_;
    GlobalTensor<U>  gradAlphaGm_;
    GlobalTensor<U>  gradBiasGm_;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> xQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> hPreAndGradHPostQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> gradHinQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> hcBeforeNormQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> invRmsQue_;

    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> xFp32Que_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> gradXFromHinQue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> gradAlphaAndBiasQue_;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> gradAlphaQue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> gradAlphaSumQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DOUBLE_BUFFER> gradBiasQue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> gradXFromHinAndAlphaAndBiasQue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> gradHResQueue_;

    TBuf<TPosition::VECCALC> gradHPreBuf_;
    TBuf<TPosition::VECCALC> zBuf_;
    TBuf<TPosition::VECCALC> normOutForwardBuf_;
    TBuf<TPosition::VECCALC> gradZBuf_;
    TBuf<TPosition::VECCALC> biasBuf_;
    TBuf<TPosition::VECCALC> gradCalcBuf1_;
    TBuf<TPosition::VECCALC> gradCalcBuf2_;

    int64_t aivNum_ = 0;
    int64_t usedAivNum_ = 0;
    // hbj
    TBuf<TPosition::VECCALC> tmpDiv3Buf_;
    TBuf<TPosition::VECCALC> tmpMulBuf_;
    TBuf<TPosition::VECCALC> sumOutBuf_;
    TBuf<TPosition::VECCALC> normOutBuf_;
    TBuf<TPosition::VECCALC> zResBuf_;
    TBuf<TPosition::VECCALC> zResMaxBuf_;
    TBuf<TPosition::VECCALC> zResMaxIdxBuf_;
    TBuf<TPosition::VECCALC> gradXRowNormedBuf_;
    TBuf<TPosition::VECCALC> skGradBuf_;
    TBuf<TPosition::VECCALC> expGradBuf_;

    int64_t batchSize_ = 0;
    int64_t seqLength_ = 0;
    uint16_t n_ = 0;
    uint16_t nn_ = 0;
    int64_t c_ = 0;
    int64_t curCoreProcessbsTask_ = 0;
    int64_t bsLoopDataLen_ = 0;
    int64_t cLoopDataLen_ = 0;
    int64_t skIterCount_ = 0;
    float eps_ = 0;
    int64_t k_ = 0;
    int64_t blockIdx_ = 0;
    int64_t mm1K_ = 0;
    int64_t mm1M_ = 0;
    int64_t mm1N_ = 0;
    int64_t mm2K_ = 0;
    int64_t mm2M_ = 0;
    int64_t mm2N_ = 0;
    int64_t blockSize_ = 0;
    int64_t nAlignFp32_ = 0;
    int64_t nnAlignFp32_ = 0;
    int64_t cLenAlignFp32_ = 0;
    int64_t cLenAlignFp16_ = 0;
    int64_t bsnAlignFp32_ = 0;
    int64_t bsAlignFp32_ = 0;
    int64_t bsnnAlignFp32_ = 0;
    int64_t gradBiasOft_ = 0;
    int64_t uTypeSize_ = sizeof(U);
    int64_t xTypeSize_ = sizeof(X_T);
    int64_t gradHinTypeSize_ = sizeof(GRADHIN_T);
    U alphaPre_, alphaPost_, alphaRes_;
    TPipe* pipe_ = nullptr;
    const MhcPreSinkhornBackwardArch35DeterminiticTilingData* tilingData_ = nullptr;
    constexpr static uint32_t V_REG_SIZE = Ops::Base::GetVRegSize();
    int64_t avcNum_ = 0;
    int64_t bsTaskCount_ = 0;
    int64_t totalAicTasks_ = 0;
    int64_t calcBSPerTask_ = 0;
    int64_t totalAicTasksAligned_ = 0;

    // vector中计算RMSNormGrad的循环
    int64_t cLoopsPerBS_ = 0;
    int64_t cCountPerBS_ = 0;
    int64_t cTailPerBS_ = 0;

    // vector中最后计算gradX的循环
    int64_t cBlockLoops_ = 0;
    int64_t cBlockTailLoops_ = 0;
    int64_t cBlockCount_ = 0;
    int64_t cBlockTailCount_ = 0;
    int64_t blockCalcNum_ = 0;
};

template<typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::Init(GM_ADDR x, GM_ADDR phi, GM_ADDR hPre, GM_ADDR gradHin, GM_ADDR gradHPost,
        GM_ADDR gradHRes, GM_ADDR alpha, GM_ADDR bias, GM_ADDR hcBeforeNorm, GM_ADDR invRms, GM_ADDR sumOut,
        GM_ADDR normOut, GM_ADDR gradX, GM_ADDR gradPhi, GM_ADDR gradAlpha, GM_ADDR gradBias,
        GM_ADDR workspace, const MhcPreSinkhornBackwardArch35DeterminiticTilingData* tilingData, TPipe* pipe)
{
    tilingData_ = tilingData;
    pipe_ = pipe;
    batchSize_ = tilingData->batchSize;
    seqLength_ = tilingData->seqLength;
    n_ = static_cast<uint16_t>(tilingData->n);
    nn_ = n_ * n_;
    c_ = tilingData->c;
    cLoopDataLen_ = tilingData->cLoopDataLen;
    bsLoopDataLen_ = tilingData->bsLoopDataLen;
    skIterCount_ = tilingData->skIterCount;
    eps_ = tilingData->eps;
    k_ = n_ * n_ + 2 * n_;
    aivNum_ = tilingData->aivNum;
    usedAivNum_ = tilingData->usedAivNum;
    blockIdx_ = GetBlockIdx();
    blockSize_ = Ops::Base::GetUbBlockSize();
    curCoreProcessbsTask_ = (blockIdx_ == usedAivNum_ - 1) ? tilingData->tailBsTaskCount : tilingData->bsTaskCount;
    nAlignFp32_ = Ops::Base::CeilAlign(int64_t(n_ * uTypeSize_), blockSize_) / uTypeSize_;
    cLenAlignFp32_ = Ops::Base::CeilAlign(int64_t(cLoopDataLen_ * uTypeSize_), blockSize_) / uTypeSize_;
    cLenAlignFp16_ = Ops::Base::CeilAlign(int64_t(cLoopDataLen_ * xTypeSize_), blockSize_) / xTypeSize_;
    nnAlignFp32_ = Ops::Base::CeilAlign(int64_t(n_ * n_ * uTypeSize_), blockSize_) / uTypeSize_;
    bsnAlignFp32_ = Ops::Base::CeilAlign(int64_t(n_ * bsLoopDataLen_ * uTypeSize_), blockSize_) / uTypeSize_;
    bsAlignFp32_ = Ops::Base::CeilAlign(int64_t(bsLoopDataLen_ * uTypeSize_), blockSize_) / uTypeSize_;
    bsnnAlignFp32_ = Ops::Base::CeilAlign(int64_t(bsLoopDataLen_ * n_ * n_ * uTypeSize_), blockSize_) / uTypeSize_;
    gradBiasOft_ = blockSize_ / uTypeSize_;

    gradPhiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(gradPhi));
    // 每个BS中nc循环
    cLoopsPerBS_ = tilingData->cLoopsPerBS;
    cCountPerBS_ = tilingData->cCountPerBS;
    cTailPerBS_ = tilingData->cTailPerBS;
    // BSNC每核循环
    cBlockLoops_ = (blockIdx_ == tilingData->aivNum - 1) ? tilingData->cBlockTailLoops : tilingData->cBlockLoops;
    cBlockCount_ = tilingData->cBlockCount;
    cBlockTailCount_ = (blockIdx_ == tilingData->aivNum - 1) ? tilingData->cBlockTailTailCount : tilingData->cBlockTailCount;
    blockCalcNum_ = blockIdx_ * cBlockCount_ + tilingData->cBlockTailCount;

    phiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(phi));
    constexpr int64_t ALIGN_256 = 256;
 	constexpr int64_t FP32_BYTE_SIZE = sizeof(U);
    int64_t wsGradAlphaOffset_ = AlignUp(bsLoopDataLen_ * n_ * c_ * FP32_BYTE_SIZE, ALIGN_256) / FP32_BYTE_SIZE;
    int64_t wsGradBiasOffset_ = wsGradAlphaOffset_ + AlignUp(64 * FP32_BYTE_SIZE, ALIGN_256) / FP32_BYTE_SIZE;
    int64_t wsGradXFromHinOffset_ = wsGradBiasOffset_ + AlignUp(n_ * FP32_BYTE_SIZE, ALIGN_256) / FP32_BYTE_SIZE;
    // gradXWs_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(workspace));       //后需添加偏移计算
    // xFp32Ws_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(workspace));
    gradHcBeforeNormWs_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(workspace));
    gradNormOutWs_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(workspace));

    xFp32Ws_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(workspace));
    gradAlphaWs_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(workspace) + wsGradAlphaOffset_);
    gradBiasWs_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(workspace) + wsGradBiasOffset_);
    gradXFromHinWs_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(workspace) + wsGradXFromHinOffset_);

    if ASCEND_IS_AIV {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_T*>(x));
        gradHinGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GRADHIN_T*>(gradHin));
        gradHPostGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(gradHPost));
        gradHResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(gradHRes));
        alphaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(alpha));
        biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(bias));
        hPreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(hPre));
        hcBeforeNormGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(hcBeforeNorm));
        invRmsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(invRms));
        sumOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(sumOut));
        normOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(normOut));
        gradXGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_T*>(gradX));
        gradAlphaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(gradAlpha));
        gradBiasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(gradBias));
        
        // Init UB
        //VECIN
        pipe_->InitBuffer(xQue_, DOUBLE_BUFFER, cLenAlignFp16_ * n_ * bsLoopDataLen_ * xTypeSize_);
        pipe_->InitBuffer(hPreAndGradHPostQue_, DOUBLE_BUFFER, bsnAlignFp32_ * uTypeSize_);
        pipe_->InitBuffer(gradHinQue_, DOUBLE_BUFFER, cLenAlignFp16_ * bsLoopDataLen_ * gradHinTypeSize_);
        pipe_->InitBuffer(hcBeforeNormQue_, DOUBLE_BUFFER, bsLoopDataLen_ * (2 * nAlignFp32_ + nnAlignFp32_) * uTypeSize_);
        pipe_->InitBuffer(invRmsQue_, DOUBLE_BUFFER, bsAlignFp32_ * uTypeSize_);
        pipe_->InitBuffer(gradHResQueue_, DOUBLE_BUFFER, bsLoopDataLen_ * nnAlignFp32_ * FP32_BYTE_SIZE);
        //VECOUT
        pipe_->InitBuffer(xFp32Que_, DOUBLE_BUFFER, cLenAlignFp32_ * n_ * bsLoopDataLen_ * uTypeSize_);
        pipe_->InitBuffer(gradXFromHinQue_, DOUBLE_BUFFER, cLenAlignFp32_ * n_ * bsLoopDataLen_ * uTypeSize_);
        pipe_->InitBuffer(gradAlphaAndBiasQue_, DOUBLE_BUFFER, nnAlignFp32_ * uTypeSize_ + blockSize_);
        
        //BUF
        pipe_->InitBuffer(gradHPreBuf_, bsnAlignFp32_ * uTypeSize_);
        pipe_->InitBuffer(gradZBuf_, bsnAlignFp32_ * uTypeSize_);
        pipe_->InitBuffer(biasBuf_, nnAlignFp32_ * uTypeSize_);        
        pipe_->InitBuffer(normOutForwardBuf_, bsnnAlignFp32_ * uTypeSize_);
        pipe_->InitBuffer(zBuf_, bsnnAlignFp32_ * uTypeSize_);

        pipe_->InitBuffer(gradCalcBuf1_, Ops::Base::CeilDiv(int64_t(k_ * uTypeSize_), blockSize_));
        pipe_->InitBuffer(gradCalcBuf2_, Ops::Base::CeilDiv(int64_t(k_ * uTypeSize_), blockSize_));
    }

    if ASCEND_IS_AIV {
        pipe_->InitBuffer(gradHPreBuf_, bsnAlignFp32_ * uTypeSize_);
        pipe_->InitBuffer(zBuf_, bskAlignFp32_ * uTypeSize_);
        pipe_->InitBuffer(normOutForwardBuf_, bskAlignFp32_ * uTypeSize_);
        pipe_->InitBuffer(gradZBuf_, bsnAlignFp32_ * uTypeSize_);
        pipe_->InitBuffer(biasBuf_, bsAlignFp32_ * uTypeSize_);

        pipe_->InitBuffer(gradCalcBuf1_, Ops::Base::CeilDiv(int64_t(k_ * uTypeSize_), blockSize_));
        pipe_->InitBuffer(gradCalcBuf2_, Ops::Base::CeilDiv(int64_t(k_ * uTypeSize_), blockSize_));
        avcNum_ = tilingData->aivNum / AIV_AIC_RATE;
        calcBSPerTask_ = tilingData->bsTaskCount * AIV_AIC_RATE;
        totalAicTasks_ = batchSize_ * seqLength_;
        totalAicTasksAligned_ = Ops::Base::CeilAlign(totalAicTasks_, avcNum_ * calcBSPerTask_);
        mm1K_ = n_ * n_ + 2 * n_;
        mm1M_ = calcBSPerTask_;
        mm1N_ = n_ * c_;
        mm2K_ = batchSize_ * seqLength_;
        mm2M_ = 1;
        mm2N_ = n_ * c_;
        // hbj
        pipe_->InitBuffer(tmpDiv3Buf_, 2 * skIterCount_ * bsLoopDataLen_ * nAlignFp32_ * FP32_BYTE_SIZE);
        pipe_->InitBuffer(tmpMulBuf_, 2 * skIterCount_ * bsLoopDataLen_ * nAlignFp32_ * FP32_BYTE_SIZE);
        pipe_->InitBuffer(sumOutBuf_, 2 * skIterCount_ * bsLoopDataLen_ * nAlignFp32_ * FP32_BYTE_SIZE);
        pipe_->InitBuffer(normOutBuf_, nnAlignFp32_ * bsLoopDataLen_ * FP32_BYTE_SIZE);
        pipe_->InitBuffer(zResBuf_, bsLoopDataLen_ * nnAlignFp32_ * FP32_BYTE_SIZE);
        pipe_->InitBuffer(zResMaxBuf_, bsLoopDataLen_ * nnAlignFp32_ * FP32_BYTE_SIZE);
        pipe_->InitBuffer(zResMaxIdxBuf_, bsLoopDataLen_ * nnAlignFp32_ * FP32_BYTE_SIZE);
        pipe_->InitBuffer(gradXRowNormedBuf_, bsLoopDataLen_ * nnAlignFp32_ * FP32_BYTE_SIZE);
        pipe_->InitBuffer(skGradBuf_, bsLoopDataLen_ * nnAlignFp32_ * FP32_BYTE_SIZE);
        pipe_->InitBuffer(expGradBuf_, bsLoopDataLen_ * nnAlignFp32_ * FP32_BYTE_SIZE);
    }
}

template<typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::CopyInXandGradHin(uint64_t xGmOffset, uint64_t gradHinGmOffset, 
                                                                                uint64_t bsLen, uint64_t cLen)
{
    LocalTensor<X_T> xLocal = xQue_.AllocTensor<X_T>();             //in
    LocalTensor<GRADHIN_T> gradHinLocal = gradHinQue_.AllocTensor<GRADHIN_T>();

    CopyIn(xLocal, xGm_[xGmOffset], bsLen * n_, cLen, 0, (c_ - cLen) * xTypeSize_);
    CopyIn(gradHinLocal, gradHinGm_[gradHinGmOffset], bsLen, cLen, 0, (c_ - cLen) * gradHinTypeSize_);

    xQue_.EnQue(xLocal);
    gradHinQue_.EnQue(gradHinLocal);
}

template<typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::CopyInHpreAndGradHPost(const GlobalTensor<U>& gMTensor, uint64_t bsLen)
{
    LocalTensor<U> hPreLocal = hPreAndGradHPostQue_.AllocTensor<U>();
    CopyIn(hPreLocal, gMTensor, 1, bsLen * n_, 0, 0);
    hPreAndGradHPostQue_.EnQue(hPreLocal);
}

template<typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::CopyInHcBeforeNorm(uint64_t bsLen, uint64_t colLen, uint64_t gmOft) 
{
    LocalTensor<U> hcBeforeNormLocal = hcBeforeNormQue_.AllocTensor<U>();
    CopyIn(hcBeforeNormLocal, hcBeforeNormGm_[gmOft], bsLen, colLen, 0, k_ - colLen);
    hcBeforeNormQue_.EnQue(hcBeforeNormLocal);
}

template<typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::CopyInInvRms(uint64_t bsLen, uint64_t gmOft) 
{
    LocalTensor<U> invRmsLocal = invRmsQue_.AllocTensor<U>();
    CopyIn(invRmsLocal, invRmsGm_[gmOft], 1, bsLen, 0, 0);
    invRmsQue_.EnQue(invRmsLocal);
}

template<typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::CopyOutAlphaAndBias(uint64_t gmAlphaOft, uint64_t gmBiasOft) 
{
    LocalTensor<U> gradAlphaLocal = gradAlphaAndBiasQue_.DeQue<U>();
    LocalTensor<U> gradBiasLocal = gradAlphaLocal[gradBiasOft_];
    SetAtomicAdd<U>();
    CopyOut(gradAlphaWs_[gmAlphaOft], gradAlphaLocal, 1, 1, 0, 0);
    CopyOut(gradBiasWs_[gmBiasOft], gradBiasLocal, 1, n_, 0, 0);
    SetAtomicNone();
    gradAlphaAndBiasQue_.FreeTensor(gradAlphaLocal);
}

template<typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::CopyInAlphaAndBias(int64_t wsBiasOft, int64_t wsAlphaOft, int64_t biasLen)
{
    LocalTensor<U> gradBiasLocal = gradBiasQue_.AllocTensor<U>();
    LocalTensor<U> gradAlphaLocal = gradAlphaQue_.AllocTensor<U>();
    CopyIn(gradBiasLocal, gradBiasWs_[wsBiasOft], usedAivNum_, biasLen, 0, 0);
    CopyIn(gradAlphaLocal, gradAlphaWs_[wsAlphaOft], 1, usedAivNum_, 0, 0);
    gradBiasQue_.EnQue(gradBiasLocal);
    gradAlphaQue_.EnQue(gradAlphaLocal);
}

template<typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::CopyOutAlphaAndBiasSum(int64_t gmBiasOft, int64_t gmAlphaOft, int64_t biasLen)
{
    LocalTensor<U> gradBiasSumLocal = gradBiasQue_.DeQue<U>();
    LocalTensor<U> gradAlphaSumLocal = gradAlphaSumQue_.DeQue<U>();
    CopyOut(gradAlphaGm_[gmAlphaOft], gradAlphaSumLocal, 1, 1, 0, 0);
    SetAtomicAdd<U>();
    CopyOut(gradBiasGm_[gmBiasOft], gradBiasSumLocal, usedAivNum_, biasLen, 0, 0);
    SetAtomicNone();
    gradBiasQue_.FreeTensor(gradBiasSumLocal);
    gradAlphaSumQue_.FreeTensor(gradAlphaSumLocal);
}

template<typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::AddAlphaAndBias(void)
{
    pipe_->Reset();

    auto aivNumAlign = Ops::Base::CeilAlign(int64_t(usedAivNum_ * uTypeSize_), blockSize_) / uTypeSize_;
    pipe_->InitBuffer(gradAlphaQue_, DOUBLE_BUFFER, aivNumAlign * uTypeSize_);
    pipe_->InitBuffer(gradAlphaSumQue_, DOUBLE_BUFFER, blockSize_);
    pipe_->InitBuffer(gradBiasQue_, DOUBLE_BUFFER, usedAivNum_ * nnAlignFp32_ * uTypeSize_);

    for (uint16_t i = 0; i < 3; i++) {
        auto biasLen = (i == 2) ? n_ * n_ : n_;
        CopyInAlphaAndBias(i * usedAivNum_ * n_, i * usedAivNum_, biasLen);
        LocalTensor<U> gradAlphaSumLocal = gradAlphaSumQue_.AllocTensor<U>();
        LocalTensor<U> gradAlphaLocal = gradAlphaQue_.DeQue<U>();
        
        __local_mem__ U* gradAlphaAddr = (__ubuf__ U*)gradAlphaLocal.GetPhyAddr();
        __local_mem__ U* gradAlphaSumAddr = (__ubuf__ U*)gradAlphaSumLocal.GetPhyAddr();
        uint32_t vfLen = V_REG_SIZE / sizeof(U);
        uint16_t loopCnt = (usedAivNum_ + vfLen - 1) / vfLen;
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<U> gradAlphaReg;
            AscendC::MicroAPI::RegTensor<U> gradAlphaSumTempReg;
            AscendC::MicroAPI::RegTensor<U> gradAlphaSumReg;
            AscendC::MicroAPI::MaskReg maskReg;
            AscendC::MicroAPI::MaskReg allMaskReg = AscendC::MicroAPI::CreateMask<U, AscendC::MicroAPI::MaskPattern::ALL>();
            uint32_t maskLen = static_cast<uint32_t>(usedAivNum_);
            MicroAPI::Duplicate(gradAlphaSumReg, U(0), allMaskReg);
            for (uint16_t i = 0; i < loopCnt; i++) {
                maskReg = AscendC::MicroAPI::UpdateMask<U>(maskLen);
                AscendC::MicroAPI::AddrReg gradAlphaOfst = AscendC::MicroAPI::CreateAddrReg<U>(i, vfLen);
                AscendC::MicroAPI::DataCopy(gradAlphaReg, gradAlphaAddr, gradAlphaOfst);
                AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::SUM, U, U>(gradAlphaSumTempReg, gradAlphaReg, maskReg); //reduce_sum
                AscendC::MicroAPI::Add(gradAlphaSumReg, gradAlphaSumReg, gradAlphaSumTempReg, maskReg);
            }
            AscendC::MicroAPI::Store(gradAlphaSumAddr, gradAlphaSumReg, 1);
        }
        gradAlphaSumQue_.EnQue(gradAlphaSumLocal);
        CopyOutAlphaAndBiasSum(i * n_, i, biasLen);
    }
}

template<typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::ComputeFirst(uint64_t bsIdx, uint64_t bsLen)
{
    LocalTensor<U> gradHPreLocal = gradHPreBuf_.Get<U>();   //buf 中间结果
    Duplicate(gradHPreLocal, float(0), bsLen * n_);

    uint64_t cLoopDataLen = cLoopDataLen_;
    uint64_t cLoopNum = Ops::Base::CeilDiv(c_, cLoopDataLen_);
    uint64_t cTailDataLen = c_ - cLoopDataLen * (cLoopNum - 1);
    uint64_t bsOffset = blockIdx_ * tilingData_->bsTaskCount + bsIdx * bsLoopDataLen_;
    uint64_t hPreOffset = bsOffset * n_;
    uint64_t gradHPostOffset = bsOffset * n_;
    uint64_t xOffset = bsOffset * n_ * c_;
    uint64_t gradHinOffset = bsOffset * c_;
    uint64_t hcBeforeNormOffset = bsOffset * k_;
    uint64_t invRmsOffset = bsOffset;
    uint64_t gradXFromHinOffset = bsOffset * n_ * c_;
    CopyInHpreAndGradHPost(hPreGm_[hPreOffset], bsLen);
    LocalTensor<U> hPreLocal = hPreAndGradHPostQue_.DeQue<U>();

    Process1Info info;
    info.bsLen = bsLen;
    info.n = n_;
    info.vRegSize = V_REG_SIZE;
    for (uint64_t cIdx = 0; cIdx < cLoopNum; cIdx++) {
        auto cLen = cIdx == cLoopNum - 1 ? cTailDataLen : cLoopDataLen;
        int64_t cLenXTAlign = Ops::Base::CeilAlign(int64_t(cLen * xTypeSize_), blockSize_) / xTypeSize_;
        int64_t cLenUAlign = Ops::Base::CeilAlign(int64_t(cLen * uTypeSize_), blockSize_) / uTypeSize_;
        int64_t cLenGRADHINTAlign = Ops::Base::CeilAlign(int64_t(cLen * gradHinTypeSize_), blockSize_) / gradHinTypeSize_;
        info.cLen = cLen;
        info.cLenXTAlign = cLenXTAlign;
        info.cLenUAlign = cLenUAlign;
        info.cLenGRADHINTAlign = cLenGRADHINTAlign;
        uint64_t xStartOft = xOffset + cIdx * cLoopDataLen;
        uint64_t gradHinStartOft = gradHinOffset + cIdx * cLoopDataLen;
        uint64_t gradXFromHinStartOft = gradXFromHinOffset + cIdx * cLoopDataLen;
        CopyInXandGradHin(xStartOft, gradHinStartOft, bsLen, cLen);
        ComputeGradHpreAndXFromIn(hPreLocal, gradHPreLocal, info);
        LocalTensor<U> xFp32Local = xFp32Que_.DeQue<U>();     //out
        LocalTensor<U> gradXFromHinLocal = gradXFromHinQue_.DeQue<U>();       //out        
        CopyOut(xFp32Ws_[xStartOft], xFp32Local, bsLen * n_, cLen, 0, (c_ - cLoopDataLen) * uTypeSize_);       //搬运x_f32到workspace
        CopyOut(gradXFromHinWs_[gradXFromHinStartOft], gradXFromHinLocal, bsLen * n_, cLen, 0, (c_ - cLoopDataLen) * uTypeSize_);       //搬运grad_x_from_hin到workspace
        xFp32Que_.FreeTensor(xFp32Local);
        gradXFromHinQue_.FreeTensor(gradXFromHinLocal);
    }
    hPreAndGradHPostQue_.FreeTensor(hPreLocal);
    CopyInInvRms(bsLen, invRmsOffset);
    LocalTensor<U> invRmsLocal = invRmsQue_.DeQue<U>();
    ComputeZAndNormOutForward(invRmsLocal, bsLen, n_, hcBeforeNormOffset, alphaPre_, 0);
    ComputeSigmoidGrad(bsLen, n_, NUMONE);
    ComputeGradAlphaAndBias(bsLen, n_);
    CopyOutAlphaAndBias(blockIdx_, blockIdx_ * n_);
    //grad_Z_pre ->workspace宏鹏哥实现，完成grad_norm_out计算，并搬运到workspace
    LocalTensor<U> gradZLocal = gradZBuf_.Get<U>();
    for (int64_t bsLoop = 0; bsLoop < bsLen; bsLoop++) {
        int64_t curBsIdx = bsOffset + bsLoop;
        U invRms = invRmsLocal(bsLoop);
        ComputeGradNormOutAndGradHcBeforeNorm(gradZLocal[bsLoop * n_], gradNormOutWs_, gradHcBeforeNormWs_, alphaPre_, invRms, n_, curBsIdx * k_);
    }

    ComputeZAndNormOutForward(invRmsLocal, bsLen, n_, n_, alphaPost_);
    CopyInHpreAndGradHPost(gradHPostGm_[gradHPostOffset], bsLen);
    ComputeSigmoidGrad(bsLen, n_, NUMTWO);
    ComputeGradAlphaAndBias(bsLen, n_);
    CopyOutAlphaAndBias(blockIdx_ + 64, blockIdx_ * n_ + 64 * n_);

    // grad_Z_post ->workspace宏鹏哥实现，完成grad_norm_out计算，并搬运到workspace
    for (int64_t bsLoop = 0; bsLoop < bsLen; bsLoop++) {
        int64_t curBsIdx = bsOffset + bsLoop;
        U invRms = invRmsLocal(bsLoop);
        ComputeGradNormOutAndGradHcBeforeNorm(gradZLocal[bsLoop * n_], gradNormOutWs_, gradHcBeforeNormWs_, alphaPost_, invRms, n_, curBsIdx * k_ + n_);
    }

    ComputeZAndNormOutForward(invRmsLocal, bsLen, n_ * n_, 2 * n_, alphaRes_);
    invRmsQue_.FreeTensor(invRmsLocal);
}

template<typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::ComputeZAndNormOutForward(const LocalTensor<U>& invRmsLocal, uint64_t bsLen, 
                                        uint64_t colLen, uint64_t oft, const U& alpha, uint64_t biasOft)
{
    CopyInHcBeforeNorm(bsLen, colLen, oft);
    LocalTensor<U> hcBeforeNormLocal = hcBeforeNormQue_.DeQue<U>();

    LocalTensor<U> zLocal = zBuf_.Get<U>();
    LocalTensor<U> normOutForwardLocal = normOutForwardBuf_.Get<U>();
    LocalTensor<U> biasLocal = biasBuf_.Get<U>();

    __local_mem__ U* zAddr = (__ubuf__ U*)zLocal.GetPhyAddr();
    __local_mem__ U* hcBeformNormAddr = (__ubuf__ U*)hcBeforeNormLocal.GetPhyAddr();
    __local_mem__ U* invRmsAddr = (__ubuf__ U*)invRmsLocal.GetPhyAddr();
    __local_mem__ U* normOutForwardAddr = (__ubuf__ U*)normOutForwardLocal.GetPhyAddr();
    __local_mem__ U* biasAddr = (__ubuf__ U*)biasLocal[biasOft].GetPhyAddr();

    auto cLenAlign = Ops::Base::CeilAlign(int64_t(colLen * uTypeSize_), blockSize_) / uTypeSize_;
    ComputeForwardPart<U>(hcBeformNormAddr, invRmsAddr, normOutForwardAddr, bsLen, colLen, cLenAlign, V_REG_SIZE, alpha);
    PipeBarrier<PIPE_V>();
    ComputeZAndForwardPart<U>(zAddr, normOutForwardAddr, biasAddr, bsLen, colLen, cLenAlign, V_REG_SIZE, alpha);
    hcBeforeNormQue_.FreeTensor(hcBeforeNormLocal);
}

template<typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::ComputeSigmoidGrad(uint64_t bsLen, uint64_t colLen, uint8_t num)
{
    LocalTensor<U> gradHLocal;
    LocalTensor<U> zPreLocal = zBuf_.Get<U>();
    LocalTensor<U> gradZLocal = gradZBuf_.Get<U>();

    __local_mem__ U* gradHAddr;
    __local_mem__ U* zPreAddr = (__ubuf__ U*)zPreLocal.GetPhyAddr();
    __local_mem__ U* gradZAddr = (__ubuf__ U*)gradZLocal.GetPhyAddr();
    if (num == NUMONE) {
        gradHLocal = gradHPreBuf_.Get<U>();
        gradHAddr = (__ubuf__ U*)gradHLocal.GetPhyAddr();
        SigmoidGrad<U, true>(gradHAddr, zPreAddr, gradZAddr, bsLen, colLen, V_REG_SIZE, eps_);
    } else {
        gradHLocal = hPreAndGradHPostQue_.DeQue<U>();
        gradHAddr = (__ubuf__ U*)gradHLocal.GetPhyAddr();
        SigmoidGrad<U, false>(gradHAddr, zPreAddr, gradZAddr, bsLen, colLen, V_REG_SIZE, eps_);
        hPreAndGradHPostQue_.FreeTensor(gradHLocal);
    }
}

template<typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::ComputeGradAlphaAndBias(uint64_t bsLen, uint64_t colLen)
{
    LocalTensor<U> normOutForwardLocal = normOutForwardBuf_.Get<U>();
    LocalTensor<U> gradZLocal = gradZBuf_.Get<U>();
    LocalTensor<U> gradAlphaLocal = gradAlphaAndBiasQue_.AllocTensor<U>();
    LocalTensor<U> gradBiasLocal = gradAlphaLocal[gradBiasOft_];

    __local_mem__ U* normOutForwardAddr = (__ubuf__ U*)normOutForwardLocal.GetPhyAddr();
    __local_mem__ U* gradZAddr = (__ubuf__ U*)gradZLocal.GetPhyAddr();
    __local_mem__ U* gradAlphaAddr = (__ubuf__ U*)gradAlphaLocal.GetPhyAddr();
    __local_mem__ U* gradBiasAddr = (__ubuf__ U*)gradBiasLocal.GetPhyAddr();

    ComputeGradAlpha<U>(normOutForwardAddr, gradZAddr, gradAlphaAddr, bsLen, colLen, V_REG_SIZE);
    ComputeGradBias<U>(gradBiasAddr, gradZAddr, bsLen, colLen, V_REG_SIZE);
    gradAlphaAndBiasQue_.EnQue<U>(gradAlphaLocal);
}

template<typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::ComputeGradHpreAndXFromIn(LocalTensor<U>& hPreLocal,
    LocalTensor<U>& gradHPreLocal, const Process1Info& info)
{
    LocalTensor<U> xFp32Local = xFp32Que_.AllocTensor<U>();
    LocalTensor<U> gradXFromHinLocal = gradXFromHinQue_.AllocTensor<U>();
    LocalTensor<X_T> xLocal = xQue_.DeQue<X_T>();
    LocalTensor<GRADHIN_T> gradHInLocal = gradHinQue_.DeQue<GRADHIN_T>();

    __local_mem__ X_T* xAddr = (__ubuf__ X_T*)xLocal.GetPhyAddr();
    __local_mem__ U* xCastAddr = (__ubuf__ U*)xFp32Local.GetPhyAddr();
    __local_mem__ GRADHIN_T* gradHInAddr = (__ubuf__ GRADHIN_T*)gradHInLocal.GetPhyAddr();
    __local_mem__ U* gradHPreAddr = (__ubuf__ U*)gradHPreLocal.GetPhyAddr();
    __local_mem__ U* hPreAddr = (__ubuf__ U*)hPreLocal.GetPhyAddr();
    __local_mem__ U* gradXFromHinAddr = (__ubuf__ U*)gradXFromHinLocal.GetPhyAddr();

    uint32_t vfLen = info.vRegSize / uTypeSize_;
    uint16_t loopCnt = (info.cLen + vfLen - 1) / vfLen;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<X_T> xReg;
        AscendC::MicroAPI::RegTensor<U> xCastReg;
        AscendC::MicroAPI::RegTensor<GRADHIN_T> gradHInReg;
        AscendC::MicroAPI::RegTensor<U> gradHInCastReg;
        AscendC::MicroAPI::RegTensor<U> gradHPreReg;
        AscendC::MicroAPI::RegTensor<U> gradXFromHinReg;
        AscendC::MicroAPI::RegTensor<U> gradHPreRegSumReg;
        AscendC::MicroAPI::MaskReg valueMaskReg;
        AscendC::MicroAPI::MaskReg OneMaskReg = AscendC::MicroAPI::CreateMask<U, AscendC::MicroAPI::MaskPattern::VL1>();

        for (uint16_t i = 0; i < info.bsLen; i++) {
            auto gradHPreStart = gradHPreAddr + i * n_;
            uint32_t maskLen = static_cast<uint32_t>(info.cLen);
            for (uint16_t j = 0; j < loopCnt; j++) {
                auto gradHInStart = gradHInAddr + i * info.cLenGRADHINTAlign + j * vfLen;
                valueMaskReg = AscendC::MicroAPI::UpdateMask<U>(maskLen);
                AscendC::MicroAPI::DataCopy<GRADHIN_T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(gradHInReg, gradHInStart);
                AscendC::MicroAPI::Cast<U, GRADHIN_T, castTrait16ToFloat>(gradHInCastReg, gradHInReg, valueMaskReg);
                for (uint16_t k = 0; k < info.n; k++) {
                    AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::SCALAR_LOAD>();
                    U hPre = *(hPreAddr + i * info.n + k);
                    U gradHPre = *(gradHPreStart + k);
                    auto xCastStart = xCastAddr + (i * n_ + k) * info.cLenUAlign + j * vfLen;
                    auto xAddrStart = xAddr + (i * n_ + k) * info.cLenXTAlign + j * vfLen;
                    auto gradXFromHinStart = gradXFromHinAddr + (i * n_ + k) * info.cLenUAlign + j * vfLen;

                    AscendC::MicroAPI::DataCopy<X_T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(xReg, xAddrStart);
                    AscendC::MicroAPI::Cast<U, X_T, castTrait16ToFloat>(xCastReg, xReg, valueMaskReg);
                    AscendC::MicroAPI::Muls(gradXFromHinReg, gradHInCastReg, hPre, valueMaskReg);          //grad_h_in_f32 * h_pre
                    AscendC::MicroAPI::Mul(gradHPreReg, xCastReg, gradHInCastReg, valueMaskReg);        //x_f32 * grad_h_in_f32
                    AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::SUM, U, U>(gradHPreReg, gradHPreReg, valueMaskReg); //reduce_sum
                    AscendC::MicroAPI::Adds(gradHPreRegSumReg, gradHPreReg, gradHPre, OneMaskReg);      //将原值进行累加

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
    gradHinQue_.FreeTensor(gradHInLocal);
}

template<typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::GetAlphaAndBias(void)
{
    LocalTensor<U> biasLocal_ = biasBuf_.Get<U>();
    alphaPre_ = alphaGm_.GetValue(0);
    alphaPost_ = alphaGm_.GetValue(1);
    alphaRes_ = alphaGm_.GetValue(2);
    DataCopyPad(biasLocal_, biasGm_, {static_cast<uint16_t>(1),
        static_cast<uint32_t>((n_ * n_ + 2 * n_) * sizeof(U)), 0, 0, 0}, {false, 0, 0, 0});
}


template<typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::SinkhornGradSimdVf(LocalTensor<float> skGradLocal, uint64_t bsIdx, uint64_t bsLen)
{

    int64_t gmOffset = blockIdx_ * curCoreProcessbsTask_ + bsIdx * bsLen;
    
    LocalTensor<float> gradHResLocal = gradHResQueue_.AllocTensor<float>();
    DataCopyPad(gradHResLocal, gradHResGm_[gmOffset * n_ * n_], {1, static_cast<uint32_t>(bsLen * n_ * n_ * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
    gradHResQueue_.EnQue(gradHResLocal);
    
    LocalTensor<float> sumOutLocal = sumOutBuf_.Get<float>();
    LocalTensor<float> normOutLocal = normOutBuf_.Get<float>();
    DataCopyPad(sumOutLocal, sumOutGm_[2 * skIterCount_ * gmOffset * n_], {1, static_cast<uint32_t>(2 * skIterCount_ * n_ * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
    DataCopyPad(normOutLocal, normOutGm_[2 * skIterCount_ * gmOffset * n_ * n_], {1, static_cast<uint32_t>(2 * skIterCount_ * n_ * n_ * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});

    gradHResLocal = gradHResQueue_.DeQue<float>();

    __local_mem__ float *gradHResPtr = (__local_mem__ float *)gradHResLocal.GetPhyAddr();
    __local_mem__ float *skGradPtr = (__local_mem__ float *)skGradLocal.GetPhyAddr();

    uint16_t bsLenLoop = static_cast<uint16_t>(bsLen);

    for (int64_t i = skIterCount_; i >= 0; --i) {

        LocalTensor<float> colSum = sumOutLocal[2 * i + 1];  // [n] 每列sum
        LocalTensor<float> rowSum = sumOutLocal[2 * i];      // [n] 每行sum
        LocalTensor<float> rowNorm = normOutLocal[2 * i];  // [tile*n*n] 归一化矩阵
        
        LocalTensor<float> colSumBroadcast = colSum;   // [n*n] broadcast后的colSum
        LocalTensor<float> rowSumBroadcast = rowSum;   // [n*n] broadcast后的rowSum
        LocalTensor<float> gradXRowNormed = gradXRowNormedBuf_.Get<float>();
        LocalTensor<float> tmpDiv3Local = tmpDiv3Buf_.Get<float>();

        __local_mem__ float *colSumPtr = (__local_mem__ float *)colSum.GetPhyAddr();
        __local_mem__ float *rowSumPtr = (__local_mem__ float *)rowSum.GetPhyAddr();
        __local_mem__ float *rowNormPtr = (__local_mem__ float *)rowNorm.GetPhyAddr();
        __local_mem__ float *colSumBroadcastPtr = (__local_mem__ float *)colSumBroadcast.GetPhyAddr();
        __local_mem__ float *rowSumBroadcastPtr = (__local_mem__ float *)rowSumBroadcast.GetPhyAddr();
        __local_mem__ float *gradXRowNormedPtr = (__local_mem__ float *)gradXRowNormed.GetPhyAddr();
        __local_mem__ float *tmpDiv3LocalPtr = (__local_mem__ float *)tmpDiv3Local.GetPhyAddr();

        __VEC_SCOPE__
        {   
            Reg::RegTensor<float> rowSumReg;
            Reg::RegTensor<float> rowSumBroadcastReg;
            Reg::RegTensor<float> colSumReg;
            Reg::RegTensor<float> colSumBroadcastReg;
            Reg::RegTensor<float> colSumSquareReg;
            Reg::UnalignRegForLoad ureg, ureg0, ureg1, ureg2, ureg3, ureg4, ureg5, ureg6, ureg7, ureg8, ureg9;
            Reg::UnalignRegForStore uregStore0, uregStore1, uregStore2, uregStore3, uregStore4, uregStore5, uregStore6, uregStore7;
            Reg::RegTensor<float> gradHResReg;
            Reg::RegTensor<float> tmpDivReg;
            Reg::RegTensor<float> tmpDiv2Reg;
            Reg::RegTensor<float> tmpDiv3Reg;
            Reg::RegTensor<float> tmpDiv4Reg;
            Reg::RegTensor<float> tmpMulReg;
            Reg::RegTensor<float> tmpMul2Reg;
            Reg::RegTensor<float> gradXRowNormedReg;
            Reg::RegTensor<float> gradXRowNormedReg1;

            Reg::RegTensor<float> rowNormReg;

            Reg::RegTensor<float> tmpSubReg;
            Reg::RegTensor<float> skGradReg;

            Reg::MaskReg maskN = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::MaskReg maskNN = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

            uint32_t maskLenN = static_cast<uint32_t>(n_);
            uint32_t maskLenNN = static_cast<uint32_t>(nn_);

            for (uint16_t j = 0; j < bsLenLoop; j++) {
                maskNN = Reg::UpdateMask<float>(maskLenNN);

                int64_t bsOffset = j * nn_;
                
                //broadcast colSumReg rowSumReg (B,S,N) -> (B,S,N,N) 
                for (uint16_t k = 0; k < n_; k++) {
                    maskN = Reg::UpdateMask<float>(maskLenN);
                    Reg::LoadUnAlignPre(ureg0, colSumPtr + bsOffset + k);
                    Reg::LoadUnAlign(colSumReg, ureg0, colSumPtr + bsOffset + k);

                    Reg::LoadUnAlignPre(ureg1, rowSumPtr + bsOffset + k);
                    Reg::LoadUnAlign(rowSumReg, ureg1, rowSumPtr + bsOffset + k);

                    Reg::Duplicate(colSumBroadcastReg, colSumReg, maskN);
                    Reg::Duplicate(rowSumBroadcastReg, rowSumReg, maskN);

                    Reg::StoreUnAlign(colSumBroadcastPtr, colSumBroadcastReg, uregStore0, bsOffset + k);
                    Reg::StoreUnAlign(rowSumBroadcastPtr, rowSumBroadcastReg, uregStore1, bsOffset + k);
                }
                Reg::StoreUnAlignPost(colSumBroadcastPtr, uregStore0, 0);
                Reg::StoreUnAlignPost(rowSumBroadcastPtr, uregStore1, 0);

                // 计算 tmpDiv2Reg
                Reg::LoadUnAlignPre(ureg4, rowNormPtr + bsOffset);
                Reg::LoadUnAlign(rowNormReg, ureg4, rowNormPtr + bsOffset);
                Reg::LoadUnAlignPre(ureg5, gradHResPtr + bsOffset);
                Reg::LoadUnAlign(gradHResReg, ureg5, gradHResPtr + bsOffset);

                Reg::LoadUnAlignPre(ureg6, colSumBroadcastPtr + bsOffset);
                Reg::LoadUnAlign(colSumBroadcastReg, ureg6, colSumBroadcastPtr + bsOffset);
                
                Reg::LoadUnAlignPre(ureg7, colSumPtr + bsOffset);
                Reg::LoadUnAlign(colSumReg, ureg7, colSumPtr + bsOffset);

                Reg::Mul(tmpMulReg, rowNormReg, gradHResReg, maskNN);
                Reg::Mul(colSumSquareReg, colSumReg, colSumReg, maskNN);
                Reg::Div(tmpDiv2Reg, tmpMulReg, colSumSquareReg, maskNN);

                Reg::StoreUnAlign(gradXRowNormedPtr, tmpDiv2Reg, uregStore2, bsOffset);
            }
            Reg::StoreUnAlignPost(gradXRowNormedPtr, uregStore2, 0);
                        // reduce sum
            for (uint16_t j = 0; j < bsLenLoop; j++) {
                maskNN = Reg::UpdateMask<float>(maskLenNN);
                maskN = Reg::UpdateMask<float>(maskLenN);

                int64_t bsOffset = j * nn_;

                // 搬入
                Reg::LoadUnAlignPre(ureg0, rowSumPtr + bsOffset);
                Reg::LoadUnAlign(rowSumReg, ureg0, rowSumPtr + bsOffset);

                Reg::LoadUnAlignPre(ureg1, colSumPtr + bsOffset);
                Reg::LoadUnAlign(colSumReg, ureg1, colSumPtr + bsOffset);

                Reg::LoadUnAlignPre(ureg2, rowNormPtr + bsOffset);
                Reg::LoadUnAlign(rowNormReg, ureg2, rowNormPtr + bsOffset);

                Reg::LoadUnAlignPre(ureg3, gradHResPtr + bsOffset);
                Reg::LoadUnAlign(gradHResReg, ureg3, gradHResPtr + bsOffset);

                Reg::LoadUnAlignPre(ureg4, colSumBroadcastPtr + bsOffset);
                Reg::LoadUnAlign(colSumBroadcastReg, ureg4, colSumBroadcastPtr + bsOffset);

                Reg::LoadUnAlignPre(ureg5, rowSumBroadcastPtr + bsOffset);
                Reg::LoadUnAlign(rowSumBroadcastReg, ureg5, rowSumBroadcastPtr + bsOffset);

                // reducesum (b,s,n,n) -> (b,s,1,n)
                for (uint16_t k = 0; k < n_; k++) {
                    Reg::LoadUnAlignPre(ureg6, gradXRowNormedPtr + bsOffset + k);
                    Reg::LoadUnAlign(gradXRowNormedReg, ureg6, gradXRowNormedPtr + bsOffset + k);
                    Reg::Add(gradXRowNormedReg1, gradXRowNormedReg1, gradXRowNormedReg, maskN);
                }

                // broadcast (b,s,1,n) -> (b,s,n,n)
                for (uint16_t k = 0; k < n_; k++) {
                    Reg::LoadUnAlignPre(ureg7, gradXRowNormedPtr + j * nn_ + k);
                    Reg::LoadUnAlign(gradXRowNormedReg, ureg7, gradXRowNormedPtr + j * nn_ + k);
                    Reg::Move(gradXRowNormedReg, gradXRowNormedReg1, maskN);
                }

                // tmpDiv3Reg、tmpDiv4Reg
                Reg::Div(tmpDivReg, gradHResReg, colSumReg, maskNN);
                Reg::Sub(tmpSubReg, tmpDivReg, gradXRowNormedReg, maskNN);
                Reg::Mul(tmpMul2Reg, tmpSubReg, rowNormReg, maskNN);
                Reg::Div(tmpDiv3Reg, tmpSubReg, rowSumReg, maskNN);
                Reg::Div(tmpDiv4Reg, tmpMul2Reg, rowSumReg, maskNN);

                Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(tmpDiv3LocalPtr, tmpDiv3Reg, j * nn_, maskNN);
                Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(skGradPtr, tmpDiv4Reg, j * nn_, maskNN);
            }

            // 计算sk_grad
            for (uint16_t j = 0; j < bsLenLoop; j++) {
                maskNN = Reg::UpdateMask<float>(maskLenNN);
                maskN = Reg::UpdateMask<float>(maskLenN);

                int64_t bsOffset = j * nn_;

                // reduce_sum (b,s,n,n) -> (b,s,n,1)
                for (uint16_t s = 0; s < n_; s++) {
                    for (uint16_t t = 0; t < n_; t++) {
                        Reg::LoadUnAlignPre(ureg1, skGradPtr + bsOffset + s * n_ + t);
                        Reg::LoadUnAlign(skGradReg, ureg1, skGradPtr + bsOffset + s * n_ + t);
                        Reg::Add(gradXRowNormedReg1, gradXRowNormedReg1, skGradReg, maskN);
                    }
                }

                // broadcast (b,s,n,1) -> (b,s,n,n)
                for (uint16_t s = 0; s < n_; s++) {
                    for (uint16_t t = 0; t < n_; t++) {
                        Reg::LoadUnAlignPre(ureg1, skGradPtr + bsOffset + s * n_ + t);
                        Reg::LoadUnAlign(skGradReg, ureg1, skGradPtr + bsOffset + s * n_ + t);
                        Reg::Move(skGradReg, gradXRowNormedReg1, maskN);
                        Reg::StoreUnAlign(skGradPtr, skGradReg, uregStore5, n_);
                    }
                    Reg::StoreUnAlignPost(skGradPtr, uregStore5, 0);
                }

                Reg::LoadUnAlignPre(ureg1, tmpDiv3LocalPtr + bsOffset);
                Reg::LoadUnAlign(tmpDiv3Reg, ureg1, tmpDiv3LocalPtr + bsOffset);
                Reg::LoadUnAlignPre(ureg2, skGradPtr + bsOffset);
                Reg::LoadUnAlign(skGradReg, ureg2, skGradPtr + bsOffset);
                Reg::Sub(skGradReg, tmpDiv3Reg, skGradReg, maskNN);

                // 搬出skGradReg
                Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(skGradPtr, skGradReg, skIterCount_ - i - 1 + bsOffset, maskNN);
            }
        }
    }
    gradHResQueue_.FreeTensor(gradHResLocal);
}

template<typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::ExpGradSimdVf(LocalTensor<float> skGrad, LocalTensor<float> expGrad, LocalTensor<float> zResLocal, uint64_t bsIdx, uint64_t bsLen)
{
    LocalTensor<float> zResMax = zResMaxBuf_.Get<float>();
    LocalTensor<float> zResMaxIdx = zResMaxIdxBuf_.Get<float>();
    LocalTensor<float> tmpMul = tmpMulBuf_.Get<float>();

    __local_mem__ float* expGradPtr = (__local_mem__ float*) expGrad.GetPhyAddr();
    __local_mem__ float* zResPtr = (__local_mem__ float*) zResLocal.GetPhyAddr();
    __local_mem__ float* skGradPtr = (__local_mem__ float*) skGrad.GetPhyAddr();
    __local_mem__ float* tmpMulPtr = (__local_mem__ float*) tmpMul.GetPhyAddr();

    uint32_t maskLenN = static_cast<uint32_t>(n_);
    uint32_t maskLenNN = static_cast<uint32_t>(nn_);

    uint16_t bsLenLoop = static_cast<uint16_t>(bsLen);

    for (uint16_t i = 0; i < bsLenLoop; i++) {

        AscendC::DataCopy(zResMaxIdx, zResLocal, maskLenNN);
        AscendC::DataCopy(zResMax, zResLocal, maskLenNN);

        __local_mem__ float* zResMaxPtr = (__local_mem__ float*) zResMax.GetPhyAddr();
        __local_mem__ float* zResMaxIdxPtr = (__local_mem__ float*) zResMaxIdx.GetPhyAddr();

        __VEC_SCOPE__
        {
            Reg::RegTensor<float> zResReg;
            Reg::RegTensor<float> zResMaxReg;
            Reg::RegTensor<float> zResMaxIdxReg;
            Reg::RegTensor<float> zResMaxBcastReg;
            Reg::RegTensor<float> zResMaxIdxBcastReg;
            
            Reg::RegTensor<float> tmpSubReg;
            Reg::RegTensor<float> zeroReg;
            Reg::RegTensor<float> oneReg;
            Reg::RegTensor<float> tmpExpReg;
            Reg::RegTensor<float> skGradReg;
            Reg::RegTensor<float> tmpMulReg;
            Reg::RegTensor<float> tmpMul2Reg;
            Reg::RegTensor<float> sumAllReg;

            Reg::RegTensor<float> isMaxReg;
            Reg::RegTensor<float> expGradReg;

            Reg::UnalignRegForLoad ureg, ureg0, ureg1, ureg2, ureg3, urege4;
            Reg::UnalignRegForStore uregStore0, uregStore1, uregStore2, uregStore3, uregStore4;
            Reg::MaskReg maxMask;
            Reg::MaskReg cmpMask;
            Reg::MaskReg mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::MaskReg pmask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::MaskReg maskN = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::MaskReg maskNN = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::MaskReg maskZero = Reg::CreateMask<float, Reg::MaskPattern::ALLF>();

            Reg::Duplicate(zeroReg, (float)0.0, mask);
            Reg::Duplicate(oneReg, (float)1.0, mask);
            maskNN = Reg::UpdateMask<float>(maskLenNN);

            // zResmax
            for (uint16_t j = 0; j < n_; j++) {  // 第三维
                for (uint16_t k = 0; k < n_; k++) {  // 第四维
                    maskN = Reg::UpdateMask<float>(maskLenN);
                    Reg::LoadUnAlignPre(ureg, zResPtr + j * n_ + k);
                    Reg::LoadUnAlign(zResReg, ureg, zResPtr + j * n_ + k);
                    Reg::Max<float>(zResMaxReg, zResMaxReg, zResReg, maskN);
                }
            }

            // zResmaxIdx
            Reg::RegTensor<float> idxReg;
            Reg::Arange(idxReg, 0);   // 不能放for里面

            for (uint16_t j = 0; j < n_; j++) {  // 第三维
                for (uint16_t k = 0; k < n_; k++) {  // 第四维

                    maskN = Reg::UpdateMask<float>(maskLenN);
                    Reg::LoadUnAlignPre(ureg, zResPtr + j * n_ + k);
                    Reg::LoadUnAlign(zResReg, ureg, zResPtr + j * n_ + k);
                    Reg::Compare<float>(maxMask, zResMaxReg, zResReg, maskN);
                    
                    Reg::Select(idxReg, idxReg, zeroReg, maxMask);
                    Reg::Max<float>(zResMaxIdxReg, zResMaxIdxReg, idxReg, maskN);
                }
            }

            //  make zResmax & zResmaxIdx broadcast
            Reg::Duplicate(zResMaxBcastReg, zResMaxReg, mask);
            Reg::Duplicate(zResMaxIdxBcastReg, zResMaxIdxReg, mask);
            for (uint16_t j = 0; j < n_; j++) {  // 第三维
                for (uint16_t k = 0; k < n_; k++) {  // 第四维
                    Reg::StoreUnAlign<float>(zResMaxPtr, zResMaxBcastReg, uregStore0, j * n_ + k);
                    Reg::StoreUnAlign<float>(zResMaxIdxPtr, zResMaxIdxReg, uregStore1, j * n_ + k);
                }
                Reg::StoreUnAlignPost(zResMaxPtr, uregStore0, 0);
                Reg::StoreUnAlignPost(zResMaxIdxPtr, uregStore1, 0);
            }

            //
            Reg::LoadUnAlignPre(ureg, zResPtr);
            Reg::LoadUnAlign(zResReg, ureg, zResPtr);
            Reg::LoadUnAlignPre(ureg, zResMaxPtr);
            Reg::LoadUnAlign(zResMaxReg, ureg, zResMaxPtr);
            Reg::LoadUnAlignPre(ureg, skGradPtr);
            Reg::LoadUnAlign(skGradReg, ureg, skGradPtr);

            Reg::Sub<float>(tmpSubReg, zResReg, zResMaxReg, maskNN);
            Reg::Exp<float>(tmpExpReg, tmpSubReg, maskNN);
            Reg::Mul<float>(tmpMulReg, tmpExpReg, skGradReg, maskNN);

            Reg::StoreUnAlign<float>(tmpMulPtr, tmpMulReg, uregStore2, i* nn_);
            Reg::StoreUnAlignPost(tmpMulPtr, uregStore2, 0);

            for (uint16_t j = 0; j < n_; j++) {  // 第三维
                for (uint16_t k = 0; k < n_; k++) {  // 第四维
                    maskN = Reg::UpdateMask<float>(maskLenN);
                    Reg::LoadUnAlignPre(ureg, tmpMulPtr + j * n_ + k);
                    Reg::LoadUnAlign(tmpMulReg, ureg, tmpMulPtr + j * n_ + k);
                    Reg::Add<float>(sumAllReg, sumAllReg, tmpMulReg, maskN);
                }
            }

            Reg::Duplicate(sumAllReg, sumAllReg, mask);
            for (uint16_t j = 0; j < n_; j++) {  // 第三维
                for (uint16_t k = 0; k < n_; k++) {  // 第四维
                    Reg::StoreUnAlign<float>(tmpMulPtr, sumAllReg, uregStore3, j * n_ + k);
                }
                Reg::StoreUnAlignPost(tmpMulPtr, uregStore3, 0);
            }

            Reg::LoadUnAlignPre(ureg, tmpMulPtr + i * nn_);
            Reg::LoadUnAlign(sumAllReg, ureg, tmpMulPtr + i * nn_);

            // 计算is_max
            Reg::RegTensor<float> idxReg1;
            Reg::Arange(idxReg1, static_cast<float>(0));   // 不能放for里面
            for (uint16_t j = 0; j < n_; j++) {

                maskN = Reg::UpdateMask<float>(maskLenN);

                Reg::LoadUnAlignPre(ureg, tmpMulPtr + i * nn_ + + j * n_);
                Reg::LoadUnAlign(sumAllReg, ureg, tmpMulPtr + i * nn_ + j * n_);

                Reg::LoadUnAlignPre(ureg, zResMaxIdxPtr + i * nn_ + j * n_);
                Reg::LoadUnAlign(zResMaxIdxReg, ureg, zResMaxIdxPtr + i * nn_ + j * n_);

                Reg::LoadUnAlignPre(ureg, tmpMulPtr + i * nn_  + j * n_);
                Reg::LoadUnAlign(tmpMulReg, ureg, tmpMulPtr + i * nn_ + j * n_);

                Reg::Compare(cmpMask, idxReg1, zResMaxIdxReg, maskN);
                Reg::Select(isMaxReg, oneReg, zeroReg, cmpMask);

                Reg::Mul(tmpMul2Reg, isMaxReg, sumAllReg, maskN);
                Reg::Sub(expGradReg, tmpMulReg, tmpMul2Reg, maskN);

                Reg::StoreUnAlign<float>(expGradPtr, expGradReg, uregStore4, i * nn_ + j * n_);
            }
            Reg::StoreUnAlignPost(expGradPtr, uregStore4, 0);
        }
    }
}

template<typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::ProcessGradXFromRms(int64_t bsIdx)
{
    auto curBsIdx = bsIdx + blockIdx_ * bsTaskCount_;
    auto curBsKOffset = curBsIdx * k_;
    auto curBsNCOffset = curBsIdx * n_ * c_;
    U curGradInvRms = 0;
    U curInvRms = invRmsGm_(curBsIdx);
    curInvRms = curInvRms * curInvRms * curInvRms;
    LocalTensor<U> gradNormOutLocal = gradCalcBuf1_.Get<U>();
    LocalTensor<U> hcBeforeNormLocal = gradCalcBuf2_.Get<U>();
    CopyIn(gradNormOutLocal, gradNormOutWs_[curBsKOffset], 0, k_, 0, 0);
    CopyIn(hcBeforeNormLocal, hcBeforeNormGm_[curBsKOffset], 0, k_, 0, 0);
    ComputeGradInvRms(gradNormOutLocal, hcBeforeNormLocal, k_, curGradInvRms);

    for (int64_t cLoop = 0; cLoop < cLoopsPerBS_; cLoop++) { // 切k
        auto calcCount= cLoop == cLoopsPerBS_ - 1 ? cTailPerBS_ : cCountPerBS_;
        auto calcOffset = curBsNCOffset + calcCount * cLoop;
        LocalTensor<U> xTmpLocal = xQue_.AllocTensor<U>();
        CopyIn(xTmpLocal, xFp32Ws_[calcOffset],  0, calcCount, 0, 0);
        ComputeGradXFromRms(xTmpLocal, curInvRms, curGradInvRms, calcCount, n_ * c_);
        CopyOut(xFp32Ws_[calcOffset], xTmpLocal, 0, calcCount, 0, 0);
        xFp32Que_.FreeTensor(xTmpLocal);
    }
}

template<typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::ComputeFinal()
{
    pipe_->Reset();
    pipe_->InitBuffer(hPreAndGradHPostQue_, DOUBLE_BUFFER, cBlockCount_ * sizeof(U));
    pipe_->InitBuffer(xQue_, DOUBLE_BUFFER, cBlockCount_ * sizeof(U));
    pipe_->InitBuffer(gradHinQue_, DOUBLE_BUFFER, cBlockCount_ * sizeof(U));
    pipe_->InitBuffer(xFp32Que_, DOUBLE_BUFFER, cBlockCount_ * sizeof(U));
    for (int64_t dataLoop = 0; dataLoop < cBlockLoops_; dataLoop++) { // 切c
        int64_t calcCount = dataLoop == cBlockLoops_ - 1 ? cBlockTailCount_ : cBlockCount_;
        int64_t calcOffset = blockIdx_ * blockCalcNum_ + dataLoop * cBlockCount_;
        LocalTensor<U> xFp32Local = xFp32Que_.AllocTensor<U>();
        LocalTensor<U> xLocal = xQue_.AllocTensor<U>();
        LocalTensor<U> gradXFromHinLocal = gradHinQue_.AllocTensor<U>();
        LocalTensor<U> gradXFromMatMulLocal = hPreAndGradHPostQue_.AllocTensor<U>();
        CopyIn(xLocal, xFp32Ws_[calcOffset], 0, calcCount, 0, 0);
        CopyIn(gradXFromHinLocal, gradXFromHinWs_[calcOffset], 0, calcCount, 0, 0);
        CopyIn(gradXFromMatMulLocal, gradXGm_[calcOffset], 0, calcCount, 0, 0);
        Add(xFp32Local, xLocal, gradXFromHinLocal, calcCount);
        Add(xFp32Local, xFp32Local, gradXFromMatMulLocal, calcCount);
        if constexpr(!std::is_same<X_T, U>::value){
            Cast(xFp32Local.template ReinterpretCast<X_T>(), xFp32Local, RoundMode::CAST_RINT, calcCount);
        }
        xFp32Que_.EnQue(xFp32Local);
        LocalTensor<X_T> xOutLocal = xFp32Que_.DeQue<X_T>();
        CopyOut(gradXGm_[calcOffset], xOutLocal, 0, calcCount, 0, 0);
        xFp32Que_.FreeTensor(xOutLocal);
        xQue_.FreeTensor(xLocal);
        gradHinQue_.FreeTensor(gradXFromHinLocal);
        hPreAndGradHPostQue_.FreeTensor(gradXFromMatMulLocal);
    }
}

template<typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::Process(void)
{
    if ASCEND_IS_AIV {
        if (blockIdx_ >= usedAivNum_) {
            return;
        }
        GetAlphaAndBias();
        uint64_t bsLoopNum = Ops::Base::CeilDiv(curCoreProcessbsTask_, bsLoopDataLen_);
        uint64_t bsTailDataLen = curCoreProcessbsTask_ - bsLoopDataLen_ * (bsLoopNum - 1);
        for (uint64_t bsIdx = 0; bsIdx < bsLoopNum; bsIdx++) {
            auto bsLen = bsIdx == bsLoopNum - 1 ? bsTailDataLen : bsLoopDataLen_;
            ComputeFirst(bsIdx, bsLen);
            // ComputeFirst(bsIdx, bsLen);

            LocalTensor<float> gradHResLocal = gradHResQueue_.AllocTensor<float>();
            LocalTensor<float> skGradLocal = skGradBuf_.Get<float>();
            LocalTensor<float> expGradLocal = expGradBuf_.Get<float>();
            LocalTensor<float> zReslocal = zResBuf_.Get<float>();

            SinkhornGradSimdVf(skGradLocal, bsIdx, bsLen);
            ExpGradSimdVf(skGradLocal, expGradLocal, zReslocal, bsIdx, bsLen);
            // 计算gradNormOut from gradZRes
            int64_t  bsOffset = blockIdx_ * curCoreProcessbsTask_ + bsIdx * bsLoopDataLen_;
            for (int64_t bsLoop = 0; bsLoop < bsLen; bsLoop++) {
                int64_t curBsIdx = bsOffset + bsLoop;
                U invRms = invRmsGm_(curBsIdx);
                ComputeGradNormOutAndGradHcBeforeNorm(expGradLocal[bsLoop * n_ * n_], gradNormOutWs_, gradHcBeforeNormWs_, alphaRes_, invRms, n_ * n_, curBsIdx * k_ + 2 * n_);
            }
        }
        // 计算GradXFromRms
        CrossCoreSetFlag<0x2, PIPE_MTE3>(0); // GradHcBeforeNorm计算完成，可以进行matmul计算
        SyncAll(); // GradNormOut需全部存储完毕再进行RMSNormGrad的计算
        for (uint64_t bsIdx = 0; bsIdx < curCoreProcessbsTask_; bsIdx++) {
            ProcessGradXFromRms(bsIdx);
        }
        
        // 计算GradX,需添加等待matmul完成
        CrossCoreWaitFlag<0x2, PIPE_MTE3>(1); // gradX matmul计算完成可以执行进行最后的计算
        SyncAll();
        ComputeFinal();
    }

    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<0x2, PIPE_MTE3>(0);
        SyncAll();
        for (int32_t taskOffset = blockIdx_ * calcBSPerTask_;
            taskOffset < totalAicTasksAligned_; taskOffset += avcNum_ * calcBSPerTask_) {
            int32_t tileTaskCount = min(calcBSPerTask_, totalAicTasks_ - taskOffset);
            if (tileTaskCount > 0) {
                ComputeGradXMatmul(taskOffset, tileTaskCount);
            }
        }
        CrossCoreSetFlag<0x2, PIPE_MTE3>(1); // grad_x matmul计算完成，需要aiv继续计算
        // 计算grad_phi 需要切K
        for (int32_t taskOffset = blockIdx_;
            taskOffset < n_ * (n_ + 2); taskOffset += avcNum_) {
            ComputeGradPhiMatmul(taskOffset, 1);
        }
    }
}

// mm1K_ = n_ * n_ + 2 * n_;
// mm1M_ = calcBSPerTask_;
// mm1N_ = n_ * c_;

template<typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::ComputeGradXMatmul(const int32_t taskOffset, const int32_t mm1M)
{
    if (mm1M <= 0)
        return;

    mm1_.SetTensorA(gradHcBeforeNormWs_[taskOffset * mm1K_]);
    mm1_.SetTensorB(phiGm_);
    mm1_.SetHF32(true, 1);
    mm1_.SetSingleShape(mm1M, mm1N_, mm1K_);
    mm1_.template IterateAll<false> (gradXGm_[taskOffset * (n_ * c_)], 1);
    mm1_.End();
}

// mm2K_ = B * S;
// mm2M_ = calcKPerTask_;
// mm2N_ = n_ * c_;

template<typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic<X_T, GRADHIN_T, U>::ComputeGradPhiMatmul(const int32_t taskOffset, const int32_t mm2M)
{
    if (mm2M <= 0)
        return;

    mm2_.SetTensorA(gradNormOutWs_[taskOffset], true);
    mm2_.SetTensorB(xFp32Ws_);
    mm2_.SetHF32(true, 1);
    mm2_.SetSingleShape(mm2M_, mm2N_, mm2K_);
    mm2_.template IterateAll<false> (gradPhiGm_[taskOffset * (n_ * c_)], 1);
    mm2_.End();
}

#endif  //  MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_H
