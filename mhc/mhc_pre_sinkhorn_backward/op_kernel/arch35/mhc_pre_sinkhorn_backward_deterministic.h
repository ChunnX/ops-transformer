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
#include "mhc_pre_sinkhorn_backward_deterministic_simd_vf.h"

template<typename X_T, typename GRADHIN_T, typename U>
class MhcPreSinkhornBackwardDeterministic {
public:
    __aicore__ inline MhcPreSinkhornBackwardDeterministic() = default;
    __aicore__ inline void Process(void);
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR phi, GM_ADDR hPre, GM_ADDR gradHin, GM_ADDR gradHPost,
        GM_ADDR gradHRes, GM_ADDR alpha, GM_ADDR bias, GM_ADDR hcBeforeNorm, GM_ADDR invRms, GM_ADDR sumOut,
        GM_ADDR normOut, GM_ADDR gradX, GM_ADDR gradPhi, GM_ADDR gradAlpha, GM_ADDR gradBias,
        GM_ADDR workspace, const MhcPreSinkhornBackwardArch35TilingData* tilingData, TPipe* pipe);

private:
    __aicore__ inline void ComputeFirst(uint64_t bsIdx, uint64_t bsLen);

private:


    using A0Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using A1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using C1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, U>;
    using C2Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;

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

    GlobalTensor<U>  xFp32Ws_;                  //中间结果
    GlobalTensor<U>  gradHcBeforeNormWs_;
    GlobalTensor<U>  gradXWs_;
    GlobalTensor<U>  gradNormOutWs_; 

    GlobalTensor<X_T>  gradXGm_;            //输出
    GlobalTensor<U>  gradPhiGm_;
    GlobalTensor<U>  gradAlphaGm_;
    GlobalTensor<U>  gradBiasGm_;



    int64_t batchSize_ = 0;
    int64_t seqLength_ = 0;
    int64_t n_ = 0;
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
    U alphaPre_, alphaPost_, alphaRes_;
    constexpr static uint32_t V_REG_SIZE = platform::GetVRegSize();

};

template<typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic::Init(GM_ADDR x, GM_ADDR phi, GM_ADDR hPre, GM_ADDR gradHin, GM_ADDR gradHPost,
        GM_ADDR gradHRes, GM_ADDR alpha, GM_ADDR bias, GM_ADDR hcBeforeNorm, GM_ADDR invRms, GM_ADDR sumOut,
        GM_ADDR normOut, GM_ADDR gradX, GM_ADDR gradPhi, GM_ADDR gradAlpha, GM_ADDR gradBias,
        GM_ADDR workspace, const MhcPreSinkhornBackwardArch35TilingData* tilingData, TPipe* pipe)
{
    batchSize_ = tilingData->batchSize;
    seqLength_ = tilingData->seqLength;
    n_ = tilingData->c;
    c_ = tilingData->n;
    cLoopDataLen_ = tilingData->cLoopDataLen;
    bsLoopDataLen_ = tilingData->bsLoopDataLen;
    skIterCount_ = tilingData->skIterCount;
    eps_ = tilingData->eps;
    k_ = n_ * n_ + 2 * n_;
    blockIdx_ = GetBlockIdx();
    curCoreProcessbsTask_ = (blockIdx_ == tilingData->aivNum - 1) ? tilingData->tailBsTaskCount_ : tilingData->bsTaskCount;

    gradPhiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ GRADHIN_T*>(gradPhi));
    phiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(phi));
    gradXWs_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(workspace));
    xFp32Ws_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(workspace));
    gradHcBeforeNormWs_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(workspace));
    gradNormOutWs_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(workspace));

    if ASCEND_IS_AIV {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_T*>(gradPhi));
        gradHPostGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(gradPhi));
        gradHResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(gradPhi));
        alphaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(gradPhi));
        biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(gradPhi));
        hPreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(gradPhi));
        hcBeforeNormGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(gradPhi));
        invRmsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(gradPhi));
        sumOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(gradPhi));
        normOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(gradPhi));


        // Init UB

    }

    if ASCEND_IS_AIV {

    }
}

template<typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic::ComputeFirst(uint64_t bsIdx, uint64_t bsLen)
{
    LocalTensor<U> hPreLocal = hPreQue_.AllocTensor<U>();
    LocalTensor<U> hcBeforeNormLocal = hcBeforeNormQue_.AllocTensor<U>();
    LocalTensor<U> invRmsLocal = invRmsQue_.AllocTensor<U>();
    LocalTensor<U> gradHPostLocal = gradHPostQue_.AllocTensor<U>();
    LocalTensor<X_T> xLocal = xQue_.AllocTensor<X_T>();
    LocalTensor<GRADHIN_T> gradHinLocal = gradHinQue_.AllocTensor<GRADHIN_T>();
    LocalTensor<U> xFp32Local = xFp32Que_.AllocTensor<U>();

    LocalTensor<U> gradHPreLocal = gradHPreBuf_.Get<U>();
    LocalTensor<U> gradXFromHinLocal = gradXFromHinBuf_.Get<U>();

    Duplicate(gradHPreLocal, float(0), bsLen * n_);

    uint64_t cLoopDataLen = cLoopDataLen_;
    uint64_t cLoopNum = ops::CeilDiv(c_, cLoopDataLen);
    uint64_t cTailDataLen = c_ - cLoopDataLen * (cLoopNum - 1);
    uint64_t bsOffset = blockIdx_ * curCoreProcessbsTask_ + bsIdx * bsLoopDataLen_;
    uint64_t hPreOffset = bsOffset * n_;
    uint64_t hcBeforeOffset = bsOffset * k_;
    uint64_t invRmsOffset = bsOffset;
    uint64_t gradHPostOffset = bsOffset * n_;
    uint64_t xOffset = bsOffset * n_ * c_;
    uint64_t gradHinOffset = bsOffset * c_;

    CopyIn(hPreLocal, hPreGm_[hPreOffset], bsLen, n_, 0, 0);

    uint32_t vfLen = V_REG_SIZE / sizeof(U);

    for (uint64_t cIdx = 0; cIdx < cLoopNum; cIdx++) {
        auto cLen = cIdx == cLoopNum - 1 ? cTailDataLen : cLoopDataLen;
        uint64_t xStartOft = xOffset + cIdx * cLoopDataLen;
        uint64_t gradHinStartOft = gradHinOffset + cIdx * cLoopDataLen;
        CopyIn(xLocal, xGm_[xStartOft], bsLen * n_, cLen, 0, (c_ - cLoopDataLen) * sizeof(X_T));
        CopyIn(gradHinLocal, gradHinGm_[gradHinStartOft], bsLen, cLen, 0, (c_ - cLoopDataLen) * sizeof(GRADHIN_T));
        ComputeProcess1(xLocal, xFp32Local, hPreLocal, gradHinLocal, gradHPreLocal, gradXFromHinLocal, bsLen, cLen, n_, c_);

        CopyOut(xFp32Ws_[cIdx * cLoopDataLen], xFp32Local, bsLen * n_, cLen, 0, (c_ - cLoopDataLen) * sizeof(U));       //搬运x_f32到workspace
/*      
        DataCopy(UB, x);
        DataCopy(UB, grad_h_in);
        Cast(x_f32);
        Cast(grad_h_in_f32);
*/

    }
    
/*      
    DataCopy(Ws, x_f32);

*/
}

template<typename X_T, typename GRADHIN_T, typename U>
void MhcPreSinkhornBackwardDeterministic::Process(void)
{
    if ASCEND_IS_AIV {
        uint64_t bsLoopDataLen = bsLoopDataLen_;
        uint64_t bsLoopNum = ops::CeilDiv(curCoreProcessbsTask_, bsLoopDataLen);
        uint64_t bsTailDataLen = curCoreProcessbsTask_ - bsLoopDataLen * (bsLoopNum - 1);
        for (uint64_t bsIdx = 0; bsIdx < bsLoopNum; bsIdx++) {
            auto bsLen = bsIdx == bsLoopNum - 1 ? bsTailDataLen : bsLoopDataLen;
            ComputeFirst(bsIdx, bsLen);



        }
    }

    if ASCEND_IS_AIC {

    }
}


#endif  //  MHC_PRE_SINKHORN_BACKWARD_DETERMINISTIC_H
