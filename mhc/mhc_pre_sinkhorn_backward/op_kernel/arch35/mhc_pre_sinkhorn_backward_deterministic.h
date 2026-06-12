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
    __aicore__ inline void CopyInHpre(uint64_t hPreGmOffset, uint64_t bsLen);
    __aicore__ inline void CopyInXandGradHin(uint64_t xGmOffset, uint64_t gradHinGmOffset, uint64_t bsLen, uint64_t cLen);
    __aicore__ inline void ComputeProcess1  (LocalTensor<U>& hPreLocal, LocalTensor<U>& gradHPreLocal, const Process1Info& info);
    
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
    GlobalTensor<U>  gradXFromHinWs_;

    GlobalTensor<X_T>  gradXGm_;            //输出
    GlobalTensor<U>  gradPhiGm_;
    GlobalTensor<U>  gradAlphaGm_;
    GlobalTensor<U>  gradBiasGm_;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> hPreQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> xQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> gradHinQue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> xFp32Que_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> gradXFromHinQue_;
    TBuf<TPosition::VECCALC> gradHPreBuf_;

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
    int64_t blockSize_ = 0;
    int64_t nAlignFp32_;
    int64_t cLenAlignFp32_;
    int64_t cLenAlignFp16_;
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
    blockSize_ = Ops::Base::GetUbBlockSize();
    curCoreProcessbsTask_ = (blockIdx_ == tilingData->aivNum - 1) ? tilingData->tailBsTaskCount_ : tilingData->bsTaskCount;
    nAlignFp32_ = ops::CeilAlign(n_ * sizeof(U), blockSize_) / sizeof(U);
    cAlignFp32_ = ops::CeilAlign(cLoopDataLen_ * sizeof(U), blockSize_) / sizeof(U);
    cAlignFp16_ = ops::CeilAlign(cLoopDataLen_ * sizeof(X_T), blockSize_) / sizeof(X_T);

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
        pipe_->InitBuffer(hPreQue_, DOUBLE_BUFFER, nAlignFp32_ * bsLoopDataLen_ * sizeof(U));
        pipe_->InitBuffer(xQue_, DOUBLE_BUFFER, cLenAlignFp16_ * n_ * bsLoopDataLen_ * sizeof(X_T));
        pipe_->InitBuffer(gradHinQue_, DOUBLE_BUFFER, cLenAlignFp16_ * bsLoopDataLen_ * sizeof(GRADHIN_T));
        pipe_->InitBuffer(xFp32Que_, DOUBLE_BUFFER, cLenAlignFp32_ * n_ * bsLoopDataLen_ * sizeof(U));
        pipe_->InitBuffer(gradXFromHinQue_, DOUBLE_BUFFER, cLenAlignFp32_ * n_ * bsLoopDataLen_ * sizeof(U));
        pipe_->InitBuffer(gradXFromHinQue_, nAlignFp32_ * bsLoopDataLen_ * sizeof(U));
    }

    if ASCEND_IS_AIV {

    }
}

__aicore__ inline void MhcPreSinkhornBackwardDeterministic::CopyInXandGradHin(uint64_t xGmOffset, uint64_t gradHinGmOffset, 
                                                                                uint64_t bsLen, uint64_t cLen)
{
    LocalTensor<X_T> xLocal = xQue_.AllocTensor<X_T>();             //in
    LocalTensor<GRADHIN_T> gradHinLocal = gradHinQue_.AllocTensor<GRADHIN_T>();

    CopyIn(xLocal, xGm_[xGmOffset], bsLen * n_, cLen, 0, (c_ - cLoopDataLen) * sizeof(X_T));
    CopyIn(gradHinLocal, gradHinGm_[gradHinGmOffset], bsLen, cLen, 0, (c_ - cLoopDataLen) * sizeof(GRADHIN_T));

    xQue_.EnQue(xLocal);
    gradHinQue_.EnQue(gradHinLocal);
}

__aicore__ inline void MhcPreSinkhornBackwardDeterministic::CopyInHpre(uint64_t hPreGmOffset, uint64_t bsLen)
{
    LocalTensor<U> hPreLocal = hPreQue_.AllocTensor<U>();

    CopyIn(hPreLocal, hPreGm_[hPreGmOffset], bsLen, n_, 0, 0);

    hPreQue_.EnQue(hPreLocal)
}

template<typename X_T, typename GRADHIN_T, typename U>
__aicore__ inline void MhcPreSinkhornBackwardDeterministic::ComputeFirst(uint64_t bsIdx, uint64_t bsLen)
{
    LocalTensor<U> gradHPreLocal = gradHPreBuf_.Get<U>();   //buf 中间结果
    
    Duplicate(gradHPreLocal, float(0), bsLen * n_);

    uint64_t cLoopDataLen = cLoopDataLen_;
    uint64_t cLoopNum = ops::CeilDiv(c_, cLoopDataLen);
    uint64_t cTailDataLen = c_ - cLoopDataLen * (cLoopNum - 1);
    uint64_t bsOffset = blockIdx_ * curCoreProcessbsTask_ + bsIdx * bsLoopDataLen_;
    uint64_t hPreOffset = bsOffset * n_;
    uint64_t xOffset = bsOffset * n_ * c_;
    uint64_t gradHinOffset = bsOffset * c_;

    CopyInHpre(hPreOffset, bsLen);
    LocalTensor<U> hPreLocal = hPreQue_.DeQue<U>();
    Process1Info info;
    info.bsLen = bsLen;
    info.cLen = cLen;
    info.n = n;
    info.n = c;
    info.vRegSize = V_REG_SIZE;
    uint32_t vfLen = V_REG_SIZE / sizeof(U);
    for (uint64_t cIdx = 0; cIdx < cLoopNum; cIdx++) {
        auto cLen = cIdx == cLoopNum - 1 ? cTailDataLen : cLoopDataLen;
        int64_t cLenXTAlign = ops::CeilAlign(cLen * sizeof(X_T), blockSize_) / sizeof(X_T);
        int64_t cLenUAlign = ops::CeilAlign(cLen * sizeof(U), blockSize_) / sizeof(U);
        int64_t cLenGRADHINTAlign = ops::CeilAlign(cLen * sizeof(GRADHIN_T), blockSize_) / sizeof(GRADHIN_T);
        info.cLenXTAlign = cLenXTAlign;
        info.cLenUAlign = cLenUAlign;
        info.cLenGRADHINTAlign = cLenGRADHINTAlign;
        uint64_t xStartOft = xOffset + cIdx * cLoopDataLen;
        uint64_t gradHinStartOft = gradHinOffset + cIdx * cLoopDataLen;
        CopyInXandGradHin(xStartOft, gradHinStartOft, bsLen, cLen);

        ComputeProcess1(hPreLocal, gradHPreLocal, info);

        LocalTensor<U> xFp32Local = xFp32Que_.DeQue<U>();     //out
        LocalTensor<U> gradXFromHinLocal = gradXFromHinQue_.DeQue<U>();       //out        
        CopyOut(xFp32Ws_[cIdx * cLoopDataLen], xFp32Local, bsLen * n_, cLen, 0, (c_ - cLoopDataLen) * sizeof(U));       //搬运x_f32到workspace
        CopyOut(gradXFromHinWs_[cIdx * cLoopDataLen], gradXFromHinLocal, bsLen * n_, cLen, 0, (c_ - cLoopDataLen) * sizeof(U));       //搬运grad_x_from_hin到workspace
        xFp32Que_.FreeTensor(xFp32Local);
        gradXFromHinQue_.FreeTensor(gradXFromHinLocal);
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
__aicore__ inline void MhcPreSinkhornBackwardDeterministic::ComputeProcess1 (LocalTensor<U>& hPreLocal,
    LocalTensor<U>& gradHPreLocal, const Process1Info& info)
{
    LocalTensor<X_T> xFp32Local = xFp32Que_.AllocTensor<U>();
    LocalTensor<U> gradXFromHinLocal = gradXFromHinQue_.AllocTensor<U>();
    LocalTensor<X_T> xLocal = xQue_.DeQue<X_T>();
    LocalTensor<GRADHIN_T> gradHInLocal = gradHinQue_.DeQue<GRADHIN_T>();

    __local_mem__ X_T* xAddr = (__ubuf__ X_T*)xLocal.GetPhyAddr();
    __local_mem__ U* xCastAddr = (__ubuf__ U*)xFp32Local.GetPhyAddr();
    __local_mem__ GRADHIN_T* gradHInAddr = (__ubuf__ GRADHIN_T*)gradHInLocal.GetPhyAddr();
    __local_mem__ U* gradHPreAddr = (__ubuf__ U*)gradHPreLocal.GetPhyAddr();
    __local_mem__ U* hPreAddr = (__ubuf__ U*)hPreLocal.GetPhyAddr();
    __local_mem__ U* gradXFromHinAddr = (__ubuf__ U*)gradXFromHinLocal.GetPhyAddr();

    uint32_t vfLen = info.vRegSize / sizeof(U);
    uint16_t loopCnt = (info.cLen + info.vfLen - 1) / vfLen;
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
        AscendC::MicroAPI::MaskReg OneMaskReg = AscendC::MicroAPI::CreateMask<U, AscendC::MicroAPI::VL1>();
        AscendC::MicroAPI::UnalignRegForStore gradHPreSumUreg;
        uint32_t maskLen = static_cast<uint32_t>(info.cLen);

        for (uint16_t i = 0; i < info.bsLen; i++) {
            auto gradHInStart = gradHInAddr + i * info.cLenGRADHINTAlign;
            for (uint16_t j = 0; j < loopCnt; j++) {
                valueMaskReg = AscendC::MicroAPI::UpdateMask<U>(maskLen);
                AscendC::MicroAPI::AddrReg gradHInAddrOfst = AscendC::MicroAPI::CreateAddrReg<GRADHIN_T>(j, vfLen);
                AscendC::MicroAPI::DataCopy<GRADHIN_T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(gradHInReg, gradHInStart, gradHInAddrOfst);
                AscendC::MicroAPI::Cast<U, GRADHIN_T, castTrait16ToFloat>(gradHInCastReg, gradHInReg, valueMaskReg);
                auto gradHPreStart = gradHPreAddr + i * n;
                for (uint16_t k = 0; k < info.n; k++) {
                    U hPre = *(hPreAddr + i * info.n + k);
                    U gradHPre = *(gradHPreStart + k);
                    auto xCastStart = xCastAddr + i * k * info.cLenUAlign;
                    auto xAddrStart = xAddr + i * k * info.cLenXTAlign;
                    auto gradXFromHinStart = gradXFromHinAddr + i * k * info.cLenUAlign;

                    AscendC::MicroAPI::AddrReg xAddrOfst = AscendC::MicroAPI::CreateAddrReg<X_T>(j, vfLen);
                    AscendC::MicroAPI::AddrReg xCastAddrOfst = AscendC::MicroAPI::CreateAddrReg<U>(j, vfLen);
                    AscendC::MicroAPI::AddrReg gradXFromHinAddrOfst = AscendC::MicroAPI::CreateAddrReg<U>(j, vfLen);

                    AscendC::MicroAPI::DataCopy<X_T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(xReg, xAddrStart, xAddrOfst);
                    AscendC::MicroAPI::Cast<U, X_T, castTrait16ToFloat>(xCastReg, xReg, valueMaskReg);
                    AscendC::MicroAPI::Muls(gradXFromHinReg, gradHInCastReg, hPre, valueMaskReg);          //grad_h_in_f32 * h_pre
                    AscendC::MicroAPI::Mul(gradHPreReg, xCastReg, gradHInCastReg, valueMaskReg);        //x_f32 * grad_h_in_f32
                    AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::SUM, U, U>(gradHPreReg, gradHPreReg, valueMaskReg); //reduce_sum
                    AscendC::MicroAPI::Adds(gradHPreRegSumReg, gradHPreReg, gradHPre, OneMaskReg);      //将原值进行累加

                    AscendC::MicroAPI::DataCopy(gradXFromHinStart, gradXFromHinReg, gradXFromHinAddrOfst, valueMaskReg);
                    AscendC::MicroAPI::DataCopy(xCastStart, xCastReg, xCastAddrOfst, valueMaskReg);
                    AscendC::MicroAPI::StoreUnAlign(gradHPreStart, gradHPreRegSumReg, gradHPreSumUreg, 1);
                }
                AscendC::MicroAPI::StoreUnAlignPost(gradHPreStart, gradHPreSumUreg, 0);
            }
        }
    }
    xFp32Que_.EnQue(xFp32Local);
    gradXFromHinQue_.EnQue(gradXFromHinLocal);
    xQue_.FreeTensor(xLocal);
    gradHinQue_.FreeTensor(gradHInLocal);
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
