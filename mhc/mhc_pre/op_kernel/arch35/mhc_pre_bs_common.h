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
 * \file mhc_pre_bs_common.h
 * \brief Common vector and buffer helpers for MHC Pre BS-derived templates
 */

#ifndef MHC_PRE_BS_COMMON_H_
#define MHC_PRE_BS_COMMON_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mhc_pre_utils.h"
#include "mhc_pre_base.h"
#include "mhc_pre_tiling_key.h"

namespace MhcPre {

using namespace AscendC;

template <class T, class P, int8_t HAS_RESI>
class MhcPreBsCommon : public MhcPreBase<T, P> {
public:
    using Base = MhcPreBase<T, P>;
    using Base::mnConfig_;
    using Base::coreNum_;
    using Base::totalLength_;
    using Base::N_;
    using Base::D_;
    using Base::outFlag_;
    using Base::hasGamma_;
    using Base::xGm_;
    using Base::phiGm_;
    using Base::alphaGm_;
    using Base::biasGm_;
    using Base::gammaGm_;
    using Base::hinGm_;
    using Base::hPostGm_;
    using Base::hResGm_;
    using Base::invRmsGm_;
    using Base::hMixGm_;
    using Base::hPreGm_;
    using Base::xFloatGm_;
    using Base::xInQueue_;
    using Base::gammaInQueue_;
    using Base::invRmsOutQueue_;
    using Base::outQueue_;
    using Base::biasInQue_;
    using Base::tmpBuff_;
    using Base::alphaBuf_;
    using Base::hPreBuff_;
    using Base::hPostBuff_;
    using Base::alphaInUb_;
    using Base::biasInUb_;
    using Base::matmulRes_;
    using Base::xLocal_;
    using Base::invRmsUb_;
    using Base::gammaUb_;
    using Base::pipe_;
    using Base::matrixInfo_;
    using Base::vectorOffset_;
    using Base::tiling_;
    using Base::chunkTSize_;
    using Base::v1ChunkDSize_;
    using Base::curSingleT_;
    using Base::coreIdx_;
    using Base::subBlockIdx_;
    using Base::scaleMean_;
    using Base::globalOffsetM_;
    using Base::vectorCount_;
    using Base::cubeCount_;
    using Base::mmCount_;
    using Base::vec1Count_;
    using Base::eleNumPerVf_;
    using Base::kDoubleBufferCount;
    using Base::kSingleBufferCount;
    using Base::kHalfSplitDivisor;
    using Base::kSupportedN4;
    using Base::kSupportedN6;
    using Base::kSupportedN8;
    using Base::ND_LENGTH;
    using Base::PARALLEL_NUM;
    using Base::hasResi_;
    using Base::V0_BASE_T;
    using Base::V1_BASE_T;

    // Compile-time residual mode.
    static constexpr int8_t kHasResi = HAS_RESI;

    // UB queue sizes shared by BS-derived templates.
    static constexpr uint32_t kXInQueueBufferBytes = 80U * 1024U;
    static constexpr uint32_t kOutQueueBufferBytes = 20U * 1024U;
    static constexpr uint32_t kTmpBufferBytes = 40U * 1024U;

    __aicore__ inline MhcPreBsCommon() = default;
    __aicore__ inline void InitHMixBuffer(InitParams initParams);
    __aicore__ inline void InitUbBuffers(bool splitAcrossSubBlocks = true,
                                         uint32_t v1BufferRows = V1_BASE_T);
    __aicore__ inline void InitBlockParams(uint64_t curblock, uint32_t tBlockNum);
    __aicore__ inline void V0PostProcess(uint32_t curblock, uint32_t tBlockNum);
    __aicore__ inline void V0Prologue(uint32_t curNdLen, uint32_t offsetNd);
};

template <class T, class P, int8_t HAS_RESI>
__aicore__ inline void MhcPreBsCommon<T, P, HAS_RESI>::InitHMixBuffer(InitParams initParams)
{
    if (outFlag_) {
        hMixGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_mix));
    } else {
        hMixGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ P *>(initParams.workspace + mnConfig_.singleCoreM * mnConfig_.singleCoreK *
                                                                    PARALLEL_NUM * sizeof(P) * coreNum_));
    }
}

template <class T, class P, int8_t HAS_RESI>
__aicore__ inline void MhcPreBsCommon<T, P, HAS_RESI>::InitUbBuffers(
    bool splitAcrossSubBlocks, uint32_t v1BufferRows)
{
    if ASCEND_IS_NOT_AIV {
        return;
    }

    pipe_->InitBuffer(xInQueue_, kDoubleBufferCount, kXInQueueBufferBytes);
    pipe_->InitBuffer(outQueue_, kDoubleBufferCount, kOutQueueBufferBytes);
    uint32_t invRmsRows = splitAcrossSubBlocks ?
                              (curSingleT_ + kHalfSplitDivisor - 1U) / kHalfSplitDivisor :
                              curSingleT_;
    uint32_t invRmsBytes = MhcPreUtils::Align(
                               invRmsRows, Base::kAlignmentBytes / sizeof(P)) *
                           sizeof(P);
    pipe_->InitBuffer(invRmsOutQueue_, kSingleBufferCount, invRmsBytes);

    if (hasGamma_) {
        pipe_->InitBuffer(gammaInQueue_, kSingleBufferCount, ND_LENGTH * sizeof(P));
    }

    pipe_->InitBuffer(tmpBuff_, kTmpBufferBytes);

    uint32_t parameterBufferBytes = MhcPreUtils::Align(
        mnConfig_.n * sizeof(P), Base::kAlignmentBytes);
    pipe_->InitBuffer(biasInQue_, kSingleBufferCount, parameterBufferBytes);
    pipe_->InitBuffer(alphaBuf_, parameterBufferBytes);
    alphaInUb_ = alphaBuf_.template Get<P>();

    uint64_t buffOffset = 0;
    hPreBuff_ = tmpBuff_.template GetWithOffset<P>(v1BufferRows * N_, buffOffset);
    buffOffset += v1BufferRows * N_ * sizeof(P);

    hPostBuff_ = tmpBuff_.template GetWithOffset<P>(v1BufferRows * N_, buffOffset);
}

template <class T, class P, int8_t HAS_RESI>
__aicore__ inline void MhcPreBsCommon<T, P, HAS_RESI>::InitBlockParams(
    uint64_t curblock, uint32_t tBlockNum)
{
    globalOffsetM_ = curblock * chunkTSize_;
    curSingleT_ = chunkTSize_;
    if (curblock == tBlockNum - 1) {
        mnConfig_.curSingleCoreM = totalLength_ - globalOffsetM_;
        curSingleT_ = matrixInfo_.totalLength - curblock * chunkTSize_;
    }
    if ASCEND_IS_AIV {
        this->VectorComputeOffset();
    }
}

template <class T, class P, int8_t HAS_RESI>
__aicore__ inline void MhcPreBsCommon<T, P, HAS_RESI>::V0Prologue(uint32_t curNdLen, uint32_t offsetNd)
{
    if (hasGamma_) {
        gammaUb_ = gammaInQueue_.template AllocTensor<P>();
        this->DataCopyGamma(curNdLen, offsetNd);
        gammaUb_ = gammaInQueue_.template DeQue<P>();
    }

    for (uint32_t offsetM = vectorOffset_.offsetMStart; offsetM < vectorOffset_.offsetMEnd; offsetM += V0_BASE_T) {
        uint32_t curMLen = V0_BASE_T;
        if (offsetM + V0_BASE_T >= vectorOffset_.offsetMEnd) {
            curMLen = vectorOffset_.offsetMEnd - offsetM;
        }
        uint64_t invRmsOffset = offsetM - vectorOffset_.offsetMStart;
        LocalTensor<P> invRmsUb = invRmsUb_[invRmsOffset];
        xLocal_ = xInQueue_.template AllocTensor<T>();
        this->DataCopyX(curMLen, curNdLen, offsetM, offsetNd);
        xLocal_ = xInQueue_.template DeQue<T>();

        LocalTensor<P> aL1Ub = outQueue_.template AllocTensor<P>();
        if (hasGamma_) {
            if (offsetNd == 0) {
                this->template VFDoV0ProcessXIn<true, true>(
                    (__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(),
                    (__ubuf__ T *)xLocal_.GetPhyAddr(), (__ubuf__ P *)gammaUb_.GetPhyAddr(),
                    curMLen, curNdLen);
            } else {
                this->template VFDoV0ProcessXIn<true, false>(
                    (__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(),
                    (__ubuf__ T *)xLocal_.GetPhyAddr(), (__ubuf__ P *)gammaUb_.GetPhyAddr(),
                    curMLen, curNdLen);
            }
        } else {
            if (offsetNd == 0) {
                this->template VFDoV0ProcessXIn<false, true>(
                    (__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(),
                    (__ubuf__ T *)xLocal_.GetPhyAddr(), nullptr, curMLen, curNdLen);
            } else {
                this->template VFDoV0ProcessXIn<false, false>(
                    (__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(),
                    (__ubuf__ T *)xLocal_.GetPhyAddr(), nullptr, curMLen, curNdLen);
            }
        }
        outQueue_.template EnQue<P>(aL1Ub);
        aL1Ub = outQueue_.template DeQue<P>();
        this->DataCopyOutToWorkSpace(aL1Ub, curMLen, curNdLen, offsetM, offsetNd);

        xInQueue_.FreeTensor(xLocal_);
        outQueue_.FreeTensor(aL1Ub);
    }

    if (hasGamma_) {
        gammaInQueue_.FreeTensor(gammaUb_);
    }
}

template <class T, class P, int8_t HAS_RESI>
__aicore__ inline void MhcPreBsCommon<T, P, HAS_RESI>::V0PostProcess(
    uint32_t curblock, uint32_t tBlockNum)
{
    if (vectorOffset_.singleCoreM == 0) {
        return;
    }

    this->VFDoV0ProcessInvRms(
        (__ubuf__ P *)invRmsUb_.GetPhyAddr(), vectorOffset_.singleCoreM, scaleMean_, matrixInfo_.normEps);
    this->DataCopyOutInvRmsUb(vectorOffset_.singleCoreM, vectorOffset_.offsetMStart);
}

} // namespace MhcPre

#endif // MHC_PRE_BS_COMMON_H_
