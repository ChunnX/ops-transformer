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
 * \file mhc_pre_split_nd.h
 * \brief MHC Pre kernel for ND split mode
 */

#ifndef MHC_PRE_SPLIT_ND_H_
#define MHC_PRE_SPLIT_ND_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mhc_pre_utils.h"
#include "mhc_pre_kernel_base.h"
#include "mhc_pre_cube_compute_arch35.h"
#include "mhc_pre_tiling_key.h"

namespace MhcPre {

using namespace matmul;
using namespace AscendC;

struct InitParamsDecode {
    GM_ADDR x;
    GM_ADDR phi;
    GM_ADDR alpha;
    GM_ADDR bias;
    GM_ADDR gamma;
    GM_ADDR hin;
    GM_ADDR h_post;
    GM_ADDR h_res;
    GM_ADDR inv_rms;
    GM_ADDR h_mix;
    GM_ADDR h_pre;
    GM_ADDR workspace;
    TPipe *tPipeIn;
    MhcPreTilingData *tilingData;
};

static constexpr uint64_t SYNC_V0toV0 = 0x1;
static constexpr uint64_t SYNC_V0toC = 0x2;
static constexpr uint64_t SYNC_CtoC = 0x3;
static constexpr uint64_t SYNC_CtoV1 = 0x4;

template <class T, class P, int8_t HAS_RESI>
class MhcPreKernelSplitND : public MhcPreKernelBase<T, P> {
public:
    using Base = MhcPreKernelBase<T, P>;
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
    using Base::kSupportedN4;
    using Base::kSupportedN6;
    using Base::hasResi_;

    static constexpr int8_t kHasResi = HAS_RESI;

    __aicore__ inline MhcPreKernelSplitND() = default;
    __aicore__ inline void Init(InitParams initParams);
    __aicore__ inline void Process();
    __aicore__ inline void AICProcess();
    __aicore__ inline void InitLocalBuffers();
    __aicore__ inline void VectorComputeOffset();
    __aicore__ inline void AIVPreLoad();
    template <bool hasGamma, bool isFirstND>
    __aicore__ inline void VFDoV0ProcessXInSingleReduce(__ubuf__ P *xDst, __ubuf__ P *invRmsDst,
        __ubuf__ T *xIn, __ubuf__ P *gamma, uint16_t mSize, uint16_t nSize);
    __aicore__ inline void DataCopyOutToWorkSpace(LocalTensor<P> &x, uint32_t curMLen, uint32_t curNdLen,
                                                  uint32_t offsetM, uint32_t offsetNd);
    __aicore__ inline void V0Prologue(uint64_t curBlock, uint64_t tBlockNum);
    __aicore__ inline void AIV1Process(uint64_t curBlock, uint64_t tBlockNum);
    __aicore__ inline void AIV1Prologue(uint64_t offsetT, uint64_t lenT, uint64_t singleCoreOffset);
    __aicore__ inline void HMixProcess(uint64_t offsetT, uint64_t lenT);

    GlobalTensor<P> mmResGm_;
    GlobalTensor<P> tempMMResGm_;
    TBuf<TPosition::A1> l1Buffer_;
    LocalTensor<P> aL1_;
    LocalTensor<P> bL1_;
    MhcPreCubeCompute mmService_;

    static constexpr uint32_t ND_LENGTH = 2048;
    static constexpr uint64_t V1_BASE_T = 8;

private:
    static constexpr uint32_t kXInQueueBufferBytes = 80 * 1024;
    static constexpr uint32_t kOutQueueBufferBytes = 16 * 1024;
    static constexpr uint32_t kTmpBufferBytes = 20 * 1024;
    static constexpr uint32_t kMinTForN4 = 2;
    static constexpr uint32_t kMinTForN6 = 4;
    static constexpr uint32_t kDefaultCurSingleM = 2;
    static constexpr uint32_t kDefaultMinT = 1;
    static constexpr uint32_t kDefaultvectorCoreNum_ = 2;
    static constexpr uint32_t kDefaultV0BaseT = 1;

    uint32_t chunNDSize_ = 320;
    uint32_t curSingleM_ = kDefaultCurSingleM;
    uint32_t minT_ = kDefaultMinT;
    uint32_t vectorCoreNum_ = kDefaultvectorCoreNum_;
    uint32_t V0_BASE_T = kDefaultV0BaseT;
};

template <class T, class P, int8_t HAS_RESI>
__aicore__ inline void MhcPreKernelSplitND<T, P, HAS_RESI>::Init(InitParams initParams)
{
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(initParams.x));
    phiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.phi));
    alphaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.alpha));
    biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.bias));
    gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.gamma));
    hinGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(initParams.hin));
    hPostGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_post));
    hResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_res));

    invRmsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.inv_rms));
    hPreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_pre));

    xFloatGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.workspace));

    this->InitFromTilingData(initParams.tilingData);

    if (N_ == kSupportedN4) {
        minT_ = kMinTForN4;
    }
    if (N_ == kSupportedN6) {
        minT_ = kMinTForN6;
    }

    chunNDSize_ = Ceil(matrixInfo_.nD, coreNum_);
    chunkTSize_ = Ceil(totalLength_, coreNum_);

    mnConfig_.m = matrixInfo_.totalLength;
    mnConfig_.n = matrixInfo_.fusionSize;
    mnConfig_.k = matrixInfo_.nD;
    mnConfig_.singleCoreM = totalLength_;
    mnConfig_.singleCoreN = mnConfig_.n;
    mnConfig_.singleCoreK = chunNDSize_;
    mnConfig_.curSingleCoreM = mnConfig_.singleCoreM;
    mnConfig_.curSingleCoreN = mnConfig_.singleCoreN;
    mnConfig_.curSingleCoreK = mnConfig_.singleCoreK;
    curSingleM_ = chunkTSize_;

    constexpr uint64_t kWorkspaceAlignBytes = 32UL;
    uint64_t xFloatWorkspaceBytes = totalLength_ * matrixInfo_.nD * sizeof(P);
    xFloatWorkspaceBytes = (xFloatWorkspaceBytes + kWorkspaceAlignBytes - 1UL) /
                        kWorkspaceAlignBytes * kWorkspaceAlignBytes;
    if (outFlag_) {
        mmResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_mix));
    } else {
        mmResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.workspace + xFloatWorkspaceBytes));
    }
    tempMMResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.workspace + xFloatWorkspaceBytes));

    pipe_ = initParams.tPipeIn;
    coreIdx_ = GetBlockIdx();
    subBlockIdx_ = GetSubBlockIdx();

    if ASCEND_IS_AIV {
        InitLocalBuffers();
    }
    if ASCEND_IS_AIC {
        pipe_->InitBuffer(l1Buffer_, MHC_PRE_LOW_API_L1_ALLOC_SIZE);
        aL1_ = l1Buffer_.Get<P>();
        bL1_ = aL1_[MHC_PRE_LOW_API_L1_BUF_NUM * MHC_PRE_LOW_API_L1_BUF_OFFSET];
        mmService_.Init(this->implMode_);
    }

    SyncAll<false>();
}


template <class T, class P, int8_t HAS_RESI>
__aicore__ inline void MhcPreKernelSplitND<T, P, HAS_RESI>::InitLocalBuffers()
{
    pipe_->InitBuffer(xInQueue_, kDoubleBufferCount, kXInQueueBufferBytes);
    pipe_->InitBuffer(outQueue_, kDoubleBufferCount, kOutQueueBufferBytes);
    pipe_->InitBuffer(invRmsOutQueue_, kSingleBufferCount, Ceil(curSingleM_, kDoubleBufferCount) * sizeof(P));

    if (hasGamma_) {
        pipe_->InitBuffer(gammaInQueue_, kSingleBufferCount, ND_LENGTH * sizeof(P));
    }
    
    pipe_->InitBuffer(tmpBuff_, kTmpBufferBytes);

    pipe_->InitBuffer(biasInQue_, kSingleBufferCount, mnConfig_.n * sizeof(P));
    pipe_->InitBuffer(alphaBuf_, mnConfig_.n * sizeof(P));
    alphaInUb_ = alphaBuf_.template Get<P>();

    uint64_t buffOffset = 0;
    hPreBuff_ = tmpBuff_.template GetWithOffset<P>(uint32_t(V1_BASE_T * N_), buffOffset);
    buffOffset += V1_BASE_T * N_ * sizeof(P);
    buffOffset = Ceil(buffOffset, 32) * 32;

    hPostBuff_ = tmpBuff_.template GetWithOffset<P>(uint32_t(V1_BASE_T * N_), buffOffset);
}

template <class T, class P, int8_t HAS_RESI>
__aicore__ inline void MhcPreKernelSplitND<T, P, HAS_RESI>::Process()
{
    if ASCEND_IS_AIV {
        coreIdx_ = GetBlockIdx() / kDoubleBufferCount;
        this->AIVPreLoad();

        uint32_t tBlockNum = Ceil(totalLength_, chunkTSize_);
        if (coreIdx_ < tBlockNum) {
            globalOffsetM_ = coreIdx_ * chunkTSize_;
            V0Prologue(coreIdx_, tBlockNum);
            AIV1Process(coreIdx_, tBlockNum);
        } else {
            AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_V0toV0);
            AscendC::CrossCoreWaitFlag(SYNC_V0toV0);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_V0toC);
        }
    }

    if ASCEND_IS_AIC {
        uint32_t ndBlockNum = Ceil(matrixInfo_.nD, chunNDSize_);
        if (coreIdx_ < ndBlockNum) {
            AICProcess();
        }
        // All launched AICs must join this barrier, including cores without a K slice.
        AscendC::CrossCoreSetFlag<0x0, PIPE_FIX>(SYNC_CtoC);
        AscendC::CrossCoreWaitFlag(SYNC_CtoC);
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(SYNC_CtoV1);
        mmService_.End(this->implMode_);
    }

    if ASCEND_IS_AIV {
        invRmsOutQueue_.FreeTensor(invRmsUb_);
        biasInQue_.FreeTensor(biasInUb_);
    }
}

template <class T, class P, int8_t HAS_RESI>
__aicore__ inline void MhcPreKernelSplitND<T, P, HAS_RESI>::AIVPreLoad()
{
    Base::AIVPreLoad();
}

template <class T, class P, int8_t HAS_RESI>
__aicore__ inline void MhcPreKernelSplitND<T, P, HAS_RESI>::AICProcess()
{
    AscendC::CrossCoreWaitFlag(SYNC_V0toC);
    
    uint64_t offset = chunNDSize_ * coreIdx_;
    uint64_t outOffset = mnConfig_.singleCoreM * mnConfig_.singleCoreN * coreIdx_;
    if (offset + mnConfig_.singleCoreK > mnConfig_.k) {
        mnConfig_.curSingleCoreK = mnConfig_.k - offset;
    }

    uint64_t mAlign = LowApiAlign(mnConfig_.singleCoreM, AscendC::BLOCK_CUBE);
    uint64_t nAlign = LowApiAlign(mnConfig_.singleCoreN, AscendC::BLOCK_CUBE);
    uint64_t maxKL1 = MHC_PRE_LOW_API_L1_BUF_OFFSET / mAlign;
    maxKL1 = LowApiAlign(maxKL1, MHC_PRE_LOW_API_C0_SIZE) - MHC_PRE_LOW_API_C0_SIZE;
    maxKL1 = AscendC::Std::max(maxKL1, static_cast<uint64_t>(MHC_PRE_LOW_API_C0_SIZE));
    uint64_t baseK = (256U / AscendC::Std::max(mAlign, nAlign)) * 32U;
    baseK = AscendC::Std::max(baseK, 32UL);
    // Keep each B GM-to-L1 transfer within the low-level cube K block.
    maxKL1 = AscendC::Std::min(maxKL1, baseK);

    uint8_t aL1BufferId = 0;
    uint64_t endOffset = offset + mnConfig_.curSingleCoreK;
    for (uint64_t kOffset = offset; kOffset < endOffset; kOffset += maxKL1) {
        uint64_t currentK = AscendC::Std::min(maxKL1, endOffset - kOffset);
        uint8_t bL1BufferId = mmService_.GetBL1BufferId();
        LocalTensor<P> currentAL1 = aL1_[aL1BufferId * MHC_PRE_LOW_API_L1_BUF_OFFSET];
        LocalTensor<P> currentBL1 = bL1_[bL1BufferId * MHC_PRE_LOW_API_L1_BUF_OFFSET];
        mmService_.CopyInA1Nd2Nz(mnConfig_.singleCoreM, currentK, mnConfig_.k,
            xFloatGm_[kOffset], currentAL1);
        mmService_.CopyInB1Nd2Nz(mnConfig_.k, currentK, mnConfig_.singleCoreN,
            phiGm_[kOffset], currentBL1);
        bool isFirst = kOffset == offset;
        bool isLast = kOffset + currentK >= endOffset;
        mmService_.Process(mnConfig_.singleCoreM, mnConfig_.singleCoreN,
            mnConfig_.singleCoreM, baseK, isFirst, isLast, currentAL1, currentBL1);
        if (isLast) {
            mmService_.CopyOut(tempMMResGm_[outOffset], mnConfig_.singleCoreN);
        }
        aL1BufferId ^= 1U;
    }
}

template <class T, class P, int8_t HAS_RESI>
__aicore__ inline void MhcPreKernelSplitND<T, P, HAS_RESI>::VectorComputeOffset()
{
    uint64_t alignSingleM = Ceil(curSingleM_, 2);
    vectorOffset_.singleCoreM = alignSingleM <  curSingleM_ ? alignSingleM : curSingleM_;
    if (subBlockIdx_ == 0) {
        vectorOffset_.offsetMStart = 0;
        vectorOffset_.offsetMEnd = vectorOffset_.singleCoreM;
    } else {
        vectorOffset_.offsetMStart = vectorOffset_.singleCoreM;
        vectorOffset_.singleCoreM = curSingleM_ - vectorOffset_.singleCoreM;
        vectorOffset_.offsetMEnd = curSingleM_;
    }
}

template <class T, class P, int8_t HAS_RESI>
__aicore__ inline void MhcPreKernelSplitND<T, P, HAS_RESI>::DataCopyOutToWorkSpace(LocalTensor<P> &x, uint32_t curMLen,
                                                                          uint32_t curNdLen, uint32_t offsetM,
                                                                          uint32_t offsetNd)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(curMLen);
    copyParams.blockLen = uint32_t(curNdLen * sizeof(P));
    copyParams.srcStride = uint32_t(0);
    copyParams.dstStride = uint32_t((matrixInfo_.nD - curNdLen) * sizeof(P));

    uint64_t offset = (globalOffsetM_ + offsetM) * matrixInfo_.nD + offsetNd;
    DataCopyPad(xFloatGm_[offset], x, copyParams);
}

template <class T, class P, int8_t HAS_RESI>
template <bool hasGamma, bool isFirstND>
__aicore__ inline void MhcPreKernelSplitND<T, P, HAS_RESI>::VFDoV0ProcessXInSingleReduce(
    __ubuf__ P *xDst, __ubuf__ P *invRmsDst, __ubuf__ T *xIn, __ubuf__ P *gamma,
    uint16_t mSize, uint16_t nSize)
{
    uint32_t nSrcUbAligned = MhcPreUtils::Align(
        nSize, static_cast<uint16_t>(MhcPreUtils::UB_ALIGN_SIZE / sizeof(T)));
    uint32_t nDstUbAligned = MhcPreUtils::Align(
        nSize, static_cast<uint16_t>(MhcPreUtils::UB_ALIGN_SIZE / sizeof(P)));
    uint16_t nLoopCnt = MhcPreUtils::CeilDiv(nSize, eleNumPerVf_);
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<P>();
        for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
            uint32_t elementNum = nSize;
            MicroAPI::RegTensor<P> squareSumReg;
            MicroAPI::Duplicate(squareSumReg, 0.0f);
            for (uint16_t vfBlockIdx = 0; vfBlockIdx < nLoopCnt; ++vfBlockIdx) {
                MicroAPI::RegTensor<T> xInReg;
                MicroAPI::RegTensor<P> gammaReg;
                MicroAPI::RegTensor<P> xFp32Reg;
                MicroAPI::RegTensor<P> xMulReg;
                MicroAPI::RegTensor<P> squareReg;
                uint32_t xInOffset = mIdx * nSrcUbAligned + vfBlockIdx * eleNumPerVf_;
                MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xInReg, xIn + xInOffset);
                MicroAPI::MaskReg blockMask = MicroAPI::UpdateMask<P>(elementNum);
                MicroAPI::Cast<P, T, ctHalf2Fp32Zero>(xFp32Reg, xInReg, blockMask);
                if constexpr (hasGamma) {
                    MicroAPI::LoadAlign(gammaReg, gamma + vfBlockIdx * eleNumPerVf_);
                    MicroAPI::Mul(xMulReg, gammaReg, xFp32Reg, blockMask);
                } else {
                    xMulReg = xFp32Reg;
                }
                uint32_t dstUbOffset = mIdx * nDstUbAligned + vfBlockIdx * eleNumPerVf_;
                MicroAPI::StoreAlign(xDst + dstUbOffset, xMulReg, blockMask);
                MicroAPI::Mul(squareReg, xFp32Reg, xFp32Reg, blockMask);
                MicroAPI::Add<P, MicroAPI::MaskMergeMode::MERGING>(
                    squareSumReg, squareSumReg, squareReg, blockMask);
            }
            MicroAPI::RegTensor<P> partialReg;
            MicroAPI::Reduce<MicroAPI::ReduceType::SUM>(partialReg, squareSumReg, fullMask);
            if constexpr (isFirstND) {
                MicroAPI::Store(invRmsDst + mIdx, partialReg, 1U);
            } else {
                MicroAPI::RegTensor<P> sumReg;
                MicroAPI::Load(sumReg, invRmsDst + mIdx);
                MicroAPI::Add(sumReg, sumReg, partialReg, fullMask);
                MicroAPI::Store(invRmsDst + mIdx, sumReg, 1U);
            }
        }
    }
}
template <class T, class P, int8_t HAS_RESI>
__aicore__ inline void MhcPreKernelSplitND<T, P, HAS_RESI>::V0Prologue(uint64_t curBlock, uint64_t tBlockNum)
{
    curSingleM_ = chunkTSize_;
    if (curBlock == tBlockNum - 1) {
        curSingleM_ = matrixInfo_.totalLength - curBlock * chunkTSize_;
    }

    VectorComputeOffset();
    if (vectorOffset_.singleCoreM == 0) {
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_V0toV0);
        AscendC::CrossCoreWaitFlag(SYNC_V0toV0);
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_V0toC);
        return;
    }

    for (uint32_t offsetNd = 0; offsetNd < matrixInfo_.nD; offsetNd += ND_LENGTH) {
        uint32_t curNdLen = ND_LENGTH;
        if (offsetNd + ND_LENGTH >= matrixInfo_.nD) {
            curNdLen = matrixInfo_.nD - offsetNd;
        }

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
                    this->template VFDoV0ProcessXInSingleReduce<true, true>(
                        (__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(),
                        (__ubuf__ T *)xLocal_.GetPhyAddr(), (__ubuf__ P *)gammaUb_.GetPhyAddr(), curMLen, curNdLen);
                } else {
                    this->template VFDoV0ProcessXInSingleReduce<true, false>(
                        (__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(),
                        (__ubuf__ T *)xLocal_.GetPhyAddr(), (__ubuf__ P *)gammaUb_.GetPhyAddr(), curMLen, curNdLen);
                }
            } else {
                if (offsetNd == 0) {
                    this->template VFDoV0ProcessXInSingleReduce<false, true>(
                        (__ubuf__ P *)aL1Ub.GetPhyAddr(), (__ubuf__ P *)invRmsUb.GetPhyAddr(),
                        (__ubuf__ T *)xLocal_.GetPhyAddr(), nullptr, curMLen, curNdLen);
                } else {
                    this->template VFDoV0ProcessXInSingleReduce<false, false>(
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
    AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(SYNC_V0toV0);
    AscendC::CrossCoreWaitFlag(SYNC_V0toV0);
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNC_V0toC);

    this->VFDoV0ProcessInvRms(
        (__ubuf__ P *)invRmsUb_.GetPhyAddr(), vectorOffset_.singleCoreM, scaleMean_, matrixInfo_.normEps);
    this->DataCopyOutInvRmsUb(vectorOffset_.singleCoreM, vectorOffset_.offsetMStart);
}

template <class T, class P, int8_t HAS_RESI>
__aicore__ inline void MhcPreKernelSplitND<T, P, HAS_RESI>::AIV1Process(uint64_t curBlock, uint64_t tBlockNum)
{
    curSingleM_ = chunkTSize_;
    if (curBlock == tBlockNum - 1) {
        curSingleM_ = matrixInfo_.totalLength - curBlock * chunkTSize_;
    }
    VectorComputeOffset();
    if (vectorOffset_.singleCoreM == 0) { // uint64_t类型永远不会小于0
        return;
    }
    AscendC::CrossCoreWaitFlag(SYNC_CtoV1);
    uint64_t lenT = 0;
    uint64_t singleCoreOffset = 0;
    for (int offsetT = vectorOffset_.offsetMStart; offsetT < vectorOffset_.offsetMEnd; offsetT += V1_BASE_T) {
        lenT = V1_BASE_T < vectorOffset_.offsetMEnd - offsetT ? V1_BASE_T : vectorOffset_.offsetMEnd - offsetT;
        AIV1Prologue(offsetT, lenT, singleCoreOffset);

        this->AIV1ProcessHPost(offsetT, lenT);
        
        this->AIV1ProcessHPre(offsetT, lenT);
        this->AIV1ProcessHInBatched(offsetT, lenT, kXInQueueBufferBytes, kOutQueueBufferBytes);

        singleCoreOffset += lenT;
    }
}

template <class T, class P, int8_t HAS_RESI>
__aicore__ inline void MhcPreKernelSplitND<T, P, HAS_RESI>::AIV1Prologue(
    uint64_t offsetT, uint64_t lenT, uint64_t singleCoreOffset)
{
    uint64_t offset = globalOffsetM_ + offsetT;
    HMixProcess(offsetT, lenT);

    matmulRes_ = xInQueue_.template DeQue<P>(); 
    LocalTensor<P> hResOutLocal;
    if constexpr (HAS_RESI == MHC_PRE_HAS_RESI) {
        hResOutLocal = outQueue_.template AllocTensor<P>();
    }

    if (N_ == 4U) {
        MhcPreVFPostSegments<P, 4U, HAS_RESI == MHC_PRE_HAS_RESI>(
            hPreBuff_, hPostBuff_, hResOutLocal, matmulRes_, invRmsUb_,
            alphaInUb_, biasInUb_, static_cast<uint16_t>(lenT),
            static_cast<uint32_t>(singleCoreOffset), mnConfig_.n);
    } else if (N_ == 6U) {
        MhcPreVFPostSegments<P, 6U, HAS_RESI == MHC_PRE_HAS_RESI>(
            hPreBuff_, hPostBuff_, hResOutLocal, matmulRes_, invRmsUb_,
            alphaInUb_, biasInUb_, static_cast<uint16_t>(lenT),
            static_cast<uint32_t>(singleCoreOffset), mnConfig_.n);
    } else {
        MhcPreVFPostSegments<P, 8U, HAS_RESI == MHC_PRE_HAS_RESI>(
            hPreBuff_, hPostBuff_, hResOutLocal, matmulRes_, invRmsUb_,
            alphaInUb_, biasInUb_, static_cast<uint16_t>(lenT),
            static_cast<uint32_t>(singleCoreOffset), mnConfig_.n);
    }
    PipeBarrier<PIPE_V>();

    if constexpr (HAS_RESI == MHC_PRE_HAS_RESI) {
        outQueue_.EnQue(hResOutLocal);
        hResOutLocal = outQueue_.template DeQue<P>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(1);
        copyParams.blockLen = uint32_t(lenT * N_ * N_ * sizeof(P));
        copyParams.srcStride = uint32_t(0);
        copyParams.dstStride = uint32_t(0);
        DataCopyPad(hResGm_[offset * N_ * N_], hResOutLocal, copyParams);
        outQueue_.FreeTensor(hResOutLocal);
    xInQueue_.FreeTensor(matmulRes_);
    }
}

template <class T, class P, int8_t HAS_RESI>
__aicore__ inline void MhcPreKernelSplitND<T, P, HAS_RESI>::HMixProcess(uint64_t offsetT, uint64_t lenT)
{
    uint32_t mmResGmBlockNum = Ceil(matrixInfo_.nD, chunNDSize_);
    uint32_t computeLen = lenT * mnConfig_.n;
    // align to 32B
    uint32_t alignLen = MhcPreUtils::Align(computeLen, 8);
    uint64_t HMixOffset = (globalOffsetM_ + offsetT) * mnConfig_.n;

    LocalTensor<P> hMixLocal = xInQueue_.template AllocTensor<P>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(mmResGmBlockNum);
    copyParams.blockLen = uint32_t(computeLen * sizeof(P));
    copyParams.srcStride = uint32_t((mnConfig_.curSingleCoreM * mnConfig_.curSingleCoreN - computeLen) * sizeof(P));
    copyParams.dstStride = uint32_t((alignLen - computeLen) * sizeof(P) / MhcPreUtils::UB_ALIGN_SIZE);
    DataCopyPadExtParams<P> copyPadParams{true, 0, 0, 0};
    DataCopyPad(hMixLocal, tempMMResGm_[HMixOffset], copyParams, copyPadParams);
    xInQueue_.EnQue(hMixLocal);
    hMixLocal = xInQueue_.template DeQue<P>();

    uint64_t addOffset = 0;
    for(uint32_t mmResGmBlockIdx = 1; mmResGmBlockIdx < mmResGmBlockNum; mmResGmBlockIdx++)
    {
        addOffset += alignLen;
        Add(hMixLocal, hMixLocal, hMixLocal[addOffset], computeLen);
        PipeBarrier<PIPE_V>();
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID4);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID4);
    if(outFlag_)
    {
        DataCopyExtParams MixOutCopyParams;
        MixOutCopyParams.blockCount = static_cast<uint16_t>(1);
        MixOutCopyParams.blockLen = uint32_t(computeLen * sizeof(P));
        MixOutCopyParams.srcStride = uint32_t(0);
        MixOutCopyParams.dstStride = uint32_t(0);
        DataCopyPad(mmResGm_[HMixOffset], hMixLocal, MixOutCopyParams);
    }
    xInQueue_.EnQue(hMixLocal);
}

} // namespace MhcPre

#endif // MHC_PRE_SPLIT_ND_H_
