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
 * \file moe_dispatch_ffn_combine.h
 * \brief
 */

#ifndef MOE_DISPATCH_FFN_COMBINE_H
#define MOE_DISPATCH_FFN_COMBINE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#include "kernel_operator_list_tensor_intf.h"
#include "moe_dispatch_ffn_combine_base.h"
#include "moe_dispatch_ffn_combine_workspace_info.h"
#include "block_epilogue_swiglu_mx_quant.h"
#include "tla/layout.hpp"
#include "moe_dispatch_ffn_combine_catlass_impl.h"

using namespace AscendC;

namespace MoeDispatchFFNCombineImpl {
using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
using BlockOffset = Shape<int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t>;

#define TemplateMoeDispatchFFNCombineTypeClass typename XType, typename OutputType, int32_t QuantMode
#define TemplateMoeDispatchFFNCombineTypeFunc XType, OutputType, QuantMode

template <TemplateMoeDispatchFFNCombineTypeClass>
class MoeDispatchFFNCombine {
public:
    using QuantOutType = typename std::conditional<((QuantMode >= 3) && (QuantMode != 4)), fp8_e5m2_t, fp8_e4m3fn_t>::type;
    using QuantScaleOutType = typename std::conditional<(QuantMode >= 3), fp8_e8m0_t, float>::type;

    using GMM1InType = QuantOutType;
    using GMM1OutType = bfloat16_t;
    using Weight1Type = QuantOutType;
    using WeightScale1Type = QuantScaleOutType;
    using Weight2Type = QuantOutType;
    using WeightScale2Type = QuantScaleOutType;
    __aicore__ inline MoeDispatchFFNCombine() {};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, 
        GM_ADDR weight1, GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2, 
        GM_ADDR scales, GM_ADDR yOut, GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, TPipe *pipe,
        MoeDispatchFFNCombineTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void BuffInit();
    __aicore__ inline void ResetWorkSpaceFlagList();
    __aicore__ inline void GetCumsumForMMAIV();
    __aicore__ inline void CrossRankSyncCntPerExpert();
    __aicore__ inline void DispatchTokens(GMMAddrInfo &gmmAddrInfo, uint32_t expertIdx);
    __aicore__ inline bool UpdateGroupParams(uint32_t expertIdx);
    __aicore__ inline void UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo, uint32_t gmmIndex);
    __aicore__ inline void ArgSortExpandedRowIdx();
    __aicore__ inline void ResetTokenPerExpert();
    __aicore__ inline void Unpermute();
    __aicore__ inline void EndSync();
    __aicore__ inline void EndGMM2Sync();
    __aicore__ inline void CrossRankSyncInWorldSize();

    __aicore__ inline void CopyGMToGMPerTokenV2(GlobalTensor<QuantOutType> dst, GlobalTensor<QuantScaleOutType> dstScale,
        GlobalTensor<QuantOutType> src, int32_t rows, int32_t hiddenSize, int32_t& pingpongId);

    __aicore__ inline void TilingByCore(int32_t totalLen, int32_t &coreLen, int32_t& coreOffset, int32_t align = 32)
    {
        int32_t coreIdx = GetBlockIdx(); // 0-63
        int32_t coreNum = GetBlockNum() * 2; // 32
        int32_t lenPerCore = CeilDiv((uint32_t)totalLen, coreNum) / align * align;
        coreLen = lenPerCore;
        coreOffset = coreIdx * lenPerCore;
        if (coreOffset + coreLen >= totalLen) {
            coreLen = totalLen - coreOffset;
        }
        if (coreOffset >= totalLen) {
            coreLen = 0;
        }
    }

    __aicore__ inline int64_t TokenPerExpertLayout(int64_t dim0, int64_t dim1, int64_t dim2)
    {
        return dim0 * perRankStridesInTokenPerExpert_ + dim1 * expertPerRank_ + dim2;
    }

    __aicore__ inline int64_t CeilAlign(int64_t a, int b)
    {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b * b;
    }

    __aicore__ inline void gmSignalWait(__gm__ int32_t *sig_addr, int32_t cmp_val) 
    {
        do {
            if (ReadGmByPassDCache(sig_addr) != cmp_val) {
                return;
            }
        } while (true);
    }

    __aicore__ inline void gmSignalWaitBarrier(__gm__ int32_t *sig_addr, int32_t cmp_val)
    {
        do {
            if (ReadGmByPassDCache(sig_addr) == cmp_val) {
                return;
            }
            // if (ReadGmByPassDCache(sig_addr) == cmp_val + 1) {
            //     return;
            // }
        } while (true);
    }

    TPipe* tpipe_{nullptr};
    TBuf<> CrossRankSyncBuf_;
    TBuf<> PrevSumBuf_;
    TBuf<> CopyTempBuf_;
    TBuf<> CumsumBuf_;
    TBuf<> ResetFlagBuf_;
    TBuf<> DataResBuf_;

    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    Params params_{};
    TupleShape problemShape_{};
    BlockOffset baseOffset_{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    GM_ADDR tokenPerExpertCurRankAddr_;

    GlobalTensor<int32_t> preSumBeforeRankWorkSpace_;
    GlobalTensor<int32_t> tokenPerExpertWin_;
    GlobalTensor<int32_t> cumsumInWorkSpace_;
    GlobalTensor<int32_t> groupFlagListWorkSpace_;
    GlobalTensor<int32_t> groupFlagListGm_;
    GlobalTensor<int32_t> groupFlagListGm2_;
    GlobalTensor<int32_t> groupList_;

    uint32_t m_ = 0;
    uint32_t k_ = 0;
    uint32_t topK_ = 0;
    uint32_t rankId_ = 0;
    uint32_t worldSize_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t groupListType_ = 0;
    uint64_t maxOutputSize_ = 0;
    int32_t prevSum_ = 0;
    int32_t dispatchRows_ = 0;
    int32_t vecSetSyncCom_ = 0;
    uint32_t startBlockIdx_ = 0;
    uint32_t dispatchBlockNumPerEP_ = 2;
    int32_t perRankStridesInTokenPerExpert_ = 0;
    uint32_t blockNum_ = GetBlockNum();
    uint32_t blockIdx_ = GetBlockIdx() / GetTaskRation();
    uint32_t aivCoreIdx_ = GetBlockIdx();
    uint32_t aicNum_ = blockNum_ / GetTaskRation();
    uint32_t subBlockIdx_ = GetSubBlockIdx();
    uint64_t preOffset_ = 0;
    uint32_t GMM2_PINGPONG_IDX = 0;
    uint32_t GMM2_PINGPONG_COUNT = 2;

    using L1TileShape = tla::Shape<tla::Int<128>, tla::Int<256>, tla::Int<512>>;
    using BlockEpilogue = BlockEpilogueSwigluQuant<L1TileShape, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, true>;
    BlockEpilogue epilogueOp_;
};

template <TemplateMoeDispatchFFNCombineTypeClass>
__aicore__ inline void MoeDispatchFFNCombine<TemplateMoeDispatchFFNCombineTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales, GM_ADDR yOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, TPipe *pipe, MoeDispatchFFNCombineTilingData *tilingData)
{
    tpipe_ = pipe;
    m_ = tilingData->m;
    k_ = tilingData->k;
    topK_ = tilingData->topK;
    rankId_ = tilingData->rank;
    worldSize_ = tilingData->EP;
    expertPerRank_ = tilingData->expertPerRank;
    dispatchBlockNumPerEP_ = tilingData->blockNumPerEP;
    dispatchRows_ = tilingData->dispatchRows;
    groupListType_ = tilingData->groupListType;
    maxOutputSize_ = tilingData->maxOutputSize;
    Get<N_VALUE>(problemShape_) = tilingData->n;
    Get<K_VALUE>(problemShape_) = k_;
    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext*>(context);
    for (int i = 0; i < worldSize_; i++) {
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
    }
    params_.aGmAddr = x;
    params_.expertIdxGmAddr = topkIds;
    params_.bGmAddr = weight1;
    params_.bScaleGmAddr = weightScales1;
    params_.b2GmAddr = weight2;
    params_.b2ScaleGmAddr = weightScales2;
    params_.y2GmAddr = yOut;
    params_.expertTokenNumsOutGmAddr = expertTokenNumsOut;
    params_.probsGmAddr = topkWeights;
    params_.workspaceInfo = WorkspaceInfo(workspaceGM, tilingData);
    params_.peermemInfo = PeermemInfo(winRankAddr_[rankId_], tilingData);
    params_.tilingData = tilingData;

    preSumBeforeRankWorkSpace_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrSumBeforeRank);
    tokenPerExpertWin_.SetGlobalBuffer((__gm__ int32_t*)params_.peermemInfo.ptrTokenPerExpert);
    cumsumInWorkSpace_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.ptrcumsumMM));
    groupList_ = cumsumInWorkSpace_[(worldSize_ - 1) * expertPerRank_];
    groupFlagListWorkSpace_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrFlagList);

    uint32_t maxResetBufAlign = CeilDiv(expertPerRank_ * INT_CACHELINE * MAX_AICORE_NUM * 2, blockNum_ * 2)
                                / ALIGN32 * ALIGN32 * sizeof(int32_t);
    tpipe_->InitBuffer(ResetFlagBuf_, maxResetBufAlign);
    ResetWorkSpaceFlagList();
    epilogueOp_.Init({params_.workspaceInfo.ptrA2, params_.workspaceInfo.ptrA2Scale, params_.workspaceInfo.ptrFlagList,
                    nullptr, nullptr, nullptr, 128, 256});
    perRankStridesInTokenPerExpert_ = CeilAlign(worldSize_ * expertPerRank_, ALIGN_128);
    tokenPerExpertCurRankAddr_ = params_.peermemInfo.ptrTokenPerExpert + TokenPerExpertLayout(rankId_, 0, 0) * sizeof(int32_t);
}

template <TemplateMoeDispatchFFNCombineTypeClass>
__aicore__ inline void MoeDispatchFFNCombine<TemplateMoeDispatchFFNCombineTypeFunc>::BuffInit()
{
    tpipe_->Reset();
    // 对CrossRankSync、Dispatch、GetCumSum过程涉及到的LocalTensor初始化管理
    uint32_t syncBufAlign = CeilAlign(worldSize_ * expertPerRank_, ALIGN_128) * sizeof(int32_t);
    uint32_t prevSumBufAlign = CeilAlign(worldSize_ * expertPerRank_, ALIGN_128)
                                * sizeof(int32_t) * expertPerRank_;
    tpipe_->InitBuffer(CrossRankSyncBuf_, syncBufAlign);
    tpipe_->InitBuffer(PrevSumBuf_, prevSumBufAlign);
    uint32_t getCumsumBufAlign = (AlignUp(worldSize_ * expertPerRank_, 128)) * worldSize_ * sizeof(int32_t);
    tpipe_->InitBuffer(CumsumBuf_, getCumsumBufAlign);
    tpipe_->InitBuffer(CopyTempBuf_, BUFFERALIGN * sizeof(QuantOutType));
    uint32_t dataResBufAlign = DATARES_MUL * k_ * sizeof(bfloat16_t);
    tpipe_->InitBuffer(DataResBuf_, dataResBufAlign);
}

template <TemplateMoeDispatchFFNCombineTypeClass>
__aicore__ inline void MoeDispatchFFNCombine<TemplateMoeDispatchFFNCombineTypeFunc>::ResetWorkSpaceFlagList()
{
    //workSpace groupFlagList清零
    groupFlagListGm_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrFlagList);
    if constexpr(g_coreType == AIV) {
        int32_t flagNum = expertPerRank_ * INT_CACHELINE * MAX_AICORE_NUM * 2;
        int32_t coreLen, coreOffset;
        TilingByCore(flagNum, coreLen, coreOffset);
        LocalTensor<int32_t> initTensor = ResetFlagBuf_.Get<int32_t>();
        Duplicate(initTensor, 0, coreLen);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(groupFlagListGm_[coreOffset], initTensor, coreLen);
    }
}

template <TemplateMoeDispatchFFNCombineTypeClass>
__aicore__ inline void MoeDispatchFFNCombine<TemplateMoeDispatchFFNCombineTypeFunc>::GetCumsumForMMAIV()
{
    int32_t expertPerRankAligned = (expertPerRank_ + 8 - 1) / 8 * 8;
    LocalTensor<int32_t> tmpBuffer1 = CumsumBuf_.Get<int32_t>();
    DataCopyPad(tmpBuffer1, tokenPerExpertWin_[rankId_ * expertPerRank_],
        {static_cast<uint16_t>(worldSize_), static_cast<uint16_t>(expertPerRank_ * sizeof(int32_t)),
         static_cast<uint16_t>((AlignUp(worldSize_ * expertPerRank_, 128) - expertPerRank_) * sizeof(int32_t)), 0},
        {});
    SyncFunc<AscendC::HardEvent::MTE2_V>();

    for (uint32_t i = 1; i < worldSize_; ++i) {
        Add(tmpBuffer1[i * expertPerRankAligned], tmpBuffer1[i * expertPerRankAligned], tmpBuffer1[(i - 1) * expertPerRankAligned], expertPerRank_);
        PipeBarrier<PIPE_V>();
    }
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(cumsumInWorkSpace_, tmpBuffer1, {static_cast<uint16_t>(worldSize_), static_cast<uint16_t>((expertPerRank_) * sizeof(int32_t)), 0, 0});
}

template <TemplateMoeDispatchFFNCombineTypeClass>
__aicore__ inline void MoeDispatchFFNCombine<TemplateMoeDispatchFFNCombineTypeFunc>::EndSync()
{
    if (vecSetSyncCom_ == 0) {
        return;
    }
    if constexpr(g_coreType == AIC) {
        WaitForVector();
    }
}

template <TemplateMoeDispatchFFNCombineTypeClass>
__aicore__ inline void MoeDispatchFFNCombine<TemplateMoeDispatchFFNCombineTypeFunc>::EndGMM2Sync()
{
    if constexpr(g_coreType == AIV) {
        return;
    }
    if (vecSetSyncCom_ <= 0) {
        return;
    } else if (vecSetSyncCom_ == 1) {
        WaitForVector();
    } else {
        WaitForVector(GMM2_PINGPONG_IDX);
        WaitForVector(1 - GMM2_PINGPONG_IDX);
    }
}

template <TemplateMoeDispatchFFNCombineTypeClass>
__aicore__ inline void MoeDispatchFFNCombine<TemplateMoeDispatchFFNCombineTypeFunc>::CrossRankSyncCntPerExpert()
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    uint32_t numPerCore = CeilAlign(worldSize_ * expertPerRank_, ALIGN_128);
    LocalTensor<int32_t> tmpBuffer = CrossRankSyncBuf_.Get<int32_t>();
    LocalTensor<int32_t> prevSumBuf = PrevSumBuf_.Get<int32_t>();
    // 给对端发送
    uint64_t offsetInDstWinAddr = perRankStridesInTokenPerExpert_ * sizeof(int32_t) * rank_ + PEERMEM_DATA_OFFSET + m_ * topK_ * (k_ + k_ / 32) sizeof(int8_t);
    for (int32_t dstEpIdx = aivCoreIdx_; dstEpIdx < worldSize_; dstEpIdx += blockNum_) {
        if (dstEpIdx == rankId_) {
            continue;
        }
        GlobalTensor<int32_t> srcAddress;
        srcAddress.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tokenPerExpertCurRankAddr_));
        GlobalTensor<int32_t> dstAddress;
        __gm__ void* dstPeermemPtr = GetRankWinAddrWithOffset(dstEpIdx, offsetInDstWinAddr);
        dstAddress.SetGlobalBuffer((__gm__ int32_t*)dstPeermemPtr);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        DataCopy(tmpBuffer, srcAddress, numPerCore);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Adds(tmpBuffer, tmpBuffer, 0x800000, numPerCore);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(dstAddress, tmpBuffer, numPerCore);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    }
    //本端接收
    int32_t intPer512 = ALIGN_512 / sizeof(int);
    for (int32_t dstEpIdx = aivCoreIdx_; dstEpIdx < worldSize_; dstEpIdx += blockNum_) {
        if (dstEpIdx != rankId_) {
            for (int32_t checkIdx = 0; checkIdx < numPerCore; checkIdx += intPer512) {
                __gm__ int32_t* syncCheck = reinterpret_cast<__gm__ int32_t*>(params_.peermemInfo.ptrTokenPerExpert) + TokenPerExpertLayout(dstEpIdx, 0, checkIdx);
                gmSignalWait(syncCheck, 0);
            }
            DataCopy(tmpBuffer, tokenPerExpertWin_[TokenPerExpertLayout(dstEpIdx, 0, 0)], numPerCore);
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            Adds(tmpBuffer, tmpBuffer, -0x800000, numPerCore);
            PipeBarrier<PIPE_V>();
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            DataCopy(tokenPerExpertWin_[TokenPerExpertLayout(dstEpIdx, 0, 0)], tmpBuffer, numPerCore);
        } else {
            DataCopy(tmpBuffer, tokenPerExpertWin_[TokenPerExpertLayout(dstEpIdx, 0, 0)], numPerCore);
            SyncFunc<AscendC::HardEvent::MTE2_V>();
        }
        PipeBarrier<PIPE_ALL>();
        int32_t tmpSum = 0;
        int32_t j = 0;
        for (int32_t i = 0; i < (rank_ + 1) * expertPerRank_; i++) {
            if (i >= rankId_ * expertPerRank_) {
                prevSumBuf(j) = tmpSum;
                j++;
            }
            tmpSum += tmpBuffer(i);
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(preSumBeforeRankWorkSpace_[dstEpIdx * expertPerRank_], prevSumBuf,
        DataCopyParams{1, static_cast<uint16_t>(expertPerRank_ * sizeof(int32_t)), 0, 0});
    }
    SyncAll<true>();
    if (aivCoreIdx_ == 0) {
        GetCumsumForMMAIV();
    }
    if (blockIdx_ < worldSize_ * dispatchBlockNumPerEP_) {
        prevSum_ = preSumBeforeRankWorkSpace_(blockIdx_ / dispatchBlockNumPerEP_ * expertPerRank_);
    }
}

template <TemplateMoeDispatchFFNCombineTypeClass>
__aicore__ inline void MoeDispatchFFNCombine<TemplateMoeDispatchFFNCombineTypeFunc>::CopyGMToGMPerTokenV2(
    GlobalTensor<QuantOutType> dst, GlobalTensor<QuantScaleOutType> dstScale, GlobalTensor<QuantOutType> src,
    int32_t rows, int32_t hiddenSize, int32_t& pingpongId) 
{
    constexpr int32_t BufferNum = 2;
    LocalTensor<QuantOutType> tmpBuffer1 = CopyTempBuf_.Get<QuantOutType>();
    LocalTensor<QuantOutType> tmpBuffer2 = tmpBuffer1[96 * 1024];
    uint32_t scaleLen = CeilDiv(hiddenSize, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    uint32_t copyInNum = hiddenSize + scaleLen;
    auto processCount = CeilDiv(rows, dispatchRows_);
    for (uint32_t processIndex = 0; processIndex < processCount; ++processIndex) {
        pingpongId = (pingpongId + 1) % BufferNum;
        TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<QuantOutType> buf = pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;
        LocalTensor<QuantScaleOutType> bufScale = buf[hiddenSize].template ReinterpretCast<QuantScaleOutType>();
        
        auto inputOffset = processIndex * dispatchRows_ * copyInNum;
        
        int32_t rowNum = dispatchRows_;
        if (processIndex == processCount - 1) {
            rowNum = rows - processIndex * dispatchRows_;
        }

        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        int64_t dataLen = rowNum * copyInNum;
        DataCopy(buf, src[inputOffset], dataLen);

        SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        auto outputOffset = processIndex * dispatchRows_ * hiddenSize;
        DataCopyPad(dst[outputOffset], 
            buf, {static_cast<uint16_t>(rowNum), static_cast<uint16_t>(hiddenSize), static_cast<uint16_t>(scaleLen / 32), 0, 0});
        DataCopyPad(dstScale[processIndex * dispatchRows_ * scaleLen], 
            bufScale, {static_cast<uint16_t>(rowNum), static_cast<uint16_t>(scaleLen), static_cast<uint16_t>(hiddenSize / 32), 0, 0});
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
    }
}

template <TemplateMoeDispatchFFNCombineTypeClass>
__aicore__ inline void MoeDispatchFFNCombine<TemplateMoeDispatchFFNCombineTypeFunc>::DispatchTokens(GMMAddrInfo &gmmAddrInfo, uint32_t expertIdx)
{
        int64_t widthA = Get<K_VALUE>(problemShape_);
        int64_t widthAScale = CeilDiv(widthA, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        int32_t pingpongId = 0;
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        for(uint32_t idx = aivCoreIdx_; idx < worldSize_ * dispatchBlockNumPerEP_; idx += blockNum_) {
            uint32_t dstEpIdx = idx / dispatchBlockNumPerEP_;
            uint32_t idxInEp = idx % dispatchBlockNumPerEP_;
            uint32_t rowStart = (dstEpIdx == 0 ? 0 : cumsumInWorkSpace_((dstEpIdx - 1) * expertPerRank_ + expertIdx));
            if (rowStart < maxOutputSize_) {
                uint32_t rows = tokenPerExpertWin_(TokenPerExpertLayout(dstEpIdx, rankId_, expertIdx));
                if (rowStart + rows > maxOutputSize_) {
                    rows = maxOutputSize_ - rowStart;
                }
                uint32_t rowSrc = prevSum_;
                prevSum_ += rows;
                uint32_t rowsPerCore = CeilDiv(rows, dispatchBlockNumPerEP_);
                uint32_t rowStartPerCore = rowStart + idxInEp * rowsPerCore;
                uint32_t rowSrcPerCore = rowSrc + idxInEp * rowsPerCore;
                if (rowStartPerCore + rowsPerCore > rowStart + rows) {
                    rowsPerCore = rowStart + rows - rowStartPerCore;
                }
                if (rowsPerCore > 0) {
                    GlobalTensor <QuantOutType> gmRemoteA, aGlobalTensor;
                    GlobalTensor <QuantScaleOutType> aScaleGlobalTensor;
                    int64_t gmOffsetA = rowStartPerCore * widthA;
                    int64_t gmOffsetAScale = rowStartPerCore * widthAScale;
                    int64_t gmOffsetPeer = rowSrcPerCore * (widthA + widthAScale);
                    gmRemoteA.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType *>(GetRankWinAddrWithOffset(dstEpIdx, PEERMEM_DATA_OFFSET)));
                    aGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType *>(gmmAddrInfo.aGlobal));
                    aScaleGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleOutType *>(gmmAddrInfo.aScaleGlobal));
                    CopyGMToGMPerTokenV2(aGlobalTensor[gmOffsetA], aScaleGlobalTensor[gmOffsetAScale], gmRemoteA[gmOffsetPeer], rowsPerCore, widthA, pingpongId);
                }
            }
            SyncFunc<AscendC::HardEvent::MTE3_S>();
            AtomicAdd((__gm__ int32_t*)(groupFlagListGm2_.GetPhyAddr()), int32_t(1));
        }
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <TemplateMoeDispatchFFNCombineTypeClass>
__aicore__ inline bool MoeDispatchFFNCombine<TemplateMoeDispatchFFNCombineTypeFunc>::UpdateGroupParams(uint32_t expertIdx)
{
    // baseOffset_ is 0 when groupIdx = 0
    if (expertIdx != 0) {
        uint64_t m = Get<M_VALUE>(problemShape_);
        uint64_t n = Get<N_VALUE>(problemShape_);
        uint64_t k = Get<K_VALUE>(problemShape_);
        Get<IDX_A_OFFSET>(baseOffset_) += m * k;
        Get<IDX_B_OFFSET>(baseOffset_) += n * k;
        // only splitM
        auto scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        // scaleAAxisBaseOffset (m, ceil(k,64), 2)
        Get<IDX_A_SCALE_OFFSET>(baseOffset_) += m * scaleK;
        // scaleBAxisBaseOffset (g, n, ceil(k,64), 2) or (g, ceil(k,64), n, 2)
        Get<IDX_B_SCALE_OFFSET>(baseOffset_) += n * scaleK;
        Get<IDX_C_OFFSET>(baseOffset_) += m * n / SWIGLU_N_HALF;
        Get<IDX_C_SCALE_OFFSET>(baseOffset_) +=
            m * CeilDiv(n / SWIGLU_N_HALF, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_FLAG_OFFSET>(baseOffset_) += 1;
        Get<IDX_B2_OFFSET>(baseOffset_) += k * n / SWIGLU_N_HALF;
        Get<IDX_B2_SCALE_OFFSET>(baseOffset_) += k * CeilDiv(n / SWIGLU_N_HALF, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_Y2_OFFSET>(baseOffset_) += m * k;
        Get<IDX_M_OFFSET>(baseOffset_) += m;
    }

    int32_t splitValue = 0;
    if (groupListType_ == 0) {
        int32_t offset = static_cast<int32_t>(groupList_.GetValue(expertIdx));
        splitValue = offset - preOffset_;
        preOffset_ = offset;
    } else {
        splitValue = static_cast<uint64_t>(groupList_.GetValue(expertIdx));
    }

    Get<M_VALUE>(problemShape_) = splitValue;
    // split_m，when m=0, skip
    if (Get<M_VALUE>(problemShape_) == 0) {
        return false;
    }
    return true;
}

template <TemplateMoeDispatchFFNCombineTypeClass>
__aicore__ inline void MoeDispatchFFNCombine<TemplateMoeDispatchFFNCombineTypeFunc>::UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo, uint32_t gmmIndex)
{
    if (gmmIndex == 0) {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.ptrA + Get<IDX_A_OFFSET>(baseOffset_) * sizeof(QuantOutType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.ptrAScale + Get<IDX_A_SCALE_OFFSET>(baseOffset_) * sizeof(QuantScaleOutType);
        if constexpr(g_coreType == AIC) {
            gmmAddrInfo.bGlobal = params_.bGmAddr + Get<IDX_B_OFFSET>(baseOffset_) * sizeof(QuantOutType);
            gmmAddrInfo.bScaleGlobal = params_.bScaleGmAddr + Get<IDX_B_SCALE_OFFSET>(baseOffset_) * sizeof(QuantScaleOutType);
        } else {
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                Get<IDX_C_OFFSET>(baseOffset_), Get<IDX_C_SCALE_OFFSET>(baseOffset_),
                Get<IDX_FLAG_OFFSET>(baseOffset_), 0L, 0L, 0L};
                epilogueOp_.UpdateGlobalAddr(vecBaseOffset);
        }
    } else if constexpr(g_coreType == AIC) {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.ptrA2 + Get<IDX_C_OFFSET>(baseOffset_) * sizeof(QuantOutType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.ptrA2Scale + Get<IDX_C_SCALE_OFFSET>(baseOffset_) * sizeof(QuantScaleOutType);
        gmmAddrInfo.bGlobal = params_.b2GmAddr + Get<IDX_B2_OFFSET>(baseOffset_) * sizeof(QuantOutType);
        gmmAddrInfo.bScaleGlobal = params_.b2ScaleGmAddr + Get<IDX_B2_SCALE_OFFSET>(baseOffset_) * sizeof(QuantScaleOutType);
    }

    groupFlagListGm_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrFlagList + Get<IDX_FLAG_OFFSET>(baseOffset) * MAX_AICORE_NUM);
    groupFlagListGm2_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrFlagList2 + Get<IDX_FLAG_OFFSET>(baseOffset) * MAX_AICORE_NUM);
}

template <TemplateMoeDispatchFFNCombineTypeClass>
__aicore__ inline void MoeDispatchFFNCombine<TemplateMoeDispatchFFNCombineTypeFunc>::ArgSortExpandedRowIdx()
{
    int32_t coreLen, coreOffset;
    TilingByCore(m_ * topK_, coreLen, coreOffset, 1);
    __gm__ int32_t *src = (__gm__ int32_t *)params_.workspaceInfo.expandedRowIdx;
    __gm__ int32_t *dst = (__gm__ int32_t *)params_.workspaceInfo.expandedRowIdxGather;
    Simt::VF_CALL<ArgsortSimt>(Simt::Dim3{MoeInitRoutingV3::SIMT_THREAD_NUM, 1, 1}, coreLen, coreOffset,
                                src, dst);
}

template <TemplateMoeDispatchFFNCombineTypeClass>
__aicore__ inline void MoeDispatchFFNCombine<TemplateMoeDispatchFFNCombineTypeFunc>::ResetTokenPerExpert()
{
    if (aivCoreIdx_ != blockNum_ - 1) {
        return;
    }
    int32_t num = worldSize_ * CeilAlign(worldSize_ * expertPerRank_, ALIGN_128);
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    LocalTensor<int32_t> tmp;
    Duplicate(tmp, 0, num);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(tokenPerExpertWin_, tmp, num);
}

template <TemplateMoeDispatchFFNCombineTypeClass>
__aicore__ inline void MoeDispatchFFNCombine<TemplateMoeDispatchFFNCombineTypeFunc>::Unpermute()
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    int32_t coreLen, coreOffset;
    TilingByCore(m_, coreLen, coreOffset, 1);
    GlobalTensor<int32_t> expandedRowIdx;
    expandedRowIdx.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.expandedRowIdxGather);
    GlobalTensor<bfloat16_t> probs;
    probs.SetGlobalBuffer((__gm__ bfloat16_t*)params_.probsGmAddr);
    GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer((__gm__ bfloat16_t*)params_.peermemInfo.ptrD);
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer((__gm__ bfloat16_t*)params_.y2GmAddr);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    for(int32_t tokenIdx = coreOffset; tokenIdx < coreLen + coreOffset; tokenIdx++) {
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        LocalTensor<bfloat16_t> dataRes = DataResBuf_.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> dataIn0 = dataRes[k_];
        LocalTensor<bfloat16_t> dataIn1 = dataRes[k_ * 2];
        for(int32_t expId = 0; expId < topK_; ++expId) {
            int32_t rowIdx = expandedRowIdx(tokenIdx * topK_ + expId);
            bfloat16_t expScale = probs(tokenIdx * topK_ + expId);
            auto event = (expId % 2 == 0) ? EVENT_ID0 : EVENT_ID1;
            auto dataIn = (expId % 2 == 0) ? dataIn0 : dataIn1;
            WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            DataCopy(dataIn, expandedX[rowIdx * k_], k_);
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            SetFlag<AscendC::HardEvent::S_V>(event);
            WaitFlag<AscendC::HardEvent::S_V>(event);
            if (expId == 0) {
                Muls(dataRes, dataIn, expScale, k_);
            } else {
                Muls(dataIn, dataIn, expScale, k_);
                PipeBarrier<PIPE_V>();
                Add(dataRes, dataRes, dataIn, k_);
                PipeBarrier<PIPE_V>();
            }
            SetFlag<AscendC::HardEvent::V_MTE2>(event);
        }
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(output[tokenIdx * k_], dataRes, k_);
    }
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
}

template <TemplateMoeDispatchFFNCombineTypeClass>
__aicore__ inline void MoeDispatchFFNCombine<TemplateMoeDispatchFFNCombineTypeFunc>::CrossRankSyncInWorldSize()
{
    __gm__ int32_t* syncRank = (__gm__ int32_t*)(params_.peermemInfo.base);
    __gm__ int32_t* syncCount = (__gm__ int32_t*)(params_.peermemInfo.base + 2048);
    int count = *(syncCount + aivCoreIdx_ * 16) + 1;

    for (int i = aivCoreIdx_; i < worldSize_; i += blockNum_) {
        __gm__ int32_t* syncRemoteAddr = (__gm__ int32_t*)(winRankAddr_[i]) + rankId_ * 16;
        WriteGmByPassDCache(syncRemoteAddr, count);
        auto syncCheck = syncRank + i * 16;
        gmSignalWaitBarrier(syncCheck, count);
    }
    SyncAll<true>();
    *(syncCount + aivCoreIdx_ * 16) = count;
}

template <TemplateMoeDispatchFFNCombineTypeClass>
__aicore__ inline void MoeDispatchFFNCombine<TemplateMoeDispatchFFNCombineTypeFunc>::Process()
{
    moe_init_routing_v3<bfloat16_t, fp8_e4m3fn_t>(params_.aGmAddr, params_.expertIdxGmAddr, nullptr, nullptr,
            params_.peermemInfo.ptrA0, params_.workspaceInfo.expandedRowIdx, tokenPerExpertCurRankAddr_, nullptr,
            params_.workspaceInfo.ptrA, tpipe_, &params_.tilingData->moeInitRoutingTilingData, 
            params_.tilingData->moeInitRoutingTilingData.tilingKey);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
    BuffInit();

    CrossRankSyncCntPerExpert();
    SyncAll<false>();

    GMMAddrInfo gmmAddrInfo;
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
        if (!UpdateGroupParams(expertIdx)) {
            continue;
        }
        UpdateGlobalBuffer(gmmAddrInfo, 0);
        if constexpr(g_coreType == AscendC::AIV) {
            if(subBlockIdx_ == 1) {
                DispatchTokens(gmmAddrInfo, expertIdx);
            }
        }
        MoeDispatchFFNCombineCatlassImpl::GroupMatmulSwigluQuant<QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
            epilogueOp_, tpipe_, params_, problemShape_, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_);
    }
    EndSync();

    preOffset_ = 0;
    vecSetSyncCom_ = 0;
    baseOffset_ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
        if (!UpdateGroupParams(expertIdx)) {
            continue;
        }
        UpdateGlobalBuffer(gmmAddrInfo, 1);
        MoeDispatchFFNCombineCatlassImpl::GroupMatmul2<QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
            tpipe_, params_, problemShape_, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_, expertIdx);
    }
    EndGMM2Sync();
    SyncAll<true>();

    ArgSortExpandedRowIdx();
    if constexpr(g_coreType == AIV) {
        ResetTokenPerExpert();
        CrossRankSyncInWorldSize();
    }
    Unpermute();
}

}   // namespace MoeDispatchFFNCombineImpl
#endif