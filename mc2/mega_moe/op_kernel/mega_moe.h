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
 * \file mega_moe.h
 * \brief
 */

#ifndef MEGA_MOE_H
#define MEGA_MOE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#include "kernel_operator_list_tensor_intf.h"
#include "mega_moe_base.h"
#include "mega_moe_workspace_info.h"
#include "block_epilogue_swiglu_mx_quant.h"
#include "mega_moe_impl.h"


using namespace AscendC;

namespace MegaMoeImpl {
using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
using BlockOffset = Shape<int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t>;

#define TemplateMegaMoeTypeClass typename XType, typename OutputType, typename TopkWeightsType, int32_t QuantMode
#define TemplateMegaMoeTypeFunc XType, OutputType, TopkWeightsType, QuantMode

template <TemplateMegaMoeTypeClass>
class MegaMoe {
public:
    using QuantOutType = typename std::conditional<((QuantMode >= 3) && (QuantMode != 4)),
        fp8_e5m2_t, fp8_e4m3fn_t>::type;
    using QuantScaleOutType = typename std::conditional<(QuantMode >= 3), fp8_e8m0_t, float>::type;
    __aicore__ inline MegaMoe() {};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights,
        GM_ADDR weight1, GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
        GM_ADDR scales, GM_ADDR yOut, GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM,
        MegaMoeTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void FirstBuffInit(TPipe *tPipe);
    __aicore__ inline void SecondBuffInit(TPipe *tPipe);
    __aicore__ inline void ResetWorkSpaceFlagList();
    __aicore__ inline void GetCumsumForMMAIV();
    __aicore__ inline void TopKIdsAllGather();
    __aicore__ inline void ExpertTokenNumCopyOut();
    __aicore__ inline void TripleInfoCalAndDispatch(GMMAddrInfo &gmmAddrInfo);
    __aicore__ inline bool UpdateGroupParams(uint32_t expertIdx);
    __aicore__ inline void UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo, uint32_t gmmIndex);
    __aicore__ inline void ArgSortExpandedRowIdx();
    __aicore__ inline void ResetTokenPerExpert();
    __aicore__ inline void Unpermute();
    __aicore__ inline void EndSync();
    __aicore__ inline void EndGMM2Sync();
    __aicore__ inline void CrossRankSyncInWorldSize();
    __aicore__ inline void CopyGMToGMPerToken(GlobalTensor<QuantOutType> dst,
                GlobalTensor<QuantScaleOutType> dstScale, GlobalTensor<QuantOutType> src,
                int32_t rows, int32_t hiddenSize, int32_t& pingpongId);

    __aicore__ inline void TilingByCore(int32_t totalLen, int32_t &coreLen,
        int32_t& coreOffset, int32_t align = ALIGN_32)
    {
        int32_t coreIdx = GetBlockIdx();
        int32_t coreNum = GetBlockNum() * 2; // 取到vec核总数
        int32_t lenPerCore = Ops::Base::CeilDiv(static_cast<uint32_t>(totalLen), static_cast<uint32_t>(coreNum));
        int32_t lenPerCoreAlign = Ops::Base::CeilAlign(static_cast<uint32_t>(lenPerCore), static_cast<uint32_t>(align));
        coreLen = lenPerCoreAlign;
        coreOffset = coreIdx * lenPerCoreAlign;
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

    __aicore__ inline void GmSignalWait(__gm__ int32_t *sig_addr, int32_t cmp_val)
    {
        do {
            if (ReadGmByPassDCache(sig_addr) != cmp_val) {
                return;
            }
        } while (true);
    }

    __aicore__ inline void GmSignalWaitBarrier(__gm__ int32_t *sig_addr, int32_t cmp_val)
    {
        do {
            if (ReadGmByPassDCache(sig_addr) == cmp_val) {
                return;
            }
        } while (true);
    }

    TBuf<> CrossRankSyncBuf_;
    TBuf<> PrevSumBuf_;
    TBuf<> CopyTempBuf_;
    TBuf<> CumsumBuf_;
    TBuf<> ResetFlagBuf_;
    TBuf<> DataResBuf_;
    TBuf<> DataResFp32Buf_;
    TBuf<> ResetTokenBuf_;
    TBuf<> topKWeightsBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> topkIdsQueue_;

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
    GlobalTensor<int32_t> expertTokenNumsOut_;
    LocalTensor<float> topKWeightsTensor_;

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
    uint32_t blockNumPerRank_ = 2;
    int32_t perRankStridesInTokenPerExpert_ = 0;
    uint32_t blockNum_ = GetBlockNum();
    uint32_t blockAivNum_ = GetBlockNum() * 2;
    uint32_t blockIdx_ = GetBlockIdx() / GetTaskRation();
    uint32_t aivCoreIdx_ = GetBlockIdx();
    uint32_t subBlockIdx_ = GetSubBlockIdx();

    uint64_t preOffset_ = 0;
    uint16_t gmm2PingPongIdx_ = 0;

    using BlockEpilogue = BlockEpilogueSwigluMxQuant<QuantOutType, float, QuantScaleOutType, QuantScaleOutType, true>;
    BlockEpilogue epilogueOp_;
};

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales, GM_ADDR yOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData *tilingData)
{
    m_ = tilingData->bs;
    k_ = tilingData->h;
    topK_ = tilingData->topK;
    worldSize_ = tilingData->epWorldSize;
    expertPerRank_ = tilingData->expertPerRank;
    blockNumPerRank_ = tilingData->blockNumPerEP;
    dispatchRows_ = tilingData->dispatchRows;
    groupListType_ = tilingData->groupListType;
    maxOutputSize_ = tilingData->maxOutputSize;
    Get<N_VALUE>(problemShape_) = tilingData->hiddenDim;
    Get<K_VALUE>(problemShape_) = k_;
    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext*>(context);
    rankId_ = mc2Context_->epRankId;
    for (int i = 0; i < worldSize_; i++) {
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
    }
    params_.aGmAddr = x;
    params_.topkIdsGmAddr = topkIds;
    params_.bGmAddr = GetTensorAddr(0, weight1);
    params_.b2GmAddr = GetTensorAddr(0, weight2);
    params_.bScaleGmAddr = GetTensorAddr(0, weightScales1);
    params_.b2ScaleGmAddr = GetTensorAddr(0, weightScales2);

    params_.y2GmAddr = yOut;
    params_.expertTokenNumsOutGmAddr = expertTokenNumsOut;
    params_.probsGmAddr = topkWeights;
    params_.workspaceInfo = WorkspaceInfo(workspaceGM, tilingData);
    params_.peermemInfo = PeermemInfo(winRankAddr_[rankId_], tilingData);
    params_.tilingData = tilingData;
    preSumBeforeRankWorkSpace_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrSumBeforeRank);
    tokenPerExpertWin_.SetGlobalBuffer((__gm__ int32_t*)params_.peermemInfo.topKInfoInWorldSizePtr);
    cumsumInWorkSpace_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.ptrcumsumMM));
    expertTokenNumsOut_.SetGlobalBuffer((__gm__ int32_t*)params_.expertTokenNumsOutGmAddr);
    groupList_ = cumsumInWorkSpace_[(worldSize_ - 1) * expertPerRank_];
    groupFlagListWorkSpace_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrFlagSwiGluToGmm2);
    epilogueOp_.Init({params_.workspaceInfo.ptrA2, params_.workspaceInfo.ptrA2Scale,
        params_.workspaceInfo.ptrFlagSwiGluToGmm2, nullptr, nullptr, nullptr, ALIGN_128, ALIGN_256});
    perRankStridesInTokenPerExpert_ = Ops::Base::CeilAlign(
        static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128));
    tokenPerExpertCurRankAddr_ = params_.peermemInfo.topKInfoInWorldSizePtr +
        TokenPerExpertLayout(rankId_, 0, 0) * sizeof(int32_t);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::FirstBuffInit(TPipe *tPipe)
{
    tpipe_->InitBuffer(topkIdsQueue_, 1, Ops::Base::CeilAlign(m_ * topK_ * sizeof(int32_t)), ALIGN_32); // 暂时未开DB
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::SecondBuffInit(TPipe *tPipe)
{
    uint32_t dataResBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(UNPERMUTE_LIST_NUM * k_ * sizeof(bfloat16_t)), static_cast<uint32_t>(ALIGN_32));
    tPipe->InitBuffer(DataResBuf_, dataResBufAlign);
    int32_t num = worldSize_ * Ops::Base::CeilAlign(
        static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128)) * sizeof(int32_t);
    tPipe->InitBuffer(ResetTokenBuf_, num);
    tPipe->InitBuffer(DataResFp32Buf_, dataResBufAlign * 2);
    tPipe->InitBuffer(topKWeightsBuf_, Ops::Base::CeilAlign(
        static_cast<uint32_t>(m_ * topK_ * sizeof(float)), static_cast<uint32_t>(ALIGN_32)));
    topKWeightsTensor_ = topKWeightsBuf_.Get<float>();
    if constexpr (Std::IsSame<TopkWeightsType, float>::value) {
        GlobalTensor<float> topKWeightsGlobalTensor_;
        topKWeightsGlobalTensor_.SetGlobalBuffer((__gm__ float*)params_.probsGmAddr);
        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(m_ * topK_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(topKWeightsTensor_, topKWeightsGlobalTensor_, copyParams, copyPadParams);
    }
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        TBuf<> tempBuf;
        tPipe->InitBuffer(tempBuf, Ops::Base::CeilAlign(
            static_cast<uint32_t>(m_ * topK_ * sizeof(bfloat16_t)), uint32_t(ALIGN_32)));
        LocalTensor<bfloat16_t> tempLocal = tempBuf.Get<bfloat16_t>();
        GlobalTensor<bfloat16_t> topkWeightsGlobalTensor;
        topkWeightsGlobalTensor.SetGlobalBuffer((__gm__ bfloat16_t*)params_.probsGmAddr);
        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(m_ * topK_ * sizeof(bfloat16_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<bfloat16_t> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(tempLocal, topkWeightsGlobalTensor, copyParams, copyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Cast(topKWeightsTensor_, tempLocal, AscendC::RoundMode::CAST_NONE, m_ * topK_);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ResetWorkSpaceFlagList()
{
    // workSpace groupFlagList清零
    groupFlagListGm_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrFlagSwiGluToGmm2);
    if constexpr(g_coreType == AIV) {
        int32_t flagNum = expertPerRank_ * INT_CACHELINE * 2;
        int32_t coreLen, coreOffset;
        TilingByCore(flagNum, coreLen, coreOffset);
        LocalTensor<int32_t> initTensor = ResetFlagBuf_.Get<int32_t>();
        if (coreLen != 0) {
            Duplicate(initTensor, 0, coreLen);
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            DataCopy(groupFlagListGm_[coreOffset], initTensor, coreLen);
        }
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::GetCumsumForMMAIV()
{
    int32_t expertPerRankAligned = (expertPerRank_ + 8 - 1) / 8 * 8;
    LocalTensor<int32_t> tmpBuffer1 = CumsumBuf_.Get<int32_t>();
    DataCopyPad(tmpBuffer1, tokenPerExpertWin_[rankId_ * expertPerRank_],
        {static_cast<uint16_t>(worldSize_), static_cast<uint16_t>(expertPerRank_ * sizeof(int32_t)),
         static_cast<uint16_t>((AlignUp(worldSize_ * expertPerRank_, ALIGN_128)
             - expertPerRank_) * sizeof(int32_t)), 0},
        {});
    SyncFunc<AscendC::HardEvent::MTE2_V>();

    for (uint32_t i = 1; i < worldSize_; ++i) {
        Add(tmpBuffer1[i * expertPerRankAligned], tmpBuffer1[i * expertPerRankAligned],
            tmpBuffer1[(i - 1) * expertPerRankAligned], expertPerRank_);
        PipeBarrier<PIPE_V>();
    }
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(cumsumInWorkSpace_, tmpBuffer1,
        {static_cast<uint16_t>(worldSize_), static_cast<uint16_t>((expertPerRank_) * sizeof(int32_t)), 0, 0});
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ExpertTokenNumCopyOut()
{
    int32_t copyNum = Ops::Base::CeilAlign(static_cast<uint32_t>(expertPerRank_), static_cast<uint32_t>(ALIGN_32));
    LocalTensor<int32_t> tmpCopyBuffer = CrossRankSyncBuf_.Get<int32_t>();
    DataCopy(tmpCopyBuffer, groupList_, copyNum);
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    DataCopy(expertTokenNumsOut_, tmpCopyBuffer, copyNum);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::EndSync()
{
    if (vecSetSyncCom_ == 0) {
        return;
    }
    if constexpr(g_coreType == AIC) {
        WaitForVector();
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::EndGMM2Sync()
{
    if constexpr(g_coreType == AIV) {
        return;
    }
    if (vecSetSyncCom_ <= 0) {
        return;
    } else if (vecSetSyncCom_ == 1) {
        WaitForVector();
    } else {
        WaitForVector(gmm2PingPongIdx_);
        WaitForVector(1 - gmm2PingPongIdx_);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::TopKIdsAllGather()
{
    // topKInfo给对端发送，当前分核逻辑为一个核负责一张卡的发送
    GlobalTensor<int32_t> srcGlobalTensor;
    GlobalTensor<int32_t> dstGlobalTensor;
    LocalTensor<int32_t> topkIdsTensor;
    srcGlobalTensor.SetGlobalBuffer((__gm__ int32_t*)params_.topkIdsGmAddr);
    DataCopyExtParams dataCopyParams{1U, static_cast<uint32_t>(m_ * topK_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
    topkIdsTensor = topkIdsQueue_.AllocTensor<int32_t>();
    DataCopyPad(topkIdsTensor, tpGlobal[startExpertId_], dataCopyParams, copyPadParams);
    topkIdsQueue_.EnQue(topkIdsTensor);
    topkIdsTensor = topkIdsQueue_.DeQue<int32_t>();
    for (int32_t dstRankIdx = aivCoreIdx_; dstRankIdx < worldSize_; dstRankIdx += blockAivNum_) {
        uint64_t offsetInDstWinAddr = PEERMEM_DATA_OFFSET + rankId_ * Ops::Base::CeilAlign((int64_t)(m_ * topK_ * sizeof(int32_t)), (int64_t)ALIGN_512);
        dstGlobalTensor.SetGlobalBuffer((__gm__ int32_t*)GetRankWinAddrWithOffset(dstRankIdx, offsetInDstWinAddr));
        DataCopyPad(dstGlobalTensor, topkIdsTensor, dataCopyParams);
    }
    topkIdsQueue_.FreeTensor(topkIdsTensor);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::QuantProcessInRank()
{
    // 分核，按照BS与aivCoreNum均分
    int32_t curentNum;
    int32_t curentOffset;
    TilingByCore(m_, curentNum, curentOffset, 1);
    GlobalTensor<XType> srcGlobalTensor;
    GlobalTensor<QuantOutType> dstGlobalTensor;
    uint64_t quantPerTokenOffset = (k_ + k_ / MXFP_SCALE_GROUP_NUM) * sizeof(int8_t);
    for (int32_t index = 0; index < curentNum; index++) {
        srcGlobalTensor.SetGlobalBuffer((__gm__ XType*)(params_.aGmAddr + (curentOffset + index) * k_ * sizeof(bfloat16_t)));
        dstGlobalTensor.SetGlobalBuffer((__gm__ QuantOutType*)(params_.PeermemInfo.quantTokenScalePtr + (curentOffset + index) * quantPerTokenOffset));
        ProcessToken(srcGlobalTensor, dstGlobalTensor); // 量化调用参考dispatch
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::SendInfoCal(uint32_t localExpertId)
{ 
    int32_t sendCnt = 0;
    int32_t totalSendCnt = 0;
    int32_t compareCount = Ops::Base::CeilAlign(m_ * topK_ * sizeof(int32_t), ALIGN_256);
    maskAlignSize_ = Ops::Base::CeilAlign(Ceil(m_ * topK_, 8), ALIGN_32);
    Duplicate<uint32_t>(cumsumInfoTensor_, 0, worldSize_ * 8);
    Duplicate<uint32_t>(gatherMaskInt32Tensor, 0, Ceil(worldSize_ * maskAlignSize_ * sizeof(uint8_t), ALIGN_32) / sizeof(uint32_t));
    GlobalTensor<int32_t> srcGlobalTensor;
    DataCopyExtParams dataCopyParams{1U, static_cast<uint32_t>(m_ * topK_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadParams{true, 0U, Ops::Base::CeilAlign(Ceil(m_ * topK_, 8)) - m_ * topK_, 0U}; // 填充数据为0
    for (int32_t calRankId = 0; calRankId < worldSize_; calRankId++) {
        uint64_t offsetInDstWinAddr = PEERMEM_DATA_OFFSET + calRankId * Ops::Base::CeilAlign((int64_t)(m_ * topK_ * sizeof(int32_t)), (int64_t)ALIGN_512);
        srcGlobalTensor.SetGlobalBuffer((__gm__ int32_t*)GetRankWinAddrWithOffset(calRankId, offsetInDstWinAddr));
        DataCopyPad(topkTensor, srcGlobalTensor, dataCopyParams, copyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        CompareScalar(gatherMask_[calRankId * maskAlignSize_], topkTensor, static_cast<half>(localExpertId), AscendC::CMPMODE::EQ, compareCount); // mask位置直接存下来
        GatherMask(validBsIndexTensor_, bsIndexTensor, maskTensorInt32, true, mask, {1, 1, 0, 0}, sendCnt);
        SyncFunc<AscendC::HardEvent::V_S>();
        totalSendCnt += sendCnt;
        cumsumInfoTensor_.SetValue(calRankId * 8, totalSendCnt);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::CopyGMToGMPerToken(int32_t rowDstOffsetPerCore, int32_t remoteRankIdx,
    int32_t copyStartIdx, int32_t copyEndIdx, int32_t hiddenSize, int32_t& pingpongId)
{
    constexpr int32_t BufferNum = 2;
    uint32_t scaleLen = Ops::Base::CeilDiv(static_cast<int32_t>(hiddenSize), static_cast<int32_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    uint32_t copyInNum = hiddenSize + scaleLen;
    int64_t widthA = Get<K_VALUE>(problemShape_); // H 长度
    int64_t widthAScale = Ops::Base::CeilDiv(widthA, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    LocalTensor<QuantOutType> tmpBuffer1 = CopyTempBuf_.Get<QuantOutType>();
    LocalTensor<QuantOutType> tmpBuffer2 = tmpBuffer1[96 * 1024];
    GlobalTensor<QuantOutType> remoteRankGlobalTensor;
    GlobalTensor<QuantOutType> tokenRevGlobalTensor;
    GlobalTensor<QuantScaleOutType> scaleRevGlobalTensor;
    tokenRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(gmmAddrInfo.aGlobal + rowDstOffsetPerCore * widthA));
    scaleRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleOutType *>(gmmAddrInfo.aScaleGlobal + rowDstOffsetPerCore * widthAScale));
    remoteRankGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(GetRankWinAddrWithOffset(remoteRankIdx,
        PEERMEM_DATA_OFFSET + worldSize_ * topkOffsetPerRank_)));
    for (uint32_t copyIndex = copyStartIdx; copyIndex < copyEndIdx; ++copyIndex) {
        pingpongId = (pingpongId + 1) % BufferNum;
        TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<QuantOutType> tokenScalebuf = pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;
        LocalTensor<QuantScaleOutType> bufScale = tokenScalebuf[widthA].template ReinterpretCast<QuantScaleOutType>();
        int32_t topkId = validTopkIdsTensor_.GetValue(copyIndex);
        int32_t topkIndex = validTopkIndexTensor_.GetValue(copyIndex);
        tripleTensor_[rowDstOffsetPerCore + copyIndex].SetValue(0, remoteRankIdx);
        tripleTensor_[rowDstOffsetPerCore + copyIndex].SetValue(1, topkId);
        tripleTensor_[rowDstOffsetPerCore + copyIndex].SetValue(2, topkIndex);
        auto remoteCopyOffset = (topkIndex / topK_) * copyInNum;
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        DataCopy(tokenScalebuf, remoteRankGlobalTensor[remoteCopyOffset], copyInNum);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        auto outputOffset = copyIndex * widthA;
        DataCopyPad(tokenRevGlobalTensor[outputOffset], tokenScalebuf,
            {1, static_cast<uint16_t>(widthA),
            static_cast<uint16_t>(scaleLen / ALIGN_32), 0, 0});
        DataCopyPad(scaleRevGlobalTensor[copyIndex * scaleLen], bufScale,
            {1, static_cast<uint16_t>(scaleLen), static_cast<uint16_t>(widthA / ALIGN_32), 0, 0});
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
    }
    DataCopy(tripleGloablTensor[rowDstOffsetPerCore * 8], tripleTensor_, (copyEndIdx - copyStartIdx) * 8); // 三元组搬出
}


template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::TripleInfoCalAndDispatch(GMMAddrInfo &gmmAddrInfo)
{
    // int32_t coreLen = 0
    // int32_t coreOffset = 0;
    // TilingByCore(totalSendCnt_, coreLen, coreOffset);
    // Cumsum();
    // CompareScalar(mask, coreOffset);
    // ScalarGetSFFValue(maskInt64);
    // 当前先保持原有按照卡分核，后续换成按照token分核Cumsum -> ComapreScalar -> ScalarGetSFFValue(拿到起始位置)

    sendTotalNum_ = m_ * topK_;
    topkOffsetPerRank_ = Ops::Base::CeilAlign((int64_t)(sendTotalNum_ * sizeof(int32_t)), (int64_t)ALIGN_512);
    int32_t pingpongId = 0;
    int32_t sendToCurExpTokenCnt = 0;
    GlobalTensor<int32_t> srcGlobalTensor;
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (uint32_t idx = blockIdx_; idx < worldSize_ * blockNumPerRank_; idx += blockNum_) {     // 需要添加本卡的cnt对应到原始卡上的哪个bs
        uint32_t dstRankIdx = idx / blockNumPerRank_;
        uint64_t offsetInDstWinAddr = PEERMEM_DATA_OFFSET + calRankId * topkOffsetPerRank_;
        srcGlobalTensor.SetGlobalBuffer((__gm__ int32_t*)GetRankWinAddrWithOffset(dstRankIdx, offsetInDstWinAddr));
        DataCopyPad(topkTensor, srcGlobalTensor, dataCopyParams, copyPadParams);
        CreateVecIndex(topkIdxTensor, 0, sendTotalNum_);
        GatherMask(validTopkIndexTensor_, topkIdxTensor, gatherMask_[dstRankIdx * maskAlignSize_], true, sendTotalNum_, {1, 1, 0, 0}, sendToCurExpTokenCnt);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        GatherMask(validTopkIdsTensor_, topkTensor_, gatherMask_[dstRankIdx * maskAlignSize_], true, sendTotalNum_, {1, 1, 0, 0}, sendToCurExpTokenCnt);
        uint32_t aivIdxInGroup = idx % blockNumPerRank_;
        uint32_t rowStartInDst = (dstRankIdx == 0 ? 0 : cumsumInfoTensor_.GetValue(dstRankIdx));  // 填到本端时的目的地址起始值
        if (rowStartInDst < maxOutputSize_) {
            int32_t rowsCopyCnt = sendToCurExpTokenCnt;
            if (rowStartInDst + rowsCopyCnt > maxOutputSize_) {
                rowsCopyCnt = maxOutputSize_ - rowStartInDst;
            }
            int32_t rowSrcOffset = prevSum_;
            prevSum_ += rowsCopyCnt;
            int32_t rowsPerCore = Ops::Base::CeilDiv(rowsCopyCnt, static_cast<int32_t>(blockNumPerRank_));
            int32_t rowDstOffsetPerCore = rowStartInDst + aivIdxInGroup * rowsPerCore;
            int32_t rowSrcOffsetPerCore = rowSrcOffset + aivIdxInGroup * rowsPerCore;
            if (rowDstOffsetPerCore + rowsPerCore > rowStartInDst + rowsCopyCnt) {
                rowsPerCore = rowStartInDst + rowsCopyCnt - rowDstOffsetPerCore;
            }
            if (rowsPerCore > 0) {
                CopyGMToGMPerToken(rowDstOffsetPerCore, dstRankIdx, rowSrcOffsetPerCore, rowSrcOffsetPerCore + rowsPerCore, pingpongId);
            }
        }
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        AtomicAdd((__gm__ int32_t*)(groupFlagListGm2_.GetPhyAddr()), int32_t(1));
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline bool MegaMoe<TemplateMegaMoeTypeFunc>::UpdateGroupParams(uint32_t expertIdx)
{
    // baseOffset_ is 0 when groupIdx = 0
    if (expertIdx != 0) {
        uint64_t m = Get<M_VALUE>(problemShape_);
        uint64_t n = Get<N_VALUE>(problemShape_);
        uint64_t k = Get<K_VALUE>(problemShape_);
        Get<IDX_A_OFFSET>(baseOffset_) += m * k;
        Get<IDX_B_OFFSET>(baseOffset_) += n * k;
        // only splitM
        auto scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_A_SCALE_OFFSET>(baseOffset_) += m * scaleK;
        Get<IDX_B_SCALE_OFFSET>(baseOffset_) += n * scaleK;
        Get<IDX_C_OFFSET>(baseOffset_) += m * n / SWIGLU_N_HALF;
        Get<IDX_C_SCALE_OFFSET>(baseOffset_) +=
            m * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_FLAG_OFFSET>(baseOffset_) += 1;
        Get<IDX_B2_OFFSET>(baseOffset_) += k * n / SWIGLU_N_HALF;
        Get<IDX_B2_SCALE_OFFSET>(baseOffset_) +=
            k * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
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

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo, uint32_t gmmIndex)
{
    if (gmmIndex == 0) {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.ptrA + Get<IDX_A_OFFSET>(baseOffset_) * sizeof(QuantOutType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.ptrAScale +
        Get<IDX_A_SCALE_OFFSET>(baseOffset_) * sizeof(QuantScaleOutType);
        if constexpr(g_coreType == AIC) {
            gmmAddrInfo.bGlobal = params_.bGmAddr + Get<IDX_B_OFFSET>(baseOffset_) * sizeof(QuantOutType);
            gmmAddrInfo.bScaleGlobal = params_.bScaleGmAddr + Get<IDX_B_SCALE_OFFSET>(baseOffset_) *
                                        sizeof(QuantScaleOutType);
        } else {
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                Get<IDX_C_OFFSET>(baseOffset_), Get<IDX_C_SCALE_OFFSET>(baseOffset_),
                Get<IDX_FLAG_OFFSET>(baseOffset_), 0L, 0L, 0L};
            epilogueOp_.UpdateGlobalAddr(vecBaseOffset);
        }
    } else if constexpr(g_coreType == AIC) {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.ptrA2 + Get<IDX_C_OFFSET>(baseOffset_) * sizeof(QuantOutType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.ptrA2Scale + Get<IDX_C_SCALE_OFFSET>(baseOffset_) *
                                    sizeof(QuantScaleOutType);
        gmmAddrInfo.bGlobal = params_.b2GmAddr + Get<IDX_B2_OFFSET>(baseOffset_) * sizeof(QuantOutType);
        gmmAddrInfo.bScaleGlobal = params_.b2ScaleGmAddr + Get<IDX_B2_SCALE_OFFSET>(baseOffset_) *
                                    sizeof(QuantScaleOutType);
    }
    gmmAddrInfo.groupFlagList = (__gm__ int32_t*)params_.workspaceInfo.ptrFlagSwiGluToGmm2 +
                                Get<IDX_FLAG_OFFSET>(baseOffset_) * INT_CACHELINE;
    gmmAddrInfo.groupFlagList2 = (__gm__ int32_t*)params_.workspaceInfo.ptrFlagDispatchToGmm1 +
                                Get<IDX_FLAG_OFFSET>(baseOffset_) * INT_CACHELINE;
    groupFlagListGm_.SetGlobalBuffer(gmmAddrInfo.groupFlagList);
    groupFlagListGm2_.SetGlobalBuffer(gmmAddrInfo.groupFlagList2);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ArgSortExpandedRowIdx()
{
    int32_t coreLen, coreOffset;
    TilingByCore(m_ * topK_, coreLen, coreOffset, 1);
    __gm__ int32_t *src = (__gm__ int32_t *)params_.workspaceInfo.expandedRowIdx;
    __gm__ int32_t *dst = (__gm__ int32_t *)params_.workspaceInfo.expandedRowIdxGather;
    Simt::VF_CALL<ArgsortSimt>(Simt::Dim3{MoeInitRoutingV3::SIMT_THREAD_NUM, 1, 1}, coreLen, coreOffset, src, dst);
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::DDR>();
    GlobalTensor<int32_t> rowIndexGatherGm;
    rowIndexGatherGm.SetGlobalBuffer((__gm__ int32_t *)params_.workspaceInfo.expandedRowIdxGather);
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(rowIndexGatherGm);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ResetTokenPerExpert()
{
    if (aivCoreIdx_ != blockAivNum_ - 1) {
        return;
    }
    int32_t num = worldSize_ * Ops::Base::CeilAlign(
            static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128));
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    LocalTensor<int32_t> tmp = ResetTokenBuf_.Get<int32_t>();
    Duplicate(tmp, 0, num);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(tokenPerExpertWin_, tmp, num);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Unpermute()
{
    int32_t coreLen, coreOffset;
    TilingByCore(m_, coreLen, coreOffset, 1);
    GlobalTensor<int32_t> expandedRowIdx;
    expandedRowIdx.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.expandedRowIdxGather);
    GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer((__gm__ bfloat16_t*)params_.peermemInfo.combineSendPtr);
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer((__gm__ bfloat16_t*)params_.y2GmAddr);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    for (int32_t tokenIdx = coreOffset; tokenIdx < coreLen + coreOffset; tokenIdx++) {
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        LocalTensor<bfloat16_t> dataResBf16 = DataResBuf_.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> dataIn0Bf16 = dataResBf16[k_];
        LocalTensor<bfloat16_t> dataIn1Bf16 = dataResBf16[k_ * 2];
        LocalTensor<float> dataResFp32 = DataResFp32Buf_.Get<float>();
        LocalTensor<float> dataIn0Fp32 = dataResFp32[k_];
        LocalTensor<float> dataIn1Fp32 = dataResFp32[k_ * 2];
        for (int32_t expId = 0; expId < topK_; ++expId) {
            int32_t rowIdx = expandedRowIdx(tokenIdx * topK_ + expId);
            float expScale = topKWeightsTensor_.GetValue(tokenIdx * topK_ + expId);
            auto event = (expId % 2 == 0) ? EVENT_ID0 : EVENT_ID1;
            auto dataInBf16 = (expId % 2 == 0) ? dataIn0Bf16 : dataIn1Bf16;
            auto dataInFp32 = (expId % 2 == 0) ? dataIn0Fp32 : dataIn1Fp32;
            WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            DataCopy(dataInBf16, expandedX[rowIdx * k_], k_);
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            SetFlag<AscendC::HardEvent::S_V>(event);
            WaitFlag<AscendC::HardEvent::S_V>(event);
            // bf16 -> fp32
            Cast(dataInFp32, dataInBf16, AscendC::RoundMode::CAST_NONE, k_);
            PipeBarrier<PIPE_V>();
            if (expId == 0) {
                Muls(dataResFp32, dataInFp32, expScale, k_);
            } else {
                Muls(dataInFp32, dataInFp32, expScale, k_);
                PipeBarrier<PIPE_V>();
                Add(dataResFp32, dataResFp32, dataInFp32, k_);
                PipeBarrier<PIPE_V>();
            }
            SetFlag<AscendC::HardEvent::V_MTE2>(event);
        }
        // fp32 -> bf16
        Cast(dataResBf16, dataResFp32, AscendC::RoundMode::CAST_RINT, k_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(output[tokenIdx * k_], dataResBf16, k_);
    }
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::CrossRankSyncInWorldSize()
{
    __gm__ int32_t* syncRank = (__gm__ int32_t*)(params_.peermemInfo.rankSyncInWorldPtr);
    __gm__ int32_t* syncCount = (__gm__ int32_t*)(params_.peermemInfo.rankSyncInWorldPtr + 48 * 1024);
    int count = *(syncCount + aivCoreIdx_ * 16) + 1;
    for (int i = aivCoreIdx_; i < worldSize_; i += blockAivNum_) {
        __gm__ int32_t* syncRemoteAddr = (__gm__ int32_t*)(winRankAddr_[i]) + rankId_ * 16;
        WriteGmByPassDCache(syncRemoteAddr, count);
        auto syncCheck = syncRank + i * 16;
        GmSignalWaitBarrier(syncCheck, count);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
    *(syncCount + aivCoreIdx_ * 16) = count;
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Process()
{
    if constexpr(g_coreType == AIV) {
        // 1. buffer申请 & workspace Init
        TPipe firstPipe;
        FirstBuffInit(&firstPipe);

        // 2. topKIds全ep域内的发送 & 本卡数据量化
        TopKIdsAllGather();
        QuantProcessInRank();

        // 3. 全卡同步
        CrossRankSyncInWorldSize();
        if (subBlockIdx_ == 0) {
            PipeBarrier<PIPE_ALL>();
            firstPipe.Destroy();
        }
    }

    // 4. dispatch & GroupMatmulSwigluQuant
    for (int localExpertId = 0; localExpertId < expertPerRank_; localExpertId++) {
        if (!UpdateGroupParams(localExpertId)) { // 待确认地址更新怎么做
            continue;
        }
        UpdateGlobalBuffer(gmmAddrInfo, 0);
        if constexpr(g_coreType == AIV) {
            if (subBlockIdx_ == 1) {
                SendInfoCal(localExpertId);
                TripleInfoCalAndDispatch(gmmAddrInfo);
            }
        }
        if (subBlockIdx_ == 0) {
            MegaMoeImpl::GroupMatmulSwigluQuant<
                QuantOutType, QuantOutType, float, QuantScaleOutType, QuantScaleOutType>(
                epilogueOp_, params_, problemShape_, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_);
        }
    }
    EndSync();
    if constexpr(g_coreType == AscendC::AIV) {
        if (aivCoreIdx_ == 1) {
            ExpertTokenNumCopyOut();
        }
        if (subBlockIdx_ == 1) {
            PipeBarrier<PIPE_ALL>();
            firstPipe.Destroy();
        }
    }

    // 5. for循环按照本卡专家id进行gmm2 & combine，内部gmm2与combine之间按照基本块进行流水
    preOffset_ = 0;
    vecSetSyncCom_ = 0;
    baseOffset_ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
        if (!UpdateGroupParams(expertIdx)) {
            continue;
        }
        UpdateGlobalBuffer(gmmAddrInfo, 1);
        if (subBlockIdx_ == 0) {
            MegaMoeImpl::GroupMatmul2<QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
                params_, problemShape_, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_, expertIdx, gmm2PingPongIdx_,
                rankId_);
        }
    }
    EndGMM2Sync();
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();

    // 6. unpermute & 清理win区上放置token的位置
    if constexpr(g_coreType == AIV) {
        ArgSortExpandedRowIdx();
        TPipe secondPipe;
        SecondBuffInit(&secondPipe);
        ResetTokenPerExpert();
        CrossRankSyncInWorldSize(); // 全卡软同步，确认combine send完成
        Unpermute();
        PipeBarrier<PIPE_ALL>();
        secondPipe.Destroy();
    }

}

}   // namespace MegaMoeImpl
#endif