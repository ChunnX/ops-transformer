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
 * \file mega_moe_wave.h
 * \brief
 */

#ifndef MEGA_MOE_WAVE_H
#define MEGA_MOE_WAVE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#include "kernel_operator_list_tensor_intf.h"
#include "mega_moe_base.h"
#include "mega_moe_workspace_info.h"
#include "block_epilogue_swiglu_mx_quant.h"
#include "mega_moe_impl.h"
#include "../../moe_distribute_dispatch_v2/op_kernel/quantize_functions.h"

using namespace AscendC;

namespace MegaMoeImpl {
using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
using BlockOffset = Shape<int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t>;

#define TemplateMegaMoeTypeClass typename XType, typename OutputType, typename TopkWeightsType, int32_t QuantMode
#define TemplateMegaMoeTypeFunc XType, OutputType, TopkWeightsType, QuantMode

template <TemplateMegaMoeTypeClass>
class MegaMoeWave {
public:
    using QuantOutType = typename std::conditional<((QuantMode >= E5M2_QUANT) && (QuantMode != E4M3_QUANT)),
        fp8_e5m2_t, fp8_e4m3fn_t>::type;
    using QuantScaleOutType = typename std::conditional<(QuantMode >= E5M2_QUANT), fp8_e8m0_t, float>::type;
    __aicore__ inline MegaMoeWave() {};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights,
        GM_ADDR weight1, GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
        GM_ADDR scales, GM_ADDR yOut, GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM,
        MegaMoeTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void FirstBuffInit();
    __aicore__ inline void BuffInit();
    __aicore__ inline void SecondBuffInit();
    __aicore__ inline void ResetFlagList();
    __aicore__ inline void TopKIdsAllGather();
    __aicore__ inline void SendInfoCal(int32_t localExpertId);
    __aicore__ inline void TripleInfoCalAndDispatch(GMMAddrInfo &gmmAddrInfo, int32_t localExpertId);
    __aicore__ inline bool UpdateGroupParams(uint32_t expertIdx, uint32_t gmmIndex);
    __aicore__ inline void UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo, uint32_t gmmIndex, uint32_t localExpertId);
    __aicore__ inline void Unpermute();
    __aicore__ inline void CrossRankSyncInWorldSize();
    __aicore__ inline void ExpertTokenNumCopyOut();
    __aicore__ inline void CopyGMToGMPerToken(int32_t rowDstOffsetInCore,
        int32_t remoteRankIdx, int32_t copyStartIdx, int32_t copyNum, int32_t& pingpongId);
    __aicore__ inline void QuantProcessInRank();

    __aicore__ inline int64_t TokenPerExpertLayout(int64_t dim0, int64_t dim1, int64_t dim2)
    {
        return dim0 * perRankStridesInTokenPerExpert_ + dim1 * expertPerRank_ + dim2;
    }

    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    ParamsWave params_{};
    TupleShape problemShape_{};
    BlockOffset baseOffset_{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    GM_ADDR tokenPerExpertCurRankAddr_;

    GlobalTensor<int32_t> tokenPerExpertWin_;
    GlobalTensor<int32_t> cumsumInWorkSpace_;
    GlobalTensor<int32_t> groupFlagListGm_;
    GlobalTensor<int32_t> groupFlagListGm2_;
    GlobalTensor<int32_t> groupList_;
    GlobalTensor<int32_t> expertTokenNumsOut_;
    GlobalTensor<int32_t> tripleGloablTensor_;
    GlobalTensor<int32_t> expertRevNumsGloablTensor_;
    

    uint32_t m_ = 0;
    uint32_t k_ = 0;
    uint32_t aicNum_ = 0;
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
    uint32_t mxFp8ScaleNumAlignPerToken_ = 0;
    uint32_t mxFp8TokenAlignBytes_ = 0;
    uint32_t mxFp8ScaleAlignBytes_ = 0;
    uint32_t mxFp8TokenScaleAlignBytes_ = 0;
    uint32_t expertBeforeCnt_ = 0;
    uint32_t dataResAddr_ = 0;
    uint32_t dataResSize_ = 0;
    uint32_t resetTokenAddr_ = 0;
    uint32_t resetTokenSize_ = 0;
    uint32_t dataResFp32Addr_ = 0;
    uint32_t dataResFp32Size_ = 0;
    uint32_t topKWeightsAddr_ = 0;
    uint32_t topKWeightsSize_ = 0;
    uint32_t ubBufferUsedAddr_ = 0;
    uint64_t preOffset_ = 0;
    uint16_t gmm2PingPongIdx_ = 0;
    uint64_t sendTotalNum_ = 0;
    uint32_t maskAlignSize_ = 0;
    uint32_t topkOffsetPerRank_ = 0;
    uint64_t totalSendCntInExp_ = 0;
    uint64_t cumsumRevCntInRank_ = 0;
    int32_t compareCount_ = 0;

    LocalTensor<int32_t> topkTensor1_;
    LocalTensor<int32_t> topkTensor2_;
    LocalTensor<int32_t> topkIndexTensor_;
    LocalTensor<uint8_t> gatherMaskTensor_;
    LocalTensor<uint32_t> gatherMaskInt32Tensor_;
    LocalTensor<int32_t> expertTokenCntTensor_;
    LocalTensor<int32_t> validTopkIndexTensor_;
    LocalTensor<int32_t> cumsumInfoTensor_;
    LocalTensor<QuantOutType> copyTmpTensor1_;
    LocalTensor<QuantOutType> copyTmpTensor2_;
    LocalTensor<int32_t> tripleTensor_;
    LocalTensor<XType> xInTensor1_;
    LocalTensor<XType> xInTensor2_;
    LocalTensor<QuantOutType> xOutTensor1_;
    LocalTensor<QuantOutType> xOutTensor2_;
    LocalTensor<uint16_t> mxTempTensor_;
    LocalTensor<int32_t> resetTensor_;
    LocalTensor<int32_t> topkIdsTensor_;
    LocalTensor<int32_t> expertTokenNumsOutTensor_;
    LocalTensor<bfloat16_t> dataResTensor_;
    LocalTensor<int32_t> resetTokenTensor_;
    LocalTensor<float> dataResFp32Tensor_;
    LocalTensor<float> topKWeightsTensor_;

    using BlockEpilogue = BlockEpilogueSwigluMxQuant<QuantOutType, float, QuantScaleOutType, QuantScaleOutType, true>;
    BlockEpilogue epilogueOp_;
};

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales, GM_ADDR yOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData *tilingData)
{
    m_ = tilingData->bs;
    k_ = tilingData->h;
    aicNum_ = tilingData->aicNum;
    topK_ = tilingData->topK;
    sendTotalNum_ = m_ * topK_;
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
    params_.expertIdxGmAddr = topkIds;
    params_.bGmAddr = GetTensorAddr(0, weight1);
    params_.b2GmAddr = GetTensorAddr(0, weight2);
    params_.bScaleGmAddr = GetTensorAddr(0, weightScales1);
    params_.b2ScaleGmAddr = GetTensorAddr(0, weightScales2);

    params_.y2GmAddr = yOut;
    params_.expertTokenNumsOutGmAddr = expertTokenNumsOut;
    params_.probsGmAddr = topkWeights;
    params_.workspaceInfo = WorkspaceInfoWave(workspaceGM, tilingData);
    params_.peermemInfo = PeermemInfoWave(winRankAddr_[rankId_], tilingData);
    params_.tilingData = tilingData;
    tokenPerExpertWin_.SetGlobalBuffer((__gm__ int32_t*)params_.peermemInfo.topKInfoInWorldSizePtr);
    expertTokenNumsOut_.SetGlobalBuffer((__gm__ int32_t*)params_.expertTokenNumsOutGmAddr);
    expertRevNumsGloablTensor_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.expertRevTokenNumsPtr);
    tripleGloablTensor_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.tripleInfoPtr);
    epilogueOp_.Init({params_.workspaceInfo.swigluQuantDataPtr, params_.workspaceInfo.swigluQuantScalePtr,
        params_.workspaceInfo.flagSwiGluToGmm2Ptr, nullptr, nullptr, nullptr, ALIGN_128, ALIGN_256});
    perRankStridesInTokenPerExpert_ = Ops::Base::CeilAlign(
        static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128));
    tokenPerExpertCurRankAddr_ = params_.peermemInfo.topKInfoInWorldSizePtr +
        TokenPerExpertLayout(rankId_, 0, 0) * sizeof(int32_t);
    topkOffsetPerRank_ = Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_512));
    mxFp8ScaleNumAlignPerToken_ = Ops::Base::CeilDiv(k_, static_cast<uint32_t>(ALIGN_32));
    mxFp8TokenAlignBytes_ = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_256)) * sizeof(QuantOutType);
    mxFp8ScaleAlignBytes_ = mxFp8ScaleNumAlignPerToken_ * sizeof(uint8_t);
    mxFp8TokenScaleAlignBytes_ = Ops::Base::CeilAlign(mxFp8TokenAlignBytes_ + mxFp8ScaleAlignBytes_,
        static_cast<uint32_t>(ALIGN_32));
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeTypeFunc>::FirstBuffInit()
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    compareCount_ = Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_256)) / sizeof(int32_t);
    maskAlignSize_ = Ops::Base::CeilAlign(Ops::Base::CeilAlign(static_cast<int64_t>(compareCount_),
        static_cast<int64_t>(INT32_PER_256B)), static_cast<int64_t>(ALIGN_32));
    uint32_t expertTokenCntTensorAddr = 0;
    uint32_t expertTokenCntTensorSize = ALIGN_32;
    expertTokenCntTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, expertTokenCntTensorAddr,
        expertTokenCntTensorSize / sizeof(int32_t));
    uint32_t cumsumInfoTensorAddr = expertTokenCntTensorAddr + expertTokenCntTensorSize;
    uint32_t cumsumInfoTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(worldSize_ * expertPerRank_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    cumsumInfoTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, cumsumInfoTensorAddr,
        cumsumInfoTensorSize / sizeof(int32_t));
    uint32_t gatherMaskTensorAddr = cumsumInfoTensorAddr + cumsumInfoTensorSize;
    uint32_t gatherMaskTensorSize = maskAlignSize_ * worldSize_;
    gatherMaskTensor_ = LocalTensor<uint8_t>(TPosition::VECCALC, gatherMaskTensorAddr,
        gatherMaskTensorSize / sizeof(uint8_t));
    gatherMaskInt32Tensor_ = LocalTensor<uint32_t>(TPosition::VECCALC, gatherMaskTensorAddr,
        gatherMaskTensorSize / sizeof(uint32_t));
    uint32_t topkTensorAddr1 = gatherMaskTensorAddr + gatherMaskTensorSize;
    uint32_t topkTensorSize1 = Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_256));
    topkTensor1_ = LocalTensor<int32_t>(TPosition::VECCALC, topkTensorAddr1, topkTensorSize1 / sizeof(int32_t));
    uint32_t topkTensorAddr2 = topkTensorAddr1 + topkTensorSize1;
    uint32_t topkTensorSize2 = topkTensorSize1;
    topkTensor2_ = LocalTensor<int32_t>(TPosition::VECCALC, topkTensorAddr2, topkTensorSize2 / sizeof(int32_t));
    uint32_t validTopkIndexTensorAddr = topkTensorAddr2 + topkTensorSize2;
    uint32_t validTopkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    validTopkIndexTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, validTopkIndexTensorAddr,
        validTopkIndexTensorSize / sizeof(int32_t));
    uint32_t topkIndexTensorAddr = validTopkIndexTensorAddr + validTopkIndexTensorSize;
    uint32_t topkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    topkIndexTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, topkIndexTensorAddr,
        topkIndexTensorSize / sizeof(int32_t));
    uint32_t copyTmpTensor1Addr = topkIndexTensorAddr + topkIndexTensorSize;
    uint32_t copyTmpTensor1Size = TOKEN_SCALE_SIZE;
    copyTmpTensor1_ = LocalTensor<QuantOutType>(TPosition::VECCALC, copyTmpTensor1Addr,
        copyTmpTensor1Size / sizeof(QuantOutType));
    uint32_t copyTmpTensor2Addr = copyTmpTensor1Addr + copyTmpTensor1Size;
    uint32_t copyTmpTensor2Size = TOKEN_SCALE_SIZE;
    copyTmpTensor2_ = LocalTensor<QuantOutType>(TPosition::VECCALC, copyTmpTensor2Addr,
        copyTmpTensor2Size / sizeof(QuantOutType));
    uint32_t expertTokenNumsOutTensorAddr = copyTmpTensor2Addr + copyTmpTensor2Size;
    uint32_t expertTokenNumsOutTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(expertPerRank_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    expertTokenNumsOutTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, expertTokenNumsOutTensorAddr,
        expertTokenNumsOutTensorSize / sizeof(int32_t));
    ubBufferUsedAddr_ = expertTokenNumsOutTensorAddr + expertTokenNumsOutTensorSize;
    CreateVecIndex(topkIndexTensor_, 0, (topkIndexTensorSize / sizeof(int32_t)));
    Duplicate<int32_t>(cumsumInfoTensor_, 0, (cumsumInfoTensorSize / sizeof(int32_t)));
    Duplicate<int32_t>(topkTensor1_, 0, (topkTensorSize1 / sizeof(int32_t)));
    Duplicate<int32_t>(topkTensor2_, 0, (topkTensorSize2 / sizeof(int32_t)));
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeTypeFunc>::BuffInit()
{
    if constexpr(g_coreType == AIC) {
        return;
    }

    uint32_t topkIdsTensorAddr = 0;
    uint32_t topkIdsTensorSize = Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_32));
    topkIdsTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, topkIdsTensorAddr, topkIdsTensorSize / sizeof(int32_t));
    uint32_t resetTensorAddr = topkIdsTensorAddr + topkIdsTensorSize;
    uint32_t resetNumPerCore = Ops::Base::CeilDiv(
        static_cast<uint64_t>(expertPerRank_ * INT_CACHELINE * 2), static_cast<uint64_t>(blockAivNum_));
    uint32_t resetTensorSize = Ops::Base::CeilAlign(static_cast<uint64_t>(resetNumPerCore),
        static_cast<uint64_t>(INT32_PER_256B)) * sizeof(int32_t);
    resetTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, resetTensorAddr, resetTensorSize / sizeof(int32_t));
    uint32_t mxTempTensorAddr = resetTensorAddr + resetTensorSize;
    uint32_t mxTempTensorSize = 2 * 1024;
    mxTempTensor_ = LocalTensor<uint16_t>(TPosition::VECCALC, mxTempTensorAddr, mxTempTensorSize / sizeof(uint16_t));
    uint32_t xOutTensorAddr1 = mxTempTensorAddr + mxTempTensorSize;
    uint32_t xOutTensorSize1 = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_256)) +
        Ops::Base::CeilDiv(k_, static_cast<uint32_t>(ALIGN_32));
    xOutTensor1_ = LocalTensor<QuantOutType>(TPosition::VECCALC, xOutTensorAddr1,
        xOutTensorSize1 / sizeof(QuantOutType));
    uint32_t xOutTensorAddr2 = xOutTensorAddr1 + xOutTensorSize1;
    uint32_t xOutTensorSize2 = xOutTensorSize1;
    xOutTensor2_ = LocalTensor<QuantOutType>(TPosition::VECCALC, xOutTensorAddr2,
        xOutTensorSize2 / sizeof(QuantOutType));
    uint32_t xInAlignAddr1 = xOutTensorAddr2 + xOutTensorSize2;
    uint32_t xInAlignSize1 = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_128)) * sizeof(XType);
    xInTensor1_ = LocalTensor<XType>(TPosition::VECCALC, xInAlignAddr1, xInAlignSize1 / sizeof(XType));
    uint32_t xInAlignAddr2 = xInAlignAddr1 + xInAlignSize1;
    uint32_t xInAlignSize2 = xInAlignSize1;
    xInTensor2_ = LocalTensor<XType>(TPosition::VECCALC, xInAlignAddr2, xInAlignSize2 / sizeof(XType));
    Duplicate<int32_t>(resetTensor_, 0, (resetTensorSize / sizeof(int32_t)));
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeTypeFunc>::ResetFlagList()
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    // workSpace groupFlagList清零
    groupFlagListGm_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.flagSwiGluToGmm2Ptr);
    int32_t flagNum = expertPerRank_ * INT_CACHELINE * TWO_FLAG;
    int32_t coreLen, coreOffset;
    TilingByCore(flagNum, coreLen, coreOffset, 1);
    DataCopyExtParams rankSyncCopyParams{1U, static_cast<uint32_t>(coreLen * sizeof(int32_t)), 0U, 0U, 0U};
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID>();
    if (coreLen != 0) {
        DataCopyPad(groupFlagListGm_[coreOffset], resetTensor_, rankSyncCopyParams);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeTypeFunc>::ExpertTokenNumCopyOut()
{
    int32_t lastRankIdx = static_cast<int32_t>(worldSize_ - 1);
    expertTokenNumsOutTensor_.SetValue(0, cumsumInfoTensor_.GetValue(lastRankIdx));
    for (int32_t expertIdx = 1; expertIdx < expertPerRank_; expertIdx++) {
        int32_t cur = cumsumInfoTensor_.GetValue(expertIdx * static_cast<int32_t>(worldSize_) + lastRankIdx);
        int32_t prev = cumsumInfoTensor_.GetValue((expertIdx - 1) * static_cast<int32_t>(worldSize_) + lastRankIdx);
        expertTokenNumsOutTensor_.SetValue(expertIdx, cur - prev);
    }
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID>();
    DataCopyExtParams copyParams{1U, static_cast<uint32_t>(expertPerRank_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(expertTokenNumsOut_, expertTokenNumsOutTensor_, copyParams);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeTypeFunc>::TopKIdsAllGather()
{
    // topKInfo给对端发送，当前分核逻辑为一个核负责一张卡的发送
    if constexpr(g_coreType == AIC) {
        return;
    }

    GlobalTensor<int32_t> srcGlobalTensor;
    GlobalTensor<int32_t> dstGlobalTensor;
    srcGlobalTensor.SetGlobalBuffer((__gm__ int32_t*)params_.expertIdxGmAddr);
    DataCopyExtParams dataCopyParams{1U, static_cast<uint32_t>(sendTotalNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(topkIdsTensor_, srcGlobalTensor, dataCopyParams, copyPadParams);
    SyncFuncStatic<AscendC::HardEvent::MTE2_MTE3, SYNC_EVENT_ID>();
    for (int32_t dstRankIdx = aivCoreIdx_; dstRankIdx < worldSize_; dstRankIdx += blockAivNum_) {
        uint64_t offsetInDstWinAddr = PEERMEM_DATA_OFFSET + rankId_ * Ops::Base::CeilAlign(
        static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_512));
        dstGlobalTensor.SetGlobalBuffer((__gm__ int32_t*)GetRankWinAddrWithOffset(dstRankIdx, offsetInDstWinAddr));
        DataCopyPad(dstGlobalTensor, topkIdsTensor_, dataCopyParams);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeTypeFunc>::QuantProcessInRank()
{
    if constexpr(g_coreType == AIC) {
        return;
    }

    // 分核，按照BS与aivCoreNum均分
    int32_t curentNum;
    int32_t curentOffset;
    TilingByCore(m_, curentNum, curentOffset, 1);
    uint32_t H = k_;
    GlobalTensor<XType> srcGlobalTensor;
    GlobalTensor<uint8_t> dstGlobalTensor;
    DataCopyParams xCopyInParams = {1U, static_cast<uint16_t>(H * sizeof(XType)), 0U, 0U};
    DataCopyPadParams xCopyInPadParams{true, 0, 0, 0};
    DataCopyExtParams xCopyOutParams = {1U,
        static_cast<uint32_t>(mxFp8TokenAlignBytes_ + mxFp8ScaleAlignBytes_), 0U, 0U, 0U};
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int32_t index = 0; index < curentNum; index++) {
        srcGlobalTensor.SetGlobalBuffer((__gm__ XType*)(params_.aGmAddr + (curentOffset + index) * H * sizeof(XType)));
        dstGlobalTensor.SetGlobalBuffer((__gm__ uint8_t*)(params_.peermemInfo.quantTokenScalePtr +
        (curentOffset + index) * mxFp8TokenScaleAlignBytes_));
        auto event = (index % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
        auto xInTensor = (index % DOUBLE_BUFFER == 0) ? xInTensor1_ : xInTensor2_;
        auto xOutTensor = (index % DOUBLE_BUFFER == 0) ? xOutTensor1_ : xOutTensor2_;
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        DataCopyPad(xInTensor, srcGlobalTensor, xCopyInParams, xCopyInPadParams);
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        __ubuf__ XType* srcAddr = (__ubuf__ XType*)xInTensor.GetPhyAddr();
        __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)mxTempTensor_.GetPhyAddr();
        __ubuf__ uint16_t* halfScaleAddr = (__ubuf__ uint16_t*)mxTempTensor_[Ops::Base::CeilAlign(
        mxFp8ScaleNumAlignPerToken_, static_cast<uint32_t>(ALIGN_32))].GetPhyAddr();
        __ubuf__ int8_t* outDataAddr = (__ubuf__ int8_t*)xOutTensor.GetPhyAddr();
        __ubuf__ uint16_t* mxScaleAddr = (__ubuf__ uint16_t*)xOutTensor[Ops::Base::CeilAlign(
        H, static_cast<uint32_t>(ALIGN_256))].GetPhyAddr();

        quant::ComputeMaxExp(srcAddr, maxExpAddr, H); // 计算最大Exp
        quant::ComputeScale<QuantOutType>(maxExpAddr, mxScaleAddr, halfScaleAddr,
        mxFp8ScaleNumAlignPerToken_); // 计算scales并填充f
        quant::ComputeFp8Data<XType, QuantOutType, AscendC::RoundMode::CAST_TRUNC,
            AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleAddr, outDataAddr, H);
        SetFlag<AscendC::HardEvent::V_MTE3>(event);
        WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        auto xOutBytesTensor = xOutTensor.template ReinterpretCast<uint8_t>();
        DataCopyPad(dstGlobalTensor, xOutBytesTensor, xCopyOutParams);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeTypeFunc>::SendInfoCal(int32_t localExpertId)
{
    totalSendCntInExp_ = 0;
    uint32_t mask = sendTotalNum_;
    int32_t compareExpertId = localExpertId + expertPerRank_ * rankId_;
    GlobalTensor<int32_t> srcGlobalTensor;
    DataCopyExtParams dataCopyParams{1U, static_cast<uint32_t>(sendTotalNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadParams{true, 0U,
        static_cast<uint8_t>(Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_), static_cast<int64_t>(INT32_PER_256B)) - sendTotalNum_), 0U}; // 填充数据为0
    Duplicate<uint32_t>(gatherMaskInt32Tensor_, 0, (worldSize_ * maskAlignSize_) / sizeof(uint32_t)); // mask位置清理
    SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID1);
    for (int32_t calRankId = 0; calRankId < worldSize_; calRankId++) {
        uint64_t sendCnt = 0;
        uint64_t offsetInDstWinAddr = PEERMEM_DATA_OFFSET + calRankId * Ops::Base::CeilAlign(
        static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_512));
        srcGlobalTensor.SetGlobalBuffer((__gm__ int32_t*)GetRankWinAddrWithOffset(calRankId, offsetInDstWinAddr));
        auto event = (calRankId % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
        auto topkTensor = (calRankId % DOUBLE_BUFFER == 0) ? topkTensor1_ : topkTensor2_;
        WaitFlag<AscendC::HardEvent::S_MTE2>(event);
        DataCopyPad(topkTensor, srcGlobalTensor, dataCopyParams, copyPadParams);
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        CompareScalar(gatherMaskTensor_[calRankId * maskAlignSize_], topkTensor, compareExpertId,
        AscendC::CMPMODE::EQ, compareCount_); // mask位置直接存下来
        GatherMask(validTopkIndexTensor_, topkIndexTensor_,
        gatherMaskInt32Tensor_[calRankId * (maskAlignSize_ / sizeof(uint32_t))], true, mask, {1, 1, 0, 0}, sendCnt);
        SetFlag<AscendC::HardEvent::V_S>(event);
        WaitFlag<AscendC::HardEvent::V_S>(event);
        totalSendCntInExp_ += sendCnt;
        cumsumRevCntInRank_ += sendCnt;
        cumsumInfoTensor_.SetValue(localExpertId * worldSize_ + calRankId, static_cast<int32_t>(cumsumRevCntInRank_)); // 先专家 - 再rank
        SetFlag<AscendC::HardEvent::S_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID1);
    expertTokenCntTensor_.SetValue(0, totalSendCntInExp_);
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID>();
    DataCopy<int32_t>(expertRevNumsGloablTensor_[localExpertId * INT32_PER_256B * aicNum_ + INT32_PER_256B * blockIdx_],
        expertTokenCntTensor_, INT32_PER_256B);
    PipeBarrier<PIPE_ALL>();
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_MTE3>(2);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeTypeFunc>::CopyGMToGMPerToken(
    int32_t rowDstOffsetInCore, int32_t remoteRankIdx, int32_t copyStartIdx, int32_t copyNum, int32_t& pingpongId)
{
    constexpr int32_t BufferNum = 2;
    int64_t widthA = Get<K_VALUE>(problemShape_); // 输出单token长度，紧密排列
    int64_t widthAScale = Ops::Base::CeilDiv(static_cast<int64_t>(widthA), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE;  // 输出token-scale长度，紧密排列
    uint32_t copyInNum = Ops::Base::CeilAlign(static_cast<int64_t>(mxFp8TokenAlignBytes_ + mxFp8ScaleAlignBytes_),
        static_cast<int64_t>(ALIGN_32)); // 输入token-scale拼接，非紧密排列
    GlobalTensor<QuantOutType> remoteRankGlobalTensor;
    GlobalTensor<QuantOutType> tokenRevGlobalTensor;
    GlobalTensor<QuantScaleOutType> scaleRevGlobalTensor;
    tokenRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(
        params_.workspaceInfo.dispatchRevDataPtr + rowDstOffsetInCore * widthA)); // rowDstOffsetInCore 是当前专家全部的累加偏移
    scaleRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleOutType *>(
        params_.workspaceInfo.dispatchRevScalePtr + rowDstOffsetInCore * widthAScale));
    remoteRankGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(
        GetRankWinAddrWithOffset(remoteRankIdx,
        PEERMEM_DATA_OFFSET + worldSize_ * topkOffsetPerRank_)));
    for (uint32_t copyIndex = 0; copyIndex < copyNum; ++copyIndex) {
        pingpongId = (pingpongId + 1) % BufferNum;
        TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<QuantOutType> tokenScalebuf = pingpongId == 0 ? copyTmpTensor1_ : copyTmpTensor2_;
        LocalTensor<QuantScaleOutType> bufScale = tokenScalebuf[mxFp8TokenAlignBytes_].template
        ReinterpretCast<QuantScaleOutType>();
        int32_t topkIndex = validTopkIndexTensor_.GetValue(copyStartIdx + copyIndex) ;
        int32_t tokenIndex = topkIndex / topK_;
        tripleTensor_[copyIndex * INT32_PER_256B].SetValue(RANK_ID, remoteRankIdx);
        tripleTensor_[copyIndex * INT32_PER_256B].SetValue(TOKEN_ID, tokenIndex);
        tripleTensor_[copyIndex * INT32_PER_256B].SetValue(TOPK_INDEX, topkIndex % topK_);
        auto remoteCopyOffset = tokenIndex * copyInNum;

        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        DataCopy(tokenScalebuf, remoteRankGlobalTensor[remoteCopyOffset], copyInNum);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        DataCopyPad(tokenRevGlobalTensor[copyIndex * widthA], tokenScalebuf,
        {1, static_cast<uint16_t>(widthA * sizeof(QuantOutType)), 0U, 0U, 0U});
        DataCopyPad(scaleRevGlobalTensor[copyIndex * widthAScale], bufScale,
        {1, static_cast<uint16_t>(widthAScale * sizeof(QuantScaleOutType)), 0U, 0U, 0U});
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
    }
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID>();
    DataCopy(tripleGloablTensor_[rowDstOffsetInCore * INT32_PER_256B], tripleTensor_, copyNum * INT32_PER_256B); // 三元组搬出
}

// template <TemplateMegaMoeTypeClass>
// __aicore__ inline void MegaMoeWave<TemplateMegaMoeTypeFunc>::TripleInfoCalAndDispatch(
//     GMMAddrInfo &gmmAddrInfo, int32_t localExpertId)
// {
//     int32_t pingpongId = 0;
//     SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
//     SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
//     PipeBarrier<PIPE_ALL>();
//     int32_t priorExpertCumsum = (localExpertId == 0) ? 0 :
//         cumsumInfoTensor_.GetValue(localExpertId * worldSize_ - 1);
//     int32_t totalTokensInExpert = static_cast<int32_t>(Get<M_VALUE>(problemShape_));
//     if (priorExpertCumsum >= static_cast<int32_t>(maxOutputSize_)) {
//         totalTokensInExpert = 0;
//     } else if (priorExpertCumsum + totalTokensInExpert > static_cast<int32_t>(maxOutputSize_)) {
//         totalTokensInExpert = static_cast<int32_t>(maxOutputSize_) - priorExpertCumsum;
//     }

//     int32_t totalCores = static_cast<int32_t>(blockNum_);
//     int32_t coreIdx = static_cast<int32_t>(blockIdx_);
//     int32_t tokensThisCore = (totalTokensInExpert > coreIdx) ?
//         ((totalTokensInExpert - 1 - coreIdx) / totalCores + 1) : 0;

//     if (tokensThisCore > 0) {
//         // 二分查找：在当前专家内，找到首个token (expert内全局idx = coreIdx) 所在的卡号
//         int32_t firstGlobalIdxInExpert = coreIdx;
//         int32_t lo = 0;
//         int32_t hi = static_cast<int32_t>(worldSize_) - 1;
//         int32_t startRank = hi;
//         while (lo <= hi) {
//             int32_t mid = (lo + hi) >> 1;
//             int32_t cumsumAtMid = cumsumInfoTensor_.GetValue(localExpertId * worldSize_ + mid) - priorExpertCumsum;
//             if (cumsumAtMid > firstGlobalIdxInExpert) {
//                 startRank = mid;
//                 hi = mid - 1;
//             } else {
//                 lo = mid + 1;
//             }
//         }

//         uint32_t tripleTensorAddr = ubBufferUsedAddr_;
//         uint32_t tripleTensorSize = tokensThisCore * ALIGN_32;
//         tripleTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, tripleTensorAddr,
//             tripleTensorSize / sizeof(int32_t));

//         SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID>();

//         constexpr int32_t BufferNum = 2;
//         int64_t widthA = Get<K_VALUE>(problemShape_);
//         int64_t widthAScale = Ops::Base::CeilDiv(
//             static_cast<int64_t>(widthA), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
//         uint32_t copyInNum = Ops::Base::CeilAlign(
//             static_cast<int64_t>(mxFp8TokenAlignBytes_ + mxFp8ScaleAlignBytes_), static_cast<int64_t>(ALIGN_32));

//         // 目的地址基址按照当前专家在dispatchRevDataPtr中的起始位置
//         GlobalTensor<QuantOutType> tokenRevGlobalTensor;
//         GlobalTensor<QuantScaleOutType> scaleRevGlobalTensor;
//         tokenRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(
//             params_.workspaceInfo.dispatchRevDataPtr + priorExpertCumsum * widthA));
//         scaleRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleOutType*>(
//             params_.workspaceInfo.dispatchRevScalePtr + priorExpertCumsum * widthAScale));

//         int32_t curRank = startRank;
//         int32_t curRankExprStart = (startRank == 0) ? 0 :
//             (cumsumInfoTensor_.GetValue(localExpertId * worldSize_ + startRank - 1) - priorExpertCumsum);
//         int32_t curRankExprEnd = cumsumInfoTensor_.GetValue(localExpertId * worldSize_ + startRank) - priorExpertCumsum;

//         uint64_t sendCntCurRank = 0;
//         GatherMask(validTopkIndexTensor_, topkIndexTensor_,
//             gatherMaskInt32Tensor_[curRank * (maskAlignSize_ / sizeof(uint32_t))],
//             true, sendTotalNum_, {1, 1, 0, 0}, sendCntCurRank);
//         SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID>();

//         GlobalTensor<QuantOutType> remoteRankGlobalTensor;
//         remoteRankGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(
//             GetRankWinAddrWithOffset(curRank, PEERMEM_DATA_OFFSET + worldSize_ * topkOffsetPerRank_)));

//         for (int32_t i_in_core = 0; i_in_core < tokensThisCore; ++i_in_core) {
//             int32_t globalIdxInExpert = coreIdx + i_in_core * totalCores;

//             bool rankChanged = false;
//             while (globalIdxInExpert >= curRankExprEnd) {
//                 ++curRank;
//                 curRankExprStart = curRankExprEnd;
//                 curRankExprEnd = cumsumInfoTensor_.GetValue(localExpertId * worldSize_ + curRank) - priorExpertCumsum;
//                 rankChanged = true;
//             }
//             if (rankChanged) {
//                 GatherMask(validTopkIndexTensor_, topkIndexTensor_,
//                     gatherMaskInt32Tensor_[curRank * (maskAlignSize_ / sizeof(uint32_t))],
//                     true, sendTotalNum_, {1, 1, 0, 0}, sendCntCurRank);
//                 SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID>();
//                 remoteRankGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(
//                     GetRankWinAddrWithOffset(curRank, PEERMEM_DATA_OFFSET + worldSize_ * topkOffsetPerRank_)));
//             }

//             int32_t withinRankIdx = globalIdxInExpert - curRankExprStart;
//             int32_t topkIndex = validTopkIndexTensor_.GetValue(withinRankIdx);
//             int32_t tokenIndex = topkIndex / topK_;

//             tripleTensor_[i_in_core * 8].SetValue(0, curRank);
//             tripleTensor_[i_in_core * 8].SetValue(1, tokenIndex);
//             tripleTensor_[i_in_core * 8].SetValue(2, topkIndex % topK_);

//             pingpongId = (pingpongId + 1) % BufferNum;
//             TEventID EVENT_ID_CUR = (pingpongId == 0) ? EVENT_ID0 : EVENT_ID1;
//             LocalTensor<QuantOutType> tokenScalebuf = (pingpongId == 0) ? copyTmpTensor1_ : copyTmpTensor2_;
//             LocalTensor<QuantScaleOutType> bufScale =
//                 tokenScalebuf[mxFp8TokenAlignBytes_].template ReinterpretCast<QuantScaleOutType>();

//             auto remoteCopyOffset = tokenIndex * copyInNum;

//             WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID_CUR);
//             DataCopy(tokenScalebuf, remoteRankGlobalTensor[remoteCopyOffset], copyInNum);
//             SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID_CUR);
//             WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID_CUR);

//             // 按核错位排布：核c的第i_in_core个token写到专家内偏移 coreIdx + i_in_core * totalCores
//             int32_t dstOffsetInExpert = coreIdx + i_in_core * totalCores;
//             DataCopyPad(tokenRevGlobalTensor[dstOffsetInExpert * widthA], tokenScalebuf,
//                 {1, static_cast<uint16_t>(widthA * sizeof(QuantOutType)), 0U, 0U, 0U});
//             DataCopyPad(scaleRevGlobalTensor[dstOffsetInExpert * widthAScale], bufScale,
//                 {1, static_cast<uint16_t>(widthAScale * sizeof(QuantScaleOutType)), 0U, 0U, 0U});
//             SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID_CUR);
//         }

//         // 三元组按错位排布搬出（与数据同顺序）
//         SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID>();
//         for (int32_t i = 0; i < tokensThisCore; ++i) {
//             int32_t dstAbsIdx = priorExpertCumsum + coreIdx + i * totalCores;
//             DataCopy(tripleGloablTensor_[dstAbsIdx * 8], tripleTensor_[i * 8], 8);
//         }
//     }

//     // 保持原有计数：总AtomicAdd次数 = worldSize_ * blockNumPerRank_，与GMM等待目标一致
//     SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID>();
//     for (uint32_t idx = blockIdx_; idx < worldSize_ * blockNumPerRank_; idx += blockNum_) {
//         AtomicAdd((__gm__ int32_t*)(groupFlagListGm2_.GetPhyAddr()), int32_t(1));
//     }

//     WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
//     WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
// }

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeTypeFunc>::TripleInfoCalAndDispatch(
    GMMAddrInfo &gmmAddrInfo, int32_t localExpertId)
{
    int32_t pingpongId = 0;
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    PipeBarrier<PIPE_ALL>();
    for (uint32_t idx = blockIdx_; idx < worldSize_ * blockNumPerRank_; idx += blockNum_) {
        // 相比对原来流程需要添加本卡的cnt对应到原始卡上的哪个bs
        uint32_t dstRankIdx = idx / blockNumPerRank_;
        uint64_t sendToCurExpTokenCnt = 0;
        GatherMask(validTopkIndexTensor_, topkIndexTensor_,
            gatherMaskInt32Tensor_[dstRankIdx * (maskAlignSize_ / sizeof(uint32_t))],
            true, sendTotalNum_, {1, 1, 0, 0}, sendToCurExpTokenCnt);
        uint32_t aivIdxInGroup = idx % blockNumPerRank_;
        uint32_t rowStartIdxInDst = ((dstRankIdx == 0 && localExpertId == 0) ? 0 :
            cumsumInfoTensor_.GetValue(localExpertId * worldSize_ + dstRankIdx - 1));  // 填到本端时的目的地址起始值
        if (rowStartIdxInDst < maxOutputSize_) {
            int32_t rowsCopyCnt = sendToCurExpTokenCnt;
            if (rowStartIdxInDst + rowsCopyCnt > maxOutputSize_) {
                rowsCopyCnt = maxOutputSize_ - rowStartIdxInDst;
            }
            int32_t rowsNumInCore = Ops::Base::CeilDiv(rowsCopyCnt, static_cast<int32_t>(blockNumPerRank_));
            int32_t rowSrcIdxInCore = aivIdxInGroup * rowsNumInCore;
            int32_t rowDstOffsetInCore = rowStartIdxInDst + rowSrcIdxInCore;
            if (rowDstOffsetInCore + rowsNumInCore > rowStartIdxInDst + rowsCopyCnt) {
                rowsNumInCore = rowStartIdxInDst + rowsCopyCnt - rowDstOffsetInCore;
            }
            if (rowsNumInCore > 0) {
                uint32_t tripleTensorAddr = ubBufferUsedAddr_;
                uint32_t tripleTensorSize = rowsNumInCore * ALIGN_32;
                tripleTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, tripleTensorAddr,
                    tripleTensorSize / sizeof(int32_t));
                SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID>();
                CopyGMToGMPerToken(rowDstOffsetInCore, dstRankIdx, rowSrcIdxInCore, rowsNumInCore, pingpongId);
            }
        }
        SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID>();
        AtomicAdd((__gm__ int32_t*)(groupFlagListGm2_.GetPhyAddr()), int32_t(1));
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline bool MegaMoeWave<TemplateMegaMoeTypeFunc>::UpdateGroupParams(uint32_t expertIdx, uint32_t gmmIndex)
{
    if (expertIdx != 0) {
        uint64_t m = Get<M_VALUE>(problemShape_);
        uint64_t n = Get<N_VALUE>(problemShape_);
        uint64_t k = Get<K_VALUE>(problemShape_);
        expertBeforeCnt_ += m;
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

    if (gmmIndex == 0) {
        if constexpr (g_coreType == AscendC::AIC) {
            CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_S>(2 + 16);
            uint64_t offsetInCnt = expertIdx * 8 * aicNum_ + 8 * blockIdx_;
            DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
                expertRevNumsGloablTensor_[offsetInCnt]);
            Get<M_VALUE>(problemShape_) = expertRevNumsGloablTensor_.GetValue(offsetInCnt);
            CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_S>(3);
        }
        if constexpr (g_coreType == AscendC::AIV) {
            if (subBlockIdx_ == 0) {
                CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_S>(3);
                uint64_t offsetInCnt = expertIdx * 8 * aicNum_ + 8 * blockIdx_;
                DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
                expertRevNumsGloablTensor_[offsetInCnt]);
                Get<M_VALUE>(problemShape_) = expertRevNumsGloablTensor_.GetValue(offsetInCnt);
            } else {
                Get<M_VALUE>(problemShape_) = totalSendCntInExp_;
            }
        }
    } else {
        uint64_t offsetInCnt = expertIdx * 8 * aicNum_ + 8 * blockIdx_;
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
                expertRevNumsGloablTensor_[offsetInCnt]);
        Get<M_VALUE>(problemShape_) = expertRevNumsGloablTensor_.GetValue(offsetInCnt);
    }

    if (Get<M_VALUE>(problemShape_) == 0) {
        return false;
    }
    return true;
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeTypeFunc>::UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo, uint32_t gmmIndex,
    uint32_t localExpertId)
{
    if (gmmIndex == 0) {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.dispatchRevDataPtr +
            Get<IDX_A_OFFSET>(baseOffset_) * sizeof(QuantOutType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.dispatchRevScalePtr +
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
        gmmAddrInfo.aGlobal = params_.workspaceInfo.swigluQuantDataPtr +
            Get<IDX_C_OFFSET>(baseOffset_) * sizeof(QuantOutType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.swigluQuantScalePtr + Get<IDX_C_SCALE_OFFSET>(baseOffset_) *
                                    sizeof(QuantScaleOutType);
        gmmAddrInfo.bGlobal = params_.b2GmAddr + Get<IDX_B2_OFFSET>(baseOffset_) * sizeof(QuantOutType);
        gmmAddrInfo.bScaleGlobal = params_.b2ScaleGmAddr + Get<IDX_B2_SCALE_OFFSET>(baseOffset_) *
                                    sizeof(QuantScaleOutType);
    }
    gmmAddrInfo.groupFlagList = (__gm__ int32_t*)params_.workspaceInfo.flagSwiGluToGmm2Ptr +
                                Get<IDX_FLAG_OFFSET>(baseOffset_) * INT_CACHELINE;
    gmmAddrInfo.groupFlagList2 = (__gm__ int32_t*)params_.workspaceInfo.flagDispatchToGmm1Ptr +
                                Get<IDX_FLAG_OFFSET>(baseOffset_) * INT_CACHELINE;
    groupFlagListGm_.SetGlobalBuffer(gmmAddrInfo.groupFlagList);
    groupFlagListGm2_.SetGlobalBuffer(gmmAddrInfo.groupFlagList2);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeTypeFunc>::SecondBuffInit()
{
    uint32_t dataResBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(UNPERMUTE_LIST_NUM * k_ * sizeof(bfloat16_t)), static_cast<uint32_t>(ALIGN_32));
    int32_t num = worldSize_ * Ops::Base::CeilAlign(
        static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128)) * sizeof(int32_t);
    uint32_t dataResFp32BufAlign = dataResBufAlign * HALF_TO_FP32;
    uint32_t topKWeightsBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(m_ * topK_ * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
    uint32_t tempBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(m_ * topK_ * sizeof(bfloat16_t)), uint32_t(ALIGN_32));

    dataResAddr_ = 0;
    dataResSize_ = dataResBufAlign / sizeof(bfloat16_t);
    dataResTensor_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, dataResAddr_, dataResSize_);
    resetTokenAddr_ = dataResAddr_ + dataResBufAlign;
    resetTokenSize_ = num / sizeof(int32_t);
    resetTokenTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, resetTokenAddr_, resetTokenSize_);
    dataResFp32Addr_ = resetTokenAddr_ + num;
    dataResFp32Size_ = dataResFp32BufAlign / sizeof(float);
    dataResFp32Tensor_ = LocalTensor<float>(TPosition::VECCALC, dataResFp32Addr_, dataResFp32Size_);
    topKWeightsAddr_ = dataResFp32Addr_ + dataResFp32BufAlign;
    topKWeightsSize_ = topKWeightsBufAlign / sizeof(float);
    topKWeightsTensor_ = LocalTensor<float>(TPosition::VECCALC, topKWeightsAddr_, topKWeightsSize_);
    uint32_t tempAddr = topKWeightsAddr_ + topKWeightsBufAlign;
    if constexpr (Std::IsSame<TopkWeightsType, float>::value) {
        GlobalTensor<float> topKWeightsGlobalTensor_;
        topKWeightsGlobalTensor_.SetGlobalBuffer((__gm__ float*)params_.probsGmAddr);
        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(m_ * topK_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(topKWeightsTensor_, topKWeightsGlobalTensor_, copyParams, copyPadParams);
    }
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        uint32_t tempSize = tempBufAlign / sizeof(bfloat16_t);
        LocalTensor<bfloat16_t> tempLocal(TPosition::VECCALC, tempAddr, tempSize);
        GlobalTensor<bfloat16_t> topkWeightsGlobalTensor;
        topkWeightsGlobalTensor.SetGlobalBuffer((__gm__ bfloat16_t*)params_.probsGmAddr);
        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(m_ * topK_ * sizeof(bfloat16_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<bfloat16_t> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(tempLocal, topkWeightsGlobalTensor, copyParams, copyPadParams);
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID>();
        Cast(topKWeightsTensor_, tempLocal, AscendC::RoundMode::CAST_NONE, m_ * topK_);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeTypeFunc>::Unpermute()
{
    int32_t coreLen, coreOffset;
    TilingByCore(m_, coreLen, coreOffset, 1);
    GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer((__gm__ bfloat16_t*)params_.peermemInfo.combineSendPtr);
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer((__gm__ bfloat16_t*)params_.y2GmAddr);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    for (int32_t tokenIdx = coreOffset; tokenIdx < coreLen + coreOffset; tokenIdx++) {
        SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID>();
        LocalTensor<bfloat16_t> dataIn0Bf16 = dataResTensor_[k_];
        LocalTensor<bfloat16_t> dataIn1Bf16 = dataResTensor_[k_ * 2];
        LocalTensor<float> dataIn0Fp32 = dataResFp32Tensor_[k_];
        LocalTensor<float> dataIn1Fp32 = dataResFp32Tensor_[k_ * 2];
        for (int32_t expId = 0; expId < topK_; ++expId) {
            float expScale = topKWeightsTensor_.GetValue(tokenIdx * topK_ + expId);
            auto event = (expId % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
            auto dataInBf16 = (expId % DOUBLE_BUFFER == 0) ? dataIn0Bf16 : dataIn1Bf16;
            auto dataInFp32 = (expId % DOUBLE_BUFFER == 0) ? dataIn0Fp32 : dataIn1Fp32;
            WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            DataCopy(dataInBf16, expandedX[(tokenIdx * topK_ + expId) * k_], k_);
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            SetFlag<AscendC::HardEvent::S_V>(event);
            WaitFlag<AscendC::HardEvent::S_V>(event);
            // bf16 -> fp32
            Cast(dataInFp32, dataInBf16, AscendC::RoundMode::CAST_NONE, k_);
            PipeBarrier<PIPE_V>();
            if (expId == 0) {
                Muls(dataResFp32Tensor_, dataInFp32, expScale, k_);
            } else {
                Muls(dataInFp32, dataInFp32, expScale, k_);
                PipeBarrier<PIPE_V>();
                Add(dataResFp32Tensor_, dataResFp32Tensor_, dataInFp32, k_);
                PipeBarrier<PIPE_V>();
            }
            SetFlag<AscendC::HardEvent::V_MTE2>(event);
        }
        // fp32 -> bf16
        Cast(dataResTensor_, dataResFp32Tensor_, AscendC::RoundMode::CAST_RINT, k_);
        SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID>();
        DataCopy(output[tokenIdx * k_], dataResTensor_, k_);
    }
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeTypeFunc>::CrossRankSyncInWorldSize()
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    __gm__ int32_t* syncRank = (__gm__ int32_t*)(params_.peermemInfo.rankSyncInWorldPtr);
    __gm__ int32_t* syncCount = (__gm__ int32_t*)(params_.peermemInfo.rankSyncInWorldPtr +
        48 * 1024 + aivCoreIdx_ * 64);
    int count = ReadGmByPassDCache(syncCount) + 1;
    for (int i = aivCoreIdx_; i < worldSize_; i += blockAivNum_) {
        __gm__ int32_t* syncRemoteAddr = (__gm__ int32_t*)(winRankAddr_[i]) + rankId_ * 16;
        WriteGmByPassDCache(syncRemoteAddr, count);
        auto syncCheck = syncRank + i * 16;
        GmSignalWaitBarrier(syncCheck, count);
    }
    WriteGmByPassDCache(syncCount, count);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeTypeFunc>::Process()
{
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    BuffInit();
    TopKIdsAllGather();
    ResetFlagList();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    QuantProcessInRank();

    // 2. 全卡同步
    CrossRankSyncInWorldSize();

    // 3. dispatch & GroupMatmulSwigluQuant
    FirstBuffInit();
    GMMAddrInfo gmmAddrInfo;
    for (int localExpertId = 0; localExpertId < expertPerRank_; localExpertId++) {
        if constexpr(g_coreType == AIV) {
            if (subBlockIdx_ == 1) {
                SendInfoCal(localExpertId);
            }
        }
        if (!UpdateGroupParams(localExpertId, 0)) {
            continue;
        }
        UpdateGlobalBuffer(gmmAddrInfo, 0, localExpertId);
        if constexpr(g_coreType == AIV) {
            if (subBlockIdx_ == 1) {
                TripleInfoCalAndDispatch(gmmAddrInfo, localExpertId);
            }
        }
        if (subBlockIdx_ == 0) {
            MegaMoeImpl::GroupMatmulSwigluQuant<
                QuantOutType, QuantOutType, float, QuantScaleOutType, QuantScaleOutType>(
                epilogueOp_, params_, problemShape_, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_);
        }
    }
    EndSync(vecSetSyncCom_);
    if constexpr(g_coreType == AIV) {
        if (subBlockIdx_ == 1) {
            ExpertTokenNumCopyOut();
        }
    }

    // 5. for循环按照本卡专家id进行gmm2 & combine，内部gmm2与combine之间按照基本块进行流水
    preOffset_ = 0;
    vecSetSyncCom_ = 0;
    expertBeforeCnt_ = 0;
    baseOffset_ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
        if (!UpdateGroupParams(expertIdx, 1)) {
            continue;
        }
        UpdateGlobalBuffer(gmmAddrInfo, 1, expertIdx);
        if (subBlockIdx_ == 0) {
            MegaMoeImpl::GroupMatmul2Wave<QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                QuantScaleOutType>(
                params_, problemShape_, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_, expertBeforeCnt_, expertIdx,
                gmm2PingPongIdx_,
                rankId_);
        }
    }
    EndGMM2Sync(vecSetSyncCom_, gmm2PingPongIdx_);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
    // 6. unpermute & 清理win区上放置token的位置
    if constexpr(g_coreType == AIV) {
        SecondBuffInit();
        CrossRankSyncInWorldSize(); // 全卡软同步，确认combine send完成
        Unpermute();
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

}   // namespace MegaMoeImpl
#endif
