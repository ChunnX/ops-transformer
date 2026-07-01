/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_ep_combine.h
 * \brief MoE Expert-Parallel Combine kernel implementation
 */
#ifndef MOE_EP_COMBINE_H
#define MOE_EP_COMBINE_H

#include "kernel_operator.h"
#include "moe_ep_combine_tiling_key.h"

#if __has_include("../common/op_kernel/moe_distribute_base.h")
#include "../common/op_kernel/moe_distribute_base.h"
#include "../common/op_kernel/mc2_kernel_utils.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#endif

#include "moe_ep_combine_base.h"
#include "moe_ep_combine_tiling.h"

#ifndef ALIGN_UP
#define ALIGN_UP(x, align) (((x) + (align) - 1) & ~((align) - 1))
#endif

namespace MoeEpCombineImpl {

using namespace AscendC;

#define TemplateMoeEpCombineTypeClass typename XType, uint32_t HasTopkWeight
#define TemplateMoeEpCombineTypeFunc XType, HasTopkWeight
#define HCOMM_INIT_SIZE 512UL

static constexpr uint32_t WIN_ADDR_ALIGN = 512;
static constexpr uint32_t RECV_META_FIELDS = 4;
constexpr uint64_t UB_ALIGN = 32UL;
constexpr uint32_t COMBINE_STATE_OFFSET = 64U * 1024U;  // 本卡状态空间偏移地址，前面的地址给dispatch用
constexpr uint32_t STATE_OFFSET = 32U;
constexpr uint32_t DCCI_OFFSET = 64U;
constexpr uint64_t ALIGNED_LEN_256 = 256UL;
constexpr uint32_t FLOAT_PER_UB_ALIGN = 8U;
constexpr size_t MASK_CALC_NEED_WORKSPACE = 20UL * 1024UL;

template <TemplateMoeEpCombineTypeClass>
class MoeEpCombine {
public:
    __aicore__ inline MoeEpCombine(){};

    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR recvSrcMetadata,
                                GM_ADDR numRecvPerExpert, GM_ADDR topkWeightsOptional, GM_ADDR bias0, GM_ADDR bias1,
                                GM_ADDR combinedX, GM_ADDR combinedTopkWeightsOptional, GM_ADDR workspace,
                                GM_ADDR tilingGM, TPipe *pipe, const MoeEpCombineInfo *tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
                                    uint32_t &endTokenId, uint32_t &tokenPerAivNum);
    __aicore__ inline void SendToken(GM_ADDR localWorkspaceAddr, GM_ADDR remoteRankWinAddr,
                                    uint32_t tokenIndex, uint32_t dstRankId);
    __aicore__ inline void SetStatus(GM_ADDR localRankStateAddr,  GM_ADDR remoteRankStateAddr,
                                    uint32_t tokenIndex, uint32_t dstRankId);
    __aicore__ inline bool WaitDispatch(uint32_t tokenIndex, uint32_t copyCount, uint32_t beginIndex);
    __aicore__ inline void ProcessTopKToken(uint32_t tokenIndex);
    __aicore__ inline void SendPhaseExpertToToken();
    __aicore__ inline void BuffInit();
    __aicore__ inline void MaskAlign(LocalTensor<half> maskCalcSelectedTensor);
    __aicore__ inline void MaskCheck();
    __aicore__ inline void RecvPhaseReduce();

    __aicore__ inline uint64_t GetCommHandle(uint32_t rankId)
    {
        uint32_t index = rankId > rankId_ ? rankId - 1 : rankId;
        return hcommHandle_[index];
    }
    __aicore__ inline GM_ADDR GetUrmaWinAddrByRankId(uint32_t rankId, uint64_t offset)
    {
        return (GM_ADDR)(winRankAddr_[rankId] + offset);
    }
    __aicore__ inline GM_ADDR GetUrmaStateAddrByRankId(uint32_t rankId, uint64_t offset)
    {
        return (GM_ADDR)(winRankAddr_[rankId] + offset);
    }

    __aicore__ inline GM_ADDR GetLocalWorkspaceDataAddr(const int32_t rankId, uint64_t offset)
    {
        return (GM_ADDR)localUrmaWorkspace_ + offset + localWsSizeDataPerRank_ * rankId;
    }
    __aicore__ inline GM_ADDR GetLocalWorkspaceStateAddr(const int32_t rankId)
    {
        return (GM_ADDR)localUrmaWorkspace_ + localWsSizeStatusPerRank_ * rankId;
    }

    TPipe *tpipe_{nullptr};
    const MoeEpCombineInfo *tilingData_{nullptr};
    __gm__ Mc2MoeContext *mc2Context_{nullptr};

    uint32_t rankId_{0};
    uint32_t epWorldSize_{0};
    uint32_t numMaxTokensPerRank_{0};
    uint32_t numTokens_{0};
    uint32_t topK_{0};
    uint32_t axisH_{0};
    uint32_t hAlignSize_{0};          // UB对齐后的hidden size
    uint64_t workspaceStateSize_{0};  // 发送侧workspace中给所有卡状态区的大小
    uint64_t CombineStateAddr_{0};    // Win区Combine 状态区的地址
    uint64_t CombineDataAddr_{0};    // Win区Combine 数据区的地址

    uint32_t hWeightAlignSize_{0};    // token+weight对齐后的hidden size
    uint32_t XTypeAlign32Size_{0};
    uint32_t perSlotBytes_{0};
    uint64_t actualA_{0};
    uint32_t aivNum_{0};
    uint32_t localmoeNum_{0};
    uint32_t totalWinSizeEp_{0};
    uint64_t localWsSizeDataPerRank_{0};
    uint64_t localWsSizeStatusPerRank_{0};
    uint64_t winStateOffset_{0};
    uint64_t winDataOffset_{0};

    uint32_t tStart_{0};
    uint32_t tEnd_{0};
    uint32_t tPerCore_{0};
    uint32_t ubXBytes_{0};
    uint32_t stateOffset_{0};

    uint32_t mask_tokenNum_{0};
    uint32_t bsKCastCnt_{0};
    uint32_t activeMaskAlignSize_{0};

    GlobalTensor<XType> xGm_;
    GlobalTensor<int32_t> topkIdxGm_;
    GlobalTensor<int32_t> recvSrcMetadataGm_;
    GlobalTensor<int64_t> numRecvPerExpertGm_;
    GlobalTensor<float> topkWeightsOptionalGm_;

    LocalTensor<XType> xTmpTensor_;
    LocalTensor<float> weightTmpTensor_;
    GlobalTensor<XType> combinedXGm_;
    GlobalTensor<float> combinedTopkWeightsGm_;

    LocalTensor<XType> ubX_;
    LocalTensor<float> ubAccFp32_;
    LocalTensor<float> ubTmpFp32_;
    LocalTensor<float> ubWeighted_;
    LocalTensor<uint8_t> hcommTensor_;
    LocalTensor<float> stateResetTensor_;

    LocalTensor<half> maskGenerateTensor_;
    LocalTensor<bool> maskStrideTensor_;
    LocalTensor<half> tokenTargetTensor_;

    TBuf<QuePosition::VECIN> ubXBuf_;
    TBuf<QuePosition::VECIN> ubAccFp32Buf_;
    TBuf<QuePosition::VECIN> ubTmpFp32Buf_;
    TBuf<QuePosition::VECIN> ubWeightedBuf_;
    TBuf<> readStateBuf_;
    TBuf<> tokenStatusBuf_;
    TBuf<> stateBuf_;
    TBuf<> stateSumBuf_;
    TBuf<> stateResetBuf_;
    TBuf<> hcommBuf_;

    TBuf<> compareBuf_;
    TBuf<> rowTmpFloatBuf_;
    TBuf<> tokenBuf_;
    TBuf<> tokenTargetTBuf_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_; // 数据队列
    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_; // todo: 通信上下文

    GM_ADDR winRankAddr_[HCCL_MAX_RANK_SIZE];
    uint64_t hcommHandle_[HCCL_MAX_RANK_SIZE];
    GM_ADDR localUrmaWorkspace_{nullptr};      // 工作空间的GM地址
    GM_ADDR maskCalcWorkspaceGM_{nullptr};
    uint32_t aivId_{0};
};

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR recvSrcMetadata, GM_ADDR numRecvPerExpert,
    GM_ADDR topkWeightsOptional, GM_ADDR bias0, GM_ADDR bias1, GM_ADDR combinedX, GM_ADDR combinedTopkWeightsOptional,
    GM_ADDR workspace, GM_ADDR tilingGM, TPipe *pipe, const MoeEpCombineInfo *tilingData)
{
    tpipe_ = pipe;
    tilingData_ = tilingData;
    aivId_ = GetBlockIdx();
    localUrmaWorkspace_ = workspace;
    maskCalcWorkspaceGM_ = workspace + aivId_ * MASK_CALC_NEED_WORKSPACE;
    epWorldSize_ = tilingData_->cfg.epWorldSize;
    numMaxTokensPerRank_ = tilingData_->cfg.numMaxTokensPerRank;
    numTokens_ = tilingData_->cfg.numTokens;
    topK_ = tilingData_->cfg.topK;
    axisH_ = tilingData_->cfg.hidden;
    perSlotBytes_ = tilingData_->cfg.perSlotBytes;
    aivNum_ = tilingData_->aivNum;
    localmoeNum_ = tilingData_->cfg.numLocalExperts;
    totalWinSizeEp_ = tilingData->totalWinSizeEp;
    localWsSizeDataPerRank_ = tilingData->localWsSizeDataPerRank;
    localWsSizeStatusPerRank_ = tilingData->localWsSizeStatusPerRank;
    winStateOffset_ = tilingData->winStateOffset;
    winDataOffset_ = tilingData->winDataOffset;
    hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN; // UB 32字节对齐
    hWeightAlignSize_ = hAlignSize_ + UB_ALIGN; // UB 32字节对齐
    stateOffset_ = STATE_OFFSET;
    
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);

    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext *>(context);
    rankId_ = mc2Context_->epRankId;
    for (uint32_t i = 0; i < epWorldSize_; ++i) {
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
        hcommHandle_[i] = mc2Context_->hcommHandle[i];
    }

    workspaceStateSize_ = static_cast<uint64_t>(numMaxTokensPerRank_) * topK_ * UB_ALIGN * epWorldSize_;
    // workspaceStateSize_ = 16 * 1024 * 1024;
    CombineStateAddr_ = tilingData->winStateOffset;
    // CombineStateAddr_ = GetWinCombineStateAddrByRankId(mc2Context_);
    CombineDataAddr_ = tilingData->winDataOffset;
    // CombineDataAddr_ = GetWinCombineDataAddrByRankId(mc2Context_);

    xGm_.SetGlobalBuffer((__gm__ XType *)x);
    topkIdxGm_.SetGlobalBuffer((__gm__ int32_t *)topkIdx);
    recvSrcMetadataGm_.SetGlobalBuffer((__gm__ int32_t *)recvSrcMetadata);
    numRecvPerExpertGm_.SetGlobalBuffer((__gm__ int64_t *)numRecvPerExpert);
    combinedXGm_.SetGlobalBuffer((__gm__ XType *)combinedX);

    // 计算actualA_的大小
    for (int i = 0; i < localmoeNum_; i++) {
        if (i % (DCCI_OFFSET / sizeof(int64_t)) == 0) {
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                numRecvPerExpertGm_[i]);
        }
        actualA_ += numRecvPerExpertGm_.GetValue(i);
    }
    ubXBytes_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;
    XTypeAlign32Size_ = hAlignSize_;
    if constexpr (HasTopkWeight == 1) {
        XTypeAlign32Size_ = hWeightAlignSize_;
        topkWeightsOptionalGm_.SetGlobalBuffer((__gm__ float *)topkWeightsOptional);
        combinedTopkWeightsGm_.SetGlobalBuffer((__gm__ float *)combinedTopkWeightsOptional);
    }
    tpipe_->InitBuffer(xQueue_, 1, XTypeAlign32Size_);
    tpipe_->InitBuffer(readStateBuf_, UB_ALIGN);  // 32
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
    uint32_t &endTokenId, uint32_t &sendTokenNum)
{
    sendTokenNum = curSendCnt / curUseAivNum;                // 每个aiv需要发送的token数
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum;  // 余数
    uint32_t newAivId = aivId_;

    startTokenId = sendTokenNum * newAivId;  // 每个aiv发送时的起始rankid
    if (newAivId < remainderTokenNum) {      // 前remainderRankNum个aiv需要多发1个卡的数据
        sendTokenNum += 1;
        startTokenId += newAivId;
    } else {
        startTokenId += remainderTokenNum;
    }
    endTokenId = startTokenId + sendTokenNum;
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendToken(
    GM_ADDR localWorkspaceAddr,  GM_ADDR remoteRankWinAddr, uint32_t tokenIndex, uint32_t dstRankId)
{
    // 处理单个token：
    // 1. 从输入GM读取token数据到UB
    // 2. 将token数据写入目标rank的窗口
    GlobalTensor<XType> outTokenGT;
    outTokenGT.SetGlobalBuffer((__gm__ XType*)localWorkspaceAddr);
    if (dstRankId == rankId_) {
        outTokenGT.SetGlobalBuffer((__gm__ XType*)remoteRankWinAddr); // 目标rank是epRankId，直接写入remoteAddr
    }
    
    DataCopyPadParams padParams = {false, 0, 0, 0};
    DataCopyParams xCopyParams = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    DataCopyParams hCommuCopyOutParams = {1U, static_cast<uint16_t>(hAlignSize_), 0U, 0U};
    DataCopyParams weightCommuOutParams = {1U, static_cast<uint16_t>(sizeof(float)), 0U, 0U};

    // 从GM读取token到UB
    xTmpTensor_ = xQueue_.AllocTensor<XType>();
    DataCopyPad(xTmpTensor_, xGm_[tokenIndex * axisH_], xCopyParams, padParams);
    if constexpr (HasTopkWeight == 1) {
        // 从GM读取WeightData到UB
        DataCopyPad(xTmpTensor_[hAlignSize_ / sizeof(XType)].template ReinterpretCast<float>(),
            topkWeightsOptionalGm_[tokenIndex], weightCommuOutParams, padParams);
        hCommuCopyOutParams = {1U, static_cast<uint16_t>(hWeightAlignSize_), 0U, 0U};
    }
    xQueue_.EnQue(xTmpTensor_);
    xTmpTensor_ = xQueue_.DeQue<XType>();

    // 写入目标rank的窗口
    DataCopyPad(outTokenGT, xTmpTensor_, hCommuCopyOutParams);
    xQueue_.FreeTensor<XType>(xTmpTensor_);
    PipeBarrier<PIPE_MTE3>(); // 是否需要这个流水
    if (dstRankId != rankId_) {
        hcomm_.WriteNbi(GetCommHandle(dstRankId), remoteRankWinAddr, localWorkspaceAddr, XTypeAlign32Size_);
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SetStatus(
    GM_ADDR localRankStateAddr,  GM_ADDR remoteRankStateAddr, uint32_t tokenIndex, uint32_t dstRankId)
{
    // ============ 将状态写入目标rank的状态区 ============
    // 写入位置: 目标rank状态区 + 当前rank的偏移
    LocalTensor<float> statusTensor = readStateBuf_.Get<float>();
    Duplicate<float>(statusTensor, (float)1, FLOAT_PER_UB_ALIGN);
    GlobalTensor<float> outStateTensor;
    outStateTensor.SetGlobalBuffer((__gm__ float*)localRankStateAddr);
    if (dstRankId == rankId_) {
        outStateTensor.SetGlobalBuffer((__gm__ float*)remoteRankStateAddr); // 目标rank是epRankId，直接写入remoteAddr
    }
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy<float>(outStateTensor, statusTensor, 8UL);
    PipeBarrier<PIPE_MTE3>();
    if (dstRankId != rankId_) {
        hcomm_.Drain(GetCommHandle(dstRankId));
        hcomm_.WriteNbi(GetCommHandle(dstRankId), remoteRankStateAddr, localRankStateAddr, UB_ALIGN);
        hcomm_.Drain(GetCommHandle(dstRankId));
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendPhaseExpertToToken()
{
    uint32_t startRankId, endRankId, sendRankNum;
    SplitToCore(epWorldSize_, aivNum_, startRankId, endRankId, sendRankNum);
    if (startRankId >= endRankId || sendRankNum == 0) {
        return;
    }
    for (uint32_t rankId = startRankId; rankId < endRankId; rankId++) {
        for (uint32_t tokenIndex = 0; tokenIndex < actualA_; ++tokenIndex) {
            int32_t src_rank = recvSrcMetadataGm_.GetValue(tokenIndex * RECV_META_FIELDS + 0);
            int32_t src_token_idx = recvSrcMetadataGm_.GetValue(tokenIndex * RECV_META_FIELDS + 1);
            int32_t src_topK_idx = recvSrcMetadataGm_.GetValue(tokenIndex * RECV_META_FIELDS + 2);
            if (src_rank < 0 || src_rank!= rankId) { // 只处理当前rank负责的token
                continue;
            }
            // 计算目标窗口地址:
            // 当前rank的workspaceGM地址（src_rank只是为了workspace内的以卡分区）
            uint64_t slotOffset = (static_cast<uint64_t>(src_token_idx) * topK_ + src_topK_idx) * perSlotBytes_;
            GM_ADDR localRankWorkSpaceAddr = GetLocalWorkspaceDataAddr(src_rank, workspaceStateSize_) +
                slotOffset;
            GM_ADDR remoteRankWinAddr = GetUrmaWinAddrByRankId(src_rank, CombineDataAddr_) +
                slotOffset;
            SendToken(localRankWorkSpaceAddr, remoteRankWinAddr, tokenIndex, src_rank);
            PipeBarrier<PIPE_MTE3>();
            GM_ADDR localRankStateAddr = GetLocalWorkspaceStateAddr(src_rank) +
                (src_token_idx * topK_ + src_topK_idx) * stateOffset_;
            GM_ADDR remoteRankStateAddr = GetUrmaStateAddrByRankId(src_rank, CombineStateAddr_) +
                (src_token_idx * topK_ + src_topK_idx) * stateOffset_;

            SetStatus(localRankStateAddr, remoteRankStateAddr, tokenIndex, src_rank);
        }
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::BuffInit()
{
    tpipe_->Reset();
    SplitToCore(numTokens_, aivNum_, tStart_, tEnd_, tPerCore_);
    if (tStart_ >= numTokens_) {return;}

    mask_tokenNum_ = tPerCore_ * topK_;
    uint32_t bsKInt32Align = Ceil(mask_tokenNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t bsKFloatAlign = Ceil(mask_tokenNum_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    uint32_t bsKHalfAlign = Ceil(mask_tokenNum_ * sizeof(half), UB_ALIGN) * UB_ALIGN;
    uint32_t bsHalfAlign = Ceil(tPerCore_ * sizeof(half), UB_ALIGN) * UB_ALIGN;

    bsKCastCnt_ = Ceil(mask_tokenNum_ * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    activeMaskAlignSize_ = tPerCore_ * Ceil(topK_ * sizeof(half), UB_ALIGN) * UB_ALIGN;
    bsKInt32Align = (bsKInt32Align > bsKCastCnt_ ? bsKInt32Align : bsKCastCnt_);
    bsKHalfAlign = (bsKHalfAlign > activeMaskAlignSize_ ? bsKHalfAlign : activeMaskAlignSize_);
    bsKFloatAlign = (bsKFloatAlign > bsKCastCnt_ ? bsKFloatAlign : bsKCastCnt_);
    bsKFloatAlign = (bsKFloatAlign > activeMaskAlignSize_ ? bsKFloatAlign : activeMaskAlignSize_);

    tpipe_->InitBuffer(tokenTargetTBuf_, bsHalfAlign);
    tpipe_->InitBuffer(compareBuf_, bsKInt32Align);
    tpipe_->InitBuffer(rowTmpFloatBuf_, bsKFloatAlign);
    tpipe_->InitBuffer(tokenBuf_, bsKHalfAlign);

    if constexpr (HasTopkWeight == 1) {
        tpipe_->InitBuffer(ubWeightedBuf_, sizeof(float));
        ubWeighted_ = ubWeightedBuf_.Get<float>();
    }
    tpipe_->InitBuffer(ubXBuf_, ubXBytes_);
    tpipe_->InitBuffer(ubAccFp32Buf_, axisH_ * sizeof(float));
    tpipe_->InitBuffer(ubTmpFp32Buf_, axisH_ * sizeof(float));

    ubX_ = ubXBuf_.Get<XType>();
    ubAccFp32_ = ubAccFp32Buf_.Get<float>();
    ubTmpFp32_ = ubTmpFp32Buf_.Get<float>();
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline bool MoeEpCombine<TemplateMoeEpCombineTypeFunc>::WaitDispatch(uint32_t tokenIndex,
     uint32_t copyCount, uint32_t beginIndex)
{
    uint32_t targetCount = copyCount;
    float target = (float)1.0 * targetCount;
    float minTarget = target - (float)0.5;
    float maxTarget = target + (float)0.5;
    // 计算地址偏移
    GM_ADDR stateGM = GetUrmaStateAddrByRankId(rankId_, CombineStateAddr_) + tokenIndex * topK_ * stateOffset_;
    GlobalTensor<float> stateGMTensor;
    stateGMTensor.SetGlobalBuffer((__gm__ float*)stateGM);
    float localState = 0;
    SumParams sumParams{1, copyCount, copyCount};
    LocalTensor<float> stateTensor = stateBuf_.Get<float>();
    LocalTensor<float> stateSumTensor = stateSumBuf_.Get<float>();
    SyncFunc<AscendC::HardEvent::S_MTE2>();
    DataCopy<float>(stateTensor, stateGMTensor, copyCount);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Sum(stateSumTensor, stateTensor, sumParams);
    SyncFunc<AscendC::HardEvent::V_S>();
    localState = stateSumTensor(0);
    if (((minTarget < localState) && (localState < maxTarget))) {
        // 计算地址偏移，清状态
        DataCopy<float>(stateGMTensor, stateResetTensor_, copyCount);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        return true;
    }
    return false;
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::ProcessTopKToken(uint32_t tokenIndex)
{
    Duplicate<float>(ubAccFp32_, (float)0, axisH_);
    DataCopyPadParams padParams = {false, 0, 0, 0};
    DataCopyParams xCopyParams = {1U, static_cast<uint16_t>(hAlignSize_), 0U, 0U};
    DataCopyParams weightCopyParams = {1U, static_cast<uint16_t>(sizeof(float)), 0U, 0U};
    for (uint32_t topkId = 0U; topkId < topK_; topkId++) {
        // 读取expert_id
        GM_ADDR wAddr = GetUrmaWinAddrByRankId(rankId_, CombineDataAddr_) +
                (tokenIndex * topK_ + topkId) * perSlotBytes_;
        GlobalTensor<XType> srcTokenTensor;
        srcTokenTensor.SetGlobalBuffer(reinterpret_cast<__gm__ XType *>(wAddr));
        DataCopyPad(ubX_, srcTokenTensor, xCopyParams, padParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Cast(ubTmpFp32_, ubX_, AscendC::RoundMode::CAST_NONE, axisH_);
        Add(ubAccFp32_, ubAccFp32_, ubTmpFp32_, axisH_);
        if constexpr (HasTopkWeight == 1) {
            GM_ADDR weightAddr = wAddr + hAlignSize_;
            GlobalTensor<float> srcWeightTensor;
            srcWeightTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weightAddr));
            DataCopyPad(ubWeighted_, srcWeightTensor, weightCopyParams, padParams);
            SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
            DataCopyPad(combinedTopkWeightsGm_[tokenIndex * topK_ + topkId], ubWeighted_, weightCopyParams);
            SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        }
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::RecvPhaseReduce()
{
    DataCopyPadParams padParams = {false, 0, 0, 0};
    DataCopyParams xCopyParams = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    tpipe_->InitBuffer(tokenStatusBuf_, Ceil(tPerCore_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);
    tpipe_->InitBuffer(stateBuf_, topK_ * STATE_OFFSET);
    tpipe_->InitBuffer(stateSumBuf_, UB_ALIGN);
    tpipe_->InitBuffer(stateResetBuf_, topK_ * STATE_OFFSET);      // 清理状态区
    stateResetTensor_ = stateResetBuf_.Get<float>();
    Duplicate<float>(stateResetTensor_, (float)0.0, static_cast<uint32_t>(topK_ * FLOAT_PER_UB_ALIGN));
    LocalTensor<int32_t> tokenStatusTensor = tokenStatusBuf_.Get<int32_t>();
    Duplicate<int32_t>(tokenStatusTensor, static_cast<int32_t>(0), tPerCore_);
    SyncFunc<AscendC::HardEvent::V_S>();
    uint32_t CompletedtokenNum = static_cast<uint32_t>(0);
    uint32_t copyCount = topK_ * FLOAT_PER_UB_ALIGN;
    while (CompletedtokenNum != tPerCore_) {
        for (uint32_t tokenIdx = tStart_; tokenIdx < tEnd_; ++tokenIdx) {
            if (tokenStatusTensor(tokenIdx - tStart_) == 1) {
                continue;
            }
            if (!WaitDispatch(tokenIdx, copyCount, tStart_)) {
                continue;
            }
            CompletedtokenNum++;
            tokenStatusTensor.SetValue(tokenIdx - tStart_, 1);
            ProcessTopKToken(tokenIdx);
            LocalTensor<XType> ubResultBf16 = ubX_;
            Cast(ubResultBf16, ubAccFp32_, RoundMode::CAST_RINT, axisH_);
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            DataCopyPad(combinedXGm_[tokenIdx * axisH_], ubResultBf16, xCopyParams);
        }
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::Process()
{
    SendPhaseExpertToToken();
    PipeBarrier<PIPE_ALL>();  // MaskCheck中包含reset操作，需确保前面操作完成
    BuffInit();
    RecvPhaseReduce();
}

} // namespace MoeEpCombineImpl

#endif // MOE_EP_COMBINE_H