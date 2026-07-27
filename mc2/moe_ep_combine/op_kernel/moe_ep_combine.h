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

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/reduce.h"
#include "adv_api/reduce/sum.h"
#include "adv_api/hcomm/hcomm.h"

#include "moe_ep_combine_tiling_key.h"
#if __has_include("../common/moe_distribute_base.h")
#include "../common/moe_distribute_base.h"
#include "../common/mc2_kernel_utils.h"
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
constexpr uint32_t COMBINE_STATE_OFFSET = 64U * 1024U; // 本卡状态空间偏移地址，前面的地址给dispatch用
constexpr uint32_t STATE_OFFSET = 32U;
constexpr uint32_t DCCI_OFFSET = 64U;
constexpr uint64_t ALIGNED_LEN_256 = 256UL;
constexpr uint32_t FLOAT_PER_UB_ALIGN = 8U;
constexpr uint32_t SEND_DOUBLE_BUFFER_NUM = 2U;
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
    __aicore__ inline void CopyTokenToQueue(uint32_t tokenIndex);
    __aicore__ inline void CopyTokenFromQueue(GM_ADDR writeAddr);
    __aicore__ inline void SendQueuedToken(int32_t srcRank, int32_t srcTokenIdx, int32_t srcTopKIdx,
                                           uint32_t channelIndex);
    __aicore__ inline bool WaitDispatch(uint32_t tokenIndex, uint32_t copyCount);
    __aicore__ inline void ProcessTopKToken(uint32_t tokenIndex);
    __aicore__ inline void SendPhaseExpertToToken();
    __aicore__ inline void GetCoreAssignment(uint32_t totalBlocks, uint32_t &targetRank,
                                              uint32_t &coreIndexInGroup, uint32_t &groupSize);
    __aicore__ inline void BuffInit();
    __aicore__ inline void MaskAlign(LocalTensor<half> maskCalcSelectedTensor);
    __aicore__ inline void MaskCheck();
    __aicore__ inline void RecvPhaseReduce();

    __aicore__ inline uint64_t GetCommHandle(uint32_t rankId, uint32_t channelIndex)
    {
        return hcommHandle_[rankId * channelsPerRank_ + channelIndex];
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
    __gm__ Mc2Aclnn::MoeCommContext *mc2Context_{nullptr};

    uint32_t rankId_{0};
    uint32_t epWorldSize_{0};
    uint32_t channelsPerRank_{1};
    uint32_t numMaxTokensPerRank_{0};
    uint32_t numTokens_{0};
    uint32_t topK_{0};
    uint32_t axisH_{0};
    uint32_t hAlignSize_{0};         // UB对齐后的hidden size
    uint64_t workspaceStateSize_{0}; // 发送侧workspace中给所有卡状态区的大小
    uint64_t CombineStateAddr_{0};   // Win区Combine 状态区的地址
    uint64_t CombineDataAddr_{0};    // Win区Combine 数据区的地址

    uint32_t hWeightAlignSize_{0}; // token+weight对齐后的hidden size
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

    GlobalTensor<XType> combinedXGm_;
    GlobalTensor<float> combinedTopkWeightsGm_;

    LocalTensor<XType> ubX_;
    LocalTensor<float> ubAccFp32_;
    LocalTensor<float> ubTmpFp32_;
    LocalTensor<float> ubWeighted_;
    LocalTensor<uint32_t> statusTensor_;
    LocalTensor<uint8_t> hcommTensor_;
    LocalTensor<uint32_t> stateResetTensor_;

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
    TBuf<> metadataBuf_; // 发送阶段recvSrcMetadata的UB缓冲，批量搬运避免GetValue

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_; // 数据队列
    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;                 // 通信上下文

    GM_ADDR winRankAddr_[Mc2Aclnn::HCCL_MAX_RANK_SIZE];
    uint64_t hcommHandle_[Mc2Aclnn::HCCL_MAX_RANK_SIZE];
    GM_ADDR localUrmaWorkspace_{nullptr}; // 工作空间的GM地址
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
    hWeightAlignSize_ = hAlignSize_ + UB_ALIGN;                      // UB 32字节对齐
    stateOffset_ = STATE_OFFSET;

    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);

    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::MoeCommContext *>(context);
    rankId_ = mc2Context_->epRankId;
    channelsPerRank_ = mc2Context_->channelsPerRank;
    if (channelsPerRank_ == 0) {
        channelsPerRank_ = 1;
    }
    if (epWorldSize_ > 0 && channelsPerRank_ > Mc2Aclnn::HCCL_MAX_RANK_SIZE / epWorldSize_) {
        channelsPerRank_ = 1;
    }
    for (uint32_t i = 0; i < epWorldSize_; ++i) {
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
    }
    uint32_t handleCount = epWorldSize_ * channelsPerRank_;
    for (uint32_t i = 0; i < handleCount; ++i) {
        hcommHandle_[i] = mc2Context_->hcommHandle[i];
    }

    workspaceStateSize_ = static_cast<uint64_t>(numMaxTokensPerRank_) * topK_ * UB_ALIGN * epWorldSize_;
    CombineStateAddr_ = tilingData->winStateOffset;
    CombineDataAddr_ = tilingData->winDataOffset;

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
    tpipe_->InitBuffer(xQueue_, SEND_DOUBLE_BUFFER_NUM, XTypeAlign32Size_);
    tpipe_->InitBuffer(readStateBuf_, UB_ALIGN); // 32
    statusTensor_ = readStateBuf_.Get<uint32_t>();
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId, uint32_t &endTokenId, uint32_t &sendTokenNum)
{
    sendTokenNum = curSendCnt / curUseAivNum;               // 每个aiv需要发送的token数
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum; // 余数
    uint32_t newAivId = aivId_;

    startTokenId = sendTokenNum * newAivId; // 每个aiv发送时的起始rankid
    if (newAivId < remainderTokenNum) {     // 前remainderRankNum个aiv需要多发1个卡的数据
        sendTokenNum += 1;
        startTokenId += newAivId;
    } else {
        startTokenId += remainderTokenNum;
    }
    endTokenId = startTokenId + sendTokenNum;
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::CopyTokenToQueue(uint32_t tokenIndex)
{
    DataCopyPadParams padParams = {false, 0, 0, 0};
    DataCopyParams xCopyParams = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};

    LocalTensor<XType> tokenTensor = xQueue_.AllocTensor<XType>();
    DataCopyPad(tokenTensor, xGm_[tokenIndex * axisH_], xCopyParams, padParams);
    if constexpr (HasTopkWeight == 1) {
        DataCopyParams weightCopyParams = {1U, static_cast<uint16_t>(sizeof(float)), 0U, 0U};
        DataCopyPad(tokenTensor[hAlignSize_ / sizeof(XType)].template ReinterpretCast<float>(),
                    topkWeightsOptionalGm_[tokenIndex], weightCopyParams, padParams);
    }
    xQueue_.EnQue(tokenTensor);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::CopyTokenFromQueue(GM_ADDR writeAddr)
{
    GlobalTensor<XType> outTokenGT;
    outTokenGT.SetGlobalBuffer((__gm__ XType *)writeAddr);

    DataCopyParams copyOutParams = {1U, static_cast<uint16_t>(XTypeAlign32Size_), 0U, 0U};
    LocalTensor<XType> tokenTensor = xQueue_.DeQue<XType>();
    DataCopyPad(outTokenGT, tokenTensor, copyOutParams);
    xQueue_.FreeTensor<XType>(tokenTensor);
    // HCOMM must not consume the workspace before the UB-to-GM copy completes.
    PipeBarrier<PIPE_MTE3>();
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendQueuedToken(
    int32_t srcRank, int32_t srcTokenIdx, int32_t srcTopKIdx, uint32_t channelIndex)
{
    uint64_t slotOffset = (static_cast<uint64_t>(srcTokenIdx) * topK_ + srcTopKIdx) * perSlotBytes_;
    if (srcRank != static_cast<int32_t>(rankId_)) {
        GM_ADDR remoteRankWinAddr = GetUrmaWinAddrByRankId(srcRank, CombineDataAddr_) + slotOffset;
        GM_ADDR localRankWorkSpaceAddr = GetLocalWorkspaceDataAddr(srcRank, workspaceStateSize_) + slotOffset;
        CopyTokenFromQueue(localRankWorkSpaceAddr);
        GM_ADDR remoteRankStateAddr = GetUrmaStateAddrByRankId(srcRank, CombineStateAddr_) +
                                      (srcTokenIdx * topK_ + srcTopKIdx) * WIN_ADDR_ALIGN;
        uint64_t commHandle = GetCommHandle(srcRank, channelIndex);
        hcomm_.WriteWithNotifyNbi(commHandle, remoteRankWinAddr, localRankWorkSpaceAddr, XTypeAlign32Size_,
                                  remoteRankStateAddr, 1);
    } else {
        GM_ADDR localRankWinAddr = GetUrmaWinAddrByRankId(srcRank, CombineDataAddr_) + slotOffset;
        CopyTokenFromQueue(localRankWinAddr);
        GM_ADDR localStateAddr = GetUrmaStateAddrByRankId(srcRank, CombineStateAddr_) +
                                 (srcTokenIdx * topK_ + srcTopKIdx) * WIN_ADDR_ALIGN;
        GlobalTensor<uint32_t> state;
        state.SetGlobalBuffer((__gm__ uint32_t *)localStateAddr);
        DataCopy(state, statusTensor_, FLOAT_PER_UB_ALIGN);
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::GetCoreAssignment(
    uint32_t totalBlocks, uint32_t &targetRank, uint32_t &coreIndexInGroup, uint32_t &groupSize)
{
    uint32_t baseGroupSize = totalBlocks / epWorldSize_;
    uint32_t remainder = totalBlocks % epWorldSize_;
    uint32_t accumulated = 0;
    for (uint32_t rank = 0; rank < epWorldSize_; ++rank) {
        uint32_t currentGroupSize = baseGroupSize + ((rank < remainder) ? 1U : 0U);
        if (aivId_ < accumulated + currentGroupSize) {
            targetRank = rank;
            groupSize = currentGroupSize;
            coreIndexInGroup = aivId_ - accumulated;
            return;
        }
        accumulated += currentGroupSize;
    }
    targetRank = epWorldSize_;
    groupSize = 0;
    coreIndexInGroup = 0;
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendPhaseExpertToToken()
{
    if (actualA_ == 0 || epWorldSize_ == 0 || aivNum_ == 0) {
        return;
    }
    uint32_t activeAivNum = aivNum_;
    uint32_t maxChannelAivNum = epWorldSize_ * channelsPerRank_;
    if (activeAivNum > maxChannelAivNum) {
        activeAivNum = maxChannelAivNum;
    }
    if (aivId_ >= activeAivNum) {
        return;
    }

    bool splitRankTokens = activeAivNum >= epWorldSize_;
    uint32_t targetRank = epWorldSize_;
    uint32_t coreIndexInGroup = 0;
    uint32_t groupSize = 1;
    if (splitRankTokens) {
        // Split cores by rank first. The cores in one rank group use fixed, distinct channels.
        GetCoreAssignment(activeAivNum, targetRank, coreIndexInGroup, groupSize);
        if (targetRank >= epWorldSize_ || groupSize == 0) {
            return;
        }
    }
    uint32_t channelIndex = splitRankTokens ? coreIndexInGroup : 0;
    uint32_t targetRankTokenIndex = 0;
    Duplicate<uint32_t>(statusTensor_, (uint32_t)1, FLOAT_PER_UB_ALIGN);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    constexpr uint32_t metaBytesPerToken = RECV_META_FIELDS * sizeof(int32_t);
    constexpr uint32_t metaChunkTokenMax = 8192U; // 分块上限，平衡UB占用与搬运次数（128KB，A5 UB 256KB余量充足）

    uint32_t metaChunkTokens = (actualA_ < metaChunkTokenMax) ? actualA_ : metaChunkTokenMax;
    uint32_t metaChunkBytes = Ceil(metaChunkTokens * metaBytesPerToken, UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(metadataBuf_, metaChunkBytes);
    LocalTensor<int32_t> metadataLocal = metadataBuf_.Get<int32_t>();
    const DataCopyPadExtParams<int32_t> metaPadParams{false, 0U, 0U, 0U};
    for (uint64_t chunkStart = 0; chunkStart < actualA_; chunkStart += metaChunkTokens) {
        uint64_t chunkEnd = (chunkStart + metaChunkTokens > actualA_) ? actualA_ : (chunkStart + metaChunkTokens);
        uint32_t curChunkTokens = static_cast<uint32_t>(chunkEnd - chunkStart);

        DataCopyExtParams metaCopyParams{1U, curChunkTokens * metaBytesPerToken, 0U, 0U, 0U};
        DataCopyPad(metadataLocal, recvSrcMetadataGm_[chunkStart * RECV_META_FIELDS], metaCopyParams, metaPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        for (uint32_t i = 0; i < curChunkTokens; ++i) {
            uint32_t tokenIndex = static_cast<uint32_t>(chunkStart) + i;
            int32_t src_rank = metadataLocal.GetValue(i * RECV_META_FIELDS + 0);
            int32_t src_token_idx = metadataLocal.GetValue(i * RECV_META_FIELDS + 1);
            int32_t src_topK_idx = metadataLocal.GetValue(i * RECV_META_FIELDS + 2);
            if (src_rank < 0 || src_rank >= static_cast<int32_t>(epWorldSize_)) {
                continue;
            }
            if (splitRankTokens) {
                if (src_rank != static_cast<int32_t>(targetRank)) {
                    continue;
                }
                uint32_t currentRankTokenIndex = targetRankTokenIndex++;
                if (currentRankTokenIndex % groupSize != coreIndexInGroup) {
                    continue;
                }
            } else if (static_cast<uint32_t>(src_rank) % activeAivNum != aivId_) {
                continue;
            }
            // TQue with two buffers manages the MTE2/MTE3 ping-pong across loop iterations.
            CopyTokenToQueue(tokenIndex);
            SendQueuedToken(src_rank, src_token_idx, src_topK_idx, channelIndex);
        }
        SyncFunc<AscendC::HardEvent::S_MTE2>(); // 确保本轮 Scalar GetValue 读完，下一块 DataCopyPad 才可覆盖 metadataLocal
    }
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(recvSrcMetadataGm_);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::BuffInit()
{
    tpipe_->Reset();
    SplitToCore(numTokens_, aivNum_, tStart_, tEnd_, tPerCore_);
    if (tStart_ >= numTokens_) {
        return;
    }

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
__aicore__ inline bool MoeEpCombine<TemplateMoeEpCombineTypeFunc>::WaitDispatch(uint32_t tokenIndex, uint32_t copyCount)
{
    // 计算地址偏移
    GM_ADDR stateGM = GetUrmaStateAddrByRankId(rankId_, CombineStateAddr_) + tokenIndex * topK_ * WIN_ADDR_ALIGN;
    GlobalTensor<uint32_t> stateGMTensor;
    stateGMTensor.SetGlobalBuffer((__gm__ uint32_t *)stateGM);

    LocalTensor<uint32_t> stateTensor = stateBuf_.Get<uint32_t>();
    SyncFunc<AscendC::HardEvent::S_MTE2>();
    DataCopyParams params = {static_cast<uint16_t>(topK_), 1, WIN_ADDR_ALIGN / UB_ALIGN - 1, 0};
    DataCopy<uint32_t>(stateTensor, stateGMTensor, params);
    SyncFunc<AscendC::HardEvent::MTE2_V>();

    LocalTensor<uint32_t> stateSumTensor = stateSumBuf_.Get<uint32_t>();
    uint32_t shape[] = {topK_, UB_ALIGN / sizeof(uint32_t)};
    ReduceSum<uint32_t, AscendC::Pattern::Reduce::RA, false>(stateSumTensor, stateTensor, shape, true);
    SyncFunc<AscendC::HardEvent::V_S>();

    uint32_t localState = stateSumTensor(0);
    if (localState == copyCount) {
        DataCopyParams resetParams = {static_cast<uint16_t>(topK_), 1, 0, WIN_ADDR_ALIGN / UB_ALIGN - 1};
        DataCopy<uint32_t>(stateGMTensor, stateResetTensor_, resetParams);
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
        uint64_t slotOffset = (static_cast<uint64_t>(tokenIndex) * topK_ + topkId) * perSlotBytes_;
        GM_ADDR wAddr = GetUrmaWinAddrByRankId(rankId_, CombineDataAddr_) + slotOffset;
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
    tpipe_->InitBuffer(stateResetBuf_, topK_ * STATE_OFFSET); // 清理状态区
    stateResetTensor_ = stateResetBuf_.Get<uint32_t>();
    Duplicate<uint32_t>(stateResetTensor_, (uint32_t)0.0, static_cast<uint32_t>(topK_ * FLOAT_PER_UB_ALIGN));
    LocalTensor<int32_t> tokenStatusTensor = tokenStatusBuf_.Get<int32_t>();
    Duplicate<int32_t>(tokenStatusTensor, static_cast<int32_t>(0), tPerCore_);
    SyncFunc<AscendC::HardEvent::V_S>();
    uint32_t CompletedtokenNum = static_cast<uint32_t>(0);
    uint32_t copyCount = topK_;
    while (CompletedtokenNum != tPerCore_) {
        for (uint32_t tokenIdx = tStart_; tokenIdx < tEnd_; ++tokenIdx) {
            if (tokenStatusTensor(tokenIdx - tStart_) == 1) {
                continue;
            }
            if (!WaitDispatch(tokenIdx, copyCount)) {
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
    PipeBarrier<PIPE_ALL>(); // MaskCheck中包含reset操作，需确保前面操作完成
    BuffInit();
    RecvPhaseReduce();
}

} // namespace MoeEpCombineImpl

#endif // MOE_EP_COMBINE_H
