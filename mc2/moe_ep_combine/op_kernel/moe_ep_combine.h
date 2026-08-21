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

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if (ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)) && \
    (HCOMM_MAJOR > 9 || (HCOMM_MAJOR == 9 && HCOMM_MINOR > 0))
#define ENABLE_MOE_EP_COMBINE_KERNEL
#endif

#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "kernel_tiling/kernel_tiling.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/reduce.h"
#include "adv_api/reduce/sum.h"
#if __has_include("adv_api/hcomm/hcomm.h")
#include "adv_api/hcomm/hcomm.h"
#endif

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

#if defined(ENABLE_MOE_EP_COMBINE_KERNEL)

using namespace AscendC;

#define TemplateMoeEpCombineTypeClass typename XType, uint32_t HasTopkWeight
#define TemplateMoeEpCombineTypeFunc XType, HasTopkWeight
#define HCOMM_INIT_SIZE 512UL

static constexpr uint32_t WIN_ADDR_ALIGN = 512;
static constexpr uint32_t RECV_META_FIELDS = 4;
// PR 111 requires each committed batch to contain fewer WQEBBs than the SQ depth.
static constexpr uint32_t HCOMM_BATCH_CAPACITY = 127;
static constexpr uint32_t HCOMM_PLAIN_WRITE_WQE_BYTES = 64;
static constexpr uint32_t HCOMM_BATCH_BUFFER_BYTES = HCOMM_BATCH_CAPACITY * HCOMM_PLAIN_WRITE_WQE_BYTES;
constexpr uint64_t UB_ALIGN = 32UL;
constexpr uint32_t COMBINE_STATE_OFFSET = 64U * 1024U; // 本卡状态空间偏移地址，前面的地址给dispatch用
constexpr uint32_t STATE_OFFSET = 32U;
constexpr uint32_t DCCI_OFFSET = 64U;
constexpr uint64_t ALIGNED_LEN_256 = 256UL;
constexpr uint32_t FLOAT_PER_UB_ALIGN = 8U;
constexpr uint32_t SEND_DOUBLE_BUFFER_NUM = 2U;
static constexpr struct UrmaWqeEntry DEFAULT_WQE_CONFIG = {
    .odr = 5,
    .fence = 1,
    .se = 0,
    .cqe = 0,
    .inlineEn = 0};
static constexpr struct UrmaWqeEntry CHANNEL_FLAG_WQE_CONFIG = {
    .odr = 6,
    .fence = 1,
    .se = 0,
    .cqe = 0,
    .inlineEn = 0};
template <TemplateMoeEpCombineTypeClass>
class MoeEpCombine {
public:
    __aicore__ inline MoeEpCombine(){};

    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR recvSrcMetadata,
                                GM_ADDR numRecvPerExpert, GM_ADDR topkWeights, GM_ADDR combinedX,
                                GM_ADDR combinedTopkWeights, GM_ADDR workspace, GM_ADDR tilingGM, TPipe *pipe,
                                const MoeEpCombineInfo *tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
                                       uint32_t &endTokenId, uint32_t &tokenPerAivNum);
    __aicore__ inline void SendLocalToken(uint32_t tokenIndex, GM_ADDR dstAddr);
    __aicore__ inline void SendSlot(uint32_t tokenIndex, int32_t srcRank, int32_t srcTokenIdx,
                                    int32_t srcTopKIdx, uint32_t channelIndex, uint32_t weight);
    __aicore__ inline void SendChannelFlag(uint32_t dstRank, uint32_t channelIndex,
                                          uint32_t completionChannelCount);
    template <auto const& config>
    __aicore__ inline void PrepareWrite(
        uint64_t commHandle, GM_ADDR dstBase, GM_ADDR dst, GM_ADDR src, uint64_t len);
    __aicore__ inline void FlushPreparedWrites();
    __aicore__ inline bool WaitDispatch(uint32_t completionChannelCount);
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
    __aicore__ inline GM_ADDR GetLocalSendDataWorkspaceAddr(const int32_t rankId)
    {
        return combineSendDataWorkspaceAddr_ + sendDataWorkspaceSizePerRank_ * rankId;
    }
    __aicore__ inline uint32_t ReduceSumWorkNeedSize(int32_t count, int32_t typeSize)
    {
        int32_t elementsPerBlock = UB_ALIGN / typeSize;
        int32_t elementsPerRepeat = ALIGNED_LEN_256 / typeSize;
        int32_t iter1OutputCount = (count + elementsPerRepeat - 1) / elementsPerRepeat;
        uint32_t iter1AlignEnd = ((iter1OutputCount + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
        return iter1AlignEnd;
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
    uint32_t hAlignSize_{0}; // UB对齐后的hidden size
    uint64_t combineStateWinOffset_{0};
    uint64_t combineDataWinOffset_{0};

    uint32_t hWeightAlignSize_{0}; // token+weight对齐后的hidden size
    uint32_t XTypeAlign32Size_{0};
    uint32_t perSlotBytes_{0};
    uint64_t actualA_{0};
    uint32_t aivNum_{0};
    uint32_t localmoeNum_{0};
    uint64_t sendDataWorkspaceSizePerRank_{0};

    uint32_t tStart_{0};
    uint32_t tEnd_{0};
    uint32_t tPerCore_{0};
    uint32_t ubXBytes_{0};
    uint32_t mask_tokenNum_{0};
    uint32_t bsKCastCnt_{0};
    uint32_t activeMaskAlignSize_{0};

    GlobalTensor<XType> xGm_;
    GlobalTensor<int32_t> topkIdxGm_;
    GlobalTensor<int32_t> recvSrcMetadataGm_;
    GlobalTensor<int64_t> numRecvPerExpertGm_;
    GlobalTensor<float> topkWeightsGm_;

    GlobalTensor<XType> combinedXGm_;
    GlobalTensor<float> combinedTopkWeightsGm_;

    LocalTensor<XType> ubX_;
    LocalTensor<float> ubAccFp32_;
    LocalTensor<float> ubTmpFp32_;
    LocalTensor<float> ubWeighted_;
    LocalTensor<uint32_t> statusTensor_;
    LocalTensor<uint8_t> hcommTensor_;
    LocalTensor<uint8_t> hcommBatchTensor_;

    LocalTensor<half> maskGenerateTensor_;
    LocalTensor<bool> maskStrideTensor_;
    LocalTensor<half> tokenTargetTensor_;

    TBuf<QuePosition::VECIN> ubXBuf_;
    TBuf<QuePosition::VECIN> ubAccFp32Buf_;
    TBuf<QuePosition::VECIN> ubTmpFp32Buf_;
    TBuf<QuePosition::VECIN> ubWeightedBuf_;
    TBuf<> readStateBuf_;
    TBuf<> stateBuf_;
    TBuf<> stateSumBuf_;
    TBuf<> hcommBuf_;
    TBuf<TPosition::VECOUT> hcommBatchBuf_;

    TBuf<> compareBuf_;
    TBuf<> rowTmpFloatBuf_;
    TBuf<> tokenBuf_;
    TBuf<> tokenTargetTBuf_;
    TBuf<> metadataBuf_; // 发送阶段recvSrcMetadata的UB缓冲，批量搬运避免GetValue
    TBuf<> weightsBuf_;
    TBuf<> sendIndexBuf_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_; // 数据队列
    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;                 // 通信上下文
    using HcommBatchHandle = AscendC::BatchHandle<AscendC::ChannelHandle>;

    GM_ADDR winRankAddr_[Mc2Aclnn::HCCL_MAX_RANK_SIZE];
    uint64_t hcommHandle_[Mc2Aclnn::HCCL_MAX_RANK_SIZE];
    GM_ADDR combineSendDataWorkspaceAddr_{nullptr};
    GM_ADDR flagWorkspaceAddr_{nullptr};
    uint32_t aivId_{0};
    HcommBatchHandle activeBatchHandle_{};
    uint64_t activeBatchChannel_{0};
    uint32_t preparedWriteCount_{0};
};

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void
MoeEpCombine<TemplateMoeEpCombineTypeFunc>::Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR recvSrcMetadata,
                                                 GM_ADDR numRecvPerExpert, GM_ADDR topkWeights, GM_ADDR combinedX,
                                                 GM_ADDR combinedTopkWeights, GM_ADDR workspace, GM_ADDR tilingGM,
                                                 TPipe *pipe, const MoeEpCombineInfo *tilingData)
{
    tpipe_ = pipe;
    tilingData_ = tilingData;
    aivId_ = GetBlockIdx();
    combineSendDataWorkspaceAddr_ = workspace;
    epWorldSize_ = tilingData_->cfg.epWorldSize;
    numMaxTokensPerRank_ = tilingData_->cfg.numMaxTokensPerRank;
    numTokens_ = tilingData_->cfg.numTokens;
    topK_ = tilingData_->cfg.topK;
    axisH_ = tilingData_->cfg.hidden;
    perSlotBytes_ = tilingData_->cfg.perSlotBytes;
    aivNum_ = tilingData_->aivNum;
    localmoeNum_ = tilingData_->cfg.numLocalExperts;
    sendDataWorkspaceSizePerRank_ = tilingData->sendDataWorkspaceSizePerRank;
    flagWorkspaceAddr_ = combineSendDataWorkspaceAddr_ + epWorldSize_ * sendDataWorkspaceSizePerRank_ +
                         static_cast<uint64_t>(aivId_) * WIN_ADDR_ALIGN;
    hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN; // UB 32字节对齐
    hWeightAlignSize_ = hAlignSize_ + UB_ALIGN;                      // UB 32字节对齐
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);
    tpipe_->InitBuffer(hcommBatchBuf_, HCOMM_BATCH_BUFFER_BYTES);
    hcommBatchTensor_ = hcommBatchBuf_.Get<uint8_t>();
    Duplicate<uint8_t>(hcommBatchTensor_, 0U, HCOMM_BATCH_BUFFER_BYTES);
    SyncFunc<AscendC::HardEvent::V_S>();

    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::MoeCommContext *>(context);
    rankId_ = mc2Context_->epRankId;
    channelsPerRank_ = mc2Context_->channelsPerRank;
    if (channelsPerRank_ == 0 ||
        (epWorldSize_ > 0 && channelsPerRank_ > Mc2Aclnn::HCCL_MAX_RANK_SIZE / epWorldSize_)) {
        channelsPerRank_ = 1;
    }
    for (uint32_t i = 0; i < epWorldSize_; ++i) {
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
    }
    uint32_t handleCount = epWorldSize_ * channelsPerRank_;
    for (uint32_t i = 0; i < handleCount; ++i) {
        hcommHandle_[i] = mc2Context_->hcommHandle[i];
    }

    combineStateWinOffset_ = tilingData->combineStateWinOffset;
    combineDataWinOffset_ = tilingData->combineDataWinOffset;

    xGm_.SetGlobalBuffer((__gm__ XType *)x);
    topkIdxGm_.SetGlobalBuffer((__gm__ int32_t *)topkIdx);
    recvSrcMetadataGm_.SetGlobalBuffer((__gm__ int32_t *)recvSrcMetadata);
    numRecvPerExpertGm_.SetGlobalBuffer((__gm__ int64_t *)numRecvPerExpert);
    combinedXGm_.SetGlobalBuffer((__gm__ XType *)combinedX);

    // 计算actualA_的大小
    uint32_t numRecvBytes = Ceil(localmoeNum_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
    uint32_t reduceTmpBytes = ReduceSumWorkNeedSize(localmoeNum_, sizeof(int64_t)) * sizeof(int64_t);
    TBuf<TPosition::VECCALC> numRecvBuf;
    tpipe_->InitBuffer(numRecvBuf, numRecvBytes + reduceTmpBytes + UB_ALIGN); // 数据 + ReduceSum tmp + 输出
    LocalTensor<int64_t> numRecvLocal = numRecvBuf.Get<int64_t>();

    DataCopyExtParams numRecvCopyParams{1U, static_cast<uint32_t>(localmoeNum_ * sizeof(int64_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int64_t> numRecvPadParams{false, 0U, 0U, 0U};
    DataCopyPad(numRecvLocal, numRecvPerExpertGm_, numRecvCopyParams, numRecvPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();

    LocalTensor<int64_t> reduceTmp = numRecvLocal[numRecvBytes / sizeof(int64_t)];
    LocalTensor<int64_t> sumOut = reduceTmp[reduceTmpBytes / sizeof(int64_t)];
    ReduceSum<int64_t>(sumOut, numRecvLocal, reduceTmp, static_cast<int32_t>(localmoeNum_));
    SyncFunc<AscendC::HardEvent::V_S>();
    actualA_ = static_cast<uint64_t>(sumOut.GetValue(0));
    ubXBytes_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;
    XTypeAlign32Size_ = hAlignSize_;
    if constexpr (HasTopkWeight == 1) {
        XTypeAlign32Size_ = hWeightAlignSize_;
        topkWeightsGm_.SetGlobalBuffer((__gm__ float *)topkWeights);
        combinedTopkWeightsGm_.SetGlobalBuffer((__gm__ float *)combinedTopkWeights);
    }
    tpipe_->InitBuffer(xQueue_, SEND_DOUBLE_BUFFER_NUM, XTypeAlign32Size_);
    tpipe_->InitBuffer(readStateBuf_, UB_ALIGN); // 32
    tpipe_->InitBuffer(ubWeightedBuf_, UB_ALIGN);
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
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendLocalToken(uint32_t tokenIndex, GM_ADDR dstAddr)
{
    GlobalTensor<XType> outToken;
    outToken.SetGlobalBuffer((__gm__ XType *)dstAddr);

    DataCopyPadParams padParams = {false, 0, 0, 0};
    DataCopyParams copyParams = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    LocalTensor<XType> tokenTensor = xQueue_.AllocTensor<XType>();
    DataCopyPad(tokenTensor, xGm_[tokenIndex * axisH_], copyParams, padParams);
    xQueue_.EnQue(tokenTensor);
    tokenTensor = xQueue_.DeQue<XType>();
    DataCopyPad(outToken, tokenTensor, copyParams);
    xQueue_.FreeTensor<XType>(tokenTensor);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::FlushPreparedWrites()
{
    if (preparedWriteCount_ == 0) {
        return;
    }
    int32_t ret = hcomm_.BatchCommit(activeBatchHandle_);
    ascendc_assert(ret == HCOMM_SUCCESS, "BatchCommit failed, ret=%d", ret);
    activeBatchHandle_ = {};
    activeBatchChannel_ = 0;
    preparedWriteCount_ = 0;
}

template <TemplateMoeEpCombineTypeClass>
template <auto const& config>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::PrepareWrite(
    uint64_t commHandle, GM_ADDR dstBase, GM_ADDR dst, GM_ADDR src, uint64_t len)
{
    if (preparedWriteCount_ != 0 && activeBatchChannel_ != commHandle) {
        FlushPreparedWrites();
    }
    if (preparedWriteCount_ == HCOMM_BATCH_CAPACITY) {
        FlushPreparedWrites();
    }
    if (preparedWriteCount_ == 0) {
        activeBatchHandle_ = hcomm_.MakeBatchHandle(
            commHandle, hcommBatchTensor_, HCOMM_BATCH_BUFFER_BYTES, dstBase);
        activeBatchChannel_ = commHandle;
    }
    int32_t ret = hcomm_.WriteNbi<config>(activeBatchHandle_, dst, src, len);
    ascendc_assert(ret == HCOMM_SUCCESS, "batch WriteNbi failed, ret=%d", ret);
    ++preparedWriteCount_;
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendSlot(
    uint32_t tokenIndex, int32_t srcRank, int32_t srcTokenIdx, int32_t srcTopKIdx, uint32_t channelIndex, uint32_t weight)
{
    uint64_t sendTokenOffset = (static_cast<uint64_t>(srcTokenIdx) * topK_ + srcTopKIdx) * perSlotBytes_;
    uint64_t recvStateOffset = (srcTokenIdx * topK_ + srcTopKIdx) * WIN_ADDR_ALIGN;

    GM_ADDR remoteRankWinAddr = GetUrmaWinAddrByRankId(srcRank, combineDataWinOffset_);
    GM_ADDR remoteRankStateAddr = GetUrmaStateAddrByRankId(srcRank, combineStateWinOffset_);

    if (srcRank != static_cast<int32_t>(rankId_)) {
        uint64_t commHandle = GetCommHandle(srcRank, channelIndex);
        GM_ADDR tokenAddr = (GM_ADDR)xGm_.GetPhyAddr(tokenIndex * axisH_);
        PrepareWrite<DEFAULT_WQE_CONFIG>(
            commHandle, winRankAddr_[srcRank], remoteRankWinAddr + sendTokenOffset, tokenAddr,
            axisH_ * sizeof(XType));
        if constexpr (HasTopkWeight == 1) {
            GM_ADDR weightAddr = (GM_ADDR)topkWeightsGm_.GetPhyAddr(tokenIndex);
            PrepareWrite<DEFAULT_WQE_CONFIG>(
                commHandle, winRankAddr_[srcRank], remoteRankStateAddr + recvStateOffset, weightAddr,
                sizeof(uint32_t));
        }
    } else {
        SendLocalToken(tokenIndex, remoteRankWinAddr + sendTokenOffset);
        if constexpr (HasTopkWeight == 1) {
            SyncFunc<AscendC::HardEvent::MTE3_S>();
            statusTensor_(0) = weight;
            SyncFunc<AscendC::HardEvent::S_MTE3>();
            GlobalTensor<uint32_t> state;
            state.SetGlobalBuffer((__gm__ uint32_t *)(remoteRankStateAddr + recvStateOffset));
            DataCopyParams weightCopyParams = {1U, static_cast<uint16_t>(sizeof(uint32_t)), 0U, 0U};
            DataCopyPad(state, statusTensor_, weightCopyParams);
        }
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendChannelFlag(
    uint32_t dstRank, uint32_t channelIndex, uint32_t completionChannelCount)
{
    uint64_t flagOffset = (static_cast<uint64_t>(numMaxTokensPerRank_) * topK_ +
                           static_cast<uint64_t>(rankId_) * completionChannelCount + channelIndex) * WIN_ADDR_ALIGN;
    GM_ADDR flagAddr = GetUrmaStateAddrByRankId(dstRank, combineStateWinOffset_) + flagOffset;
    if (dstRank == rankId_) {
        GlobalTensor<uint64_t> localFlag;
        localFlag.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(flagAddr));
        DataCopy(localFlag, statusTensor_.ReinterpretCast<uint64_t>(), UB_ALIGN / sizeof(uint64_t));
        return;
    }
    PrepareWrite<CHANNEL_FLAG_WQE_CONFIG>(
        GetCommHandle(dstRank, channelIndex), winRankAddr_[dstRank], flagAddr, flagWorkspaceAddr_, WIN_ADDR_ALIGN);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::GetCoreAssignment(
    uint32_t totalBlocks, uint32_t &targetRank, uint32_t &coreIndexInGroup, uint32_t &groupSize)
{
    uint32_t baseGroupSize = totalBlocks / epWorldSize_;
    uint32_t remainder = totalBlocks % epWorldSize_;
    uint32_t maxChannelAivNum = epWorldSize_ * channelsPerRank_;
    // Keep channelsPerRank_ cores for every remote rank and give all excess AIVs to the local rank.
    bool assignRemainingToLocal = totalBlocks > maxChannelAivNum;
    uint32_t localGroupSize =
        assignRemainingToLocal ? totalBlocks - (epWorldSize_ - 1U) * channelsPerRank_ : 0U;
    uint32_t accumulated = 0;
    for (uint32_t rank = 0; rank < epWorldSize_; ++rank) {
        uint32_t currentGroupSize = assignRemainingToLocal
                                        ? (rank == rankId_ ? localGroupSize : channelsPerRank_)
                                        : baseGroupSize + ((rank < remainder) ? 1U : 0U);
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
    if (epWorldSize_ == 0 || aivNum_ == 0) {
        return;
    }
    uint32_t activeAivNum = aivNum_;
    uint32_t maxChannelAivNum = epWorldSize_ * channelsPerRank_;
    bool useRemainingAivForLocal = activeAivNum > maxChannelAivNum;

    activeBatchHandle_ = {};
    activeBatchChannel_ = 0;
    preparedWriteCount_ = 0;

    bool splitRankTokens = activeAivNum >= epWorldSize_;
    uint32_t targetRank = epWorldSize_;
    uint32_t coreIndexInGroup = 0;
    uint32_t groupSize = 1;
    if (splitRankTokens && aivId_ < activeAivNum) {
        GetCoreAssignment(activeAivNum, targetRank, coreIndexInGroup, groupSize);
    }
    uint32_t channelIndex = splitRankTokens ? coreIndexInGroup : 0;

    bool sendsTokens = actualA_ != 0 && aivId_ < activeAivNum;
    bool sendsLocalTokens = sendsTokens &&
                            (splitRankTokens ? targetRank == rankId_ : rankId_ % activeAivNum == aivId_);
    constexpr uint32_t metaBytesPerToken = RECV_META_FIELDS * sizeof(int32_t);
    constexpr uint32_t metaChunkTokenMax = 2048U;
    constexpr uint32_t indexStride = UB_ALIGN / sizeof(uint64_t);
    uint32_t metaChunkTokens = 0;
    uint32_t metaChunkBytes = 0;
    uint32_t sendIndexBytes = 0;
    LocalTensor<int32_t> metadataLocal;
    LocalTensor<uint64_t> sendIndexLocal;
    GlobalTensor<uint64_t> sendIndexGm;
    uint64_t slotsPerRank = static_cast<uint64_t>(numMaxTokensPerRank_) * topK_;
    uint64_t workspaceStride = sendDataWorkspaceSizePerRank_ / sizeof(uint64_t);
    if (sendsTokens) {
        metaChunkTokens = (actualA_ < metaChunkTokenMax) ? actualA_ : metaChunkTokenMax;
        metaChunkBytes = Ceil(metaChunkTokens * metaBytesPerToken, UB_ALIGN) * UB_ALIGN;
        sendIndexBytes = metaChunkTokens * UB_ALIGN;
        tpipe_->InitBuffer(metadataBuf_, metaChunkBytes);
        tpipe_->InitBuffer(sendIndexBuf_, sendIndexBytes);
        if constexpr (HasTopkWeight == 1) {
            uint32_t weightChunkBytes = Ceil(metaChunkTokens * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
            tpipe_->InitBuffer(weightsBuf_, weightChunkBytes);
        }
        metadataLocal = metadataBuf_.Get<int32_t>();
        sendIndexLocal = sendIndexBuf_.Get<uint64_t>();
        sendIndexGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(combineSendDataWorkspaceAddr_));

        uint32_t zeroIndexCount = metaChunkBytes / sizeof(uint64_t);
        uint32_t zeroSlotCount = zeroIndexCount / indexStride;
        LocalTensor<uint64_t> zeroIndexLocal = metadataLocal.ReinterpretCast<uint64_t>();
        Duplicate<uint64_t>(zeroIndexLocal, 0U, zeroIndexCount);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        uint64_t clearCount = slotsPerRank / activeAivNum;
        uint64_t clearRemainder = slotsPerRank % activeAivNum;
        uint64_t clearStart = aivId_ * clearCount + (aivId_ < clearRemainder ? aivId_ : clearRemainder);
        clearCount += aivId_ < clearRemainder ? 1U : 0U;
        for (uint32_t dstRank = 0; dstRank < epWorldSize_; ++dstRank) {
            for (uint64_t offset = 0; offset < clearCount; offset += zeroSlotCount) {
                uint32_t curClearCount = static_cast<uint32_t>(
                    clearCount - offset > zeroSlotCount ? zeroSlotCount : clearCount - offset);
                DataCopyExtParams clearParams{
                    1U, static_cast<uint32_t>(curClearCount * UB_ALIGN), 0U, 0U, 0U};
                uint64_t clearOffset = (clearStart + offset) * indexStride;
                DataCopyPad(sendIndexGm[static_cast<uint64_t>(dstRank) * workspaceStride + clearOffset],
                            zeroIndexLocal, clearParams);
            }
        }
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        SyncAll<true>();

        uint64_t rowsPerCore = actualA_ / activeAivNum;
        uint64_t rowRemainder = actualA_ % activeAivNum;
        uint64_t rowStart = aivId_ * rowsPerCore + (aivId_ < rowRemainder ? aivId_ : rowRemainder);
        uint64_t rowCount = rowsPerCore + (aivId_ < rowRemainder ? 1U : 0U);
        uint64_t rowEnd = rowStart + rowCount;
        const DataCopyPadExtParams<int32_t> metaPadParams{false, 0U, 0U, 0U};
        DataCopyExtParams indexCopyParams{1U, static_cast<uint32_t>(sizeof(uint64_t)), 0U, 0U, 0U};
        for (uint64_t chunkStart = rowStart; chunkStart < rowEnd; chunkStart += metaChunkTokens) {
            uint64_t chunkEnd = (chunkStart + metaChunkTokens > rowEnd) ? rowEnd : chunkStart + metaChunkTokens;
            uint32_t curChunkTokens = static_cast<uint32_t>(chunkEnd - chunkStart);
            LocalTensor<uint32_t> weightsForNotify;
            if constexpr (HasTopkWeight == 1) {
                weightsForNotify = weightsBuf_.Get<uint32_t>();
                DataCopyExtParams weightCopyParams{
                    1U, static_cast<uint32_t>(curChunkTokens * sizeof(uint32_t)), 0U, 0U, 0U};
                const DataCopyPadExtParams<uint32_t> weightPadParams{false, 0U, 0U, 0U};
                DataCopyPad(weightsForNotify, topkWeightsGm_[chunkStart].template ReinterpretCast<uint32_t>(),
                            weightCopyParams, weightPadParams);
            }
            DataCopyExtParams metaCopyParams{1U, curChunkTokens * metaBytesPerToken, 0U, 0U, 0U};
            DataCopyPad(
                metadataLocal, recvSrcMetadataGm_[chunkStart * RECV_META_FIELDS], metaCopyParams, metaPadParams);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            for (uint32_t i = 0; i < curChunkTokens; ++i) {
                int32_t srcRank = metadataLocal.GetValue(i * RECV_META_FIELDS + 0);
                int32_t srcTokenIdx = metadataLocal.GetValue(i * RECV_META_FIELDS + 1);
                int32_t srcTopKIdx = metadataLocal.GetValue(i * RECV_META_FIELDS + 2);
                bool valid = srcRank >= 0 && srcRank < static_cast<int32_t>(epWorldSize_) && srcTokenIdx >= 0 &&
                             srcTokenIdx < static_cast<int32_t>(numMaxTokensPerRank_) && srcTopKIdx >= 0 &&
                             srcTopKIdx < static_cast<int32_t>(topK_);
                uint64_t sendIndex = 0;
                if (valid) {
                    sendIndex = static_cast<uint32_t>(chunkStart + i) + 1U;
                    if constexpr (HasTopkWeight == 1) {
                        sendIndex |= static_cast<uint64_t>(weightsForNotify.GetValue(i)) << 32U;
                    }
                }
                sendIndexLocal.SetValue(i * indexStride, sendIndex);
            }
            SyncFunc<AscendC::HardEvent::S_MTE3>();
            for (uint32_t i = 0; i < curChunkTokens; ++i) {
                uint64_t sendIndex = sendIndexLocal.GetValue(i * indexStride);
                if (sendIndex == 0) {
                    continue;
                }
                int32_t srcRank = metadataLocal.GetValue(i * RECV_META_FIELDS + 0);
                uint64_t dstSlot = static_cast<uint64_t>(metadataLocal.GetValue(i * RECV_META_FIELDS + 1)) * topK_ +
                                   metadataLocal.GetValue(i * RECV_META_FIELDS + 2);
                uint64_t indexOffset = dstSlot * indexStride;
                DataCopyPad(sendIndexGm[static_cast<uint64_t>(srcRank) * workspaceStride + indexOffset],
                            sendIndexLocal[i * indexStride], indexCopyParams);
            }
            SyncFunc<AscendC::HardEvent::MTE3_S>();
        }
        SyncAll<true>();
    }
    if (sendsTokens) {
        const DataCopyPadExtParams<uint64_t> indexPadParams{false, 0U, 0U, 0U};
        uint32_t firstRank = splitRankTokens ? targetRank : aivId_;
        uint32_t rankStep = splitRankTokens ? epWorldSize_ : activeAivNum;
        uint32_t indexChunkMax = sendIndexBytes / UB_ALIGN;
        for (uint32_t dstRank = firstRank; dstRank < epWorldSize_; dstRank += rankStep) {
            uint64_t slotStart = 0;
            uint64_t slotEnd = slotsPerRank;
            if (splitRankTokens && groupSize > 1U) {
                uint64_t slotsPerCore = slotsPerRank / groupSize;
                uint64_t slotRemainder = slotsPerRank % groupSize;
                slotStart = coreIndexInGroup * slotsPerCore +
                            (coreIndexInGroup < slotRemainder ? coreIndexInGroup : slotRemainder);
                slotEnd = slotStart + slotsPerCore + (coreIndexInGroup < slotRemainder ? 1U : 0U);
            }
            for (uint64_t chunkStart = slotStart; chunkStart < slotEnd; chunkStart += indexChunkMax) {
                uint64_t chunkEnd = (chunkStart + indexChunkMax > slotEnd) ? slotEnd : chunkStart + indexChunkMax;
                uint32_t curIndexCount = static_cast<uint32_t>(chunkEnd - chunkStart);
                DataCopyExtParams mapCopyParams{
                    1U, static_cast<uint32_t>(curIndexCount * UB_ALIGN), 0U, 0U, 0U};
                DataCopyPad(sendIndexLocal,
                            sendIndexGm[static_cast<uint64_t>(dstRank) * workspaceStride + chunkStart * indexStride],
                            mapCopyParams, indexPadParams);
                SyncFunc<AscendC::HardEvent::MTE2_S>();
                for (uint32_t i = 0; i < curIndexCount; ++i) {
                    uint64_t sendIndex = sendIndexLocal.GetValue(i * indexStride);
                    uint32_t tokenIndexPlusOne = static_cast<uint32_t>(sendIndex);
                    if (tokenIndexPlusOne == 0U) {
                        continue;
                    }
                    uint64_t dstSlot = chunkStart + i;
                    uint32_t srcTokenIdx = static_cast<uint32_t>(dstSlot / topK_);
                    uint32_t srcTopKIdx = static_cast<uint32_t>(dstSlot % topK_);
                    uint32_t weight = static_cast<uint32_t>(sendIndex >> 32U);
                    SendSlot(tokenIndexPlusOne - 1U, dstRank, srcTokenIdx, srcTopKIdx, channelIndex, weight);
                }
                SyncFunc<AscendC::HardEvent::S_MTE2>();
            }
        }
    }

    // Append one ODR 6 completion write after all payload writes on this channel.
    // Channels without payload also publish completion so the receiver cannot wait forever.
    if (sendsLocalTokens) {
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
    if (useRemainingAivForLocal) {
        // Extra local AIVs do not publish additional flags. Wait for their local MTE3 writes before
        // the configured channel AIVs publish the original number of completion flags.
        SyncAll<true>();
    }
    uint32_t completionChannelCount = useRemainingAivForLocal ? channelsPerRank_ : groupSize;
    bool publishesChannelFlag = aivId_ < activeAivNum &&
                                (!splitRankTokens || coreIndexInGroup < completionChannelCount);
    if (publishesChannelFlag) {
        LocalTensor<uint64_t> flagTensor = statusTensor_.ReinterpretCast<uint64_t>();
        Duplicate<uint64_t>(flagTensor, 1U, UB_ALIGN / sizeof(uint64_t));
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        GlobalTensor<uint64_t> flagWorkspace;
        flagWorkspace.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(flagWorkspaceAddr_));
        DataCopy(flagWorkspace, flagTensor, UB_ALIGN / sizeof(uint64_t));
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        if (splitRankTokens) {
            SendChannelFlag(targetRank, channelIndex, completionChannelCount);
        } else {
            for (uint32_t dstRank = aivId_; dstRank < epWorldSize_; dstRank += activeAivNum) {
                SendChannelFlag(dstRank, 0U, 1U);
            }
        }
    }
    FlushPreparedWrites();
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
        tpipe_->InitBuffer(ubWeightedBuf_, UB_ALIGN);
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
__aicore__ inline bool MoeEpCombine<TemplateMoeEpCombineTypeFunc>::WaitDispatch(
    uint32_t completionChannelCount)
{
    uint32_t completionFlagCount = epWorldSize_ * completionChannelCount;
    uint64_t flagOffset = static_cast<uint64_t>(numMaxTokensPerRank_) * topK_ * WIN_ADDR_ALIGN;
    GM_ADDR stateGM = GetUrmaStateAddrByRankId(rankId_, combineStateWinOffset_) + flagOffset;
    GlobalTensor<uint32_t> stateGMTensor;
    stateGMTensor.SetGlobalBuffer((__gm__ uint32_t *)stateGM);

    LocalTensor<uint32_t> stateTensor = stateBuf_.Get<uint32_t>();
    SyncFunc<AscendC::HardEvent::S_MTE2>();
    DataCopyExtParams params = {
        static_cast<uint16_t>(completionFlagCount), sizeof(uint32_t), WIN_ADDR_ALIGN - sizeof(uint32_t), 0, 0};
    DataCopyPadExtParams<uint32_t> padParams = {true, 0, 0, 0};

    DataCopyPad<uint32_t>(stateTensor, stateGMTensor, params, padParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();

    LocalTensor<uint32_t> stateSumTensor = stateSumBuf_.Get<uint32_t>();
    uint32_t shape[] = {completionFlagCount, UB_ALIGN / sizeof(uint32_t)};
    ReduceSum<uint32_t, AscendC::Pattern::Reduce::RA, false>(stateSumTensor, stateTensor, shape, true);
    SyncFunc<AscendC::HardEvent::V_S>();
    return stateSumTensor(0) == completionFlagCount;
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::ProcessTopKToken(uint32_t tokenIndex)
{
    Duplicate<float>(ubAccFp32_, (float)0, axisH_);
    DataCopyPadParams padParams = {false, 0, 0, 0};
    DataCopyParams xCopyParams = {1U, static_cast<uint16_t>(hAlignSize_), 0U, 0U};
    DataCopyParams weightCopyParams = {1U, static_cast<uint16_t>(sizeof(float)), 0U, 0U};
    for (uint32_t topkId = 0U; topkId < topK_; topkId++) {
        if (topkId != 0U) {
            SyncFunc<AscendC::HardEvent::V_MTE2>();
        }
        // 读取expert_id
        uint64_t slotOffset = (static_cast<uint64_t>(tokenIndex) * topK_ + topkId) * perSlotBytes_;
        GM_ADDR wAddr = GetUrmaWinAddrByRankId(rankId_, combineDataWinOffset_) + slotOffset;
        GlobalTensor<XType> srcTokenTensor;
        srcTokenTensor.SetGlobalBuffer(reinterpret_cast<__gm__ XType *>(wAddr));
        DataCopyPad(ubX_, srcTokenTensor, xCopyParams, padParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Cast(ubTmpFp32_, ubX_, AscendC::RoundMode::CAST_NONE, axisH_);
        Add(ubAccFp32_, ubAccFp32_, ubTmpFp32_, axisH_);
        if constexpr (HasTopkWeight == 1) {
            GM_ADDR weightAddr = GetUrmaStateAddrByRankId(rankId_, combineStateWinOffset_) +
                                 (tokenIndex * topK_ + topkId) * WIN_ADDR_ALIGN;
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
    uint32_t activeAivNum = aivNum_;
    uint32_t maxChannelAivNum = epWorldSize_ * channelsPerRank_;
    uint32_t completionChannelCount = 1U;
    if (activeAivNum > maxChannelAivNum) {
        completionChannelCount = channelsPerRank_;
    } else if (activeAivNum >= epWorldSize_) {
        completionChannelCount =
            activeAivNum / epWorldSize_ + (rankId_ < activeAivNum % epWorldSize_ ? 1U : 0U);
    }
    uint32_t completionFlagCount = epWorldSize_ * completionChannelCount;
    uint32_t flagBufferBytes = completionFlagCount * STATE_OFFSET;
    tpipe_->InitBuffer(stateBuf_, flagBufferBytes);
    tpipe_->InitBuffer(stateSumBuf_, UB_ALIGN);
    if (aivId_ == 0U) {
        while (!WaitDispatch(completionChannelCount)) {
        }
    }
    SyncAll<true>();
    if (aivId_ == 0U) {
        LocalTensor<uint32_t> stateTensor = stateBuf_.Get<uint32_t>();
        SyncFunc<AscendC::HardEvent::S_V>();
        Duplicate<uint32_t>(stateTensor, static_cast<uint32_t>(0), flagBufferBytes / sizeof(uint32_t));
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        uint64_t flagOffset = static_cast<uint64_t>(numMaxTokensPerRank_) * topK_ * WIN_ADDR_ALIGN;
        GM_ADDR stateGM = GetUrmaStateAddrByRankId(rankId_, combineStateWinOffset_) + flagOffset;
        GlobalTensor<uint32_t> stateGMTensor;
        stateGMTensor.SetGlobalBuffer((__gm__ uint32_t *)stateGM);
        DataCopyExtParams clearParams = {
            static_cast<uint16_t>(completionFlagCount), STATE_OFFSET, 0, WIN_ADDR_ALIGN - STATE_OFFSET, 0};
        DataCopyPad<uint32_t>(stateGMTensor, stateTensor, clearParams);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
    SyncAll<true>();
    for (uint32_t tokenIdx = tStart_; tokenIdx < tEnd_; ++tokenIdx) {
        ProcessTopKToken(tokenIdx);
        LocalTensor<XType> ubResultBf16 = ubX_;
        Cast(ubResultBf16, ubAccFp32_, RoundMode::CAST_RINT, axisH_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyPad(combinedXGm_[tokenIdx * axisH_], ubResultBf16, xCopyParams);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
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

#endif

} // namespace MoeEpCombineImpl

#endif // MOE_EP_COMBINE_H
