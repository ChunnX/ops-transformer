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
 * \file ffn_to_attention_urma.h
 * \brief FFNToAttentionV2 URMA implementation.
 */
#ifndef FFN_TO_ATTENTION_URMA_H
#define FFN_TO_ATTENTION_URMA_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "adv_api/hcomm/hcomm.h"
#include "ffn_to_attention_v2_tiling.h"

#if __has_include("../common/moe_distribute_base.h")
#include "../common/moe_distribute_base.h"
#include "../common/attention_ffn_context.h"
#include "../common/mc2_kernel_utils.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/attention_ffn_context.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#endif

namespace FFNToAttentionImpl {

constexpr uint8_t URMA_BUFFER_NUM = 2U;
constexpr uint32_t URMA_UB_ALIGN = 32U;
constexpr uint64_t URMA_WIN_ADDR_ALIGN = 512UL;

constexpr uint32_t URMA_HCOMM_INIT_SIZE = 512U;
constexpr uint32_t URMA_FLAG_SLOT_SIZE = 32U;
constexpr int32_t URMA_FLAG_VALUE = 1;

constexpr uint32_t URMA_WQE_SIZE = 64U;
constexpr uint32_t URMA_BATCH_WQE_CAPACITY = 256U;
constexpr uint32_t URMA_BATCH_BUFFER_SIZE = URMA_BATCH_WQE_CAPACITY * URMA_WQE_SIZE;
constexpr uint32_t HCOMM_SQ_MAX_PENDING = 32767U;
constexpr uint32_t METADATA_CHUNK_TOKEN_NUM = 256U;
constexpr uint32_t METADATA_FIELD_NUM = 4U;
constexpr uint32_t METADATA_TOKEN_BATCH_OFFSET_FIELD_INDEX = 2U;
constexpr uint32_t METADATA_TOKEN_TOPK_OFFSET_FIELD_INDEX = 3U;
constexpr uint32_t METADATA_CHUNK_BUFFER_SIZE = METADATA_CHUNK_TOKEN_NUM * METADATA_FIELD_NUM * sizeof(int32_t);
constexpr uint32_t REMOTE_RELAY_INFO_FIELD_NUM = 4U;
constexpr uint32_t REMOTE_FLAG_OFFSET_FIELD_INDEX = 1U;
constexpr uint32_t SOURCE_TOKEN_INDEX_FIELD_INDEX = 2U;
constexpr uint32_t NEXT_TOKEN_INDEX_FIELD_INDEX = 3U;
constexpr uint32_t REMOTE_RELAY_INFO_SIZE = REMOTE_RELAY_INFO_FIELD_NUM * sizeof(uint64_t);
constexpr uint64_t INVALID_TOKEN_INDEX = 0xFFFFFFFFFFFFFFFFULL;
constexpr uint32_t INVALID_TOKEN_INDEX_LOW32 = 0xFFFFFFFFU;
constexpr uint32_t REMOTE_RELAY_LIST_NUM = 2U;
constexpr uint32_t UINT32_PER_UINT64 = sizeof(uint64_t) / sizeof(uint32_t);
constexpr uint32_t URMA_BLOCK_UINT32_NUM = URMA_UB_ALIGN / sizeof(uint32_t);
static constexpr AscendC::UrmaWqeEntry REMOTE_WQE_CONFIG = {.odr = 5, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
static constexpr AscendC::UrmaWqeEntry REMOTE_CQE_WQE_CONFIG = {.odr = 5, .fence = 1, .se = 0, .cqe = 1, .inlineEn = 0};
static constexpr AscendC::UrmaWqeEntry REMOTE_ORDERED_FLAG_WQE_CONFIG = {
    .odr = 6, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
static constexpr AscendC::UrmaWqeEntry REMOTE_ORDERED_FLAG_CQE_WQE_CONFIG = {
    .odr = 6, .fence = 1, .se = 0, .cqe = 1, .inlineEn = 0};

#define TemplateFFNToAttentionUrmaTypeClass typename xType, bool isInputRankTable
#define TemplateFFNToAttentionUrmaTypeFunc xType, isInputRankTable

using namespace AscendC;

using HcommChannelHandle = AscendC::ChannelHandle;
using HcommBatchHandle = AscendC::BatchHandle<HcommChannelHandle>;

struct RemoteRelayInfo {
    uint64_t remoteDataOffset;
    uint64_t remoteFlagOffset;
    uint64_t sourceTokenIndex;
    uint64_t nextTokenIndex;
};

template <TemplateFFNToAttentionUrmaTypeClass>
class FFNToAttentionUrma {
public:
    __aicore__ inline FFNToAttentionUrma(){};
    __aicore__ inline void Init(GM_ADDR mc2Context, GM_ADDR x, GM_ADDR sessionIds, GM_ADDR microBatchIds,
                                GM_ADDR tokenIds, GM_ADDR expertOffsets, GM_ADDR actualTokenNum, GM_ADDR attnRankTable,
                                GM_ADDR workspaceGM, TPipe *pipe, const FFNToAttentionV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void WriteLocalUint64(const LocalTensor<uint64_t> &tensor, uint32_t index, uint64_t value)
    {
        // Duplicate operates on 32-bit lanes; preserve both halves of the 64-bit value.
        constexpr uint32_t valuesPerBlock = URMA_UB_ALIGN / sizeof(uint64_t);
        const uint32_t blockStart = index / valuesPerBlock * valuesPerBlock;
        const uint32_t lane = (index % valuesPerBlock) * UINT32_PER_UINT64;
        auto block = tensor[blockStart].template ReinterpretCast<uint32_t>();
        uint64_t lowMask[2] = {1ULL << lane, 0ULL};
        uint64_t highMask[2] = {1ULL << (lane + 1U), 0ULL};
        // Synchronization belongs to the caller so several fields can be written as one group.
        PipeBarrier<PIPE_V>();
        Duplicate<uint32_t>(block, static_cast<uint32_t>(value), lowMask, 1, 1, URMA_BLOCK_UINT32_NUM);
        PipeBarrier<PIPE_V>();
        Duplicate<uint32_t>(block, static_cast<uint32_t>(value >> 32U), highMask, 1, 1, URMA_BLOCK_UINT32_NUM);
    }

    __aicore__ inline void PrefetchTokenMetaData(uint64_t tokenStart, uint32_t tokenNum);
    __aicore__ inline void ReadTokenMetaDataFromLocal(ReadTokenMetaDataStruct &metaDataStruct, uint32_t localOffset);
    __aicore__ inline GM_ADDR GetWindowAddr(uint32_t curAttenWorkRank);
    __aicore__ inline uint64_t GetUrmaCommHandle(uint32_t dstRank);
    __aicore__ inline void HcommInit();
    __aicore__ inline void InitLocalFlag();
    __aicore__ inline void SendToLocal(GM_ADDR remoteDataAddr, GM_ADDR remoteFlagAddr, uint64_t yOffset,
                                       const DataCopyExtParams &xCopyParams);
    __aicore__ inline void AppendRemoteDataWrite(HcommBatchHandle &batchHandle, GM_ADDR remoteWindowAddr,
                                                 const RemoteRelayInfo &relayInfo, uint32_t dstRank, bool enableCqe);
    __aicore__ inline void AppendRemoteFlagWrite(HcommBatchHandle &batchHandle, GM_ADDR remoteWindowAddr,
                                                 const RemoteRelayInfo &relayInfo, uint32_t dstRank, bool firstFlag,
                                                 bool enableCqe);
    __aicore__ inline void BuildRemoteRelayLists();
    __aicore__ inline void StoreRemoteRelayInfo(uint64_t tokenIndex, const RemoteRelayInfo &relayInfo);
    __aicore__ inline void LoadRemoteRelayInfo(uint64_t tokenIndex, RemoteRelayInfo &relayInfo);
    __aicore__ inline void StoreNextTokenIndex(uint64_t tokenIndex, uint64_t nextTokenIndex);
    __aicore__ inline void CommitAndDrainChannel(HcommBatchHandle &batchHandle, HcommChannelHandle channel,
                                                 uint32_t &preparedWqeCount, uint32_t &sqWriteCount);

    TPipe *tpipe_{nullptr};
    Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
    __gm__ Mc2Aclnn::AttentionFFNContext *mc2Context_{nullptr};

    GM_ADDR inputDataAddr_{nullptr};
    GM_ADDR localFlagAddr_{nullptr};

    GlobalTensor<xType> xGMTensor_;
    GlobalTensor<int32_t> sessionIdsGMTensor_;
    GlobalTensor<int32_t> microBatchIdsGMTensor_;
    GlobalTensor<int32_t> tokenIdsGMTensor_;
    GlobalTensor<int32_t> expertOffsetsGMTensor_;
    GlobalTensor<int64_t> actualTokenNumGMTensor_;
    GlobalTensor<int32_t> attnRankTableGMTensor_;
    GlobalTensor<uint64_t> remoteRelayInfoGMTensor_;
    GlobalTensor<uint64_t> remoteRelayListHeadGMTensor_;
    GlobalTensor<uint64_t> remoteRelayListTailGMTensor_;
    LocalTensor<xType> xTmpTensor_;
    LocalTensor<int32_t> statusTensor_;
    LocalTensor<uint8_t> hcommTensor_;
    LocalTensor<uint8_t> hcommBatchTensor_;
    LocalTensor<int32_t> metadataTensor_;
    LocalTensor<uint64_t> transferTensor_;
    LocalTensor<uint64_t> relayListHeadTensor_;
    LocalTensor<uint64_t> relayListTailTensor_;

    TBuf<> statusBuf_;
    TBuf<> hcommBuf_;
    TBuf<> metadataBuf_;
    TBuf<> transferBuf_;
    TBuf<> relayListBuf_;
    TBuf<TPosition::VECOUT> hcommBatchBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_;

    uint32_t aivId_{0};
    uint32_t rankId_{0};
    uint64_t actualTokenNum_{0};
    uint64_t maxTokenNum_{0};
    uint32_t axisH_{0};
    uint32_t axisHS_{0};
    uint32_t axisA_{0};
    uint32_t microBatchNum_{0};
    uint32_t axisBS_{0};
    uint32_t expertNumPerToken_{0};
    uint32_t aivNum_{0};
    uint32_t worldSize_{0};
    uint64_t batchSizeSendCnt_{0};
    uint64_t hSize_{0};
    uint64_t winTokenInfoTableSize_{0};
    bool localWritesMayAliasInput_{false};
};

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::Init(
    GM_ADDR mc2Context, GM_ADDR x, GM_ADDR sessionIds, GM_ADDR microBatchIds, GM_ADDR tokenIds, GM_ADDR expertOffsets,
    GM_ADDR actualTokenNum, GM_ADDR attnRankTable, GM_ADDR workspaceGM, TPipe *pipe,
    const FFNToAttentionV2TilingData *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::AttentionFFNContext *>(mc2Context);
    rankId_ = mc2Context_->epRankId;

    axisBS_ = tilingData->ffnToAttentionV2Info.BS;
    axisH_ = tilingData->ffnToAttentionV2Info.H;
    axisHS_ = tilingData->ffnToAttentionV2Info.HS;
    axisA_ = tilingData->ffnToAttentionV2Info.A;
    microBatchNum_ = tilingData->ffnToAttentionV2Info.microBatchNum;
    expertNumPerToken_ = tilingData->ffnToAttentionV2Info.expertNumPerToken;
    aivNum_ = tilingData->ffnToAttentionV2Info.aivNum;
    worldSize_ = tilingData->ffnToAttentionV2Info.worldSize;
    maxTokenNum_ = tilingData->ffnToAttentionV2Info.maxTokenNum;

    inputDataAddr_ = x;
    xGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(x));
    sessionIdsGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sessionIds));
    microBatchIdsGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(microBatchIds));
    tokenIdsGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tokenIds));
    expertOffsetsGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(expertOffsets));
    actualTokenNumGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(actualTokenNum));
    if constexpr (isInputRankTable) {
        attnRankTableGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(attnRankTable));
    }

    tpipe_->InitBuffer(transferBuf_, URMA_UB_ALIGN);
    transferTensor_ = transferBuf_.Get<uint64_t>();
    actualTokenNum_ = static_cast<uint64_t>(actualTokenNumGMTensor_.GetValue(0UL));
    hSize_ = static_cast<uint64_t>(axisH_) * sizeof(xType);
    batchSizeSendCnt_ = static_cast<uint64_t>(axisBS_) * expertNumPerToken_;
    const uint64_t tokenSlotNum = static_cast<uint64_t>(microBatchNum_) * batchSizeSendCnt_;
    winTokenInfoTableSize_ = Ceil(tokenSlotNum * sizeof(int32_t), URMA_WIN_ADDR_ALIGN) * URMA_WIN_ADDR_ALIGN;
    const uint64_t tokenWindowEnd = winTokenInfoTableSize_ + tokenSlotNum * axisHS_ * sizeof(xType);
    const uint64_t relayWindowOffset =
        Ceil(tokenWindowEnd, static_cast<uint64_t>(URMA_FLAG_SLOT_SIZE)) * URMA_FLAG_SLOT_SIZE;
    // Flag sources and relay metadata live in this rank's window tail, outside peer-visible token slots.
    GM_ADDR localScratchAddr = GetWindowAddr(rankId_) + relayWindowOffset;
    localFlagAddr_ = localScratchAddr + static_cast<uint64_t>(aivId_) * URMA_FLAG_SLOT_SIZE;
    GM_ADDR remoteRelayInfoAddr = localScratchAddr + static_cast<uint64_t>(aivNum_) * URMA_FLAG_SLOT_SIZE;
    remoteRelayInfoGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(remoteRelayInfoAddr));
    GM_ADDR remoteRelayListHeadAddr = remoteRelayInfoAddr + maxTokenNum_ * REMOTE_RELAY_INFO_SIZE;
    remoteRelayListHeadGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(remoteRelayListHeadAddr));
    GM_ADDR remoteRelayListTailAddr = remoteRelayListHeadAddr + static_cast<uint64_t>(worldSize_) * sizeof(uint64_t);
    remoteRelayListTailGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(remoteRelayListTailAddr));
    // Validate that the context buffer (whose size is claimed via cclBufferSize attribute,
    // passed through tilingData->urmaWorkspaceOffset) is large enough for the current shape.
    // This catches context-reuse scenarios where the buffer was allocated for a smaller shape.
    const uint64_t cclBufferSize = tilingData->ffnToAttentionV2Info.urmaWorkspaceOffset;
    const uint64_t flagWindowSize = static_cast<uint64_t>(aivNum_) * URMA_FLAG_SLOT_SIZE;
    const uint64_t relayInfoWindowSize = maxTokenNum_ * REMOTE_RELAY_INFO_SIZE;
    const uint64_t relayListWindowSize = static_cast<uint64_t>(worldSize_) * sizeof(uint64_t) * REMOTE_RELAY_LIST_NUM;
    const uint64_t scratchWindowSize = flagWindowSize + relayInfoWindowSize + relayListWindowSize;
    const uint64_t requiredWindowSize = relayWindowOffset + scratchWindowSize;
    ascendc_assert(cclBufferSize >= requiredWindowSize,
                   "cclBufferSize (%llu) < requiredWindowSize (%llu), "
                   "context buffer may have been allocated for a different shape",
                   static_cast<unsigned long long>(cclBufferSize), static_cast<unsigned long long>(requiredWindowSize));
    // x may be a view of registered memory, but its payload must not alias the scratch tail.
    const uint64_t inputAddr = reinterpret_cast<uint64_t>(inputDataAddr_);
    const uint64_t scratchBegin = reinterpret_cast<uint64_t>(localScratchAddr);
    const uint64_t scratchEnd =
        reinterpret_cast<uint64_t>(remoteRelayListTailAddr) + static_cast<uint64_t>(worldSize_) * sizeof(uint64_t);
    ascendc_assert(inputAddr >= scratchEnd || inputAddr + maxTokenNum_ * hSize_ <= scratchBegin,
                   "x overlaps local flag/relay window tail");
    const uint64_t localWindowBegin = reinterpret_cast<uint64_t>(GetWindowAddr(rankId_));
    localWritesMayAliasInput_ =
        inputAddr < localWindowBegin + tokenWindowEnd && inputAddr + maxTokenNum_ * hSize_ > localWindowBegin;

    tpipe_->InitBuffer(xQueue_, URMA_BUFFER_NUM, hSize_);
    tpipe_->InitBuffer(statusBuf_, URMA_UB_ALIGN);
    statusTensor_ = statusBuf_.Get<int32_t>();
    tpipe_->InitBuffer(metadataBuf_, METADATA_CHUNK_BUFFER_SIZE);
    metadataTensor_ = metadataBuf_.Get<int32_t>();
    if (aivId_ == 0U) {
        const uint32_t listBytes =
            Ceil(worldSize_ * static_cast<uint32_t>(sizeof(uint64_t)), URMA_UB_ALIGN) * URMA_UB_ALIGN;
        tpipe_->InitBuffer(relayListBuf_, REMOTE_RELAY_LIST_NUM * listBytes);
        relayListHeadTensor_ = relayListBuf_.Get<uint64_t>();
        relayListTailTensor_ = relayListHeadTensor_[listBytes / sizeof(uint64_t)];
    }
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::HcommInit()
{
    tpipe_->InitBuffer(hcommBuf_, URMA_HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, URMA_HCOMM_INIT_SIZE);

    tpipe_->InitBuffer(hcommBatchBuf_, URMA_BATCH_BUFFER_SIZE);
    hcommBatchTensor_ = hcommBatchBuf_.Get<uint8_t>();
    Duplicate<uint8_t>(hcommBatchTensor_, 0U, URMA_BATCH_BUFFER_SIZE);

    SyncFunc<HardEvent::V_S>();
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::InitLocalFlag()
{
    constexpr uint32_t flagSlotElementNum = URMA_FLAG_SLOT_SIZE / sizeof(int32_t);
    Duplicate<int32_t>(statusTensor_, URMA_FLAG_VALUE, flagSlotElementNum);
    SyncFunc<HardEvent::V_MTE3>();
    GlobalTensor<int32_t> localFlagGMTensor;
    localFlagGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(localFlagAddr_));
    DataCopy(localFlagGMTensor, statusTensor_, flagSlotElementNum);
    PipeBarrier<PIPE_MTE3>();
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline GM_ADDR FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::GetWindowAddr(
    uint32_t curAttenWorkRank)
{
    return (GM_ADDR)mc2Context_->epHcclBuffer_[curAttenWorkRank];
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline uint64_t FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::GetUrmaCommHandle(uint32_t dstRank)
{
    uint32_t index = dstRank > rankId_ ? dstRank - 1U : dstRank;
    return mc2Context_->hcommHandle_[index];
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::PrefetchTokenMetaData(
    uint64_t tokenStart, uint32_t tokenNum)
{
    SyncFunc<HardEvent::S_MTE2>(); // Finish scalar reads before reusing the chunk buffer.
    DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(tokenNum * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> padParams{false, 0U, 0U, 0U};
    DataCopyPad(metadataTensor_, sessionIdsGMTensor_[tokenStart], copyParams, padParams);
    DataCopyPad(metadataTensor_[METADATA_CHUNK_TOKEN_NUM], microBatchIdsGMTensor_[tokenStart], copyParams, padParams);
    DataCopyPad(metadataTensor_[METADATA_TOKEN_BATCH_OFFSET_FIELD_INDEX * METADATA_CHUNK_TOKEN_NUM],
                tokenIdsGMTensor_[tokenStart], copyParams, padParams);
    DataCopyPad(metadataTensor_[METADATA_TOKEN_TOPK_OFFSET_FIELD_INDEX * METADATA_CHUNK_TOKEN_NUM],
                expertOffsetsGMTensor_[tokenStart], copyParams, padParams);
    SyncFunc<HardEvent::MTE2_S>();
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::ReadTokenMetaDataFromLocal(
    ReadTokenMetaDataStruct &metaDataStruct, uint32_t localOffset)
{
    int32_t curAttenWorkId = metadataTensor_.GetValue(localOffset);
    metaDataStruct.curAttenWorkIds = static_cast<uint32_t>(curAttenWorkId);
    metaDataStruct.curAttenWorkRank = metaDataStruct.curAttenWorkIds;

    if constexpr (isInputRankTable) {
        int32_t curAttenWorkRank = attnRankTableGMTensor_.GetValue(metaDataStruct.curAttenWorkIds);
        metaDataStruct.curAttenWorkRank = static_cast<uint32_t>(curAttenWorkRank);
    }

    int32_t curMicroBatchId = metadataTensor_.GetValue(METADATA_CHUNK_TOKEN_NUM + localOffset);
    metaDataStruct.curMicroBatchIds = static_cast<uint32_t>(curMicroBatchId);

    int32_t curTokenBatchOffset =
        metadataTensor_.GetValue(METADATA_TOKEN_BATCH_OFFSET_FIELD_INDEX * METADATA_CHUNK_TOKEN_NUM + localOffset);
    metaDataStruct.curTokenBatchOffset = static_cast<uint32_t>(curTokenBatchOffset);

    int32_t curTokenTopkOffset =
        metadataTensor_.GetValue(METADATA_TOKEN_TOPK_OFFSET_FIELD_INDEX * METADATA_CHUNK_TOKEN_NUM + localOffset);
    metaDataStruct.curTokenTopkOffset = static_cast<uint32_t>(curTokenTopkOffset);
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::SendToLocal(
    GM_ADDR remoteDataAddr, GM_ADDR remoteFlagAddr, uint64_t yOffset, const DataCopyExtParams &xCopyParams)
{
    DataCopyPadExtParams<xType> copyPadExtParams{false, 0U, 0U, 0U};
    xTmpTensor_ = xQueue_.AllocTensor<xType>();
    DataCopyPad(xTmpTensor_, xGMTensor_[static_cast<uint64_t>(yOffset) * axisH_], xCopyParams, copyPadExtParams);
    xQueue_.EnQue(xTmpTensor_);
    xTmpTensor_ = xQueue_.DeQue<xType>();

    GlobalTensor<xType> tokenDataGMTensor;
    tokenDataGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(remoteDataAddr));
    DataCopyPad(tokenDataGMTensor, xTmpTensor_, xCopyParams);
    xQueue_.FreeTensor<xType>(xTmpTensor_);
    PipeBarrier<PIPE_MTE3>();

    GlobalTensor<int32_t> tokenInfoTableGMTensor;
    tokenInfoTableGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(remoteFlagAddr));
    DataCopyExtParams flagCopyParams = {1U, sizeof(int32_t), 0U, 0U, 0U};
    DataCopyPad(tokenInfoTableGMTensor, statusTensor_, flagCopyParams);
    SyncFunc<HardEvent::MTE3_S>();
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::AppendRemoteDataWrite(
    HcommBatchHandle &batchHandle, GM_ADDR remoteWindowAddr, const RemoteRelayInfo &relayInfo, uint32_t dstRank,
    bool enableCqe)
{
    GM_ADDR sourceDataAddr = inputDataAddr_ + relayInfo.sourceTokenIndex * hSize_;
    if (enableCqe) {
        (void)hcomm_.WriteNbi<REMOTE_CQE_WQE_CONFIG>(batchHandle, remoteWindowAddr + relayInfo.remoteDataOffset,
                                                     sourceDataAddr, static_cast<uint32_t>(hSize_));
    } else {
        (void)hcomm_.WriteNbi<REMOTE_WQE_CONFIG>(batchHandle, remoteWindowAddr + relayInfo.remoteDataOffset,
                                                 sourceDataAddr, static_cast<uint32_t>(hSize_));
    }
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::AppendRemoteFlagWrite(
    HcommBatchHandle &batchHandle, GM_ADDR remoteWindowAddr, const RemoteRelayInfo &relayInfo, uint32_t dstRank,
    bool firstFlag, bool enableCqe)
{
    if (firstFlag) {
        if (enableCqe) {
            (void)hcomm_.WriteNbi<REMOTE_ORDERED_FLAG_CQE_WQE_CONFIG>(
                batchHandle, remoteWindowAddr + relayInfo.remoteFlagOffset, localFlagAddr_,
                static_cast<uint32_t>(sizeof(int32_t)));
        } else {
            (void)hcomm_.WriteNbi<REMOTE_ORDERED_FLAG_WQE_CONFIG>(
                batchHandle, remoteWindowAddr + relayInfo.remoteFlagOffset, localFlagAddr_,
                static_cast<uint32_t>(sizeof(int32_t)));
        }
    } else if (enableCqe) {
        (void)hcomm_.WriteNbi<REMOTE_CQE_WQE_CONFIG>(batchHandle, remoteWindowAddr + relayInfo.remoteFlagOffset,
                                                     localFlagAddr_, static_cast<uint32_t>(sizeof(int32_t)));
    } else {
        (void)hcomm_.WriteNbi<REMOTE_WQE_CONFIG>(batchHandle, remoteWindowAddr + relayInfo.remoteFlagOffset,
                                                 localFlagAddr_, static_cast<uint32_t>(sizeof(int32_t)));
    }
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::StoreRemoteRelayInfo(
    uint64_t tokenIndex, const RemoteRelayInfo &relayInfo)
{
    const uint64_t base = tokenIndex * REMOTE_RELAY_INFO_FIELD_NUM;
    SyncFunc<HardEvent::MTE3_V>();
    WriteLocalUint64(transferTensor_, 0U, relayInfo.remoteDataOffset);                             //<-
    WriteLocalUint64(transferTensor_, REMOTE_FLAG_OFFSET_FIELD_INDEX, relayInfo.remoteFlagOffset); //<-
    WriteLocalUint64(transferTensor_, SOURCE_TOKEN_INDEX_FIELD_INDEX, relayInfo.sourceTokenIndex); //<-
    WriteLocalUint64(transferTensor_, NEXT_TOKEN_INDEX_FIELD_INDEX, relayInfo.nextTokenIndex);     //<-
    DataCopyExtParams copyParams = {1U, REMOTE_RELAY_INFO_SIZE, 0U, 0U, 0U};
    SyncFunc<HardEvent::V_MTE3>();
    DataCopyPad(remoteRelayInfoGMTensor_[base], transferTensor_, copyParams);
    // The next staging write waits with MTE3_V; list construction flushes before traversal.
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::LoadRemoteRelayInfo(
    uint64_t tokenIndex, RemoteRelayInfo &relayInfo)
{
    const uint64_t base = tokenIndex * REMOTE_RELAY_INFO_FIELD_NUM;
    // Fetch the four contiguous uint64 fields with one 32B transfer.
    DataCopyExtParams copyParams = {1U, REMOTE_RELAY_INFO_SIZE, 0U, 0U, 0U};
    DataCopyPadExtParams<uint64_t> padParams = {false, 0U, 0U, 0UL};
    SyncFunc<HardEvent::S_MTE2>(); // Previous scalar reads of this UB slot must finish before it is overwritten.
    DataCopyPad(transferTensor_, remoteRelayInfoGMTensor_[base], copyParams, padParams);
    SyncFunc<HardEvent::MTE2_S>();
    relayInfo.remoteDataOffset = transferTensor_.GetValue(0U);
    relayInfo.remoteFlagOffset = transferTensor_.GetValue(REMOTE_FLAG_OFFSET_FIELD_INDEX);
    relayInfo.sourceTokenIndex = transferTensor_.GetValue(SOURCE_TOKEN_INDEX_FIELD_INDEX);
    relayInfo.nextTokenIndex = transferTensor_.GetValue(NEXT_TOKEN_INDEX_FIELD_INDEX);
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::StoreNextTokenIndex(
    uint64_t tokenIndex, uint64_t nextTokenIndex)
{
    SyncFunc<HardEvent::MTE3_V>();
    WriteLocalUint64(transferTensor_, 0U, nextTokenIndex); //<-
    DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(sizeof(uint64_t)), 0U, 0U, 0U};
    SyncFunc<HardEvent::V_MTE3>();
    // Copy exactly 8B: copying a full data block here would overwrite the following relay record.
    DataCopyPad(remoteRelayInfoGMTensor_[tokenIndex * REMOTE_RELAY_INFO_FIELD_NUM + NEXT_TOKEN_INDEX_FIELD_INDEX],
                transferTensor_, copyParams);
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::BuildRemoteRelayLists()
{
    Duplicate<uint32_t>(relayListHeadTensor_.template ReinterpretCast<uint32_t>(), INVALID_TOKEN_INDEX_LOW32,
                        worldSize_ * REMOTE_RELAY_LIST_NUM); //<-
    Duplicate<uint32_t>(relayListTailTensor_.template ReinterpretCast<uint32_t>(), INVALID_TOKEN_INDEX_LOW32,
                        worldSize_ * REMOTE_RELAY_LIST_NUM); //<-
    SyncFunc<HardEvent::V_S>();

    for (uint64_t tokenStart = 0UL; tokenStart < actualTokenNum_; tokenStart += METADATA_CHUNK_TOKEN_NUM) {
        const uint32_t tokenNum = static_cast<uint32_t>(actualTokenNum_ - tokenStart < METADATA_CHUNK_TOKEN_NUM ?
                                                            actualTokenNum_ - tokenStart :
                                                            METADATA_CHUNK_TOKEN_NUM);
        PrefetchTokenMetaData(tokenStart, tokenNum);

        for (uint32_t localOffset = 0U; localOffset < tokenNum; ++localOffset) {
            const uint64_t yOffset = tokenStart + localOffset;
            ReadTokenMetaDataStruct metaDataStruct;
            ReadTokenMetaDataFromLocal(metaDataStruct, localOffset); //<-
            if (metaDataStruct.curAttenWorkRank == rankId_) {
                continue;
            }

            const uint64_t tokenSlot = (static_cast<uint64_t>(metaDataStruct.curMicroBatchIds) * axisBS_ +
                                        metaDataStruct.curTokenBatchOffset) *
                                           expertNumPerToken_ +
                                       metaDataStruct.curTokenTopkOffset;
            const uint64_t remoteDataOffset = winTokenInfoTableSize_ + tokenSlot * axisHS_ * sizeof(xType);
            const uint64_t remoteFlagOffset = tokenSlot * sizeof(int32_t);
            const uint32_t dstRank = metaDataStruct.curAttenWorkRank;
            RemoteRelayInfo relayInfo{remoteDataOffset, remoteFlagOffset, yOffset, INVALID_TOKEN_INDEX};
            StoreRemoteRelayInfo(yOffset, relayInfo);
            SyncFunc<HardEvent::V_S>(); // Observe the previous vector update to this rank's tail.
            const uint64_t tailTokenIndex = relayListTailTensor_.GetValue(dstRank);
            SyncFunc<HardEvent::S_V>(); // Finish the tail read before updating head/tail in the same UB block.
            if (tailTokenIndex == INVALID_TOKEN_INDEX) {
                WriteLocalUint64(relayListHeadTensor_, dstRank, yOffset);
            } else {
                StoreNextTokenIndex(tailTokenIndex, yOffset);
            }
            WriteLocalUint64(relayListTailTensor_, dstRank, yOffset); //<-
        }
    }
    DataCopyExtParams listCopyParams = {1U, worldSize_ * static_cast<uint32_t>(sizeof(uint64_t)), 0U, 0U, 0U};
    SyncFunc<HardEvent::V_MTE3>();
    DataCopyPad(remoteRelayListHeadGMTensor_, relayListHeadTensor_, listCopyParams);
    DataCopyPad(remoteRelayListTailGMTensor_, relayListTailTensor_, listCopyParams);
    SyncFunc<HardEvent::MTE3_S>();
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::CommitAndDrainChannel(
    HcommBatchHandle &batchHandle, HcommChannelHandle channel, uint32_t &preparedWqeCount, uint32_t &sqWriteCount)
{
    if (preparedWqeCount != 0U) {
        (void)hcomm_.BatchCommit(batchHandle);
        preparedWqeCount = 0U;
    }
    int32_t ret = hcomm_.Drain(channel);
    ascendc_assert(ret == 0, "Urma drain failed, ret=%d", ret);
    sqWriteCount = 0U;
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        if (aivId_ == 0U) {
            HcommInit();
        }
        InitLocalFlag();

        DataCopyExtParams xCopyParams = {1U, static_cast<uint32_t>(hSize_), 0U, 0U, 0U};
        // Assign contiguous metadata chunks to cores, retaining exactly one owner per token.
        const uint64_t tokensPerCore = Ceil(actualTokenNum_, static_cast<uint64_t>(aivNum_));
        const uint32_t localChunkSize =
            tokensPerCore == 0UL ?
                1U :
                static_cast<uint32_t>(tokensPerCore < METADATA_CHUNK_TOKEN_NUM ? tokensPerCore :
                                                                                 METADATA_CHUNK_TOKEN_NUM);
        const uint64_t chunkStride = static_cast<uint64_t>(aivNum_) * localChunkSize;
        for (uint64_t tokenStart = static_cast<uint64_t>(aivId_) * localChunkSize; tokenStart < actualTokenNum_;
             tokenStart += chunkStride) {
            const uint32_t tokenNum = static_cast<uint32_t>(
                actualTokenNum_ - tokenStart < localChunkSize ? actualTokenNum_ - tokenStart : localChunkSize);
            PrefetchTokenMetaData(tokenStart, tokenNum);
            for (uint32_t localOffset = 0U; localOffset < tokenNum; ++localOffset) {
                const uint64_t yOffset = tokenStart + localOffset;
                ReadTokenMetaDataStruct metaDataStruct;
                ReadTokenMetaDataFromLocal(metaDataStruct, localOffset);
                if (metaDataStruct.curAttenWorkRank == rankId_) {
                    GM_ADDR curRankWinAddr = GetWindowAddr(metaDataStruct.curAttenWorkRank);
                    const uint64_t tokenSlot = (static_cast<uint64_t>(metaDataStruct.curMicroBatchIds) * axisBS_ +
                                                metaDataStruct.curTokenBatchOffset) *
                                                   expertNumPerToken_ +
                                               metaDataStruct.curTokenTopkOffset;
                    const uint64_t remoteDataOffset = winTokenInfoTableSize_ + tokenSlot * axisHS_ * sizeof(xType);
                    const uint64_t remoteFlagOffset = tokenSlot * sizeof(int32_t);
                    SendToLocal(curRankWinAddr + remoteDataOffset, curRankWinAddr + remoteFlagOffset, yOffset,
                                xCopyParams);
                }
            }
        }

        // Remote sends use x directly, not worker staging. Overlap them with other cores' local writes
        // when x is disjoint from the local window; retain ordering for registered-window aliases.
        if (localWritesMayAliasInput_) {
            SyncAll<true>();
        }
        if (aivId_ == 0U) {
            BuildRemoteRelayLists();
            for (uint32_t dstRank = 0U; dstRank < worldSize_; ++dstRank) {
                if (dstRank == rankId_) {
                    continue;
                }
                const uint64_t firstTokenIndex = relayListHeadTensor_.GetValue(dstRank);
                if (firstTokenIndex == INVALID_TOKEN_INDEX) {
                    continue;
                }
                GM_ADDR remoteWindowAddr = GetWindowAddr(dstRank);
                HcommChannelHandle channel = GetUrmaCommHandle(dstRank);
                HcommBatchHandle batchHandle =
                    hcomm_.MakeBatchHandle(channel, hcommBatchTensor_, URMA_BATCH_BUFFER_SIZE, remoteWindowAddr);

                uint32_t preparedWqeCount = 0U;
                uint32_t sqWriteCount = 0U;
                uint64_t tokenIndex = firstTokenIndex;
                while (tokenIndex != INVALID_TOKEN_INDEX) {
                    RemoteRelayInfo relayInfo;
                    LoadRemoteRelayInfo(tokenIndex, relayInfo);
                    tokenIndex = relayInfo.nextTokenIndex;
                    const bool enableCqe = sqWriteCount + 1U >= HCOMM_SQ_MAX_PENDING;
                    AppendRemoteDataWrite(batchHandle, remoteWindowAddr, relayInfo, dstRank, enableCqe);
                    ++preparedWqeCount;
                    ++sqWriteCount;
                    if (enableCqe) {
                        CommitAndDrainChannel(batchHandle, channel, preparedWqeCount, sqWriteCount);
                    } else if (preparedWqeCount == URMA_BATCH_WQE_CAPACITY) {
                        (void)hcomm_.BatchCommit(batchHandle);
                        preparedWqeCount = 0U;
                    }
                }

                bool firstFlag = true;
                tokenIndex = firstTokenIndex;
                while (tokenIndex != INVALID_TOKEN_INDEX) {
                    RemoteRelayInfo relayInfo;
                    LoadRemoteRelayInfo(tokenIndex, relayInfo);
                    tokenIndex = relayInfo.nextTokenIndex;
                    const bool enableCqe = sqWriteCount + 1U >= HCOMM_SQ_MAX_PENDING;
                    AppendRemoteFlagWrite(batchHandle, remoteWindowAddr, relayInfo, dstRank, firstFlag, enableCqe);
                    firstFlag = false;
                    ++preparedWqeCount;
                    ++sqWriteCount;
                    if (enableCqe) {
                        CommitAndDrainChannel(batchHandle, channel, preparedWqeCount, sqWriteCount);
                    } else if (preparedWqeCount == URMA_BATCH_WQE_CAPACITY) {
                        (void)hcomm_.BatchCommit(batchHandle);
                        preparedWqeCount = 0U;
                    }
                }

                if (preparedWqeCount != 0U) {
                    (void)hcomm_.BatchCommit(batchHandle);
                }
            }
        }

        // Keep all AIVs alive until AIV0 finishes submitting remote WQEs.
        SyncAll<true>();
    }
}

} // namespace FFNToAttentionImpl
#endif // FFN_TO_ATTENTION_URMA_H
