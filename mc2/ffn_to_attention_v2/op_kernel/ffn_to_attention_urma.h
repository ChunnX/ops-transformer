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
constexpr uint64_t URMA_WQE_MAX_DATA_SIZE = static_cast<uint64_t>(0xFFFFFFFFU);
constexpr uint32_t METADATA_CHUNK_TOKEN_NUM = 256U;
constexpr uint32_t METADATA_FIELD_NUM = 4U;
constexpr uint32_t METADATA_CHUNK_BUFFER_SIZE = METADATA_CHUNK_TOKEN_NUM * METADATA_FIELD_NUM * sizeof(int32_t);
constexpr uint32_t REMOTE_RELAY_INFO_FIELD_NUM = 4U;
constexpr uint32_t REMOTE_RELAY_INFO_SIZE = REMOTE_RELAY_INFO_FIELD_NUM * sizeof(uint64_t);
constexpr uint64_t INVALID_TOKEN_INDEX = 0xFFFFFFFFFFFFFFFFULL;
constexpr uint64_t URMA_RESERVED_WORKSPACE_SIZE = 64ULL * 1024ULL * 1024ULL;
static constexpr AscendC::UrmaWqeEntry REMOTE_DATA_WQE_CONFIG = {.odr = 5, .fence = 1, .se = 0, .inlineEn = 0};
static constexpr AscendC::UrmaWqeEntry REMOTE_ORDERED_FLAG_WQE_CONFIG = {.odr = 6, .fence = 1, .se = 0, .inlineEn = 0};
static constexpr AscendC::UrmaWqeEntry REMOTE_FLAG_WQE_CONFIG = {.odr = 5, .fence = 1, .se = 0, .inlineEn = 0};

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

static_assert(sizeof(RemoteRelayInfo) == REMOTE_RELAY_INFO_SIZE, "RemoteRelayInfo has an unexpected size");

template <TemplateFFNToAttentionUrmaTypeClass>
class FFNToAttentionUrma {
public:
    __aicore__ inline FFNToAttentionUrma(){};
    __aicore__ inline void Init(GM_ADDR mc2Context, GM_ADDR x, GM_ADDR sessionIds, GM_ADDR microBatchIds,
                                GM_ADDR tokenIds, GM_ADDR expertOffsets, GM_ADDR actualTokenNum, GM_ADDR attnRankTable,
                                GM_ADDR workspaceGM, TPipe *pipe, const FFNToAttentionV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ReadTokenMetaData(ReadTokenMetaDataStruct &metaDataStruct, uint64_t yOffset);
    __aicore__ inline void ReadTokenMetaDataFromLocal(ReadTokenMetaDataStruct &metaDataStruct, uint32_t localOffset,
                                                      uint64_t yOffset);
    __aicore__ inline GM_ADDR GetWindowAddr(uint32_t curAttenWorkRank);
    __aicore__ inline uint64_t GetUrmaCommHandle(uint32_t dstRank);
    __aicore__ inline void HcommInit();
    __aicore__ inline void InitLocalFlag();
    __aicore__ inline void SendToLocal(GM_ADDR remoteDataAddr, GM_ADDR remoteFlagAddr, uint64_t yOffset,
                                       const DataCopyExtParams &xCopyParams);
    __aicore__ inline void AppendRemoteDataWrite(HcommBatchHandle &batchHandle, GM_ADDR remoteWindowAddr,
                                                 const RemoteRelayInfo &relayInfo, uint32_t dstRank);
    __aicore__ inline void AppendRemoteFlagWrite(HcommBatchHandle &batchHandle, GM_ADDR remoteWindowAddr,
                                                 const RemoteRelayInfo &relayInfo, uint32_t dstRank, bool firstFlag,
                                                 bool lastFlag);
    __aicore__ inline void BuildRemoteRelayLists();
    __aicore__ inline void StoreRemoteRelayInfo(uint64_t tokenIndex, const RemoteRelayInfo &relayInfo);
    __aicore__ inline void LoadRemoteRelayInfo(uint64_t tokenIndex, RemoteRelayInfo &relayInfo);
    __aicore__ inline void CommitBatch(HcommBatchHandle &batchHandle);

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

    TBuf<> statusBuf_;
    TBuf<> hcommBuf_;
    TBuf<> metadataBuf_;
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
    uint64_t dataWorkspaceStride_{0};
    uint64_t winTokenInfoTableSize_{0};
    uint64_t urmaWorkspaceOffset_{0};
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
    urmaWorkspaceOffset_ = tilingData->ffnToAttentionV2Info.urmaWorkspaceOffset;
    ascendc_assert(workspaceGM != nullptr, "workspace address must not be null");

    ascendc_assert(aivNum_ > 0U, "aivNum must be greater than zero");

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

    int64_t actualTokenNumValue = actualTokenNumGMTensor_.GetValue(0);
    ascendc_assert(actualTokenNumValue >= 0, "actualTokenNum must be non-negative, actualTokenNum=%ld",
                   actualTokenNumValue);
    actualTokenNum_ = static_cast<uint64_t>(actualTokenNumValue);
    ascendc_assert(actualTokenNum_ <= maxTokenNum_,
                   "actualTokenNum exceeds x dim0, actualTokenNum=%lu, maxTokenNum=%lu", actualTokenNum_, maxTokenNum_);
    hSize_ = static_cast<uint64_t>(axisH_) * sizeof(xType);
    ascendc_assert(hSize_ > 0U && hSize_ <= URMA_WQE_MAX_DATA_SIZE && hSize_ % URMA_FLAG_SLOT_SIZE == 0U,
                   "direct URMA source requires a non-zero 32B-aligned token size within the WQE limit, hSize=%lu",
                   hSize_);
    dataWorkspaceStride_ = Ceil(hSize_, static_cast<uint64_t>(URMA_FLAG_SLOT_SIZE)) * URMA_FLAG_SLOT_SIZE;
    const uint64_t rankRelayListWorkspaceSize = static_cast<uint64_t>(worldSize_) * 2U * sizeof(uint64_t);
    ascendc_assert(rankRelayListWorkspaceSize <= URMA_RESERVED_WORKSPACE_SIZE,
                   "rank relay lists exceed reserved workspace, listSize=%lu, reservedSize=%lu",
                   rankRelayListWorkspaceSize, URMA_RESERVED_WORKSPACE_SIZE);
    ascendc_assert(maxTokenNum_ <= (URMA_RESERVED_WORKSPACE_SIZE - rankRelayListWorkspaceSize) / REMOTE_RELAY_INFO_SIZE,
                   "relay info exceeds reserved workspace, maxTokenNum=%lu, relayInfoSize=%u, reservedSize=%lu",
                   maxTokenNum_, REMOTE_RELAY_INFO_SIZE, URMA_RESERVED_WORKSPACE_SIZE);
    localFlagAddr_ = workspaceGM + urmaWorkspaceOffset_ + maxTokenNum_ * dataWorkspaceStride_ +
                     static_cast<uint64_t>(aivId_) * URMA_FLAG_SLOT_SIZE;
    GM_ADDR remoteRelayInfoAddr = workspaceGM + urmaWorkspaceOffset_ + maxTokenNum_ * dataWorkspaceStride_ +
                                  static_cast<uint64_t>(aivNum_) * URMA_FLAG_SLOT_SIZE;
    remoteRelayInfoGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(remoteRelayInfoAddr));
    GM_ADDR remoteRelayListHeadAddr = remoteRelayInfoAddr + maxTokenNum_ * REMOTE_RELAY_INFO_SIZE;
    remoteRelayListHeadGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(remoteRelayListHeadAddr));
    GM_ADDR remoteRelayListTailAddr = remoteRelayListHeadAddr + static_cast<uint64_t>(worldSize_) * sizeof(uint64_t);
    remoteRelayListTailGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(remoteRelayListTailAddr));

    tpipe_->InitBuffer(xQueue_, URMA_BUFFER_NUM, hSize_);
    tpipe_->InitBuffer(statusBuf_, URMA_UB_ALIGN);
    statusTensor_ = statusBuf_.Get<int32_t>();
    tpipe_->InitBuffer(metadataBuf_, METADATA_CHUNK_BUFFER_SIZE);
    metadataTensor_ = metadataBuf_.Get<int32_t>();

    batchSizeSendCnt_ = static_cast<uint64_t>(axisBS_) * expertNumPerToken_;
    winTokenInfoTableSize_ =
        Ceil(static_cast<uint64_t>(microBatchNum_) * batchSizeSendCnt_ * sizeof(int32_t), URMA_WIN_ADDR_ALIGN) *
        URMA_WIN_ADDR_ALIGN;
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
    SyncFunc<HardEvent::MTE3_S>();
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
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::ReadTokenMetaData(
    ReadTokenMetaDataStruct &metaDataStruct, uint64_t yOffset)
{
    int32_t curAttenWorkId = sessionIdsGMTensor_.GetValue(yOffset);
    metaDataStruct.curAttenWorkIds = static_cast<uint32_t>(curAttenWorkId);
    metaDataStruct.curAttenWorkRank = metaDataStruct.curAttenWorkIds;

    if constexpr (isInputRankTable) {
        ascendc_assert(metaDataStruct.curAttenWorkIds < axisA_,
                       "sessionId exceeds attnRankTable, sessionId=%u, attnRankNum=%u, yOffset=%lu",
                       metaDataStruct.curAttenWorkIds, axisA_, yOffset);
        int32_t curAttenWorkRank = attnRankTableGMTensor_.GetValue(metaDataStruct.curAttenWorkIds);
        metaDataStruct.curAttenWorkRank = static_cast<uint32_t>(curAttenWorkRank);
    }
    ascendc_assert(metaDataStruct.curAttenWorkRank < worldSize_,
                   "attention rank exceeds worldSize, rank=%u, worldSize=%u, yOffset=%lu",
                   metaDataStruct.curAttenWorkRank, worldSize_, yOffset);

    int32_t curMicroBatchId = microBatchIdsGMTensor_.GetValue(yOffset);
    metaDataStruct.curMicroBatchIds = static_cast<uint32_t>(curMicroBatchId);

    int32_t curTokenBatchOffset = tokenIdsGMTensor_.GetValue(yOffset);
    metaDataStruct.curTokenBatchOffset = static_cast<uint32_t>(curTokenBatchOffset);

    int32_t curTokenTopkOffset = expertOffsetsGMTensor_.GetValue(yOffset);
    metaDataStruct.curTokenTopkOffset = static_cast<uint32_t>(curTokenTopkOffset);
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::ReadTokenMetaDataFromLocal(
    ReadTokenMetaDataStruct &metaDataStruct, uint32_t localOffset, uint64_t yOffset)
{
    int32_t curAttenWorkId = metadataTensor_.GetValue(localOffset);
    metaDataStruct.curAttenWorkIds = static_cast<uint32_t>(curAttenWorkId);
    metaDataStruct.curAttenWorkRank = metaDataStruct.curAttenWorkIds;

    if constexpr (isInputRankTable) {
        ascendc_assert(metaDataStruct.curAttenWorkIds < axisA_,
                       "sessionId exceeds attnRankTable, sessionId=%u, attnRankNum=%u, yOffset=%lu",
                       metaDataStruct.curAttenWorkIds, axisA_, yOffset);
        int32_t curAttenWorkRank = attnRankTableGMTensor_.GetValue(metaDataStruct.curAttenWorkIds);
        metaDataStruct.curAttenWorkRank = static_cast<uint32_t>(curAttenWorkRank);
    }
    ascendc_assert(metaDataStruct.curAttenWorkRank < worldSize_,
                   "attention rank exceeds worldSize, rank=%u, worldSize=%u, yOffset=%lu",
                   metaDataStruct.curAttenWorkRank, worldSize_, yOffset);

    int32_t curMicroBatchId = metadataTensor_.GetValue(METADATA_CHUNK_TOKEN_NUM + localOffset);
    metaDataStruct.curMicroBatchIds = static_cast<uint32_t>(curMicroBatchId);

    int32_t curTokenBatchOffset = metadataTensor_.GetValue(2U * METADATA_CHUNK_TOKEN_NUM + localOffset);
    metaDataStruct.curTokenBatchOffset = static_cast<uint32_t>(curTokenBatchOffset);

    int32_t curTokenTopkOffset = metadataTensor_.GetValue(3U * METADATA_CHUNK_TOKEN_NUM + localOffset);
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
    HcommBatchHandle &batchHandle, GM_ADDR remoteWindowAddr, const RemoteRelayInfo &relayInfo, uint32_t dstRank)
{
    GM_ADDR sourceDataAddr = inputDataAddr_ + relayInfo.sourceTokenIndex * hSize_;
    int32_t ret = hcomm_.WriteNbi<REMOTE_DATA_WQE_CONFIG>(batchHandle, remoteWindowAddr + relayInfo.remoteDataOffset,
                                                          sourceDataAddr, static_cast<uint32_t>(hSize_));
    ascendc_assert(ret == 0, "batch data WriteNbi failed, ret=%d, rankId=%u, dstRank=%u, tokenIndex=%lu", ret, rankId_,
                   dstRank, relayInfo.sourceTokenIndex);
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::AppendRemoteFlagWrite(
    HcommBatchHandle &batchHandle, GM_ADDR remoteWindowAddr, const RemoteRelayInfo &relayInfo, uint32_t dstRank,
    bool firstFlag, bool lastFlag)
{
    int32_t ret;
    if (firstFlag || lastFlag) {
        ret =
            hcomm_.WriteNbi<REMOTE_ORDERED_FLAG_WQE_CONFIG>(batchHandle, remoteWindowAddr + relayInfo.remoteFlagOffset,
                                                            localFlagAddr_, static_cast<uint32_t>(sizeof(int32_t)));
    } else {
        ret = hcomm_.WriteNbi<REMOTE_FLAG_WQE_CONFIG>(batchHandle, remoteWindowAddr + relayInfo.remoteFlagOffset,
                                                      localFlagAddr_, static_cast<uint32_t>(sizeof(int32_t)));
    }
    ascendc_assert(ret == 0, "batch flag WriteNbi failed, ret=%d, rankId=%u, dstRank=%u, tokenIndex=%lu", ret, rankId_,
                   dstRank, relayInfo.sourceTokenIndex);
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::StoreRemoteRelayInfo(
    uint64_t tokenIndex, const RemoteRelayInfo &relayInfo)
{
    const uint64_t base = tokenIndex * REMOTE_RELAY_INFO_FIELD_NUM;
    remoteRelayInfoGMTensor_.SetValue(base, relayInfo.remoteDataOffset);
    remoteRelayInfoGMTensor_.SetValue(base + 1U, relayInfo.remoteFlagOffset);
    remoteRelayInfoGMTensor_.SetValue(base + 2U, relayInfo.sourceTokenIndex);
    remoteRelayInfoGMTensor_.SetValue(base + 3U, relayInfo.nextTokenIndex);
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::LoadRemoteRelayInfo(
    uint64_t tokenIndex, RemoteRelayInfo &relayInfo)
{
    const uint64_t base = tokenIndex * REMOTE_RELAY_INFO_FIELD_NUM;
    relayInfo.remoteDataOffset = remoteRelayInfoGMTensor_.GetValue(base);
    relayInfo.remoteFlagOffset = remoteRelayInfoGMTensor_.GetValue(base + 1U);
    relayInfo.sourceTokenIndex = remoteRelayInfoGMTensor_.GetValue(base + 2U);
    relayInfo.nextTokenIndex = remoteRelayInfoGMTensor_.GetValue(base + 3U);
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::BuildRemoteRelayLists()
{
    for (uint32_t dstRank = 0U; dstRank < worldSize_; ++dstRank) {
        remoteRelayListHeadGMTensor_.SetValue(dstRank, INVALID_TOKEN_INDEX);
        remoteRelayListTailGMTensor_.SetValue(dstRank, INVALID_TOKEN_INDEX);
    }

    for (uint64_t tokenStart = 0UL; tokenStart < actualTokenNum_; tokenStart += METADATA_CHUNK_TOKEN_NUM) {
        const uint32_t tokenNum = static_cast<uint32_t>(actualTokenNum_ - tokenStart < METADATA_CHUNK_TOKEN_NUM ?
                                                            actualTokenNum_ - tokenStart :
                                                            METADATA_CHUNK_TOKEN_NUM);
        if (tokenStart != 0UL) {
            SyncFunc<HardEvent::S_MTE2>();
        }
        DataCopyExtParams metadataCopyParams = {1U, static_cast<uint32_t>(tokenNum * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> metadataPadParams = {false, 0U, 0U, 0U};
        DataCopyPad(metadataTensor_, sessionIdsGMTensor_[tokenStart], metadataCopyParams, metadataPadParams);
        DataCopyPad(metadataTensor_[METADATA_CHUNK_TOKEN_NUM], microBatchIdsGMTensor_[tokenStart], metadataCopyParams,
                    metadataPadParams);
        DataCopyPad(metadataTensor_[2U * METADATA_CHUNK_TOKEN_NUM], tokenIdsGMTensor_[tokenStart], metadataCopyParams,
                    metadataPadParams);
        DataCopyPad(metadataTensor_[3U * METADATA_CHUNK_TOKEN_NUM], expertOffsetsGMTensor_[tokenStart],
                    metadataCopyParams, metadataPadParams);
        SyncFunc<HardEvent::MTE2_S>();

        for (uint32_t localOffset = 0U; localOffset < tokenNum; ++localOffset) {
            const uint64_t yOffset = tokenStart + localOffset;
            ReadTokenMetaDataStruct metaDataStruct;
            ReadTokenMetaDataFromLocal(metaDataStruct, localOffset, yOffset);
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
            const uint64_t tailTokenIndex = remoteRelayListTailGMTensor_.GetValue(dstRank);
            if (tailTokenIndex == INVALID_TOKEN_INDEX) {
                remoteRelayListHeadGMTensor_.SetValue(dstRank, yOffset);
            } else {
                remoteRelayInfoGMTensor_.SetValue(tailTokenIndex * REMOTE_RELAY_INFO_FIELD_NUM + 3U, yOffset);
            }
            remoteRelayListTailGMTensor_.SetValue(dstRank, yOffset);
        }
    }
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::CommitBatch(
    HcommBatchHandle &batchHandle)
{
    int32_t ret = hcomm_.BatchCommit(batchHandle);
    ascendc_assert(ret == 0, "BatchCommit failed, ret=%d, rankId=%u", ret, rankId_);
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        if (aivId_ == 0U) {
            HcommInit();
        }
        InitLocalFlag();

        const uint64_t roundNum = Ceil(actualTokenNum_, static_cast<uint64_t>(aivNum_));
        DataCopyExtParams xCopyParams = {1U, static_cast<uint32_t>(hSize_), 0U, 0U, 0U};
        for (uint64_t round = 0; round < roundNum; ++round) {
            const uint64_t yOffset = static_cast<uint64_t>(aivId_) + round * aivNum_;
            if (yOffset < actualTokenNum_) {
                ReadTokenMetaDataStruct metaDataStruct;
                ReadTokenMetaData(metaDataStruct, yOffset);
                if (metaDataStruct.curAttenWorkRank == rankId_) {
                    GM_ADDR curRankWinAddr = GetWindowAddr(metaDataStruct.curAttenWorkRank);
                    ascendc_assert(curRankWinAddr != nullptr, "window address is null, dstRank=%u",
                                   metaDataStruct.curAttenWorkRank);
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

        // AIV0 starts remote writes only after every AIV completes its local path.
        SyncAll<true>();
        if (aivId_ == 0U) {
            BuildRemoteRelayLists();
            for (uint32_t dstRank = 0U; dstRank < worldSize_; ++dstRank) {
                if (dstRank == rankId_) {
                    continue;
                }
                const uint64_t firstTokenIndex = remoteRelayListHeadGMTensor_.GetValue(dstRank);
                if (firstTokenIndex == INVALID_TOKEN_INDEX) {
                    continue;
                }
                GM_ADDR remoteWindowAddr = GetWindowAddr(dstRank);
                ascendc_assert(remoteWindowAddr != nullptr, "window address is null, dstRank=%u", dstRank);
                HcommChannelHandle channel = GetUrmaCommHandle(dstRank);
                HcommBatchHandle batchHandle =
                    hcomm_.MakeBatchHandle(channel, hcommBatchTensor_, URMA_BATCH_BUFFER_SIZE, remoteWindowAddr);

                uint32_t preparedWqeCount = 0U;
                uint64_t tokenIndex = firstTokenIndex;
                while (tokenIndex != INVALID_TOKEN_INDEX) {
                    RemoteRelayInfo relayInfo;
                    LoadRemoteRelayInfo(tokenIndex, relayInfo);
                    tokenIndex = relayInfo.nextTokenIndex;
                    AppendRemoteDataWrite(batchHandle, remoteWindowAddr, relayInfo, dstRank);
                    ++preparedWqeCount;
                    if (preparedWqeCount == URMA_BATCH_WQE_CAPACITY) {
                        CommitBatch(batchHandle);
                        preparedWqeCount = 0U;
                    }
                }

                bool firstFlag = true;
                tokenIndex = firstTokenIndex;
                while (tokenIndex != INVALID_TOKEN_INDEX) {
                    RemoteRelayInfo relayInfo;
                    LoadRemoteRelayInfo(tokenIndex, relayInfo);
                    tokenIndex = relayInfo.nextTokenIndex;
                    const bool lastFlag = tokenIndex == INVALID_TOKEN_INDEX;
                    AppendRemoteFlagWrite(batchHandle, remoteWindowAddr, relayInfo, dstRank, firstFlag, lastFlag);
                    firstFlag = false;
                    ++preparedWqeCount;
                    if (preparedWqeCount == URMA_BATCH_WQE_CAPACITY) {
                        CommitBatch(batchHandle);
                        preparedWqeCount = 0U;
                    }
                }

                if (preparedWqeCount != 0U) {
                    CommitBatch(batchHandle);
                }
            }
        }

        // Keep all AIVs alive until AIV0 finishes submitting remote WQEs.
        SyncAll<true>();
    }
}

} // namespace FFNToAttentionImpl
#endif // FFN_TO_ATTENTION_URMA_H
