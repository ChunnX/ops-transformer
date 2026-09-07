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
constexpr uint32_t REMOTE_RELAY_INFO_SIZE = 32U;
constexpr uint32_t PERF_FIELD_NUM = 12U;
constexpr uint32_t PERF_STORAGE_FIELD_NUM = 16U;
static constexpr AscendC::UrmaWqeEntry REMOTE_DATA_WQE_CONFIG = {
    .odr = 5, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
static constexpr AscendC::UrmaWqeEntry REMOTE_FIRST_FLAG_WQE_CONFIG = {
    .odr = 6, .fence = 1, .se = 0, .cqe = 1, .inlineEn = 0};
static constexpr AscendC::UrmaWqeEntry REMOTE_FLAG_WQE_CONFIG = {
    .odr = 5, .fence = 1, .se = 0, .cqe = 1, .inlineEn = 0};

#define TemplateFFNToAttentionUrmaTypeClass typename xType, bool isInputRankTable
#define TemplateFFNToAttentionUrmaTypeFunc xType, isInputRankTable

using namespace AscendC;

using HcommChannelHandle = AscendC::ChannelHandle;
using HcommBatchHandle = AscendC::BatchHandle<HcommChannelHandle>;

struct alignas(REMOTE_RELAY_INFO_SIZE) RemoteRelayInfo {
    uint64_t remoteDataOffset;
    uint64_t remoteFlagOffset;
    uint64_t sourceTokenIndex;
    uint32_t dstRank;
};

static_assert(sizeof(RemoteRelayInfo) == REMOTE_RELAY_INFO_SIZE, "RemoteRelayInfo must occupy 32 bytes");

template <TemplateFFNToAttentionUrmaTypeClass>
class FFNToAttentionUrma {
public:
    __aicore__ inline FFNToAttentionUrma(){};
    __aicore__ inline void Init(GM_ADDR mc2Context, GM_ADDR x, GM_ADDR sessionIds, GM_ADDR microBatchIds,
                                GM_ADDR tokenIds, GM_ADDR expertOffsets, GM_ADDR actualTokenNum, GM_ADDR attnRankTable,
                                GM_ADDR perfData, GM_ADDR workspaceGM, TPipe *pipe,
                                const FFNToAttentionV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ReadTokenMetaData(ReadTokenMetaDataStruct &metaDataStruct, uint64_t yOffset);
    __aicore__ inline GM_ADDR GetWindowAddr(uint32_t curAttenWorkRank);
    __aicore__ inline uint64_t GetUrmaCommHandle(uint32_t dstRank);
    __aicore__ inline void HcommInit();
    __aicore__ inline void InitLocalFlag();
    __aicore__ inline void SendToLocal(GM_ADDR remoteDataAddr, GM_ADDR remoteFlagAddr, uint64_t yOffset,
                                       const DataCopyExtParams &xCopyParams);
    __aicore__ inline void StageRemoteData(uint64_t yOffset, const DataCopyExtParams &xCopyParams);
    __aicore__ inline void AppendRemoteDataWrite(HcommBatchHandle &batchHandle, GM_ADDR remoteWindowAddr,
                                                 const RemoteRelayInfo &relayInfo);
    __aicore__ inline void AppendRemoteFlagWrite(HcommBatchHandle &batchHandle, GM_ADDR remoteWindowAddr,
                                                 const RemoteRelayInfo &relayInfo, bool firstFlag);
    __aicore__ inline void CommitBatch(HcommBatchHandle &batchHandle, uint64_t &commitCycles);
    __aicore__ inline void WritePerfData(const uint64_t (&cycles)[PERF_FIELD_NUM]);

    TPipe *tpipe_{nullptr};
    Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
    __gm__ Mc2Aclnn::AttentionFFNContext *mc2Context_{nullptr};

    GM_ADDR localDataAddr_{nullptr};
    GM_ADDR localFlagAddr_{nullptr};

    GlobalTensor<xType> xGMTensor_;
    GlobalTensor<int32_t> sessionIdsGMTensor_;
    GlobalTensor<int32_t> microBatchIdsGMTensor_;
    GlobalTensor<int32_t> tokenIdsGMTensor_;
    GlobalTensor<int32_t> expertOffsetsGMTensor_;
    GlobalTensor<int64_t> actualTokenNumGMTensor_;
    GlobalTensor<int32_t> attnRankTableGMTensor_;
    GlobalTensor<int64_t> perfDataGMTensor_;
    LocalTensor<xType> xTmpTensor_;
    LocalTensor<int32_t> statusTensor_;
    LocalTensor<uint8_t> hcommTensor_;
    LocalTensor<uint8_t> hcommBatchTensor_;

    TBuf<> statusBuf_;
    TBuf<> hcommBuf_;
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
    uint64_t requiredWindowSize_{0};
    uint64_t urmaWorkspaceOffset_{0};
    bool perfEnabled_{false};
};

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::Init(
    GM_ADDR mc2Context, GM_ADDR x, GM_ADDR sessionIds, GM_ADDR microBatchIds, GM_ADDR tokenIds, GM_ADDR expertOffsets,
    GM_ADDR actualTokenNum, GM_ADDR attnRankTable, GM_ADDR perfData, GM_ADDR workspaceGM, TPipe *pipe,
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
    ascendc_assert(axisH_ <= axisHS_, "H must be <= HS, H=%u, HS=%u", axisH_, axisHS_);

    xGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(x));
    sessionIdsGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sessionIds));
    microBatchIdsGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(microBatchIds));
    tokenIdsGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(tokenIds));
    expertOffsetsGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(expertOffsets));
    actualTokenNumGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(actualTokenNum));
    if constexpr (isInputRankTable) {
        attnRankTableGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(attnRankTable));
    }
    perfEnabled_ = perfData != nullptr;
    if (perfEnabled_) {
        perfDataGMTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(perfData));
    }

    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(actualTokenNumGMTensor_);
    int64_t actualTokenNumValue = actualTokenNumGMTensor_.GetValue(0);
    ascendc_assert(actualTokenNumValue >= 0, "actualTokenNum must be non-negative, actualTokenNum=%ld",
                   actualTokenNumValue);
    actualTokenNum_ = static_cast<uint64_t>(actualTokenNumValue);
    ascendc_assert(actualTokenNum_ <= maxTokenNum_,
                   "actualTokenNum exceeds x dim0, actualTokenNum=%lu, maxTokenNum=%lu", actualTokenNum_, maxTokenNum_);
    hSize_ = static_cast<uint64_t>(axisH_) * sizeof(xType);
    dataWorkspaceStride_ = Ceil(hSize_, static_cast<uint64_t>(URMA_FLAG_SLOT_SIZE)) * URMA_FLAG_SLOT_SIZE;
    localDataAddr_ = workspaceGM + urmaWorkspaceOffset_;
    localFlagAddr_ = workspaceGM + urmaWorkspaceOffset_ + maxTokenNum_ * dataWorkspaceStride_ +
                     static_cast<uint64_t>(aivId_) * URMA_FLAG_SLOT_SIZE;

    tpipe_->InitBuffer(xQueue_, URMA_BUFFER_NUM, hSize_);
    tpipe_->InitBuffer(statusBuf_, URMA_UB_ALIGN);
    statusTensor_ = statusBuf_.Get<int32_t>();

    batchSizeSendCnt_ = static_cast<uint64_t>(axisBS_) * expertNumPerToken_;
    winTokenInfoTableSize_ =
        Ceil(static_cast<uint64_t>(microBatchNum_) * batchSizeSendCnt_ * sizeof(int32_t), URMA_WIN_ADDR_ALIGN) *
        URMA_WIN_ADDR_ALIGN;
    const uint64_t totalSlotNum = static_cast<uint64_t>(microBatchNum_) * axisBS_ * expertNumPerToken_;
    requiredWindowSize_ = winTokenInfoTableSize_ + totalSlotNum * axisHS_ * sizeof(xType);
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
    ascendc_assert(curAttenWorkId >= 0, "sessionId must be non-negative, sessionId=%d, yOffset=%lu", curAttenWorkId,
                   yOffset);
    metaDataStruct.curAttenWorkIds = static_cast<uint32_t>(curAttenWorkId);
    metaDataStruct.curAttenWorkRank = metaDataStruct.curAttenWorkIds;

    if constexpr (isInputRankTable) {
        ascendc_assert(metaDataStruct.curAttenWorkIds < axisA_,
                       "sessionId exceeds attnRankTable, sessionId=%u, attnRankNum=%u, yOffset=%lu",
                       metaDataStruct.curAttenWorkIds, axisA_, yOffset);
        int32_t curAttenWorkRank = attnRankTableGMTensor_.GetValue(metaDataStruct.curAttenWorkIds);
        ascendc_assert(curAttenWorkRank >= 0, "attention rank must be non-negative, rank=%d, sessionId=%u",
                       curAttenWorkRank, metaDataStruct.curAttenWorkIds);
        metaDataStruct.curAttenWorkRank = static_cast<uint32_t>(curAttenWorkRank);
    }
    ascendc_assert(metaDataStruct.curAttenWorkRank < worldSize_,
                   "attention rank exceeds worldSize, rank=%u, worldSize=%u, yOffset=%lu",
                   metaDataStruct.curAttenWorkRank, worldSize_, yOffset);

    int32_t curMicroBatchId = microBatchIdsGMTensor_.GetValue(yOffset);
    ascendc_assert(curMicroBatchId >= 0 && static_cast<uint32_t>(curMicroBatchId) < microBatchNum_,
                   "microBatchId out of range, microBatchId=%d, microBatchNum=%u, yOffset=%lu", curMicroBatchId,
                   microBatchNum_, yOffset);
    metaDataStruct.curMicroBatchIds = static_cast<uint32_t>(curMicroBatchId);

    int32_t curTokenBatchOffset = tokenIdsGMTensor_.GetValue(yOffset);
    ascendc_assert(curTokenBatchOffset >= 0 && static_cast<uint32_t>(curTokenBatchOffset) < axisBS_,
                   "tokenId out of range, tokenId=%d, BS=%u, yOffset=%lu", curTokenBatchOffset, axisBS_, yOffset);
    metaDataStruct.curTokenBatchOffset = static_cast<uint32_t>(curTokenBatchOffset);

    int32_t curTokenTopkOffset = expertOffsetsGMTensor_.GetValue(yOffset);
    ascendc_assert(curTokenTopkOffset >= 0 && static_cast<uint32_t>(curTokenTopkOffset) < expertNumPerToken_,
                   "expertOffset out of range, expertOffset=%d, expertNumPerToken=%u, yOffset=%lu", curTokenTopkOffset,
                   expertNumPerToken_, yOffset);
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
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::StageRemoteData(
    uint64_t yOffset, const DataCopyExtParams &xCopyParams)
{
    DataCopyPadExtParams<xType> copyPadExtParams{false, 0U, 0U, 0U};
    xTmpTensor_ = xQueue_.AllocTensor<xType>();
    DataCopyPad(xTmpTensor_, xGMTensor_[yOffset * axisH_], xCopyParams, copyPadExtParams);
    xQueue_.EnQue(xTmpTensor_);
    xTmpTensor_ = xQueue_.DeQue<xType>();

    GlobalTensor<xType> localDataGMTensor;
    localDataGMTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ xType *>(localDataAddr_ + yOffset * dataWorkspaceStride_));
    DataCopyPad(localDataGMTensor, xTmpTensor_, xCopyParams);
    xQueue_.FreeTensor<xType>(xTmpTensor_);
    SyncFunc<HardEvent::MTE3_S>();
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::AppendRemoteDataWrite(
    HcommBatchHandle &batchHandle, GM_ADDR remoteWindowAddr, const RemoteRelayInfo &relayInfo)
{
    ascendc_assert(aivId_ == 0U, "only AIV0 may append remote WQEs, aivId=%u", aivId_);
    ascendc_assert(relayInfo.sourceTokenIndex < actualTokenNum_,
                   "source token index out of range, tokenIndex=%lu, actualTokenNum=%lu", relayInfo.sourceTokenIndex,
                   actualTokenNum_);
    GM_ADDR sourceDataAddr = localDataAddr_ + relayInfo.sourceTokenIndex * dataWorkspaceStride_;
    int32_t ret = hcomm_.WriteNbi<REMOTE_DATA_WQE_CONFIG>(batchHandle, remoteWindowAddr + relayInfo.remoteDataOffset,
                                                          sourceDataAddr, static_cast<uint32_t>(hSize_));
    ascendc_assert(ret == 0, "batch data WriteNbi failed, ret=%d, rankId=%u, dstRank=%u, tokenIndex=%lu", ret, rankId_,
                   relayInfo.dstRank, relayInfo.sourceTokenIndex);
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::AppendRemoteFlagWrite(
    HcommBatchHandle &batchHandle, GM_ADDR remoteWindowAddr, const RemoteRelayInfo &relayInfo, bool firstFlag)
{
    int32_t ret;
    if (firstFlag) {
        ret = hcomm_.WriteNbi<REMOTE_FIRST_FLAG_WQE_CONFIG>(batchHandle, remoteWindowAddr + relayInfo.remoteFlagOffset,
                                                            localFlagAddr_, static_cast<uint32_t>(sizeof(int32_t)));
    } else {
        ret = hcomm_.WriteNbi<REMOTE_FLAG_WQE_CONFIG>(batchHandle, remoteWindowAddr + relayInfo.remoteFlagOffset,
                                                      localFlagAddr_, static_cast<uint32_t>(sizeof(int32_t)));
    }
    ascendc_assert(ret == 0, "batch flag WriteNbi failed, ret=%d, rankId=%u, dstRank=%u, tokenIndex=%lu", ret, rankId_,
                   relayInfo.dstRank, relayInfo.sourceTokenIndex);
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::CommitBatch(
    HcommBatchHandle &batchHandle, uint64_t &commitCycles)
{
    // prof11 BatchCommit
    const uint64_t prof11Start = perfEnabled_ ? static_cast<uint64_t>(GetSystemCycle()) : 0UL;
    int32_t ret = hcomm_.BatchCommit(batchHandle);
    ascendc_assert(ret == 0, "BatchCommit failed, ret=%d, rankId=%u", ret, rankId_);
    if (perfEnabled_) {
        commitCycles += static_cast<uint64_t>(GetSystemCycle()) - prof11Start;
    }
    // prof11 BatchCommit
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::WritePerfData(
    const uint64_t (&cycles)[PERF_FIELD_NUM])
{
    if (!perfEnabled_) {
        return;
    }
    const uint64_t base = static_cast<uint64_t>(aivId_) * PERF_STORAGE_FIELD_NUM;
    for (uint32_t i = 0U; i < PERF_FIELD_NUM; ++i) {
        perfDataGMTensor_.SetValue(base + i, static_cast<int64_t>(cycles[i]));
    }
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(perfDataGMTensor_[base]);
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        perfDataGMTensor_[base + 8U]);
}

template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        uint64_t perfCycles[PERF_FIELD_NUM] = {0UL};
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
                // prof1 ReadTokenMetaData
                const uint64_t prof1Start = perfEnabled_ ? static_cast<uint64_t>(GetSystemCycle()) : 0UL;
                ReadTokenMetaData(metaDataStruct, yOffset);
                if (perfEnabled_) {
                    perfCycles[0] += static_cast<uint64_t>(GetSystemCycle()) - prof1Start;
                }
                // prof1 ReadTokenMetaData
                if (metaDataStruct.curAttenWorkRank == rankId_) {
                    // prof2 LocalAddressAndOffset
                    const uint64_t prof2Start = perfEnabled_ ? static_cast<uint64_t>(GetSystemCycle()) : 0UL;
                    GM_ADDR curRankWinAddr = GetWindowAddr(metaDataStruct.curAttenWorkRank);
                    ascendc_assert(curRankWinAddr != nullptr, "window address is null, dstRank=%u",
                                   metaDataStruct.curAttenWorkRank);
                    const uint64_t tokenSlot = (static_cast<uint64_t>(metaDataStruct.curMicroBatchIds) * axisBS_ +
                                                metaDataStruct.curTokenBatchOffset) *
                                                   expertNumPerToken_ +
                                               metaDataStruct.curTokenTopkOffset;
                    const uint64_t remoteDataOffset = winTokenInfoTableSize_ + tokenSlot * axisHS_ * sizeof(xType);
                    const uint64_t remoteFlagOffset = tokenSlot * sizeof(int32_t);
                    ascendc_assert(
                        remoteDataOffset <= requiredWindowSize_ && hSize_ <= requiredWindowSize_ - remoteDataOffset,
                        "token data address exceeds window, dataOffset=%lu, dataSize=%lu, windowSize=%lu",
                        remoteDataOffset, hSize_, requiredWindowSize_);
                    ascendc_assert(remoteFlagOffset <= winTokenInfoTableSize_ &&
                                       sizeof(int32_t) <= winTokenInfoTableSize_ - remoteFlagOffset,
                                   "token flag address exceeds info table, flagOffset=%lu, infoSize=%lu",
                                   remoteFlagOffset, winTokenInfoTableSize_);
                    if (perfEnabled_) {
                        perfCycles[1] += static_cast<uint64_t>(GetSystemCycle()) - prof2Start;
                    }
                    // prof2 LocalAddressAndOffset
                    // prof3 SendToLocal
                    const uint64_t prof3Start = perfEnabled_ ? static_cast<uint64_t>(GetSystemCycle()) : 0UL;
                    SendToLocal(curRankWinAddr + remoteDataOffset, curRankWinAddr + remoteFlagOffset, yOffset,
                                xCopyParams);
                    if (perfEnabled_) {
                        perfCycles[2] += static_cast<uint64_t>(GetSystemCycle()) - prof3Start;
                    }
                    // prof3 SendToLocal
                } else {
                    // prof4 StageRemoteData
                    const uint64_t prof4Start = perfEnabled_ ? static_cast<uint64_t>(GetSystemCycle()) : 0UL;
                    StageRemoteData(yOffset, xCopyParams);
                    if (perfEnabled_) {
                        perfCycles[3] += static_cast<uint64_t>(GetSystemCycle()) - prof4Start;
                    }
                    // prof4 StageRemoteData
                }
            }
        }

        // All remote tokens have unique staging slots. AIV0 may consume them after every AIV finishes staging.
        SyncAll<true>();
        if (aivId_ == 0U) {
            for (uint32_t dstRank = 0U; dstRank < worldSize_; ++dstRank) {
                if (dstRank == rankId_) {
                    continue;
                }

                // prof5 CountRemoteTokens
                const uint64_t prof5Start = perfEnabled_ ? static_cast<uint64_t>(GetSystemCycle()) : 0UL;
                uint64_t rankTokenCount = 0UL;
                for (uint64_t yOffset = 0UL; yOffset < actualTokenNum_; ++yOffset) {
                    ReadTokenMetaDataStruct metaDataStruct;
                    ReadTokenMetaData(metaDataStruct, yOffset);
                    if (metaDataStruct.curAttenWorkRank == dstRank) {
                        ++rankTokenCount;
                    }
                }
                if (perfEnabled_) {
                    perfCycles[4] += static_cast<uint64_t>(GetSystemCycle()) - prof5Start;
                }
                // prof5 CountRemoteTokens
                if (rankTokenCount == 0U) {
                    continue;
                }
                // prof6 MakeBatchHandle
                const uint64_t prof6Start = perfEnabled_ ? static_cast<uint64_t>(GetSystemCycle()) : 0UL;
                GM_ADDR remoteWindowAddr = GetWindowAddr(dstRank);
                ascendc_assert(remoteWindowAddr != nullptr, "window address is null, dstRank=%u", dstRank);
                HcommChannelHandle channel = GetUrmaCommHandle(dstRank);
                HcommBatchHandle batchHandle =
                    hcomm_.MakeBatchHandle(channel, hcommBatchTensor_, URMA_BATCH_BUFFER_SIZE, remoteWindowAddr);
                if (perfEnabled_) {
                    perfCycles[5] += static_cast<uint64_t>(GetSystemCycle()) - prof6Start;
                }
                // prof6 MakeBatchHandle

                uint32_t preparedWqeCount = 0U;
                for (uint64_t yOffset = 0UL; yOffset < actualTokenNum_; ++yOffset) {
                    ReadTokenMetaDataStruct metaDataStruct;
                    ReadTokenMetaData(metaDataStruct, yOffset);
                    if (metaDataStruct.curAttenWorkRank != dstRank) {
                        continue;
                    }
                    // prof7 BuildRemoteDataInfo
                    const uint64_t prof7Start = perfEnabled_ ? static_cast<uint64_t>(GetSystemCycle()) : 0UL;
                    const uint64_t tokenSlot = (static_cast<uint64_t>(metaDataStruct.curMicroBatchIds) * axisBS_ +
                                                metaDataStruct.curTokenBatchOffset) *
                                                   expertNumPerToken_ +
                                               metaDataStruct.curTokenTopkOffset;
                    const uint64_t remoteDataOffset = winTokenInfoTableSize_ + tokenSlot * axisHS_ * sizeof(xType);
                    const uint64_t remoteFlagOffset = tokenSlot * sizeof(int32_t);
                    ascendc_assert(
                        remoteDataOffset <= requiredWindowSize_ && hSize_ <= requiredWindowSize_ - remoteDataOffset,
                        "token data address exceeds window, dataOffset=%lu, dataSize=%lu, windowSize=%lu",
                        remoteDataOffset, hSize_, requiredWindowSize_);
                    ascendc_assert(remoteFlagOffset <= winTokenInfoTableSize_ &&
                                       sizeof(int32_t) <= winTokenInfoTableSize_ - remoteFlagOffset,
                                   "token flag address exceeds info table, flagOffset=%lu, infoSize=%lu",
                                   remoteFlagOffset, winTokenInfoTableSize_);
                    RemoteRelayInfo relayInfo{remoteDataOffset, remoteFlagOffset, yOffset, dstRank};
                    if (perfEnabled_) {
                        perfCycles[6] += static_cast<uint64_t>(GetSystemCycle()) - prof7Start;
                    }
                    // prof7 BuildRemoteDataInfo
                    // prof8 AppendRemoteDataWrite
                    const uint64_t prof8Start = perfEnabled_ ? static_cast<uint64_t>(GetSystemCycle()) : 0UL;
                    AppendRemoteDataWrite(batchHandle, remoteWindowAddr, relayInfo);
                    if (perfEnabled_) {
                        perfCycles[7] += static_cast<uint64_t>(GetSystemCycle()) - prof8Start;
                    }
                    // prof8 AppendRemoteDataWrite
                    ++preparedWqeCount;
                    if (preparedWqeCount == URMA_BATCH_WQE_CAPACITY) {
                        CommitBatch(batchHandle, perfCycles[10]);
                        preparedWqeCount = 0U;
                    }
                }

                bool firstFlag = true;
                for (uint64_t yOffset = 0UL; yOffset < actualTokenNum_; ++yOffset) {
                    ReadTokenMetaDataStruct metaDataStruct;
                    ReadTokenMetaData(metaDataStruct, yOffset);
                    if (metaDataStruct.curAttenWorkRank != dstRank) {
                        continue;
                    }
                    // prof9 BuildRemoteFlagInfo
                    const uint64_t prof9Start = perfEnabled_ ? static_cast<uint64_t>(GetSystemCycle()) : 0UL;
                    const uint64_t tokenSlot = (static_cast<uint64_t>(metaDataStruct.curMicroBatchIds) * axisBS_ +
                                                metaDataStruct.curTokenBatchOffset) *
                                                   expertNumPerToken_ +
                                               metaDataStruct.curTokenTopkOffset;
                    RemoteRelayInfo relayInfo{winTokenInfoTableSize_ + tokenSlot * axisHS_ * sizeof(xType),
                                              tokenSlot * sizeof(int32_t), yOffset, dstRank};
                    if (perfEnabled_) {
                        perfCycles[8] += static_cast<uint64_t>(GetSystemCycle()) - prof9Start;
                    }
                    // prof9 BuildRemoteFlagInfo
                    // prof10 AppendRemoteFlagWrite
                    const uint64_t prof10Start = perfEnabled_ ? static_cast<uint64_t>(GetSystemCycle()) : 0UL;
                    AppendRemoteFlagWrite(batchHandle, remoteWindowAddr, relayInfo, firstFlag);
                    firstFlag = false;
                    if (perfEnabled_) {
                        perfCycles[9] += static_cast<uint64_t>(GetSystemCycle()) - prof10Start;
                    }
                    // prof10 AppendRemoteFlagWrite
                    ++preparedWqeCount;
                    if (preparedWqeCount == URMA_BATCH_WQE_CAPACITY) {
                        CommitBatch(batchHandle, perfCycles[10]);
                        preparedWqeCount = 0U;
                    }
                }

                if (preparedWqeCount != 0U) {
                    CommitBatch(batchHandle, perfCycles[10]);
                }
                // prof12 Drain
                const uint64_t prof12Start = perfEnabled_ ? static_cast<uint64_t>(GetSystemCycle()) : 0UL;
                int32_t ret = hcomm_.Drain(batchHandle);
                ascendc_assert(ret == 0, "batch drain failed, ret=%d, rankId=%u, dstRank=%u", ret, rankId_, dstRank);
                if (perfEnabled_) {
                    perfCycles[11] += static_cast<uint64_t>(GetSystemCycle()) - prof12Start;
                }
                // prof12 Drain
            }
        }

        // Keep all AIVs alive until AIV0 has drained every destination channel.
        SyncAll<true>();
        if (perfEnabled_) {
            WritePerfData(perfCycles);
        }
    }
}

} // namespace FFNToAttentionImpl
#endif // FFN_TO_ATTENTION_URMA_H
