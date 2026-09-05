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

#include <cstddef>

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
#include "../common/moe_ep_exception_dump_writer.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#include "../../common/op_kernel/moe_ep_exception_dump_writer.h"
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
static constexpr uint32_t COMBINE_CHANNEL_COUNT = 7U;
static constexpr uint32_t RECV_META_FIELDS = 5U;
static constexpr uint32_t META_TOKEN_IDX_OFFSET = 1U;
static constexpr uint32_t META_TOPK_IDX_OFFSET = 2U;
static constexpr uint32_t META_RECV_X_IDX_OFFSET = 4U;
static constexpr uint32_t HCOMM_CHANNEL_TOKEN_CAPACITY = 32368U;
// Keep one SQ entry unused. A token that would make the following token cross this limit requests a CQE and drains.
static constexpr uint32_t HCOMM_SQ_MAX_PENDING = 32767U;
// PR 111 requires each committed batch to contain fewer WQEBBs than the SQ depth.
static constexpr uint32_t HCOMM_BATCH_CAPACITY = 256;
static constexpr uint32_t HCOMM_PLAIN_WRITE_WQE_BYTES = 64;
static constexpr uint32_t HCOMM_BATCH_BUFFER_BYTES = HCOMM_BATCH_CAPACITY * HCOMM_PLAIN_WRITE_WQE_BYTES;
constexpr uint64_t UB_ALIGN = 32UL;
constexpr uint32_t SEND_DOUBLE_BUFFER_NUM = 2U;
static constexpr uint32_t META_CHUNK_TOKEN_MAX = 2048U;
static constexpr struct UrmaWqeEntry DEFAULT_WQE_CONFIG = {.odr = 5, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
static constexpr struct UrmaWqeEntry DEFAULT_CQE_WQE_CONFIG = {.odr = 5, .fence = 1, .se = 0, .cqe = 1, .inlineEn = 0};
static constexpr struct UrmaWqeEntry CHANNEL_FLAG_WQE_CONFIG = {.odr = 6, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
template <TemplateMoeEpCombineTypeClass>
class MoeEpCombine {
public:
    __aicore__ inline MoeEpCombine(){};

    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR recvSrcMetadata,
                                GM_ADDR recvRankOffsets, GM_ADDR numRecvPerExpert, GM_ADDR topkWeights,
                                GM_ADDR workspace, GM_ADDR tilingGM, TPipe *pipe, const MoeEpCombineInfo *tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void SendLocalToken(uint32_t tokenIndex, GM_ADDR dstAddr);
    __aicore__ inline void SendLocalSlot(uint32_t tokenIndex, int32_t srcTokenIdx, int32_t srcTopKIdx,
                                         GM_ADDR localDataBase, GM_ADDR localStateBase);
    __aicore__ inline void SendRemoteSlot(uint32_t tokenIndex, int32_t srcTokenIdx, int32_t srcTopKIdx,
                                          GM_ADDR remoteDataBase, GM_ADDR remoteStateBase, uint64_t tokenBytes);
    __aicore__ inline void SendChannelFlag(uint32_t dstRank, uint32_t channelIndex);
    __aicore__ inline void BeginPreparedWrites(uint32_t dstRank, uint32_t channelIndex);
    template <auto const &config>
    __aicore__ inline void PrepareWrite(GM_ADDR dst, GM_ADDR src, uint64_t len);
    __aicore__ inline void FlushPreparedWrites(bool keepHandle = false);
    __aicore__ inline void BuildAddressTable(uint32_t entryStart, uint32_t entryEnd);
    __aicore__ inline uint32_t GetAddressTableCount(uint32_t targetRank);
    __aicore__ inline void SendAddressTableRange(uint32_t targetRank, uint32_t entryStart, uint32_t entryEnd,
                                                 uint32_t channelIndex);
    __aicore__ inline void SendPhaseExpertToToken();
    __aicore__ inline void GetCoreAssignment(uint32_t totalBlocks, uint32_t &targetRank, uint32_t &coreIndexInGroup,
                                             uint32_t &groupSize);

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
    TPipe *tpipe_{nullptr};
    const MoeEpCombineInfo *tilingData_{nullptr};
    __gm__ Mc2Aclnn::MoeCommContext *mc2Context_{nullptr};
    MoeEpExceptionDump::MoeEpCoreDiagWriter diagWriter_;

    uint32_t rankId_{0};
    uint32_t epWorldSize_{0};
    uint32_t channelsPerRank_{1};
    uint32_t combineChannelCount_{1};
    uint32_t numMaxTokensPerRank_{0};
    uint32_t topK_{0};
    uint32_t axisH_{0};
    uint32_t hAlignSize_{0}; // UB对齐后的hidden size
    uint64_t combineStateWinOffset_{0};
    uint64_t combineDataWinOffset_{0};

    uint32_t XTypeAlign32Size_{0};
    uint32_t perSlotBytes_{0};
    uint64_t actualA_{0};
    uint64_t recvCapacity_{0};
    uint32_t aivNum_{0};
    uint32_t metadataChunkTokens_{1};

    GlobalTensor<XType> xGm_;
    GlobalTensor<int32_t> recvSrcMetadataGm_;
    GlobalTensor<int32_t> recvRankOffsetsGm_;
    GlobalTensor<float> topkWeightsGm_;

    LocalTensor<uint32_t> statusTensor_;
    LocalTensor<int32_t> rankOffsetsTensor_;
    LocalTensor<uint8_t> hcommTensor_;
    LocalTensor<uint8_t> hcommBatchTensor_;

    TBuf<> readStateBuf_;
    TBuf<> rankOffsetsBuf_;
    TBuf<> hcommBuf_;
    TBuf<TPosition::VECOUT> hcommBatchBuf_;

    TBuf<> metadataBuf_; // 发送阶段recvSrcMetadata的UB缓冲，批量搬运避免GetValue
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_; // 数据队列
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> weightQueue_;
    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_; // 通信上下文
    using HcommBatchHandle = AscendC::BatchHandle<AscendC::ChannelHandle>;

    GM_ADDR winRankAddr_[Mc2Aclnn::HCCL_MAX_RANK_SIZE];
    uint64_t hcommHandle_[Mc2Aclnn::HCCL_MAX_RANK_SIZE];
    GM_ADDR flagSourceWinAddr_{nullptr};
    uint32_t aivId_{0};
    HcommBatchHandle activeBatchHandle_{};
    uint64_t activeBatchChannel_{0};
    uint32_t preparedWriteCount_{0};
    uint32_t sqWriteCount_{0};
    bool activeBatchInitialized_{false};
};

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx,
                                                                        GM_ADDR recvSrcMetadata,
                                                                        GM_ADDR recvRankOffsets,
                                                                        GM_ADDR numRecvPerExpert, GM_ADDR topkWeights,
                                                                        GM_ADDR workspace, GM_ADDR tilingGM,
                                                                        TPipe *pipe, const MoeEpCombineInfo *tilingData)
{
    (void)topkIdx;
    (void)numRecvPerExpert;
    (void)workspace;
    tpipe_ = pipe;
    tilingData_ = tilingData;
    aivId_ = GetBlockIdx();
    epWorldSize_ = tilingData_->cfg.epWorldSize;
    numMaxTokensPerRank_ = tilingData_->cfg.numMaxTokensPerRank;
    topK_ = tilingData_->cfg.topK;
    axisH_ = tilingData_->cfg.hidden;
    perSlotBytes_ = tilingData_->cfg.perSlotBytes;
    aivNum_ = tilingData_->aivNum;
    recvCapacity_ = tilingData_->recvCapacity;
    hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN; // UB 32字节对齐
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);
    tpipe_->InitBuffer(hcommBatchBuf_, HCOMM_BATCH_BUFFER_BYTES);
    hcommBatchTensor_ = hcommBatchBuf_.Get<uint8_t>();
    Duplicate<uint8_t>(hcommBatchTensor_, 0U, HCOMM_BATCH_BUFFER_BYTES);
    SyncFunc<AscendC::HardEvent::V_S>();

    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::MoeCommContext *>(context);
    rankId_ = mc2Context_->epRankId;
    constexpr size_t metadataOffset =
        offsetof(MoeEpCombineTilingData, moeEpCombineInfo) + offsetof(MoeEpCombineInfo, dumpMetadata);
    MoeEpExceptionDump::WriteMetadata(context, tilingGM + metadataOffset);
    diagWriter_.Init(context, MOE_EP_CORE_DIAG_COMBINE, tpipe_);
    channelsPerRank_ = mc2Context_->channelsPerRank;
    if (channelsPerRank_ == 0U) {
        channelsPerRank_ = 1U;
    }

    const uint64_t tokenSlotCount = static_cast<uint64_t>(numMaxTokensPerRank_) * topK_;
    combineChannelCount_ =
        static_cast<uint32_t>(Ceil(tokenSlotCount, static_cast<uint64_t>(HCOMM_CHANNEL_TOKEN_CAPACITY)));
    combineChannelCount_ = combineChannelCount_ == 0U ? 1U : combineChannelCount_;
    combineChannelCount_ = combineChannelCount_ < channelsPerRank_ ? combineChannelCount_ : channelsPerRank_;
    combineChannelCount_ = combineChannelCount_ < COMBINE_CHANNEL_COUNT ? combineChannelCount_ : COMBINE_CHANNEL_COUNT;

    for (uint32_t i = 0; i < epWorldSize_; ++i) {
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
    }
    uint32_t handleCount = epWorldSize_ * channelsPerRank_;
    for (uint32_t i = 0; i < handleCount; ++i) {
        hcommHandle_[i] = mc2Context_->hcommHandle[i];
    }

    combineStateWinOffset_ = tilingData->combineStateWinOffset;
    combineDataWinOffset_ = tilingData->combineDataWinOffset;
    flagSourceWinAddr_ = GetUrmaStateAddrByRankId(rankId_, tilingData->combineFlagSourceWinOffset);

    xGm_.SetGlobalBuffer((__gm__ XType *)x);
    recvSrcMetadataGm_.SetGlobalBuffer((__gm__ int32_t *)recvSrcMetadata);
    recvRankOffsetsGm_.SetGlobalBuffer((__gm__ int32_t *)recvRankOffsets);

    uint32_t rankOffsetsBytes = Ceil(static_cast<uint64_t>(epWorldSize_ + 1U) * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(rankOffsetsBuf_, rankOffsetsBytes);
    rankOffsetsTensor_ = rankOffsetsBuf_.Get<int32_t>();
    DataCopyExtParams rankOffsetsCopyParams{1U, static_cast<uint32_t>((epWorldSize_ + 1U) * sizeof(int32_t)), 0U, 0U,
                                            0U};
    DataCopyPadExtParams<int32_t> rankOffsetsPadParams{false, 0U, 0U, 0U};
    DataCopyPad(rankOffsetsTensor_, recvRankOffsetsGm_, rankOffsetsCopyParams, rankOffsetsPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    int32_t actualASigned = rankOffsetsTensor_.GetValue(epWorldSize_);
    actualA_ = actualASigned > 0 ? static_cast<uint64_t>(actualASigned) : 0U;
    actualA_ = actualA_ < recvCapacity_ ? actualA_ : recvCapacity_;

    XTypeAlign32Size_ = hAlignSize_;
    if constexpr (HasTopkWeight == 1) {
        topkWeightsGm_.SetGlobalBuffer((__gm__ float *)topkWeights);
    }
    tpipe_->InitBuffer(xQueue_, SEND_DOUBLE_BUFFER_NUM, XTypeAlign32Size_);
    if constexpr (HasTopkWeight == 1) {
        tpipe_->InitBuffer(weightQueue_, SEND_DOUBLE_BUFFER_NUM, UB_ALIGN);
    }
    metadataChunkTokens_ = actualA_ < META_CHUNK_TOKEN_MAX ? static_cast<uint32_t>(actualA_) : META_CHUNK_TOKEN_MAX;
    metadataChunkTokens_ = metadataChunkTokens_ == 0U ? 1U : metadataChunkTokens_;
    uint32_t metadataChunkBytes =
        Ceil(static_cast<uint64_t>(metadataChunkTokens_) * RECV_META_FIELDS * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(metadataBuf_, metadataChunkBytes);
    tpipe_->InitBuffer(readStateBuf_, WIN_ADDR_ALIGN);
    statusTensor_ = readStateBuf_.Get<uint32_t>();
    diagWriter_.RunPosRecord(MOE_EP_COMBINE_RUN_POS_INIT_DONE);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::BuildAddressTable(uint32_t entryStart,
                                                                                     uint32_t entryEnd)
{
    // Sorted metadata plus recv_rank_offsets forms the implicit address table. AIV0 initializes the persistent
    // non-inline flag source while the AIV group assigned to the local rank copies its part of the local range.
    if (aivId_ == 0U) {
        LocalTensor<uint64_t> flagTensor = statusTensor_.ReinterpretCast<uint64_t>();
        Duplicate<uint64_t>(flagTensor, 1U, WIN_ADDR_ALIGN / sizeof(uint64_t));
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        GlobalTensor<uint64_t> flagSource;
        flagSource.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(flagSourceWinAddr_));
        DataCopy(flagSource, flagTensor, WIN_ADDR_ALIGN / sizeof(uint64_t));
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }

    uint64_t localBegin = static_cast<uint32_t>(rankOffsetsTensor_.GetValue(rankId_));
    uint64_t scanStart = localBegin + entryStart;
    uint64_t scanEnd = localBegin + entryEnd;

    constexpr uint32_t metaBytesPerToken = RECV_META_FIELDS * sizeof(int32_t);
    LocalTensor<int32_t> metadataLocal = metadataBuf_.Get<int32_t>();
    const DataCopyPadExtParams<int32_t> metaPadParams{false, 0U, 0U, 0U};
    GM_ADDR localDataBase = GetUrmaWinAddrByRankId(rankId_, combineDataWinOffset_);
    GM_ADDR localStateBase = GetUrmaStateAddrByRankId(rankId_, combineStateWinOffset_);

    for (uint64_t chunkStart = scanStart; chunkStart < scanEnd; chunkStart += metadataChunkTokens_) {
        uint64_t chunkEnd = chunkStart + metadataChunkTokens_ > scanEnd ? scanEnd : chunkStart + metadataChunkTokens_;
        uint32_t chunkCount = static_cast<uint32_t>(chunkEnd - chunkStart);
        DataCopyExtParams metaCopyParams{1U, chunkCount * metaBytesPerToken, 0U, 0U, 0U};
        DataCopyPad(metadataLocal, recvSrcMetadataGm_[chunkStart * RECV_META_FIELDS], metaCopyParams, metaPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        for (uint32_t i = 0U; i < chunkCount; ++i) {
            uint32_t metaOffset = i * RECV_META_FIELDS;
            int32_t srcTokenIdx = metadataLocal.GetValue(metaOffset + META_TOKEN_IDX_OFFSET);
            int32_t srcTopKIdx = metadataLocal.GetValue(metaOffset + META_TOPK_IDX_OFFSET);
            int32_t recvXIdx = metadataLocal.GetValue(metaOffset + META_RECV_X_IDX_OFFSET);
            SendLocalSlot(static_cast<uint32_t>(recvXIdx), srcTokenIdx, srcTopKIdx, localDataBase, localStateBase);
        }
        SyncFunc<AscendC::HardEvent::S_MTE2>();
    }
    if (scanStart < scanEnd) {
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
}
template <TemplateMoeEpCombineTypeClass>
__aicore__ inline uint32_t MoeEpCombine<TemplateMoeEpCombineTypeFunc>::GetAddressTableCount(uint32_t targetRank)
{
    uint32_t rankBegin = static_cast<uint32_t>(rankOffsetsTensor_.GetValue(targetRank));
    uint32_t rankEnd = static_cast<uint32_t>(rankOffsetsTensor_.GetValue(targetRank + 1U));
    return rankEnd - rankBegin;
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendAddressTableRange(uint32_t targetRank,
                                                                                         uint32_t entryStart,
                                                                                         uint32_t entryEnd,
                                                                                         uint32_t channelIndex)
{
    // entryStart/entryEnd are relative to the target rank's range in the sorted metadata.
    if (entryStart >= entryEnd) {
        return;
    }
    if (targetRank == rankId_) {
        return;
    }
    BeginPreparedWrites(targetRank, channelIndex);

    constexpr uint32_t metaBytesPerToken = RECV_META_FIELDS * sizeof(int32_t);
    LocalTensor<int32_t> metadataLocal = metadataBuf_.Get<int32_t>();
    const DataCopyPadExtParams<int32_t> padParams{false, 0U, 0U, 0U};
    GM_ADDR remoteDataBase = GetUrmaWinAddrByRankId(targetRank, combineDataWinOffset_);
    GM_ADDR remoteStateBase = GetUrmaStateAddrByRankId(targetRank, combineStateWinOffset_);
    uint64_t tokenBytes = static_cast<uint64_t>(axisH_) * sizeof(XType);
    uint64_t rankBegin = static_cast<uint32_t>(rankOffsetsTensor_.GetValue(targetRank));
    uint64_t rangeBegin = rankBegin + entryStart;
    uint64_t rangeEnd = rankBegin + entryEnd;

    for (uint64_t chunkStart = rangeBegin; chunkStart < rangeEnd; chunkStart += metadataChunkTokens_) {
        uint64_t chunkEnd = chunkStart + metadataChunkTokens_ > rangeEnd ? rangeEnd : chunkStart + metadataChunkTokens_;
        uint32_t chunkCount = static_cast<uint32_t>(chunkEnd - chunkStart);
        DataCopyExtParams copyParams{1U, chunkCount * metaBytesPerToken, 0U, 0U, 0U};
        DataCopyPad(metadataLocal, recvSrcMetadataGm_[chunkStart * RECV_META_FIELDS], copyParams, padParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        for (uint32_t i = 0U; i < chunkCount; ++i) {
            uint32_t metaOffset = i * RECV_META_FIELDS;
            int32_t srcTokenIdx = metadataLocal.GetValue(metaOffset + META_TOKEN_IDX_OFFSET);
            int32_t srcTopKIdx = metadataLocal.GetValue(metaOffset + META_TOPK_IDX_OFFSET);
            int32_t recvXIdx = metadataLocal.GetValue(metaOffset + META_RECV_X_IDX_OFFSET);
            SendRemoteSlot(static_cast<uint32_t>(recvXIdx), srcTokenIdx, srcTopKIdx, remoteDataBase, remoteStateBase,
                           tokenBytes);
        }
        SyncFunc<AscendC::HardEvent::S_MTE2>();
    }
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
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::FlushPreparedWrites(bool keepHandle)
{
    if (preparedWriteCount_ != 0) {
        int32_t ret = hcomm_.BatchCommit(activeBatchHandle_);
        (void)ret;
        preparedWriteCount_ = 0;
    }
    if (!keepHandle) {
        activeBatchHandle_ = {};
        activeBatchChannel_ = 0;
        activeBatchInitialized_ = false;
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::BeginPreparedWrites(uint32_t dstRank,
                                                                                       uint32_t channelIndex)
{
    uint64_t commHandle = GetCommHandle(dstRank, channelIndex);
    if (activeBatchInitialized_ && activeBatchChannel_ == commHandle) {
        return;
    }
    if (activeBatchInitialized_) {
        FlushPreparedWrites();
    }
    activeBatchHandle_ =
        hcomm_.MakeBatchHandle(commHandle, hcommBatchTensor_, HCOMM_BATCH_BUFFER_BYTES, winRankAddr_[dstRank]);
    activeBatchChannel_ = commHandle;
    sqWriteCount_ = 0;
    activeBatchInitialized_ = true;
}

template <TemplateMoeEpCombineTypeClass>
template <auto const &config>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::PrepareWrite(GM_ADDR dst, GM_ADDR src, uint64_t len)
{
    if (preparedWriteCount_ == HCOMM_BATCH_CAPACITY) {
        // BatchCommit resets the WQE count in the handle; keep using it for this channel as PR 10309 does.
        FlushPreparedWrites(true);
    }
    int32_t ret = hcomm_.WriteNbi<config>(activeBatchHandle_, dst, src, len);
    (void)ret;
    ++preparedWriteCount_;
    ++sqWriteCount_;
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendLocalSlot(
    uint32_t tokenIndex, int32_t srcTokenIdx, int32_t srcTopKIdx, GM_ADDR localDataBase, GM_ADDR localStateBase)
{
    uint64_t sendTokenOffset = (static_cast<uint64_t>(srcTokenIdx) * topK_ + srcTopKIdx) * perSlotBytes_;
    uint64_t recvStateOffset = (srcTokenIdx * topK_ + srcTopKIdx) * WIN_ADDR_ALIGN;

    SendLocalToken(tokenIndex, localDataBase + sendTokenOffset);
    if constexpr (HasTopkWeight == 1) {
        GlobalTensor<float> state;
        state.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(localStateBase + recvStateOffset));
        LocalTensor<float> weightTensor = weightQueue_.AllocTensor<float>();
        DataCopyParams weightCopyParams{1U, static_cast<uint16_t>(sizeof(float)), 0U, 0U};
        DataCopyPadParams weightPadParams{false, 0U, 0U, 0U};
        DataCopyPad(weightTensor, topkWeightsGm_[tokenIndex], weightCopyParams, weightPadParams);
        weightQueue_.EnQue(weightTensor);
        weightTensor = weightQueue_.DeQue<float>();
        DataCopyPad(state, weightTensor, weightCopyParams);
        weightQueue_.FreeTensor<float>(weightTensor);
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendRemoteSlot(
    uint32_t tokenIndex, int32_t srcTokenIdx, int32_t srcTopKIdx, GM_ADDR remoteDataBase, GM_ADDR remoteStateBase,
    uint64_t tokenBytes)
{
    uint64_t sendTokenOffset = (static_cast<uint64_t>(srcTokenIdx) * topK_ + srcTopKIdx) * perSlotBytes_;
    uint64_t recvStateOffset = (srcTokenIdx * topK_ + srcTopKIdx) * WIN_ADDR_ALIGN;
    GM_ADDR tokenAddr = (GM_ADDR)xGm_.GetPhyAddr(tokenIndex * axisH_);
    constexpr uint32_t writesPerToken = HasTopkWeight == 1 ? 2U : 1U;
    bool drainAfterToken = sqWriteCount_ + 2U * writesPerToken > HCOMM_SQ_MAX_PENDING;
    if constexpr (HasTopkWeight == 1) {
        PrepareWrite<DEFAULT_WQE_CONFIG>(remoteDataBase + sendTokenOffset, tokenAddr, tokenBytes);
    } else {
        if (drainAfterToken) {
            PrepareWrite<DEFAULT_CQE_WQE_CONFIG>(remoteDataBase + sendTokenOffset, tokenAddr, tokenBytes);
        } else {
            PrepareWrite<DEFAULT_WQE_CONFIG>(remoteDataBase + sendTokenOffset, tokenAddr, tokenBytes);
        }
    }
    if constexpr (HasTopkWeight == 1) {
        GM_ADDR weightAddr = reinterpret_cast<GM_ADDR>(topkWeightsGm_.GetPhyAddr(tokenIndex));
        if (drainAfterToken) {
            PrepareWrite<DEFAULT_CQE_WQE_CONFIG>(remoteStateBase + recvStateOffset, weightAddr, sizeof(float));
        } else {
            PrepareWrite<DEFAULT_WQE_CONFIG>(remoteStateBase + recvStateOffset, weightAddr, sizeof(float));
        }
    }
    if (drainAfterToken) {
        FlushPreparedWrites(true);
        int32_t ret = hcomm_.Drain(activeBatchChannel_);
        (void)ret;
        sqWriteCount_ = 0U;
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendChannelFlag(uint32_t dstRank,
                                                                                   uint32_t channelIndex)
{
    uint64_t flagIndex = static_cast<uint64_t>(rankId_) * combineChannelCount_ + channelIndex;
    uint64_t flagOffset =
        static_cast<uint64_t>(numMaxTokensPerRank_) * topK_ * WIN_ADDR_ALIGN + flagIndex * WIN_ADDR_ALIGN;
    GM_ADDR flagAddr = GetUrmaStateAddrByRankId(dstRank, combineStateWinOffset_) + flagOffset;
    if (dstRank == rankId_) {
        GlobalTensor<uint64_t> localFlag;
        localFlag.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(flagAddr));
        localFlag.SetValue(0, 1U);
        DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(localFlag);
        return;
    }
    BeginPreparedWrites(dstRank, channelIndex);
    PrepareWrite<CHANNEL_FLAG_WQE_CONFIG>(flagAddr, flagSourceWinAddr_, WIN_ADDR_ALIGN);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::GetCoreAssignment(uint32_t totalBlocks,
                                                                                     uint32_t &targetRank,
                                                                                     uint32_t &coreIndexInGroup,
                                                                                     uint32_t &groupSize)
{
    uint32_t maxChannelAivNum = epWorldSize_ * combineChannelCount_;
    bool assignRemainingToLocal = totalBlocks > maxChannelAivNum;
    if (assignRemainingToLocal) {
        // Put all remote communication AIVs first, then give every remaining AIV to local copies.
        uint32_t remoteAivNum = (epWorldSize_ - 1U) * combineChannelCount_;
        if (aivId_ < remoteAivNum) {
            uint32_t remoteRankIndex = aivId_ / combineChannelCount_;
            targetRank = remoteRankIndex < rankId_ ? remoteRankIndex : remoteRankIndex + 1U;
            coreIndexInGroup = aivId_ % combineChannelCount_;
            groupSize = combineChannelCount_;
        } else {
            targetRank = rankId_;
            coreIndexInGroup = aivId_ - remoteAivNum;
            groupSize = totalBlocks - remoteAivNum;
        }
        return;
    }

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
    if (epWorldSize_ == 0 || aivNum_ == 0) {
        return;
    }
    uint32_t activeAivNum = aivNum_;

    activeBatchHandle_ = {};
    activeBatchChannel_ = 0;
    preparedWriteCount_ = 0;
    sqWriteCount_ = 0;
    activeBatchInitialized_ = false;

    bool splitRankTokens = activeAivNum >= epWorldSize_;
    uint32_t targetRank = epWorldSize_;
    uint32_t coreIndexInGroup = 0;
    uint32_t groupSize = 1;
    if (splitRankTokens && aivId_ < activeAivNum) {
        GetCoreAssignment(activeAivNum, targetRank, coreIndexInGroup, groupSize);
    }
    bool sendsTokens = actualA_ != 0 && aivId_ < activeAivNum;
    uint32_t entryStart = 0U;
    uint32_t entryEnd = 0U;
    if (splitRankTokens) {
        uint32_t entryCount = GetAddressTableCount(targetRank);
        uint32_t entriesPerCore = entryCount / groupSize;
        uint32_t remainder = entryCount % groupSize;
        entryStart =
            coreIndexInGroup * entriesPerCore + ((coreIndexInGroup < remainder) ? coreIndexInGroup : remainder);
        entryEnd = entryStart + entriesPerCore + ((coreIndexInGroup < remainder) ? 1U : 0U);
    } else if (rankId_ % activeAivNum == aivId_) {
        entryEnd = GetAddressTableCount(rankId_);
    }

    // Local MTE copies and remote URMA writes use disjoint rank ranges and can run concurrently on their assigned
    // AIVs. BuildAddressTable is called by every AIV so AIV0 always initializes the shared flag source.
    if (!splitRankTokens || targetRank == rankId_) {
        BuildAddressTable(entryStart, entryEnd);
    } else {
        BuildAddressTable(0U, 0U);
    }

    if (sendsTokens) {
        if (splitRankTokens) {
            if (targetRank != rankId_) {
                SendAddressTableRange(targetRank, entryStart, entryEnd, coreIndexInGroup);
            }
        } else {
            for (uint32_t rank = aivId_; rank < epWorldSize_; rank += activeAivNum) {
                if (rank == rankId_) {
                    continue;
                }
                uint32_t entryCount = GetAddressTableCount(rank);
                SendAddressTableRange(rank, 0U, entryCount, 0U);
            }
        }
    }

    // Submit the last partial remote batch so it can progress while slower local copies finish. The barrier makes
    // every local write visible before the local completion flags are published. Remote flags remain ordered after
    // their data writes on the same communication channel.
    FlushPreparedWrites(true);
    SyncAll<true>();

    bool publishesChannelFlag = aivId_ < activeAivNum && (!splitRankTokens || coreIndexInGroup < combineChannelCount_);
    if (publishesChannelFlag) {
        if (splitRankTokens) {
            SendChannelFlag(targetRank, coreIndexInGroup);
        } else {
            for (uint32_t dstRank = aivId_; dstRank < epWorldSize_; dstRank += activeAivNum) {
                SendChannelFlag(dstRank, 0U);
            }
        }
    }
    FlushPreparedWrites();
    if (sendsTokens) {
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(recvSrcMetadataGm_);
    }
    diagWriter_.RunPosRecord(MOE_EP_COMBINE_RUN_POS_URMA_REQUESTS_ISSUE_DONE);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::Process()
{
    SendPhaseExpertToToken();
}

#endif

} // namespace MoeEpCombineImpl

#endif // MOE_EP_COMBINE_H
