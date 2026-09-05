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
static constexpr uint32_t COMBINE_CHANNEL_COUNT = 1U;
static constexpr uint32_t RECV_META_FIELDS = 5U;
static constexpr uint32_t META_SRC_RANK_OFFSET = 0U;
static constexpr uint32_t META_TOKEN_IDX_OFFSET = 1U;
static constexpr uint32_t META_TOPK_IDX_OFFSET = 2U;
static constexpr uint32_t META_SLOT_IDX_OFFSET = 3U;
static constexpr uint32_t META_RECV_X_IDX_OFFSET = 4U;
// Keep one SQ entry unused. A token that would make the following token cross this limit requests a CQE and drains.
static constexpr uint32_t HCOMM_SQ_MAX_PENDING = 32767U;
// PR 111 requires each committed batch to contain fewer WQEBBs than the SQ depth.
static constexpr uint32_t HCOMM_BATCH_CAPACITY = 256;
static constexpr uint32_t HCOMM_PLAIN_WRITE_WQE_BYTES = 64;
static constexpr uint32_t HCOMM_BATCH_BUFFER_BYTES = HCOMM_BATCH_CAPACITY * HCOMM_PLAIN_WRITE_WQE_BYTES;
constexpr uint64_t UB_ALIGN = 32UL;
constexpr uint64_t ALIGNED_LEN_256 = 256UL;
constexpr uint32_t SEND_DOUBLE_BUFFER_NUM = 2U;
static constexpr uint32_t ADDRESS_ENTRY_UB_BYTES = UB_ALIGN;
static constexpr uint32_t ADDRESS_ENTRY_GM_BYTES = WIN_ADDR_ALIGN;
static constexpr uint32_t ADDRESS_TABLE_BUFFER_BYTES = 40U * 1024U;
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
                                GM_ADDR workspace, GM_ADDR tilingGM, TPipe *pipe,
                                const MoeEpCombineInfo *tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void SendLocalToken(uint32_t tokenIndex, GM_ADDR dstAddr);
    // TODO(delete): legacy address-table path only; the direct metadata path uses SendLocalMetadataSlot().
    __aicore__ inline void SendLocalSlot(uint32_t tokenIndex, int32_t srcTokenIdx, int32_t srcTopKIdx, uint32_t weight,
                                         GM_ADDR localDataBase, GM_ADDR localStateBase);
    // TODO(delete): legacy address-table path only; the direct metadata path uses SendRemoteMetadataSlot().
    __aicore__ inline void SendRemoteSlot(uint32_t tokenIndex, int32_t srcTokenIdx, int32_t srcTopKIdx, uint32_t weight,
                                          GM_ADDR remoteDataBase, GM_ADDR remoteStateBase, uint64_t tokenBytes);
    __aicore__ inline void SendChannelFlag(uint32_t dstRank, uint32_t channelIndex);
    __aicore__ inline void BeginPreparedWrites(uint32_t dstRank, uint32_t channelIndex);
    __aicore__ inline void InitFlagSource();
    template <auto const &config>
    __aicore__ inline void PrepareWrite(GM_ADDR dst, GM_ADDR src, uint64_t len);
    __aicore__ inline void FlushPreparedWrites(bool keepHandle = false);
    // TODO(delete): legacy address-table construction path; no longer called by the direct metadata path.
    __aicore__ inline void BuildAddressTable();
    // TODO(delete): legacy address-table path only.
    __aicore__ inline uint32_t GetAddressTableCount(uint32_t targetRank);
    // TODO(delete): legacy address-table path only.
    __aicore__ inline void SendAddressTableRange(uint32_t targetRank, uint32_t entryStart, uint32_t entryEnd,
                                                 uint32_t channelIndex);
    // TODO(delete): legacy address-table send entry; its Process() call site is intentionally commented out.
    __aicore__ inline void SendPhaseExpertToToken();
    // TODO(delete): legacy address-table path only.
    __aicore__ inline void GetCoreAssignment(uint32_t totalBlocks, uint32_t &targetRank, uint32_t &coreIndexInGroup,
                                             uint32_t &groupSize);
    // TODO(delete): rank offsets replaced metadata rank probing.
    __aicore__ inline int32_t ReadMetadataRank(uint64_t metadataRow);
    // TODO(delete): rank offsets replaced metadata lower-bound lookup.
    __aicore__ inline uint64_t LowerBoundSrcRank(uint32_t targetRank);
    __aicore__ inline void SplitRange(uint64_t rangeBegin, uint64_t rangeEnd, uint32_t coreCount, uint32_t coreIndex,
                                      uint64_t &coreBegin, uint64_t &coreEnd);
    // TODO(delete): the direct metadata hot path trusts the Dispatch Epilogue metadata contract.
    __aicore__ inline bool IsMetadataRowValid(uint32_t targetRank, int32_t srcRank, int32_t srcTokenIdx,
                                              int32_t srcTopKIdx, int32_t recvXIdx);
    __aicore__ inline void SendLocalWeight(uint32_t recvXIdx, GM_ADDR dstAddr);
    __aicore__ inline void SendLocalMetadataSlot(uint32_t recvXIdx, int32_t srcTokenIdx, int32_t srcTopKIdx,
                                                 GM_ADDR localDataBase, GM_ADDR localStateBase);
    __aicore__ inline void SendRemoteMetadataSlot(uint32_t recvXIdx, int32_t srcTokenIdx, int32_t srcTopKIdx,
                                                  GM_ADDR remoteDataBase, GM_ADDR remoteStateBase, uint64_t tokenBytes);
    __aicore__ inline bool ProcessLocalMetadataRange(uint64_t rangeBegin, uint64_t rangeEnd);
    __aicore__ inline void ProcessRemoteMetadataRange(uint32_t targetRank, uint64_t rangeBegin, uint64_t rangeEnd);
    __aicore__ inline void SendPhaseDirectFromMetadata();

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
    // TODO(delete): legacy address-table workspace accessor.
    __aicore__ inline GM_ADDR GetLocalSendDataWorkspaceAddr(const int32_t rankId)
    {
        return combineSendDataWorkspaceAddr_ + sendDataWorkspaceSizePerRank_ * rankId;
    }
    // TODO(delete): only used by the commented legacy actualA reduction.
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

    uint32_t hWeightAlignSize_{0}; // token+weight对齐后的hidden size
    uint32_t XTypeAlign32Size_{0};
    uint32_t perSlotBytes_{0};
    uint64_t actualA_{0};
    uint64_t recvCapacity_{0};
    uint32_t aivNum_{0};
    uint32_t localmoeNum_{0};
    uint32_t metadataChunkTokens_{1};
    uint64_t sendDataWorkspaceSizePerRank_{0};

    GlobalTensor<XType> xGm_;
    GlobalTensor<int32_t> recvSrcMetadataGm_;
    GlobalTensor<int32_t> recvRankOffsetsGm_;
    GlobalTensor<int64_t> numRecvPerExpertGm_;
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
    TBuf<> weightsBuf_;
    TBuf<> addressTableBuf_;
    TBuf<> rankInfoBuf_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_; // 数据队列
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> weightQueue_;
    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_; // 通信上下文
    using HcommBatchHandle = AscendC::BatchHandle<AscendC::ChannelHandle>;

    GM_ADDR winRankAddr_[Mc2Aclnn::HCCL_MAX_RANK_SIZE];
    uint64_t hcommHandle_[Mc2Aclnn::HCCL_MAX_RANK_SIZE];
    GM_ADDR combineSendDataWorkspaceAddr_{nullptr};
    GM_ADDR flagWorkspaceAddr_{nullptr};
    GM_ADDR rankCountWorkspaceAddr_{nullptr};
    GM_ADDR perCoreRankCountWorkspaceAddr_{nullptr};
    uint64_t perCoreRankCountStride_{0};
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
                                                                        GM_ADDR numRecvPerExpert,
                                                                        GM_ADDR topkWeights, GM_ADDR workspace,
                                                                        GM_ADDR tilingGM, TPipe *pipe,
                                                                        const MoeEpCombineInfo *tilingData)
{
    tpipe_ = pipe;
    tilingData_ = tilingData;
    aivId_ = GetBlockIdx();
    combineSendDataWorkspaceAddr_ = workspace;
    epWorldSize_ = tilingData_->cfg.epWorldSize;
    numMaxTokensPerRank_ = tilingData_->cfg.numMaxTokensPerRank;
    topK_ = tilingData_->cfg.topK;
    axisH_ = tilingData_->cfg.hidden;
    perSlotBytes_ = tilingData_->cfg.perSlotBytes;
    aivNum_ = tilingData_->aivNum;
    localmoeNum_ = tilingData_->cfg.numLocalExperts;
    recvCapacity_ = tilingData_->recvCapacity;
    sendDataWorkspaceSizePerRank_ = tilingData->sendDataWorkspaceSizePerRank;
    // TODO(delete): legacy address-table workspace placed flag storage after every rank's send table.
    // flagWorkspaceAddr_ = combineSendDataWorkspaceAddr_ + epWorldSize_ * sendDataWorkspaceSizePerRank_ +
    //                      static_cast<uint64_t>(aivId_) * WIN_ADDR_ALIGN;
    flagWorkspaceAddr_ = combineSendDataWorkspaceAddr_ + static_cast<uint64_t>(aivId_) * WIN_ADDR_ALIGN;
    rankCountWorkspaceAddr_ = combineSendDataWorkspaceAddr_ + epWorldSize_ * sendDataWorkspaceSizePerRank_ +
                              static_cast<uint64_t>(aivNum_) * WIN_ADDR_ALIGN;
    perCoreRankCountStride_ = Ceil(static_cast<uint64_t>(epWorldSize_) * UB_ALIGN, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    perCoreRankCountWorkspaceAddr_ = rankCountWorkspaceAddr_ + static_cast<uint64_t>(epWorldSize_) * WIN_ADDR_ALIGN;
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
    constexpr size_t metadataOffset =
        offsetof(MoeEpCombineTilingData, moeEpCombineInfo) + offsetof(MoeEpCombineInfo, dumpMetadata);
    MoeEpExceptionDump::WriteMetadata(context, tilingGM + metadataOffset);
    diagWriter_.Init(context, MOE_EP_CORE_DIAG_COMBINE, tpipe_);
    channelsPerRank_ = mc2Context_->channelsPerRank;
    if (channelsPerRank_ == 0 || (epWorldSize_ > 0 && channelsPerRank_ > Mc2Aclnn::HCCL_MAX_RANK_SIZE / epWorldSize_)) {
        channelsPerRank_ = 1;
    }

    combineChannelCount_ = COMBINE_CHANNEL_COUNT;

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
    recvSrcMetadataGm_.SetGlobalBuffer((__gm__ int32_t *)recvSrcMetadata);
    recvRankOffsetsGm_.SetGlobalBuffer((__gm__ int32_t *)recvRankOffsets);
    numRecvPerExpertGm_.SetGlobalBuffer((__gm__ int64_t *)numRecvPerExpert);

    uint32_t rankOffsetsBytes =
        Ceil(static_cast<uint64_t>(epWorldSize_ + 1U) * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(rankOffsetsBuf_, rankOffsetsBytes);
    rankOffsetsTensor_ = rankOffsetsBuf_.Get<int32_t>();
    DataCopyExtParams rankOffsetsCopyParams{
        1U, static_cast<uint32_t>((epWorldSize_ + 1U) * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> rankOffsetsPadParams{false, 0U, 0U, 0U};
    DataCopyPad(rankOffsetsTensor_, recvRankOffsetsGm_, rankOffsetsCopyParams, rankOffsetsPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    int32_t actualASigned = rankOffsetsTensor_.GetValue(epWorldSize_);

    // TODO(delete): recv_rank_offsets[epWorldSize] replaces the legacy per-core reduction below.
    /*
    uint32_t numRecvBytes = Ceil(localmoeNum_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
    uint32_t reduceTmpBytes = ReduceSumWorkNeedSize(localmoeNum_, sizeof(int64_t)) * sizeof(int64_t);
    TBuf<TPosition::VECCALC> numRecvBuf;
    tpipe_->InitBuffer(numRecvBuf, numRecvBytes + reduceTmpBytes + UB_ALIGN);
    LocalTensor<int64_t> numRecvLocal = numRecvBuf.Get<int64_t>();
    DataCopyExtParams numRecvCopyParams{1U, static_cast<uint32_t>(localmoeNum_ * sizeof(int64_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int64_t> numRecvPadParams{false, 0U, 0U, 0U};
    DataCopyPad(numRecvLocal, numRecvPerExpertGm_, numRecvCopyParams, numRecvPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int64_t> reduceTmp = numRecvLocal[numRecvBytes / sizeof(int64_t)];
    LocalTensor<int64_t> sumOut = reduceTmp[reduceTmpBytes / sizeof(int64_t)];
    ReduceSum<int64_t>(sumOut, numRecvLocal, reduceTmp, static_cast<int32_t>(localmoeNum_));
    SyncFunc<AscendC::HardEvent::V_S>();
    int64_t actualASigned = sumOut.GetValue(0);
    */
    actualA_ = (actualASigned > 0) ? static_cast<uint64_t>(actualASigned) : 0U;
    if (actualA_ > recvCapacity_) {
        actualA_ = recvCapacity_;
    }
    XTypeAlign32Size_ = hAlignSize_;
    if constexpr (HasTopkWeight == 1) {
        topkWeightsGm_.SetGlobalBuffer((__gm__ float *)topkWeights);
    }
    tpipe_->InitBuffer(xQueue_, SEND_DOUBLE_BUFFER_NUM, XTypeAlign32Size_);
    if constexpr (HasTopkWeight == 1) {
        tpipe_->InitBuffer(weightQueue_, SEND_DOUBLE_BUFFER_NUM, UB_ALIGN);
    }
    metadataChunkTokens_ = (actualA_ < META_CHUNK_TOKEN_MAX) ? static_cast<uint32_t>(actualA_) : META_CHUNK_TOKEN_MAX;
    metadataChunkTokens_ = (metadataChunkTokens_ == 0U) ? 1U : metadataChunkTokens_;
    uint32_t metadataChunkBytes =
        Ceil(static_cast<uint64_t>(metadataChunkTokens_) * RECV_META_FIELDS * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(metadataBuf_, metadataChunkBytes);
    tpipe_->InitBuffer(readStateBuf_, WIN_ADDR_ALIGN);
    statusTensor_ = readStateBuf_.Get<uint32_t>();
    diagWriter_.RunPosRecord(MOE_EP_COMBINE_RUN_POS_INIT_DONE);
}

// TODO(delete): legacy address-table construction path.
template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::BuildAddressTable()
{
    // Follow dispatch PR 10309: count per core, derive an exclusive prefix, then batch-write records.
    constexpr uint32_t metaBytesPerToken = RECV_META_FIELDS * sizeof(int32_t);
    constexpr uint32_t rankCountBlockElements = UB_ALIGN / sizeof(int32_t);
    uint32_t metaChunkTokens = (actualA_ < META_CHUNK_TOKEN_MAX) ? actualA_ : META_CHUNK_TOKEN_MAX;
    metaChunkTokens = (metaChunkTokens == 0U) ? 1U : metaChunkTokens;
    uint32_t metaChunkBytes = Ceil(metaChunkTokens * metaBytesPerToken, UB_ALIGN) * UB_ALIGN;
    uint32_t rankCountElements = epWorldSize_ * rankCountBlockElements;
    tpipe_->InitBuffer(metadataBuf_, metaChunkBytes);
    if constexpr (HasTopkWeight == 1) {
        uint32_t weightChunkBytes = Ceil(metaChunkTokens * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(weightsBuf_, weightChunkBytes);
    }
    tpipe_->InitBuffer(addressTableBuf_, ADDRESS_TABLE_BUFFER_BYTES);
    tpipe_->InitBuffer(rankInfoBuf_, 3U * rankCountElements * sizeof(int32_t));

    LocalTensor<int32_t> rankCounts = rankInfoBuf_.Get<int32_t>();
    LocalTensor<int32_t> rankPrefix = rankCounts[rankCountElements];
    LocalTensor<int32_t> prefixSums = rankCounts[2U * rankCountElements];
    Duplicate<int32_t>(rankCounts, 0, rankCountElements);
    Duplicate<int32_t>(rankPrefix, 0, rankCountElements);
    SyncFunc<AscendC::HardEvent::V_S>();

    DataCopyParams rankCountCopyParams{static_cast<uint16_t>(epWorldSize_), static_cast<uint16_t>(UB_ALIGN), 0U,
                                       static_cast<uint16_t>(WIN_ADDR_ALIGN - UB_ALIGN)};

    uint64_t tokensPerCore = actualA_ / aivNum_;
    uint64_t remainder = actualA_ % aivNum_;
    uint64_t scanStart = static_cast<uint64_t>(aivId_) * tokensPerCore + ((aivId_ < remainder) ? aivId_ : remainder);
    uint64_t scanEnd = scanStart + tokensPerCore + ((aivId_ < remainder) ? 1U : 0U);

    LocalTensor<int32_t> metadataLocal = metadataBuf_.Get<int32_t>();
    const DataCopyPadExtParams<int32_t> metaPadParams{false, 0U, 0U, 0U};
    for (uint64_t chunkStart = scanStart; chunkStart < scanEnd; chunkStart += metaChunkTokens) {
        uint64_t chunkEnd = (chunkStart + metaChunkTokens > scanEnd) ? scanEnd : chunkStart + metaChunkTokens;
        uint32_t curChunkTokens = static_cast<uint32_t>(chunkEnd - chunkStart);

        DataCopyExtParams metaCopyParams{1U, curChunkTokens * metaBytesPerToken, 0U, 0U, 0U};
        DataCopyPad(metadataLocal, recvSrcMetadataGm_[chunkStart * RECV_META_FIELDS], metaCopyParams, metaPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        for (uint32_t i = 0; i < curChunkTokens; ++i) {
            int32_t srcRank = metadataLocal.GetValue(i * RECV_META_FIELDS);
            if (srcRank < 0 || srcRank >= static_cast<int32_t>(epWorldSize_) ||
                srcRank == static_cast<int32_t>(rankId_)) {
                continue;
            }
            uint32_t countOffset = static_cast<uint32_t>(srcRank) * rankCountBlockElements;
            rankCounts.SetValue(countOffset, rankCounts.GetValue(countOffset) + 1);
        }
        SyncFunc<AscendC::HardEvent::S_MTE2>();
    }

    // Publish this core's row. Address-table positions and totals are derived from these rows without atomic ops.
    GlobalTensor<int32_t> perCoreRankCounts;
    perCoreRankCounts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
        perCoreRankCountWorkspaceAddr_ + static_cast<uint64_t>(aivId_) * perCoreRankCountStride_));
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(perCoreRankCounts, rankCounts, rankCountElements);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    SyncAll<true>();

    // Group preceding rows into a few MTE2 copies, as in PR 10309 GetSlotStartNum().
    uint32_t rankCountRowBytes = rankCountElements * sizeof(int32_t);
    uint32_t prefixRowsPerBatch = ADDRESS_TABLE_BUFFER_BYTES / rankCountRowBytes;
    LocalTensor<int32_t> prefixRows = addressTableBuf_.Get<int32_t>();
    GlobalTensor<int32_t> allPerCoreRankCounts;
    allPerCoreRankCounts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(perCoreRankCountWorkspaceAddr_));
    const DataCopyPadExtParams<int32_t> prefixPadParams{false, 0U, 0U, 0U};
    for (uint32_t coreStart = 0; coreStart < aivId_; coreStart += prefixRowsPerBatch) {
        uint32_t copyRows = (coreStart + prefixRowsPerBatch > aivId_) ? aivId_ - coreStart : prefixRowsPerBatch;
        DataCopyExtParams prefixCopyParams{static_cast<uint16_t>(copyRows), rankCountRowBytes,
                                           static_cast<uint32_t>(perCoreRankCountStride_ - rankCountRowBytes), 0U, 0U};
        DataCopyPad(prefixRows,
                    allPerCoreRankCounts[static_cast<uint64_t>(coreStart) * perCoreRankCountStride_ / sizeof(int32_t)],
                    prefixCopyParams, prefixPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        const uint32_t prefixShape[] = {copyRows, rankCountElements};
        ReduceSum<int32_t, AscendC::Pattern::Reduce::RA, true>(prefixSums, prefixRows, prefixShape, false);
        Add(rankPrefix, rankPrefix, prefixSums, rankCountElements);
        if (coreStart + copyRows < aivId_) {
            SyncFunc<AscendC::HardEvent::V_MTE2>();
        }
    }
    SyncFunc<AscendC::HardEvent::V_S>();

    // rankCounts still holds this core's row. The last AIV writes prefix + own count as each remote rank's total.
    if (aivId_ == aivNum_ - 1U) {
        for (uint32_t rank = 0; rank < epWorldSize_; ++rank) {
            uint32_t countOffset = rank * rankCountBlockElements;
            rankCounts.SetValue(countOffset, rankPrefix.GetValue(countOffset) + rankCounts.GetValue(countOffset));
        }
        GlobalTensor<int32_t> totalRankCounts;
        totalRankCounts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(rankCountWorkspaceAddr_));
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(totalRankCounts, rankCounts, rankCountCopyParams);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }

    // Each group fits even if all records target the same rank. Records are grouped by rank in UB.
    uint32_t entriesPerRank = ADDRESS_TABLE_BUFFER_BYTES / epWorldSize_ / ADDRESS_ENTRY_UB_BYTES;
    LocalTensor<uint32_t> addressEntries = addressTableBuf_.Get<uint32_t>();
    constexpr uint32_t entryElements = ADDRESS_ENTRY_UB_BYTES / sizeof(uint32_t);
    // DataCopyParams uses 32-byte blocks, unlike the byte-based DataCopyPad parameters above.
    DataCopyParams addressCopyParams{
        0U, static_cast<uint16_t>(ADDRESS_ENTRY_UB_BYTES / UB_ALIGN), 0U,
        static_cast<uint16_t>((ADDRESS_ENTRY_GM_BYTES - ADDRESS_ENTRY_UB_BYTES) / UB_ALIGN)};
    GM_ADDR localDataBase = GetUrmaWinAddrByRankId(rankId_, combineDataWinOffset_);
    GM_ADDR localStateBase = GetUrmaStateAddrByRankId(rankId_, combineStateWinOffset_);
    for (uint64_t groupStart = scanStart; groupStart < scanEnd; groupStart += entriesPerRank) {
        uint64_t groupEnd = (groupStart + entriesPerRank > scanEnd) ? scanEnd : groupStart + entriesPerRank;
        uint32_t groupTokens = static_cast<uint32_t>(groupEnd - groupStart);
        DataCopyExtParams metaCopyParams{1U, groupTokens * metaBytesPerToken, 0U, 0U, 0U};
        DataCopyPad(metadataLocal, recvSrcMetadataGm_[groupStart * RECV_META_FIELDS], metaCopyParams, metaPadParams);

        LocalTensor<uint32_t> weightsLocal;
        if constexpr (HasTopkWeight == 1) {
            weightsLocal = weightsBuf_.Get<uint32_t>();
            DataCopyExtParams weightCopyParams{1U, static_cast<uint32_t>(groupTokens * sizeof(uint32_t)), 0U, 0U, 0U};
            const DataCopyPadExtParams<uint32_t> weightPadParams{false, 0U, 0U, 0U};
            DataCopyPad(weightsLocal, topkWeightsGm_[groupStart].template ReinterpretCast<uint32_t>(), weightCopyParams,
                        weightPadParams);
        }
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        Duplicate<int32_t>(rankCounts, 0, rankCountElements);
        SyncFunc<AscendC::HardEvent::V_S>();
        for (uint32_t i = 0; i < groupTokens; ++i) {
            int32_t srcRank = metadataLocal.GetValue(i * RECV_META_FIELDS);
            if (srcRank < 0 || srcRank >= static_cast<int32_t>(epWorldSize_)) {
                continue;
            }
            uint32_t tokenIndex = static_cast<uint32_t>(groupStart) + i;
            int32_t srcTokenIdx = metadataLocal.GetValue(i * RECV_META_FIELDS + 1);
            int32_t srcTopKIdx = metadataLocal.GetValue(i * RECV_META_FIELDS + 2);
            uint32_t weight = 0U;
            if constexpr (HasTopkWeight == 1) {
                weight = weightsLocal.GetValue(i);
            }
            if (srcRank == static_cast<int32_t>(rankId_)) {
                SendLocalSlot(tokenIndex, srcTokenIdx, srcTopKIdx, weight, localDataBase, localStateBase);
                continue;
            }
            uint32_t countOffset = static_cast<uint32_t>(srcRank) * rankCountBlockElements;
            uint32_t localIndex = static_cast<uint32_t>(rankCounts.GetValue(countOffset));
            uint32_t entryOffset = (static_cast<uint32_t>(srcRank) * entriesPerRank + localIndex) * entryElements;
            addressEntries.SetValue(entryOffset, tokenIndex);
            addressEntries.SetValue(entryOffset + 1, static_cast<uint32_t>(srcTokenIdx));
            addressEntries.SetValue(entryOffset + 2, static_cast<uint32_t>(srcTopKIdx));
            addressEntries.SetValue(entryOffset + 3, weight);
            rankCounts.SetValue(countOffset, static_cast<int32_t>(localIndex + 1));
        }

        SyncFunc<AscendC::HardEvent::S_MTE3>();
        for (uint32_t rank = 0; rank < epWorldSize_; ++rank) {
            uint32_t countOffset = rank * rankCountBlockElements;
            uint32_t count = static_cast<uint32_t>(rankCounts.GetValue(countOffset));
            if (count == 0U) {
                continue;
            }
            uint32_t prefixOffset = rank * rankCountBlockElements;
            uint32_t writeStart = static_cast<uint32_t>(rankPrefix.GetValue(prefixOffset));
            GlobalTensor<uint32_t> addressTable;
            addressTable.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(GetLocalSendDataWorkspaceAddr(rank)));
            addressCopyParams.blockCount = static_cast<uint16_t>(count);
            DataCopy(addressTable[static_cast<uint64_t>(writeStart) * ADDRESS_ENTRY_GM_BYTES / sizeof(uint32_t)],
                     addressEntries[rank * entriesPerRank * entryElements], addressCopyParams);
            rankPrefix.SetValue(prefixOffset, static_cast<int32_t>(writeStart + count));
        }
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        SyncFunc<AscendC::HardEvent::S_MTE2>();
    }
    SyncAll<true>();
}

// TODO(delete): legacy address-table path only.
template <TemplateMoeEpCombineTypeClass>
__aicore__ inline uint32_t MoeEpCombine<TemplateMoeEpCombineTypeFunc>::GetAddressTableCount(uint32_t targetRank)
{
    GlobalTensor<uint32_t> totalRankCounts;
    totalRankCounts.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(rankCountWorkspaceAddr_));
    DataCopy(statusTensor_, totalRankCounts[static_cast<uint64_t>(targetRank) * WIN_ADDR_ALIGN / sizeof(uint32_t)],
             UB_ALIGN / sizeof(uint32_t));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint32_t count = statusTensor_.GetValue(0);
    SyncFunc<AscendC::HardEvent::S_MTE2>();
    return count;
}

// TODO(delete): legacy address-table path only.
template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendAddressTableRange(uint32_t targetRank,
                                                                                         uint32_t entryStart,
                                                                                         uint32_t entryEnd,
                                                                                         uint32_t channelIndex)
{
    if (entryStart >= entryEnd) {
        return;
    }
    if (targetRank != rankId_) {
        BeginPreparedWrites(targetRank, channelIndex);
    }

    uint32_t metaChunkTokens = (actualA_ < META_CHUNK_TOKEN_MAX) ? actualA_ : META_CHUNK_TOKEN_MAX;
    uint32_t metaChunkBytes = Ceil(metaChunkTokens * RECV_META_FIELDS * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t entryChunkMax = metaChunkBytes / ADDRESS_ENTRY_UB_BYTES;
    LocalTensor<uint32_t> entriesLocal = metadataBuf_.Get<uint32_t>();
    GlobalTensor<uint32_t> addressTable;
    addressTable.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(GetLocalSendDataWorkspaceAddr(targetRank)));
    const DataCopyPadExtParams<uint32_t> padParams{false, 0U, 0U, 0U};
    constexpr uint32_t localEntryStride = ADDRESS_ENTRY_UB_BYTES / sizeof(uint32_t);
    constexpr uint32_t globalEntryStride = ADDRESS_ENTRY_GM_BYTES / sizeof(uint32_t);
    GM_ADDR remoteDataBase = GetUrmaWinAddrByRankId(targetRank, combineDataWinOffset_);
    GM_ADDR remoteStateBase = GetUrmaStateAddrByRankId(targetRank, combineStateWinOffset_);
    uint64_t tokenBytes = static_cast<uint64_t>(axisH_) * sizeof(XType);

    for (uint32_t chunkStart = entryStart; chunkStart < entryEnd; chunkStart += entryChunkMax) {
        uint32_t chunkEnd = (chunkStart + entryChunkMax > entryEnd) ? entryEnd : chunkStart + entryChunkMax;
        uint32_t curEntries = chunkEnd - chunkStart;
        DataCopyExtParams copyParams{static_cast<uint16_t>(curEntries), ADDRESS_ENTRY_UB_BYTES,
                                     ADDRESS_ENTRY_GM_BYTES - ADDRESS_ENTRY_UB_BYTES, 0U, 0U};
        DataCopyPad(entriesLocal, addressTable[chunkStart * globalEntryStride], copyParams, padParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        for (uint32_t i = 0; i < curEntries; ++i) {
            uint32_t offset = i * localEntryStride;
            uint32_t tokenIndex = entriesLocal.GetValue(offset);
            int32_t srcTokenIdx = static_cast<int32_t>(entriesLocal.GetValue(offset + 1));
            int32_t srcTopKIdx = static_cast<int32_t>(entriesLocal.GetValue(offset + 2));
            uint32_t weight = entriesLocal.GetValue(offset + 3);
            SendRemoteSlot(tokenIndex, srcTokenIdx, srcTopKIdx, weight, remoteDataBase, remoteStateBase, tokenBytes);
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
    DataCopyPad(tokenTensor, xGm_[static_cast<uint64_t>(tokenIndex) * axisH_], copyParams, padParams);
    xQueue_.EnQue(tokenTensor);
    tokenTensor = xQueue_.DeQue<XType>();
    DataCopyPad(outToken, tokenTensor, copyParams);
    xQueue_.FreeTensor<XType>(tokenTensor);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::FlushPreparedWrites(bool keepHandle)
{
    if (preparedWriteCount_ != 0) {
        (void)hcomm_.BatchCommit(activeBatchHandle_);
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
    (void)hcomm_.WriteNbi<config>(activeBatchHandle_, dst, src, len);
    ++preparedWriteCount_;
    ++sqWriteCount_;
}

// TODO(delete): legacy address-table path only.
template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendLocalSlot(uint32_t tokenIndex,
                                                                                 int32_t srcTokenIdx,
                                                                                 int32_t srcTopKIdx, uint32_t weight,
                                                                                 GM_ADDR localDataBase,
                                                                                 GM_ADDR localStateBase)
{
    uint64_t sendTokenOffset = (static_cast<uint64_t>(srcTokenIdx) * topK_ + srcTopKIdx) * perSlotBytes_;
    uint64_t recvStateOffset = (srcTokenIdx * topK_ + srcTopKIdx) * WIN_ADDR_ALIGN;

    SendLocalToken(tokenIndex, localDataBase + sendTokenOffset);
    if constexpr (HasTopkWeight == 1) {
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        statusTensor_(0) = weight;
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        GlobalTensor<uint32_t> state;
        state.SetGlobalBuffer((__gm__ uint32_t *)(localStateBase + recvStateOffset));
        DataCopyParams weightCopyParams = {1U, static_cast<uint16_t>(sizeof(uint32_t)), 0U, 0U};
        DataCopyPad(state, statusTensor_, weightCopyParams);
    }
}

// TODO(delete): legacy address-table path only.
template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendRemoteSlot(
    uint32_t tokenIndex, int32_t srcTokenIdx, int32_t srcTopKIdx, uint32_t weight, GM_ADDR remoteDataBase,
    GM_ADDR remoteStateBase, uint64_t tokenBytes)
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
        GM_ADDR weightAddr = (GM_ADDR)topkWeightsGm_.GetPhyAddr(tokenIndex);
        if (drainAfterToken) {
            PrepareWrite<DEFAULT_CQE_WQE_CONFIG>(remoteStateBase + recvStateOffset, weightAddr, sizeof(uint32_t));
        } else {
            PrepareWrite<DEFAULT_WQE_CONFIG>(remoteStateBase + recvStateOffset, weightAddr, sizeof(uint32_t));
        }
    }
    if (drainAfterToken) {
        FlushPreparedWrites(true);
        (void)hcomm_.Drain(activeBatchChannel_);
        sqWriteCount_ = 0;
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendChannelFlag(uint32_t dstRank,
                                                                                   uint32_t channelIndex)
{
    uint64_t flagIndex = static_cast<uint64_t>(rankId_) * COMBINE_CHANNEL_COUNT + channelIndex;
    uint64_t flagOffset =
        static_cast<uint64_t>(numMaxTokensPerRank_) * topK_ * WIN_ADDR_ALIGN + flagIndex * WIN_ADDR_ALIGN;
    GM_ADDR flagAddr = GetUrmaStateAddrByRankId(dstRank, combineStateWinOffset_) + flagOffset;
    if (dstRank == rankId_) {
        GlobalTensor<uint64_t> localFlag;
        localFlag.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(flagAddr));
        LocalTensor<uint64_t> flagTensor = statusTensor_.ReinterpretCast<uint64_t>();
        DataCopy(localFlag, flagTensor, UB_ALIGN / sizeof(uint64_t));
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(localFlag);
        return;
    }
    BeginPreparedWrites(dstRank, channelIndex);
    PrepareWrite<CHANNEL_FLAG_WQE_CONFIG>(flagAddr, flagWorkspaceAddr_, WIN_ADDR_ALIGN);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::InitFlagSource()
{
    LocalTensor<uint64_t> flagTensor = statusTensor_.ReinterpretCast<uint64_t>();
    Duplicate<uint64_t>(flagTensor, 1U, WIN_ADDR_ALIGN / sizeof(uint64_t));
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    GlobalTensor<uint64_t> flagWorkspace;
    flagWorkspace.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(flagWorkspaceAddr_));
    DataCopy(flagWorkspace, flagTensor, WIN_ADDR_ALIGN / sizeof(uint64_t));
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

// TODO(delete): rank offsets replaced metadata rank probing.
template <TemplateMoeEpCombineTypeClass>
__aicore__ inline int32_t MoeEpCombine<TemplateMoeEpCombineTypeFunc>::ReadMetadataRank(uint64_t metadataRow)
{
    if (metadataRow >= actualA_) {
        return static_cast<int32_t>(epWorldSize_);
    }
    LocalTensor<int32_t> metadataLocal = metadataBuf_.Get<int32_t>();
    DataCopyExtParams copyParams{1U, RECV_META_FIELDS * sizeof(int32_t), 0U, 0U, 0U};
    const DataCopyPadExtParams<int32_t> padParams{false, 0U, 0U, 0U};
    DataCopyPad(metadataLocal, recvSrcMetadataGm_[metadataRow * RECV_META_FIELDS], copyParams, padParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    int32_t srcRank = metadataLocal.GetValue(META_SRC_RANK_OFFSET);
    SyncFunc<AscendC::HardEvent::S_MTE2>();
    return srcRank;
}

// TODO(delete): rank offsets replaced metadata lower-bound lookup.
template <TemplateMoeEpCombineTypeClass>
__aicore__ inline uint64_t MoeEpCombine<TemplateMoeEpCombineTypeFunc>::LowerBoundSrcRank(uint32_t targetRank)
{
    uint64_t left = 0U;
    uint64_t right = actualA_;
    while (left < right) {
        uint64_t mid = left + ((right - left) >> 1U);
        int32_t srcRank = ReadMetadataRank(mid);
        if (srcRank < static_cast<int32_t>(targetRank)) {
            left = mid + 1U;
        } else {
            right = mid;
        }
    }
    return left;
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SplitRange(uint64_t rangeBegin, uint64_t rangeEnd,
                                                                              uint32_t coreCount, uint32_t coreIndex,
                                                                              uint64_t &coreBegin, uint64_t &coreEnd)
{
    if (rangeBegin >= rangeEnd || coreCount == 0U || coreIndex >= coreCount) {
        coreBegin = rangeBegin;
        coreEnd = rangeBegin;
        return;
    }
    uint64_t count = rangeEnd - rangeBegin;
    uint64_t base = count / coreCount;
    uint64_t remainder = count % coreCount;
    uint64_t prefix = static_cast<uint64_t>(coreIndex) * base +
                      ((static_cast<uint64_t>(coreIndex) < remainder) ? coreIndex : remainder);
    uint64_t coreCountValue = base + ((static_cast<uint64_t>(coreIndex) < remainder) ? 1U : 0U);
    coreBegin = rangeBegin + prefix;
    coreEnd = coreBegin + coreCountValue;
}

// TODO(delete): the direct metadata hot path trusts the Dispatch Epilogue metadata contract.
template <TemplateMoeEpCombineTypeClass>
__aicore__ inline bool MoeEpCombine<TemplateMoeEpCombineTypeFunc>::IsMetadataRowValid(
    uint32_t targetRank, int32_t srcRank, int32_t srcTokenIdx, int32_t srcTopKIdx, int32_t recvXIdx)
{
    return srcRank == static_cast<int32_t>(targetRank) && srcTokenIdx >= 0 &&
           srcTokenIdx < static_cast<int32_t>(numMaxTokensPerRank_) && srcTopKIdx >= 0 &&
           srcTopKIdx < static_cast<int32_t>(topK_) && recvXIdx >= 0 && static_cast<uint64_t>(recvXIdx) < actualA_;
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendLocalWeight(uint32_t recvXIdx, GM_ADDR dstAddr)
{
    if constexpr (HasTopkWeight == 1) {
        GlobalTensor<float> outWeight;
        outWeight.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dstAddr));
        LocalTensor<float> weightTensor = weightQueue_.AllocTensor<float>();
        DataCopyParams copyParams{1U, static_cast<uint16_t>(sizeof(float)), 0U, 0U};
        DataCopyPadParams padParams{false, 0U, 0U, 0U};
        DataCopyPad(weightTensor, topkWeightsGm_[recvXIdx], copyParams, padParams);
        weightQueue_.EnQue(weightTensor);
        weightTensor = weightQueue_.DeQue<float>();
        DataCopyPad(outWeight, weightTensor, copyParams);
        weightQueue_.FreeTensor<float>(weightTensor);
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendLocalMetadataSlot(
    uint32_t recvXIdx, int32_t srcTokenIdx, int32_t srcTopKIdx, GM_ADDR localDataBase, GM_ADDR localStateBase)
{
    uint64_t dstSlot =
        static_cast<uint64_t>(static_cast<uint32_t>(srcTokenIdx)) * topK_ + static_cast<uint32_t>(srcTopKIdx);
    SendLocalToken(recvXIdx, localDataBase + dstSlot * perSlotBytes_);
    if constexpr (HasTopkWeight == 1) {
        SendLocalWeight(recvXIdx, localStateBase + dstSlot * WIN_ADDR_ALIGN);
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendRemoteMetadataSlot(
    uint32_t recvXIdx, int32_t srcTokenIdx, int32_t srcTopKIdx, GM_ADDR remoteDataBase, GM_ADDR remoteStateBase,
    uint64_t tokenBytes)
{
    uint64_t dstSlot =
        static_cast<uint64_t>(static_cast<uint32_t>(srcTokenIdx)) * topK_ + static_cast<uint32_t>(srcTopKIdx);
    GM_ADDR tokenAddr = reinterpret_cast<GM_ADDR>(xGm_.GetPhyAddr(static_cast<uint64_t>(recvXIdx) * axisH_));
    PrepareWrite<DEFAULT_WQE_CONFIG>(remoteDataBase + dstSlot * perSlotBytes_, tokenAddr, tokenBytes);
    if constexpr (HasTopkWeight == 1) {
        GM_ADDR weightAddr = reinterpret_cast<GM_ADDR>(topkWeightsGm_.GetPhyAddr(recvXIdx));
        PrepareWrite<DEFAULT_WQE_CONFIG>(remoteStateBase + dstSlot * WIN_ADDR_ALIGN, weightAddr, sizeof(float));
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline bool MoeEpCombine<TemplateMoeEpCombineTypeFunc>::ProcessLocalMetadataRange(uint64_t rangeBegin,
                                                                                             uint64_t rangeEnd)
{
    if (rangeBegin >= rangeEnd) {
        return false;
    }
    constexpr uint32_t metaBytesPerToken = RECV_META_FIELDS * sizeof(int32_t);
    LocalTensor<int32_t> metadataLocal = metadataBuf_.Get<int32_t>();
    const DataCopyPadExtParams<int32_t> padParams{false, 0U, 0U, 0U};
    GM_ADDR localDataBase = GetUrmaWinAddrByRankId(rankId_, combineDataWinOffset_);
    GM_ADDR localStateBase = GetUrmaStateAddrByRankId(rankId_, combineStateWinOffset_);
    bool didLocalWrite = false;

    for (uint64_t chunkStart = rangeBegin; chunkStart < rangeEnd; chunkStart += metadataChunkTokens_) {
        uint64_t chunkEnd =
            (chunkStart + metadataChunkTokens_ > rangeEnd) ? rangeEnd : chunkStart + metadataChunkTokens_;
        uint32_t chunkCount = static_cast<uint32_t>(chunkEnd - chunkStart);
        DataCopyExtParams copyParams{1U, chunkCount * metaBytesPerToken, 0U, 0U, 0U};
        DataCopyPad(metadataLocal, recvSrcMetadataGm_[chunkStart * RECV_META_FIELDS], copyParams, padParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        for (uint32_t i = 0; i < chunkCount; ++i) {
            uint32_t metaOffset = i * RECV_META_FIELDS;
            // int32_t srcRank = metadataLocal.GetValue(metaOffset + META_SRC_RANK_OFFSET); // TODO(delete): validation
            // only
            int32_t srcTokenIdx = metadataLocal.GetValue(metaOffset + META_TOKEN_IDX_OFFSET);
            int32_t srcTopKIdx = metadataLocal.GetValue(metaOffset + META_TOPK_IDX_OFFSET);
            int32_t recvXIdx = metadataLocal.GetValue(metaOffset + META_RECV_X_IDX_OFFSET);
            // TODO(delete): remove the legacy per-row validation helper after metadata contract validation.
            // Kernel trusts metadata produced by Dispatch Epilogue; do not validate every row on the hot path.
            // if (!IsMetadataRowValid(rankId_, srcRank, srcTokenIdx, srcTopKIdx, recvXIdx)) {
            //     continue;
            // }
            SendLocalMetadataSlot(static_cast<uint32_t>(recvXIdx), srcTokenIdx, srcTopKIdx, localDataBase,
                                  localStateBase);
            didLocalWrite = true;
        }
        SyncFunc<AscendC::HardEvent::S_MTE2>();
    }
    return didLocalWrite;
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::ProcessRemoteMetadataRange(uint32_t targetRank,
                                                                                              uint64_t rangeBegin,
                                                                                              uint64_t rangeEnd)
{
    constexpr uint32_t metaBytesPerToken = RECV_META_FIELDS * sizeof(int32_t);
    LocalTensor<int32_t> metadataLocal = metadataBuf_.Get<int32_t>();
    const DataCopyPadExtParams<int32_t> padParams{false, 0U, 0U, 0U};
    GM_ADDR remoteDataBase = GetUrmaWinAddrByRankId(targetRank, combineDataWinOffset_);
    GM_ADDR remoteStateBase = GetUrmaStateAddrByRankId(targetRank, combineStateWinOffset_);
    uint64_t tokenBytes = static_cast<uint64_t>(axisH_) * sizeof(XType);

    for (uint64_t chunkStart = rangeBegin; chunkStart < rangeEnd; chunkStart += metadataChunkTokens_) {
        uint64_t chunkEnd =
            (chunkStart + metadataChunkTokens_ > rangeEnd) ? rangeEnd : chunkStart + metadataChunkTokens_;
        uint32_t chunkCount = static_cast<uint32_t>(chunkEnd - chunkStart);
        DataCopyExtParams copyParams{1U, chunkCount * metaBytesPerToken, 0U, 0U, 0U};
        DataCopyPad(metadataLocal, recvSrcMetadataGm_[chunkStart * RECV_META_FIELDS], copyParams, padParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        for (uint32_t i = 0; i < chunkCount; ++i) {
            uint32_t metaOffset = i * RECV_META_FIELDS;
            // int32_t srcRank = metadataLocal.GetValue(metaOffset + META_SRC_RANK_OFFSET); // TODO(delete): validation
            // only
            int32_t srcTokenIdx = metadataLocal.GetValue(metaOffset + META_TOKEN_IDX_OFFSET);
            int32_t srcTopKIdx = metadataLocal.GetValue(metaOffset + META_TOPK_IDX_OFFSET);
            int32_t recvXIdx = metadataLocal.GetValue(metaOffset + META_RECV_X_IDX_OFFSET);
            // TODO(delete): remove the legacy per-row validation helper after metadata contract validation.
            // Kernel trusts metadata produced by Dispatch Epilogue; do not validate every row on the hot path.
            // if (!IsMetadataRowValid(targetRank, srcRank, srcTokenIdx, srcTopKIdx, recvXIdx)) {
            //     continue;
            // }
            SendRemoteMetadataSlot(static_cast<uint32_t>(recvXIdx), srcTokenIdx, srcTopKIdx, remoteDataBase,
                                   remoteStateBase, tokenBytes);
        }
        SyncFunc<AscendC::HardEvent::S_MTE2>();
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendPhaseDirectFromMetadata()
{
    if (epWorldSize_ == 0U || aivNum_ == 0U) {
        return;
    }

    InitFlagSource();

    // TODO(delete): rank boundaries now come directly from Dispatch Epilogue.
    // uint64_t localBegin = LowerBoundSrcRank(rankId_);
    // uint64_t localEnd = LowerBoundSrcRank(rankId_ + 1U);
    uint64_t localBegin = static_cast<uint32_t>(rankOffsetsTensor_.GetValue(rankId_));
    uint64_t localEnd = static_cast<uint32_t>(rankOffsetsTensor_.GetValue(rankId_ + 1U));
    uint64_t localCoreBegin = 0U;
    uint64_t localCoreEnd = 0U;
    SplitRange(localBegin, localEnd, aivNum_, aivId_, localCoreBegin, localCoreEnd);
    bool didLocalWrite = ProcessLocalMetadataRange(localCoreBegin, localCoreEnd);
    if (didLocalWrite) {
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }

    // All AIVs must reach this barrier, including empty local ranges and actualA_ == 0.
    SyncAll<true>();

    if (aivId_ == 0U) {
        SendChannelFlag(rankId_, 0U);
    }

    activeBatchHandle_ = {};
    activeBatchChannel_ = 0U;
    preparedWriteCount_ = 0U;
    activeBatchInitialized_ = false;

    uint64_t remoteOrdinalBegin = 0U;
    uint64_t remoteOrdinalEnd = 0U;
    uint32_t remoteRankCount = epWorldSize_ - 1U;
    SplitRange(0U, remoteRankCount, aivNum_, aivId_, remoteOrdinalBegin, remoteOrdinalEnd);
    for (uint64_t ordinal = remoteOrdinalBegin; ordinal < remoteOrdinalEnd; ++ordinal) {
        uint32_t dstRank = (ordinal < rankId_) ? static_cast<uint32_t>(ordinal) : static_cast<uint32_t>(ordinal + 1U);
        // TODO(delete): keep the legacy lower-bound calls commented until the rank-offset path is validated.
        // uint64_t rankBegin = LowerBoundSrcRank(dstRank);
        // uint64_t rankEnd = LowerBoundSrcRank(dstRank + 1U);
        uint64_t rankBegin = static_cast<uint32_t>(rankOffsetsTensor_.GetValue(dstRank));
        uint64_t rankEnd = static_cast<uint32_t>(rankOffsetsTensor_.GetValue(dstRank + 1U));
        BeginPreparedWrites(dstRank, 0U);
        ProcessRemoteMetadataRange(dstRank, rankBegin, rankEnd);
        SendChannelFlag(dstRank, 0U);
        FlushPreparedWrites();
    }
    if (actualA_ != 0U) {
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(recvSrcMetadataGm_);
    }
    diagWriter_.RunPosRecord(MOE_EP_COMBINE_RUN_POS_URMA_REQUESTS_ISSUE_DONE);
}

// TODO(delete): legacy address-table path only.
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

// TODO(delete): legacy address-table send entry; its Process() call site remains commented out.
template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendPhaseExpertToToken()
{
    if (epWorldSize_ == 0 || aivNum_ == 0) {
        return;
    }
    uint32_t activeAivNum = aivNum_;

    // Build once with all AIVs; communication AIVs consume only their compact rank table below.
    BuildAddressTable();

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
    if (sendsTokens) {
        if (splitRankTokens) {
            uint32_t entryCount = GetAddressTableCount(targetRank);
            uint32_t entriesPerCore = entryCount / groupSize;
            uint32_t remainder = entryCount % groupSize;
            uint32_t entryStart =
                coreIndexInGroup * entriesPerCore + ((coreIndexInGroup < remainder) ? coreIndexInGroup : remainder);
            uint32_t entryEnd = entryStart + entriesPerCore + ((coreIndexInGroup < remainder) ? 1U : 0U);
            SendAddressTableRange(targetRank, entryStart, entryEnd, coreIndexInGroup);
        } else {
            for (uint32_t rank = aivId_; rank < epWorldSize_; rank += activeAivNum) {
                uint32_t entryCount = GetAddressTableCount(rank);
                SendAddressTableRange(rank, 0U, entryCount, 0U);
            }
        }
    }

    // Local copies completed before BuildAddressTable's final barrier. Each communication channel publishes one
    // completion flag without a second all-core barrier here.
    bool publishesChannelFlag = aivId_ < activeAivNum && (!splitRankTokens || coreIndexInGroup < combineChannelCount_);
    if (publishesChannelFlag) {
        LocalTensor<uint64_t> flagTensor = statusTensor_.ReinterpretCast<uint64_t>();
        Duplicate<uint64_t>(flagTensor, 1U, UB_ALIGN / sizeof(uint64_t));
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        GlobalTensor<uint64_t> flagWorkspace;
        flagWorkspace.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(flagWorkspaceAddr_));
        DataCopy(flagWorkspace, flagTensor, UB_ALIGN / sizeof(uint64_t));
        SyncFunc<AscendC::HardEvent::MTE3_S>();
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
    // TODO(delete): remove the legacy address-table entry point and its helpers after direct-path validation.
    // SendPhaseExpertToToken();
    SendPhaseDirectFromMetadata();
}

#endif

} // namespace MoeEpCombineImpl

#endif // MOE_EP_COMBINE_H
