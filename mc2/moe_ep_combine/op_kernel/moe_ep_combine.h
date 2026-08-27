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
static constexpr uint32_t COMBINE_CHANNEL_COUNT = 1U;
static constexpr uint32_t RECV_META_FIELDS = 5;
static constexpr uint32_t META_SRC_RANK_OFFSET = 0U;
static constexpr uint32_t META_TOKEN_IDX_OFFSET = 1U;
static constexpr uint32_t META_TOPK_IDX_OFFSET = 2U;
static constexpr uint32_t META_RECV_X_IDX_OFFSET = 4U;
// PR 111 requires each committed batch to contain fewer WQEBBs than the SQ depth.
static constexpr uint32_t HCOMM_BATCH_CAPACITY = 256;
static constexpr uint32_t HCOMM_PLAIN_WRITE_WQE_BYTES = 64;
static constexpr uint32_t HCOMM_BATCH_BUFFER_BYTES = HCOMM_BATCH_CAPACITY * HCOMM_PLAIN_WRITE_WQE_BYTES;
constexpr uint64_t UB_ALIGN = 32UL;
constexpr uint64_t ALIGNED_LEN_256 = 256UL;
constexpr uint32_t SEND_DOUBLE_BUFFER_NUM = 2U;
static constexpr uint32_t META_CHUNK_TOKEN_MAX = 2048U;
static constexpr struct UrmaWqeEntry DEFAULT_WQE_CONFIG = {.odr = 5, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
static constexpr struct UrmaWqeEntry CHANNEL_FLAG_WQE_CONFIG = {.odr = 6, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
template <TemplateMoeEpCombineTypeClass>
class MoeEpCombine {
public:
    __aicore__ inline MoeEpCombine(){};

    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR recvSrcMetadata,
                                GM_ADDR numRecvPerExpert, GM_ADDR topkWeights, GM_ADDR workspace, GM_ADDR tilingGM,
                                TPipe *pipe, const MoeEpCombineInfo *tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void SendLocalToken(uint32_t tokenIndex, GM_ADDR dstAddr);
    __aicore__ inline void SendLocalSlot(uint32_t tokenIndex, int32_t srcTokenIdx, int32_t srcTopKIdx, uint32_t weight,
                                         GM_ADDR localDataBase, GM_ADDR localStateBase);
    __aicore__ inline void SendRemoteSlot(uint32_t tokenIndex, int32_t srcTokenIdx, int32_t srcTopKIdx, uint32_t weight,
                                          GM_ADDR remoteDataBase, GM_ADDR remoteStateBase, uint64_t tokenBytes);
    __aicore__ inline void SendChannelFlag(uint32_t dstRank);
    __aicore__ inline void BeginPreparedWrites(uint32_t dstRank);
    template <auto const &config>
    __aicore__ inline void PrepareWrite(GM_ADDR dst, GM_ADDR src, uint64_t len);
    __aicore__ inline void FlushPreparedWrites(bool keepHandle = false);
    __aicore__ inline void BuildRankOffsets();
    __aicore__ inline void SendMetadataRange(uint32_t targetRank, uint32_t rowStart, uint32_t rowEnd);
    __aicore__ inline void SendPhaseExpertToToken();
    __aicore__ inline void GetCoreAssignment(uint32_t totalBlocks, uint32_t &targetRank, uint32_t &coreIndexInGroup,
                                             uint32_t &groupSize);

    __aicore__ inline uint64_t GetCommHandle(uint32_t rankId)
    {
        return hcommHandle_[rankId * channelsPerRank_];
    }
    __aicore__ inline GM_ADDR GetUrmaWinAddrByRankId(uint32_t rankId, uint64_t offset)
    {
        return (GM_ADDR)(winRankAddr_[rankId] + offset);
    }
    __aicore__ inline GM_ADDR GetUrmaStateAddrByRankId(uint32_t rankId, uint64_t offset)
    {
        return (GM_ADDR)(winRankAddr_[rankId] + offset);
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

    GlobalTensor<XType> xGm_;
    GlobalTensor<int32_t> recvSrcMetadataGm_;
    GlobalTensor<int64_t> numRecvPerExpertGm_;
    GlobalTensor<float> topkWeightsGm_;

    LocalTensor<uint32_t> statusTensor_;
    LocalTensor<uint8_t> hcommTensor_;
    LocalTensor<uint8_t> hcommBatchTensor_;

    TBuf<> readStateBuf_;
    TBuf<> hcommBuf_;
    TBuf<TPosition::VECOUT> hcommBatchBuf_;

    TBuf<> metadataBuf_; // 发送阶段recvSrcMetadata的UB缓冲，批量搬运避免GetValue
    TBuf<> weightsBuf_;
    TBuf<> rankInfoBuf_;
    LocalTensor<int32_t> rankOffsets_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_; // 数据队列
    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;                 // 通信上下文
    using HcommBatchHandle = AscendC::BatchHandle<AscendC::ChannelHandle>;

    GM_ADDR winRankAddr_[Mc2Aclnn::HCCL_MAX_RANK_SIZE];
    uint64_t hcommHandle_[Mc2Aclnn::HCCL_MAX_RANK_SIZE];
    GM_ADDR flagSourceAddr_{nullptr};
    GM_ADDR perCoreRankCountWorkspaceAddr_{nullptr};
    GM_ADDR rankOffsetsWorkspaceAddr_{nullptr};
    uint64_t perCoreRankCountStride_{0};
    uint32_t aivId_{0};
    HcommBatchHandle activeBatchHandle_{};
    uint64_t activeBatchChannel_{0};
    uint32_t preparedWriteCount_{0};
    bool activeBatchInitialized_{false};
};

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx,
                                                                        GM_ADDR recvSrcMetadata,
                                                                        GM_ADDR numRecvPerExpert, GM_ADDR topkWeights,
                                                                        GM_ADDR workspace, GM_ADDR tilingGM,
                                                                        TPipe *pipe, const MoeEpCombineInfo *tilingData)
{
    tpipe_ = pipe;
    tilingData_ = tilingData;
    aivId_ = GetBlockIdx();
    epWorldSize_ = tilingData_->cfg.epWorldSize;
    numMaxTokensPerRank_ = tilingData_->cfg.numMaxTokensPerRank;
    topK_ = tilingData_->cfg.topK;
    axisH_ = tilingData_->cfg.hidden;
    perSlotBytes_ = tilingData_->cfg.perSlotBytes;
    aivNum_ = tilingData_->aivNum;
    localmoeNum_ = tilingData_->cfg.numLocalExperts;
    perCoreRankCountWorkspaceAddr_ = workspace;
    perCoreRankCountStride_ =
        Ceil(static_cast<uint64_t>(epWorldSize_) * sizeof(int32_t), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    rankOffsetsWorkspaceAddr_ =
        perCoreRankCountWorkspaceAddr_ + static_cast<uint64_t>(aivNum_) * perCoreRankCountStride_;
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
    if (channelsPerRank_ == 0 || (epWorldSize_ > 0 && channelsPerRank_ > Mc2Aclnn::HCCL_MAX_RANK_SIZE / epWorldSize_)) {
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
    uint64_t flagSourceOffset = (static_cast<uint64_t>(numMaxTokensPerRank_) * topK_ + epWorldSize_) * WIN_ADDR_ALIGN;
    flagSourceAddr_ = GetUrmaStateAddrByRankId(rankId_, combineStateWinOffset_) + flagSourceOffset;

    xGm_.SetGlobalBuffer((__gm__ XType *)x);
    recvSrcMetadataGm_.SetGlobalBuffer((__gm__ int32_t *)recvSrcMetadata);
    numRecvPerExpertGm_.SetGlobalBuffer((__gm__ int64_t *)numRecvPerExpert);

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
    XTypeAlign32Size_ = hAlignSize_;
    if constexpr (HasTopkWeight == 1) {
        XTypeAlign32Size_ = hWeightAlignSize_;
        topkWeightsGm_.SetGlobalBuffer((__gm__ float *)topkWeights);
    }
    tpipe_->InitBuffer(xQueue_, SEND_DOUBLE_BUFFER_NUM, XTypeAlign32Size_);
    tpipe_->InitBuffer(readStateBuf_, UB_ALIGN); // 32
    statusTensor_ = readStateBuf_.Get<uint32_t>();
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::BuildRankOffsets()
{
    // Metadata is already rank ordered. Count in parallel, then publish only the compact rank boundaries.
    constexpr uint32_t metaBytesPerToken = RECV_META_FIELDS * sizeof(int32_t);
    constexpr uint32_t rankAlignElements = UB_ALIGN / sizeof(int32_t);
    uint32_t metaChunkTokens = (actualA_ < META_CHUNK_TOKEN_MAX) ? actualA_ : META_CHUNK_TOKEN_MAX;
    metaChunkTokens = (metaChunkTokens == 0U) ? 1U : metaChunkTokens;
    uint32_t metaChunkBytes = Ceil(metaChunkTokens * metaBytesPerToken, UB_ALIGN) * UB_ALIGN;
    uint32_t rankCountElements = Ceil(epWorldSize_, rankAlignElements) * rankAlignElements;
    uint32_t rankOffsetElements = Ceil(epWorldSize_ + 1U, rankAlignElements) * rankAlignElements;
    tpipe_->InitBuffer(metadataBuf_, metaChunkBytes);
    if constexpr (HasTopkWeight == 1) {
        tpipe_->InitBuffer(weightsBuf_, UB_ALIGN);
    }
    tpipe_->InitBuffer(rankInfoBuf_, (rankCountElements + rankOffsetElements) * sizeof(int32_t));

    LocalTensor<int32_t> rankCounts = rankInfoBuf_.Get<int32_t>();
    rankOffsets_ = rankCounts[rankCountElements];
    Duplicate<int32_t>(rankCounts, 0, rankCountElements);
    SyncFunc<AscendC::HardEvent::V_S>();

    uint64_t tokensPerCore = actualA_ / aivNum_;
    uint64_t remainder = actualA_ % aivNum_;
    uint64_t scanStart = static_cast<uint64_t>(aivId_) * tokensPerCore + ((aivId_ < remainder) ? aivId_ : remainder);
    uint64_t scanEnd = scanStart + tokensPerCore + ((aivId_ < remainder) ? 1U : 0U);
    LocalTensor<int32_t> metadataLocal = metadataBuf_.Get<int32_t>();
    const DataCopyPadExtParams<int32_t> metaPadParams{false, 0U, 0U, 0U};
    for (uint64_t chunkStart = scanStart; chunkStart < scanEnd; chunkStart += metaChunkTokens) {
        uint64_t chunkEnd = (chunkStart + metaChunkTokens > scanEnd) ? scanEnd : chunkStart + metaChunkTokens;
        uint32_t curChunkTokens = static_cast<uint32_t>(chunkEnd - chunkStart);
        DataCopyExtParams metaCopyParams{1U, static_cast<uint32_t>(curChunkTokens * metaBytesPerToken), 0U, 0U, 0U};
        DataCopyPad(metadataLocal, recvSrcMetadataGm_[chunkStart * RECV_META_FIELDS], metaCopyParams, metaPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        for (uint32_t i = 0; i < curChunkTokens; ++i) {
            int32_t srcRank = metadataLocal.GetValue(i * RECV_META_FIELDS + META_SRC_RANK_OFFSET);
            if (srcRank >= 0 && srcRank < static_cast<int32_t>(epWorldSize_)) {
                rankCounts.SetValue(srcRank, rankCounts.GetValue(srcRank) + 1);
            }
        }
        SyncFunc<AscendC::HardEvent::S_MTE2>();
    }

    GlobalTensor<int32_t> perCoreRankCounts;
    perCoreRankCounts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
        perCoreRankCountWorkspaceAddr_ + static_cast<uint64_t>(aivId_) * perCoreRankCountStride_));
    DataCopyExtParams rankCountCopyParams{1U, static_cast<uint32_t>(epWorldSize_ * sizeof(int32_t)), 0U, 0U, 0U};
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(perCoreRankCounts, rankCounts, rankCountCopyParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    SyncAll<true>();

    if (aivId_ == 0U) {
        Duplicate<int32_t>(rankOffsets_, 0, rankOffsetElements);
        SyncFunc<AscendC::HardEvent::V_MTE2>();
        GlobalTensor<int32_t> allPerCoreRankCounts;
        allPerCoreRankCounts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(perCoreRankCountWorkspaceAddr_));
        const DataCopyPadExtParams<int32_t> rankPadParams{false, 0U, 0U, 0U};
        for (uint32_t coreId = 0; coreId < aivNum_; ++coreId) {
            DataCopyPad(rankCounts,
                        allPerCoreRankCounts[static_cast<uint64_t>(coreId) * perCoreRankCountStride_ / sizeof(int32_t)],
                        rankCountCopyParams, rankPadParams);
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            Add(rankOffsets_, rankOffsets_, rankCounts, rankCountElements);
            if (coreId + 1U < aivNum_) {
                SyncFunc<AscendC::HardEvent::V_MTE2>();
            }
        }
        SyncFunc<AscendC::HardEvent::V_S>();
        int32_t rankPrefix = 0;
        for (uint32_t rank = 0; rank < epWorldSize_; ++rank) {
            int32_t rankCount = rankOffsets_.GetValue(rank);
            rankOffsets_.SetValue(rank, rankPrefix);
            rankPrefix += rankCount;
        }
        rankOffsets_.SetValue(epWorldSize_, rankPrefix);

        GlobalTensor<int32_t> rankOffsetsGm;
        rankOffsetsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(rankOffsetsWorkspaceAddr_));
        DataCopyExtParams rankOffsetCopyParams{1U, static_cast<uint32_t>((epWorldSize_ + 1U) * sizeof(int32_t)), 0U, 0U,
                                               0U};
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(rankOffsetsGm, rankOffsets_, rankOffsetCopyParams);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
    SyncAll<true>();

    GlobalTensor<int32_t> rankOffsetsGm;
    rankOffsetsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(rankOffsetsWorkspaceAddr_));
    DataCopyExtParams rankOffsetCopyParams{1U, static_cast<uint32_t>((epWorldSize_ + 1U) * sizeof(int32_t)), 0U, 0U,
                                           0U};
    const DataCopyPadExtParams<int32_t> rankOffsetPadParams{false, 0U, 0U, 0U};
    DataCopyPad(rankOffsets_, rankOffsetsGm, rankOffsetCopyParams, rankOffsetPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendMetadataRange(uint32_t targetRank,
                                                                                     uint32_t rowStart, uint32_t rowEnd)
{
    if (rowStart >= rowEnd) {
        return;
    }
    if (targetRank != rankId_) {
        BeginPreparedWrites(targetRank);
    }

    constexpr uint32_t metaBytesPerToken = RECV_META_FIELDS * sizeof(int32_t);
    uint32_t metaChunkTokens = (actualA_ < META_CHUNK_TOKEN_MAX) ? actualA_ : META_CHUNK_TOKEN_MAX;
    LocalTensor<int32_t> metadataLocal = metadataBuf_.Get<int32_t>();
    LocalTensor<uint32_t> weightLocal;
    if constexpr (HasTopkWeight == 1) {
        weightLocal = weightsBuf_.Get<uint32_t>();
    }
    const DataCopyPadExtParams<int32_t> metaPadParams{false, 0U, 0U, 0U};
    const DataCopyPadExtParams<uint32_t> weightPadParams{false, 0U, 0U, 0U};
    GM_ADDR dataBase = GetUrmaWinAddrByRankId(targetRank, combineDataWinOffset_);
    GM_ADDR stateBase = GetUrmaStateAddrByRankId(targetRank, combineStateWinOffset_);
    uint64_t tokenBytes = static_cast<uint64_t>(axisH_) * sizeof(XType);

    for (uint32_t chunkStart = rowStart; chunkStart < rowEnd; chunkStart += metaChunkTokens) {
        uint32_t chunkEnd = (chunkStart + metaChunkTokens > rowEnd) ? rowEnd : chunkStart + metaChunkTokens;
        uint32_t curRows = chunkEnd - chunkStart;
        DataCopyExtParams metaCopyParams{1U, static_cast<uint32_t>(curRows * metaBytesPerToken), 0U, 0U, 0U};
        DataCopyPad(metadataLocal, recvSrcMetadataGm_[static_cast<uint64_t>(chunkStart) * RECV_META_FIELDS],
                    metaCopyParams, metaPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        for (uint32_t i = 0; i < curRows; ++i) {
            uint32_t metaBase = i * RECV_META_FIELDS;
            int32_t srcRank = metadataLocal.GetValue(metaBase + META_SRC_RANK_OFFSET);
            int32_t srcTokenIdx = metadataLocal.GetValue(metaBase + META_TOKEN_IDX_OFFSET);
            int32_t srcTopKIdx = metadataLocal.GetValue(metaBase + META_TOPK_IDX_OFFSET);
            int32_t recvXIdx = metadataLocal.GetValue(metaBase + META_RECV_X_IDX_OFFSET);
            if (srcRank != static_cast<int32_t>(targetRank) || srcTokenIdx < 0 ||
                srcTokenIdx >= static_cast<int32_t>(numMaxTokensPerRank_) || srcTopKIdx < 0 ||
                srcTopKIdx >= static_cast<int32_t>(topK_) || recvXIdx < 0 ||
                recvXIdx >= static_cast<int32_t>(actualA_)) {
                continue;
            }

            uint32_t recvXRow = static_cast<uint32_t>(recvXIdx);
            if (targetRank == rankId_) {
                uint32_t weight = 0U;
                if constexpr (HasTopkWeight == 1) {
                    DataCopyExtParams weightCopyParams{1U, static_cast<uint32_t>(sizeof(uint32_t)), 0U, 0U, 0U};
                    DataCopyPad(weightLocal, topkWeightsGm_[recvXRow].template ReinterpretCast<uint32_t>(),
                                weightCopyParams, weightPadParams);
                    SyncFunc<AscendC::HardEvent::MTE2_S>();
                    weight = weightLocal.GetValue(0);
                    SyncFunc<AscendC::HardEvent::S_MTE2>();
                }
                SendLocalSlot(recvXRow, srcTokenIdx, srcTopKIdx, weight, dataBase, stateBase);
            } else {
                SendRemoteSlot(recvXRow, srcTokenIdx, srcTopKIdx, 0U, dataBase, stateBase, tokenBytes);
            }
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
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::BeginPreparedWrites(uint32_t dstRank)
{
    uint64_t commHandle = GetCommHandle(dstRank);
    if (activeBatchInitialized_ && activeBatchChannel_ == commHandle) {
        return;
    }
    if (activeBatchInitialized_) {
        FlushPreparedWrites();
    }
    activeBatchHandle_ =
        hcomm_.MakeBatchHandle(commHandle, hcommBatchTensor_, HCOMM_BATCH_BUFFER_BYTES, winRankAddr_[dstRank]);
    activeBatchChannel_ = commHandle;
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
}

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

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendRemoteSlot(
    uint32_t tokenIndex, int32_t srcTokenIdx, int32_t srcTopKIdx, uint32_t weight, GM_ADDR remoteDataBase,
    GM_ADDR remoteStateBase, uint64_t tokenBytes)
{
    uint64_t sendTokenOffset = (static_cast<uint64_t>(srcTokenIdx) * topK_ + srcTopKIdx) * perSlotBytes_;
    uint64_t recvStateOffset = (srcTokenIdx * topK_ + srcTopKIdx) * WIN_ADDR_ALIGN;
    GM_ADDR tokenAddr = (GM_ADDR)xGm_.GetPhyAddr(tokenIndex * axisH_);
    PrepareWrite<DEFAULT_WQE_CONFIG>(remoteDataBase + sendTokenOffset, tokenAddr, tokenBytes);
    if constexpr (HasTopkWeight == 1) {
        GM_ADDR weightAddr = (GM_ADDR)topkWeightsGm_.GetPhyAddr(tokenIndex);
        PrepareWrite<DEFAULT_WQE_CONFIG>(remoteStateBase + recvStateOffset, weightAddr, sizeof(uint32_t));
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendChannelFlag(uint32_t dstRank)
{
    uint64_t flagOffset = (static_cast<uint64_t>(numMaxTokensPerRank_) * topK_ + rankId_) * WIN_ADDR_ALIGN;
    GM_ADDR flagAddr = GetUrmaStateAddrByRankId(dstRank, combineStateWinOffset_) + flagOffset;
    if (dstRank == rankId_) {
        GlobalTensor<uint64_t> localFlag;
        localFlag.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(flagAddr));
        localFlag.SetValue(0, 1U);
        DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(localFlag);
        return;
    }
    BeginPreparedWrites(dstRank);
    PrepareWrite<CHANNEL_FLAG_WQE_CONFIG>(flagAddr, flagSourceAddr_, WIN_ADDR_ALIGN);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::GetCoreAssignment(uint32_t totalBlocks,
                                                                                     uint32_t &targetRank,
                                                                                     uint32_t &coreIndexInGroup,
                                                                                     uint32_t &groupSize)
{
    uint32_t maxChannelAivNum = epWorldSize_ * COMBINE_CHANNEL_COUNT;
    bool assignRemainingToLocal = totalBlocks > maxChannelAivNum;
    if (assignRemainingToLocal) {
        // Put all remote communication AIVs first, then give every remaining AIV to local copies.
        uint32_t remoteAivNum = (epWorldSize_ - 1U) * COMBINE_CHANNEL_COUNT;
        if (aivId_ < remoteAivNum) {
            uint32_t remoteRankIndex = aivId_ / COMBINE_CHANNEL_COUNT;
            targetRank = remoteRankIndex < rankId_ ? remoteRankIndex : remoteRankIndex + 1U;
            coreIndexInGroup = aivId_ % COMBINE_CHANNEL_COUNT;
            groupSize = COMBINE_CHANNEL_COUNT;
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
    // Build only the small rank boundary table. Token payloads stay in recv_x and are addressed by metadata column 5.
    BuildRankOffsets();

    activeBatchHandle_ = {};
    activeBatchChannel_ = 0;
    preparedWriteCount_ = 0;
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
            uint32_t rankStart = static_cast<uint32_t>(rankOffsets_.GetValue(targetRank));
            uint32_t rankEnd = static_cast<uint32_t>(rankOffsets_.GetValue(targetRank + 1U));
            uint32_t rankRows = rankEnd - rankStart;
            uint32_t rowsPerCore = rankRows / groupSize;
            uint32_t remainder = rankRows % groupSize;
            uint32_t rowStart = rankStart + coreIndexInGroup * rowsPerCore +
                                ((coreIndexInGroup < remainder) ? coreIndexInGroup : remainder);
            uint32_t rowEnd = rowStart + rowsPerCore + ((coreIndexInGroup < remainder) ? 1U : 0U);
            SendMetadataRange(targetRank, rowStart, rowEnd);
        } else {
            for (uint32_t rank = aivId_; rank < epWorldSize_; rank += activeAivNum) {
                uint32_t rankStart = static_cast<uint32_t>(rankOffsets_.GetValue(rank));
                uint32_t rankEnd = static_cast<uint32_t>(rankOffsets_.GetValue(rank + 1U));
                SendMetadataRange(rank, rankStart, rankEnd);
            }
        }
    }

    // Local rank rows may be split across more AIVs than the published channel count. Wait for every local copy
    // before any AIV publishes completion; remote payload and flag WQEs remain ordered within each batch handle.
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    SyncAll<true>();
    constexpr uint32_t completionChannelCount = COMBINE_CHANNEL_COUNT;
    bool publishesChannelFlag =
        aivId_ < activeAivNum && (!splitRankTokens || coreIndexInGroup < completionChannelCount);
    if (publishesChannelFlag) {
        LocalTensor<uint64_t> flagTensor = statusTensor_.ReinterpretCast<uint64_t>();
        Duplicate<uint64_t>(flagTensor, 1U, UB_ALIGN / sizeof(uint64_t));
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        GlobalTensor<uint64_t> flagSource;
        flagSource.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(flagSourceAddr_));
        DataCopy(flagSource, flagTensor, UB_ALIGN / sizeof(uint64_t));
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        if (splitRankTokens) {
            SendChannelFlag(targetRank);
        } else {
            for (uint32_t dstRank = aivId_; dstRank < epWorldSize_; dstRank += activeAivNum) {
                SendChannelFlag(dstRank);
            }
        }
    }
    FlushPreparedWrites();
    if (sendsTokens) {
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(recvSrcMetadataGm_);
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::Process()
{
    SendPhaseExpertToToken();
}

#endif

} // namespace MoeEpCombineImpl

#endif // MOE_EP_COMBINE_H
