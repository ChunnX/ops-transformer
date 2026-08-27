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
static constexpr uint32_t RECV_META_FIELDS = 4;
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
    __aicore__ inline void BuildAddressTable();
    __aicore__ inline uint32_t GetAddressTableCount(uint32_t targetRank);
    __aicore__ inline void SendAddressTableRange(uint32_t targetRank, uint32_t entryStart, uint32_t entryEnd);
    __aicore__ inline void SendPhaseExpertToToken();
    __aicore__ inline void GetCoreAssignment(uint32_t totalBlocks, uint32_t &targetRank, uint32_t &coreIndexInGroup,
                                             uint32_t &groupSize);

    __aicore__ inline uint64_t GetCommHandle(uint32_t rankId)
    {
        PipeBarrier<PIPE_ALL>();
        return hcommHandle_[rankId * channelsPerRank_];
    }
    __aicore__ inline GM_ADDR GetUrmaWinAddrByRankId(uint32_t rankId, uint64_t offset)
    {
        PipeBarrier<PIPE_ALL>();
        return (GM_ADDR)(winRankAddr_[rankId] + offset);
    }
    __aicore__ inline GM_ADDR GetUrmaStateAddrByRankId(uint32_t rankId, uint64_t offset)
    {
        PipeBarrier<PIPE_ALL>();
        return (GM_ADDR)(winRankAddr_[rankId] + offset);
    }
    __aicore__ inline GM_ADDR GetLocalSendDataWorkspaceAddr(const int32_t rankId)
    {
        PipeBarrier<PIPE_ALL>();
        return combineSendDataWorkspaceAddr_ + sendDataWorkspaceSizePerRank_ * rankId;
    }
    __aicore__ inline uint32_t ReduceSumWorkNeedSize(int32_t count, int32_t typeSize)
    {
        PipeBarrier<PIPE_ALL>();
        int32_t elementsPerBlock = UB_ALIGN / typeSize;
        PipeBarrier<PIPE_ALL>();
        int32_t elementsPerRepeat = ALIGNED_LEN_256 / typeSize;
        PipeBarrier<PIPE_ALL>();
        int32_t iter1OutputCount = (count + elementsPerRepeat - 1) / elementsPerRepeat;
        PipeBarrier<PIPE_ALL>();
        uint32_t iter1AlignEnd = ((iter1OutputCount + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
        PipeBarrier<PIPE_ALL>();
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
    uint64_t sendDataWorkspaceSizePerRank_{0};

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
    TBuf<> addressTableBuf_;
    TBuf<> rankInfoBuf_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_; // 数据队列
    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;                 // 通信上下文
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
    bool activeBatchInitialized_{false};
};

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx,
                                                                        GM_ADDR recvSrcMetadata,
                                                                        GM_ADDR numRecvPerExpert, GM_ADDR topkWeights,
                                                                        GM_ADDR workspace, GM_ADDR tilingGM,
                                                                        TPipe *pipe, const MoeEpCombineInfo *tilingData)
{
    PipeBarrier<PIPE_ALL>();
    tpipe_ = pipe;
    PipeBarrier<PIPE_ALL>();
    tilingData_ = tilingData;
    PipeBarrier<PIPE_ALL>();
    aivId_ = GetBlockIdx();
    PipeBarrier<PIPE_ALL>();
    combineSendDataWorkspaceAddr_ = workspace;
    PipeBarrier<PIPE_ALL>();
    epWorldSize_ = tilingData_->cfg.epWorldSize;
    PipeBarrier<PIPE_ALL>();
    numMaxTokensPerRank_ = tilingData_->cfg.numMaxTokensPerRank;
    PipeBarrier<PIPE_ALL>();
    topK_ = tilingData_->cfg.topK;
    PipeBarrier<PIPE_ALL>();
    axisH_ = tilingData_->cfg.hidden;
    PipeBarrier<PIPE_ALL>();
    perSlotBytes_ = tilingData_->cfg.perSlotBytes;
    PipeBarrier<PIPE_ALL>();
    aivNum_ = tilingData_->aivNum;
    PipeBarrier<PIPE_ALL>();
    localmoeNum_ = tilingData_->cfg.numLocalExperts;
    PipeBarrier<PIPE_ALL>();
    sendDataWorkspaceSizePerRank_ = tilingData->sendDataWorkspaceSizePerRank;
    PipeBarrier<PIPE_ALL>();
    flagWorkspaceAddr_ = combineSendDataWorkspaceAddr_ + epWorldSize_ * sendDataWorkspaceSizePerRank_ +
                         static_cast<uint64_t>(aivId_) * WIN_ADDR_ALIGN;
    PipeBarrier<PIPE_ALL>();
    rankCountWorkspaceAddr_ = combineSendDataWorkspaceAddr_ + epWorldSize_ * sendDataWorkspaceSizePerRank_ +
                              static_cast<uint64_t>(aivNum_) * WIN_ADDR_ALIGN;
    PipeBarrier<PIPE_ALL>();
    perCoreRankCountStride_ = Ceil(static_cast<uint64_t>(epWorldSize_) * UB_ALIGN, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    PipeBarrier<PIPE_ALL>();
    perCoreRankCountWorkspaceAddr_ = rankCountWorkspaceAddr_ + static_cast<uint64_t>(epWorldSize_) * WIN_ADDR_ALIGN;
    PipeBarrier<PIPE_ALL>();
    hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN; // UB 32字节对齐
    PipeBarrier<PIPE_ALL>();
    hWeightAlignSize_ = hAlignSize_ + UB_ALIGN; // UB 32字节对齐
    PipeBarrier<PIPE_ALL>();
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    PipeBarrier<PIPE_ALL>();
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    PipeBarrier<PIPE_ALL>();
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);
    PipeBarrier<PIPE_ALL>();
    tpipe_->InitBuffer(hcommBatchBuf_, HCOMM_BATCH_BUFFER_BYTES);
    PipeBarrier<PIPE_ALL>();
    hcommBatchTensor_ = hcommBatchBuf_.Get<uint8_t>();
    PipeBarrier<PIPE_ALL>();
    Duplicate<uint8_t>(hcommBatchTensor_, 0U, HCOMM_BATCH_BUFFER_BYTES);
    PipeBarrier<PIPE_ALL>();
    SyncFunc<AscendC::HardEvent::V_S>();

    PipeBarrier<PIPE_ALL>();
    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::MoeCommContext *>(context);
    PipeBarrier<PIPE_ALL>();
    rankId_ = mc2Context_->epRankId;
    PipeBarrier<PIPE_ALL>();
    channelsPerRank_ = mc2Context_->channelsPerRank;
    PipeBarrier<PIPE_ALL>();
    if (channelsPerRank_ == 0 || (epWorldSize_ > 0 && channelsPerRank_ > Mc2Aclnn::HCCL_MAX_RANK_SIZE / epWorldSize_)) {
        PipeBarrier<PIPE_ALL>();
        channelsPerRank_ = 1;
    }
    PipeBarrier<PIPE_ALL>();
    for (uint32_t i = 0; i < epWorldSize_; ++i) {
        PipeBarrier<PIPE_ALL>();
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
    }
    PipeBarrier<PIPE_ALL>();
    uint32_t handleCount = epWorldSize_ * channelsPerRank_;
    PipeBarrier<PIPE_ALL>();
    for (uint32_t i = 0; i < handleCount; ++i) {
        PipeBarrier<PIPE_ALL>();
        hcommHandle_[i] = mc2Context_->hcommHandle[i];
    }

    PipeBarrier<PIPE_ALL>();
    combineStateWinOffset_ = tilingData->combineStateWinOffset;
    PipeBarrier<PIPE_ALL>();
    combineDataWinOffset_ = tilingData->combineDataWinOffset;

    PipeBarrier<PIPE_ALL>();
    xGm_.SetGlobalBuffer((__gm__ XType *)x);
    PipeBarrier<PIPE_ALL>();
    recvSrcMetadataGm_.SetGlobalBuffer((__gm__ int32_t *)recvSrcMetadata);
    PipeBarrier<PIPE_ALL>();
    numRecvPerExpertGm_.SetGlobalBuffer((__gm__ int64_t *)numRecvPerExpert);

    // 计算actualA_的大小
    PipeBarrier<PIPE_ALL>();
    uint32_t numRecvBytes = Ceil(localmoeNum_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
    PipeBarrier<PIPE_ALL>();
    uint32_t reduceTmpBytes = ReduceSumWorkNeedSize(localmoeNum_, sizeof(int64_t)) * sizeof(int64_t);
    PipeBarrier<PIPE_ALL>();
    TBuf<TPosition::VECCALC> numRecvBuf;
    PipeBarrier<PIPE_ALL>();
    tpipe_->InitBuffer(numRecvBuf, numRecvBytes + reduceTmpBytes + UB_ALIGN); // 数据 + ReduceSum tmp + 输出
    PipeBarrier<PIPE_ALL>();
    LocalTensor<int64_t> numRecvLocal = numRecvBuf.Get<int64_t>();

    PipeBarrier<PIPE_ALL>();
    DataCopyExtParams numRecvCopyParams{1U, static_cast<uint32_t>(localmoeNum_ * sizeof(int64_t)), 0U, 0U, 0U};
    PipeBarrier<PIPE_ALL>();
    DataCopyPadExtParams<int64_t> numRecvPadParams{false, 0U, 0U, 0U};
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(numRecvLocal, numRecvPerExpertGm_, numRecvCopyParams, numRecvPadParams);
    PipeBarrier<PIPE_ALL>();
    SyncFunc<AscendC::HardEvent::MTE2_V>();

    PipeBarrier<PIPE_ALL>();
    LocalTensor<int64_t> reduceTmp = numRecvLocal[numRecvBytes / sizeof(int64_t)];
    PipeBarrier<PIPE_ALL>();
    LocalTensor<int64_t> sumOut = reduceTmp[reduceTmpBytes / sizeof(int64_t)];
    PipeBarrier<PIPE_ALL>();
    ReduceSum<int64_t>(sumOut, numRecvLocal, reduceTmp, static_cast<int32_t>(localmoeNum_));
    PipeBarrier<PIPE_ALL>();
    SyncFunc<AscendC::HardEvent::V_S>();
    PipeBarrier<PIPE_ALL>();
    actualA_ = static_cast<uint64_t>(sumOut.GetValue(0));
    PipeBarrier<PIPE_ALL>();
    XTypeAlign32Size_ = hAlignSize_;
    PipeBarrier<PIPE_ALL>();
    if constexpr (HasTopkWeight == 1) {
        XTypeAlign32Size_ = hWeightAlignSize_;
        PipeBarrier<PIPE_ALL>();
        topkWeightsGm_.SetGlobalBuffer((__gm__ float *)topkWeights);
    }
    PipeBarrier<PIPE_ALL>();
    tpipe_->InitBuffer(xQueue_, SEND_DOUBLE_BUFFER_NUM, XTypeAlign32Size_);
    PipeBarrier<PIPE_ALL>();
    tpipe_->InitBuffer(readStateBuf_, UB_ALIGN); // 32
    PipeBarrier<PIPE_ALL>();
    statusTensor_ = readStateBuf_.Get<uint32_t>();
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::BuildAddressTable()
{
    // Follow dispatch PR 10309: count per core, derive an exclusive prefix, then batch-write records.
    PipeBarrier<PIPE_ALL>();
    constexpr uint32_t metaBytesPerToken = RECV_META_FIELDS * sizeof(int32_t);
    PipeBarrier<PIPE_ALL>();
    constexpr uint32_t rankCountBlockElements = UB_ALIGN / sizeof(int32_t);
    PipeBarrier<PIPE_ALL>();
    uint32_t metaChunkTokens = (actualA_ < META_CHUNK_TOKEN_MAX) ? actualA_ : META_CHUNK_TOKEN_MAX;
    PipeBarrier<PIPE_ALL>();
    metaChunkTokens = (metaChunkTokens == 0U) ? 1U : metaChunkTokens;
    PipeBarrier<PIPE_ALL>();
    uint32_t metaChunkBytes = Ceil(metaChunkTokens * metaBytesPerToken, UB_ALIGN) * UB_ALIGN;
    PipeBarrier<PIPE_ALL>();
    uint32_t rankCountElements = epWorldSize_ * rankCountBlockElements;
    PipeBarrier<PIPE_ALL>();
    tpipe_->InitBuffer(metadataBuf_, metaChunkBytes);
    PipeBarrier<PIPE_ALL>();
    if constexpr (HasTopkWeight == 1) {
        uint32_t weightChunkBytes = Ceil(metaChunkTokens * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
        PipeBarrier<PIPE_ALL>();
        tpipe_->InitBuffer(weightsBuf_, weightChunkBytes);
    }
    PipeBarrier<PIPE_ALL>();
    tpipe_->InitBuffer(addressTableBuf_, ADDRESS_TABLE_BUFFER_BYTES);
    PipeBarrier<PIPE_ALL>();
    tpipe_->InitBuffer(rankInfoBuf_, 3U * rankCountElements * sizeof(int32_t));

    PipeBarrier<PIPE_ALL>();
    LocalTensor<int32_t> rankCounts = rankInfoBuf_.Get<int32_t>();
    PipeBarrier<PIPE_ALL>();
    LocalTensor<int32_t> rankPrefix = rankCounts[rankCountElements];
    PipeBarrier<PIPE_ALL>();
    LocalTensor<int32_t> prefixSums = rankCounts[2U * rankCountElements];
    PipeBarrier<PIPE_ALL>();
    Duplicate<int32_t>(rankCounts, 0, rankCountElements);
    PipeBarrier<PIPE_ALL>();
    Duplicate<int32_t>(rankPrefix, 0, rankCountElements);
    PipeBarrier<PIPE_ALL>();
    SyncFunc<AscendC::HardEvent::V_S>();

    PipeBarrier<PIPE_ALL>();
    DataCopyParams rankCountCopyParams{static_cast<uint16_t>(epWorldSize_), static_cast<uint16_t>(UB_ALIGN), 0U,
                                       static_cast<uint16_t>(WIN_ADDR_ALIGN - UB_ALIGN)};

    PipeBarrier<PIPE_ALL>();
    uint64_t tokensPerCore = actualA_ / aivNum_;
    PipeBarrier<PIPE_ALL>();
    uint64_t remainder = actualA_ % aivNum_;
    PipeBarrier<PIPE_ALL>();
    uint64_t scanStart = static_cast<uint64_t>(aivId_) * tokensPerCore + ((aivId_ < remainder) ? aivId_ : remainder);
    PipeBarrier<PIPE_ALL>();
    uint64_t scanEnd = scanStart + tokensPerCore + ((aivId_ < remainder) ? 1U : 0U);

    PipeBarrier<PIPE_ALL>();
    LocalTensor<int32_t> metadataLocal = metadataBuf_.Get<int32_t>();
    PipeBarrier<PIPE_ALL>();
    const DataCopyPadExtParams<int32_t> metaPadParams{false, 0U, 0U, 0U};
    PipeBarrier<PIPE_ALL>();
    for (uint64_t chunkStart = scanStart; chunkStart < scanEnd; chunkStart += metaChunkTokens) {
        PipeBarrier<PIPE_ALL>();
        uint64_t chunkEnd = (chunkStart + metaChunkTokens > scanEnd) ? scanEnd : chunkStart + metaChunkTokens;
        PipeBarrier<PIPE_ALL>();
        uint32_t curChunkTokens = static_cast<uint32_t>(chunkEnd - chunkStart);

        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams metaCopyParams{1U, curChunkTokens * metaBytesPerToken, 0U, 0U, 0U};
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(metadataLocal, recvSrcMetadataGm_[chunkStart * RECV_META_FIELDS], metaCopyParams, metaPadParams);
        PipeBarrier<PIPE_ALL>();
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        PipeBarrier<PIPE_ALL>();
        for (uint32_t i = 0; i < curChunkTokens; ++i) {
            PipeBarrier<PIPE_ALL>();
            int32_t srcRank = metadataLocal.GetValue(i * RECV_META_FIELDS);
            PipeBarrier<PIPE_ALL>();
            if (srcRank < 0 || srcRank >= static_cast<int32_t>(epWorldSize_) ||
                srcRank == static_cast<int32_t>(rankId_)) {
                PipeBarrier<PIPE_ALL>();
                continue;
            }
            PipeBarrier<PIPE_ALL>();
            uint32_t countOffset = static_cast<uint32_t>(srcRank) * rankCountBlockElements;
            PipeBarrier<PIPE_ALL>();
            rankCounts.SetValue(countOffset, rankCounts.GetValue(countOffset) + 1);
        }
        PipeBarrier<PIPE_ALL>();
        SyncFunc<AscendC::HardEvent::S_MTE2>();
    }

    // Publish this core's row. Address-table positions and totals are derived from these rows without atomic ops.
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<int32_t> perCoreRankCounts;
    PipeBarrier<PIPE_ALL>();
    perCoreRankCounts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
        perCoreRankCountWorkspaceAddr_ + static_cast<uint64_t>(aivId_) * perCoreRankCountStride_));
    PipeBarrier<PIPE_ALL>();
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    PipeBarrier<PIPE_ALL>();
    DataCopy(perCoreRankCounts, rankCounts, rankCountElements);
    PipeBarrier<PIPE_ALL>();
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();

    // Group preceding rows into a few MTE2 copies, as in PR 10309 GetSlotStartNum().
    PipeBarrier<PIPE_ALL>();
    uint32_t rankCountRowBytes = rankCountElements * sizeof(int32_t);
    PipeBarrier<PIPE_ALL>();
    uint32_t prefixRowsPerBatch = ADDRESS_TABLE_BUFFER_BYTES / rankCountRowBytes;
    PipeBarrier<PIPE_ALL>();
    LocalTensor<int32_t> prefixRows = addressTableBuf_.Get<int32_t>();
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<int32_t> allPerCoreRankCounts;
    PipeBarrier<PIPE_ALL>();
    allPerCoreRankCounts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(perCoreRankCountWorkspaceAddr_));
    PipeBarrier<PIPE_ALL>();
    const DataCopyPadExtParams<int32_t> prefixPadParams{false, 0U, 0U, 0U};
    PipeBarrier<PIPE_ALL>();
    for (uint32_t coreStart = 0; coreStart < aivId_; coreStart += prefixRowsPerBatch) {
        PipeBarrier<PIPE_ALL>();
        uint32_t copyRows = (coreStart + prefixRowsPerBatch > aivId_) ? aivId_ - coreStart : prefixRowsPerBatch;
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams prefixCopyParams{static_cast<uint16_t>(copyRows), rankCountRowBytes,
                                           static_cast<uint32_t>(perCoreRankCountStride_ - rankCountRowBytes), 0U, 0U};
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(prefixRows,
                    allPerCoreRankCounts[static_cast<uint64_t>(coreStart) * perCoreRankCountStride_ / sizeof(int32_t)],
                    prefixCopyParams, prefixPadParams);
        PipeBarrier<PIPE_ALL>();
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        PipeBarrier<PIPE_ALL>();
        const uint32_t prefixShape[] = {copyRows, rankCountElements};
        PipeBarrier<PIPE_ALL>();
        ReduceSum<int32_t, AscendC::Pattern::Reduce::RA, true>(prefixSums, prefixRows, prefixShape, false);
        PipeBarrier<PIPE_ALL>();
        Add(rankPrefix, rankPrefix, prefixSums, rankCountElements);
        PipeBarrier<PIPE_ALL>();
        if (coreStart + copyRows < aivId_) {
            PipeBarrier<PIPE_ALL>();
            SyncFunc<AscendC::HardEvent::V_MTE2>();
        }
    }
    PipeBarrier<PIPE_ALL>();
    SyncFunc<AscendC::HardEvent::V_S>();

    // rankCounts still holds this core's row. The last AIV writes prefix + own count as each remote rank's total.
    PipeBarrier<PIPE_ALL>();
    if (aivId_ == aivNum_ - 1U) {
        PipeBarrier<PIPE_ALL>();
        for (uint32_t rank = 0; rank < epWorldSize_; ++rank) {
            PipeBarrier<PIPE_ALL>();
            uint32_t countOffset = rank * rankCountBlockElements;
            PipeBarrier<PIPE_ALL>();
            rankCounts.SetValue(countOffset, rankPrefix.GetValue(countOffset) + rankCounts.GetValue(countOffset));
        }
        PipeBarrier<PIPE_ALL>();
        GlobalTensor<int32_t> totalRankCounts;
        PipeBarrier<PIPE_ALL>();
        totalRankCounts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(rankCountWorkspaceAddr_));
        PipeBarrier<PIPE_ALL>();
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(totalRankCounts, rankCounts, rankCountCopyParams);
        PipeBarrier<PIPE_ALL>();
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }

    // Each group fits even if all records target the same rank. Records are grouped by rank in UB.
    PipeBarrier<PIPE_ALL>();
    uint32_t entriesPerRank = ADDRESS_TABLE_BUFFER_BYTES / epWorldSize_ / ADDRESS_ENTRY_UB_BYTES;
    PipeBarrier<PIPE_ALL>();
    LocalTensor<uint32_t> addressEntries = addressTableBuf_.Get<uint32_t>();
    PipeBarrier<PIPE_ALL>();
    constexpr uint32_t entryElements = ADDRESS_ENTRY_UB_BYTES / sizeof(uint32_t);
    // DataCopyParams uses 32-byte blocks, unlike the byte-based DataCopyPad parameters above.
    PipeBarrier<PIPE_ALL>();
    DataCopyParams addressCopyParams{
        0U, static_cast<uint16_t>(ADDRESS_ENTRY_UB_BYTES / UB_ALIGN), 0U,
        static_cast<uint16_t>((ADDRESS_ENTRY_GM_BYTES - ADDRESS_ENTRY_UB_BYTES) / UB_ALIGN)};
    PipeBarrier<PIPE_ALL>();
    GM_ADDR localDataBase = GetUrmaWinAddrByRankId(rankId_, combineDataWinOffset_);
    PipeBarrier<PIPE_ALL>();
    GM_ADDR localStateBase = GetUrmaStateAddrByRankId(rankId_, combineStateWinOffset_);
    PipeBarrier<PIPE_ALL>();
    for (uint64_t groupStart = scanStart; groupStart < scanEnd; groupStart += entriesPerRank) {
        PipeBarrier<PIPE_ALL>();
        uint64_t groupEnd = (groupStart + entriesPerRank > scanEnd) ? scanEnd : groupStart + entriesPerRank;
        PipeBarrier<PIPE_ALL>();
        uint32_t groupTokens = static_cast<uint32_t>(groupEnd - groupStart);
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams metaCopyParams{1U, groupTokens * metaBytesPerToken, 0U, 0U, 0U};
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(metadataLocal, recvSrcMetadataGm_[groupStart * RECV_META_FIELDS], metaCopyParams, metaPadParams);

        PipeBarrier<PIPE_ALL>();
        LocalTensor<uint32_t> weightsLocal;
        PipeBarrier<PIPE_ALL>();
        if constexpr (HasTopkWeight == 1) {
            weightsLocal = weightsBuf_.Get<uint32_t>();
            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams weightCopyParams{1U, static_cast<uint32_t>(groupTokens * sizeof(uint32_t)), 0U, 0U, 0U};
            PipeBarrier<PIPE_ALL>();
            const DataCopyPadExtParams<uint32_t> weightPadParams{false, 0U, 0U, 0U};
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(weightsLocal, topkWeightsGm_[groupStart].template ReinterpretCast<uint32_t>(), weightCopyParams,
                        weightPadParams);
        }
        PipeBarrier<PIPE_ALL>();
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        // rankCounts was updated by scalar in the counting pass. Drain scalar accesses before V clears it.
        PipeBarrier<PIPE_ALL>();
        SyncFunc<AscendC::HardEvent::S_V>();
        PipeBarrier<PIPE_ALL>();
        Duplicate<int32_t>(rankCounts, 0, rankCountElements);
        PipeBarrier<PIPE_ALL>();
        SyncFunc<AscendC::HardEvent::V_S>();
        PipeBarrier<PIPE_ALL>();
        for (uint32_t i = 0; i < groupTokens; ++i) {
            PipeBarrier<PIPE_ALL>();
            int32_t srcRank = metadataLocal.GetValue(i * RECV_META_FIELDS);
            PipeBarrier<PIPE_ALL>();
            if (srcRank < 0 || srcRank >= static_cast<int32_t>(epWorldSize_)) {
                PipeBarrier<PIPE_ALL>();
                continue;
            }
            PipeBarrier<PIPE_ALL>();
            uint32_t tokenIndex = static_cast<uint32_t>(groupStart) + i;
            PipeBarrier<PIPE_ALL>();
            int32_t srcTokenIdx = metadataLocal.GetValue(i * RECV_META_FIELDS + 1);
            PipeBarrier<PIPE_ALL>();
            int32_t srcTopKIdx = metadataLocal.GetValue(i * RECV_META_FIELDS + 2);
            PipeBarrier<PIPE_ALL>();
            uint32_t weight = 0U;
            PipeBarrier<PIPE_ALL>();
            if constexpr (HasTopkWeight == 1) {
                weight = weightsLocal.GetValue(i);
            }
            PipeBarrier<PIPE_ALL>();
            if (srcRank == static_cast<int32_t>(rankId_)) {
                PipeBarrier<PIPE_ALL>();
                SendLocalSlot(tokenIndex, srcTokenIdx, srcTopKIdx, weight, localDataBase, localStateBase);
                PipeBarrier<PIPE_ALL>();
                continue;
            }
            PipeBarrier<PIPE_ALL>();
            uint32_t countOffset = static_cast<uint32_t>(srcRank) * rankCountBlockElements;
            PipeBarrier<PIPE_ALL>();
            uint32_t localIndex = static_cast<uint32_t>(rankCounts.GetValue(countOffset));
            PipeBarrier<PIPE_ALL>();
            uint32_t entryOffset = (static_cast<uint32_t>(srcRank) * entriesPerRank + localIndex) * entryElements;
            PipeBarrier<PIPE_ALL>();
            addressEntries.SetValue(entryOffset, tokenIndex);
            PipeBarrier<PIPE_ALL>();
            addressEntries.SetValue(entryOffset + 1, static_cast<uint32_t>(srcTokenIdx));
            PipeBarrier<PIPE_ALL>();
            addressEntries.SetValue(entryOffset + 2, static_cast<uint32_t>(srcTopKIdx));
            PipeBarrier<PIPE_ALL>();
            addressEntries.SetValue(entryOffset + 3, weight);
            PipeBarrier<PIPE_ALL>();
            rankCounts.SetValue(countOffset, static_cast<int32_t>(localIndex + 1));
        }

        PipeBarrier<PIPE_ALL>();
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        PipeBarrier<PIPE_ALL>();
        for (uint32_t rank = 0; rank < epWorldSize_; ++rank) {
            PipeBarrier<PIPE_ALL>();
            uint32_t countOffset = rank * rankCountBlockElements;
            PipeBarrier<PIPE_ALL>();
            uint32_t count = static_cast<uint32_t>(rankCounts.GetValue(countOffset));
            PipeBarrier<PIPE_ALL>();
            if (count == 0U) {
                PipeBarrier<PIPE_ALL>();
                continue;
            }
            PipeBarrier<PIPE_ALL>();
            uint32_t prefixOffset = rank * rankCountBlockElements;
            PipeBarrier<PIPE_ALL>();
            uint32_t writeStart = static_cast<uint32_t>(rankPrefix.GetValue(prefixOffset));
            PipeBarrier<PIPE_ALL>();
            GlobalTensor<uint32_t> addressTable;
            PipeBarrier<PIPE_ALL>();
            addressTable.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(GetLocalSendDataWorkspaceAddr(rank)));
            PipeBarrier<PIPE_ALL>();
            addressCopyParams.blockCount = static_cast<uint16_t>(count);
            PipeBarrier<PIPE_ALL>();
            DataCopy(addressTable[static_cast<uint64_t>(writeStart) * ADDRESS_ENTRY_GM_BYTES / sizeof(uint32_t)],
                     addressEntries[rank * entriesPerRank * entryElements], addressCopyParams);
            PipeBarrier<PIPE_ALL>();
            rankPrefix.SetValue(prefixOffset, static_cast<int32_t>(writeStart + count));
        }
        PipeBarrier<PIPE_ALL>();
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        PipeBarrier<PIPE_ALL>();
        SyncFunc<AscendC::HardEvent::S_MTE2>();
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline uint32_t MoeEpCombine<TemplateMoeEpCombineTypeFunc>::GetAddressTableCount(uint32_t targetRank)
{
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<uint32_t> totalRankCounts;
    PipeBarrier<PIPE_ALL>();
    totalRankCounts.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(rankCountWorkspaceAddr_));
    PipeBarrier<PIPE_ALL>();
    DataCopy(statusTensor_, totalRankCounts[static_cast<uint64_t>(targetRank) * WIN_ADDR_ALIGN / sizeof(uint32_t)],
             UB_ALIGN / sizeof(uint32_t));
    PipeBarrier<PIPE_ALL>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    PipeBarrier<PIPE_ALL>();
    uint32_t count = statusTensor_.GetValue(0);
    PipeBarrier<PIPE_ALL>();
    SyncFunc<AscendC::HardEvent::S_MTE2>();
    PipeBarrier<PIPE_ALL>();
    return count;
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendAddressTableRange(uint32_t targetRank,
                                                                                         uint32_t entryStart,
                                                                                         uint32_t entryEnd)
{
    PipeBarrier<PIPE_ALL>();
    if (entryStart >= entryEnd) {
        PipeBarrier<PIPE_ALL>();
        return;
    }
    PipeBarrier<PIPE_ALL>();
    if (targetRank != rankId_) {
        PipeBarrier<PIPE_ALL>();
        BeginPreparedWrites(targetRank);
    }

    PipeBarrier<PIPE_ALL>();
    uint32_t metaChunkTokens = (actualA_ < META_CHUNK_TOKEN_MAX) ? actualA_ : META_CHUNK_TOKEN_MAX;
    PipeBarrier<PIPE_ALL>();
    uint32_t metaChunkBytes = Ceil(metaChunkTokens * RECV_META_FIELDS * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    PipeBarrier<PIPE_ALL>();
    uint32_t entryChunkMax = metaChunkBytes / ADDRESS_ENTRY_UB_BYTES;
    PipeBarrier<PIPE_ALL>();
    LocalTensor<uint32_t> entriesLocal = metadataBuf_.Get<uint32_t>();
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<uint32_t> addressTable;
    PipeBarrier<PIPE_ALL>();
    addressTable.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(GetLocalSendDataWorkspaceAddr(targetRank)));
    PipeBarrier<PIPE_ALL>();
    const DataCopyPadExtParams<uint32_t> padParams{false, 0U, 0U, 0U};
    PipeBarrier<PIPE_ALL>();
    constexpr uint32_t localEntryStride = ADDRESS_ENTRY_UB_BYTES / sizeof(uint32_t);
    PipeBarrier<PIPE_ALL>();
    constexpr uint32_t globalEntryStride = ADDRESS_ENTRY_GM_BYTES / sizeof(uint32_t);
    PipeBarrier<PIPE_ALL>();
    GM_ADDR remoteDataBase = GetUrmaWinAddrByRankId(targetRank, combineDataWinOffset_);
    PipeBarrier<PIPE_ALL>();
    GM_ADDR remoteStateBase = GetUrmaStateAddrByRankId(targetRank, combineStateWinOffset_);
    PipeBarrier<PIPE_ALL>();
    uint64_t tokenBytes = static_cast<uint64_t>(axisH_) * sizeof(XType);

    PipeBarrier<PIPE_ALL>();
    for (uint32_t chunkStart = entryStart; chunkStart < entryEnd; chunkStart += entryChunkMax) {
        PipeBarrier<PIPE_ALL>();
        uint32_t chunkEnd = (chunkStart + entryChunkMax > entryEnd) ? entryEnd : chunkStart + entryChunkMax;
        PipeBarrier<PIPE_ALL>();
        uint32_t curEntries = chunkEnd - chunkStart;
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams copyParams{static_cast<uint16_t>(curEntries), ADDRESS_ENTRY_UB_BYTES,
                                     ADDRESS_ENTRY_GM_BYTES - ADDRESS_ENTRY_UB_BYTES, 0U, 0U};
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(entriesLocal, addressTable[chunkStart * globalEntryStride], copyParams, padParams);
        PipeBarrier<PIPE_ALL>();
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        PipeBarrier<PIPE_ALL>();
        for (uint32_t i = 0; i < curEntries; ++i) {
            PipeBarrier<PIPE_ALL>();
            uint32_t offset = i * localEntryStride;
            PipeBarrier<PIPE_ALL>();
            uint32_t tokenIndex = entriesLocal.GetValue(offset);
            PipeBarrier<PIPE_ALL>();
            int32_t srcTokenIdx = static_cast<int32_t>(entriesLocal.GetValue(offset + 1));
            PipeBarrier<PIPE_ALL>();
            int32_t srcTopKIdx = static_cast<int32_t>(entriesLocal.GetValue(offset + 2));
            PipeBarrier<PIPE_ALL>();
            uint32_t weight = entriesLocal.GetValue(offset + 3);
            PipeBarrier<PIPE_ALL>();
            SendRemoteSlot(tokenIndex, srcTokenIdx, srcTopKIdx, weight, remoteDataBase, remoteStateBase, tokenBytes);
        }
        PipeBarrier<PIPE_ALL>();
        SyncFunc<AscendC::HardEvent::S_MTE2>();
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendLocalToken(uint32_t tokenIndex, GM_ADDR dstAddr)
{
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<XType> outToken;
    PipeBarrier<PIPE_ALL>();
    outToken.SetGlobalBuffer((__gm__ XType *)dstAddr);

    PipeBarrier<PIPE_ALL>();
    DataCopyPadParams padParams = {false, 0, 0, 0};
    PipeBarrier<PIPE_ALL>();
    DataCopyParams copyParams = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    PipeBarrier<PIPE_ALL>();
    LocalTensor<XType> tokenTensor = xQueue_.AllocTensor<XType>();
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(tokenTensor, xGm_[tokenIndex * axisH_], copyParams, padParams);
    PipeBarrier<PIPE_ALL>();
    xQueue_.EnQue(tokenTensor);
    PipeBarrier<PIPE_ALL>();
    tokenTensor = xQueue_.DeQue<XType>();
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(outToken, tokenTensor, copyParams);
    PipeBarrier<PIPE_ALL>();
    xQueue_.FreeTensor<XType>(tokenTensor);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::FlushPreparedWrites(bool keepHandle)
{
    PipeBarrier<PIPE_ALL>();
    if (preparedWriteCount_ != 0) {
        PipeBarrier<PIPE_ALL>();
        (void)hcomm_.BatchCommit(activeBatchHandle_);
        PipeBarrier<PIPE_ALL>();
        preparedWriteCount_ = 0;
    }
    PipeBarrier<PIPE_ALL>();
    if (!keepHandle) {
        PipeBarrier<PIPE_ALL>();
        activeBatchHandle_ = {};
        PipeBarrier<PIPE_ALL>();
        activeBatchChannel_ = 0;
        PipeBarrier<PIPE_ALL>();
        activeBatchInitialized_ = false;
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::BeginPreparedWrites(uint32_t dstRank)
{
    PipeBarrier<PIPE_ALL>();
    uint64_t commHandle = GetCommHandle(dstRank);
    PipeBarrier<PIPE_ALL>();
    if (activeBatchInitialized_ && activeBatchChannel_ == commHandle) {
        PipeBarrier<PIPE_ALL>();
        return;
    }
    PipeBarrier<PIPE_ALL>();
    if (activeBatchInitialized_) {
        PipeBarrier<PIPE_ALL>();
        FlushPreparedWrites();
    }
    PipeBarrier<PIPE_ALL>();
    activeBatchHandle_ =
        hcomm_.MakeBatchHandle(commHandle, hcommBatchTensor_, HCOMM_BATCH_BUFFER_BYTES, winRankAddr_[dstRank]);
    PipeBarrier<PIPE_ALL>();
    activeBatchChannel_ = commHandle;
    PipeBarrier<PIPE_ALL>();
    activeBatchInitialized_ = true;
}

template <TemplateMoeEpCombineTypeClass>
template <auto const &config>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::PrepareWrite(GM_ADDR dst, GM_ADDR src, uint64_t len)
{
    PipeBarrier<PIPE_ALL>();
    if (preparedWriteCount_ == HCOMM_BATCH_CAPACITY) {
        // BatchCommit resets the WQE count in the handle; keep using it for this channel as PR 10309 does.
        PipeBarrier<PIPE_ALL>();
        FlushPreparedWrites(true);
    }
    PipeBarrier<PIPE_ALL>();
    (void)hcomm_.WriteNbi<config>(activeBatchHandle_, dst, src, len);
    PipeBarrier<PIPE_ALL>();
    ++preparedWriteCount_;
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendLocalSlot(uint32_t tokenIndex,
                                                                                 int32_t srcTokenIdx,
                                                                                 int32_t srcTopKIdx, uint32_t weight,
                                                                                 GM_ADDR localDataBase,
                                                                                 GM_ADDR localStateBase)
{
    PipeBarrier<PIPE_ALL>();
    uint64_t sendTokenOffset = (static_cast<uint64_t>(srcTokenIdx) * topK_ + srcTopKIdx) * perSlotBytes_;
    PipeBarrier<PIPE_ALL>();
    uint64_t recvStateOffset = (srcTokenIdx * topK_ + srcTopKIdx) * WIN_ADDR_ALIGN;

    PipeBarrier<PIPE_ALL>();
    SendLocalToken(tokenIndex, localDataBase + sendTokenOffset);
    // Keep the diagnostic path deliberately conservative: without top-k weights there used to be no MTE3 drain
    // between local token copies and the completion flag. The weight path happened to provide one below.
    PipeBarrier<PIPE_ALL>();
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    PipeBarrier<PIPE_ALL>();
    if constexpr (HasTopkWeight == 1) {
        statusTensor_(0) = weight;
        PipeBarrier<PIPE_ALL>();
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        PipeBarrier<PIPE_ALL>();
        GlobalTensor<uint32_t> state;
        PipeBarrier<PIPE_ALL>();
        state.SetGlobalBuffer((__gm__ uint32_t *)(localStateBase + recvStateOffset));
        PipeBarrier<PIPE_ALL>();
        DataCopyParams weightCopyParams = {1U, static_cast<uint16_t>(sizeof(uint32_t)), 0U, 0U};
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(state, statusTensor_, weightCopyParams);
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendRemoteSlot(
    uint32_t tokenIndex, int32_t srcTokenIdx, int32_t srcTopKIdx, uint32_t weight, GM_ADDR remoteDataBase,
    GM_ADDR remoteStateBase, uint64_t tokenBytes)
{
    PipeBarrier<PIPE_ALL>();
    uint64_t sendTokenOffset = (static_cast<uint64_t>(srcTokenIdx) * topK_ + srcTopKIdx) * perSlotBytes_;
    PipeBarrier<PIPE_ALL>();
    uint64_t recvStateOffset = (srcTokenIdx * topK_ + srcTopKIdx) * WIN_ADDR_ALIGN;
    PipeBarrier<PIPE_ALL>();
    GM_ADDR tokenAddr = (GM_ADDR)xGm_.GetPhyAddr(tokenIndex * axisH_);
    PipeBarrier<PIPE_ALL>();
    PrepareWrite<DEFAULT_WQE_CONFIG>(remoteDataBase + sendTokenOffset, tokenAddr, tokenBytes);
    PipeBarrier<PIPE_ALL>();
    if constexpr (HasTopkWeight == 1) {
        GM_ADDR weightAddr = (GM_ADDR)topkWeightsGm_.GetPhyAddr(tokenIndex);
        PipeBarrier<PIPE_ALL>();
        PrepareWrite<DEFAULT_WQE_CONFIG>(remoteStateBase + recvStateOffset, weightAddr, sizeof(uint32_t));
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendChannelFlag(uint32_t dstRank)
{
    PipeBarrier<PIPE_ALL>();
    uint64_t flagOffset = (static_cast<uint64_t>(numMaxTokensPerRank_) * topK_ + rankId_) * WIN_ADDR_ALIGN;
    PipeBarrier<PIPE_ALL>();
    GM_ADDR flagAddr = GetUrmaStateAddrByRankId(dstRank, combineStateWinOffset_) + flagOffset;
    PipeBarrier<PIPE_ALL>();
    if (dstRank == rankId_) {
        PipeBarrier<PIPE_ALL>();
        GlobalTensor<uint64_t> localFlag;
        PipeBarrier<PIPE_ALL>();
        localFlag.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(flagAddr));
        PipeBarrier<PIPE_ALL>();
        localFlag.SetValue(0, 1U);
        PipeBarrier<PIPE_ALL>();
        DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(localFlag);
        PipeBarrier<PIPE_ALL>();
        return;
    }
    PipeBarrier<PIPE_ALL>();
    BeginPreparedWrites(dstRank);
    PipeBarrier<PIPE_ALL>();
    PrepareWrite<CHANNEL_FLAG_WQE_CONFIG>(flagAddr, flagWorkspaceAddr_, WIN_ADDR_ALIGN);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::GetCoreAssignment(uint32_t totalBlocks,
                                                                                     uint32_t &targetRank,
                                                                                     uint32_t &coreIndexInGroup,
                                                                                     uint32_t &groupSize)
{
    PipeBarrier<PIPE_ALL>();
    uint32_t maxChannelAivNum = epWorldSize_ * COMBINE_CHANNEL_COUNT;
    PipeBarrier<PIPE_ALL>();
    bool assignRemainingToLocal = totalBlocks > maxChannelAivNum;
    PipeBarrier<PIPE_ALL>();
    if (assignRemainingToLocal) {
        // Put all remote communication AIVs first, then give every remaining AIV to local copies.
        PipeBarrier<PIPE_ALL>();
        uint32_t remoteAivNum = (epWorldSize_ - 1U) * COMBINE_CHANNEL_COUNT;
        PipeBarrier<PIPE_ALL>();
        if (aivId_ < remoteAivNum) {
            PipeBarrier<PIPE_ALL>();
            uint32_t remoteRankIndex = aivId_ / COMBINE_CHANNEL_COUNT;
            PipeBarrier<PIPE_ALL>();
            targetRank = remoteRankIndex < rankId_ ? remoteRankIndex : remoteRankIndex + 1U;
            PipeBarrier<PIPE_ALL>();
            coreIndexInGroup = aivId_ % COMBINE_CHANNEL_COUNT;
            PipeBarrier<PIPE_ALL>();
            groupSize = COMBINE_CHANNEL_COUNT;
        } else {
            PipeBarrier<PIPE_ALL>();
            targetRank = rankId_;
            PipeBarrier<PIPE_ALL>();
            coreIndexInGroup = aivId_ - remoteAivNum;
            PipeBarrier<PIPE_ALL>();
            groupSize = totalBlocks - remoteAivNum;
        }
        PipeBarrier<PIPE_ALL>();
        return;
    }

    PipeBarrier<PIPE_ALL>();
    uint32_t baseGroupSize = totalBlocks / epWorldSize_;
    PipeBarrier<PIPE_ALL>();
    uint32_t remainder = totalBlocks % epWorldSize_;
    PipeBarrier<PIPE_ALL>();
    uint32_t accumulated = 0;
    PipeBarrier<PIPE_ALL>();
    for (uint32_t rank = 0; rank < epWorldSize_; ++rank) {
        PipeBarrier<PIPE_ALL>();
        uint32_t currentGroupSize = baseGroupSize + ((rank < remainder) ? 1U : 0U);
        PipeBarrier<PIPE_ALL>();
        if (aivId_ < accumulated + currentGroupSize) {
            PipeBarrier<PIPE_ALL>();
            targetRank = rank;
            PipeBarrier<PIPE_ALL>();
            groupSize = currentGroupSize;
            PipeBarrier<PIPE_ALL>();
            coreIndexInGroup = aivId_ - accumulated;
            PipeBarrier<PIPE_ALL>();
            return;
        }
        PipeBarrier<PIPE_ALL>();
        accumulated += currentGroupSize;
    }
    PipeBarrier<PIPE_ALL>();
    targetRank = epWorldSize_;
    PipeBarrier<PIPE_ALL>();
    groupSize = 0;
    PipeBarrier<PIPE_ALL>();
    coreIndexInGroup = 0;
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendPhaseExpertToToken()
{
    PipeBarrier<PIPE_ALL>();
    if (epWorldSize_ == 0 || aivNum_ == 0) {
        PipeBarrier<PIPE_ALL>();
        return;
    }
    PipeBarrier<PIPE_ALL>();
    uint32_t activeAivNum = aivNum_;

    // Build once with all AIVs; communication AIVs consume only their compact rank table below.
    PipeBarrier<PIPE_ALL>();
    BuildAddressTable();

    PipeBarrier<PIPE_ALL>();
    activeBatchHandle_ = {};
    PipeBarrier<PIPE_ALL>();
    activeBatchChannel_ = 0;
    PipeBarrier<PIPE_ALL>();
    preparedWriteCount_ = 0;
    PipeBarrier<PIPE_ALL>();
    activeBatchInitialized_ = false;

    PipeBarrier<PIPE_ALL>();
    bool splitRankTokens = activeAivNum >= epWorldSize_;
    PipeBarrier<PIPE_ALL>();
    uint32_t targetRank = epWorldSize_;
    PipeBarrier<PIPE_ALL>();
    uint32_t coreIndexInGroup = 0;
    PipeBarrier<PIPE_ALL>();
    uint32_t groupSize = 1;
    PipeBarrier<PIPE_ALL>();
    if (splitRankTokens && aivId_ < activeAivNum) {
        PipeBarrier<PIPE_ALL>();
        GetCoreAssignment(activeAivNum, targetRank, coreIndexInGroup, groupSize);
    }
    PipeBarrier<PIPE_ALL>();
    bool sendsTokens = actualA_ != 0 && aivId_ < activeAivNum;
    PipeBarrier<PIPE_ALL>();
    if (sendsTokens) {
        PipeBarrier<PIPE_ALL>();
        if (splitRankTokens) {
            PipeBarrier<PIPE_ALL>();
            uint32_t entryCount = GetAddressTableCount(targetRank);
            PipeBarrier<PIPE_ALL>();
            uint32_t entriesPerCore = entryCount / groupSize;
            PipeBarrier<PIPE_ALL>();
            uint32_t remainder = entryCount % groupSize;
            PipeBarrier<PIPE_ALL>();
            uint32_t entryStart =
                coreIndexInGroup * entriesPerCore + ((coreIndexInGroup < remainder) ? coreIndexInGroup : remainder);
            PipeBarrier<PIPE_ALL>();
            uint32_t entryEnd = entryStart + entriesPerCore + ((coreIndexInGroup < remainder) ? 1U : 0U);
            PipeBarrier<PIPE_ALL>();
            SendAddressTableRange(targetRank, entryStart, entryEnd);
        } else {
            PipeBarrier<PIPE_ALL>();
            for (uint32_t rank = aivId_; rank < epWorldSize_; rank += activeAivNum) {
                PipeBarrier<PIPE_ALL>();
                uint32_t entryCount = GetAddressTableCount(rank);
                PipeBarrier<PIPE_ALL>();
                SendAddressTableRange(rank, 0U, entryCount);
            }
        }
    }

    // Commit every token batch before publishing completion. Local copies use MTE3, so drain that pipeline first;
    // then wait for all AIVs because only one AIV publishes the completion flag for each rank.
    PipeBarrier<PIPE_ALL>();
    FlushPreparedWrites();
    PipeBarrier<PIPE_ALL>();
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();

    // Each remote rank publishes one completion flag through channel 0 after all token writes have been issued.
    PipeBarrier<PIPE_ALL>();
    constexpr uint32_t completionChannelCount = COMBINE_CHANNEL_COUNT;
    PipeBarrier<PIPE_ALL>();
    bool publishesChannelFlag =
        aivId_ < activeAivNum && (!splitRankTokens || coreIndexInGroup < completionChannelCount);
    PipeBarrier<PIPE_ALL>();
    if (publishesChannelFlag) {
        PipeBarrier<PIPE_ALL>();
        LocalTensor<uint64_t> flagTensor = statusTensor_.ReinterpretCast<uint64_t>();
        PipeBarrier<PIPE_ALL>();
        SyncFunc<AscendC::HardEvent::S_V>();
        PipeBarrier<PIPE_ALL>();
        Duplicate<uint64_t>(flagTensor, 1U, UB_ALIGN / sizeof(uint64_t));
        PipeBarrier<PIPE_ALL>();
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        PipeBarrier<PIPE_ALL>();
        GlobalTensor<uint64_t> flagWorkspace;
        PipeBarrier<PIPE_ALL>();
        flagWorkspace.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(flagWorkspaceAddr_));
        PipeBarrier<PIPE_ALL>();
        DataCopy(flagWorkspace, flagTensor, UB_ALIGN / sizeof(uint64_t));
        PipeBarrier<PIPE_ALL>();
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        PipeBarrier<PIPE_ALL>();
        if (splitRankTokens) {
            PipeBarrier<PIPE_ALL>();
            SendChannelFlag(targetRank);
        } else {
            PipeBarrier<PIPE_ALL>();
            for (uint32_t dstRank = aivId_; dstRank < epWorldSize_; dstRank += activeAivNum) {
                PipeBarrier<PIPE_ALL>();
                SendChannelFlag(dstRank);
            }
        }
    }
    PipeBarrier<PIPE_ALL>();
    FlushPreparedWrites();
    PipeBarrier<PIPE_ALL>();
    if (sendsTokens) {
        PipeBarrier<PIPE_ALL>();
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(recvSrcMetadataGm_);
    }
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::Process()
{
    PipeBarrier<PIPE_ALL>();
    SendPhaseExpertToToken();
}

#endif

} // namespace MoeEpCombineImpl

#endif // MOE_EP_COMBINE_H
