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
constexpr uint64_t ALIGNED_LEN_256 = 256UL;
constexpr uint32_t SEND_DOUBLE_BUFFER_NUM = 2U;
static constexpr uint32_t ADDRESS_ENTRY_UB_BYTES = UB_ALIGN;
static constexpr uint32_t ADDRESS_ENTRY_GM_BYTES = WIN_ADDR_ALIGN;
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
                                GM_ADDR numRecvPerExpert, GM_ADDR topkWeights, GM_ADDR workspace, GM_ADDR tilingGM,
                                TPipe *pipe, const MoeEpCombineInfo *tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void SendLocalToken(uint32_t tokenIndex, GM_ADDR dstAddr);
    __aicore__ inline void SendSlot(uint32_t tokenIndex, int32_t srcRank, int32_t srcTokenIdx,
                                    int32_t srcTopKIdx, uint32_t channelIndex, uint32_t weight);
    __aicore__ inline void SendChannelFlag(uint32_t dstRank, uint32_t channelIndex,
                                          uint32_t completionChannelCount);
    template <auto const& config>
    __aicore__ inline void PrepareWrite(
        uint64_t commHandle, GM_ADDR dstBase, GM_ADDR dst, GM_ADDR src, uint64_t len);
    __aicore__ inline void FlushPreparedWrites();
    __aicore__ inline void BuildAddressTable();
    __aicore__ inline void SendAddressTableRange(uint32_t targetRank, uint32_t entryStart,
                                                 uint32_t entryEnd, uint32_t channelIndex);
    __aicore__ inline void SendPhaseExpertToToken();
    __aicore__ inline void GetCoreAssignment(uint32_t totalBlocks, uint32_t &targetRank,
                                             uint32_t &coreIndexInGroup, uint32_t &groupSize);

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

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_; // 数据队列
    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;                 // 通信上下文
    using HcommBatchHandle = AscendC::BatchHandle<AscendC::ChannelHandle>;

    GM_ADDR winRankAddr_[Mc2Aclnn::HCCL_MAX_RANK_SIZE];
    uint64_t hcommHandle_[Mc2Aclnn::HCCL_MAX_RANK_SIZE];
    GM_ADDR combineSendDataWorkspaceAddr_{nullptr};
    GM_ADDR flagWorkspaceAddr_{nullptr};
    GM_ADDR rankCountWorkspaceAddr_{nullptr};
    uint32_t aivId_{0};
    HcommBatchHandle activeBatchHandle_{};
    uint64_t activeBatchChannel_{0};
    uint32_t preparedWriteCount_{0};
};

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void
MoeEpCombine<TemplateMoeEpCombineTypeFunc>::Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR recvSrcMetadata,
                                                 GM_ADDR numRecvPerExpert, GM_ADDR topkWeights, GM_ADDR workspace,
                                                 GM_ADDR tilingGM, TPipe *pipe, const MoeEpCombineInfo *tilingData)
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
    sendDataWorkspaceSizePerRank_ = tilingData->sendDataWorkspaceSizePerRank;
    flagWorkspaceAddr_ = combineSendDataWorkspaceAddr_ + epWorldSize_ * sendDataWorkspaceSizePerRank_ +
                         static_cast<uint64_t>(aivId_) * WIN_ADDR_ALIGN;
    rankCountWorkspaceAddr_ = combineSendDataWorkspaceAddr_ + epWorldSize_ * sendDataWorkspaceSizePerRank_ +
                              static_cast<uint64_t>(aivNum_) * WIN_ADDR_ALIGN;
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
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::BuildAddressTable()
{
    // Keep one 512-byte counter slot per rank so AtomicAdd on different ranks never shares a cache line.
    if (aivId_ == 0) {
        for (uint32_t rank = 0; rank < epWorldSize_; ++rank) {
            __gm__ int32_t *countAddr = reinterpret_cast<__gm__ int32_t *>(
                rankCountWorkspaceAddr_ + static_cast<uint64_t>(rank) * WIN_ADDR_ALIGN);
            AscendC::WriteGmByPassDCache(countAddr, int32_t(0));
        }
        PipeBarrier<PIPE_ALL>();
    }
    SyncAll<true>();

    if (actualA_ == 0) {
        return;
    }

    constexpr uint32_t metaBytesPerToken = RECV_META_FIELDS * sizeof(int32_t);
    constexpr uint32_t metaChunkTokenMax = 8192U;
    uint32_t metaChunkTokens = (actualA_ < metaChunkTokenMax) ? actualA_ : metaChunkTokenMax;
    uint32_t metaChunkBytes = Ceil(metaChunkTokens * metaBytesPerToken, UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(metadataBuf_, metaChunkBytes);
    if constexpr (HasTopkWeight == 1) {
        uint32_t weightChunkBytes = Ceil(metaChunkTokens * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(weightsBuf_, weightChunkBytes);
    }

    uint64_t tokensPerCore = actualA_ / aivNum_;
    uint64_t remainder = actualA_ % aivNum_;
    uint64_t scanStart = static_cast<uint64_t>(aivId_) * tokensPerCore +
                         ((aivId_ < remainder) ? aivId_ : remainder);
    uint64_t scanEnd = scanStart + tokensPerCore + ((aivId_ < remainder) ? 1U : 0U);

    LocalTensor<int32_t> metadataLocal = metadataBuf_.Get<int32_t>();
    const DataCopyPadExtParams<int32_t> metaPadParams{false, 0U, 0U, 0U};
    for (uint64_t chunkStart = scanStart; chunkStart < scanEnd; chunkStart += metaChunkTokens) {
        uint64_t chunkEnd = (chunkStart + metaChunkTokens > scanEnd) ? scanEnd : chunkStart + metaChunkTokens;
        uint32_t curChunkTokens = static_cast<uint32_t>(chunkEnd - chunkStart);

        LocalTensor<uint32_t> weightsLocal;
        if constexpr (HasTopkWeight == 1) {
            weightsLocal = weightsBuf_.Get<uint32_t>();
            DataCopyExtParams weightCopyParams{
                1U, static_cast<uint32_t>(curChunkTokens * sizeof(uint32_t)), 0U, 0U, 0U};
            const DataCopyPadExtParams<uint32_t> weightPadParams{false, 0U, 0U, 0U};
            DataCopyPad(weightsLocal, topkWeightsGm_[chunkStart].template ReinterpretCast<uint32_t>(),
                        weightCopyParams, weightPadParams);
        }

        DataCopyExtParams metaCopyParams{1U, curChunkTokens * metaBytesPerToken, 0U, 0U, 0U};
        DataCopyPad(metadataLocal, recvSrcMetadataGm_[chunkStart * RECV_META_FIELDS],
                    metaCopyParams, metaPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        for (uint32_t i = 0; i < curChunkTokens; ++i) {
            int32_t srcRank = metadataLocal.GetValue(i * RECV_META_FIELDS);
            if (srcRank < 0 || srcRank >= static_cast<int32_t>(epWorldSize_)) {
                continue;
            }

            __gm__ int32_t *countAddr = reinterpret_cast<__gm__ int32_t *>(
                rankCountWorkspaceAddr_ + static_cast<uint64_t>(srcRank) * WIN_ADDR_ALIGN);
            int32_t entryIndex = AscendC::AtomicAdd(countAddr, int32_t(1));
            ascendc_assert(entryIndex >= 0 &&
                               static_cast<uint64_t>(entryIndex) <
                                   static_cast<uint64_t>(numMaxTokensPerRank_) * topK_,
                           "combine address table overflow, rank=%d, index=%d", srcRank, entryIndex);

            __gm__ uint32_t *entryAddr = reinterpret_cast<__gm__ uint32_t *>(
                GetLocalSendDataWorkspaceAddr(srcRank) +
                static_cast<uint64_t>(entryIndex) * ADDRESS_ENTRY_GM_BYTES);
            AscendC::WriteGmByPassDCache(entryAddr, static_cast<uint32_t>(chunkStart) + i);
            AscendC::WriteGmByPassDCache(entryAddr + 1,
                static_cast<uint32_t>(metadataLocal.GetValue(i * RECV_META_FIELDS + 1)));
            AscendC::WriteGmByPassDCache(entryAddr + 2,
                static_cast<uint32_t>(metadataLocal.GetValue(i * RECV_META_FIELDS + 2)));
            uint32_t weight = 0;
            if constexpr (HasTopkWeight == 1) {
                weight = weightsLocal.GetValue(i);
            }
            AscendC::WriteGmByPassDCache(entryAddr + 3, weight);
        }
        SyncFunc<AscendC::HardEvent::S_MTE2>();
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::SendAddressTableRange(
    uint32_t targetRank, uint32_t entryStart, uint32_t entryEnd, uint32_t channelIndex)
{
    if (entryStart >= entryEnd) {
        return;
    }

    uint32_t metaChunkTokens = (actualA_ < 8192U) ? actualA_ : 8192U;
    uint32_t metaChunkBytes = Ceil(metaChunkTokens * RECV_META_FIELDS * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t entryChunkMax = metaChunkBytes / ADDRESS_ENTRY_UB_BYTES;
    LocalTensor<uint32_t> entriesLocal = metadataBuf_.Get<uint32_t>();
    GlobalTensor<uint32_t> addressTable;
    addressTable.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(GetLocalSendDataWorkspaceAddr(targetRank)));
    const DataCopyPadExtParams<uint32_t> padParams{false, 0U, 0U, 0U};
    constexpr uint32_t localEntryStride = ADDRESS_ENTRY_UB_BYTES / sizeof(uint32_t);
    constexpr uint32_t globalEntryStride = ADDRESS_ENTRY_GM_BYTES / sizeof(uint32_t);

    for (uint32_t chunkStart = entryStart; chunkStart < entryEnd; chunkStart += entryChunkMax) {
        uint32_t chunkEnd = (chunkStart + entryChunkMax > entryEnd) ? entryEnd : chunkStart + entryChunkMax;
        uint32_t curEntries = chunkEnd - chunkStart;
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(curEntries), ADDRESS_ENTRY_UB_BYTES,
            ADDRESS_ENTRY_GM_BYTES - ADDRESS_ENTRY_UB_BYTES, 0U, 0U};
        DataCopyPad(entriesLocal, addressTable[chunkStart * globalEntryStride], copyParams, padParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        for (uint32_t i = 0; i < curEntries; ++i) {
            uint32_t offset = i * localEntryStride;
            uint32_t tokenIndex = entriesLocal.GetValue(offset);
            int32_t srcTokenIdx = static_cast<int32_t>(entriesLocal.GetValue(offset + 1));
            int32_t srcTopKIdx = static_cast<int32_t>(entriesLocal.GetValue(offset + 2));
            uint32_t weight = entriesLocal.GetValue(offset + 3);
            SendSlot(tokenIndex, static_cast<int32_t>(targetRank), srcTokenIdx, srcTopKIdx, channelIndex, weight);
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
        localFlag.SetValue(0, 1U);
        DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(localFlag);
        return;
    }
    PrepareWrite<CHANNEL_FLAG_WQE_CONFIG>(
        GetCommHandle(dstRank, channelIndex), winRankAddr_[dstRank], flagAddr, flagWorkspaceAddr_, WIN_ADDR_ALIGN);
}

template <TemplateMoeEpCombineTypeClass>
__aicore__ inline void MoeEpCombine<TemplateMoeEpCombineTypeFunc>::GetCoreAssignment(
    uint32_t totalBlocks, uint32_t &targetRank, uint32_t &coreIndexInGroup, uint32_t &groupSize)
{
    uint32_t maxChannelAivNum = epWorldSize_ * channelsPerRank_;
    bool assignRemainingToLocal = totalBlocks > maxChannelAivNum;
    if (assignRemainingToLocal) {
        // Put all remote communication AIVs first, then give every remaining AIV to local copies.
        uint32_t remoteAivNum = (epWorldSize_ - 1U) * channelsPerRank_;
        if (aivId_ < remoteAivNum) {
            uint32_t remoteRankIndex = aivId_ / channelsPerRank_;
            targetRank = remoteRankIndex < rankId_ ? remoteRankIndex : remoteRankIndex + 1U;
            coreIndexInGroup = aivId_ % channelsPerRank_;
            groupSize = channelsPerRank_;
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
    uint32_t maxChannelAivNum = epWorldSize_ * channelsPerRank_;
    bool useRemainingAivForLocal = activeAivNum > maxChannelAivNum;

    // Build once with all AIVs; communication AIVs consume only their compact rank table below.
    BuildAddressTable();

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
    if (sendsTokens) {
        if (splitRankTokens) {
            __gm__ int32_t *countAddr = reinterpret_cast<__gm__ int32_t *>(
                rankCountWorkspaceAddr_ + static_cast<uint64_t>(targetRank) * WIN_ADDR_ALIGN);
            uint32_t entryCount = static_cast<uint32_t>(AscendC::ReadGmByPassDCache(countAddr));
            uint32_t entriesPerCore = entryCount / groupSize;
            uint32_t remainder = entryCount % groupSize;
            uint32_t entryStart = coreIndexInGroup * entriesPerCore +
                                  ((coreIndexInGroup < remainder) ? coreIndexInGroup : remainder);
            uint32_t entryEnd = entryStart + entriesPerCore + ((coreIndexInGroup < remainder) ? 1U : 0U);
            SendAddressTableRange(targetRank, entryStart, entryEnd, channelIndex);
        } else {
            for (uint32_t rank = aivId_; rank < epWorldSize_; rank += activeAivNum) {
                __gm__ int32_t *countAddr = reinterpret_cast<__gm__ int32_t *>(
                    rankCountWorkspaceAddr_ + static_cast<uint64_t>(rank) * WIN_ADDR_ALIGN);
                uint32_t entryCount = static_cast<uint32_t>(AscendC::ReadGmByPassDCache(countAddr));
                SendAddressTableRange(rank, 0U, entryCount, 0U);
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
