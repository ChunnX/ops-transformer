/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_ep_combine_epilogue.h
 * \brief MoE Expert-Parallel Combine Epilogue kernel — recv + reduce phase.
 *        Reads expert outputs from HCCL Window, accumulates per-token topK results,
 *        and writes combined_x / combined_topk_weights.
 */
#ifndef MOE_EP_COMBINE_EPILOGUE_H
#define MOE_EP_COMBINE_EPILOGUE_H

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if (ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)) && \
    (HCOMM_MAJOR > 9 || (HCOMM_MAJOR == 9 && HCOMM_MINOR > 0))
#define ENABLE_MOE_EP_COMBINE_EPILOGUE_KERNEL
#endif

#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "kernel_tiling/kernel_tiling.h"
#include "adv_api/reduce/reduce.h"
#include "adv_api/reduce/sum.h"
#include <cstddef>
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#include "../../common/op_kernel/mc2_moe_context.h"
#include "../../common/op_kernel/moe_ep_exception_dump_writer.h"

#include "moe_ep_combine_epilogue_tiling_key.h"
#include "moe_ep_combine_epilogue_tiling.h"

namespace MoeEpCombineEpilogueImpl {

#if defined(ENABLE_MOE_EP_COMBINE_EPILOGUE_KERNEL)

using namespace AscendC;

#define TemplateMoeEpCombineEpilogueTypeClass typename XType, uint32_t HasTopkWeight
#define TemplateMoeEpCombineEpilogueTypeFunc XType, HasTopkWeight

static constexpr uint32_t WIN_ADDR_ALIGN = 512;
constexpr uint64_t UB_ALIGN = 32UL;
constexpr uint32_t STATE_OFFSET = 32U;
constexpr uint64_t ALIGNED_LEN_256 = 256UL;
constexpr uint32_t DOUBLE_BUFFER_NUM = 2U;
static constexpr uint32_t RECV_META_FIELDS = 5U;
static constexpr uint32_t META_TOKEN_IDX_OFFSET = 1U;
static constexpr uint32_t META_TOPK_IDX_OFFSET = 2U;
static constexpr uint32_t META_RECV_X_IDX_OFFSET = 4U;
static constexpr uint32_t META_CHUNK_TOKEN_MAX = 128U;
static constexpr uint32_t LOCAL_INDEX_STAGE_STRIDE = UB_ALIGN / sizeof(int32_t);
static constexpr uint32_t LOCAL_INDEX_STAGE_COUNT = 128U;

template <TemplateMoeEpCombineEpilogueTypeClass>
class MoeEpCombineEpilogue {
public:
    __aicore__ inline MoeEpCombineEpilogue(){};

    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR recvSrcMetadata,
                                GM_ADDR recvRankOffsets, GM_ADDR topkWeights, GM_ADDR combinedX,
                                GM_ADDR combinedTopkWeights, GM_ADDR workspace, GM_ADDR tilingGM, TPipe *pipe,
                                const MoeEpCombineEpilogueInfo *tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
                                       uint32_t &endTokenId, uint32_t &tokenPerAivNum);
    __aicore__ inline void BuffInit();
    __aicore__ inline uint32_t GetCompletionChannelCount();
    __aicore__ inline void MaskCheck();
    __aicore__ inline bool WaitDispatch(uint32_t completionChannelCount);
    __aicore__ inline void ClearCompletionFlags();
    __aicore__ inline void BuildLocalIndexMap();
    __aicore__ inline void ProcessTopKToken(uint32_t tokenIndex);
    __aicore__ inline void RecvPhaseReduce();
    __aicore__ inline GM_ADDR GetUrmaWinAddrByRankId(uint32_t rankId, uint64_t offset)
    {
        return (GM_ADDR)(winRankAddr_[rankId] + offset);
    }

    __aicore__ inline GM_ADDR GetUrmaStateAddrByRankId(uint32_t rankId, uint64_t offset)
    {
        return (GM_ADDR)(winRankAddr_[rankId] + offset);
    }

    TPipe *tpipe_{nullptr};
    const MoeEpCombineEpilogueInfo *tilingData_{nullptr};
    __gm__ Mc2Aclnn::MoeCommContext *mc2Context_{nullptr};
    MoeEpExceptionDump::MoeEpCoreDiagWriter diagWriter_;

    uint32_t rankId_{0};
    uint32_t epWorldSize_{0};
    uint32_t combineChannelCount_{1};
    uint32_t numMaxTokensPerRank_{0};
    uint32_t numTokens_{0};
    uint32_t topK_{0};
    uint32_t axisH_{0};
    uint32_t hAlignSize_{0};
    uint64_t combineStateWinOffset_{0};
    uint64_t combineDataWinOffset_{0};

    uint32_t aivNum_{0};
    uint32_t aivId_{0};
    uint64_t recvCapacity_{0};
    uint64_t actualA_{0};
    uint32_t metadataChunkTokens_{1};

    uint32_t tStart_{0};
    uint32_t tEnd_{0};
    uint32_t tPerCore_{0};
    uint32_t maskTokenNum_{0};
    uint32_t bsKCastCnt_{0};
    uint32_t activeMaskAlignSize_{0};

    GlobalTensor<XType> xGm_;
    GlobalTensor<int32_t> recvSrcMetadataGm_;
    GlobalTensor<int32_t> recvRankOffsetsGm_;
    GlobalTensor<float> topkWeightsGm_;
    GlobalTensor<int32_t> localRecvXIndexGm_;
    GlobalTensor<XType> combinedXGm_;
    GlobalTensor<int32_t> topkIdxGm_;
    GlobalTensor<float> combinedTopkWeightsGm_;

    LocalTensor<float> ubAccFp32_;
    LocalTensor<float> ubTmpFp32_;
    LocalTensor<bool> maskGenerateTensor_;
    LocalTensor<bool> maskStrideTensor_;
    LocalTensor<half> tokenTargetTensor_;

    TQue<QuePosition::VECIN, 1> xInQue_;
    TQue<QuePosition::VECOUT, 1> xOutQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> weightQue_;
    TBuf<QuePosition::VECIN> ubAccFp32Buf_;
    TBuf<QuePosition::VECIN> ubTmpFp32Buf_;
    TBuf<> stateBuf_;
    TBuf<> waitSumBuf_;
    TBuf<> rankOffsetsBuf_;
    TBuf<> metadataBuf_;
    TBuf<> localIndexBuf_;
    LocalTensor<int32_t> rankOffsetsTensor_;
    LocalTensor<int32_t> localIndexTensor_;

    TBuf<> compareBuf_;
    TBuf<> rowTmpFloatBuf_;
    TBuf<> tokenBuf_;
    TBuf<> tokenTargetTBuf_;

    GM_ADDR winRankAddr_[Mc2Aclnn::HCCL_MAX_RANK_SIZE];
};

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline void MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR recvSrcMetadata, GM_ADDR recvRankOffsets, GM_ADDR topkWeights,
    GM_ADDR combinedX, GM_ADDR combinedTopkWeights, GM_ADDR workspace, GM_ADDR tilingGM, TPipe *pipe,
    const MoeEpCombineEpilogueInfo *tilingData)
{
    tpipe_ = pipe;
    tilingData_ = tilingData;
    aivId_ = GetBlockIdx();
    epWorldSize_ = tilingData_->cfg.epWorldSize;
    numMaxTokensPerRank_ = tilingData_->cfg.numMaxTokensPerRank;
    numTokens_ = tilingData_->cfg.numTokens;
    topK_ = tilingData_->cfg.topK;
    axisH_ = tilingData_->cfg.hidden;
    aivNum_ = tilingData->aivNum;
    recvCapacity_ = tilingData->recvCapacity;
    hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;

    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::MoeCommContext *>(context);
    rankId_ = mc2Context_->epRankId;
    combineChannelCount_ = mc2Context_->channelsPerRank;
    combineChannelCount_ = combineChannelCount_ == 0U ? 1U : combineChannelCount_;
    for (uint32_t i = 0; i < epWorldSize_; ++i) {
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
    }

    combineStateWinOffset_ = tilingData->combineStateWinOffset;
    combineDataWinOffset_ = tilingData->combineDataWinOffset;

    xGm_.SetGlobalBuffer((__gm__ XType *)x);
    recvSrcMetadataGm_.SetGlobalBuffer((__gm__ int32_t *)recvSrcMetadata);
    recvRankOffsetsGm_.SetGlobalBuffer((__gm__ int32_t *)recvRankOffsets);
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);
    localRecvXIndexGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(userWorkspace + tilingData_->localIndexWorkspaceOffset),
        static_cast<uint64_t>(numTokens_) * topK_);
    if constexpr (HasTopkWeight == 1) {
        topkWeightsGm_.SetGlobalBuffer((__gm__ float *)topkWeights);
    }
    combinedXGm_.SetGlobalBuffer((__gm__ XType *)combinedX);
    topkIdxGm_.SetGlobalBuffer((__gm__ int32_t *)topkIdx);

    if constexpr (HasTopkWeight == 1) {
        combinedTopkWeightsGm_.SetGlobalBuffer((__gm__ float *)combinedTopkWeights);
    }

    constexpr size_t metadataOffset = offsetof(MoeEpCombineEpilogueTilingData, moeEpCombineEpilogueInfo) +
                                      offsetof(MoeEpCombineEpilogueInfo, dumpMetadata);
    MoeEpExceptionDump::WriteMetadata(context, tilingGM + metadataOffset);
    diagWriter_.Init(context, MOE_EP_CORE_DIAG_COMBINE_EPILOGUE, tpipe_);
    diagWriter_.RunPosRecord(MOE_EP_COMBINE_EPILOGUE_RUN_POS_INIT_DONE);
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline void MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId, uint32_t &endTokenId, uint32_t &sendTokenNum)
{
    sendTokenNum = curSendCnt / curUseAivNum;
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum;
    uint32_t newAivId = aivId_;

    startTokenId = sendTokenNum * newAivId;
    if (newAivId < remainderTokenNum) {
        sendTokenNum += 1;
        startTokenId += newAivId;
    } else {
        startTokenId += remainderTokenNum;
    }
    endTokenId = startTokenId + sendTokenNum;
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline void MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::BuffInit()
{
    SplitToCore(numTokens_, aivNum_, tStart_, tEnd_, tPerCore_);
    if (tStart_ < numTokens_) {
        maskTokenNum_ = tPerCore_ * topK_;
        uint32_t bsKInt32Align = Ceil(maskTokenNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
        uint32_t bsKFloatAlign = Ceil(maskTokenNum_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
        uint32_t bsKHalfAlign = Ceil(maskTokenNum_ * sizeof(half), UB_ALIGN) * UB_ALIGN;
        uint32_t bsHalfAlign = Ceil(tPerCore_ * sizeof(half), UB_ALIGN) * UB_ALIGN;

        bsKCastCnt_ = Ceil(maskTokenNum_ * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
        activeMaskAlignSize_ = tPerCore_ * Ceil(topK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN * sizeof(half);
        bsKInt32Align = (bsKInt32Align > bsKCastCnt_ ? bsKInt32Align : bsKCastCnt_);
        bsKHalfAlign = (bsKHalfAlign > activeMaskAlignSize_ ? bsKHalfAlign : activeMaskAlignSize_);
        bsKFloatAlign = (bsKFloatAlign > bsKCastCnt_ ? bsKFloatAlign : bsKCastCnt_);
        bsKFloatAlign = (bsKFloatAlign > activeMaskAlignSize_ ? bsKFloatAlign : activeMaskAlignSize_);

        tpipe_->InitBuffer(tokenTargetTBuf_, bsHalfAlign);
        tpipe_->InitBuffer(compareBuf_, bsKInt32Align);
        tpipe_->InitBuffer(rowTmpFloatBuf_, bsKFloatAlign);
        tpipe_->InitBuffer(tokenBuf_, bsKHalfAlign);
    }
    if constexpr (HasTopkWeight == 1) {
        tpipe_->InitBuffer(weightQue_, DOUBLE_BUFFER_NUM, UB_ALIGN);
    }
    tpipe_->InitBuffer(xInQue_, DOUBLE_BUFFER_NUM, hAlignSize_);
    if (tStart_ < numTokens_) {
        tpipe_->InitBuffer(xOutQue_, DOUBLE_BUFFER_NUM, hAlignSize_);
        uint32_t ubFp32Bytes = Ceil(axisH_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(ubAccFp32Buf_, ubFp32Bytes);
        tpipe_->InitBuffer(ubTmpFp32Buf_, ubFp32Bytes);
        ubAccFp32_ = ubAccFp32Buf_.Get<float>();
        ubTmpFp32_ = ubTmpFp32Buf_.Get<float>();
    }

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

    metadataChunkTokens_ = actualA_ < META_CHUNK_TOKEN_MAX ? static_cast<uint32_t>(actualA_) : META_CHUNK_TOKEN_MAX;
    metadataChunkTokens_ = metadataChunkTokens_ == 0U ? 1U : metadataChunkTokens_;
    uint32_t metadataChunkBytes =
        Ceil(static_cast<uint64_t>(metadataChunkTokens_) * RECV_META_FIELDS * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(metadataBuf_, metadataChunkBytes);
    tpipe_->InitBuffer(localIndexBuf_, LOCAL_INDEX_STAGE_COUNT * UB_ALIGN);
    localIndexTensor_ = localIndexBuf_.Get<int32_t>();
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline uint32_t MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::GetCompletionChannelCount()
{
    if (aivNum_ < epWorldSize_) {
        return 1U;
    }
    uint32_t maxChannelAivNum = epWorldSize_ * combineChannelCount_;
    if (aivNum_ > maxChannelAivNum) {
        return combineChannelCount_;
    }
    uint32_t baseGroupSize = aivNum_ / epWorldSize_;
    uint32_t remainder = aivNum_ % epWorldSize_;
    return baseGroupSize + ((rankId_ < remainder) ? 1U : 0U);
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline bool MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::WaitDispatch(
    uint32_t completionChannelCount)
{
    uint64_t flagOffset = static_cast<uint64_t>(numMaxTokensPerRank_) * topK_ * WIN_ADDR_ALIGN;
    GM_ADDR stateGM = GetUrmaStateAddrByRankId(rankId_, combineStateWinOffset_) + flagOffset;
    LocalTensor<uint32_t> stateTensor = stateBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> waitSum = waitSumBuf_.Get<uint32_t>();

    uint32_t totalFlagCount = epWorldSize_ * combineChannelCount_;
    GlobalTensor<uint32_t> stateGMTensor;
    stateGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(stateGM));
    DataCopyParams params = {static_cast<uint16_t>(totalFlagCount), 1U,
                             static_cast<uint16_t>((WIN_ADDR_ALIGN - UB_ALIGN) / UB_ALIGN), 0U};
    DataCopy(stateTensor, stateGMTensor, params);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    uint32_t shape[] = {totalFlagCount, UB_ALIGN / sizeof(uint32_t)};
    ReduceSum<uint32_t, AscendC::Pattern::Reduce::RA, false>(waitSum, stateTensor, shape, true);
    SyncFunc<AscendC::HardEvent::V_S>();
    uint32_t expectedFlagSum = epWorldSize_ * completionChannelCount;
    return waitSum.GetValue(0) == expectedFlagSum;
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline void MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::MaskCheck()
{
    if (tStart_ >= numTokens_) {
        return;
    }
    LocalTensor<half> maskCalcTensor = tokenBuf_.Get<half>();
    LocalTensor<int32_t> topkIdsTensor = compareBuf_.Get<int32_t>();
    LocalTensor<float> topkIdsFloatTensor = rowTmpFloatBuf_.Get<float>();
    LocalTensor<uint8_t> maskTensor = compareBuf_.Get<uint8_t>();
    LocalTensor<half> maskCalcSelectedTensor = rowTmpFloatBuf_.Get<half>();
    maskGenerateTensor_ = compareBuf_.Get<bool>();
    LocalTensor<half> tempTensor = rowTmpFloatBuf_.Get<half>();
    maskStrideTensor_ = tokenBuf_.Get<bool>();
    tokenTargetTensor_ = tokenTargetTBuf_.Get<half>();

    // 构造二维全true的mask
    Duplicate<half>(maskCalcTensor, static_cast<half>(1),
                    Ceil(maskTokenNum_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half));
    // 拷入topkIds
    DataCopyExtParams topkIdsCntParams = {1U, static_cast<uint32_t>(maskTokenNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> topkIdsCntCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(topkIdsTensor, topkIdxGm_[tStart_ * topK_], topkIdsCntParams, topkIdsCntCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    // 根据topkIds大于0，得到考虑(-1专家) 后的mask
    uint32_t calcCnt = bsKCastCnt_ / sizeof(int32_t);
    Cast(topkIdsFloatTensor, topkIdsTensor, RoundMode::CAST_NONE, calcCnt);
    CompareScalar(maskTensor, topkIdsFloatTensor, static_cast<float>(0), AscendC::CMPMODE::GE, calcCnt);
    Select(maskCalcSelectedTensor, maskTensor, maskCalcTensor, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
           calcCnt);
    Cast(maskGenerateTensor_.ReinterpretCast<uint8_t>(), maskCalcSelectedTensor, RoundMode::CAST_NONE, calcCnt);
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline void MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::ClearCompletionFlags()
{
    uint64_t flagOffset = static_cast<uint64_t>(numMaxTokensPerRank_) * topK_ * WIN_ADDR_ALIGN;
    GM_ADDR stateGM = GetUrmaStateAddrByRankId(rankId_, combineStateWinOffset_) + flagOffset;
    LocalTensor<uint32_t> stateTensor = stateBuf_.Get<uint32_t>();
    uint32_t totalFlagCount = epWorldSize_ * combineChannelCount_;
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<uint32_t>(stateTensor, 0U, totalFlagCount * STATE_OFFSET / sizeof(uint32_t));
    SyncFunc<AscendC::HardEvent::V_MTE3>();

    GlobalTensor<uint32_t> stateGMTensor;
    stateGMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(stateGM));
    DataCopyParams clearParams = {static_cast<uint16_t>(totalFlagCount), 1U, 0U,
                                  static_cast<uint16_t>((WIN_ADDR_ALIGN - UB_ALIGN) / UB_ALIGN)};
    DataCopy(stateGMTensor, stateTensor, clearParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline void MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::BuildLocalIndexMap()
{
    uint32_t slotStart = 0U;
    uint32_t slotEnd = 0U;
    uint32_t slotCount = 0U;
    SplitToCore(numTokens_ * topK_, aivNum_, slotStart, slotEnd, slotCount);
    Duplicate<int32_t>(localIndexTensor_, 0, LOCAL_INDEX_STAGE_COUNT * LOCAL_INDEX_STAGE_STRIDE);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    for (uint32_t clearStart = slotStart; clearStart < slotEnd; clearStart += LOCAL_INDEX_STAGE_COUNT) {
        uint32_t clearCount = slotEnd - clearStart < LOCAL_INDEX_STAGE_COUNT ?
                                  slotEnd - clearStart :
                                  LOCAL_INDEX_STAGE_COUNT;
        DataCopyExtParams clearParams{1U, clearCount * sizeof(int32_t), 0U, 0U, 0U};
        DataCopyPad(localRecvXIndexGm_[clearStart], localIndexTensor_, clearParams);
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    // Clearing is partitioned by destination slot, while metadata is partitioned by source row.
    SyncAll<true>();

    uint64_t localBegin = static_cast<uint32_t>(rankOffsetsTensor_.GetValue(rankId_));
    uint64_t localEnd = static_cast<uint32_t>(rankOffsetsTensor_.GetValue(rankId_ + 1U));
    localBegin = localBegin < actualA_ ? localBegin : actualA_;
    localEnd = localEnd < actualA_ ? localEnd : actualA_;
    if (localBegin >= localEnd) {
        return;
    }

    uint32_t localCoreStart = 0U;
    uint32_t localCoreEnd = 0U;
    uint32_t localCoreCount = 0U;
    SplitToCore(static_cast<uint32_t>(localEnd - localBegin), aivNum_, localCoreStart, localCoreEnd, localCoreCount);
    uint64_t rangeBegin = localBegin + localCoreStart;
    uint64_t rangeEnd = localBegin + localCoreEnd;
    if (rangeBegin >= rangeEnd) {
        return;
    }

    constexpr uint32_t metaBytesPerToken = RECV_META_FIELDS * sizeof(int32_t);
    LocalTensor<int32_t> metadataLocal = metadataBuf_.Get<int32_t>();
    DataCopyPadExtParams<int32_t> metadataPadParams{false, 0U, 0U, 0U};
    for (uint64_t chunkStart = rangeBegin; chunkStart < rangeEnd; chunkStart += metadataChunkTokens_) {
        uint64_t chunkEnd = chunkStart + metadataChunkTokens_ < rangeEnd ? chunkStart + metadataChunkTokens_ : rangeEnd;
        uint32_t chunkCount = static_cast<uint32_t>(chunkEnd - chunkStart);
        DataCopyExtParams copyParams{1U, chunkCount * metaBytesPerToken, 0U, 0U, 0U};
        DataCopyPad(metadataLocal, recvSrcMetadataGm_[chunkStart * RECV_META_FIELDS], copyParams, metadataPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        for (uint32_t i = 0U; i < chunkCount; ++i) {
            uint32_t metaOffset = i * RECV_META_FIELDS;
            int32_t srcTokenIdx = metadataLocal.GetValue(metaOffset + META_TOKEN_IDX_OFFSET);
            int32_t srcTopKIdx = metadataLocal.GetValue(metaOffset + META_TOPK_IDX_OFFSET);
            int32_t recvXIdx = metadataLocal.GetValue(metaOffset + META_RECV_X_IDX_OFFSET);
            localIndexTensor_.SetValue(i * LOCAL_INDEX_STAGE_STRIDE, recvXIdx + 1);
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyExtParams indexCopyParams{1U, static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
        for (uint32_t i = 0U; i < chunkCount; ++i) {
            uint32_t metaOffset = i * RECV_META_FIELDS;
            int32_t srcTokenIdx = metadataLocal.GetValue(metaOffset + META_TOKEN_IDX_OFFSET);
            int32_t srcTopKIdx = metadataLocal.GetValue(metaOffset + META_TOPK_IDX_OFFSET);
            uint64_t dstSlot =
                static_cast<uint64_t>(static_cast<uint32_t>(srcTokenIdx)) * topK_ + static_cast<uint32_t>(srcTopKIdx);
            DataCopyPad(localRecvXIndexGm_[dstSlot], localIndexTensor_[i * LOCAL_INDEX_STAGE_STRIDE], indexCopyParams);
        }
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        SyncFunc<AscendC::HardEvent::S_MTE2>();
    }
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline void MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::ProcessTopKToken(uint32_t tokenIndex)
{
    Duplicate<float>(ubAccFp32_, (float)0, axisH_);
    DataCopyPadParams padParams = {false, 0, 0, 0};
    DataCopyParams xCopyParams = {1U, static_cast<uint16_t>(hAlignSize_), 0U, 0U};
    DataCopyParams weightCopyParams = {1U, static_cast<uint16_t>(sizeof(float)), 0U, 0U};
    DataCopyExtParams indexCopyParams{1U, static_cast<uint32_t>(topK_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> indexPadParams{false, 0U, 0U, 0U};
    DataCopyPad(localIndexTensor_, localRecvXIndexGm_[static_cast<uint64_t>(tokenIndex) * topK_], indexCopyParams,
                indexPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    for (uint32_t topkId = 0U; topkId < topK_; topkId++) {
        bool maskExpertFlag = maskGenerateTensor_.GetValue((tokenIndex - tStart_) * topK_ + topkId);
        if (!maskExpertFlag) {
            continue;
        }
        int32_t localRecvXIndex = localIndexTensor_.GetValue(topkId) - 1;
        GM_ADDR tokenAddr = nullptr;
        if (localRecvXIndex >= 0) {
            tokenAddr = reinterpret_cast<GM_ADDR>(
                xGm_.GetPhyAddr(static_cast<uint64_t>(localRecvXIndex) * axisH_));
        } else {
            uint64_t slotOffset =
                (static_cast<uint64_t>(tokenIndex) * topK_ + topkId) * tilingData_->cfg.perSlotBytes;
            tokenAddr = GetUrmaWinAddrByRankId(rankId_, combineDataWinOffset_) + slotOffset;
        }
        GlobalTensor<XType> srcTokenTensor;
        srcTokenTensor.SetGlobalBuffer(reinterpret_cast<__gm__ XType *>(tokenAddr));
        LocalTensor<XType> xLocal = xInQue_.AllocTensor<XType>();
        DataCopyPad(xLocal, srcTokenTensor, xCopyParams, padParams);
        xInQue_.EnQue(xLocal);
        LocalTensor<XType> xIn = xInQue_.DeQue<XType>();
        Cast(ubTmpFp32_, xIn, AscendC::RoundMode::CAST_NONE, axisH_);
        Add(ubAccFp32_, ubAccFp32_, ubTmpFp32_, axisH_);
        xInQue_.FreeTensor(xIn);

        if constexpr (HasTopkWeight == 1) {
            GM_ADDR weightAddr = localRecvXIndex >= 0 ?
                                     reinterpret_cast<GM_ADDR>(topkWeightsGm_.GetPhyAddr(localRecvXIndex)) :
                                     GetUrmaStateAddrByRankId(rankId_, combineStateWinOffset_) +
                                         (tokenIndex * topK_ + topkId) * WIN_ADDR_ALIGN;
            GlobalTensor<float> srcWeightTensor;
            srcWeightTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weightAddr));
            LocalTensor<float> weightLocal = weightQue_.AllocTensor<float>();
            DataCopyPad(weightLocal, srcWeightTensor, weightCopyParams, padParams);
            weightQue_.EnQue(weightLocal);
            LocalTensor<float> weightOut = weightQue_.DeQue<float>();
            DataCopyPad(combinedTopkWeightsGm_[tokenIndex * topK_ + topkId], weightOut, weightCopyParams);
            weightQue_.FreeTensor(weightOut);
        }
    }
    LocalTensor<XType> ubResultBf16 = xOutQue_.AllocTensor<XType>();
    Cast(ubResultBf16, ubAccFp32_, RoundMode::CAST_RINT, axisH_);
    xOutQue_.EnQue(ubResultBf16);
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline void MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::RecvPhaseReduce()
{
    uint32_t completionChannelCount = GetCompletionChannelCount();
    uint32_t totalFlagCount = epWorldSize_ * combineChannelCount_;
    uint32_t flagBufferBytes = totalFlagCount * STATE_OFFSET;
    tpipe_->InitBuffer(stateBuf_, flagBufferBytes);
    tpipe_->InitBuffer(waitSumBuf_, UB_ALIGN);

    // Remote NBI writes can progress while every AIV builds its share of the local direct-read index.
    BuildLocalIndexMap();
    if (aivId_ == 0U) {
        while (!WaitDispatch(completionChannelCount)) {
        }
        ClearCompletionFlags();
    }
    SyncAll<true>();
    diagWriter_.RunPosRecord(MOE_EP_COMBINE_EPILOGUE_RUN_POS_WAIT_DONE);

    if (tPerCore_ == 0) {
        return;
    }

    DataCopyPadParams padParams = {false, 0, 0, 0};
    DataCopyParams xCopyParams = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    for (uint32_t tokenIdx = tStart_; tokenIdx < tEnd_; ++tokenIdx) {
        ProcessTopKToken(tokenIdx);
        LocalTensor<XType> ubResult = xOutQue_.DeQue<XType>();
        DataCopyPad(combinedXGm_[tokenIdx * axisH_], ubResult, xCopyParams);
        xOutQue_.FreeTensor(ubResult);
    }
    diagWriter_.RunPosRecord(MOE_EP_COMBINE_EPILOGUE_RUN_POS_OUTPUT_DONE);
}

template <TemplateMoeEpCombineEpilogueTypeClass>
__aicore__ inline void MoeEpCombineEpilogue<TemplateMoeEpCombineEpilogueTypeFunc>::Process()
{
    BuffInit();
    MaskCheck();
    RecvPhaseReduce();
}

#endif

} // namespace MoeEpCombineEpilogueImpl

#endif // MOE_EP_COMBINE_EPILOGUE_H
