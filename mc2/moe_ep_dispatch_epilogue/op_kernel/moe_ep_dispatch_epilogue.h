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
 * \file moe_ep_dispatch_epilogue.h
 * \brief
 */

#ifndef MOE_EP_DISPATCH_EPILOGUE_H
#define MOE_EP_DISPATCH_EPILOGUE_H

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if (ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)) && \
    (HCOMM_MAJOR > 9 || (HCOMM_MAJOR == 9 && HCOMM_MINOR > 0))
#define ENABLE_MOE_EP_DISPATCH_EPILOGUE_KERNEL
#endif

#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "moe_ep_dispatch_epilogue_tiling_key.h"
#include "moe_ep_dispatch_epilogue_tiling.h"

#if __has_include("../common/moe_distribute_base.h")
#include "../common/moe_distribute_base.h"
#include "../common/mc2_kernel_utils.h"
#include "../common/mc2_moe_context.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#include "../../common/op_kernel/mc2_moe_context.h"
#endif

namespace MoeEpDispatchEpilogueImpl {

#if defined(ENABLE_MOE_EP_DISPATCH_EPILOGUE_KERNEL)

using namespace AscendC;

static constexpr uint32_t UB_ALIGN = 32U;
static constexpr uint32_t WIN_ADDR_ALIGN = 512;
// recv_src_metadata is compact in GM: [srcRank, srcToken, srcTopk, srcSlot, recvXIdx].
// UB staging still uses an aligned stride so that every non-aligned 20-byte copy starts from a 32-byte boundary.
static constexpr uint32_t RECV_META_FIELDS = 5;
static constexpr uint8_t BUFFER_NUM = 2;
static constexpr uint32_t ELEM_ALIGN = 8U;
static constexpr uint32_t META_TOPK_SECTION = 2U;
static constexpr uint32_t META_EXTRA_FIELDS = 2U;
static constexpr uint32_t META_SRC_RANK_OFFSET = 0U;
static constexpr uint32_t META_TOKEN_IDX_OFFSET = 1U;
static constexpr uint32_t META_TOPK_IDX_OFFSET = 2U;
static constexpr uint32_t META_SLOT_IDX_OFFSET = 3U;
static constexpr uint32_t META_RECV_X_IDX_OFFSET = 4U;
static constexpr uint32_t HIT_ROW_OFFSET = 0U;
static constexpr uint32_t HIT_TOPK_OFFSET = 1U;
static constexpr uint32_t HIT_ENTRY_SIZE = 2U;
// Five-field metadata needs 20 bytes per row. A 4096-row tile keeps this cached staging buffer at 80 KiB.
static constexpr uint32_t CACHED_META_TILE = 4096U;
static constexpr uint32_t ALIGNED_LEN_256 = 256U;
static constexpr uint32_t SLOTS_TILE = 128U;

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
class MoeEpDispatchEpilogue {
public:
    __aicore__ inline MoeEpDispatchEpilogue(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR dstBufferSlotIdx, GM_ADDR numRecvPerRank,
                                GM_ADDR numRecvPerExpert, GM_ADDR cachedRecvSrcMetadata, GM_ADDR recvX,
                                GM_ADDR recvSrcMetadata, GM_ADDR recvTopkWeights, GM_ADDR recvScales, GM_ADDR workspace,
                                GM_ADDR tilingGM, TPipe *pipe, const MoeEpDispatchEpilogueInfo *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ComputePrefixSums();
    __aicore__ inline void CountHits();
    __aicore__ inline void BuildRankRowStarts();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void CopyFromWindowByExpert();
    __aicore__ inline void CopyFromWindowByCachedMeta();

    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startId, uint32_t &endId,
                                       uint32_t &sendNum);
    __aicore__ inline GM_ADDR GetWinAddrByRankId(__gm__ Mc2Aclnn::MoeCommContext *ctx, uint32_t rankId, uint64_t offset)
    {
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        return (GM_ADDR)ctx->epHcclBuffer[rankId] + offset;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }
    __aicore__ inline uint32_t ReduceSumWorkNeedSize(int32_t count, int32_t typeSize)
    {
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        int32_t elementsPerBlock = UB_ALIGN / typeSize;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        int32_t elementsPerRepeat = ALIGNED_LEN_256 / typeSize;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        int32_t iter1OutputCount = (count + elementsPerRepeat - 1) / elementsPerRepeat;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint32_t iter1AlignEnd = ((iter1OutputCount + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        return iter1AlignEnd;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }

    TPipe *tpipe_{nullptr};
    __gm__ Mc2Aclnn::MoeCommContext *mc2Context_{nullptr};
    uint32_t epRankId_{0};
    uint32_t aivId_{0};
    GlobalTensor<int32_t> numRecvPerRankGm_;
    GlobalTensor<int64_t> numRecvPerExpertGm_;

    GlobalTensor<XType> recvXGm_;
    GlobalTensor<float> recvTopkWeightsGm_;
    GlobalTensor<int32_t> recvSrcMetadataGm_;
    GlobalTensor<ScalesType> recvScalesGm_;
    GlobalTensor<int32_t> cachedRecvSrcMetadataGm_; // cached 路径专用：来自上一轮 dispatch 的 recv_src_metadata

    GlobalTensor<int32_t> hitCountGm_;
    GlobalTensor<int32_t> rankHitCountGm_;

    LocalTensor<int32_t> ubHitCount_;
    LocalTensor<int32_t> ubRankHitCount_;
    LocalTensor<int32_t> ubRankRowStart_;
    LocalTensor<int64_t> ubRowStart_;
    LocalTensor<int32_t> ubMeta_;
    LocalTensor<int32_t> ubTopkIds_;
    LocalTensor<int32_t> ubTargetExpertId_;
    LocalTensor<int32_t> ubRecvCnt_;
    LocalTensor<int64_t> ubExpertPfx_;
    LocalTensor<int64_t> ubHitCountRowI64_;
    LocalTensor<float> ubStageWeights_;
    LocalTensor<int32_t> ubStageMeta_;
    LocalTensor<int32_t> ubLocalCursor_;
    LocalTensor<int64_t> ubHitList_;
    LocalTensor<int32_t> ubWaitStatus_;
    LocalTensor<int32_t> ubWaitSum_;

    TBuf<QuePosition::VECIN> ubHitCountBuf_;
    TBuf<QuePosition::VECIN> ubRankHitCountBuf_;
    TBuf<QuePosition::VECIN> ubRankRowStartBuf_;
    TBuf<QuePosition::VECIN> ubRowStartBuf_;
    TBuf<QuePosition::VECIN> ubMetaBuf_;
    TBuf<QuePosition::VECIN> ubTopkIdsBuf_;
    TBuf<QuePosition::VECIN> ubTargetExpertIdBuf_;
    TBuf<QuePosition::VECIN> ubRecvCntBuf_;
    TBuf<QuePosition::VECIN> ubExpertPfxBuf_;
    TBuf<QuePosition::VECIN> ubHitCountRowI64Buf_;
    TBuf<QuePosition::VECIN> ubStageWeightsBuf_;
    TBuf<QuePosition::VECIN> ubStageMetaBuf_;
    TBuf<QuePosition::VECIN> ubLocalCursorBuf_;
    TBuf<QuePosition::VECIN> ubHitListBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> tokenQueue_;
    TBuf<> waitStatusBuf_;
    TBuf<> waitSumBuf_;
    TBuf<> sharedTmpBuf_;

    uint32_t expertSum_{0};
    GM_ADDR workspaceGM_{nullptr};
    GM_ADDR localWinAddr_{nullptr};
    GM_ADDR localSlotStateWinAddr_{nullptr};
    uint32_t scalesOffset_{0};
    uint32_t scalesBytes_{0};
    uint32_t scalesElems_{0};
    uint32_t metaOffset_{0};
    uint32_t tokenQueueBufBytes_{0};
    uint32_t hitCountOffset_{0};
    uint32_t metaBytes_{0};
    uint32_t paddedMetaElems_{0};
    uint32_t axisKAlign_{0};
    uint32_t paddedTopkElems_{0};
    uint32_t numLocalExperts_{0};
    uint32_t hitCountStride_{0}; // 对齐到 32B 的 hitCount 存储步长 (int32 元素数)
    uint32_t rankHitCountStride_{0}; // per-core rank count/start stride, aligned to 32 bytes
    uint32_t aivNum_{0};
    uint32_t axisK_{0};
    uint32_t axisH_{0};
    uint32_t epWorldSize_{0};
    uint32_t numMaxTokensPerRank_{0};
    uint32_t perSlotBytes_{0};
    uint32_t dispatchNotifyCount_{1};
    uint32_t totalNotifyCnt_{0};
    uint64_t winDataOffset_{0};
    uint64_t slotWinStateOffset_{0};
    int32_t ppEvtSToMte3_[2] = {0, 0};
    int32_t ppEvtMte3ToS_[2] = {0, 0};
};

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startId, uint32_t &endId, uint32_t &sendNum)
{
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    sendNum = curSendCnt / curUseAivNum;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    uint32_t remainderNum = curSendCnt % curUseAivNum;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    uint32_t newAivId = aivId_;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    startId = sendNum * newAivId;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    if (newAivId < remainderNum) {
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        sendNum += 1;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        startId += newAivId;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    } else {
        startId += remainderNum;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }
    endId = startId + sendNum;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::Init(
    GM_ADDR context, GM_ADDR dstBufferSlotIdx, GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert,
    GM_ADDR cachedRecvSrcMetadata, GM_ADDR recvX, GM_ADDR recvSrcMetadata, GM_ADDR recvTopkWeights, GM_ADDR recvScales,
    GM_ADDR workspace, GM_ADDR tilingGM, TPipe *pipe, const MoeEpDispatchEpilogueInfo *tilingData)
{
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    tpipe_ = pipe;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    aivId_ = GetBlockIdx();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    workspaceGM_ = workspace;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    numLocalExperts_ = tilingData->cfg.numLocalExperts;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    aivNum_ = tilingData->aivNum;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    axisK_ = tilingData->cfg.topK;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    axisH_ = tilingData->cfg.hidden;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    epWorldSize_ = tilingData->cfg.epWorldSize;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    numMaxTokensPerRank_ = tilingData->cfg.numMaxTokensPerRank;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    perSlotBytes_ = tilingData->cfg.perSlotBytes;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    dispatchNotifyCount_ = tilingData->dispatchNotifyCount;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    winDataOffset_ = tilingData->winDataOffset;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    slotWinStateOffset_ = tilingData->slotWinStateOffset;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::MoeCommContext *>(context);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    epRankId_ = mc2Context_->epRankId;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    localSlotStateWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, slotWinStateOffset_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    localWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, winDataOffset_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    metaOffset_ = Ceil((uint32_t)(axisH_ * sizeof(XType)), UB_ALIGN) * UB_ALIGN;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    numRecvPerRankGm_.SetGlobalBuffer((__gm__ int32_t *)numRecvPerRank);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    numRecvPerExpertGm_.SetGlobalBuffer((__gm__ int64_t *)numRecvPerExpert);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    cachedRecvSrcMetadataGm_.SetGlobalBuffer((__gm__ int32_t *)cachedRecvSrcMetadata);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    recvXGm_.SetGlobalBuffer((__gm__ XType *)recvX);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    recvSrcMetadataGm_.SetGlobalBuffer((__gm__ int32_t *)recvSrcMetadata);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    if constexpr (HasTopkWeights) {
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        recvTopkWeightsGm_.SetGlobalBuffer((__gm__ float *)recvTopkWeights);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }

    hitCountGm_.SetGlobalBuffer((__gm__ int32_t *)(workspace));
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    rankHitCountGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(workspace + tilingData->rankHitCountOffset));
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    axisKAlign_ = Ceil(axisK_, ELEM_ALIGN) * ELEM_ALIGN;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    metaBytes_ = (META_TOPK_SECTION * axisKAlign_) * (uint32_t)sizeof(int32_t) + UB_ALIGN;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    totalNotifyCnt_ = epWorldSize_ * dispatchNotifyCount_;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    uint32_t ubExpertPfxBytes = Ceil((uint32_t)(numLocalExperts_ * sizeof(int64_t)), UB_ALIGN) * UB_ALIGN;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    uint32_t expertReduceTmpBytes =
        ReduceSumWorkNeedSize(static_cast<int32_t>(numLocalExperts_), sizeof(int64_t)) * sizeof(int64_t);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    uint32_t statusReduceTmpBytes =
        ReduceSumWorkNeedSize(static_cast<int32_t>(totalNotifyCnt_), sizeof(float)) * sizeof(float);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    uint32_t sharedBytes = expertReduceTmpBytes > statusReduceTmpBytes ? expertReduceTmpBytes : statusReduceTmpBytes;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    tpipe_->InitBuffer(ubExpertPfxBuf_, ubExpertPfxBytes);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    tpipe_->InitBuffer(waitStatusBuf_, totalNotifyCnt_ * UB_ALIGN);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    tpipe_->InitBuffer(waitSumBuf_, UB_ALIGN);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    tpipe_->InitBuffer(sharedTmpBuf_, sharedBytes);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    ubExpertPfx_ = ubExpertPfxBuf_.Get<int64_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    ubWaitStatus_ = waitStatusBuf_.Get<int32_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    ubWaitSum_ = waitSumBuf_.Get<int32_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    if constexpr (!IsCached) {
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        hitCountStride_ = Ceil(numLocalExperts_, ELEM_ALIGN) * ELEM_ALIGN;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        rankHitCountStride_ = Ceil(epWorldSize_, ELEM_ALIGN) * ELEM_ALIGN;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        paddedMetaElems_ = Ceil(META_TOPK_SECTION * axisKAlign_ + META_EXTRA_FIELDS, ELEM_ALIGN) * ELEM_ALIGN;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        paddedTopkElems_ = axisKAlign_;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint32_t ubHitCountBytes = Ceil((uint32_t)(numLocalExperts_ * sizeof(int32_t)), UB_ALIGN) * UB_ALIGN;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint32_t ubRankCountBytes = Ceil((uint32_t)(epWorldSize_ * sizeof(int32_t)), UB_ALIGN) * UB_ALIGN;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint32_t ubRowStartBytes = Ceil((uint32_t)(numLocalExperts_ * sizeof(int64_t)), UB_ALIGN) * UB_ALIGN;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint32_t ubMetaBytes =
            Ceil((uint32_t)(SLOTS_TILE * paddedMetaElems_ * sizeof(int32_t)), UB_ALIGN) * UB_ALIGN;
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint32_t ubTopkIdsBytes =
            Ceil((uint32_t)(SLOTS_TILE * paddedTopkElems_ * sizeof(int32_t)), UB_ALIGN) * UB_ALIGN;
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint32_t ubRecvCntBytes = Ceil((uint32_t)(epWorldSize_ * sizeof(int32_t)), UB_ALIGN) * UB_ALIGN;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint32_t ubHitCountRowI64Bytes = Ceil((uint32_t)(numLocalExperts_ * sizeof(int64_t)), UB_ALIGN) * UB_ALIGN;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint32_t ubStageMetaBytes = axisK_ * UB_ALIGN * 2;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint32_t ubStageWeightsBytes = axisK_ * UB_ALIGN * 2;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint32_t ubHitListBytes = Ceil((uint32_t)(axisK_ * HIT_ENTRY_SIZE * sizeof(int64_t)), UB_ALIGN) * UB_ALIGN;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint32_t ubLocalCursorBytes = Ceil((uint32_t)(numLocalExperts_ * sizeof(int32_t)), UB_ALIGN) * UB_ALIGN;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

        tpipe_->InitBuffer(ubRecvCntBuf_, ubRecvCntBytes);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        tpipe_->InitBuffer(ubHitCountBuf_, ubHitCountBytes);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        tpipe_->InitBuffer(ubRankHitCountBuf_, ubRankCountBytes);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        tpipe_->InitBuffer(ubRankRowStartBuf_, ubRankCountBytes);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        tpipe_->InitBuffer(ubRowStartBuf_, ubRowStartBytes);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        tpipe_->InitBuffer(ubMetaBuf_, ubMetaBytes);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        tpipe_->InitBuffer(ubTopkIdsBuf_, ubTopkIdsBytes);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        tpipe_->InitBuffer(ubTargetExpertIdBuf_, ubTopkIdsBytes);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        tpipe_->InitBuffer(ubHitCountRowI64Buf_, ubHitCountRowI64Bytes);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        tpipe_->InitBuffer(ubStageWeightsBuf_, ubStageWeightsBytes);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        tpipe_->InitBuffer(ubStageMetaBuf_, ubStageMetaBytes);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        tpipe_->InitBuffer(ubHitListBuf_, ubHitListBytes);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        tpipe_->InitBuffer(ubLocalCursorBuf_, ubLocalCursorBytes);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        ubRecvCnt_ = ubRecvCntBuf_.Get<int32_t>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        ubHitCount_ = ubHitCountBuf_.Get<int32_t>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        ubRankHitCount_ = ubRankHitCountBuf_.Get<int32_t>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        ubRankRowStart_ = ubRankRowStartBuf_.Get<int32_t>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        ubRowStart_ = ubRowStartBuf_.Get<int64_t>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        ubMeta_ = ubMetaBuf_.Get<int32_t>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        ubTopkIds_ = ubTopkIdsBuf_.Get<int32_t>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        ubTargetExpertId_ = ubTargetExpertIdBuf_.Get<int32_t>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        ubHitCountRowI64_ = ubHitCountRowI64Buf_.Get<int64_t>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        ubStageWeights_ = ubStageWeightsBuf_.Get<float>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        ubStageMeta_ = ubStageMetaBuf_.Get<int32_t>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        ubHitList_ = ubHitListBuf_.Get<int64_t>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        ubLocalCursor_ = ubLocalCursorBuf_.Get<int32_t>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        ppEvtSToMte3_[0] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        ppEvtSToMte3_[1] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        ppEvtMte3ToS_[0] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S));
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        ppEvtMte3ToS_[1] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S));
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }

    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        scalesOffset_ = Ceil((uint32_t)(axisH_ * sizeof(XType)), UB_ALIGN) * UB_ALIGN;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        scalesBytes_ = tilingData->cfg.scalesBytes;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        scalesElems_ = (scalesBytes_ == 0U) ? 0U : (scalesBytes_ / sizeof(ScalesType));
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint32_t scalesBytesAlign = Ceil(scalesBytes_, UB_ALIGN) * UB_ALIGN;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        metaOffset_ += scalesBytesAlign;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        recvScalesGm_.SetGlobalBuffer((__gm__ ScalesType *)recvScales);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }

    tokenQueueBufBytes_ = metaOffset_;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    tpipe_->InitBuffer(tokenQueue_, BUFFER_NUM, tokenQueueBufBytes_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    DataCopyExtParams expertPfxCopyParams{1U, static_cast<uint32_t>(numLocalExperts_ * sizeof(int64_t)), 0U, 0U, 0U};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyPadExtParams<int64_t> expertPfxPadParams{false, 0U, 0U, 0};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyPad(ubExpertPfx_, numRecvPerExpertGm_, expertPfxCopyParams, expertPfxPadParams);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    LocalTensor<int64_t> expertReduceTmp = sharedTmpBuf_.Get<int64_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<int64_t> expertSumOut = waitSumBuf_.Get<int64_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    ReduceSum<int64_t>(expertSumOut, ubExpertPfx_, expertReduceTmp, static_cast<int32_t>(numLocalExperts_));
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    SyncFunc<AscendC::HardEvent::V_S>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    expertSum_ = static_cast<uint32_t>(expertSumOut.GetValue(0));
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::Process()
{
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    if constexpr (!IsCached) {
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        ComputePrefixSums();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        WaitDispatch();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        SyncAll<true>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        CountHits();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        SyncAll<true>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

        // Every core builds its own compact metadata row starts from the shared per-core rank-count matrix.
        // The starts stay in local UB, so no post-build cross-core synchronization is needed.
        BuildRankRowStarts();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

        uint32_t totalHits = 0;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        for (uint32_t localExpertIdx = 0; localExpertIdx < numLocalExperts_; ++localExpertIdx) {
            totalHits += static_cast<uint32_t>(ubHitCount_.GetValue(localExpertIdx));
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        }

        if (totalHits != 0) {
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            CopyFromWindowByExpert();
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        }
    } else {
        WaitDispatch();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        SyncAll<true>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        CopyFromWindowByCachedMeta();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::WaitDispatch()
{
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    if (aivId_ != aivNum_ - 1) {
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        return;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }

    uint32_t mask = 1;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    int32_t sumOfFlag = 0;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    int32_t commpareFlag = static_cast<int32_t>(totalNotifyCnt_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    GlobalTensor<int32_t> statusGMTensor;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<float> sharedTmp = sharedTmpBuf_.Get<float>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<float> ubWaitStatusFp32 = ubWaitStatus_.template ReinterpretCast<float>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<float> ubWaitSumFp32 = ubWaitSum_.template ReinterpretCast<float>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    statusGMTensor.SetGlobalBuffer((__gm__ int32_t *)localSlotStateWinAddr_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyParams statusCopyParams = {static_cast<uint16_t>(totalNotifyCnt_), 1U,
                                       static_cast<uint16_t>((WIN_ADDR_ALIGN - UB_ALIGN) / UB_ALIGN), 0U};
                                       PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyParams clearStatusCopyParams = {static_cast<uint16_t>(totalNotifyCnt_), 1U, 0U,
                                            static_cast<uint16_t>((WIN_ADDR_ALIGN - UB_ALIGN) / UB_ALIGN)};
                                            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    SyncFunc<AscendC::HardEvent::S_V>(); // 确保expertSum_计算完成
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    while (sumOfFlag != commpareFlag) {
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        DataCopy(ubWaitStatus_, statusGMTensor, statusCopyParams);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        ReduceSum(ubWaitSumFp32, ubWaitStatusFp32, sharedTmp, mask, totalNotifyCnt_, 1);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        SyncFunc<AscendC::HardEvent::V_S>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        sumOfFlag = ubWaitSum_.GetValue(0);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }
    Duplicate<int32_t>(ubWaitStatus_, 0, totalNotifyCnt_ * UB_ALIGN / sizeof(int32_t));
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopy(statusGMTensor, ubWaitStatus_, clearStatusCopyParams);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::ComputePrefixSums()
{
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    int64_t cumulativeRowOffset = 0;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    for (uint32_t localExpertIdx = 0; localExpertIdx < numLocalExperts_; ++localExpertIdx) {
        int64_t expertTokenCnt = ubExpertPfx_.GetValue(localExpertIdx);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        ubExpertPfx_.SetValue(localExpertIdx, cumulativeRowOffset);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        cumulativeRowOffset += expertTokenCnt;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }
    SyncFunc<AscendC::HardEvent::S_V>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::CountHits()
{
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyExtParams recvCntCopyParams{1U, static_cast<uint32_t>(epWorldSize_ * sizeof(int32_t)), 0U, 0U, 0U};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyPadExtParams<int32_t> recvCntPadParams{false, 0U, 0U, 0};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyPad(ubRecvCnt_, numRecvPerRankGm_, recvCntCopyParams, recvCntPadParams);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    Duplicate(ubHitCount_, (int32_t)0, numLocalExperts_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    Duplicate(ubRankHitCount_, (int32_t)0, epWorldSize_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    for (uint32_t rankId = 0; rankId < epWorldSize_; ++rankId) {
        int32_t slotCnt = ubRecvCnt_.GetValue(rankId);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        if (slotCnt == 0) {
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            continue;
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        }

        int32_t rankHitCount = 0;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

        uint32_t slotStart, slotEnd, slotCntPerAiv;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        SplitToCore(static_cast<uint32_t>(slotCnt), aivNum_, slotStart, slotEnd, slotCntPerAiv);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        if (slotStart >= slotEnd) {
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            continue;
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        }

        GM_ADDR srcRankBase = localWinAddr_ + (int64_t)rankId * numMaxTokensPerRank_ * perSlotBytes_;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        GlobalTensor<int32_t> srcTopkIdsGm;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        srcTopkIdsGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(srcRankBase + (int64_t)slotStart * perSlotBytes_ + metaOffset_));
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

        uint32_t topkBytes = axisK_ * sizeof(int32_t);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        LocalTensor<uint8_t> sharedTmpInt8 = ubMetaBuf_.Get<uint8_t>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        LocalTensor<uint32_t> sharedTmpInt32 = ubMetaBuf_.Get<uint32_t>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

        for (uint32_t tileStart = 0; tileStart < slotCntPerAiv; tileStart += SLOTS_TILE) {
            uint32_t tileCnt = (slotCntPerAiv - tileStart > SLOTS_TILE) ? SLOTS_TILE : (slotCntPerAiv - tileStart);
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

            DataCopyExtParams topkCopyParams{static_cast<uint16_t>(tileCnt), static_cast<uint32_t>(topkBytes),
                                             static_cast<int64_t>(perSlotBytes_ - topkBytes), 0, 0};
                                             PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            DataCopyPadExtParams<int32_t> topkPadParams{true, 0, static_cast<uint8_t>(paddedTopkElems_ - axisK_), -1};
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            DataCopyPad(ubTopkIds_, srcTopkIdsGm[(int64_t)tileStart * (perSlotBytes_ / sizeof(int32_t))],
                        topkCopyParams, topkPadParams);
                        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

            int32_t calCnt = static_cast<int32_t>(tileCnt * paddedTopkElems_);
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            constexpr int32_t CMP_ALIGN = ALIGNED_LEN_256 / sizeof(int32_t);
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            int32_t calCntAlign = Ceil(calCnt, CMP_ALIGN) * CMP_ALIGN;
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            for (uint32_t localExpertId = 0; localExpertId < numLocalExperts_; ++localExpertId) {
                int32_t targetExpertId = static_cast<int32_t>(epRankId_ * numLocalExperts_ + localExpertId);
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                uint64_t rsvdCnt = 0;
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                CompareScalar(sharedTmpInt8, ubTopkIds_, static_cast<int32_t>(targetExpertId), AscendC::CMPMODE::EQ,
                              calCntAlign);
                              PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                GatherMask(ubTargetExpertId_, ubTopkIds_, sharedTmpInt32, true, calCnt, {1, 1, 0, 0}, rsvdCnt);
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                SyncFunc<AscendC::HardEvent::V_S>();
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                int32_t curExpertCnt = rsvdCnt;
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                int32_t currentHits = ubHitCount_.GetValue(localExpertId);
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                ubHitCount_.SetValue(localExpertId, currentHits + curExpertCnt);
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                rankHitCount += curExpertCnt;
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            }
            SyncFunc<AscendC::HardEvent::V_MTE2>();
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        }
        ubRankHitCount_.SetValue(rankId, rankHitCount);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }

    SyncFunc<AscendC::HardEvent::V_MTE3>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyExtParams hitCountCopyParams{1U, static_cast<uint32_t>(numLocalExperts_ * sizeof(int32_t)), 0U, 0U, 0U};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyPad(hitCountGm_[(int64_t)aivId_ * hitCountStride_], ubHitCount_, hitCountCopyParams);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyExtParams rankHitCountCopyParams{1U, static_cast<uint32_t>(epWorldSize_ * sizeof(int32_t)), 0U, 0U, 0U};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyPad(rankHitCountGm_[(int64_t)aivId_ * rankHitCountStride_], ubRankHitCount_, rankHitCountCopyParams);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void
MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::BuildRankRowStarts()
{
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    // CountHits used ubRankHitCount_ as an MTE3 source. Finish that read before reusing the buffer as the
    // per-rank prefix contributed by cores preceding this core.
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    // rankTotal[r] = sum(count[core][r]) for all cores.
    // corePrefix[r] = sum(count[core][r]) for core < aivId_.
    Duplicate(ubRankRowStart_, (int32_t)0, epWorldSize_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    Duplicate(ubRankHitCount_, (int32_t)0, epWorldSize_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    // Reuse the metadata staging buffer to batch contiguous core rows. The loop processes the complete matrix in
    // one tile when it fits, and otherwise splits it by complete core rows.
    uint32_t ubMetaBytes =
        Ceil((uint32_t)(SLOTS_TILE * paddedMetaElems_ * sizeof(int32_t)), UB_ALIGN) * UB_ALIGN;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    uint32_t rankRowBytes = rankHitCountStride_ * sizeof(int32_t);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    uint32_t coreRowsPerTile = ubMetaBytes / rankRowBytes;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    coreRowsPerTile = coreRowsPerTile > aivNum_ ? aivNum_ : coreRowsPerTile;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    DataCopyPadExtParams<int32_t> rankMatrixPadParams{false, 0U, 0U, 0};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    for (uint32_t coreBase = 0; coreBase < aivNum_; coreBase += coreRowsPerTile) {
        uint32_t rowsThisTile =
            (aivNum_ - coreBase > coreRowsPerTile) ? coreRowsPerTile : (aivNum_ - coreBase);
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint32_t copyBytes = rowsThisTile * rankRowBytes;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        DataCopyExtParams rankMatrixCopyParams{1U, copyBytes, 0U, 0U, 0U};
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        DataCopyPad(ubMeta_, rankHitCountGm_[(int64_t)coreBase * rankHitCountStride_], rankMatrixCopyParams,
                    rankMatrixPadParams);
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

        for (uint32_t localCoreId = 0; localCoreId < rowsThisTile; ++localCoreId) {
            uint32_t coreId = coreBase + localCoreId;
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            LocalTensor<int32_t> rankCountRow = ubMeta_[localCoreId * rankHitCountStride_];
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            Add(ubRankRowStart_, ubRankRowStart_, rankCountRow, epWorldSize_);
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            if (coreId < aivId_) {
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                Add(ubRankHitCount_, ubRankHitCount_, rankCountRow, epWorldSize_);
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            }
        }

        // The next tile, or CopyFromWindowByExpert after the last tile, reuses ubMeta_ through MTE2. Wait until all
        // vector reads of this tile have completed before allowing that overwrite.
        SyncFunc<AscendC::HardEvent::V_MTE2>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }

    // Convert rank totals and this core's within-rank prefix to the final exclusive row starts. The total number of
    // rows is bounded by INT32_MAX in tiling. Both the producer and the following consumer use the scalar pipeline,
    // so no extra synchronization is needed after these SetValue calls.
    SyncFunc<AscendC::HardEvent::V_S>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    int32_t rankPrefix = 0;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    for (uint32_t rankId = 0; rankId < epWorldSize_; ++rankId) {
        int32_t rankCount = ubRankRowStart_.GetValue(rankId);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        int32_t corePrefix = ubRankHitCount_.GetValue(rankId);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        ubRankRowStart_.SetValue(rankId, rankPrefix + corePrefix);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        rankPrefix += rankCount;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::CopyFromWindowByExpert()
{
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyExtParams hitCountOneCopyParams{1U, static_cast<uint32_t>(numLocalExperts_ * sizeof(int32_t)), 0U, 0U, 0U};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyPadExtParams<int32_t> hitCountOnePadParams{false, 0U, 0U, 0};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    Adds(ubRowStart_, ubExpertPfx_, static_cast<int64_t>(0), numLocalExperts_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    for (uint32_t aiv = 0; aiv < aivId_; ++aiv) {
        DataCopyPad(ubHitCount_, hitCountGm_[(int64_t)aiv * hitCountStride_], hitCountOneCopyParams,
                    hitCountOnePadParams);
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        Cast(ubHitCountRowI64_, ubHitCount_, RoundMode::CAST_NONE, numLocalExperts_);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        Add(ubRowStart_, ubRowStart_, ubHitCountRowI64_, numLocalExperts_);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        SyncFunc<AscendC::HardEvent::V_MTE2>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }

    Duplicate(ubLocalCursor_, (int32_t)0, numLocalExperts_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    SyncFunc<AscendC::HardEvent::V_S>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyPadExtParams<int32_t> metaPadParams{false, 0, 0, 0};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyParams tokenCopyParams{1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyPadParams tokenPadParams{false, 0, 0, 0};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyExtParams metaOutParams{1U, static_cast<uint32_t>(RECV_META_FIELDS * sizeof(int32_t)), 0U, 0U, 0U};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyExtParams weightOutParams{1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    int32_t rankExpertBase = static_cast<int32_t>(epRankId_ * numLocalExperts_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    int32_t rankExpertEnd = rankExpertBase + static_cast<int32_t>(numLocalExperts_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    for (uint32_t rankId = 0; rankId < epWorldSize_; ++rankId) {
        int32_t slotCnt = ubRecvCnt_.GetValue(rankId);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        if (slotCnt == 0) {
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            continue;
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        }

        int32_t metadataRow = ubRankRowStart_.GetValue(rankId);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

        uint32_t slotStart, slotEnd, slotCntPerAiv;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        SplitToCore(static_cast<uint32_t>(slotCnt), aivNum_, slotStart, slotEnd, slotCntPerAiv);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        if (slotStart >= slotEnd) {
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            continue;
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        }

        GM_ADDR srcRankBase = localWinAddr_ + (int64_t)rankId * numMaxTokensPerRank_ * perSlotBytes_;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        GlobalTensor<int32_t> srcMetaGm;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        srcMetaGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(srcRankBase + (int64_t)slotStart * perSlotBytes_ + metaOffset_));
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

        for (uint32_t tileStart = 0; tileStart < slotCntPerAiv; tileStart += SLOTS_TILE) {
            uint32_t tileCnt = (slotCntPerAiv - tileStart > SLOTS_TILE) ? SLOTS_TILE : (slotCntPerAiv - tileStart);
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

            DataCopyExtParams metaCopyParams{static_cast<uint16_t>(tileCnt), metaBytes_,
                                             perSlotBytes_ - metaBytes_, 0, 0};
                                             PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            DataCopyPad(ubMeta_, srcMetaGm[(int64_t)tileStart * (perSlotBytes_ / sizeof(int32_t))],
                        metaCopyParams, metaPadParams);
                        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

            for (uint32_t localSlot = 0; localSlot < tileCnt; ++localSlot) {
                uint32_t metaBase = localSlot * paddedMetaElems_;
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                uint32_t slotBufId = localSlot & 1U;
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                uint32_t stageOff = slotBufId * axisK_ * ELEM_ALIGN;
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                uint32_t hitCnt = 0;
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                int32_t srcRankMeta = ubMeta_.GetValue(metaBase + META_TOPK_SECTION * axisKAlign_);
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                int32_t tokenIdxMeta = ubMeta_.GetValue(metaBase + META_TOPK_SECTION * axisKAlign_ + 1);
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                GM_ADDR slotAddr = srcRankBase + (int64_t)(slotStart + tileStart + localSlot) * perSlotBytes_;
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                for (uint32_t topkIdx = 0; topkIdx < axisK_; ++topkIdx) {
                    int32_t expertId = ubMeta_.GetValue(metaBase + topkIdx);
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    if (expertId < rankExpertBase || expertId >= rankExpertEnd) {
                        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                        continue;
                        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    }
                    uint32_t localExpertId = static_cast<uint32_t>(expertId - rankExpertBase);
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    int64_t expertRowStart = ubRowStart_.GetValue(localExpertId);
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    int32_t cursor = ubLocalCursor_.GetValue(localExpertId);
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    ubLocalCursor_.SetValue(localExpertId, cursor + 1);
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    int64_t recvXRow = expertRowStart + cursor;
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    ubHitList_.SetValue(hitCnt * HIT_ENTRY_SIZE + HIT_ROW_OFFSET, recvXRow);
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    ubHitList_.SetValue(hitCnt * HIT_ENTRY_SIZE + HIT_TOPK_OFFSET, static_cast<int64_t>(topkIdx));
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    hitCnt++;
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                }
                if (hitCnt == 0) {
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    if (localSlot >= 2U) {
                        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                        WaitFlag<AscendC::HardEvent::MTE3_S>(ppEvtMte3ToS_[slotBufId]);
                        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    }
                    SetFlag<AscendC::HardEvent::MTE3_S>(ppEvtMte3ToS_[slotBufId]);
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    continue;
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                }

                if (localSlot >= 2U) {
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    WaitFlag<AscendC::HardEvent::MTE3_S>(ppEvtMte3ToS_[slotBufId]);
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                }

                GlobalTensor<XType> srcTokenTensor;
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                srcTokenTensor.SetGlobalBuffer(reinterpret_cast<__gm__ XType *>(slotAddr), axisH_);
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

                LocalTensor<XType> tokenTensor = tokenQueue_.AllocTensor<XType>();
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                DataCopyPad(tokenTensor, srcTokenTensor, tokenCopyParams, tokenPadParams);
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    GlobalTensor<ScalesType> srcScalesTensor;
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    srcScalesTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ScalesType *>(slotAddr + scalesOffset_),
                                                    scalesElems_);
                                                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    DataCopyParams scalesCopyParams{1U, static_cast<uint16_t>(scalesElems_ * sizeof(ScalesType)),
                                                    0U, 0U};
                                                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    DataCopyPadParams scalesPadParams{false, 0, 0, 0};
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    DataCopyPad(tokenTensor[scalesOffset_ / sizeof(XType)].template ReinterpretCast<ScalesType>(),
                                srcScalesTensor, scalesCopyParams, scalesPadParams);
                                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                }
                tokenQueue_.EnQue(tokenTensor);
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                LocalTensor<XType> tokenOut = tokenQueue_.DeQue<XType>();
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

                for (uint32_t i = 0; i < hitCnt; i++) {
                    int64_t recvXRow = ubHitList_.GetValue(i * HIT_ENTRY_SIZE + HIT_ROW_OFFSET);
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    uint32_t topkIdx = static_cast<uint32_t>(ubHitList_.GetValue(i * HIT_ENTRY_SIZE + HIT_TOPK_OFFSET));
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

                    DataCopyPad(recvXGm_[recvXRow * axisH_], tokenOut, tokenCopyParams);
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
                        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                        DataCopyParams scalesCopyParams{1U, static_cast<uint16_t>(scalesElems_ * sizeof(ScalesType)),
                                                        0U, 0U};
                                                        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                        DataCopyPad(recvScalesGm_[recvXRow * scalesElems_],
                                    tokenOut[scalesOffset_ / sizeof(XType)].template ReinterpretCast<ScalesType>(),
                                    scalesCopyParams);
                                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    }

                    if constexpr (HasTopkWeights) {
                        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                        float weights = ubMeta_.ReinterpretCast<float>().GetValue(metaBase + axisKAlign_ + topkIdx);
                        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                        ubStageWeights_.SetValue(stageOff + i * ELEM_ALIGN, weights);
                        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    }
                    ubStageMeta_.SetValue(stageOff + i * ELEM_ALIGN + META_SRC_RANK_OFFSET, srcRankMeta);
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    ubStageMeta_.SetValue(stageOff + i * ELEM_ALIGN + META_TOKEN_IDX_OFFSET, tokenIdxMeta);
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    ubStageMeta_.SetValue(stageOff + i * ELEM_ALIGN + META_TOPK_IDX_OFFSET,
                                          static_cast<int32_t>(topkIdx));
                                          PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    ubStageMeta_.SetValue(stageOff + i * ELEM_ALIGN + META_SLOT_IDX_OFFSET,
                                          static_cast<int32_t>(slotStart + tileStart + localSlot));
                                          PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    ubStageMeta_.SetValue(stageOff + i * ELEM_ALIGN + META_RECV_X_IDX_OFFSET,
                                          static_cast<int32_t>(recvXRow));
                                          PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                }
                tokenQueue_.FreeTensor(tokenOut);
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

                SetFlag<AscendC::HardEvent::S_MTE3>(ppEvtSToMte3_[slotBufId]);
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                WaitFlag<AscendC::HardEvent::S_MTE3>(ppEvtSToMte3_[slotBufId]);
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                for (uint32_t i = 0; i < hitCnt; i++) {
                    int64_t recvXRow = ubHitList_.GetValue(i * HIT_ENTRY_SIZE + HIT_ROW_OFFSET);
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    if constexpr (HasTopkWeights) {
                        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                        DataCopyPad(recvTopkWeightsGm_[recvXRow], ubStageWeights_[stageOff + i * ELEM_ALIGN],
                                    weightOutParams);
                                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    }
                    DataCopyPad(recvSrcMetadataGm_[(int64_t)metadataRow * RECV_META_FIELDS],
                                ubStageMeta_[stageOff + i * ELEM_ALIGN], metaOutParams);
                                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    metadataRow++;
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                }
                SetFlag<AscendC::HardEvent::MTE3_S>(ppEvtMte3ToS_[slotBufId]);
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            }
            if (tileCnt >= 1U) {
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                WaitFlag<AscendC::HardEvent::MTE3_S>(ppEvtMte3ToS_[(tileCnt - 1U) & 1U]);
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            }
            if (tileCnt >= 2U) {
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                WaitFlag<AscendC::HardEvent::MTE3_S>(ppEvtMte3ToS_[(tileCnt - 2U) & 1U]);
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            }
            SyncFunc<AscendC::HardEvent::S_MTE2>();
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        }
    }
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::CopyFromWindowByCachedMeta()
{
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    if (expertSum_ == 0) {
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        return;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }

    uint32_t startId, endId, cnt;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    SplitToCore(expertSum_, aivNum_, startId, endId, cnt);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    if (startId >= endId) {
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        return;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }

    uint32_t ubMetaBytes = Ceil(metaBytes_, UB_ALIGN) * UB_ALIGN;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    // Cached metadata is rank ordered, while weights remain recv_x ordered. A single aligned staging block is enough
    // for the optional per-row scatter and avoids reserving one padded block per metadata row.
    uint32_t ubStageWeightsBytes = UB_ALIGN;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    uint32_t ubStageMetaBytes =
        Ceil((uint32_t)(CACHED_META_TILE * RECV_META_FIELDS * sizeof(int32_t)), UB_ALIGN) * UB_ALIGN;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    tpipe_->InitBuffer(ubMetaBuf_, ubMetaBytes);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    tpipe_->InitBuffer(ubStageWeightsBuf_, ubStageWeightsBytes);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    tpipe_->InitBuffer(ubStageMetaBuf_, ubStageMetaBytes);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    ubMeta_ = ubMetaBuf_.Get<int32_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    ubStageWeights_ = ubStageWeightsBuf_.Get<float>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    ubStageMeta_ = ubStageMetaBuf_.Get<int32_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    DataCopyPadExtParams<int32_t> metaPadParams{false, 0, 0, 0};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyParams tokenCopyParams{1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyPadParams tokenPadParams{false, 0, 0, 0};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    uint32_t processed = 0;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    while (processed < cnt) {
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint32_t tileCnt = (cnt - processed > CACHED_META_TILE) ? CACHED_META_TILE : (cnt - processed);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint32_t tileStartGlobal = startId + processed;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

        DataCopyExtParams metaInParams{
            1U, static_cast<uint32_t>(tileCnt * RECV_META_FIELDS * sizeof(int32_t)), 0U, 0U, 0U};
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        DataCopyPad(ubStageMeta_, cachedRecvSrcMetadataGm_[(int64_t)tileStartGlobal * RECV_META_FIELDS],
                    metaInParams, metaPadParams);
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

        for (uint32_t i = 0; i < tileCnt; ++i) {
            uint32_t metaBase = i * RECV_META_FIELDS;
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            int32_t srcRankId = ubStageMeta_.GetValue(metaBase + META_SRC_RANK_OFFSET);
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            int32_t srcTopkIdx = ubStageMeta_.GetValue(metaBase + META_TOPK_IDX_OFFSET);
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            int32_t slotIdx = ubStageMeta_.GetValue(metaBase + META_SLOT_IDX_OFFSET);
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            int32_t recvXIdx = ubStageMeta_.GetValue(metaBase + META_RECV_X_IDX_OFFSET);
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

            // recvXIdx is produced by the first non-cached Epilogue. Guard it before using a user-visible cached
            // handle so malformed metadata cannot turn into an out-of-bounds GM write.
            if (recvXIdx < 0 || recvXIdx >= static_cast<int32_t>(expertSum_)) {
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                continue;
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            }

            GM_ADDR slotAddr = localWinAddr_ + (int64_t)srcRankId * numMaxTokensPerRank_ * perSlotBytes_ +
                               (int64_t)slotIdx * perSlotBytes_;
                               PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

            GlobalTensor<XType> srcTokenTensor;
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            srcTokenTensor.SetGlobalBuffer(reinterpret_cast<__gm__ XType *>(slotAddr), axisH_);
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            LocalTensor<XType> tokenTensor = tokenQueue_.AllocTensor<XType>();
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            DataCopyPad(tokenTensor, srcTokenTensor, tokenCopyParams, tokenPadParams);
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

            if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                GlobalTensor<ScalesType> srcScalesTensor;
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                srcScalesTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ScalesType *>(slotAddr + scalesOffset_),
                                                scalesElems_);
                                                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                DataCopyParams scalesCopyParams{1U, static_cast<uint16_t>(scalesElems_ * sizeof(ScalesType)), 0U, 0U};
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                DataCopyPadParams scalesPadParams{false, 0, 0, 0};
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                DataCopyPad(tokenTensor[scalesOffset_ / sizeof(XType)].template ReinterpretCast<ScalesType>(),
                            srcScalesTensor, scalesCopyParams, scalesPadParams);
                            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            }

            if constexpr (HasTopkWeights) {
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                GlobalTensor<int32_t> srcMetaGm;
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                srcMetaGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(slotAddr + metaOffset_));
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                DataCopyExtParams slotMetaParams{1U, metaBytes_, 0U, 0U, 0U};
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                DataCopyPad(ubMeta_, srcMetaGm, slotMetaParams, metaPadParams);
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                SyncFunc<AscendC::HardEvent::MTE2_S>();
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                float weight = ubMeta_.ReinterpretCast<float>().GetValue(axisKAlign_ + srcTopkIdx);
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                ubStageWeights_.SetValue(0, weight);
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                SyncFunc<AscendC::HardEvent::S_MTE2>();
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            }

            tokenQueue_.EnQue(tokenTensor);
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            LocalTensor<XType> tokenOut = tokenQueue_.DeQue<XType>();
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            uint32_t recvXRow = static_cast<uint32_t>(recvXIdx);
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            DataCopyPad(recvXGm_[(int64_t)recvXRow * axisH_], tokenOut, tokenCopyParams);
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                DataCopyParams scalesCopyParams{1U, static_cast<uint16_t>(scalesElems_ * sizeof(ScalesType)), 0U, 0U};
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                DataCopyPad(recvScalesGm_[(int64_t)recvXRow * scalesElems_],
                            tokenOut[scalesOffset_ / sizeof(XType)].template ReinterpretCast<ScalesType>(),
                            scalesCopyParams);
                            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            }
            tokenQueue_.FreeTensor(tokenOut);
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

            if constexpr (HasTopkWeights) {
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                SyncFunc<AscendC::HardEvent::S_MTE3>();
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                DataCopyExtParams weightOutParams{1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                DataCopyPad(recvTopkWeightsGm_[recvXRow], ubStageWeights_, weightOutParams);
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                // Wait before reusing the single aligned staging block for the next metadata row.
                SyncFunc<AscendC::HardEvent::MTE3_S>();
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            }
        }
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        DataCopyExtParams metaOutParams{
            1U, static_cast<uint32_t>(tileCnt * RECV_META_FIELDS * sizeof(int32_t)), 0U, 0U, 0U};
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        DataCopyPad(recvSrcMetadataGm_[(int64_t)tileStartGlobal * RECV_META_FIELDS], ubStageMeta_, metaOutParams);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

        processed += tileCnt;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }
}
#endif

} // namespace MoeEpDispatchEpilogueImpl

#endif // MOE_EP_DISPATCH_EPILOGUE_H
