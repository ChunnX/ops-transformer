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
 * \file moe_ep_dispatch_epilogue_base.h
 * \brief dispatch epilogue 两条路径（非 cached / cached）共用的状态、初始化与工具函数。
 *
 * 本文件只放两条路径都需要的东西；路径专有的成员与流水放在
 * moe_ep_dispatch_epilogue_uncached.h 和 moe_ep_dispatch_epilogue_cached.h 里。
 * 版本校验与公共 include 也集中在这里，派生头文件直接包含本文件即可。
 */

#ifndef MOE_EP_DISPATCH_EPILOGUE_BASE_H
#define MOE_EP_DISPATCH_EPILOGUE_BASE_H

#include <cstddef>

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

#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#include "../../common/op_kernel/mc2_moe_context.h"
#include "../../common/op_kernel/moe_ep_exception_dump_writer.h"
#include "../../common/op_kernel/moe_ep_send_completion.h"

namespace MoeEpDispatchEpilogueImpl {

#if defined(ENABLE_MOE_EP_DISPATCH_EPILOGUE_KERNEL)

using namespace AscendC;

static constexpr uint32_t UB_ALIGN = 32U;
static constexpr uint32_t WIN_ADDR_ALIGN = 512;
static constexpr uint32_t NETWORK_HYBRID = 1U; // 1 = hybrid dispatch, 0 = direct
// recv_src_metadata is compact in GM: [srcRank, srcToken, srcTopk, srcSlot, recvXIdx].
// The valid prefix is ordered by (srcRank, recvXIdx); each rank range follows increasing send-source rows.
// UB staging still uses an aligned stride so that every non-aligned 20-byte copy starts from a 32-byte boundary.
static constexpr uint32_t RECV_META_FIELDS = 5;
static constexpr uint32_t ELEM_ALIGN = 8U;
static constexpr uint32_t META_TOPK_SECTION = 2U;
static constexpr uint32_t META_SRC_RANK_OFFSET = 0U;
static constexpr uint32_t META_TOKEN_IDX_OFFSET = 1U;
static constexpr uint32_t META_TOPK_IDX_OFFSET = 2U;
static constexpr uint32_t META_SLOT_IDX_OFFSET = 3U;
static constexpr uint32_t META_RECV_X_IDX_OFFSET = 4U;
static constexpr uint32_t ALIGNED_LEN_256 = 256U;

template <typename XType, typename ScalesType, bool HasTopkWeights>
class MoeEpDispatchEpilogueBase {
protected:
    __aicore__ inline MoeEpDispatchEpilogueBase(){};
    // 两条路径的 Init 前半段完全一致：读 tiling、绑公共 GM、开公共 UB。
    __aicore__ inline void InitCommon(GM_ADDR context, GM_ADDR x, GM_ADDR numRecvPerExpert, GM_ADDR recvX,
                                      GM_ADDR recvSrcMetadata, GM_ADDR recvTopkWeights, GM_ADDR recvScales,
                                      GM_ADDR workspace, GM_ADDR tilingGM, TPipe *pipe,
                                      const MoeEpDispatchEpilogueInfo *tilingData);
    // 后半段也一致，但必须排在路径专有的 InitBuffer 之后：这一步用了 sharedTmpBuf_/waitSumBuf_。
    __aicore__ inline void FinalizeInit();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startId, uint32_t &endId,
                                       uint32_t &sendNum);
    __aicore__ inline GM_ADDR GetWinAddrByRankId(__gm__ Mc2Aclnn::MoeCommContext *ctx, uint32_t rankId, uint64_t offset)
    {
        return (GM_ADDR)ctx->epHcclBuffer[rankId] + offset;
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
    __gm__ Mc2Aclnn::MoeCommContext *mc2Context_{nullptr};
    MoeEpExceptionDump::MoeEpCoreDiagWriter diagWriter_;
    uint32_t epRankId_{0};
    uint32_t networkMode_{0}; // 0 = direct(自段 x 直读输入), 1 = hybrid(自段 x 读win)
    uint32_t aivId_{0};
    GlobalTensor<int64_t> numRecvPerExpertGm_;

    GlobalTensor<XType> xGm_; // 本端源 x（direct: 从输入直接获取）
    GlobalTensor<XType> recvXGm_;
    GlobalTensor<float> recvTopkWeightsGm_;
    GlobalTensor<int32_t> recvSrcMetadataGm_;
    GlobalTensor<int32_t> recvRankOffsetsGm_;
    GlobalTensor<ScalesType> recvScalesGm_;

    LocalTensor<int32_t> ubRankOffsets_;
    LocalTensor<int64_t> ubExpertPfx_;
    LocalTensor<int32_t> ubWaitStatus_;
    LocalTensor<int32_t> ubWaitSum_;

    TBuf<QuePosition::VECIN> ubRankOffsetsBuf_;
    TBuf<QuePosition::VECIN> ubExpertPfxBuf_;
    TBuf<> waitStatusBuf_;
    TBuf<> waitSumBuf_;
    TBuf<> sharedTmpBuf_;

    uint32_t expertSum_{0};
    GM_ADDR workspaceGM_{nullptr};
    GM_ADDR localWinAddr_{nullptr};
    GM_ADDR localSlotStateWinAddr_{nullptr};
    uint32_t scalesOffset_{0};
    uint32_t scalesStride_{0}; // token UB 槽内 scales 暂存偏移(32B 对齐, 与 GM slot 紧拼偏移解耦)
    uint32_t scalesBytes_{0};
    uint32_t scalesBytesAlign_{0};
    uint32_t scalesElems_{0};
    uint32_t metaOffset_{0};
    uint32_t tokenQueueBufBytes_{0};
    uint32_t metaBytes_{0};
    uint32_t axisKAlign_{0};
    // hybrid 槽布局下 token 段的对齐字节数，也是 scales 暂存的起点；派生类算 token buffer 大小时要用。
    uint32_t hAlignSize_{0};
    uint32_t numLocalExperts_{0};
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
};

template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueBase<XType, ScalesType, HasTopkWeights>::InitCommon(
    GM_ADDR context, GM_ADDR x, GM_ADDR numRecvPerExpert, GM_ADDR recvX, GM_ADDR recvSrcMetadata,
    GM_ADDR recvTopkWeights, GM_ADDR recvScales, GM_ADDR workspace, GM_ADDR tilingGM, TPipe *pipe,
    const MoeEpDispatchEpilogueInfo *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    workspaceGM_ = workspace;
    numLocalExperts_ = tilingData->cfg.numLocalExperts;
    aivNum_ = tilingData->aivNum;
    networkMode_ = tilingData->networkMode;
    axisK_ = tilingData->cfg.topK;
    axisH_ = tilingData->cfg.hidden;
    epWorldSize_ = tilingData->cfg.epWorldSize;
    numMaxTokensPerRank_ = tilingData->cfg.numMaxTokensPerRank;
    perSlotBytes_ = tilingData->cfg.perSlotBytes;
    dispatchNotifyCount_ = tilingData->dispatchNotifyCount;
    winDataOffset_ = tilingData->winDataOffset;
    slotWinStateOffset_ = tilingData->slotWinStateOffset;

    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::MoeCommContext *>(context);
    epRankId_ = mc2Context_->epRankId;
    constexpr size_t metadataOffset = offsetof(MoeEpDispatchEpilogueTilingData, moeEpDispatchEpilogueInfo) +
                                      offsetof(MoeEpDispatchEpilogueInfo, dumpMetadata);
    MoeEpExceptionDump::WriteMetadata(context, tilingGM + metadataOffset);
    diagWriter_.Init(context, MOE_EP_CORE_DIAG_DISPATCH_EPILOGUE, tpipe_);
    localSlotStateWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, slotWinStateOffset_);
    localWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, winDataOffset_);
    // 槽内 scales/meta 基址: direct 紧拼 tokenSize; hybrid 为 ALIGN32(tokenSize)（hybrid dispatch 整槽布局）
    hAlignSize_ = Ceil((uint32_t)(axisH_ * sizeof(XType)), UB_ALIGN) * UB_ALIGN;
    uint32_t slotMetaBase = (tilingData->networkMode == NETWORK_HYBRID) ? hAlignSize_ : axisH_ * sizeof(XType);
    metaOffset_ = slotMetaBase;
    scalesStride_ = hAlignSize_ / sizeof(XType);

    numRecvPerExpertGm_.SetGlobalBuffer((__gm__ int64_t *)numRecvPerExpert);
    xGm_.SetGlobalBuffer((__gm__ XType *)x);
    recvXGm_.SetGlobalBuffer((__gm__ XType *)recvX);
    recvSrcMetadataGm_.SetGlobalBuffer((__gm__ int32_t *)recvSrcMetadata);
    recvRankOffsetsGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(recvSrcMetadata + tilingData->metadataRankOffsetsOffset));
    if constexpr (HasTopkWeights) {
        recvTopkWeightsGm_.SetGlobalBuffer((__gm__ float *)recvTopkWeights);
    }

    axisKAlign_ = Ceil(axisK_, ELEM_ALIGN) * ELEM_ALIGN;
    metaBytes_ = (META_TOPK_SECTION * axisKAlign_) * (uint32_t)sizeof(int32_t) + UB_ALIGN;
    totalNotifyCnt_ = epWorldSize_ * dispatchNotifyCount_;
    uint32_t ubExpertPfxBytes = Ceil((uint32_t)(numLocalExperts_ * sizeof(int64_t)), UB_ALIGN) * UB_ALIGN;
    uint32_t expertReduceTmpBytes =
        ReduceSumWorkNeedSize(static_cast<int32_t>(numLocalExperts_), sizeof(int64_t)) * sizeof(int64_t);
    uint32_t statusReduceTmpBytes =
        ReduceSumWorkNeedSize(static_cast<int32_t>(totalNotifyCnt_), sizeof(float)) * sizeof(float);
    uint32_t sharedBytes = expertReduceTmpBytes > statusReduceTmpBytes ? expertReduceTmpBytes : statusReduceTmpBytes;
    tpipe_->InitBuffer(ubExpertPfxBuf_, ubExpertPfxBytes);
    tpipe_->InitBuffer(waitStatusBuf_, totalNotifyCnt_ * UB_ALIGN);
    tpipe_->InitBuffer(waitSumBuf_, UB_ALIGN);
    tpipe_->InitBuffer(sharedTmpBuf_, sharedBytes);
    ubExpertPfx_ = ubExpertPfxBuf_.Get<int64_t>();
    ubWaitStatus_ = waitStatusBuf_.Get<int32_t>();
    ubWaitSum_ = waitSumBuf_.Get<int32_t>();
    uint32_t ubRankOffsetsBytes = Ceil((epWorldSize_ + 1U) * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(ubRankOffsetsBuf_, ubRankOffsetsBytes);
    ubRankOffsets_ = ubRankOffsetsBuf_.Get<int32_t>();

    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
        scalesOffset_ = slotMetaBase;
        scalesBytes_ = tilingData->cfg.scalesBytes;
        scalesElems_ = (scalesBytes_ == 0U) ? 0U : (scalesBytes_ / sizeof(ScalesType));
        scalesBytesAlign_ = Ceil(scalesBytes_, UB_ALIGN) * UB_ALIGN;
        metaOffset_ += scalesBytesAlign_;
        recvScalesGm_.SetGlobalBuffer((__gm__ ScalesType *)recvScales);
    }
}

template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueBase<XType, ScalesType, HasTopkWeights>::FinalizeInit()
{
    DataCopyExtParams expertPfxCopyParams{1U, static_cast<uint32_t>(numLocalExperts_ * sizeof(int64_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int64_t> expertPfxPadParams{false, 0U, 0U, 0};
    DataCopyPad(ubExpertPfx_, numRecvPerExpertGm_, expertPfxCopyParams, expertPfxPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();

    LocalTensor<int64_t> expertReduceTmp = sharedTmpBuf_.Get<int64_t>();
    LocalTensor<int64_t> expertSumOut = waitSumBuf_.Get<int64_t>();
    ReduceSum<int64_t>(expertSumOut, ubExpertPfx_, expertReduceTmp, static_cast<int32_t>(numLocalExperts_));
    SyncFunc<AscendC::HardEvent::V_S>();
    expertSum_ = static_cast<uint32_t>(expertSumOut.GetValue(0));
    diagWriter_.RunPosRecord(MOE_EP_DISPATCH_EPILOGUE_RUN_POS_INIT_DONE);
}

template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueBase<XType, ScalesType, HasTopkWeights>::WaitDispatch()
{
    if (aivId_ != aivNum_ - 1) {
        return;
    }

    uint32_t mask = 1;
    int32_t sumOfFlag = 0;
    int32_t commpareFlag = static_cast<int32_t>(totalNotifyCnt_);
    GlobalTensor<int32_t> statusGMTensor;
    LocalTensor<float> sharedTmp = sharedTmpBuf_.Get<float>();
    LocalTensor<float> ubWaitStatusFp32 = ubWaitStatus_.template ReinterpretCast<float>();
    LocalTensor<float> ubWaitSumFp32 = ubWaitSum_.template ReinterpretCast<float>();
    statusGMTensor.SetGlobalBuffer((__gm__ int32_t *)localSlotStateWinAddr_);
    DataCopyParams statusCopyParams = {static_cast<uint16_t>(totalNotifyCnt_), 1U,
                                       static_cast<uint16_t>((WIN_ADDR_ALIGN - UB_ALIGN) / UB_ALIGN), 0U};
    DataCopyParams clearStatusCopyParams = {static_cast<uint16_t>(totalNotifyCnt_), 1U, 0U,
                                            static_cast<uint16_t>((WIN_ADDR_ALIGN - UB_ALIGN) / UB_ALIGN)};

    SyncFunc<AscendC::HardEvent::S_V>(); // 确保expertSum_计算完成
    // 2.3us
    while (sumOfFlag != commpareFlag) {
        DataCopy(ubWaitStatus_, statusGMTensor, statusCopyParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(ubWaitSumFp32, ubWaitStatusFp32, sharedTmp, mask, totalNotifyCnt_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = ubWaitSum_.GetValue(0);
    }
    Duplicate<int32_t>(ubWaitStatus_, 0, totalNotifyCnt_ * UB_ALIGN / sizeof(int32_t));
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(statusGMTensor, ubWaitStatus_, clearStatusCopyParams);
    // Dispatch reuses these notification slots in the next continue round.  SyncAll only synchronizes AIVs; it does
    // not replace the MTE3 completion required before a later kernel can publish the next round's notifications.
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueBase<XType, ScalesType, HasTopkWeights>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startId, uint32_t &endId, uint32_t &sendNum)
{
    sendNum = curSendCnt / curUseAivNum;
    uint32_t remainderNum = curSendCnt % curUseAivNum;
    uint32_t newAivId = aivId_;
    startId = sendNum * newAivId;
    if (newAivId < remainderNum) {
        sendNum += 1;
        startId += newAivId;
    } else {
        startId += remainderNum;
    }
    endId = startId + sendNum;
}

#endif

} // namespace MoeEpDispatchEpilogueImpl

#endif // MOE_EP_DISPATCH_EPILOGUE_BASE_H
