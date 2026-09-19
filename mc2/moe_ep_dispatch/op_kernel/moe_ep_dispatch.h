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
 * \file moe_ep_dispatch.h
 * \brief
 */

#ifndef MOE_EP_DISPATCH_H
#define MOE_EP_DISPATCH_H

#include <cstddef>

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if (ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)) && \
    (HCOMM_MAJOR > 9 || (HCOMM_MAJOR == 9 && HCOMM_MINOR > 0))
#define ENABLE_MOE_EP_KERNEL
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
#include "moe_ep_dispatch_tiling.h"
#include "moe_ep_dispatch_base.h"
#include "moe_ep_dispatch_schedule.h"

#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#include "../../common/op_kernel/mc2_moe_context.h"
#include "../../common/op_kernel/moe_shared_jetty_context.h"
#include "../../common/op_kernel/moe_shared_jetty_kernel_utils.h"
#include "../../common/op_kernel/moe_ep_exception_dump_writer.h"

namespace MoeEpDispatchImpl {

#define TemplateMoeEpDispatchTypeClass \
    typename XType, typename ScalesType, bool DoCpuSync, bool IsCached, bool IsTopkWeights
#define TemplateMoeEpDispatchTypeFunc XType, ScalesType, DoCpuSync, IsCached, IsTopkWeights

#if defined(ENABLE_MOE_EP_KERNEL)

using namespace AscendC;
using namespace Mc2Kernel;

constexpr uint32_t UB_ALIGN = 32U;
constexpr uint32_t WIN_ADDR_ALIGN = 512U;
constexpr uint32_t ALIGNED_LEN_256 = 256U;
constexpr uint32_t TOPK_INFO_SIZE = 4U;  // sizeof(int32_t)=sizeof(float)=4B
constexpr uint32_t UB_STRIDE = 8U;       // UB_ALIGN/sizeof(int32_t)=8
constexpr uint32_t INT64_UB_STRIDE = 4U; // UB_ALIGN/sizeof(int64_t)=4
constexpr uint32_t BITS_PER_BYTE = 8U;
constexpr uint32_t STATE_STRIDE = 15U; // (WIN_ADDR_ALIGN-UB_ALIGN)/UB_ALIGN=15
constexpr uint32_t HCOMM_INIT_SIZE = 512U;
constexpr uint32_t PER_GROUP_SIZE = 32 * 1024U; // 计算count 32KB per group
constexpr uint32_t HISTOGRAM_SCOPE_SIZE = 256U; // 直方图uint8, 统计范围0-255
constexpr uint8_t BUFFER_NUM = 2;
constexpr uint32_t NOTIFY_VAL_SHIFT = 32U;                        // notify 高32位为 sendCnt, 低32位为 state(=1)
constexpr uint32_t MAX_BATCH_SIZE = 160 * 1024U;                  // srcTokenIdx 160KB
constexpr uint32_t SGE_PER_SLOT = 2U;                             // 每 slot 两个 SGE: x 与 meta
constexpr uint32_t SGE_PER_SQE = 12U;                             // 单 SQE 的 SGE 数
constexpr uint32_t SLOT_NUM_PER_SQE = SGE_PER_SQE / SGE_PER_SLOT; // 单 SQE 的 slot 数
constexpr uint32_t HCOMM_BATCH_CAPACITY = 128U;                   // UB capacity in full 12-SGE requests
// Start DMA after eight full 12-SGE requests; subsequent batches use the existing UB capacity.
constexpr uint32_t HCOMM_FIRST_BATCH_WQEBB = 32U;
constexpr uint32_t HCOMM_SGE_BYTES = 16U;
constexpr uint32_t HCOMM_SQE_BYTES = 48U;
constexpr uint32_t HCOMM_WQE_BYTES = 64U;
constexpr struct UrmaWqeEntry DATA_CFG = {.odr = 5, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
constexpr struct UrmaWqeEntry NOTIFY_CFG = {.odr = 6, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};

__aicore__ inline void CheckSharedJettyResult(int32_t ret)
{
    if (ret != 0) {
        AscendC::printf("MoE shared Jetty batch failed, ret=%d\n", ret);
        AscendC::Trap();
    }
}

template <TemplateMoeEpDispatchTypeClass>
class MoeEpDispatch {
public:
    __aicore__ inline MoeEpDispatch(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights, GM_ADDR scales,
                                GM_ADDR cachedSlotIdx, GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert,
                                GM_ADDR dstBufferSlotIdx, GM_ADDR workspaceGM, GM_ADDR tilingGM, TPipe *pipe,
                                const MoeEpDispatchTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitTilingAndRoles(const MoeEpDispatchInfo &info);
    __aicore__ inline void InitDerivedValues();
    __aicore__ inline void BindInputTensors(GM_ADDR topkIdx, GM_ADDR topkWeights, GM_ADDR scales,
                                            GM_ADDR cachedSlotIdx, GM_ADDR numRecvPerRank,
                                            GM_ADDR numRecvPerExpert, GM_ADDR dstBufferSlotIdx);
    __aicore__ inline void InitCopyParams();
    __aicore__ inline void BindWorkspaceAndWindow(GM_ADDR workspaceGM);
    __aicore__ inline void BufferInit();
    __aicore__ inline void ResetCounters();
    __aicore__ inline void CalSendCntHistograms(LocalTensor<int16_t> dstTensor, LocalTensor<int16_t> &tempTensor,
                                                LocalTensor<uint32_t> sendCntTensor, uint32_t calCnt,
                                                uint32_t scopeSize);
    __aicore__ inline void DedupAndSend(LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt,
                                       uint32_t calCntAlign, uint32_t tmpOffset);
    __aicore__ inline void CalSendCntPerRank(uint32_t calCnt, uint32_t calCntAlign, uint32_t tmpOffset);
    __aicore__ inline void CalSendCntPerExpert(LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt,
                                               uint32_t calCntAlign, uint32_t tmpOffset);
    __aicore__ inline void CalSendCntGroup(uint32_t group, uint32_t groupCnt, uint32_t &calCnt,
                                           LocalTensor<int32_t> topkIdsGroupTensor,
                                           LocalTensor<int16_t> tempTensorInt16,
                                           const DataCopyPadExtParams<int32_t> &copyPadParams);
    __aicore__ inline void PublishSendCounts();
    __aicore__ inline void CalSendCnt();
    __aicore__ inline void Communication();
    __aicore__ inline void SendCountToPeer(uint32_t peer, GM_ADDR dst, GM_ADDR src,
                                           GM_ADDR notifyAddr, uint64_t notifyVal);
    __aicore__ inline void GetRecvCount();
    __aicore__ inline void SetRecvNumPerExpert();
    __aicore__ inline void SetRecvNumPerRank(LocalTensor<int32_t> recvTmpTensor);
    __aicore__ inline void WriteToRemoteWindow();
    __aicore__ inline void GetTokenRange(uint32_t dstRank, uint32_t &tokenStart, uint32_t &tokenNum);
    __aicore__ inline void CommBufferInit();
    __aicore__ inline void SendTokenChunkToPeer(uint32_t dstRankId, GM_ADDR notifyAddr);
    __aicore__ inline void SendTokenData(uint32_t tokenStart, uint32_t tokenNum, uint32_t dstRankId,
                                         GM_ADDR remoteWinAddr, BufDesc *localDescs);
    __aicore__ inline void WritePayload(GM_ADDR dst, BufDesc *localDescs, uint32_t sgeCount);
    __aicore__ inline void SendCompletion(uint32_t dstRankId, GM_ADDR notifyAddr);
    __aicore__ inline void ReserveBatchSpace(uint32_t wqebbCount);
    __aicore__ inline void CommitBatch(bool force);
    __aicore__ inline uint32_t GetTokenBoundary(uint32_t dstRank, uint32_t tokenBoundary);
    __aicore__ inline uint32_t LowerBoundToken(uint32_t dstRank, uint32_t begin, uint32_t end,
                                               uint32_t tokenBoundary);
    __aicore__ inline void SetSrcTokenIdxAndDstSlot(uint32_t topkOffset, uint32_t calCnt, uint32_t calCntAlign,
                                                    uint32_t &sendLocalCnt, LocalTensor<int32_t> &cntLocalTensor);
    __aicore__ inline void ProcessMetaSlot(uint32_t startId, uint32_t tokenCnt, uint32_t sendLocalCnt,
                                           uint32_t &topkOffset, uint32_t &curLocalSlot);
    __aicore__ inline void SendPhase();
    __aicore__ inline void GetSlotStartNum();
    __aicore__ inline void CopyInMetaInfo(uint32_t tokenId, uint32_t topkOffset);
    __aicore__ inline void PublishPrepareReady();
    __aicore__ inline void WaitPrepareReady();
    __aicore__ inline void GetOverlappingCores(uint32_t begin, uint32_t end, uint32_t coreCount,
                                              uint32_t &first, uint32_t &last) const;
    __aicore__ inline void SplitToCore(uint32_t itemCount, uint32_t coreCount, uint32_t coreIndex,
                                       uint32_t &startId, uint32_t &endId, uint32_t &itemCountPerCore);

    __aicore__ inline uint64_t GetOwnedJettyHandle()
    {
        return mc2Context_->multiChannelHandles[aivId_];
    }

    TPipe *tpipe_{nullptr};
    __gm__ Mc2Aclnn::MoeSharedJettyContext *mc2Context_{nullptr};
    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_; // 通信上下文
    AscendC::BatchHandle<AscendC::MultiChannelHandle> payloadBatch_{};
    MoeEpExceptionDump::MoeEpCoreDiagWriter diagWriter_;

    GlobalTensor<int32_t> topkIdxGMTensor_;
    GlobalTensor<float> topkWeightsGMTensor_;
    GlobalTensor<int32_t> dstSlotIdxGMTensor_;
    GlobalTensor<int32_t> numRecvPerRankGMTensor_;
    GlobalTensor<int64_t> numRecvPerExpertGMTensor_;
    GlobalTensor<int32_t> cachedSlotIdxGMTensor_;
    GlobalTensor<int32_t> sendCounterGMTensor_;
    GlobalTensor<int32_t> recvCounterGMTensor_;
    GlobalTensor<int32_t> sendCntPerRankGMTensor_;
    GlobalTensor<int32_t> sendCntPerExpertGMTensor_;
    GlobalTensor<int32_t> srcTokenIdxGMTensor_;
    GlobalTensor<int16_t> dstRankGMTensor_;
    GlobalTensor<int32_t> metaSlotGMTensor_;
    GlobalTensor<int32_t> localMetaSlotGMTensor_;
    GlobalTensor<ScalesType> scalesGMTensor_;

    LocalTensor<int32_t> dstSlotIdxTensor_;
    LocalTensor<int32_t> metaLocalTensor_;
    LocalTensor<int32_t> numRecvPerRankTensor_;
    LocalTensor<int64_t> numRecvPerExpertTensor_;
    LocalTensor<int32_t> sendCntPerExpertTensor_;
    LocalTensor<int32_t> sendCntPerRankTensor_;
    LocalTensor<int32_t> slotIdxPerRankTensor_;
    LocalTensor<int16_t> dstRankTensor_;
    LocalTensor<int16_t> dstRankIdTensor_;
    LocalTensor<uint8_t> hcommTensor_;
    LocalTensor<uint8_t> hcommBatchTensor_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> metaSlotQueue_;
    TBuf<> topkIdsBuf_;
    TBuf<> tempBuf_;
    TBuf<> dstExpBuf_;
    TBuf<> maskBuf_;
    TBuf<> hcommBuf_; // 通信
    TBuf<> sendCntRankBuf_;
    TBuf<> sendCntExpertBuf_;
    TBuf<> numRecvPerRankBuf_;
    TBuf<> numRecvPerExpertBuf_;

    GM_ADDR workspaceGM_{nullptr};
    GM_ADDR hostPinnedCounterAddrGM_{nullptr};
    GM_ADDR xAddr_{nullptr};
    GM_ADDR sendCounterAddr_{nullptr};
    GM_ADDR sendCntWorkspaceAddr_{nullptr};
    GM_ADDR dstRankInfoAddr_{nullptr};
    GM_ADDR srcTokenTableAddr_{nullptr};
    GM_ADDR localCntStateWinAddr_{nullptr};
    GM_ADDR localSlotStateWinAddr_{nullptr};
    GM_ADDR localSlotWinAddr_{nullptr};
    GM_ADDR payloadStashWinAddr_{nullptr};
    GM_ADDR payloadStashStateWinAddr_{nullptr};
    GM_ADDR completionFlagSourceWinAddr_{nullptr};

    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t epWorldSize_{0};
    uint32_t moeExpertNumPerRank_{0};
    uint32_t axisMaxBS_{0};
    uint32_t scalesBytes_{0};
    uint32_t perSlotBytes_{0};
    uint32_t moeExpertNum_{0};
    uint32_t aivNum_{0};
    uint32_t epRankId_{0};
    uint32_t aivId_{0};
    uint32_t startTokenId_{0};
    uint32_t endTokenId_{0};
    uint32_t sendTokenNum_{0};
    uint32_t perGroupTokenNum_{0};
    uint32_t sendCoreCount_{0};
    uint32_t prepareCoreCount_{0};
    uint32_t prepareCoreStart_{0};
    uint32_t prepareCoreId_{0};
    uint32_t hcommBatchBufferBytes_{0};
    uint32_t pendingWqebb_{0};
    uint32_t activePeerIndex_{0};
    bool firstBatchCommitted_{false};
    uint32_t tokenSize_{0};
    uint32_t kAlignSize_{0};
    uint32_t axisKAlign_{0};
    uint32_t epWorldSizeAlign_{0};
    uint32_t epWorldSizeAlign512_{0};
    uint32_t counterCnt_{0};
    uint32_t counterAlign512_{0};
    uint32_t perGroupSizeAlign_{0};
    uint32_t maskBytesAlign_{0};
    uint32_t sendCntRankSizeAlign_{0};
    uint32_t moeNumPerRankSize_{0};
    uint32_t moeNumPerRankAlign_{0};
    uint32_t expertPerRankAlign_{0};
    uint32_t moeExpertNumAlign_{0};
    uint32_t moeNumPerRankAlign512_{0};
    uint32_t moeExpertNumAlign512_{0};
    uint32_t cntFlagWinSize_{0};
    uint32_t dispatchNotifyCount_{1};
    uint32_t metaSlotBytes_{0};
    uint32_t topkCopyAlign_{0};
    uint32_t srcTokenListBytes_{0};
    uint32_t metaSlotStride_{0};
    uint32_t tokenCntAlign_{0};
    uint64_t dstRankInfoOffset_{0};
    uint64_t srcTokenTableOffset_{0};
    uint64_t cntWinStateOffset_{0};
    uint64_t slotWinStateOffset_{0};
    uint64_t payloadWinStateOffset_{0};
    uint64_t winDataOffset_{0};
    uint64_t payloadStashWinOffset_{0};
    uint64_t completionFlagSourceWinOffset_{0};
    uint64_t dstRankWinOffset_{0};
    uint64_t dstStateWinOffset_{0};
    bool isSendCore_{false};
    bool isPrepareCore_{false};

    DataCopyParams statusCopyParams_;
    DataCopyParams clearStatusCopyParams_;
    DataCopyParams sendPerExpertParams_;
    DataCopyParams sendPerRankParams_;
    DataCopyParams metaSlotCopyParams_; // stash 元数据槽整槽拷贝参数
    DataCopyParams topkCopyParams_;
    DataCopyParams scalesCopyParams_;
    DataCopyPadParams padParams_;
};

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::InitTilingAndRoles(
    const MoeEpDispatchInfo &info)
{
    metaSlotBytes_ = info.metaSlotBytes;
    axisBS_ = info.cfg.numTokens;
    axisH_ = info.cfg.hidden;
    axisK_ = info.cfg.topK;
    epWorldSize_ = info.cfg.epWorldSize;
    if (mc2Context_->epRankSize != epWorldSize_ || epRankId_ >= epWorldSize_ || epRankId_ != info.cfg.epRankId) {
        AscendC::printf("MoE shared Jetty rank mismatch, context=%u/%u, tiling world=%u\n", epRankId_,
                        mc2Context_->epRankSize, epWorldSize_);
        AscendC::Trap();
    }
    moeExpertNum_ = info.cfg.numExperts;
    moeExpertNumPerRank_ = info.cfg.numLocalExperts;
    axisMaxBS_ = info.cfg.numMaxTokensPerRank;
    scalesBytes_ = info.scalesBytes;
    perSlotBytes_ = info.perSlotBytes;
    aivNum_ = info.aivNum;
    sendCoreCount_ = Mc2Aclnn::MOE_SHARED_JETTY_NUM;
    prepareCoreStart_ = sendCoreCount_;
    prepareCoreCount_ = aivNum_ > prepareCoreStart_ ? aivNum_ - prepareCoreStart_ : 0U;
    isSendCore_ = aivId_ < sendCoreCount_;
    isPrepareCore_ = aivId_ >= prepareCoreStart_ && aivId_ < prepareCoreStart_ + prepareCoreCount_;
    prepareCoreId_ = isPrepareCore_ ? aivId_ - prepareCoreStart_ : 0U;
    srcTokenListBytes_ = info.workspace.srcTokenListBytes;
    dstRankInfoOffset_ = info.workspace.dstRankInfoOffset;
    srcTokenTableOffset_ = info.workspace.srcTokenTableOffset;
    cntWinStateOffset_ = info.window.cntWinStateOffset;
    slotWinStateOffset_ = info.window.slotWinStateOffset;
    payloadWinStateOffset_ = info.window.payloadWinStateOffset;
    winDataOffset_ = info.window.winDataOffset;
    payloadStashWinOffset_ = info.window.payloadStashWinOffset;
    completionFlagSourceWinOffset_ = info.window.completionFlagSourceWinOffset;
    dispatchNotifyCount_ = info.dispatchNotifyCount;
    if (dispatchNotifyCount_ != Mc2Aclnn::MOE_SHARED_JETTY_NUM) {
        AscendC::printf("MoE dispatch notify count mismatch, got=%u\n", dispatchNotifyCount_);
        AscendC::Trap();
    }
    hostPinnedCounterAddrGM_ = reinterpret_cast<GM_ADDR>(info.hostPinnedCounterAddr);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::InitDerivedValues()
{
    tokenSize_ = axisH_ * sizeof(XType);
    kAlignSize_ = Ceil(axisK_ * TOPK_INFO_SIZE, UB_ALIGN) * UB_ALIGN;
    axisKAlign_ = kAlignSize_ / TOPK_INFO_SIZE;
    metaSlotStride_ = metaSlotBytes_ / sizeof(int32_t);
    epWorldSizeAlign_ = Ceil(epWorldSize_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    perGroupTokenNum_ = PER_GROUP_SIZE / sizeof(int16_t) / axisK_;
    perGroupSizeAlign_ = Ceil(perGroupTokenNum_ * axisK_ * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    maskBytesAlign_ = Ceil(Ceil(perGroupSizeAlign_ / sizeof(int16_t), BITS_PER_BYTE), UB_ALIGN) * UB_ALIGN;
    epWorldSizeAlign512_ = Ceil(epWorldSize_ * sizeof(int32_t), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    sendCntRankSizeAlign_ = Ceil(epWorldSize_ * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    counterCnt_ = epWorldSizeAlign_ / sizeof(int32_t);
    counterAlign512_ = epWorldSizeAlign512_ / sizeof(int32_t);
    moeNumPerRankSize_ = moeExpertNumPerRank_ * sizeof(int32_t);
    moeNumPerRankAlign_ = Ceil(moeNumPerRankSize_, UB_ALIGN) * UB_ALIGN;
    expertPerRankAlign_ = moeNumPerRankAlign_ / sizeof(int32_t);
    moeExpertNumAlign_ = Ceil(moeExpertNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    moeNumPerRankAlign512_ = Ceil(moeNumPerRankSize_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    moeExpertNumAlign512_ = Ceil(moeExpertNum_ * sizeof(int32_t), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    cntFlagWinSize_ = epWorldSize_ * WIN_ADDR_ALIGN;
    dstRankWinOffset_ = static_cast<uint64_t>(epRankId_) * axisMaxBS_ * perSlotBytes_;
    tokenCntAlign_ = srcTokenListBytes_ / sizeof(int32_t);
    dstStateWinOffset_ = static_cast<uint64_t>(epRankId_) * dispatchNotifyCount_ * WIN_ADDR_ALIGN;
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::BindInputTensors(
    GM_ADDR topkIdx, GM_ADDR topkWeights, GM_ADDR scales, GM_ADDR cachedSlotIdx, GM_ADDR numRecvPerRank,
    GM_ADDR numRecvPerExpert, GM_ADDR dstBufferSlotIdx)
{
    topkIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)topkIdx);
    if constexpr (IsTopkWeights) {
        topkWeightsGMTensor_.SetGlobalBuffer((__gm__ float *)topkWeights);
    }
    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
        scalesGMTensor_.SetGlobalBuffer((__gm__ ScalesType *)scales);
        topkCopyAlign_ = Ceil(scalesBytes_, UB_ALIGN) * UB_ALIGN / sizeof(int32_t);
    }
    if constexpr (IsCached) {
        cachedSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)cachedSlotIdx);
    }
    numRecvPerRankGMTensor_.SetGlobalBuffer((__gm__ int32_t *)numRecvPerRank);
    numRecvPerExpertGMTensor_.SetGlobalBuffer((__gm__ int64_t *)numRecvPerExpert);
    dstSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)dstBufferSlotIdx);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::InitCopyParams()
{
    statusCopyParams_ = {static_cast<uint16_t>(epWorldSize_), 1U, static_cast<uint16_t>(STATE_STRIDE), 0U};
    clearStatusCopyParams_ = {static_cast<uint16_t>(epWorldSize_), 1U, 0U, static_cast<uint16_t>(STATE_STRIDE)};
    sendPerExpertParams_ = {1U, static_cast<uint16_t>(moeExpertNum_ * sizeof(int32_t)), 0U, 0U};
    sendPerRankParams_ = {1U, static_cast<uint16_t>(epWorldSize_ * sizeof(int32_t)), 0U, 0U};
    metaSlotCopyParams_ = {1U, static_cast<uint16_t>(metaSlotBytes_), 0U, 0U};
    topkCopyParams_ = {1U, static_cast<uint16_t>(axisK_ * TOPK_INFO_SIZE), 0U, 0U};
    scalesCopyParams_ = {1U, static_cast<uint16_t>(scalesBytes_), 0U, 0U};
    padParams_ = {true, 0, 0, 0};
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::BindWorkspaceAndWindow(GM_ADDR workspaceGM)
{
    sendCounterAddr_ = workspaceGM;
    sendCntWorkspaceAddr_ = workspaceGM + static_cast<uint64_t>(aivNum_) * epWorldSizeAlign512_;
    dstRankInfoAddr_ = workspaceGM + dstRankInfoOffset_;
    srcTokenTableAddr_ = workspaceGM + srcTokenTableOffset_;
    localCntStateWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, cntWinStateOffset_);
    localSlotStateWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, slotWinStateOffset_);
    payloadStashStateWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, payloadWinStateOffset_);
    localSlotWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, winDataOffset_) + dstRankWinOffset_;
    payloadStashWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, payloadStashWinOffset_);
    completionFlagSourceWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, completionFlagSourceWinOffset_);
    sendCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t *)sendCounterAddr_);
    recvCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t *)localCntStateWinAddr_);
    metaSlotGMTensor_.SetGlobalBuffer((__gm__ int32_t *)payloadStashWinAddr_);
    sendCntPerRankGMTensor_.SetGlobalBuffer((__gm__ int32_t *)sendCntWorkspaceAddr_);
    sendCntPerExpertGMTensor_.SetGlobalBuffer((__gm__ int32_t *)(sendCntWorkspaceAddr_ + epWorldSizeAlign512_));
    dstRankGMTensor_.SetGlobalBuffer((__gm__ int16_t *)dstRankInfoAddr_);
    srcTokenIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)srcTokenTableAddr_);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights, GM_ADDR scales, GM_ADDR cachedSlotIdx,
    GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert, GM_ADDR dstBufferSlotIdx, GM_ADDR workspaceGM, GM_ADDR tilingGM,
    TPipe *pipe, const MoeEpDispatchTilingData *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    workspaceGM_ = workspaceGM;
    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::MoeSharedJettyContext *>(context);
    if (mc2Context_->sharedJettyNum != Mc2Aclnn::MOE_SHARED_JETTY_NUM) {
        AscendC::printf("MoE shared Jetty count mismatch, got=%u\n", mc2Context_->sharedJettyNum);
        AscendC::Trap();
    }
    epRankId_ = mc2Context_->epRankId;
    constexpr size_t metadataOffset =
        offsetof(MoeEpDispatchTilingData, moeEpDispatchInfo) + offsetof(MoeEpDispatchInfo, dumpMetadata);
    MoeEpExceptionDump::WriteMetadata<Mc2Aclnn::MoeSharedJettyContext>(context, tilingGM + metadataOffset);
    diagWriter_.Init<Mc2Aclnn::MoeSharedJettyContext>(context, MOE_EP_CORE_DIAG_DISPATCH, tpipe_);

    const auto &info = tilingData->moeEpDispatchInfo;
    xAddr_ = x;
    InitTilingAndRoles(info);
    InitDerivedValues();
    BindInputTensors(topkIdx, topkWeights, scales, cachedSlotIdx, numRecvPerRank, numRecvPerExpert,
                     dstBufferSlotIdx);
    InitCopyParams();
    BindWorkspaceAndWindow(workspaceGM);

    diagWriter_.RunPosRecord(MOE_EP_DISPATCH_RUN_POS_INIT_DONE);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SplitToCore(
    uint32_t itemCount, uint32_t coreCount, uint32_t coreIndex, uint32_t &startId, uint32_t &endId,
    uint32_t &itemCountPerCore)
{
    auto range = MoeEpDispatchSchedule::Partition(itemCount, coreCount, coreIndex);
    startId = range.begin;
    endId = range.end;
    itemCountPerCore = endId - startId;
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::BufferInit()
{
    tpipe_->InitBuffer(metaSlotQueue_, BUFFER_NUM, metaSlotBytes_);
    tpipe_->InitBuffer(topkIdsBuf_, 2 * perGroupSizeAlign_);
    tpipe_->InitBuffer(tempBuf_, perGroupSizeAlign_);
    tpipe_->InitBuffer(dstExpBuf_, 2 * perGroupSizeAlign_);
    tpipe_->InitBuffer(maskBuf_, 3 * maskBytesAlign_);
    tpipe_->InitBuffer(sendCntRankBuf_, sendCntRankSizeAlign_);
    tpipe_->InitBuffer(sendCntExpertBuf_, moeExpertNumAlign_);
    tpipe_->InitBuffer(numRecvPerExpertBuf_, 2 * moeNumPerRankAlign_);
    tpipe_->InitBuffer(numRecvPerRankBuf_, epWorldSizeAlign_);
    numRecvPerRankTensor_ = numRecvPerRankBuf_.Get<int32_t>();
    sendCntPerRankTensor_ = sendCntRankBuf_.Get<int32_t>();
    sendCntPerExpertTensor_ = sendCntExpertBuf_.Get<int32_t>();
    Duplicate<int32_t>(sendCntPerRankTensor_, 0, counterCnt_);
    Duplicate<int32_t>(sendCntPerExpertTensor_, 0, moeExpertNumAlign_ / sizeof(int32_t));
    ResetCounters();
    SyncAll<true>();
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::ResetCounters()
{
    if (aivId_ == 0U) {
        LocalTensor<int32_t> flagTensor = maskBuf_.Get<int32_t>();
        Duplicate<int32_t>(flagTensor, 1, WIN_ADDR_ALIGN / sizeof(int32_t));
        GlobalTensor<int32_t> flagSource;
        flagSource.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(completionFlagSourceWinAddr_));
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(flagSource, flagTensor, WIN_ADDR_ALIGN / sizeof(int32_t));
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
    if (isSendCore_) {
        GlobalTensor<int32_t> readyState;
        readyState.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(payloadStashStateWinAddr_ + aivId_ * WIN_ADDR_ALIGN));
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(readyState, sendCntPerRankTensor_, UB_STRIDE);
    }
    if (aivId_ == aivNum_ - 1U) {
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyPad(sendCntPerExpertGMTensor_, sendCntPerExpertTensor_, sendPerExpertParams_);
        DataCopy(sendCntPerRankGMTensor_, sendCntPerRankTensor_, counterCnt_);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CalSendCntHistograms(
    LocalTensor<int16_t> dstTensor, LocalTensor<int16_t> &tempTensor, LocalTensor<uint32_t> sendCntTensor,
    uint32_t calCnt, uint32_t scopeSize)
{
    uint64_t rsvdCnt = 0;
    LocalTensor<uint16_t> gatherMaskTensor = maskBuf_.Get<uint16_t>();
    LocalTensor<uint8_t> histoSrcTensor = sendCntTensor.template ReinterpretCast<uint8_t>();
    LocalTensor<uint16_t> histoDstTensor = tempTensor.template ReinterpretCast<uint16_t>();
    GatherMask(tempTensor, dstTensor, gatherMaskTensor, true, calCnt, {1, 1, 0, 0}, rsvdCnt);
    SyncFunc<AscendC::HardEvent::V_S>();
    if (rsvdCnt == 0) {
        Duplicate<uint32_t>(sendCntTensor, 0, scopeSize);
        return;
    }
    Cast(histoSrcTensor, tempTensor, RoundMode::CAST_NONE, rsvdCnt);
    GetExpertFreq(histoDstTensor, histoSrcTensor, rsvdCnt);
    Cast(sendCntTensor, histoDstTensor, RoundMode::CAST_NONE, scopeSize);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::DedupAndSend(
    LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt, uint32_t calCntAlign, uint32_t tmpOffset)
{
    uint32_t maskCnt = Ceil(calCntAlign, BITS_PER_BYTE);
    LocalTensor<int16_t> internalIndexTensor = dstExpBuf_.GetWithOffset<int16_t>(calCntAlign, tmpOffset);
    LocalTensor<int16_t> tempTensorInt16 = tempBuf_.Get<int16_t>();
    LocalTensor<int16_t> subTensorInt16 = topkIdsBuf_.GetWithOffset<int16_t>(calCntAlign, 0);
    LocalTensor<int32_t> gatherIdxTensor = topkIdsBuf_.Get<int32_t>();
    LocalTensor<uint8_t> selectMaskTensor = maskBuf_.GetWithOffset<uint8_t>(maskBytesAlign_, 0);
    LocalTensor<uint8_t> equalMaskTensor = maskBuf_.GetWithOffset<uint8_t>(maskBytesAlign_, maskBytesAlign_);
    LocalTensor<uint8_t> compareMaskTensor = maskBuf_.GetWithOffset<uint8_t>(maskBytesAlign_, 2 * maskBytesAlign_);

    Duplicate<int16_t>(internalIndexTensor, static_cast<int16_t>(moeExpertNumPerRank_), calCnt);
    SyncFunc<AscendC::HardEvent::MTE3_V>(); // 等待上一轮 MTE3 结束
    Div(dstRankTensor_, expertIdsTensor, internalIndexTensor, calCnt);

    if constexpr (IsCached) { // cache下复用 dstSlotIdx
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        CompareScalar(selectMaskTensor, dstSlotIdxTensor_, static_cast<int32_t>(0), AscendC::CMPMODE::GE, calCntAlign);
        return;
    }

    CompareScalar(selectMaskTensor, expertIdsTensor, static_cast<int16_t>(0), AscendC::CMPMODE::GE, calCntAlign);
    Select(dstRankTensor_, selectMaskTensor, dstRankTensor_, static_cast<int16_t>(-1),
           AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, calCnt); // 筛选无效expert id，避免数据污染
    Not(selectMaskTensor, selectMaskTensor, maskCnt);

    Duplicate<int16_t>(subTensorInt16, static_cast<int16_t>(axisK_), calCnt);
    CreateVecIndex(internalIndexTensor, static_cast<int16_t>(0), calCnt);
    Div(tempTensorInt16, internalIndexTensor, subTensorInt16, calCnt);
    Muls(tempTensorInt16, tempTensorInt16, static_cast<int16_t>(axisK_), calCnt);
    Sub(internalIndexTensor, internalIndexTensor, tempTensorInt16, calCnt);
    Muls(tempTensorInt16, tempTensorInt16, static_cast<int16_t>(sizeof(int16_t)), calCnt); // gather offset
    Cast(gatherIdxTensor, tempTensorInt16, RoundMode::CAST_NONE, calCnt);

    for (uint32_t k = 0; k < axisK_; k++) {
        Gather(tempTensorInt16, dstRankTensor_, gatherIdxTensor.template ReinterpretCast<uint32_t>(),
               static_cast<uint32_t>(0), calCnt);
        Compare(equalMaskTensor, dstRankTensor_, tempTensorInt16, AscendC::CMPMODE::EQ, calCntAlign);
        CompareScalar(compareMaskTensor, internalIndexTensor, static_cast<int16_t>(k), AscendC::CMPMODE::GT,
                      calCntAlign);
        And(compareMaskTensor, equalMaskTensor, compareMaskTensor, maskCnt);
        Or(selectMaskTensor, selectMaskTensor, compareMaskTensor, maskCnt);
        Adds(gatherIdxTensor, gatherIdxTensor, static_cast<int32_t>(sizeof(int16_t)), calCnt);
    }
    Not(selectMaskTensor, selectMaskTensor, maskCnt);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CalSendCntPerRank(uint32_t calCnt,
                                                                                       uint32_t calCntAlign,
                                                                                       uint32_t tmpOffset)
{
    LocalTensor<int16_t> tempTensor = topkIdsBuf_.GetWithOffset<int16_t>(calCntAlign, 0);
    LocalTensor<uint32_t> sendCntTensor = dstExpBuf_.GetWithOffset<uint32_t>(tmpOffset / sizeof(uint32_t), tmpOffset);
    LocalTensor<uint8_t> compareMaskTensor = maskBuf_.GetWithOffset<uint8_t>(maskBytesAlign_, 0);
    Select(dstRankTensor_, compareMaskTensor, dstRankTensor_, static_cast<int16_t>(-1),
           AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, calCnt); // 掩码直接使用

    if (epWorldSize_ <= HISTOGRAM_SCOPE_SIZE) {
        CompareScalar(compareMaskTensor, dstRankTensor_, static_cast<int16_t>(0), AscendC::CMPMODE::GE, calCntAlign);
        CalSendCntHistograms(dstRankTensor_, tempTensor, sendCntTensor, calCnt, epWorldSize_);
        Add(sendCntPerRankTensor_, sendCntPerRankTensor_, sendCntTensor.template ReinterpretCast<int32_t>(),
            static_cast<int32_t>(epWorldSize_));
        return;
    }

    uint32_t groupNum = Ceil(epWorldSize_, HISTOGRAM_SCOPE_SIZE);
    uint32_t maskCnt = Ceil(calCntAlign, BITS_PER_BYTE);
    LocalTensor<int16_t> curGroupTensor = topkIdsBuf_.GetWithOffset<int16_t>(calCntAlign, tmpOffset);
    LocalTensor<uint8_t> ltMaskTensor = maskBuf_.GetWithOffset<uint8_t>(maskBytesAlign_, maskBytesAlign_);
    for (uint32_t group = 0; group < groupNum; group++) {
        uint32_t baseId = group * HISTOGRAM_SCOPE_SIZE;
        uint32_t curGroupSize = (group == groupNum - 1) ? (epWorldSize_ - baseId) : HISTOGRAM_SCOPE_SIZE;
        Subs(curGroupTensor, dstRankTensor_, static_cast<int16_t>(baseId), calCnt);
        CompareScalar(compareMaskTensor, curGroupTensor, static_cast<int16_t>(0), AscendC::CMPMODE::GE, calCntAlign);
        CompareScalar(ltMaskTensor, curGroupTensor, static_cast<int16_t>(curGroupSize - 1), AscendC::CMPMODE::LE,
                      calCntAlign);
        And(compareMaskTensor, compareMaskTensor, ltMaskTensor, maskCnt);
        CalSendCntHistograms(curGroupTensor, tempTensor, sendCntTensor, calCnt, curGroupSize);
        Add(sendCntPerRankTensor_[baseId], sendCntPerRankTensor_[baseId],
            sendCntTensor.template ReinterpretCast<int32_t>(), static_cast<int32_t>(curGroupSize));
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CalSendCntPerExpert(
    LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt, uint32_t calCntAlign, uint32_t tmpOffset)
{
    LocalTensor<int16_t> tempTensor = topkIdsBuf_.GetWithOffset<int16_t>(calCntAlign, 0);
    LocalTensor<uint32_t> sendCntTensor = dstExpBuf_.GetWithOffset<uint32_t>(tmpOffset / sizeof(uint32_t), tmpOffset);
    LocalTensor<uint8_t> compareMaskTensor = maskBuf_.GetWithOffset<uint8_t>(maskBytesAlign_, 0);
    if (moeExpertNum_ <= HISTOGRAM_SCOPE_SIZE) {
        CompareScalar(compareMaskTensor, expertIdsTensor, static_cast<int16_t>(0), AscendC::CMPMODE::GE, calCntAlign);
        CalSendCntHistograms(expertIdsTensor, tempTensor, sendCntTensor, calCnt, moeExpertNum_);
        Add(sendCntPerExpertTensor_, sendCntPerExpertTensor_, sendCntTensor.template ReinterpretCast<int32_t>(),
            static_cast<int32_t>(moeExpertNum_));
        return;
    }

    uint32_t groupNum = Ceil(moeExpertNum_, HISTOGRAM_SCOPE_SIZE);
    uint32_t maskCnt = Ceil(calCntAlign, BITS_PER_BYTE);
    LocalTensor<int16_t> curGroupTensor = topkIdsBuf_.GetWithOffset<int16_t>(calCntAlign, tmpOffset);
    LocalTensor<uint8_t> ltMaskTensor = maskBuf_.GetWithOffset<uint8_t>(maskBytesAlign_, maskBytesAlign_);
    for (uint32_t group = 0; group < groupNum; group++) {
        uint32_t baseId = group * HISTOGRAM_SCOPE_SIZE;
        uint32_t curGroupSize = (group == groupNum - 1) ? (moeExpertNum_ - baseId) : HISTOGRAM_SCOPE_SIZE;
        Subs(curGroupTensor, expertIdsTensor, static_cast<int16_t>(baseId), calCnt);
        CompareScalar(compareMaskTensor, curGroupTensor, static_cast<int16_t>(0), AscendC::CMPMODE::GE, calCntAlign);
        CompareScalar(ltMaskTensor, curGroupTensor, static_cast<int16_t>(curGroupSize - 1), AscendC::CMPMODE::LE,
                      calCntAlign);
        And(compareMaskTensor, compareMaskTensor, ltMaskTensor, maskCnt);
        CalSendCntHistograms(curGroupTensor, tempTensor, sendCntTensor, calCnt, curGroupSize);
        Add(sendCntPerExpertTensor_[baseId], sendCntPerExpertTensor_[baseId],
            sendCntTensor.template ReinterpretCast<int32_t>(), static_cast<int32_t>(curGroupSize));
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CalSendCntGroup(
    uint32_t group, uint32_t groupCnt, uint32_t &calCnt, LocalTensor<int32_t> topkIdsGroupTensor,
    LocalTensor<int16_t> tempTensorInt16, const DataCopyPadExtParams<int32_t> &copyPadParams)
{
    uint32_t groupOffset = group * perGroupTokenNum_;
    if (group == groupCnt - 1U) {
        calCnt = (sendTokenNum_ - groupOffset) * axisK_;
    }
    uint32_t topkIdxOffset = (startTokenId_ + groupOffset) * axisK_;
    uint32_t tmpOffset = Ceil(calCnt * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    uint32_t calCntAlign = tmpOffset / sizeof(int16_t);
    DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(calCnt * sizeof(int32_t)), 0U, 0U, 0U};
    dstRankTensor_ = dstExpBuf_.GetWithOffset<int16_t>(calCntAlign, 0);
    if (group > 0U) {
        SyncFunc<AscendC::HardEvent::V_MTE2>();
    }
    DataCopyPad(topkIdsGroupTensor, topkIdxGMTensor_[topkIdxOffset], copyParams, copyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(tempTensorInt16, topkIdsGroupTensor, RoundMode::CAST_NONE, calCnt);
    CalSendCntPerExpert(tempTensorInt16, calCnt, calCntAlign, tmpOffset);
    if constexpr (IsCached) {
        dstSlotIdxTensor_ = topkIdsBuf_.Get<int32_t>();
        SyncFunc<AscendC::HardEvent::V_MTE2>();
        DataCopyPad(dstSlotIdxTensor_, cachedSlotIdxGMTensor_[topkIdxOffset], copyParams, copyPadParams);
    }
    DedupAndSend(tempTensorInt16, calCnt, calCntAlign, tmpOffset);
    if constexpr (IsCached) {
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
        DataCopyPad(dstSlotIdxGMTensor_[topkIdxOffset], dstSlotIdxTensor_, copyParams);
        SyncFunc<AscendC::HardEvent::MTE3_V>();
    }
    CalSendCntPerRank(calCnt, calCntAlign, tmpOffset);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyExtParams dstRankCopyParams = {1U, static_cast<uint32_t>(calCnt * sizeof(int16_t)), 0U, 0U, 0U};
    DataCopyPad(dstRankGMTensor_[topkIdxOffset], dstRankTensor_, dstRankCopyParams);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::PublishSendCounts()
{
    DataCopy(sendCounterGMTensor_[prepareCoreId_ * counterAlign512_], sendCntPerRankTensor_, counterCnt_);
    SetAtomicAdd<int32_t>();
    DataCopyPad(sendCntPerRankGMTensor_, sendCntPerRankTensor_, sendPerRankParams_);
    DataCopyPad(sendCntPerExpertGMTensor_, sendCntPerExpertTensor_, sendPerExpertParams_);
    SetAtomicNone();
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CalSendCnt()
{
    if (!isPrepareCore_) {
        return;
    }
    SplitToCore(axisBS_, prepareCoreCount_, prepareCoreId_, startTokenId_, endTokenId_, sendTokenNum_);
    if (startTokenId_ >= axisBS_) {
        DataCopy(sendCounterGMTensor_[prepareCoreId_ * counterAlign512_], sendCntPerRankTensor_, counterCnt_);
        return;
    }

    uint32_t groupCnt = Ceil(sendTokenNum_, perGroupTokenNum_);
    uint32_t calCnt = perGroupTokenNum_ * axisK_;
    LocalTensor<int32_t> topkIdsGroupTensor = topkIdsBuf_.Get<int32_t>();
    LocalTensor<int16_t> tempTensorInt16 = tempBuf_.Get<int16_t>();
    DataCopyPadExtParams<int32_t> topkIdsCntCopyPadParams{false, 0U, 0U, 0U};

    for (uint32_t group = 0; group < groupCnt; ++group) {
        CalSendCntGroup(group, groupCnt, calCnt, topkIdsGroupTensor, tempTensorInt16, topkIdsCntCopyPadParams);
    }
    PublishSendCounts();
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SendCountToPeer(
    uint32_t peer, GM_ADDR dst, GM_ADDR src, GM_ADDR notifyAddr, uint64_t notifyVal)
{
    auto multiHandle = static_cast<MultiChannelHandle>(GetOwnedJettyHandle());
    auto batch = hcomm_.MakeBatchHandle(multiHandle, hcommTensor_, HCOMM_INIT_SIZE);
    Mc2Kernel::CheckSharedJettyQueueDepth(batch);
    uint32_t peerIndex = peer < epRankId_ ? peer : peer - 1;
    auto &handle = hcomm_.GetHandleRef(batch, peerIndex, dst);
    CheckSharedJettyResult(hcomm_.WriteWithNotifyNbi<DATA_CFG>(handle, dst, src,
                                                           moeNumPerRankSize_, notifyAddr, notifyVal));
    CheckSharedJettyResult(hcomm_.BatchCommit(batch));
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::Communication()
{
    if (!isSendCore_) {
        return;
    }
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    Duplicate<uint8_t>(hcommTensor_, 0, HCOMM_INIT_SIZE);
    SyncFunc<AscendC::HardEvent::V_S>();

    DataCopy(sendCntPerRankTensor_, sendCntPerRankGMTensor_, counterCnt_);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    for (uint32_t dstRankId = aivId_; dstRankId < epWorldSize_; dstRankId += sendCoreCount_) {
        int32_t sendCnt = sendCntPerRankTensor_.GetValue(dstRankId);
        uint64_t notifyVal = (static_cast<uint64_t>(sendCnt) << NOTIFY_VAL_SHIFT) | 1U;
        uint32_t srcStride = dstRankId * moeExpertNumPerRank_;
        // 计算目标窗口地址:
        GM_ADDR remoteStateAddr = GetWinAddrByRankId(mc2Context_, dstRankId, cntWinStateOffset_);
        GM_ADDR notifyAddr = remoteStateAddr + epRankId_ * WIN_ADDR_ALIGN;
        GM_ADDR remoteCountAddr = remoteStateAddr + cntFlagWinSize_ + epRankId_ * moeNumPerRankAlign512_;
        GM_ADDR srcWorkspaceAddr = sendCntWorkspaceAddr_ + epWorldSizeAlign512_ + srcStride * sizeof(int32_t);

        if (dstRankId != epRankId_) { // 远端 使用URMA发送 count + state
            SendCountToPeer(dstRankId, remoteCountAddr, srcWorkspaceAddr, notifyAddr, notifyVal);
            continue;
        }

        // 本端
        GlobalTensor<int32_t> countGMTensor;
        GlobalTensor<uint64_t> notifyGMTensor;
        countGMTensor.SetGlobalBuffer((__gm__ int32_t *)remoteCountAddr);
        notifyGMTensor.SetGlobalBuffer((__gm__ uint64_t *)notifyAddr);
        LocalTensor<uint64_t> notifyLocalTensor = tempBuf_.GetWithOffset<uint64_t>(INT64_UB_STRIDE, 0);
        LocalTensor<int32_t> cntPerExpertTensor = tempBuf_.GetWithOffset<int32_t>(expertPerRankAlign_, UB_ALIGN);
        DataCopyParams expertCntCopyParams = {1U, static_cast<uint16_t>(moeNumPerRankSize_), 0U, 0U};
        DataCopyPad(cntPerExpertTensor, sendCntPerExpertGMTensor_[srcStride], expertCntCopyParams, padParams_);
        notifyLocalTensor.SetValue(0, notifyVal);
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
        DataCopyPad(countGMTensor, cntPerExpertTensor, expertCntCopyParams);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        PipeBarrier<PIPE_MTE3>(); // perExpert 写完再写notifyVal
        DataCopy(notifyGMTensor, notifyLocalTensor, INT64_UB_STRIDE);
    }
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>(); // tempBuf_ 复用
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::GetRecvCount()
{
    if (aivId_ != aivNum_ - 1) { // 最后一个核处理recvCount
        return;
    }

    uint32_t mask = 1;
    int32_t sumOfFlag = -1;
    int32_t commpareFlag = static_cast<int32_t>(epWorldSize_);
    LocalTensor<int32_t> recvCounterTensor = tempBuf_.GetWithOffset<int32_t>(epWorldSize_ * UB_STRIDE, 0);
    LocalTensor<float> tempFp32 = topkIdsBuf_.GetWithOffset<float>(epWorldSize_ * UB_STRIDE, 0);
    LocalTensor<float> recvCounterTensorFp32 = recvCounterTensor.template ReinterpretCast<float>();
    LocalTensor<float> numRecvPerRankTensorFp32 = numRecvPerRankTensor_.template ReinterpretCast<float>();
    SyncFunc<AscendC::HardEvent::MTE3_V>(); // buf 复用
    while (sumOfFlag != commpareFlag) {     // 状态位check
        DataCopy(recvCounterTensor, recvCounterGMTensor_, statusCopyParams_);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(numRecvPerRankTensorFp32, recvCounterTensorFp32, tempFp32, mask, epWorldSize_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = numRecvPerRankTensor_.GetValue(0);
    }
    SetRecvNumPerExpert();                // 计算本卡上各专家接收的token总数
    SetRecvNumPerRank(recvCounterTensor); // 计算本端接收来自各卡的token总数
    // status clear
    Duplicate<int32_t>(recvCounterTensor, 0, epWorldSize_ * UB_STRIDE);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(recvCounterGMTensor_, recvCounterTensor, clearStatusCopyParams_);
    diagWriter_.RunPosRecord(MOE_EP_DISPATCH_RUN_POS_COUNT_READY);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SetRecvNumPerExpert()
{
    uint32_t recvNumAlign = epWorldSize_ * expertPerRankAlign_;
    GlobalTensor<int32_t> localExpertRecvGMTensor;
    localExpertRecvGMTensor.SetGlobalBuffer((__gm__ int32_t *)(localCntStateWinAddr_ + cntFlagWinSize_));
    numRecvPerExpertTensor_ = numRecvPerExpertBuf_.Get<int64_t>();
    LocalTensor<int64_t> tempRecvTensorInt64 = dstExpBuf_.Get<int64_t>();
    LocalTensor<int64_t> recvCntTensor = topkIdsBuf_.GetWithOffset<int64_t>(INT64_UB_STRIDE, 0);
    LocalTensor<int32_t> recvTensorInt32 = topkIdsBuf_.GetWithOffset<int32_t>(recvNumAlign, UB_ALIGN);
    LocalTensor<int64_t> sharedTmp = recvTensorInt32.template ReinterpretCast<int64_t>();
    LocalTensor<uint8_t> sharedTmpInt8 = recvTensorInt32.template ReinterpretCast<uint8_t>();

    const uint32_t shape[] = {epWorldSize_, expertPerRankAlign_};
    DataCopyParams inRecvCntParams = {static_cast<uint16_t>(epWorldSize_), static_cast<uint16_t>(moeNumPerRankAlign_),
                                      static_cast<uint16_t>(moeNumPerRankAlign512_ - moeNumPerRankAlign_), 0U};
    DataCopyParams recvPerExpertParams = {1U, static_cast<uint16_t>(moeExpertNumPerRank_ * sizeof(int64_t)), 0U, 0U};
    DataCopyPad(recvTensorInt32, localExpertRecvGMTensor, inRecvCntParams, padParams_);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(tempRecvTensorInt64, recvTensorInt32, RoundMode::CAST_NONE, recvNumAlign);
    ReduceSum<int64_t, AscendC::Pattern::Reduce::RA, true>(numRecvPerExpertTensor_, tempRecvTensorInt64, sharedTmpInt8,
                                                           shape, true);

    if constexpr (DoCpuSync) { // 计算actualA，并写入host pin
        GlobalTensor<int64_t> hostPinnedCounterTensor;
        hostPinnedCounterTensor.SetGlobalBuffer((__gm__ int64_t *)hostPinnedCounterAddrGM_);
        ReduceSum(recvCntTensor, numRecvPerExpertTensor_, sharedTmp, moeExpertNumPerRank_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(hostPinnedCounterTensor, recvCntTensor, INT64_UB_STRIDE);
    }

    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(numRecvPerExpertGMTensor_, numRecvPerExpertTensor_, recvPerExpertParams);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SetRecvNumPerRank(
    LocalTensor<int32_t> recvTmpTensor)
{
    LocalTensor<uint32_t> gatherTmpTensor = topkIdsBuf_.GetWithOffset<uint32_t>(UB_STRIDE, epWorldSize_ * UB_ALIGN);
    SyncFunc<AscendC::HardEvent::V_S>();
    gatherTmpTensor.SetValue(0, 2); // 设置掩码，取源操作数每个datablock中的第2个元素
    uint32_t mask = 2;              // 源操作数每个datablock只需要处理两个元素
    uint64_t rsvdCnt = 0;
    GatherMaskParams recvMaskParams = {1, static_cast<uint16_t>(epWorldSize_), 1, 0};
    DataCopyParams recvPerRankParams = {1U, static_cast<uint16_t>(epWorldSize_ * sizeof(int32_t)), 0U, 0U};
    SyncFunc<AscendC::HardEvent::S_V>();
    GatherMask(numRecvPerRankTensor_, recvTmpTensor, gatherTmpTensor, true, mask, recvMaskParams, rsvdCnt);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(numRecvPerRankGMTensor_, numRecvPerRankTensor_, recvPerRankParams);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::GetSlotStartNum()
{
    slotIdxPerRankTensor_ = sendCntExpertBuf_.GetWithOffset<int32_t>(counterCnt_, 0);
    Duplicate<int32_t>(slotIdxPerRankTensor_, 0, epWorldSize_);
    if (prepareCoreId_ == 0U) {
        return;
    }
    uint32_t groupCnt = Ceil(prepareCoreId_ * epWorldSizeAlign_, perGroupSizeAlign_ * 2);
    uint32_t copyNumPerGroup = perGroupSizeAlign_ * 2 / epWorldSizeAlign_;
    LocalTensor<int32_t> counterTmpTensor = topkIdsBuf_.Get<int32_t>();
    LocalTensor<int32_t> hitTmpTensor = sendCntRankBuf_.Get<int32_t>();
    LocalTensor<uint8_t> sharedTmpTensor = tempBuf_.Get<uint8_t>();
    DataCopyParams counterCopyParams = {1U, static_cast<uint16_t>(epWorldSizeAlign_),
                                        static_cast<uint16_t>(epWorldSizeAlign512_ - epWorldSizeAlign_), 0U};
    for (uint32_t i = 0; i < groupCnt; i++) {
        uint32_t copyNum =
            (i == groupCnt - 1) ? (prepareCoreId_ - copyNumPerGroup * i) : copyNumPerGroup;
        uint32_t gmOffset = i * copyNumPerGroup * epWorldSizeAlign512_ / sizeof(int32_t);
        counterCopyParams.blockCount = static_cast<uint16_t>(copyNum);
        DataCopyPad(counterTmpTensor, sendCounterGMTensor_[gmOffset], counterCopyParams, padParams_);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        const uint32_t shape[] = {copyNum, counterCnt_};
        ReduceSum<int32_t, AscendC::Pattern::Reduce::RA, true>(hitTmpTensor, counterTmpTensor, sharedTmpTensor, shape,
                                                               true);
        Add(slotIdxPerRankTensor_, slotIdxPerRankTensor_, hitTmpTensor, epWorldSize_);
        SyncFunc<AscendC::HardEvent::V_MTE2>();
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SetSrcTokenIdxAndDstSlot(
    uint32_t topkOffset, uint32_t calCnt, uint32_t calCntAlign, uint32_t &sendLocalCnt,
    LocalTensor<int32_t> &cntLocalTensor)
{
    uint64_t rsvdCnt = 0;
    LocalTensor<int16_t> internalIndexTensor = topkIdsBuf_.GetWithOffset<int16_t>(calCntAlign, 0);
    LocalTensor<int16_t> scatterTensor = topkIdsBuf_.GetWithOffset<int16_t>(calCntAlign, calCntAlign * sizeof(int16_t));
    LocalTensor<int32_t> dstSlotIdxTensor = topkIdsBuf_.Get<int32_t>();
    LocalTensor<int32_t> tempLocalTensor = dstExpBuf_.Get<int32_t>();
    LocalTensor<uint16_t> gatherMaskTensor = maskBuf_.Get<uint16_t>();
    LocalTensor<uint8_t> compareMaskTensor = maskBuf_.Get<uint8_t>();
    DataCopyParams srcTokenCopyParams = {1U, 0U, 0U, 0U};

    for (uint32_t rankId = 0; rankId < epWorldSize_; ++rankId) {
        // 调整处理顺序，先处理其他对端卡，本端卡最后处理，后续处理token可复用
        uint32_t dstRankId = rankId >= epRankId_ ? (rankId + 1) : rankId;
        dstRankId = (rankId == epWorldSize_ - 1) ? epRankId_ : dstRankId;
        CreateVecIndex(internalIndexTensor, static_cast<int16_t>(0), calCnt); // 生成元素索引
        CompareScalar(compareMaskTensor, dstRankIdTensor_, static_cast<int16_t>(dstRankId), AscendC::CMPMODE::EQ,
                      calCntAlign);
        if (rankId > 0) {
            SyncFunc<AscendC::HardEvent::MTE3_V>();
        }
        SyncFunc<AscendC::HardEvent::S_V>(); // 需要等待上一轮本端命中判断结束
        GatherMask(internalIndexTensor, internalIndexTensor, gatherMaskTensor, true, calCnt, {1, 1, 0, 0}, rsvdCnt);
        if (dstRankId == epRankId_) {
            sendLocalCnt = static_cast<uint32_t>(rsvdCnt);
        }
        if (rsvdCnt == 0) {
            continue;
        }

        SyncFunc<AscendC::HardEvent::V_S>();
        int32_t curSlot = cntLocalTensor.GetValue(dstRankId);
        int32_t curCnt = static_cast<int32_t>(rsvdCnt);
        if constexpr (!IsCached) {
            // slot 加 epWorldSize_ 偏移：散射值 >= epWorldSize_，避免与后续卡匹配错误
            CreateVecIndex(scatterTensor, static_cast<int16_t>(curSlot + epWorldSize_), curCnt);
            Cast(tempLocalTensor, internalIndexTensor, RoundMode::CAST_NONE, curCnt);
            Muls(tempLocalTensor, tempLocalTensor, static_cast<int32_t>(sizeof(int16_t)), curCnt);
            Scatter(dstRankIdTensor_, scatterTensor, tempLocalTensor.template ReinterpretCast<uint32_t>(),
                    static_cast<uint32_t>(0), curCnt);
        }
        Duplicate<int16_t>(scatterTensor, static_cast<int16_t>(axisK_), curCnt);
        Div(internalIndexTensor, internalIndexTensor, scatterTensor, curCnt);
        Cast(tempLocalTensor, internalIndexTensor, RoundMode::CAST_NONE, curCnt);
        Adds(tempLocalTensor, tempLocalTensor, static_cast<int32_t>(topkOffset / axisK_), curCnt);
        uint32_t dstIndex = dstRankId * tokenCntAlign_ + curSlot;
        srcTokenCopyParams.blockLen = static_cast<uint32_t>(curCnt * sizeof(int32_t));
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyPad(srcTokenIdxGMTensor_[dstIndex], tempLocalTensor, srcTokenCopyParams);
        cntLocalTensor.SetValue(dstRankId, curSlot + curCnt);
        rsvdCnt = 0;
    }
    if constexpr (!IsCached) {
        // 去偏移，有效位还原为 slot，未散射位(重复/无效)保持 -1
        CompareScalar(compareMaskTensor, dstRankIdTensor_, static_cast<int16_t>(0), AscendC::CMPMODE::GE, calCntAlign);
        Subs(dstRankIdTensor_, dstRankIdTensor_, static_cast<int16_t>(epWorldSize_), calCntAlign);
        Select(dstRankIdTensor_, compareMaskTensor, dstRankIdTensor_, static_cast<int16_t>(-1),
               AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, calCntAlign);
        Cast(dstSlotIdxTensor, dstRankIdTensor_, RoundMode::CAST_NONE, calCnt);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyExtParams dstSlotCopyParams = {1U, static_cast<uint32_t>(calCnt * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(dstSlotIdxGMTensor_[topkOffset], dstSlotIdxTensor, dstSlotCopyParams);
    }
    SyncFunc<AscendC::HardEvent::MTE3_V>(); // 下一轮等待这一轮做完
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CopyInMetaInfo(uint32_t tokenId,
                                                                                    uint32_t topkOffset)
{
    LocalTensor<int32_t> metaCopyInTensor = metaSlotQueue_.AllocTensor<int32_t>();
    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
        DataCopyPad(metaCopyInTensor.template ReinterpretCast<uint8_t>(),
                    scalesGMTensor_.template ReinterpretCast<uint8_t>()[tokenId * scalesBytes_], scalesCopyParams_,
                    padParams_);
    }
    DataCopyPad(metaCopyInTensor[topkCopyAlign_], topkIdxGMTensor_[topkOffset], topkCopyParams_, padParams_);
    if constexpr (IsTopkWeights) {
        DataCopyPad(metaCopyInTensor[topkCopyAlign_ + axisKAlign_].template ReinterpretCast<float>(),
                    topkWeightsGMTensor_[topkOffset], topkCopyParams_, padParams_);
    }
    metaSlotQueue_.EnQue(metaCopyInTensor);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::ProcessMetaSlot(
    uint32_t startId, uint32_t tokenCnt, uint32_t sendLocalCnt, uint32_t &topkOffset, uint32_t &curLocalSlot)
{
    // meta 流水写 stash + 本端命中内联写本地窗 meta 区
    LocalTensor<int32_t> srcTokenTensor = dstExpBuf_.Get<int32_t>();
    uint32_t endId = startId + tokenCnt;
    uint32_t localIdx = 0;
    uint32_t metaSetStride = topkCopyAlign_ + 2 * axisKAlign_; // meta 槽内 srcRank/tokenIdx 的下标
    CopyInMetaInfo(startId, topkOffset);
    for (uint32_t tokenId = startId; tokenId < endId; ++tokenId) {
        metaLocalTensor_ = metaSlotQueue_.DeQue<int32_t>();
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        metaLocalTensor_.SetValue(metaSetStride, epRankId_);
        metaLocalTensor_.SetValue(metaSetStride + 1, tokenId);
        if (tokenId + 1 < endId) {
            topkOffset += axisK_;
            CopyInMetaInfo(tokenId + 1, topkOffset);
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(metaSlotGMTensor_[tokenId * metaSlotStride_], metaLocalTensor_, metaSlotCopyParams_);
        // 本端 x 不拷入窗口（epilogue 自段直读 x）
        if ((localIdx < sendLocalCnt) &&
            (static_cast<uint32_t>(srcTokenTensor.GetValue(localIdx)) == tokenId)) { // 本端命中判断
            localMetaSlotGMTensor_.SetGlobalBuffer(
                (__gm__ int32_t *)(localSlotWinAddr_ + static_cast<uint64_t>(curLocalSlot) * perSlotBytes_ +
                                   tokenSize_));
            DataCopyPad(localMetaSlotGMTensor_, metaLocalTensor_, metaSlotCopyParams_);
            localIdx++;
            curLocalSlot++;
        }
        metaSlotQueue_.FreeTensor<int32_t>(metaLocalTensor_);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SendPhase()
{
    if (!isPrepareCore_ || startTokenId_ >= axisBS_) {
        return;
    }
    GetSlotStartNum(); // 计算起始slot id

    uint32_t groupCnt = Ceil(sendTokenNum_, perGroupTokenNum_);
    uint32_t tokenCnt = perGroupTokenNum_;
    uint32_t strideAlign = ALIGNED_LEN_256 / sizeof(int16_t);
    dstRankIdTensor_ = tempBuf_.Get<int16_t>();
    LocalTensor<int32_t> cntLocalTensor = numRecvPerRankBuf_.Get<int32_t>();
    DataCopyPadExtParams<int16_t> dstRankCopyPadParams{false, 0U, 0U, 0U};
    SyncFunc<AscendC::HardEvent::V_S>();
    uint32_t curLocalSlot = static_cast<uint32_t>(slotIdxPerRankTensor_.GetValue(epRankId_));
    uint32_t sendLocalCnt = 0;
    Copy<int32_t>(cntLocalTensor, slotIdxPerRankTensor_, counterCnt_);

    for (uint32_t group = 0; group < groupCnt; group++) {
        if (group == groupCnt - 1) {
            tokenCnt = sendTokenNum_ - group * perGroupTokenNum_;
        }
        uint32_t calCnt = tokenCnt * axisK_;
        uint32_t startId = startTokenId_ + group * perGroupTokenNum_;
        uint32_t topkOffset = startId * axisK_;
        uint32_t calCntAlign = Ceil(calCnt, strideAlign) * strideAlign;
        DataCopyExtParams dstRankCopyParams = {1U, static_cast<uint32_t>(calCnt * sizeof(int16_t)), 0U, 0U, 0U};
        DataCopyPad(dstRankIdTensor_, dstRankGMTensor_[topkOffset], dstRankCopyParams, dstRankCopyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        SetSrcTokenIdxAndDstSlot(topkOffset, calCnt, calCntAlign, sendLocalCnt, cntLocalTensor);
        SyncFunc<AscendC::HardEvent::V_S>();
        ProcessMetaSlot(startId, tokenCnt, sendLocalCnt, topkOffset, curLocalSlot);
        if (group < groupCnt - 1) {
            SyncFunc<AscendC::HardEvent::V_MTE2>();
        }
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::PublishPrepareReady()
{
    if (!isPrepareCore_ || startTokenId_ == endTokenId_) {
        return;
    }
    uint32_t firstSender = 0U;
    uint32_t lastSender = 0U;
    GetOverlappingCores(startTokenId_, endTokenId_, sendCoreCount_, firstSender, lastSender);
    LocalTensor<int32_t> statusTensor = maskBuf_.Get<int32_t>();
    Duplicate<int32_t>(statusTensor, 1, UB_STRIDE);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    // Publish only after all source indices and metadata in this prepare interval are visible.
    PipeBarrier<PIPE_MTE3>();
    SetAtomicAdd<int32_t>();
    for (uint32_t sender = firstSender; sender < lastSender; ++sender) {
        GlobalTensor<int32_t> readyCounter;
        readyCounter.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(payloadStashStateWinAddr_ + sender * WIN_ADDR_ALIGN));
        DataCopy(readyCounter, statusTensor, UB_STRIDE);
    }
    SetAtomicNone();
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::WaitPrepareReady()
{
    uint32_t prepareStart = 0U;
    uint32_t prepareEnd = 0U;
    SplitToCore(axisBS_, sendCoreCount_, aivId_, startTokenId_, endTokenId_, sendTokenNum_);
    GetOverlappingCores(startTokenId_, endTokenId_, prepareCoreCount_, prepareStart, prepareEnd);
    uint32_t expectedReady = prepareEnd - prepareStart;
    if (expectedReady == 0U) {
        return;
    }
    LocalTensor<int32_t> statusTensor = maskBuf_.Get<int32_t>();
    GlobalTensor<int32_t> readyCounter;
    readyCounter.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(payloadStashStateWinAddr_ + aivId_ * WIN_ADDR_ALIGN));
    int32_t ready = 0;
    while (ready != static_cast<int32_t>(expectedReady)) {
        DataCopy(statusTensor, readyCounter, UB_STRIDE);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        ready = statusTensor.GetValue(0);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::GetOverlappingCores(
    uint32_t begin, uint32_t end, uint32_t coreCount, uint32_t &first, uint32_t &last) const
{
    auto overlap = MoeEpDispatchSchedule::Overlap(axisBS_, coreCount, begin, end);
    first = overlap.begin;
    last = overlap.end;
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CommBufferInit()
{
    PipeBarrier<PIPE_ALL>();
    tpipe_->Reset();
    // 通信初始化
    uint32_t hcommWriteWqeCnt = Ceil(HCOMM_SQE_BYTES + SGE_PER_SQE * HCOMM_SGE_BYTES, HCOMM_WQE_BYTES);
    hcommBatchBufferBytes_ = HCOMM_BATCH_CAPACITY * hcommWriteWqeCnt * HCOMM_WQE_BYTES;
    uint32_t hcommBatchSize = hcommBatchBufferBytes_ / sizeof(uint8_t);
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE + hcommBatchBufferBytes_);
    hcommTensor_ = hcommBuf_.GetWithOffset<uint8_t>(HCOMM_INIT_SIZE, 0);
    hcommBatchTensor_ = hcommBuf_.GetWithOffset<uint8_t>(hcommBatchSize, HCOMM_INIT_SIZE);
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);
    Duplicate<uint8_t>(hcommBatchTensor_, 0, hcommBatchSize);
    SyncFunc<AscendC::HardEvent::V_S>();

    // ub buffer init
    tpipe_->InitBuffer(sendCntRankBuf_, sendCntRankSizeAlign_);
    tpipe_->InitBuffer(tempBuf_, MAX_BATCH_SIZE);
    tpipe_->InitBuffer(maskBuf_, UB_ALIGN);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::GetTokenRange(
    uint32_t dstRank, uint32_t &tokenStart, uint32_t &tokenNum)
{
    tokenStart = GetTokenBoundary(dstRank, startTokenId_);
    tokenNum = 0U;
    if (startTokenId_ < endTokenId_) {
        tokenNum = GetTokenBoundary(dstRank, endTokenId_) - tokenStart;
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline uint32_t MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::GetTokenBoundary(
    uint32_t dstRank, uint32_t tokenBoundary)
{
    uint32_t prepare = tokenBoundary == axisBS_ ? prepareCoreCount_ :
                        MoeEpDispatchSchedule::Owner(axisBS_, prepareCoreCount_, tokenBoundary);
    uint32_t base = 0U;
    for (uint32_t row = 0U; row < prepare; ++row) {
        base += static_cast<uint32_t>(sendCounterGMTensor_.GetValue(row * counterAlign512_ + dstRank));
    }
    if (tokenBoundary == axisBS_) {
        return base;
    }
    auto range = MoeEpDispatchSchedule::Partition(axisBS_, prepareCoreCount_, prepare);
    // At a prepare boundary its predecessor's count suffices; the next producer may still be running.
    if (tokenBoundary == range.begin) {
        return base;
    }
    uint32_t peerCount = static_cast<uint32_t>(
        sendCounterGMTensor_.GetValue(prepare * counterAlign512_ + dstRank));
    return LowerBoundToken(dstRank, base, base + peerCount, tokenBoundary);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline uint32_t MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::LowerBoundToken(
    uint32_t dstRank, uint32_t begin, uint32_t end, uint32_t tokenBoundary)
{
    // Only search the ready boundary producer's sorted peer slice, never unfinished neighboring slices.
    uint32_t peerBase = dstRank * tokenCntAlign_;
    while (begin < end) {
        uint32_t middle = begin + (end - begin) / 2U;
        uint32_t token = static_cast<uint32_t>(srcTokenIdxGMTensor_.GetValue(peerBase + middle));
        if (token < tokenBoundary) {
            begin = middle + 1U;
        } else {
            end = middle;
        }
    }
    return begin;
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::WriteToRemoteWindow()
{
    if (!isSendCore_) {
        return;
    }
    CommBufferInit();
    WaitPrepareReady();
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(sendCounterGMTensor_);
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(srcTokenIdxGMTensor_);
    payloadBatch_ = hcomm_.MakeBatchHandle(static_cast<MultiChannelHandle>(GetOwnedJettyHandle()),
                                          hcommBatchTensor_, hcommBatchBufferBytes_);
    Mc2Kernel::CheckSharedJettyQueueDepth(payloadBatch_);
    pendingWqebb_ = 0U;
    firstBatchCommitted_ = false;

    // Stagger the first peer so all 32 Jettys do not burst into the same destination at once.
    for (uint32_t peerOffset = 0; peerOffset < epWorldSize_; ++peerOffset) {
        uint32_t dstRankId = (aivId_ + peerOffset) % epWorldSize_;
        GM_ADDR notifyAddr = GetWinAddrByRankId(mc2Context_, dstRankId, slotWinStateOffset_) + dstStateWinOffset_;
        notifyAddr += aivId_ * WIN_ADDR_ALIGN;
        if (unlikely(dstRankId == epRankId_)) {
            SendCompletion(dstRankId, notifyAddr);
            continue;
        }
        activePeerIndex_ = dstRankId < epRankId_ ? dstRankId : dstRankId - 1U;
        SendTokenChunkToPeer(dstRankId, notifyAddr);
    }
    CommitBatch(true);
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(topkIdxGMTensor_);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SendTokenData(
    uint32_t tokenStart, uint32_t tokenNum, uint32_t dstRankId, GM_ADDR remoteWinAddr,
    BufDesc *localDescs)
{
    uint32_t processed = 0U;
    uint32_t maxBatchCnt = MAX_BATCH_SIZE / sizeof(int32_t);
    DataCopyExtParams copyParams = {1U, 0U, 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
    LocalTensor<int32_t> srcTokenIdxTensor = tempBuf_.Get<int32_t>();
    GlobalTensor<int32_t> srcTokenIdxGMTensor;
    srcTokenIdxGMTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(srcTokenTableAddr_ + static_cast<uint64_t>(dstRankId) * srcTokenListBytes_));
    while (processed < tokenNum) {
        uint32_t batchCnt = (tokenNum - processed > maxBatchCnt) ? maxBatchCnt : tokenNum - processed;
        copyParams.blockLen = static_cast<uint32_t>(batchCnt * sizeof(int32_t));
        DataCopyPad(srcTokenIdxTensor, srcTokenIdxGMTensor[tokenStart + processed], copyParams, copyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        uint32_t batchGroup = Ceil(batchCnt, SLOT_NUM_PER_SQE);
        uint32_t srcIdx = 0U;
        for (uint32_t i = 0U; i < batchGroup; ++i) {
            uint32_t slotNum = i == batchGroup - 1U ? batchCnt - i * SLOT_NUM_PER_SQE : SLOT_NUM_PER_SQE;
            uint32_t sgePerSqe = slotNum * SGE_PER_SLOT;
            for (uint32_t j = 0U; j < sgePerSqe; j += SGE_PER_SLOT) {
                uint64_t srcTokenId = static_cast<uint64_t>(srcTokenIdxTensor.GetValue(srcIdx++));
                localDescs[j].addr = xAddr_ + srcTokenId * tokenSize_;
                localDescs[j + 1U].addr = payloadStashWinAddr_ + srcTokenId * metaSlotBytes_;
            }
            WritePayload(remoteWinAddr, localDescs, sgePerSqe);
            remoteWinAddr += slotNum * perSlotBytes_;
        }
        processed += batchCnt;
        // The next peer reuses this UB even when the payload batch has not been committed.
        SyncFunc<AscendC::HardEvent::S_MTE2>();
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SendTokenChunkToPeer(
    uint32_t dstRankId, GM_ADDR notifyAddr)
{
    uint32_t tokenStart = 0U;
    uint32_t tokenNum = 0U;
    GetTokenRange(dstRankId, tokenStart, tokenNum);
    BufDesc localDescs[SGE_PER_SQE] = {};
    for (uint32_t i = 0; i < SGE_PER_SQE; i += SGE_PER_SLOT) { // srcLen 预填
        localDescs[i].len = tokenSize_;
        localDescs[i + 1].len = metaSlotBytes_;
    }
    GM_ADDR remoteWinBaseAddr = GetWinAddrByRankId(mc2Context_, dstRankId, winDataOffset_) + dstRankWinOffset_ +
                                static_cast<uint64_t>(tokenStart) * perSlotBytes_;
    SendTokenData(tokenStart, tokenNum, dstRankId, remoteWinBaseAddr, localDescs);
    SendCompletion(dstRankId, notifyAddr);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::ReserveBatchSpace(uint32_t wqebbCount)
{
    uint32_t limit = firstBatchCommitted_ ? hcommBatchBufferBytes_ / HCOMM_WQE_BYTES : HCOMM_FIRST_BATCH_WQEBB;
    if (pendingWqebb_ + wqebbCount > limit) {
        CommitBatch(true);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CommitBatch(bool force)
{
    uint32_t limit = firstBatchCommitted_ ? hcommBatchBufferBytes_ / HCOMM_WQE_BYTES : HCOMM_FIRST_BATCH_WQEBB;
    if (pendingWqebb_ != 0U && (force || pendingWqebb_ >= limit)) {
        CheckSharedJettyResult(hcomm_.BatchCommit(payloadBatch_));
        pendingWqebb_ = 0U;
        firstBatchCommitted_ = true;
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::WritePayload(
    GM_ADDR dst, BufDesc *localDescs, uint32_t sgeCount)
{
    uint32_t wqebbCount = Ceil(HCOMM_SQE_BYTES + sgeCount * HCOMM_SGE_BYTES, HCOMM_WQE_BYTES);
    ReserveBatchSpace(wqebbCount);
    auto &handle = hcomm_.GetHandleRef(payloadBatch_, activePeerIndex_, dst);
    CheckSharedJettyResult(hcomm_.WriteNbi<DATA_CFG>(handle, dst, localDescs, sgeCount));
    pendingWqebb_ += wqebbCount;
    CommitBatch(false);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SendCompletion(
    uint32_t dstRankId, GM_ADDR notifyAddr)
{
    if (dstRankId == epRankId_) {
        LocalTensor<int32_t> statusTensor = maskBuf_.Get<int32_t>();
        GlobalTensor<int32_t> source;
        GlobalTensor<int32_t> target;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(completionFlagSourceWinAddr_));
        target.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(notifyAddr));
        DataCopy(statusTensor, source, UB_STRIDE);
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
        DataCopy(target, statusTensor, UB_STRIDE);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        return;
    }
    constexpr uint32_t flagWqebb = (HCOMM_SQE_BYTES + HCOMM_SGE_BYTES) / HCOMM_WQE_BYTES;
    ReserveBatchSpace(flagWqebb);
    auto &handle = hcomm_.GetHandleRef(payloadBatch_, activePeerIndex_, notifyAddr);
    CheckSharedJettyResult(hcomm_.WriteNbi<NOTIFY_CFG>(handle, notifyAddr, completionFlagSourceWinAddr_, WIN_ADDR_ALIGN));
    pendingWqebb_ += flagWqebb;
    CommitBatch(false);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::Process()
{
    if ASCEND_IS_AIV { // 全aiv处理
        BufferInit();
        CalSendCnt();
        SyncAll<true>();
        Communication();
        SendPhase();
        PublishPrepareReady();
        GetRecvCount();
        WriteToRemoteWindow();
        diagWriter_.RunPosRecord(MOE_EP_DISPATCH_RUN_POS_URMA_REQUESTS_ISSUE_DONE);
    }
}

#endif

} // namespace MoeEpDispatchImpl

#endif // MOE_EP_DISPATCH_H
