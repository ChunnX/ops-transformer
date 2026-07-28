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
 * \file moe_ep_dispatch_hybrid.h
 * \brief
 */

#ifndef MOE_EP_DISPATCH_HYBRID_H
#define MOE_EP_DISPATCH_HYBRID_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "kernel_tiling/kernel_tiling.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/reduce.h"
#include "adv_api/reduce/sum.h"
#include "adv_api/hcomm/hcomm.h"
#include "moe_ep_dispatch_tiling.h"
#include "moe_ep_dispatch_base.h"

#if __has_include("../common/moe_distribute_base.h")
#include "../common/moe_distribute_base.h"
#include "../common/mc2_kernel_utils.h"
#include "../common/mc2_moe_context.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#include "../../common/op_kernel/mc2_moe_context.h"
#endif

namespace MoeEpDispatchHybridImpl {

#define TemplateMoeEpDispatchHybridTypeClass                                                                           \
    typename XType, typename ScalesType, bool DoCpuSync, bool IsCached, bool IsTopkWeights, uint8_t NetworkMode
#define TemplateMoeEpDispatchHybridTypeFunc XType, ScalesType, DoCpuSync, IsCached, IsTopkWeights, NetworkMode

using namespace AscendC;
using namespace Mc2Kernel;

constexpr uint32_t UB_ALIGN = 32U;
constexpr uint32_t WIN_ADDR_ALIGN = 512U;
constexpr uint32_t ALIGNED_LEN_256 = 256U;
constexpr uint32_t TOPK_INFO_SIZE = 4U; // sizeof(int32_t)=sizeof(float)=4B
constexpr uint32_t UB_STRIDE = 8U;      // UB_ALIGN/sizeof(int32_t)=8
constexpr uint32_t INT64_UB_STRIDE = 4U; // UB_ALIGN/sizeof(int64_t)=4
constexpr int32_t BITS_PER_BYTE = 8;
constexpr uint32_t HCOMM_INIT_SIZE = 512U;
constexpr uint32_t PER_GROUP_SIZE = 40 * 1024U; // token 分核后 40KB 足够，并为并存 counter 预留 UB
constexpr uint32_t EXPERT_NUM_PER_GROUP = 256U; // GetExpertFreq 单次最多256个bin
constexpr uint32_t DATA_BLOCK_NUM = 8U;         // 256B/32B=8
constexpr uint8_t BUFFER_NUM = 2;
constexpr int64_t STATUS_POLL_BACKOFF_CYCLES = 100; // ready 未就绪时的轮询退避周期
constexpr int64_t HANG_TRACE_TIMEOUT_CYCLES = 1000000000; // 仅在异常长时间轮询时输出一次诊断

template <TemplateMoeEpDispatchHybridTypeClass>
class MoeEpDispatchHybrid {
public:
    __aicore__ inline MoeEpDispatchHybrid(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights, GM_ADDR scales,
                                GM_ADDR cachedSlotIdx, GM_ADDR cachedRouteCount, GM_ADDR cachedRouteDstScaleout,
                                GM_ADDR cachedRouteScaleoutSlot, GM_ADDR cachedRouteScaleupMask,
                                GM_ADDR cachedRouteScaleupSlot, GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert,
                                GM_ADDR dstBufferSlotIdx, GM_ADDR routeCount, GM_ADDR routeDstScaleout,
                                GM_ADDR routeScaleoutSlot, GM_ADDR routeScaleupMask, GM_ADDR routeScaleupSlot,
                                GM_ADDR workspaceGM, TPipe *pipe, const MoeEpDispatchTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void PrepareDispatchPayloads();
    __aicore__ inline void TransferDispatchPayloads();
    __aicore__ inline void InitTilingFields(const MoeEpDispatchTilingData *tilingData);
    __aicore__ inline void InitAlignmentFields(const MoeEpDispatchTilingData *tilingData);
    __aicore__ inline void InitGlobalTensors(GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights, GM_ADDR scales,
                                             GM_ADDR cachedSlotIdx, GM_ADDR cachedRouteCount,
                                             GM_ADDR cachedRouteDstScaleout, GM_ADDR cachedRouteScaleoutSlot,
                                             GM_ADDR cachedRouteScaleupMask, GM_ADDR cachedRouteScaleupSlot,
                                             GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert,
                                             GM_ADDR dstBufferSlotIdx, GM_ADDR routeCount, GM_ADDR routeDstScaleout,
                                             GM_ADDR routeScaleoutSlot, GM_ADDR routeScaleupMask,
                                             GM_ADDR routeScaleupSlot);
    __aicore__ inline void InitCopyParams();
    __aicore__ inline void InitGlobalAddresses(const MoeEpDispatchTilingData *tilingData);
    __aicore__ inline void CalSendCntPerRank(LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt);
    __aicore__ inline void CalSendCntPerExpert(LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt);
    __aicore__ inline void CalSendCntPerScaleout(LocalTensor<int16_t> rankIdsTensor, uint32_t calCnt,
                                                 LocalTensor<int32_t> scaleoutCounterTensor);
    __aicore__ inline void CalSendCnt();
    __aicore__ inline LocalTensor<int32_t> InitSendCountBuffers();
    __aicore__ inline void ResetSendCountWorkspace();
    __aicore__ inline void FlushSendCountToWorkspace(LocalTensor<int32_t> scaleoutCounterTensor);
    __aicore__ inline void ExchangeCount();
    __aicore__ inline void SendCountToRemoteRank(uint32_t dstRankId);
    __aicore__ inline void CopyLocalCountToWindow(uint32_t dstRankId, uint64_t notifyVal);
    __aicore__ inline void GetRecvCount();
    __aicore__ inline void SetRecvNumPerExpert();
    __aicore__ inline void SetRecvNumPerRank(LocalTensor<int32_t> recvTmpTensor);
    __aicore__ inline uint32_t DedupAndStage(uint32_t tokenId);
    __aicore__ inline void GetSlotStartNum();
    __aicore__ inline void ReduceCounterRange(GlobalTensor<int32_t> counterGMTensor, LocalTensor<int32_t> dstTensor,
                                              uint32_t counterAlign512, uint32_t counterCnt,
                                              uint32_t counterValueCount, uint32_t counterCoreCount);
    __aicore__ inline void ReduceCounterPrefix(GlobalTensor<int32_t> counterGMTensor, LocalTensor<int32_t> dstTensor,
                                               uint32_t counterAlign512, uint32_t counterCnt,
                                               uint32_t counterValueCount);
    __aicore__ inline uint32_t FindOrCreateRemoteRouteEntry(uint32_t tokenId, uint32_t dstScaleoutIndex,
                                                            uint32_t &routeEntryCount, bool &isNewRouteEntry);
    __aicore__ inline void UpdateRemoteRouteEntryTarget(uint32_t routeEntryIndex, uint32_t dstScaleupIndex,
                                                        uint32_t scaleupSlot);
    __aicore__ inline void WriteRouteTableToGM(uint32_t tokenId, uint32_t routeEntryCount);
    __aicore__ inline uint32_t CopyCachedRouteForToken(uint32_t tokenId);
    __aicore__ inline void AllocateScaleupSlotForExpert(uint32_t topkIndex, uint32_t dstRankId,
                                                        uint32_t &scaleupSlot);
    __aicore__ inline void UpdateScaleoutRouteForExpert(uint32_t tokenId, uint32_t dstRankId, uint32_t scaleupSlot,
                                                        uint32_t &routeEntryCount);
    __aicore__ inline GM_ADDR GetScaleoutSlotAddr(GM_ADDR baseAddr, uint32_t scaleoutIndex, uint32_t slot);
    __aicore__ inline uint32_t GetCurrentScaleupIndex();
    __aicore__ inline uint32_t GetProxyRankId(uint32_t dstScaleoutIndex);
    __aicore__ inline void WriteScaleoutPayloadFromRoute(uint32_t routeEntryCount);
    __aicore__ inline bool InitScaleupSendRange();
    __aicore__ inline void WriteScaleupSlotsToRank(uint32_t dstRankId, LocalTensor<int32_t> sendNumPerRankTensor);
    __aicore__ inline void SendScaleoutPayloadsToProxy();
    __aicore__ inline void ReduceScaleoutCounterGroup(uint32_t counterGroup,
                                                      LocalTensor<int32_t> counterSumTensor);
    __aicore__ inline void WriteScaleoutNotifyToWorkspace(uint32_t dstScaleoutIndex, uint32_t scaleoutSlotCount);
    __aicore__ inline void SendScaleoutPayloadToProxy(uint32_t dstScaleoutIndex, uint32_t proxyRankId,
                                                      uint32_t scaleoutSlotCount);
    __aicore__ inline void FanoutReceivedScaleoutPayloads();
    __aicore__ inline void DistributeScaleoutPayloads(uint32_t srcScaleoutIndex, uint32_t srcRankId,
                                                      uint32_t scaleoutSlotCount);
    __aicore__ inline void DistributeScaleoutPayload(GM_ADDR scaleoutSlotAddr, uint32_t srcRankId,
                                                     LocalTensor<int32_t> fanoutCountTensor);
    __aicore__ inline void WriteFanoutPayloadToDestination(LocalTensor<XType> payloadTensor, uint32_t srcRankId,
                                                           uint32_t dstRankId, uint32_t dstScaleupIndex,
                                                           uint32_t scaleupSlot);
    __aicore__ inline void WriteFanoutCounts(LocalTensor<int32_t> fanoutCountTensor);
    __aicore__ inline void ReduceFanoutCounts();
    __aicore__ inline void SendStagedFanoutPayloads(uint32_t srcRankId, uint32_t startScaleupIndex,
                                                    uint32_t endScaleupIndex);
    __aicore__ inline void SendStagedFanoutPayloadsToRank(uint32_t srcRankId, uint32_t dstRankId,
                                                          uint32_t fanoutSlotCount);
    __aicore__ inline void WriteFinalReadyStatusToRank(uint32_t srcRankId, uint32_t dstRankId);
    __aicore__ inline uint32_t WaitScaleoutPayloadReady(GM_ADDR statusAddr);
    __aicore__ inline void PollBackoff();
    __aicore__ inline void ClearScaleoutReceiveStatus(GM_ADDR localScaleoutStatusBase, uint32_t statusOffset);
    __aicore__ inline void WriteSlotToLocal(uint32_t dstRankId, uint32_t slot);
    __aicore__ inline void WriteToRemoteWindow();
    __aicore__ inline void InitPreparePayloadsBuffers();
    __aicore__ inline void InitRouteStateTensors(uint32_t slotIndexBytes, uint32_t rankSlotBytes,
                                                 uint32_t routeStateBytes, uint32_t routeCountOffset);
    __aicore__ inline void InitRouteStateForToken(uint32_t tokenId);
    __aicore__ inline void PrepareTokenPayload(uint32_t tokenId, uint32_t topkOffset);
    __aicore__ inline uint32_t ProcessCachedTokenRoutes(uint32_t tokenId, uint32_t topkOffset);
    __aicore__ inline void PreparePayloads();
    __aicore__ inline bool IsActiveRank(uint32_t rankId);
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startId, uint32_t &endId,
                                       uint32_t &sendNum);
    __aicore__ inline void SplitTokensToPayloadWorkers(uint32_t tokenCount, uint32_t &startTokenId,
                                                       uint32_t &endTokenId, uint32_t &tokenCountPerWorker);

    TPipe *tpipe_{nullptr};
    __gm__ Mc2Aclnn::MoeCommContext *mc2Context_{nullptr};
    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_; // 通信上下文

    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<int32_t> topkIdxGMTensor_;
    GlobalTensor<float> topkWeightsGMTensor_;
    GlobalTensor<int32_t> dstSlotIdxGMTensor_;
    GlobalTensor<int32_t> numRecvPerRankGMTensor_;
    GlobalTensor<int64_t> numRecvPerExpertGMTensor_;
    GlobalTensor<int32_t> cachedSlotIdxGMTensor_;
    GlobalTensor<int32_t> cachedRouteCountGMTensor_;
    GlobalTensor<int32_t> cachedRouteDstScaleoutGMTensor_;
    GlobalTensor<int32_t> cachedRouteScaleoutSlotGMTensor_;
    GlobalTensor<int32_t> cachedRouteScaleupMaskGMTensor_;
    GlobalTensor<int32_t> cachedRouteScaleupSlotGMTensor_;
    GlobalTensor<int32_t> routeCountGMTensor_;
    GlobalTensor<int32_t> routeDstScaleoutGMTensor_;
    GlobalTensor<int32_t> routeScaleoutSlotGMTensor_;
    GlobalTensor<int32_t> routeScaleupMaskGMTensor_;
    GlobalTensor<int32_t> routeScaleupSlotGMTensor_;
    GlobalTensor<int32_t> scaleupCounterGMTensor_;
    GlobalTensor<int32_t> scaleoutCounterGMTensor_;
    GlobalTensor<int32_t> recvCounterGMTensor_;
    GlobalTensor<int32_t> sendCntGMTensor_;
    GlobalTensor<ScalesType> scalesGMTensor_;

    LocalTensor<XType> xLocalTensor_;
    LocalTensor<XType> tokenSlotTensor_;
    LocalTensor<int32_t> topkIdxTensor_;
    LocalTensor<int32_t> dstSlotIdxTensor_;
    LocalTensor<int32_t> metaLocalTensor_;
    LocalTensor<int32_t> numRecvPerRankTensor_;
    LocalTensor<int64_t> numRecvPerExpertTensor_;
    LocalTensor<int32_t> sendCntPerRankTensor_;
    LocalTensor<int32_t> sendCntPerExpertTensor_;
    LocalTensor<uint8_t> hcommTensor_;
    // PreparePayloads 使用的 UB tensor 切片，在 PreparePayloads 初始化时从 numRecvPerRankBuf_ 切出
    LocalTensor<int32_t> slotIdxPerRankTensor_;
    LocalTensor<int32_t> counterSumTensor_; // prefix/fanout 归约结果，发送阶段与 rank slot 映射复用 UB
    LocalTensor<int32_t> sendRankSlotTensor_;
    LocalTensor<int32_t> routeScaleoutIndexTensor_;
    LocalTensor<int32_t> routeScaleoutSlotTensor_;
    LocalTensor<int32_t> routeScaleupMaskTensor_;
    LocalTensor<int32_t> routeScaleupSlotTensor_;
    LocalTensor<int32_t> sendRouteIndexByScaleoutTensor_; // 发送阶段记录 scaleout 到 route entry 的映射
    LocalTensor<int32_t> routeCountTensor_; // route 表项数的固定 UB 搬出块，避免 GM DCache 伪共享

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> perSlotQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> dstSlotQueue_;
    TBuf<> topkIdsBuf_;
    TBuf<> tempBuf_;
    TBuf<> dstExpBuf_;
    TBuf<> hcommBuf_; // 通信
    TBuf<> numRecvBuf_;
    TBuf<> numRecvPerRankBuf_;
    TBuf<> numRecvPerExpertBuf_;
    TBuf<> recvCntBuf_;
    TBuf<> recvTempBuf_;

    GM_ADDR workspaceGM_{nullptr};
    GM_ADDR hostPinnedCounterAddrGM_{nullptr};
    GM_ADDR scaleupCounterAddr_{nullptr};
    GM_ADDR scaleoutNotifyWorkspaceAddr_{nullptr};
    GM_ADDR sendCntWorkspaceAddr_{nullptr};
    GM_ADDR scaleupSendWinAddr_{nullptr};
    GM_ADDR scaleoutSendWinAddr_{nullptr};
    GM_ADDR localWinAddr_{nullptr};

    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t epWorldSize_{0};
    uint32_t moeExpertNumPerRank_{0};
    uint32_t axisMaxBS_{0};
    uint32_t scalesBytes_{0};
    uint32_t perSlotBytes_{0};
    uint32_t scaleoutSlotBytes_{0};
    uint32_t moeExpertNum_{0};
    uint32_t numScaleoutRanks_{0};
    uint32_t aivNum_{0};
    uint32_t epRankId_{0};
    uint32_t aivId_{0};
    uint32_t rankSizePerServer_{0};
    uint32_t currentServerIndex_{0};
    uint32_t serverStartRank_{0};
    uint32_t serverEndRank_{0};
    uint32_t startTokenId_{0};
    uint32_t endTokenId_{0};
    uint32_t sendTokenNum_{0};
    uint32_t startRankId_{0};
    uint32_t endRankId_{0};
    uint32_t rankNumPerCore_{0};
    uint32_t hAlignSize_{0};
    uint32_t kAlignSize_{0};
    uint32_t axisKAlign_{0};
    uint32_t scaleoutMaskWordCount_{0};
    uint32_t routeScaleupSlotEntryElements_{0}; // routeScaleupSlot 每个 route 表项的 32B 对齐 stride
    uint32_t routeScaleupMaskEntryElements_{0}; // routeScaleupMask 每个 route 表项的 32B 对齐 stride
    uint32_t routeScaleupSlotBytes_{0};          // routeScaleupSlot 对齐后的 UB 缓存大小
    uint32_t routeScaleupMaskBytes_{0};          // routeScaleupMask 对齐后的 UB 缓存大小
    uint32_t numScaleoutRanksAlign_{0};  // scaleout rank 数对齐到 UB_ALIGN
    uint32_t perGroupSizeAlign_{0};      // 每组 token 数 * axisK * sizeof(int16_t) 对齐到 256
    uint32_t perGroupTokenNum_{0};       // 每组 token 数
    uint32_t epWorldSizeAlign_{0};
    uint32_t epWorldSizeAlign512_{0};
    uint32_t moeNumPerRankAlign_{0};
    uint32_t moeExpertNumAlign_{0};
    uint32_t moeNumPerRankAlign512_{0};
    uint32_t cntPerRankSizeAlign512_{0};
    uint32_t counterCnt_{0};
    uint32_t counterAlign512_{0};
    uint32_t fanoutCounterCnt_{0}; // fanout 只统计本 server 内 rank，按 32B 对齐
    uint32_t scaleoutCounterCnt_{0};
    uint32_t scaleoutCounterAlign512_{0};
    uint32_t metaOffset_{0};
    uint64_t cntWinStateOffset_{0};
    uint64_t slotWinStateOffset_{0};
    uint64_t winDataOffset_{0};
    uint64_t scaleoutRecvDataOffset_{0};
    uint64_t scaleoutRecvStatusOffset_{0};

    DataCopyParams statusCopyParams_;
    DataCopyParams clearStatusCopyParams_;
    DataCopyParams topkCopyParams_;
    DataCopyParams xCopyParams_;      // PreparePayloads 复制单 token x payload
    DataCopyParams scalesCopyParams_; // PreparePayloads 复制 fp8 scales payload
    DataCopyParams fanoutPayloadCopyParams_; // fanout 搬运完整 slot payload
    DataCopyExtParams routeCountCopyParams_;
    DataCopyExtParams routeCopyParams_;
    DataCopyExtParams routeMaskTableCopyParams_;
    DataCopyExtParams routeSlotTableCopyParams_;
    DataCopyExtParams routeMaskEntryCopyParams_;
    DataCopyExtParams routeSlotEntryCopyParams_;
    DataCopyPadParams padParams_;
    DataCopyPadExtParams<int32_t> routeCopyPadParams_{true, 0U, 0U, 0};
};

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights, GM_ADDR scales, GM_ADDR cachedSlotIdx,
    GM_ADDR cachedRouteCount, GM_ADDR cachedRouteDstScaleout, GM_ADDR cachedRouteScaleoutSlot,
    GM_ADDR cachedRouteScaleupMask, GM_ADDR cachedRouteScaleupSlot, GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert,
    GM_ADDR dstBufferSlotIdx, GM_ADDR routeCount, GM_ADDR routeDstScaleout, GM_ADDR routeScaleoutSlot,
    GM_ADDR routeScaleupMask, GM_ADDR routeScaleupSlot, GM_ADDR workspaceGM, TPipe *pipe,
    const MoeEpDispatchTilingData *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    workspaceGM_ = workspaceGM;
    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::MoeCommContext *>(context);
    epRankId_ = mc2Context_->epRankId;

    InitTilingFields(tilingData);
    InitGlobalTensors(x, topkIdx, topkWeights, scales, cachedSlotIdx, cachedRouteCount, cachedRouteDstScaleout,
                      cachedRouteScaleoutSlot, cachedRouteScaleupMask, cachedRouteScaleupSlot, numRecvPerRank,
                      numRecvPerExpert, dstBufferSlotIdx, routeCount, routeDstScaleout, routeScaleoutSlot,
                      routeScaleupMask, routeScaleupSlot);
    InitCopyParams();
    InitGlobalAddresses(tilingData);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::InitTilingFields(
    const MoeEpDispatchTilingData *tilingData)
{
    const auto &info = tilingData->moeEpDispatchInfo;
    axisBS_ = info.cfg.numTokens;
    axisH_ = info.cfg.hidden;
    axisK_ = info.cfg.topK;
    epWorldSize_ = info.cfg.epWorldSize;
    moeExpertNumPerRank_ = info.cfg.numLocalExperts;
    axisMaxBS_ = info.cfg.numMaxTokensPerRank;
    scalesBytes_ = info.cfg.scalesBytes;
    perSlotBytes_ = info.cfg.perSlotBytes;
    scaleoutSlotBytes_ = info.scaleoutSlotAlignedBytes;
    numScaleoutRanks_ = info.numScaleoutRanks;
    aivNum_ = info.aivNum;
    rankSizePerServer_ = info.rankSizePerServer;
    currentServerIndex_ = MoeEpDispatchBase::GetCurrentServerIndex(epRankId_, rankSizePerServer_);
    serverStartRank_ = MoeEpDispatchBase::GetServerStartRank(epRankId_, rankSizePerServer_);
    serverEndRank_ = MoeEpDispatchBase::GetServerEndRank(epRankId_, rankSizePerServer_, epWorldSize_);
    cntWinStateOffset_ = info.cntWinStateOffset;
    slotWinStateOffset_ = info.slotWinStateOffset;
    winDataOffset_ = info.winDataOffset;
    scaleoutRecvDataOffset_ = info.scaleoutRecvDataOffset;
    scaleoutRecvStatusOffset_ = info.scaleoutRecvStatusOffset;
    hostPinnedCounterAddrGM_ = (GM_ADDR)info.hostPinnedCounterAddr;
    moeExpertNum_ = moeExpertNumPerRank_ * epWorldSize_;

    InitAlignmentFields(tilingData);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::InitAlignmentFields(
    const MoeEpDispatchTilingData *tilingData)
{
    const auto &info = tilingData->moeEpDispatchInfo;
    hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;
    kAlignSize_ = Ceil(axisK_ * TOPK_INFO_SIZE, UB_ALIGN) * UB_ALIGN;
    axisKAlign_ = kAlignSize_ / TOPK_INFO_SIZE;
    scaleoutMaskWordCount_ = info.scaleoutMaskWordCount;
    routeScaleupSlotEntryElements_ =
        Ceil(rankSizePerServer_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN / sizeof(int32_t);
    routeScaleupMaskEntryElements_ =
        Ceil(scaleoutMaskWordCount_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN / sizeof(int32_t);
    routeScaleupSlotBytes_ = axisKAlign_ * routeScaleupSlotEntryElements_ * sizeof(int32_t);
    routeScaleupMaskBytes_ = axisKAlign_ * routeScaleupMaskEntryElements_ * sizeof(int32_t);
    numScaleoutRanksAlign_ = Ceil(numScaleoutRanks_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN / sizeof(int32_t);
    metaOffset_ = hAlignSize_;
    epWorldSizeAlign_ = Ceil(epWorldSize_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    counterCnt_ = epWorldSizeAlign_ / sizeof(int32_t);
    fanoutCounterCnt_ = Ceil(rankSizePerServer_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN / sizeof(int32_t);
    scaleoutCounterCnt_ = Ceil(numScaleoutRanks_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN / sizeof(int32_t);
    // GM workspace perRankCount 按 512B/rank 对齐, 与 direct 一致
    epWorldSizeAlign512_ = Ceil(epWorldSize_ * sizeof(int32_t), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    counterAlign512_ = epWorldSizeAlign512_ / sizeof(int32_t);
    scaleoutCounterAlign512_ =
        Ceil(numScaleoutRanks_ * sizeof(int32_t), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN / sizeof(int32_t);
    // GM workspace perRankCount: 每 rank 512B (state + dstRankRecvNum), 与 direct 一致
    cntPerRankSizeAlign512_ = epWorldSize_ * WIN_ADDR_ALIGN;
    moeNumPerRankAlign_ = Ceil(moeExpertNumPerRank_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    moeExpertNumAlign_ = Ceil(moeExpertNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    moeNumPerRankAlign512_ = Ceil(moeExpertNumPerRank_ * sizeof(int32_t), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    perGroupTokenNum_ = PER_GROUP_SIZE / sizeof(int16_t) / axisK_;
    perGroupSizeAlign_ = Ceil(perGroupTokenNum_ * axisK_ * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::InitGlobalTensors(
    GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights, GM_ADDR scales, GM_ADDR cachedSlotIdx, GM_ADDR cachedRouteCount,
    GM_ADDR cachedRouteDstScaleout, GM_ADDR cachedRouteScaleoutSlot, GM_ADDR cachedRouteScaleupMask,
    GM_ADDR cachedRouteScaleupSlot, GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert, GM_ADDR dstBufferSlotIdx,
    GM_ADDR routeCount, GM_ADDR routeDstScaleout, GM_ADDR routeScaleoutSlot, GM_ADDR routeScaleupMask,
    GM_ADDR routeScaleupSlot)
{
    xGMTensor_.SetGlobalBuffer((__gm__ XType *)x);
    topkIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)topkIdx);
    if constexpr (IsTopkWeights) {
        topkWeightsGMTensor_.SetGlobalBuffer((__gm__ float *)topkWeights);
    }
    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
        scalesGMTensor_.SetGlobalBuffer((__gm__ ScalesType *)scales);
        metaOffset_ += Ceil(scalesBytes_, UB_ALIGN) * UB_ALIGN;
    }
    if constexpr (IsCached) {
        cachedSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)cachedSlotIdx);
        cachedRouteCountGMTensor_.SetGlobalBuffer((__gm__ int32_t *)cachedRouteCount);
        cachedRouteDstScaleoutGMTensor_.SetGlobalBuffer((__gm__ int32_t *)cachedRouteDstScaleout);
        cachedRouteScaleoutSlotGMTensor_.SetGlobalBuffer((__gm__ int32_t *)cachedRouteScaleoutSlot);
        cachedRouteScaleupMaskGMTensor_.SetGlobalBuffer((__gm__ int32_t *)cachedRouteScaleupMask);
        cachedRouteScaleupSlotGMTensor_.SetGlobalBuffer((__gm__ int32_t *)cachedRouteScaleupSlot);
    }
    numRecvPerRankGMTensor_.SetGlobalBuffer((__gm__ int32_t *)numRecvPerRank);
    numRecvPerExpertGMTensor_.SetGlobalBuffer((__gm__ int64_t *)numRecvPerExpert);
    dstSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)dstBufferSlotIdx);
    routeCountGMTensor_.SetGlobalBuffer((__gm__ int32_t *)routeCount);
    routeDstScaleoutGMTensor_.SetGlobalBuffer((__gm__ int32_t *)routeDstScaleout);
    routeScaleoutSlotGMTensor_.SetGlobalBuffer((__gm__ int32_t *)routeScaleoutSlot);
    routeScaleupMaskGMTensor_.SetGlobalBuffer((__gm__ int32_t *)routeScaleupMask);
    routeScaleupSlotGMTensor_.SetGlobalBuffer((__gm__ int32_t *)routeScaleupSlot);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::InitCopyParams()
{
    tpipe_->InitBuffer(numRecvPerRankBuf_, epWorldSizeAlign_);
    numRecvPerRankTensor_ = numRecvPerRankBuf_.Get<int32_t>();

    statusCopyParams_ = {static_cast<uint16_t>(epWorldSize_), 1U,
                         static_cast<uint16_t>((WIN_ADDR_ALIGN - UB_ALIGN) / UB_ALIGN), 0U};
    clearStatusCopyParams_ = {static_cast<uint16_t>(epWorldSize_), 1U, 0U,
                              static_cast<uint16_t>((WIN_ADDR_ALIGN - UB_ALIGN) / UB_ALIGN)};
    topkCopyParams_ = {1U, static_cast<uint16_t>(axisK_ * TOPK_INFO_SIZE), 0U, 0U};
    xCopyParams_ = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    scalesCopyParams_ = {1U, static_cast<uint16_t>(scalesBytes_), 0U, 0U};
    fanoutPayloadCopyParams_ = {1U, static_cast<uint16_t>(perSlotBytes_), 0U, 0U};
    routeCountCopyParams_ = {1U, static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
    routeCopyParams_ = {1U, static_cast<uint32_t>(axisK_ * sizeof(int32_t)), 0U, 0U, 0U};
    // 每个 route entry 单独搬运，UB 侧按 32B 步进，GM 中的 handle 仍保持紧凑格式。
    routeMaskTableCopyParams_ = {
        static_cast<uint16_t>(axisK_), static_cast<uint32_t>(scaleoutMaskWordCount_ * sizeof(int32_t)), 0U, 0U, 0U};
    routeSlotTableCopyParams_ = {
        static_cast<uint16_t>(axisK_), static_cast<uint32_t>(rankSizePerServer_ * sizeof(int32_t)), 0U, 0U, 0U};
    routeMaskEntryCopyParams_ = {
        1U, static_cast<uint32_t>(scaleoutMaskWordCount_ * sizeof(int32_t)), 0U, 0U, 0U};
    routeSlotEntryCopyParams_ = {
        1U, static_cast<uint32_t>(rankSizePerServer_ * sizeof(int32_t)), 0U, 0U, 0U};
    padParams_ = {true, 0, 0, 0};
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::InitGlobalAddresses(
    const MoeEpDispatchTilingData *tilingData)
{
    const auto &info = tilingData->moeEpDispatchInfo;
    scaleupCounterAddr_ = workspaceGM_;
    sendCntWorkspaceAddr_ = scaleupCounterAddr_ + aivNum_ * epWorldSizeAlign512_;
    GM_ADDR scaleoutCounterAddr = workspaceGM_ + info.routeWorkspaceOffset;
    uint64_t scaleoutCounterWorkspaceBytes =
        static_cast<uint64_t>(aivNum_) * Ceil(numScaleoutRanks_ * sizeof(int32_t), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    scaleoutNotifyWorkspaceAddr_ = scaleoutCounterAddr + scaleoutCounterWorkspaceBytes;
    scaleupSendWinAddr_ =
        MoeEpDispatchBase::GetWindowAddrByRankId(mc2Context_, epRankId_, info.scaleupSendWinOffset);
    scaleoutSendWinAddr_ =
        MoeEpDispatchBase::GetWindowAddrByRankId(mc2Context_, epRankId_, info.scaleoutSendWinOffset);
    GM_ADDR localCntStateWinAddr =
        MoeEpDispatchBase::GetWindowAddrByRankId(mc2Context_, epRankId_, cntWinStateOffset_);
    localWinAddr_ =
        MoeEpDispatchBase::GetWindowAddrByRankId(mc2Context_, epRankId_, info.scaleupFinalRecvDataOffset);
    scaleupCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t *)scaleupCounterAddr_);
    scaleoutCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t *)scaleoutCounterAddr);
    recvCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t *)localCntStateWinAddr);
    sendCntGMTensor_.SetGlobalBuffer((__gm__ int32_t *)sendCntWorkspaceAddr_);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline bool MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::IsActiveRank(uint32_t rankId)
{
    return rankId >= serverStartRank_ && rankId < serverEndRank_;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startId, uint32_t &endId, uint32_t &sendNum)
{
    sendNum = curSendCnt / curUseAivNum;               // 每个aiv需要发送的数量
    uint32_t remainderNum = curSendCnt % curUseAivNum; // 余数
    startId = sendNum * aivId_;  // 每个aiv发送时的起始rankid
    if (aivId_ < remainderNum) { // 前remainderRankNum个aiv需要多发1个卡的数据
        sendNum++;
        startId += aivId_;
    } else {
        startId += remainderNum;
    }
    endId = startId + sendNum;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SplitTokensToPayloadWorkers(
    uint32_t tokenCount, uint32_t &startTokenId, uint32_t &endTokenId, uint32_t &tokenCountPerWorker)
{
    if (aivNum_ <= 1U) {
        SplitToCore(tokenCount, aivNum_, startTokenId, endTokenId, tokenCountPerWorker);
        return;
    }
    if (aivId_ == 0U) {
        startTokenId = 0U;
        endTokenId = 0U;
        tokenCountPerWorker = 0U;
        return;
    }

    // AIV0负责等待接收count，其余AIV均分token，避免等待完成后再串行构造AIV0的payload。
    uint32_t payloadWorkerCount = aivNum_ - 1U;
    uint32_t payloadWorkerIndex = aivId_ - 1U;
    tokenCountPerWorker = tokenCount / payloadWorkerCount;
    uint32_t remainderTokenCount = tokenCount % payloadWorkerCount;
    startTokenId = tokenCountPerWorker * payloadWorkerIndex;
    if (payloadWorkerIndex < remainderTokenCount) {
        tokenCountPerWorker++;
        startTokenId += payloadWorkerIndex;
    } else {
        startTokenId += remainderTokenCount;
    }
    endTokenId = startTokenId + tokenCountPerWorker;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::CalSendCntPerRank(
    LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt)
{
    uint32_t tmpOffset = Ceil(calCnt * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    uint32_t tokenCnt = calCnt / axisK_;
    uint32_t calCntAlign = tmpOffset / sizeof(int16_t);
    uint32_t tokenCntAlign = Ceil(tokenCnt * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int16_t);
    uint32_t mask = tokenCnt;
    uint32_t shape[2] = {tokenCnt, axisK_};
    LocalTensor<int16_t> dstTensorInt16 = dstExpBuf_.Get<int16_t>();
    LocalTensor<int16_t> tempTensorInt16 = topkIdsBuf_.GetWithOffset<int16_t>(calCntAlign, 0);
    LocalTensor<uint16_t> maskTensorInt16 = topkIdsBuf_.GetWithOffset<uint16_t>(calCntAlign, tmpOffset);
    LocalTensor<uint8_t> gatherMaskTensorInt8 = maskTensorInt16.template ReinterpretCast<uint8_t>();

    Duplicate<int16_t>(dstTensorInt16, static_cast<int16_t>(moeExpertNumPerRank_), calCnt);
    Div(tempTensorInt16, expertIdsTensor, dstTensorInt16, calCnt);
    // 筛选无效expert id 消除影响
    CompareScalar(gatherMaskTensorInt8, expertIdsTensor, static_cast<int16_t>(0), AscendC::CMPMODE::GE, calCntAlign);
    Select(dstTensorInt16, gatherMaskTensorInt8, tempTensorInt16, static_cast<int16_t>(-1),
           AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, calCnt);

    for (uint32_t dstRankId = 0U; dstRankId < epWorldSize_; dstRankId++) {
        // 筛选出发送到目标卡的token
        uint64_t rsvdCnt = 0;
        Subs(expertIdsTensor, dstTensorInt16, static_cast<int16_t>(dstRankId), calCnt);
        Abs(tempTensorInt16, expertIdsTensor, calCnt);
        ReduceMin<int16_t, Pattern::Reduce::AR, true>(expertIdsTensor, tempTensorInt16, shape, false); // 0为目标
        Duplicate<uint16_t>(maskTensorInt16, 0, tokenCntAlign); // GatherMask前清0
        CompareScalar(gatherMaskTensorInt8, expertIdsTensor, static_cast<int16_t>(0), AscendC::CMPMODE::EQ,
                      tokenCntAlign);
        GatherMask(tempTensorInt16, expertIdsTensor, maskTensorInt16, true, mask, {1, 1, 0, 0}, rsvdCnt);
        SyncFunc<AscendC::HardEvent::V_S>();
        uint32_t offset = dstRankId * UB_STRIDE + 1U;
        int32_t curRankCnt = sendCntPerRankTensor_.GetValue(offset) + static_cast<int32_t>(rsvdCnt);
        sendCntPerRankTensor_.SetValue(offset, curRankCnt);
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::CalSendCntPerExpert(
    LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt)
{
    // 使用 GetExpertFreq 硬件向量化统计 expert 频率，替代逐 expert scalar 循环
    uint32_t tmpOffset = Ceil(calCnt * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    uint32_t calCntAlign = tmpOffset / sizeof(int16_t);
    LocalTensor<int16_t> tempTensorInt16 = topkIdsBuf_.GetWithOffset<int16_t>(calCntAlign, tmpOffset);
    LocalTensor<uint16_t> gatherMaskTensorU16 = topkIdsBuf_.Get<uint16_t>();
    LocalTensor<uint16_t> gatherMaskU16GE = topkIdsBuf_.GetWithOffset<uint16_t>(calCntAlign, 0);
    LocalTensor<uint16_t> gatherMaskU16LT = topkIdsBuf_.GetWithOffset<uint16_t>(calCntAlign, tmpOffset);
    LocalTensor<uint8_t> gatherMaskU8GE = gatherMaskU16GE.ReinterpretCast<uint8_t>();
    LocalTensor<uint8_t> gatherMaskU8LT = gatherMaskU16LT.ReinterpretCast<uint8_t>();
    LocalTensor<int16_t> dstTensorInt16 = dstExpBuf_.Get<int16_t>();
    LocalTensor<uint8_t> dstTensorU8 = dstExpBuf_.Get<uint8_t>();
    LocalTensor<uint32_t> freqTensorU32 = dstExpBuf_.Get<uint32_t>();
    uint32_t maskU16Cnt = Ceil(calCntAlign, BITS_PER_BYTE * sizeof(uint16_t));

    uint32_t groupNum = Ceil(moeExpertNum_, EXPERT_NUM_PER_GROUP);
    for (uint32_t group = 0; group < groupNum; group++) {
        uint32_t baseExpertId = group * EXPERT_NUM_PER_GROUP;
        uint32_t curGroupSize = (baseExpertId + EXPERT_NUM_PER_GROUP > moeExpertNum_)
                                    ? (moeExpertNum_ - baseExpertId)
                                    : EXPERT_NUM_PER_GROUP;
        uint64_t rsvdCnt = 0;
        Subs(dstTensorInt16, expertIdsTensor, static_cast<int16_t>(baseExpertId), calCnt);
        Duplicate<uint16_t>(gatherMaskTensorU16, 0, calCntAlign * 2);
        CompareScalar(gatherMaskU8GE, dstTensorInt16, static_cast<int16_t>(0), AscendC::CMPMODE::GE, calCntAlign);
        CompareScalar(gatherMaskU8LT, dstTensorInt16, static_cast<int16_t>(curGroupSize - 1),
                      AscendC::CMPMODE::LE, calCntAlign);
        And(gatherMaskU16GE, gatherMaskU16GE, gatherMaskU16LT, maskU16Cnt);
        GatherMask(tempTensorInt16, dstTensorInt16, gatherMaskU16GE, true, calCnt, {1, 1, 0, 0}, rsvdCnt);
        SyncFunc<AscendC::HardEvent::V_S>();
        if (rsvdCnt == 0) {
            continue;
        }

        Cast(dstTensorU8, tempTensorInt16, RoundMode::CAST_NONE, rsvdCnt);
        MoeEpDispatchBase::GetExpertFreq(gatherMaskTensorU16, dstTensorU8, rsvdCnt);
        Cast(freqTensorU32, gatherMaskTensorU16, RoundMode::CAST_NONE, curGroupSize);
        Add(sendCntPerExpertTensor_[baseExpertId], sendCntPerExpertTensor_[baseExpertId],
            freqTensorU32.ReinterpretCast<int32_t>(), static_cast<int32_t>(curGroupSize));
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::CalSendCntPerScaleout(
    LocalTensor<int16_t> rankIdsTensor, uint32_t calCnt, LocalTensor<int32_t> scaleoutCounterTensor)
{
    if (numScaleoutRanks_ == 0U) {
        return;
    }

    uint32_t tmpOffset = Ceil(calCnt * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    uint32_t tokenCnt = calCnt / axisK_;
    uint32_t calCntAlign = tmpOffset / sizeof(int16_t);
    uint32_t tokenCntAlign = Ceil(tokenCnt * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int16_t);
    uint32_t mask = tokenCnt;
    uint32_t shape[2] = {tokenCnt, axisK_};
    LocalTensor<int16_t> scaleoutIdsTensor = tempBuf_.Get<int16_t>();
    LocalTensor<int16_t> tempTensorInt16 = topkIdsBuf_.GetWithOffset<int16_t>(calCntAlign, 0);
    LocalTensor<uint16_t> maskTensorInt16 = topkIdsBuf_.GetWithOffset<uint16_t>(calCntAlign, tmpOffset);
    LocalTensor<uint8_t> gatherMaskTensorInt8 = maskTensorInt16.template ReinterpretCast<uint8_t>();

    Duplicate<int16_t>(scaleoutIdsTensor, static_cast<int16_t>(rankSizePerServer_), calCnt);
    Div(tempTensorInt16, rankIdsTensor, scaleoutIdsTensor, calCnt);
    CompareScalar(gatherMaskTensorInt8, rankIdsTensor, static_cast<int16_t>(0), AscendC::CMPMODE::GE, calCntAlign);
    Select(scaleoutIdsTensor, gatherMaskTensorInt8, tempTensorInt16, static_cast<int16_t>(-1),
           AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, calCnt);

    for (uint32_t dstScaleoutIndex = 0U; dstScaleoutIndex < numScaleoutRanks_; dstScaleoutIndex++) {
        if (dstScaleoutIndex == currentServerIndex_) {
            continue;
        }
        uint64_t rsvdCnt = 0;
        Subs(rankIdsTensor, scaleoutIdsTensor, static_cast<int16_t>(dstScaleoutIndex), calCnt);
        Abs(tempTensorInt16, rankIdsTensor, calCnt);
        ReduceMin<int16_t, Pattern::Reduce::AR, true>(rankIdsTensor, tempTensorInt16, shape, false);
        Duplicate<uint16_t>(maskTensorInt16, 0, tokenCntAlign);
        CompareScalar(gatherMaskTensorInt8, rankIdsTensor, static_cast<int16_t>(0), AscendC::CMPMODE::EQ,
                      tokenCntAlign);
        GatherMask(tempTensorInt16, rankIdsTensor, maskTensorInt16, true, mask, {1, 1, 0, 0}, rsvdCnt);
        SyncFunc<AscendC::HardEvent::V_S>();
        int32_t scaleoutCount =
            scaleoutCounterTensor.GetValue(dstScaleoutIndex) + static_cast<int32_t>(rsvdCnt);
        scaleoutCounterTensor.SetValue(dstScaleoutIndex, scaleoutCount);
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline LocalTensor<int32_t>
MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::InitSendCountBuffers()
{
    uint32_t sendCntRankSizeAlign =
        Ceil(epWorldSize_ * UB_STRIDE * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    uint32_t scaleoutCounterBytes = scaleoutCounterCnt_ * sizeof(int32_t);
    uint32_t sendCntBufferBytes = sendCntRankSizeAlign + moeExpertNumAlign_ + scaleoutCounterBytes;
    tpipe_->InitBuffer(topkIdsBuf_, 2 * perGroupSizeAlign_);
    tpipe_->InitBuffer(tempBuf_, perGroupSizeAlign_);
    tpipe_->InitBuffer(dstExpBuf_, perGroupSizeAlign_);
    tpipe_->InitBuffer(numRecvBuf_, sendCntBufferBytes);
    sendCntPerRankTensor_ =
        numRecvBuf_.GetWithOffset<int32_t>(sendCntRankSizeAlign / sizeof(int32_t), 0U);
    sendCntPerExpertTensor_ = numRecvBuf_.GetWithOffset<int32_t>(
        moeExpertNumAlign_ / sizeof(int32_t), sendCntRankSizeAlign);
    LocalTensor<int32_t> scaleoutCounterTensor = numRecvBuf_.GetWithOffset<int32_t>(
        scaleoutCounterCnt_, sendCntRankSizeAlign + moeExpertNumAlign_);
    Duplicate<int32_t>(sendCntPerRankTensor_, 0, sendCntRankSizeAlign / sizeof(int32_t));
    Duplicate<int32_t>(sendCntPerExpertTensor_, 0, moeExpertNumAlign_ / sizeof(int32_t));
    Duplicate<int32_t>(scaleoutCounterTensor, 0, scaleoutCounterCnt_);
    return scaleoutCounterTensor;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::ResetSendCountWorkspace()
{
    if (aivId_ == aivNum_ - 1U) {
        uint32_t expertCntOffset = cntPerRankSizeAlign512_ / sizeof(int32_t);
        DataCopyParams expertCntCopyParams = {
            1U, static_cast<uint16_t>(moeExpertNum_ * sizeof(int32_t)), 0U, 0U};
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(sendCntGMTensor_, sendCntPerRankTensor_, clearStatusCopyParams_);
        DataCopyPad(sendCntGMTensor_[expertCntOffset], sendCntPerExpertTensor_, expertCntCopyParams);
        PipeBarrier<PIPE_MTE3>(); // workspace 清零完成后，其他核才能开始 AtomicAdd
    }
    SyncAll<true>();
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::FlushSendCountToWorkspace(
    LocalTensor<int32_t> scaleoutCounterTensor)
{
    LocalTensor<uint32_t> gatherIndexTensor =
        topkIdsBuf_.GetWithOffset<uint32_t>(UB_STRIDE, epWorldSize_ * UB_ALIGN);
    SyncFunc<AscendC::HardEvent::V_S>(); // 前序 Vector 计算复用 topkIdsBuf_，完成后才能写 gather index
    gatherIndexTensor.SetValue(0, 2U); // 源操作数每个datablock取下标为1的元素
    uint32_t compactMask = 2U;
    uint64_t reservedCount = 0UL;
    GatherMaskParams gatherMaskParams = {1, static_cast<uint16_t>(epWorldSize_), 1, 0};
    DataCopyParams sendPerRankParams = {static_cast<uint16_t>(epWorldSize_), static_cast<uint16_t>(UB_ALIGN), 0U,
                                        static_cast<uint16_t>(WIN_ADDR_ALIGN - UB_ALIGN)};
    DataCopyParams sendPerExpertParams = {
        1U, static_cast<uint16_t>(moeExpertNum_ * sizeof(int32_t)), 0U, 0U};
    SyncFunc<AscendC::HardEvent::S_V>();
    GatherMask(numRecvPerRankTensor_, sendCntPerRankTensor_, gatherIndexTensor, true, compactMask, gatherMaskParams,
               reservedCount);

    SyncFunc<AscendC::HardEvent::S_MTE3>();
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    SetAtomicAdd<int32_t>();
    DataCopyPad(sendCntGMTensor_, sendCntPerRankTensor_, sendPerRankParams);
    DataCopyPad(sendCntGMTensor_[cntPerRankSizeAlign512_ / sizeof(int32_t)], sendCntPerExpertTensor_,
                sendPerExpertParams);
    SetAtomicNone();
    DataCopy(scaleupCounterGMTensor_[aivId_ * counterAlign512_], numRecvPerRankTensor_, counterCnt_);
    if (numScaleoutRanks_ > 0U) {
        DataCopy(scaleoutCounterGMTensor_[aivId_ * scaleoutCounterAlign512_], scaleoutCounterTensor,
                 scaleoutCounterCnt_);
    }
    PipeBarrier<PIPE_MTE3>(); // count 和 route counter 发布完成后，其他核才可读取
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::CalSendCnt()
{
    SplitTokensToPayloadWorkers(axisBS_, startTokenId_, endTokenId_, sendTokenNum_);
    uint32_t groupCnt = Ceil(sendTokenNum_, perGroupTokenNum_);
    uint32_t calCnt = perGroupTokenNum_ * axisK_;
    LocalTensor<int32_t> scaleoutCounterTensor = InitSendCountBuffers();
    DataCopyPadExtParams<int32_t> topkIdsCntCopyPadParams{false, 0U, 0U, 0U};

    LocalTensor<int32_t> topkIdsGroupTensor = topkIdsBuf_.Get<int32_t>();
    LocalTensor<int16_t> tempTensorInt16 = tempBuf_.Get<int16_t>();

    ResetSendCountWorkspace();
    if (aivId_ == 0U) { // 状态位仅做1次累加
        uint64_t mask[2] = {0x101010101010101, 0}; // 一次性操作256字节，也是8个datablock，每8个数将首个设置为1
        Duplicate<int32_t>(sendCntPerRankTensor_, 1, mask, Ceil(epWorldSize_, DATA_BLOCK_NUM), 1, DATA_BLOCK_NUM);
    }

    for (uint32_t group = 0; group < groupCnt; group++) {
        if (group == groupCnt - 1) {
            calCnt = (sendTokenNum_ - group * perGroupTokenNum_) * axisK_;
        }
        if (group > 0) {
            SyncFunc<AscendC::HardEvent::V_MTE2>();
        }
        uint32_t topkIdxOffset = (startTokenId_ + group * perGroupTokenNum_) * axisK_;
        DataCopyExtParams topkIdsCntParams = {1U, static_cast<uint32_t>(calCnt * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(topkIdsGroupTensor, topkIdxGMTensor_[topkIdxOffset], topkIdsCntParams,
                    topkIdsCntCopyPadParams); // copy topkId
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Cast(tempTensorInt16, topkIdsGroupTensor, RoundMode::CAST_NONE, calCnt);

        // 每核只扫描自己的 token 分片，并在同一批 topk 上生成 count 和两级 route counter。
        CalSendCntPerExpert(tempTensorInt16, calCnt);
        CalSendCntPerRank(tempTensorInt16, calCnt);
        CalSendCntPerScaleout(dstExpBuf_.Get<int16_t>(), calCnt, scaleoutCounterTensor);
    }

    FlushSendCountToWorkspace(scaleoutCounterTensor);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SendCountToRemoteRank(
    uint32_t dstRankId)
{
    // 计算目标窗口地址:
    GM_ADDR remoteStateAddr =
        MoeEpDispatchBase::GetWindowAddrByRankId(mc2Context_, dstRankId, cntWinStateOffset_);
    GM_ADDR notifyAddr = remoteStateAddr + epRankId_ * WIN_ADDR_ALIGN;
    GM_ADDR remoteCountAddr = remoteStateAddr + epWorldSize_ * WIN_ADDR_ALIGN + epRankId_ * moeNumPerRankAlign512_;
    GM_ADDR srcWorkspaceAddr =
        sendCntWorkspaceAddr_ + cntPerRankSizeAlign512_ + dstRankId * moeExpertNumPerRank_ * sizeof(int32_t);
    uint64_t commHandle = MoeEpDispatchBase::GetCommHandle(mc2Context_, epRankId_, dstRankId);

    // 远端 使用URMA发送 count + state
    hcomm_.WriteNbi(commHandle, remoteCountAddr, srcWorkspaceAddr,
                    static_cast<uint64_t>(moeExpertNumPerRank_ * sizeof(int32_t)));
    hcomm_.Drain(commHandle);
    hcomm_.WriteNbi(commHandle, notifyAddr, sendCntWorkspaceAddr_ + dstRankId * WIN_ADDR_ALIGN, sizeof(uint64_t));
    hcomm_.Drain(commHandle);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::CopyLocalCountToWindow(
    uint32_t dstRankId, uint64_t notifyVal)
{
    // 本端
    // 计算目标窗口地址:
    GM_ADDR remoteStateAddr =
        MoeEpDispatchBase::GetWindowAddrByRankId(mc2Context_, dstRankId, cntWinStateOffset_);
    GM_ADDR notifyAddr = remoteStateAddr + epRankId_ * WIN_ADDR_ALIGN;
    GM_ADDR remoteCountAddr = remoteStateAddr + epWorldSize_ * WIN_ADDR_ALIGN + epRankId_ * moeNumPerRankAlign512_;
    GlobalTensor<int32_t> countGMTensor;
    GlobalTensor<uint64_t> notifyGMTensor;
    countGMTensor.SetGlobalBuffer((__gm__ int32_t *)remoteCountAddr);
    notifyGMTensor.SetGlobalBuffer((__gm__ uint64_t *)notifyAddr);
    LocalTensor<int32_t> cntPerExpertTensor = tempBuf_.Get<int32_t>();
    DataCopyParams expertCntCopyParams = {1U, static_cast<uint16_t>(moeExpertNumPerRank_ * sizeof(int32_t)), 0U, 0U};
    uint32_t srcOffset = cntPerRankSizeAlign512_ / sizeof(int32_t) + dstRankId * moeExpertNumPerRank_;
    DataCopyPad(cntPerExpertTensor, sendCntGMTensor_[srcOffset], expertCntCopyParams, padParams_);
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    DataCopyPad(countGMTensor, cntPerExpertTensor, expertCntCopyParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>(); // perExpert 写完再写notifyVal
    notifyGMTensor.SetValue(0, notifyVal);
    DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(notifyGMTensor);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::ExchangeCount()
{
    SplitToCore(epWorldSize_, aivNum_, startRankId_, endRankId_, rankNumPerCore_);
    if (startRankId_ >= epWorldSize_) {
        return;
    }

    // 通信初始化
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);

    GlobalTensor<uint64_t> numSendGMTensorInt64;
    numSendGMTensorInt64.SetGlobalBuffer(
        (__gm__ uint64_t *)(sendCntWorkspaceAddr_ + startRankId_ * WIN_ADDR_ALIGN));
    LocalTensor<uint64_t> sendCntPerRankInt64 = dstExpBuf_.Get<uint64_t>();
    // GM 512B/rank 散开读, UB 每 rank 32B (UB_ALIGN), 只取前 8B (state+count)
    DataCopyParams cntCopyParams = {static_cast<uint16_t>(rankNumPerCore_), static_cast<uint16_t>(UB_ALIGN),
                                    static_cast<uint16_t>(WIN_ADDR_ALIGN - UB_ALIGN), 0U};
    DataCopyPad(sendCntPerRankInt64, numSendGMTensorInt64, cntCopyParams, padParams_);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    PipeBarrier<PIPE_MTE3>(); // 数据拷贝到workspace后再发远端

    for (uint32_t dstRankId = startRankId_; dstRankId < endRankId_; dstRankId++) {
        uint32_t notifyStride = (dstRankId - startRankId_) * INT64_UB_STRIDE;
        if (dstRankId != epRankId_) {
            SendCountToRemoteRank(dstRankId);
        } else {
            CopyLocalCountToWindow(dstRankId, sendCntPerRankInt64.GetValue(notifyStride));
        }
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::GetRecvCount()
{
    if (aivId_ != 0U) {
        return;
    }

    uint32_t serverRankCount = epWorldSize_;
    uint32_t mask = 1U;
    int32_t sumOfFlag = -1;
    int32_t compareFlag = static_cast<int32_t>(serverRankCount);
    LocalTensor<int32_t> recvCounterTensor = topkIdsBuf_.GetWithOffset<int32_t>(serverRankCount * UB_STRIDE, 0);
    LocalTensor<float> tempFp32 =
        topkIdsBuf_.GetWithOffset<float>(serverRankCount * UB_STRIDE, serverRankCount * UB_ALIGN);
    LocalTensor<float> recvCounterTensorFp32 = recvCounterTensor.template ReinterpretCast<float>();
    LocalTensor<float> numRecvPerRankTensorFp32 = numRecvPerRankTensor_.template ReinterpretCast<float>();
    int64_t waitStartCycle = GetSystemCycle();
    bool hasPrintedWaitStatus = false;
    SyncFunc<AscendC::HardEvent::MTE3_V>(); // 等待本核上轮计数器清0(buf 复用)
    while (sumOfFlag != compareFlag) {      // 状态位check
        DataCopy(recvCounterTensor, recvCounterGMTensor_, statusCopyParams_);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(numRecvPerRankTensorFp32, recvCounterTensorFp32, tempFp32, mask, serverRankCount, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = numRecvPerRankTensor_.GetValue(0);
        if (!hasPrintedWaitStatus && GetSystemCycle() - waitStartCycle > HANG_TRACE_TIMEOUT_CYCLES) {
            uint64_t invalidStatusMask = 0UL;
            for (uint32_t srcRankId = 0U; srcRankId < serverRankCount && srcRankId < 64U; srcRankId++) {
                if (recvCounterTensor.GetValue(srcRankId * UB_STRIDE) != 1) {
                    invalidStatusMask |= 1UL << srcRankId;
                }
            }
            printf("[MOE_EP_HANG] rank=%u aiv=%u wait=count status_sum=%d expected=%d "
                   "invalid_mask_low=%u invalid_mask_high=%u\n",
                   epRankId_, aivId_, sumOfFlag, compareFlag, static_cast<uint32_t>(invalidStatusMask),
                   static_cast<uint32_t>(invalidStatusMask >> 32U));
            hasPrintedWaitStatus = true;
        }
    }
    SetRecvNumPerExpert();                // 计算本卡上各专家接收的token总数
    SetRecvNumPerRank(recvCounterTensor); // 计算本端接收来自各卡的token总数
    // status clear
    Duplicate<int32_t>(recvCounterTensor, 0, serverRankCount * UB_STRIDE);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(recvCounterGMTensor_, recvCounterTensor, clearStatusCopyParams_);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SetRecvNumPerExpert()
{
    uint32_t recvExpertSizeAlign = moeNumPerRankAlign_;
    uint32_t recvExpertAlign = recvExpertSizeAlign / sizeof(int32_t);
    uint32_t serverRankCount = epWorldSize_;
    tpipe_->InitBuffer(recvCntBuf_, serverRankCount * recvExpertSizeAlign);
    tpipe_->InitBuffer(numRecvPerExpertBuf_, 2 * moeNumPerRankAlign_);
    tpipe_->InitBuffer(recvTempBuf_, UB_ALIGN);
    numRecvPerExpertTensor_ = numRecvPerExpertBuf_.Get<int64_t>();
    LocalTensor<int32_t> recvTensorInt32 = recvCntBuf_.Get<int32_t>();
    LocalTensor<int64_t> tempRecvTensorInt64 = dstExpBuf_.Get<int64_t>();
    LocalTensor<int64_t> recvCntTensor = recvTempBuf_.Get<int64_t>();
    LocalTensor<int64_t> sharedTmp = recvTensorInt32.template ReinterpretCast<int64_t>();
    LocalTensor<uint8_t> sharedTmpInt8 = recvTensorInt32.template ReinterpretCast<uint8_t>();

    const uint32_t shape[] = {serverRankCount, recvExpertAlign};
    DataCopyParams inRecvCntParams = {static_cast<uint16_t>(serverRankCount),
                                      static_cast<uint16_t>(recvExpertSizeAlign),
                                      static_cast<uint16_t>(moeNumPerRankAlign512_ - recvExpertSizeAlign), 0U};
    DataCopyParams recvPerExpertParams = {1U, static_cast<uint16_t>(moeExpertNumPerRank_ * sizeof(int64_t)), 0U, 0U};
    uint32_t recvExpertOffset = epWorldSize_ * WIN_ADDR_ALIGN / sizeof(int32_t);
    DataCopyPad(recvTensorInt32, recvCounterGMTensor_[recvExpertOffset], inRecvCntParams, padParams_);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(tempRecvTensorInt64, recvTensorInt32, RoundMode::CAST_NONE, serverRankCount * recvExpertAlign);
    ReduceSum<int64_t, AscendC::Pattern::Reduce::RA, true>(numRecvPerExpertTensor_, tempRecvTensorInt64, sharedTmpInt8,
                                                           shape, true);

    if constexpr (DoCpuSync) { // 计算actualA，并写入host pin
        GlobalTensor<int64_t> hostPinnedCounterTensor;
        hostPinnedCounterTensor.SetGlobalBuffer((__gm__ int64_t *)hostPinnedCounterAddrGM_);
        ReduceSum(recvCntTensor, numRecvPerExpertTensor_, sharedTmp, moeExpertNumPerRank_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(hostPinnedCounterTensor, recvCntTensor, UB_ALIGN / sizeof(int64_t));
    }

    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(numRecvPerExpertGMTensor_, numRecvPerExpertTensor_, recvPerExpertParams);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void
MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SetRecvNumPerRank(LocalTensor<int32_t> recvTmpTensor)
{
    LocalTensor<uint32_t> gatherIndexTensor =
        topkIdsBuf_.GetWithOffset<uint32_t>(UB_STRIDE, epWorldSize_ * UB_ALIGN);
    gatherIndexTensor.SetValue(0, 2U); // 源操作数每个datablock取下标为1的元素
    uint32_t compactMask = 2U;
    uint64_t reservedCount = 0UL;
    GatherMaskParams recvMaskParams = {1, static_cast<uint16_t>(epWorldSize_), 1, 0};
    DataCopyParams recvPerRankParams = {1U, static_cast<uint16_t>(epWorldSize_ * sizeof(int32_t)), 0U, 0U};
    SyncFunc<AscendC::HardEvent::S_V>();
    GatherMask(numRecvPerRankTensor_, recvTmpTensor, gatherIndexTensor, true, compactMask, recvMaskParams,
               reservedCount);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(numRecvPerRankGMTensor_, numRecvPerRankTensor_, recvPerRankParams);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::ReduceCounterRange(
    GlobalTensor<int32_t> counterGMTensor, LocalTensor<int32_t> dstTensor, uint32_t counterAlign512,
    uint32_t counterCnt, uint32_t counterValueCount, uint32_t counterCoreCount)
{
    LocalTensor<int32_t> counterTmpTensor = topkIdsBuf_.Get<int32_t>();
    uint32_t counterBytes = counterCnt * sizeof(int32_t);
    uint32_t counterAlignBytes = counterAlign512 * sizeof(int32_t);
    uint32_t copyNumPerGroup = perGroupSizeAlign_ * 2 / counterBytes;
    if (copyNumPerGroup == 0U) {
        copyNumPerGroup = 1U;
    }
    uint32_t groupCnt = Ceil(counterCoreCount, copyNumPerGroup);
    for (uint32_t groupIndex = 0U; groupIndex < groupCnt; groupIndex++) {
        uint32_t copyNum =
            (groupIndex == groupCnt - 1U) ? (counterCoreCount - copyNumPerGroup * groupIndex) : copyNumPerGroup;
        uint32_t gmOffset = groupIndex * copyNumPerGroup * counterAlign512;
        DataCopyParams counterCopyParams = {static_cast<uint16_t>(copyNum),
                                            static_cast<uint16_t>(counterBytes),
                                            static_cast<uint16_t>(counterAlignBytes - counterBytes), 0U};
        DataCopyPad(counterTmpTensor, counterGMTensor[gmOffset], counterCopyParams, padParams_);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        const uint32_t counterShape[] = {copyNum, counterCnt};
        ReduceSum<int32_t, AscendC::Pattern::Reduce::RA, true>(counterSumTensor_, counterTmpTensor, counterShape,
                                                               false);
        Add(dstTensor, dstTensor, counterSumTensor_, counterValueCount);
        if (groupIndex + 1U < groupCnt) {
            SyncFunc<AscendC::HardEvent::V_MTE2>();
        }
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::ReduceCounterPrefix(
    GlobalTensor<int32_t> counterGMTensor, LocalTensor<int32_t> dstTensor, uint32_t counterAlign512,
    uint32_t counterCnt, uint32_t counterValueCount)
{
    ReduceCounterRange(counterGMTensor, dstTensor, counterAlign512, counterCnt, counterValueCount, aivId_);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::GetSlotStartNum()
{
    // 预聚合前面所有核的 scaleup counter，得到本核起始 slot 偏移，避免循环内 AtomicAdd
    Duplicate<int32_t>(slotIdxPerRankTensor_, 0, epWorldSize_);
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>(); // 各核 counter 写入完成后再由 MTE2 读取前缀
    ReduceCounterPrefix(scaleupCounterGMTensor_, slotIdxPerRankTensor_, counterAlign512_, counterCnt_, epWorldSize_);

    // scaleout slot 起始偏移预聚合后存在 slotIdxPerRankTensor_ 后半段
    LocalTensor<int32_t> scaleoutSlotIdxTensor =
        numRecvPerRankBuf_.GetWithOffset<int32_t>(numScaleoutRanksAlign_, epWorldSizeAlign_);
    if (numScaleoutRanks_ > 0U) {
        if (aivId_ > 0U) {
            SyncFunc<AscendC::HardEvent::V_MTE2>();
        }
        Duplicate<int32_t>(scaleoutSlotIdxTensor, 0, numScaleoutRanksAlign_);
        ReduceCounterPrefix(scaleoutCounterGMTensor_, scaleoutSlotIdxTensor, scaleoutCounterAlign512_,
                            scaleoutCounterCnt_, numScaleoutRanks_);
    }
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline uint32_t MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::FindOrCreateRemoteRouteEntry(
    uint32_t tokenId, uint32_t dstScaleoutIndex, uint32_t &routeEntryCount, bool &isNewRouteEntry)
{
    // 直接索引 O(1) 查找：sendRouteIndexByScaleoutTensor_[dstScaleoutIndex] 存已建表项下标
    int32_t existingIdx = sendRouteIndexByScaleoutTensor_.GetValue(dstScaleoutIndex);
    if (existingIdx >= 0) {
        isNewRouteEntry = false;
        return static_cast<uint32_t>(existingIdx);
    }

    isNewRouteEntry = true;
    uint32_t routeEntryIndex = routeEntryCount;
    ASCENDC_ASSERT(routeEntryIndex < axisK_, {
        KERNEL_LOG(KERNEL_ERROR,
                   "remote route entry count (%u) exceeds axisK (%u), token %u dstScaleoutIndex %u",
                   routeEntryIndex, axisK_, tokenId, dstScaleoutIndex);
    });
    routeEntryCount++;

    routeScaleoutIndexTensor_.SetValue(routeEntryIndex, static_cast<int32_t>(dstScaleoutIndex));
    sendRouteIndexByScaleoutTensor_.SetValue(dstScaleoutIndex, static_cast<int32_t>(routeEntryIndex));
    return routeEntryIndex;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::UpdateRemoteRouteEntryTarget(
    uint32_t routeEntryIndex, uint32_t dstScaleupIndex, uint32_t scaleupSlot)
{
    // 用 bitmask 标记 scaleup 目标，UB 本地操作
    uint32_t maskBase = routeEntryIndex * routeScaleupMaskEntryElements_;
    uint32_t maskWordIndex = dstScaleupIndex / 32U;
    uint32_t maskBitIndex = dstScaleupIndex % 32U;
    uint32_t routeMaskOffset = maskBase + maskWordIndex;
    uint32_t currentMask = static_cast<uint32_t>(routeScaleupMaskTensor_.GetValue(routeMaskOffset));
    currentMask |= (1U << maskBitIndex);
    routeScaleupMaskTensor_.SetValue(routeMaskOffset, static_cast<int32_t>(currentMask));

    uint32_t routeSlotOffset = routeEntryIndex * routeScaleupSlotEntryElements_ + dstScaleupIndex;
    routeScaleupSlotTensor_.SetValue(routeSlotOffset, static_cast<int32_t>(scaleupSlot));
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::WriteRouteTableToGM(
    uint32_t tokenId, uint32_t routeEntryCount)
{
    // token 处理结束后批量写回 GM，避免循环内逐元素 GM scalar 写
    if (tokenId > startTokenId_) {
        SyncFunc<AscendC::HardEvent::MTE3_S>(); // 上一 token 的 route count 搬出后再复用固定 UB
    }
    routeCountTensor_.SetValue(0, static_cast<int32_t>(routeEntryCount));
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(routeCountGMTensor_[tokenId], routeCountTensor_, routeCountCopyParams_);
    DataCopyPad(routeDstScaleoutGMTensor_[tokenId * axisK_], routeScaleoutIndexTensor_, routeCopyParams_);
    DataCopyPad(routeScaleoutSlotGMTensor_[tokenId * axisK_], routeScaleoutSlotTensor_, routeCopyParams_);
    DataCopyPad(routeScaleupMaskGMTensor_[tokenId * axisK_ * scaleoutMaskWordCount_], routeScaleupMaskTensor_,
                routeMaskTableCopyParams_);
    DataCopyPad(routeScaleupSlotGMTensor_[tokenId * axisK_ * rankSizePerServer_], routeScaleupSlotTensor_,
                routeSlotTableCopyParams_);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline uint32_t
MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::CopyCachedRouteForToken(uint32_t tokenId)
{
    // cache 模式也返回本次输出，保持和 dst slot cache 路径一致的 handle 契约。
    if (tokenId > startTokenId_) {
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>(); // 上一轮 route 输出读 UB，本轮 cache route 重新搬入 UB
        SyncFunc<AscendC::HardEvent::S_MTE2>();    // 上一轮 scaleout workspace 写入前读取 route index/slot
    }
    uint32_t routeEntryCount = static_cast<uint32_t>(cachedRouteCountGMTensor_.GetValue(tokenId));
    DataCopyPad(routeScaleoutIndexTensor_, cachedRouteDstScaleoutGMTensor_[tokenId * axisK_], routeCopyParams_,
                routeCopyPadParams_);
    DataCopyPad(routeScaleoutSlotTensor_, cachedRouteScaleoutSlotGMTensor_[tokenId * axisK_], routeCopyParams_,
                routeCopyPadParams_);

    DataCopyPad(routeScaleupMaskTensor_, cachedRouteScaleupMaskGMTensor_[tokenId * axisK_ * scaleoutMaskWordCount_],
                routeMaskTableCopyParams_, routeCopyPadParams_);
    DataCopyPad(routeScaleupSlotTensor_, cachedRouteScaleupSlotGMTensor_[tokenId * axisK_ * rankSizePerServer_],
                routeSlotTableCopyParams_, routeCopyPadParams_);

    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    WriteRouteTableToGM(tokenId, routeEntryCount);
    return routeEntryCount;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::AllocateScaleupSlotForExpert(
    uint32_t topkIndex, uint32_t dstRankId, uint32_t &scaleupSlot)
{
    int32_t rankSlot = sendRankSlotTensor_.GetValue(dstRankId);
    if (rankSlot >= 0) {
        // 同一 rank 已分配过 slot，直接复用
        scaleupSlot = static_cast<uint32_t>(rankSlot);
        dstSlotIdxTensor_.SetValue(topkIndex, rankSlot);
        return;
    }

    // 本地递增分配 slot，无 AtomicAdd
    scaleupSlot = static_cast<uint32_t>(slotIdxPerRankTensor_.GetValue(dstRankId));
    dstSlotIdxTensor_.SetValue(topkIndex, static_cast<int32_t>(scaleupSlot));
    sendRankSlotTensor_.SetValue(dstRankId, static_cast<int32_t>(scaleupSlot));
    slotIdxPerRankTensor_.SetValue(dstRankId, static_cast<int32_t>(scaleupSlot + 1U));
    if (IsActiveRank(dstRankId)) {
        // 写入workspace对应区域
        WriteSlotToLocal(dstRankId, scaleupSlot);
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::UpdateScaleoutRouteForExpert(
    uint32_t tokenId, uint32_t dstRankId, uint32_t scaleupSlot, uint32_t &routeEntryCount)
{
    uint32_t dstScaleoutIndex = rankSizePerServer_ == 0U ? 0U : dstRankId / rankSizePerServer_;
    uint32_t dstScaleupIndex = rankSizePerServer_ == 0U ? dstRankId : dstRankId % rankSizePerServer_;
    bool isNewRouteEntry = false;
    uint32_t routeEntryIndex =
        FindOrCreateRemoteRouteEntry(tokenId, dstScaleoutIndex, routeEntryCount, isNewRouteEntry);
    // 新建表项时分配 scaleout slot（本地递增）
    if (isNewRouteEntry) {
        uint32_t scaleoutSlotOffset = epWorldSizeAlign_ / sizeof(int32_t) + dstScaleoutIndex;
        uint32_t scaleoutSlot = static_cast<uint32_t>(slotIdxPerRankTensor_.GetValue(scaleoutSlotOffset));
        routeScaleoutSlotTensor_.SetValue(routeEntryIndex, static_cast<int32_t>(scaleoutSlot));
        slotIdxPerRankTensor_.SetValue(scaleoutSlotOffset, static_cast<int32_t>(scaleoutSlot + 1U));
    }
    UpdateRemoteRouteEntryTarget(routeEntryIndex, dstScaleupIndex, scaleupSlot);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline uint32_t MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::DedupAndStage(uint32_t tokenId)
{
    uint32_t routeEntryCount = 0U;
    for (uint32_t topkIndex = 0; topkIndex < axisK_; topkIndex++) {
        int32_t expertId = topkIdxTensor_.GetValue(topkIndex);
        if (expertId < 0) {
            dstSlotIdxTensor_.SetValue(topkIndex, -1);
            continue;
        }

        uint32_t dstRankId = static_cast<uint32_t>(expertId) / moeExpertNumPerRank_;
        uint32_t scaleupSlot = 0U;
        AllocateScaleupSlotForExpert(topkIndex, dstRankId, scaleupSlot);

        if (IsActiveRank(dstRankId)) {
            continue;
        }

        UpdateScaleoutRouteForExpert(tokenId, dstRankId, scaleupSlot, routeEntryCount);
    }
    // 批量写回 route 表到 GM
    WriteRouteTableToGM(tokenId, routeEntryCount);
    return routeEntryCount;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline GM_ADDR MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::GetScaleoutSlotAddr(
    GM_ADDR baseAddr, uint32_t scaleoutIndex, uint32_t slot)
{
    return baseAddr +
           (static_cast<uint64_t>(scaleoutIndex) * axisMaxBS_ + static_cast<uint64_t>(slot)) * scaleoutSlotBytes_;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline uint32_t MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::GetCurrentScaleupIndex()
{
    return rankSizePerServer_ == 0U ? epRankId_ : epRankId_ % rankSizePerServer_;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline uint32_t MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::GetProxyRankId(
    uint32_t dstScaleoutIndex)
{
    return dstScaleoutIndex * rankSizePerServer_ + GetCurrentScaleupIndex();
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void
MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::WriteScaleoutPayloadFromRoute(uint32_t routeEntryCount)
{
    for (uint32_t routeEntryIndex = 0; routeEntryIndex < routeEntryCount; routeEntryIndex++) {
        int32_t dstScaleoutValue = routeScaleoutIndexTensor_.GetValue(routeEntryIndex);
        int32_t scaleoutSlotValue = routeScaleoutSlotTensor_.GetValue(routeEntryIndex);
        if (dstScaleoutValue < 0 || scaleoutSlotValue < 0) {
            continue;
        }

        uint32_t dstScaleoutIndex = static_cast<uint32_t>(dstScaleoutValue);
        uint32_t scaleoutSlot = static_cast<uint32_t>(scaleoutSlotValue);
        GM_ADDR scaleoutSlotAddr = GetScaleoutSlotAddr(scaleoutSendWinAddr_, dstScaleoutIndex, scaleoutSlot);
        GlobalTensor<XType> scaleoutPayloadTensor;
        scaleoutPayloadTensor.SetGlobalBuffer((__gm__ XType *)scaleoutSlotAddr);
        DataCopyPad(scaleoutPayloadTensor, tokenSlotTensor_, fanoutPayloadCopyParams_);

        GlobalTensor<int32_t> scaleoutRouteTensor;
        scaleoutRouteTensor.SetGlobalBuffer((__gm__ int32_t *)(scaleoutSlotAddr + perSlotBytes_));
        DataCopyPad(scaleoutRouteTensor, routeScaleupMaskTensor_[routeEntryIndex * routeScaleupMaskEntryElements_],
                    routeMaskEntryCopyParams_);
        DataCopyPad(scaleoutRouteTensor[scaleoutMaskWordCount_],
                    routeScaleupSlotTensor_[routeEntryIndex * routeScaleupSlotEntryElements_],
                    routeSlotEntryCopyParams_);
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::WriteSlotToLocal(uint32_t dstRankId,
                                                                                                  uint32_t slot)
{
    GlobalTensor<XType> slotGMTensor;
    if (dstRankId == epRankId_) {
        // 本卡: 写入 win 区 scaleup_recv[epRankId_][slot], 第一维是 srcGlobalRank
        GM_ADDR slotAddr = localWinAddr_ + (static_cast<uint64_t>(dstRankId) * axisMaxBS_ + slot) * perSlotBytes_;
        slotGMTensor.SetGlobalBuffer((__gm__ XType *)slotAddr);
        DataCopyPad(slotGMTensor, tokenSlotTensor_, fanoutPayloadCopyParams_);
        return;
    }
    // 非本卡: 写入本 rank 的 win staging 区，按 server 内局部 dstRank 偏移。
    uint32_t localDstRankId = dstRankId - serverStartRank_;
    GM_ADDR slotAddr =
        scaleupSendWinAddr_ + (static_cast<uint64_t>(localDstRankId) * axisMaxBS_ + slot) * perSlotBytes_;
    slotGMTensor.SetGlobalBuffer((__gm__ XType *)slotAddr);
    DataCopyPad(slotGMTensor, tokenSlotTensor_, fanoutPayloadCopyParams_);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline bool MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::InitScaleupSendRange()
{
    SplitToCore(epWorldSize_, aivNum_, startRankId_, endRankId_, rankNumPerCore_);
    if (endRankId_ <= serverStartRank_ || startRankId_ >= serverEndRank_) {
        return false;
    }
    if (startRankId_ < serverStartRank_) {
        startRankId_ = serverStartRank_;
    }
    if (endRankId_ > serverEndRank_) {
        endRankId_ = serverEndRank_;
    }
    rankNumPerCore_ = endRankId_ - startRankId_;
    return true;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::WriteScaleupSlotsToRank(
    uint32_t dstRankId, LocalTensor<int32_t> sendNumPerRankTensor)
{
    uint32_t sendTokenNum = sendNumPerRankTensor.GetValue((dstRankId - startRankId_) * UB_STRIDE + 1);
    GM_ADDR notifyAddr =
        MoeEpDispatchBase::GetWindowAddrByRankId(mc2Context_, dstRankId, slotWinStateOffset_) +
        epRankId_ * WIN_ADDR_ALIGN;
    if (unlikely(dstRankId == epRankId_)) {
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        GlobalTensor<int32_t> statusGMTensor;
        statusGMTensor.SetGlobalBuffer((__gm__ int32_t *)(notifyAddr));
        statusGMTensor.SetValue(0, static_cast<int32_t>(1));
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(statusGMTensor);
        return; //  本端slot已经写入win
    }

    uint64_t commHandle = MoeEpDispatchBase::GetCommHandle(mc2Context_, epRankId_, dstRankId);
    if (sendTokenNum > 0) {
        uint64_t sendDataSize = static_cast<uint64_t>(perSlotBytes_) * sendTokenNum;
        // win staging 按 server 内局部 dstRank 偏移，容量为 U*Nmax。
        uint32_t localDstRankId = dstRankId - serverStartRank_;
        uint64_t srcWinOffset = static_cast<uint64_t>(localDstRankId) * axisMaxBS_ * perSlotBytes_;
        uint64_t dstRankWinOffset =
            static_cast<uint64_t>(epRankId_) * axisMaxBS_ * perSlotBytes_; // 计算目标窗口地址偏移
        GM_ADDR remoteWinAddr =
            MoeEpDispatchBase::GetWindowAddrByRankId(mc2Context_, dstRankId, winDataOffset_) + dstRankWinOffset;
        GM_ADDR localSendWinAddr = scaleupSendWinAddr_ + srcWinOffset;
        MoeEpDispatchBase::WriteRemoteWindowByChunks(hcomm_, commHandle, remoteWinAddr, localSendWinAddr,
                                                     sendDataSize);
    }
    hcomm_.WriteNbi(commHandle, notifyAddr, sendCntWorkspaceAddr_ + dstRankId * WIN_ADDR_ALIGN,
                    static_cast<uint64_t>(sizeof(uint64_t)));
    hcomm_.Drain(commHandle);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::WriteToRemoteWindow()
{
    if (!InitScaleupSendRange()) {
        return;
    }

    uint32_t sendCntSize = rankNumPerCore_ * UB_ALIGN;
    tpipe_->InitBuffer(recvTempBuf_, Ceil(sendCntSize, UB_ALIGN) * UB_ALIGN);
    LocalTensor<int32_t> sendNumPerRankTensor = recvTempBuf_.Get<int32_t>();
    // GM 512B/rank 散开读, UB 每 rank 32B (UB_ALIGN), 只取前 8B (state+count)
    DataCopyParams sendCntCopyParams = {static_cast<uint16_t>(rankNumPerCore_), static_cast<uint16_t>(UB_ALIGN),
                                        static_cast<uint16_t>(WIN_ADDR_ALIGN - UB_ALIGN), 0U};
    DataCopyPad(sendNumPerRankTensor,
                sendCntGMTensor_[startRankId_ * WIN_ADDR_ALIGN / sizeof(int32_t)], sendCntCopyParams, padParams_);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    PipeBarrier<PIPE_MTE3>(); // 数据拷贝到 win staging 后再发远端

    for (uint32_t dstRankId = startRankId_; dstRankId < endRankId_; dstRankId++) {
        WriteScaleupSlotsToRank(dstRankId, sendNumPerRankTensor);
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void
MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::ReduceScaleoutCounterGroup(
    uint32_t counterGroup, LocalTensor<int32_t> counterSumTensor)
{
    // 每次只归约 owner 需要的 8 个相邻 server counter，避免每个 owner 重复扫描全部 server 列。
    LocalTensor<int32_t> counterRowsTensor = topkIdsBuf_.Get<int32_t>();
    DataCopyParams counterCopyParams = {
        static_cast<uint16_t>(aivNum_), 1U,
        static_cast<uint16_t>(scaleoutCounterAlign512_ / UB_STRIDE - 1U), 0U};
    const uint32_t counterShape[] = {aivNum_, UB_STRIDE};
    uint32_t counterOffset = counterGroup * UB_STRIDE;
    DataCopy(counterRowsTensor, scaleoutCounterGMTensor_[counterOffset], counterCopyParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    ReduceSum<int32_t, AscendC::Pattern::Reduce::RA, true>(
        counterSumTensor, counterRowsTensor, counterShape, false);
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::WriteScaleoutNotifyToWorkspace(
    uint32_t dstScaleoutIndex, uint32_t scaleoutSlotCount)
{
    GlobalTensor<int32_t> notifyTensor;
    notifyTensor.SetGlobalBuffer(
        (__gm__ int32_t *)(scaleoutNotifyWorkspaceAddr_ + dstScaleoutIndex * sizeof(uint64_t)));
    notifyTensor.SetValue(0, 1);
    notifyTensor.SetValue(1, static_cast<int32_t>(scaleoutSlotCount));
    // notify 会作为 HCOMM 源地址，必须把 Scalar 写入的 DCache 数据发布到 GM。
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(notifyTensor);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SendScaleoutPayloadToProxy(
    uint32_t dstScaleoutIndex, uint32_t proxyRankId, uint32_t scaleoutSlotCount)
{
    uint64_t commHandle = MoeEpDispatchBase::GetCommHandle(mc2Context_, epRankId_, proxyRankId);
    if (scaleoutSlotCount > 0U) {
        uint64_t sendDataSize = static_cast<uint64_t>(scaleoutSlotBytes_) * scaleoutSlotCount;
        GM_ADDR localScaleoutAddr = GetScaleoutSlotAddr(scaleoutSendWinAddr_, dstScaleoutIndex, 0U);
        GM_ADDR remoteScaleoutAddr =
            MoeEpDispatchBase::GetWindowAddrByRankId(mc2Context_, proxyRankId, scaleoutRecvDataOffset_) +
            (static_cast<uint64_t>(currentServerIndex_) * axisMaxBS_) * scaleoutSlotBytes_;
        uint64_t splitCount =
            (sendDataSize + MoeEpDispatchBase::DISPATCH_CQE_MAX_WRITE_SIZE - 1UL) /
            MoeEpDispatchBase::DISPATCH_CQE_MAX_WRITE_SIZE;
        uint64_t dataBytes = MoeEpDispatchBase::DISPATCH_CQE_MAX_WRITE_SIZE;
        for (uint64_t splitIndex = 0UL; splitIndex < splitCount; splitIndex++) {
            if (splitIndex == splitCount - 1UL) {
                dataBytes = sendDataSize - splitIndex * MoeEpDispatchBase::DISPATCH_CQE_MAX_WRITE_SIZE;
            }
            hcomm_.WriteNbi(commHandle, remoteScaleoutAddr, localScaleoutAddr, dataBytes);
            localScaleoutAddr += dataBytes;
            remoteScaleoutAddr += dataBytes;
        }
        hcomm_.Drain(commHandle);
        printf("[MOE_EP_HANG] rank=%u aiv=%u scaleout_payload_drain=complete proxy=%u bytes=%u slots=%u\n",
               epRankId_, aivId_, proxyRankId, static_cast<uint32_t>(sendDataSize), scaleoutSlotCount);
    }

    GM_ADDR remoteNotifyAddr =
        MoeEpDispatchBase::GetWindowAddrByRankId(mc2Context_, proxyRankId, scaleoutRecvStatusOffset_) +
        static_cast<uint64_t>(currentServerIndex_) * axisMaxBS_ * WIN_ADDR_ALIGN;
    hcomm_.WriteNbi(commHandle, remoteNotifyAddr,
                    scaleoutNotifyWorkspaceAddr_ + dstScaleoutIndex * sizeof(uint64_t), sizeof(uint64_t));
    hcomm_.Drain(commHandle);
    printf("[MOE_EP_HANG] rank=%u aiv=%u scaleout_notify_drain=complete proxy=%u\n",
           epRankId_, aivId_, proxyRankId);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SendScaleoutPayloadsToProxy()
{
    // payload 与 count 复用相同的目标 rank 分核，使同一连接始终由同一个 AIV 使用。
    uint32_t communicationStartRankId = 0U;
    uint32_t communicationEndRankId = 0U;
    uint32_t communicationRankNumPerCore = 0U;
    SplitToCore(epWorldSize_, aivNum_, communicationStartRankId, communicationEndRankId,
                communicationRankNumPerCore);
    if (communicationStartRankId >= epWorldSize_) {
        return;
    }
    bool ownsRemoteProxy = false;
    for (uint32_t dstScaleoutIndex = 0U; dstScaleoutIndex < numScaleoutRanks_; dstScaleoutIndex++) {
        if (dstScaleoutIndex == currentServerIndex_) {
            continue;
        }
        uint32_t proxyRankId = GetProxyRankId(dstScaleoutIndex);
        if (proxyRankId >= communicationStartRankId && proxyRankId < communicationEndRankId &&
            proxyRankId < epWorldSize_) {
            ownsRemoteProxy = true;
            break;
        }
    }
    if (!ownsRemoteProxy) {
        return;
    }

    LocalTensor<int32_t> scaleoutCounterSumTensor = counterSumTensor_;
    uint32_t counterGroupCount = Ceil(numScaleoutRanks_, UB_STRIDE);
    uint32_t currentCounterGroup = counterGroupCount;
    for (uint32_t dstScaleoutIndex = 0U; dstScaleoutIndex < numScaleoutRanks_; dstScaleoutIndex++) {
        if (dstScaleoutIndex == currentServerIndex_) {
            continue;
        }
        uint32_t proxyRankId = GetProxyRankId(dstScaleoutIndex);
        if (proxyRankId < communicationStartRankId || proxyRankId >= communicationEndRankId ||
            proxyRankId >= epWorldSize_) {
            continue;
        }

        uint32_t counterGroup = dstScaleoutIndex / UB_STRIDE;
        if (counterGroup != currentCounterGroup) {
            if (currentCounterGroup < counterGroupCount) {
                SyncFunc<AscendC::HardEvent::V_MTE2>();
            }
            ReduceScaleoutCounterGroup(counterGroup, scaleoutCounterSumTensor);
            currentCounterGroup = counterGroup;
        }
        uint32_t scaleoutSlotCount =
            static_cast<uint32_t>(scaleoutCounterSumTensor.GetValue(dstScaleoutIndex % UB_STRIDE));
        WriteScaleoutNotifyToWorkspace(dstScaleoutIndex, scaleoutSlotCount);
        SendScaleoutPayloadToProxy(dstScaleoutIndex, proxyRankId, scaleoutSlotCount);
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::DistributeScaleoutPayload(
    GM_ADDR scaleoutSlotAddr, uint32_t srcRankId, LocalTensor<int32_t> fanoutCountTensor)
{
    GlobalTensor<XType> scaleoutPayloadTensor;
    GlobalTensor<int32_t> scaleoutRouteTensor;
    scaleoutPayloadTensor.SetGlobalBuffer((__gm__ XType *)scaleoutSlotAddr);
    scaleoutRouteTensor.SetGlobalBuffer((__gm__ int32_t *)(scaleoutSlotAddr + perSlotBytes_));
    SyncFunc<AscendC::HardEvent::S_MTE2>(); // 上一 slot 的 route scalar 读取完成后再复用 route UB
    LocalTensor<XType> payloadTensor = perSlotQueue_.AllocTensor<XType>();
    DataCopyPad(payloadTensor, scaleoutPayloadTensor, fanoutPayloadCopyParams_, padParams_);
    DataCopyPad(routeScaleupMaskTensor_, scaleoutRouteTensor, routeMaskEntryCopyParams_, routeCopyPadParams_);
    DataCopyPad(routeScaleupSlotTensor_, scaleoutRouteTensor[scaleoutMaskWordCount_], routeSlotEntryCopyParams_,
                routeCopyPadParams_);
    perSlotQueue_.EnQue(payloadTensor);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    LocalTensor<XType> payloadOutTensor = perSlotQueue_.DeQue<XType>();

    for (uint32_t maskWordIndex = 0U; maskWordIndex < scaleoutMaskWordCount_; maskWordIndex++) {
        uint32_t routeMask = static_cast<uint32_t>(routeScaleupMaskTensor_.GetValue(maskWordIndex));
        for (uint32_t maskBitIndex = 0U; maskBitIndex < 32U; maskBitIndex++) {
            uint32_t dstScaleupIndex = maskWordIndex * 32U + maskBitIndex;
            if (dstScaleupIndex >= rankSizePerServer_) {
                break;
            }
            if ((routeMask & (1U << maskBitIndex)) == 0U) {
                continue;
            }
            int32_t scaleupSlotValue = routeScaleupSlotTensor_.GetValue(dstScaleupIndex);
            if (scaleupSlotValue < 0) {
                continue;
            }
            uint32_t dstRankId = serverStartRank_ + dstScaleupIndex;
            if (dstRankId >= serverEndRank_ || dstRankId >= epWorldSize_) {
                continue;
            }

            WriteFanoutPayloadToDestination(payloadOutTensor, srcRankId, dstRankId, dstScaleupIndex,
                                             static_cast<uint32_t>(scaleupSlotValue));
            int32_t targetCount = fanoutCountTensor.GetValue(dstScaleupIndex);
            fanoutCountTensor.SetValue(dstScaleupIndex, targetCount + 1);
        }
    }
    perSlotQueue_.FreeTensor<XType>(payloadOutTensor);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::WriteFanoutPayloadToDestination(
    LocalTensor<XType> payloadTensor, uint32_t srcRankId, uint32_t dstRankId, uint32_t dstScaleupIndex,
    uint32_t scaleupSlot)
{
    GM_ADDR destinationSlotAddr = nullptr;
    if (dstRankId == epRankId_) {
        destinationSlotAddr = localWinAddr_ +
            (static_cast<uint64_t>(srcRankId) * axisMaxBS_ + scaleupSlot) * perSlotBytes_;
    } else {
        // win staging 按 server 内局部 dstRank 偏移，容量为 U*Nmax。
        destinationSlotAddr = scaleupSendWinAddr_ +
            (static_cast<uint64_t>(dstScaleupIndex) * axisMaxBS_ + scaleupSlot) * perSlotBytes_;
    }
    GlobalTensor<XType> destinationSlotTensor;
    destinationSlotTensor.SetGlobalBuffer((__gm__ XType *)destinationSlotAddr);
    DataCopyPad(destinationSlotTensor, payloadTensor, fanoutPayloadCopyParams_);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::WriteFanoutCounts(
    LocalTensor<int32_t> fanoutCountTensor)
{
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(scaleupCounterGMTensor_[aivId_ * counterAlign512_], fanoutCountTensor, fanoutCounterCnt_);
    PipeBarrier<PIPE_MTE3>(); // fanout payload 和 count 均会在跨核同步后被消费
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::DistributeScaleoutPayloads(
    uint32_t srcScaleoutIndex, uint32_t srcRankId, uint32_t scaleoutSlotCount)
{
    LocalTensor<int32_t> fanoutCountTensor = slotIdxPerRankTensor_;
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(fanoutCountTensor, 0, fanoutCounterCnt_);
    SyncFunc<AscendC::HardEvent::V_S>();

    uint32_t startScaleoutSlot = 0U;
    uint32_t endScaleoutSlot = 0U;
    uint32_t scaleoutSlotNumPerCore = 0U;
    SplitToCore(scaleoutSlotCount, aivNum_, startScaleoutSlot, endScaleoutSlot, scaleoutSlotNumPerCore);
    GM_ADDR localScaleoutBase =
        MoeEpDispatchBase::GetWindowAddrByRankId(mc2Context_, epRankId_, scaleoutRecvDataOffset_);
    for (uint32_t scaleoutSlot = startScaleoutSlot; scaleoutSlot < endScaleoutSlot; scaleoutSlot++) {
        GM_ADDR scaleoutSlotAddr = GetScaleoutSlotAddr(localScaleoutBase, srcScaleoutIndex, scaleoutSlot);
        DistributeScaleoutPayload(scaleoutSlotAddr, srcRankId, fanoutCountTensor);
    }
    WriteFanoutCounts(fanoutCountTensor);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::ReduceFanoutCounts()
{
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(slotIdxPerRankTensor_, 0, fanoutCounterCnt_);
    ReduceCounterRange(scaleupCounterGMTensor_, slotIdxPerRankTensor_, counterAlign512_, fanoutCounterCnt_,
                       rankSizePerServer_, aivNum_);
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SendStagedFanoutPayloadsToRank(
    uint32_t srcRankId, uint32_t dstRankId, uint32_t fanoutSlotCount)
{
    if (dstRankId != epRankId_ && fanoutSlotCount > 0U) {
        uint64_t commHandle = MoeEpDispatchBase::GetCommHandle(mc2Context_, epRankId_, dstRankId);
        // win staging 按 server 内局部 dstRank 偏移，容量为 U*Nmax。
        uint32_t localDstRankId = dstRankId - serverStartRank_;
        GM_ADDR stagingPayloadAddr = scaleupSendWinAddr_ +
            static_cast<uint64_t>(localDstRankId) * axisMaxBS_ * perSlotBytes_;
        GM_ADDR finalPayloadAddr =
            MoeEpDispatchBase::GetWindowAddrByRankId(mc2Context_, dstRankId, winDataOffset_) +
            static_cast<uint64_t>(srcRankId) * axisMaxBS_ * perSlotBytes_;
        uint64_t fanoutPayloadBytes = static_cast<uint64_t>(fanoutSlotCount) * perSlotBytes_;
        MoeEpDispatchBase::WriteRemoteWindowByChunks(hcomm_, commHandle, finalPayloadAddr, stagingPayloadAddr,
                                                     fanoutPayloadBytes);
    }
    WriteFinalReadyStatusToRank(srcRankId, dstRankId);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SendStagedFanoutPayloads(
    uint32_t srcRankId, uint32_t startScaleupIndex, uint32_t endScaleupIndex)
{
    for (uint32_t dstScaleupIndex = startScaleupIndex; dstScaleupIndex < endScaleupIndex; dstScaleupIndex++) {
        uint32_t dstRankId = serverStartRank_ + dstScaleupIndex;
        if (dstRankId >= serverEndRank_ || dstRankId >= epWorldSize_) {
            continue;
        }
        uint32_t fanoutSlotCount =
            static_cast<uint32_t>(slotIdxPerRankTensor_.GetValue(dstScaleupIndex));
        SendStagedFanoutPayloadsToRank(srcRankId, dstRankId, fanoutSlotCount);
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::WriteFinalReadyStatusToRank(
    uint32_t srcRankId, uint32_t dstRankId)
{
    GM_ADDR notifyAddr =
        MoeEpDispatchBase::GetWindowAddrByRankId(mc2Context_, dstRankId, slotWinStateOffset_) +
        static_cast<uint64_t>(srcRankId) * WIN_ADDR_ALIGN;
    if (dstRankId == epRankId_) {
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        GlobalTensor<int32_t> statusTensor;
        statusTensor.SetGlobalBuffer((__gm__ int32_t *)notifyAddr);
        statusTensor.SetValue(0, 1);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(statusTensor);
        return;
    }

    uint64_t commHandle = MoeEpDispatchBase::GetCommHandle(mc2Context_, epRankId_, dstRankId);
    GM_ADDR readySourceAddr = sendCntWorkspaceAddr_ + dstRankId * WIN_ADDR_ALIGN;
    hcomm_.WriteNbi(commHandle, notifyAddr, readySourceAddr, sizeof(uint64_t));
    hcomm_.Drain(commHandle);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::PollBackoff()
{
    int64_t backoffStartCycle = GetSystemCycle();
    while (GetSystemCycle() - backoffStartCycle < STATUS_POLL_BACKOFF_CYCLES) {
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline uint32_t MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::WaitScaleoutPayloadReady(
    GM_ADDR statusAddr)
{
    __gm__ int32_t *statusValueAddr = (__gm__ int32_t *)statusAddr;
    int64_t waitStartCycle = GetSystemCycle();
    bool hasPrintedWaitStatus = false;
    while (ReadGmByPassDCache(statusValueAddr) != 1) {
        if (!hasPrintedWaitStatus && GetSystemCycle() - waitStartCycle > HANG_TRACE_TIMEOUT_CYCLES) {
            printf("[MOE_EP_HANG] rank=%u aiv=%u wait=scaleout_ready status=%d slot_count=%d\n",
                   epRankId_, aivId_, ReadGmByPassDCache(statusValueAddr),
                   ReadGmByPassDCache(statusValueAddr + 1));
            hasPrintedWaitStatus = true;
        }
        PollBackoff();
    }
    return static_cast<uint32_t>(ReadGmByPassDCache(statusValueAddr + 1));
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::ClearScaleoutReceiveStatus(
    GM_ADDR localScaleoutStatusBase, uint32_t statusOffset)
{
    GlobalTensor<int32_t> currentStatusTensor;
    currentStatusTensor.SetGlobalBuffer(
        (__gm__ int32_t *)(localScaleoutStatusBase + statusOffset * sizeof(int32_t)));
    currentStatusTensor.SetValue(0, 0);
    currentStatusTensor.SetValue(1, 0);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(currentStatusTensor);
    PipeBarrier<PIPE_ALL>(); // status 清零完成后才能结束本轮或处理下一 source
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::FanoutReceivedScaleoutPayloads()
{
    GM_ADDR localScaleoutStatusBase =
        MoeEpDispatchBase::GetWindowAddrByRankId(mc2Context_, epRankId_, scaleoutRecvStatusOffset_);
    uint32_t startScaleupIndex = 0U;
    uint32_t endScaleupIndex = 0U;
    uint32_t rankNumPerCore = 0U;
    uint32_t communicationStartRankId = 0U;
    uint32_t communicationEndRankId = 0U;
    SplitToCore(epWorldSize_, aivNum_, communicationStartRankId, communicationEndRankId, rankNumPerCore);
    if (communicationEndRankId > serverStartRank_ && communicationStartRankId < serverEndRank_) {
        uint32_t serverCommunicationStartRankId = communicationStartRankId;
        uint32_t serverCommunicationEndRankId = communicationEndRankId;
        if (serverCommunicationStartRankId < serverStartRank_) {
            serverCommunicationStartRankId = serverStartRank_;
        }
        if (serverCommunicationEndRankId > serverEndRank_) {
            serverCommunicationEndRankId = serverEndRank_;
        }
        startScaleupIndex = serverCommunicationStartRankId - serverStartRank_;
        endScaleupIndex = serverCommunicationEndRankId - serverStartRank_;
    }
    uint32_t currentScaleupIndex = GetCurrentScaleupIndex();
    for (uint32_t srcScaleoutIndex = 0U; srcScaleoutIndex < numScaleoutRanks_; srcScaleoutIndex++) {
        if (srcScaleoutIndex == currentServerIndex_) {
            continue;
        }
        uint32_t srcRankId = srcScaleoutIndex * rankSizePerServer_ + currentScaleupIndex;
        if (srcRankId >= epWorldSize_) {
            continue;
        }

        uint32_t statusOffset =
            static_cast<uint32_t>(srcScaleoutIndex * axisMaxBS_ * WIN_ADDR_ALIGN / sizeof(int32_t));
        GM_ADDR currentStatusAddr = localScaleoutStatusBase + statusOffset * sizeof(int32_t);
        if (aivId_ == 0U) {
            WaitScaleoutPayloadReady(currentStatusAddr);
        }

        // ready 由控制核确认后，各核才能读取 slot count
        SyncAll<true>();
        uint32_t scaleoutSlotCount = static_cast<uint32_t>(
            ReadGmByPassDCache((__gm__ int32_t *)currentStatusAddr + 1));
        DistributeScaleoutPayloads(srcScaleoutIndex, srcRankId, scaleoutSlotCount);
        // fanout payload 和每核 fanout count 写完后，目标 rank 发送核才能读取
        SyncAll<true>();

        if (startScaleupIndex < rankSizePerServer_) {
            ReduceFanoutCounts();
            SendStagedFanoutPayloads(srcRankId, startScaleupIndex, endScaleupIndex);
        }
        // 所有目标 rank 的 payload 和 final ready 发布完成后才能清理接收区
        SyncAll<true>();
        if (aivId_ == 0U) {
            ClearScaleoutReceiveStatus(localScaleoutStatusBase, statusOffset);
        }
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::InitRouteStateTensors(
    uint32_t slotIndexBytes, uint32_t rankSlotBytes, uint32_t routeStateBytes, uint32_t routeCountOffset)
{
    // 从 numRecvPerRankBuf_ 切片出各 UB tensor，存为成员变量供后续函数使用
    uint32_t rankSlotOffset = slotIndexBytes;
    uint32_t routeScaleoutIndexOffset = rankSlotOffset + rankSlotBytes;
    uint32_t routeScaleoutSlotOffset = routeScaleoutIndexOffset + routeStateBytes;
    uint32_t routeScaleupMaskOffset = routeScaleoutSlotOffset + routeStateBytes;
    uint32_t routeScaleupSlotOffset = routeScaleupMaskOffset + routeScaleupMaskBytes_;
    uint32_t routeScaleoutStateOffset = routeScaleupSlotOffset + routeScaleupSlotBytes_;
    slotIdxPerRankTensor_ = numRecvPerRankBuf_.Get<int32_t>();
    counterSumTensor_ = numRecvPerRankBuf_.GetWithOffset<int32_t>(counterCnt_, rankSlotOffset);
    sendRankSlotTensor_ = numRecvPerRankBuf_.GetWithOffset<int32_t>(epWorldSize_, rankSlotOffset);
    routeScaleoutIndexTensor_ = numRecvPerRankBuf_.GetWithOffset<int32_t>(axisKAlign_, routeScaleoutIndexOffset);
    routeScaleoutSlotTensor_ =
        numRecvPerRankBuf_.GetWithOffset<int32_t>(axisKAlign_, routeScaleoutSlotOffset);
    routeScaleupMaskTensor_ = numRecvPerRankBuf_.GetWithOffset<int32_t>(
        axisKAlign_ * routeScaleupMaskEntryElements_, routeScaleupMaskOffset);
    routeScaleupSlotTensor_ = numRecvPerRankBuf_.GetWithOffset<int32_t>(
        axisKAlign_ * routeScaleupSlotEntryElements_, routeScaleupSlotOffset);
    sendRouteIndexByScaleoutTensor_ = numRecvPerRankBuf_.GetWithOffset<int32_t>(
        numScaleoutRanksAlign_, routeScaleoutStateOffset);
    routeCountTensor_ = numRecvPerRankBuf_.GetWithOffset<int32_t>(UB_STRIDE, routeCountOffset);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::InitPreparePayloadsBuffers()
{
    tpipe_->InitBuffer(perSlotQueue_, BUFFER_NUM, perSlotBytes_);
    tpipe_->InitBuffer(dstSlotQueue_, 1, kAlignSize_);
    tpipe_->InitBuffer(topkIdsBuf_, 2 * perGroupSizeAlign_); // GetSlotStartNum 需要复用
    // UB 布局：slotIdxPerRank + scaleoutSlotIdx + prefix 归约/rank slot 复用切片 + routeScaleoutIndex
    //         + routeScaleoutSlot + routeScaleupMask + routeScaleupSlot
    uint32_t routeStateBytes = Ceil(axisK_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t scaleoutIdxBytes = Ceil(numScaleoutRanks_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t slotIndexBytes = epWorldSizeAlign_ + scaleoutIdxBytes;
    uint32_t rankSlotBytes = epWorldSizeAlign_;
    uint32_t routeCountBytes = UB_ALIGN;
    // prefix 归约结果和 token 内 rank slot 映射生命周期不重叠，复用同一切片。
    uint32_t sendBufSize = slotIndexBytes + rankSlotBytes + routeStateBytes * 2U + routeScaleupMaskBytes_ +
                           routeScaleupSlotBytes_ + scaleoutIdxBytes + routeCountBytes;
    tpipe_->InitBuffer(numRecvPerRankBuf_, sendBufSize);
    InitRouteStateTensors(slotIndexBytes, rankSlotBytes, routeStateBytes, sendBufSize - routeCountBytes);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::InitRouteStateForToken(
    uint32_t tokenId)
{
    if (tokenId > startTokenId_) {
        SyncFunc<AscendC::HardEvent::S_V>();
    }
    if constexpr (!IsCached) {
        if (tokenId > startTokenId_) {
            SyncFunc<AscendC::HardEvent::MTE3_V>(); // 非 cache 下一轮用 V 重新初始化 route UB
        }
    }
    Duplicate<int32_t>(sendRankSlotTensor_, -1, epWorldSize_);
    if constexpr (!IsCached) {
        Duplicate<int32_t>(routeScaleoutIndexTensor_, -1, axisKAlign_);
        Duplicate<int32_t>(routeScaleoutSlotTensor_, -1, axisKAlign_);
        Duplicate<int32_t>(routeScaleupMaskTensor_, 0, axisKAlign_ * routeScaleupMaskEntryElements_);
        Duplicate<int32_t>(routeScaleupSlotTensor_, -1, axisKAlign_ * routeScaleupSlotEntryElements_);
        Duplicate<int32_t>(sendRouteIndexByScaleoutTensor_, -1, numScaleoutRanksAlign_);
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::PrepareTokenPayload(
    uint32_t tokenId, uint32_t topkOffset)
{
    xLocalTensor_ = perSlotQueue_.AllocTensor<XType>();
    metaLocalTensor_ = xLocalTensor_[metaOffset_ / sizeof(XType)].template ReinterpretCast<int32_t>();
    DataCopyPad(xLocalTensor_, xGMTensor_[tokenId * axisH_], xCopyParams_, padParams_);
    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
        DataCopyPad(xLocalTensor_[hAlignSize_ / sizeof(XType)].template ReinterpretCast<ScalesType>(),
                    scalesGMTensor_[tokenId * scalesBytes_ / sizeof(ScalesType)], scalesCopyParams_, padParams_);
    }
    DataCopyPad(metaLocalTensor_, topkIdxGMTensor_[topkOffset], topkCopyParams_, padParams_);
    if constexpr (IsTopkWeights) {
        DataCopyPad(metaLocalTensor_[axisKAlign_].template ReinterpretCast<float>(), topkWeightsGMTensor_[topkOffset],
                    topkCopyParams_, padParams_);
    }
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    metaLocalTensor_.SetValue(2 * axisKAlign_, epRankId_);
    metaLocalTensor_.SetValue(2 * axisKAlign_ + 1, tokenId);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    perSlotQueue_.EnQue(xLocalTensor_);
    tokenSlotTensor_ = perSlotQueue_.DeQue<XType>();
    topkIdxTensor_ = tokenSlotTensor_[metaOffset_ / sizeof(XType)].template ReinterpretCast<int32_t>();
    dstSlotIdxTensor_ = dstSlotQueue_.AllocTensor<int32_t>();
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline uint32_t MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::ProcessCachedTokenRoutes(
    uint32_t tokenId, uint32_t topkOffset)
{
    DataCopyPad(dstSlotIdxTensor_, cachedSlotIdxGMTensor_[topkOffset], topkCopyParams_, padParams_);
    uint32_t routeEntryCount = CopyCachedRouteForToken(tokenId);
    dstSlotQueue_.EnQue(dstSlotIdxTensor_);
    dstSlotIdxTensor_ = dstSlotQueue_.DeQue<int32_t>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    for (uint32_t topkIndex = 0; topkIndex < axisK_; topkIndex++) {
        int32_t slot = dstSlotIdxTensor_.GetValue(topkIndex);
        if (slot == -1) {
            continue;
        }
        int32_t expertId = topkIdxTensor_.GetValue(topkIndex);
        uint32_t dstRankId = expertId / moeExpertNumPerRank_;
        if (!IsActiveRank(dstRankId)) {
            continue;
        }
        if (sendRankSlotTensor_.GetValue(dstRankId) == slot) {
            continue;
        }
        WriteSlotToLocal(dstRankId, slot);
        sendRankSlotTensor_.SetValue(dstRankId, slot);
    }
    return routeEntryCount;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::PreparePayloads()
{
    InitPreparePayloadsBuffers();

    // count 阶段已经按 token 分核生成 counter，这里只计算本核 slot 前缀。
    GetSlotStartNum();

    for (uint32_t tokenId = startTokenId_; tokenId < endTokenId_; ++tokenId) {
        uint32_t topkOffset = tokenId * axisK_;
        InitRouteStateForToken(tokenId);
        PrepareTokenPayload(tokenId, topkOffset);

        uint32_t routeEntryCount = 0U;
        if constexpr (!IsCached) {
            routeEntryCount = DedupAndStage(tokenId);
            dstSlotQueue_.EnQue(dstSlotIdxTensor_);
            dstSlotIdxTensor_ = dstSlotQueue_.DeQue<int32_t>();
        } else {
            routeEntryCount = ProcessCachedTokenRoutes(tokenId, topkOffset);
        }
        WriteScaleoutPayloadFromRoute(routeEntryCount);
        DataCopyPad(dstSlotIdxGMTensor_[topkOffset], dstSlotIdxTensor_, topkCopyParams_);
        perSlotQueue_.FreeTensor<XType>(tokenSlotTensor_);
        dstSlotQueue_.FreeTensor<int32_t>(dstSlotIdxTensor_);
    }
    if (sendTokenNum_ > 0U) {
        PipeBarrier<PIPE_ALL>(); // PreparePayloads 写 GM 后，SyncAll 后会继续消费这些 GM 输出
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::PrepareDispatchPayloads()
{
    CalSendCnt();
    SyncAll<true>();

    ExchangeCount();
    GetRecvCount();
    PipeBarrier<PIPE_ALL>();
    tpipe_->Reset();

    PreparePayloads();
    SyncAll<true>();

    // 通信初始化
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::TransferDispatchPayloads()
{
    // 先完成本超节点内直发和到远端 proxy 的聚合发送。
    WriteToRemoteWindow();
    SendScaleoutPayloadsToProxy();
    // 等待所有发送核完成后，再执行目标 rank fanout 和接收区确认。
    SyncAll<true>(); // fanout 按目标 rank 分核使用 HCOMM，先等待其他 AIV 完成发送
    FanoutReceivedScaleoutPayloads();
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::Process()
{
    if ASCEND_IS_AIV { // 全aiv处理
        PrepareDispatchPayloads();
        TransferDispatchPayloads();
        if (aivId_ == 0U) {
            printf("[MOE_EP_HANG] rank=%u aiv=%u dispatch=complete\n", epRankId_, aivId_);
        }
    }
}
} // namespace MoeEpDispatchHybridImpl

#endif // MOE_EP_DISPATCH_HYBRID_H
