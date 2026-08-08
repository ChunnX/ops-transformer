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
 * \file mega_moe.h
 * \brief
 */

#ifndef MEGA_MOE_H
#define MEGA_MOE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#if __has_include("../../common/mc2_kernel_utils.h")
#include "../../common/mc2_kernel_utils.h"
#else
#include "../../../common/op_kernel/mc2_kernel_utils.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "common/mega_moe_types.h"
#include "common/mega_moe_workspace.h"
#include "common/mega_moe_utils.h"
#include "stage/block_epilogue_swiglu_mx_quant.h"
#include "stage/mega_moe_token_quant.h"
#include "stage/mega_moe_send_mask.h"
#include "stage/mega_moe_workspace_reset.h"
#include "stage/mega_moe_shared_expert_input.h"
#include "stage/mega_moe_token_dispatch.h"
#include "stage/mega_moe_gmm1_swiglu.h"
#include "stage/mega_moe_gmm2_combine.h"
#include "stage/mega_moe_unpermute.h"
#if __has_include("../../moe_distribute_dispatch_v2/quantize_functions.h")
#include "../../moe_distribute_dispatch_v2/quantize_functions.h"
#else
#include "../../../moe_distribute_dispatch_v2/op_kernel/quantize_functions.h"
#endif

using namespace AscendC;

namespace MegaMoeImpl {

// 预留：XType OutputType TopkWeightsType Weight1Type
#define TemplateMegaMoeTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode, \
        int32_t CombineQuantMode, bool TopkWeightsPrefetch
#define TemplateMegaMoeTypeFunc \
    XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode, TopkWeightsPrefetch

// 编排普通模板独立的 Dispatch、GMM1/SwiGLU 和 GMM2/Combine 阶段。
template <uint8_t CombineMode, typename QuantOutType, typename ActivationType, typename WeightType,
          typename SwigluOutType, typename QuantScaleType, bool EnableA8W4, bool EnableA4W4,
          uint32_t Gmm1TileM, uint32_t EpilogueTileM, bool TopkWeightsPrefetch, typename BlockEpilogue>
__aicore__ inline void ProcessMoeExperts(
    const TokenDispatchContext &dispatchContext, const TokenDispatchArgs &dispatchArgs,
    const Gmm1SwigluContext &gmm1Context, const Gmm1SwigluArgs &gmm1Args,
    const Gmm2Context &gmm2Context, const Gmm2Args &gmm2Args,
    const QuantCombineContext &combineContext, const QuantCombineBufferConfig &combineBufferConfig,
    const Params &params, BlockEpilogue &epilogueOp, TokenDispatchScratch<ActivationType> &dispatchScratch,
    Gmm1SwigluScratch &gmm1Scratch, Gmm2Scratch &gmm2Scratch, GM_ADDR expertTokenNumsOutPtr,
    int32_t &gmTileSequence, uint16_t &gmm2PingPongIdx)
{
    if (dispatchContext.blockJob.totalJobs == 0U ||
        dispatchContext.blockJob.jobIndex >= dispatchContext.blockJob.totalJobs ||
        dispatchContext.countWorkspace.blockNum == 0U ||
        dispatchContext.countWorkspace.blockIdx >= dispatchContext.countWorkspace.blockNum ||
        gmm1Context.expertPerRank == 0U || dispatchContext.blockNumPerRank == 0U) {
        return;
    }

    InitTokenDispatchBuffers<ActivationType, EnableA8W4>(dispatchContext, dispatchArgs, dispatchScratch);
    gmm1Scratch.expertRevNumsGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(gmm1Args.expertRevTokenNumsPtr));
    Gmm1ExpertLoopState gmm1State = CreateGmm1ExpertLoopState(gmm1Context);
    GMMAddrInfo gmm1AddrInfo;
    uint32_t gmm1StartBlockIdx = 0U;
    int32_t gmm1VecSetSyncCom = 0;
    uint16_t gmm1PingPongIdx = 0U;
    Gmm1SwigluState gmm1RuntimeState{
        gmm1StartBlockIdx, gmm1VecSetSyncCom, gmTileSequence, gmm1PingPongIdx};
    uint64_t currentSendCnt = 0U;
    uint64_t nextSendCnt = 0U;

    RunMoeExpertDispatchStage<ActivationType, QuantScaleType, EnableA8W4, Gmm1TileM, TopkWeightsPrefetch>(
        dispatchContext, dispatchArgs, dispatchScratch, 0U, nextSendCnt);
    for (uint32_t expertIdx = 0U; expertIdx < gmm1Context.expertPerRank; ++expertIdx) {
        currentSendCnt = nextSendCnt;
        if (expertIdx + 1U < gmm1Context.expertPerRank) {
            RunMoeExpertDispatchStage<ActivationType, QuantScaleType, EnableA8W4, Gmm1TileM,
                                      TopkWeightsPrefetch>(
                dispatchContext, dispatchArgs, dispatchScratch, expertIdx + 1U, nextSendCnt);
        }
        RunMoeExpertGmm1SwigluStage<QuantOutType, ActivationType, WeightType, SwigluOutType,
                                    QuantScaleType, EnableA8W4, Gmm1TileM, EpilogueTileM,
                                    TopkWeightsPrefetch>(
            gmm1Context, gmm1Args, params, epilogueOp, gmm1Scratch, gmm1State, gmm1AddrInfo,
            gmm1RuntimeState, expertIdx, currentSendCnt);
    }
    FinishMoeGmm1SwigluStage<EnableA8W4, TopkWeightsPrefetch>(gmm1Context, gmm1Args, gmm1RuntimeState);

    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() == 1U) {
            ExportExpertTokenCounts<ActivationType, EnableA8W4>(
                dispatchContext, dispatchScratch, expertTokenNumsOutPtr);
        }
    }
    if constexpr (TopkWeightsPrefetch) {
        SyncAll<false>();
    }

    uint32_t gmm2StartBlockIdx = gmm1RuntimeState.startBlockIdx;
    int32_t gmm2VecSetSyncCom = 0;
    Gmm2RuntimeState gmm2RuntimeState{
        gmm2StartBlockIdx, gmm2VecSetSyncCom, gmTileSequence, gmm2PingPongIdx};
    RunMoeGmm2CombineStage<CombineMode, QuantOutType, SwigluOutType, WeightType,
                           QuantScaleType, ActivationType, EnableA8W4, EnableA4W4,
                           Gmm1TileM, TopkWeightsPrefetch>(
        gmm2Context, gmm2Args, gmm2Scratch, combineContext, combineBufferConfig, params, gmm2RuntimeState);
}

template <TemplateMegaMoeTypeClass>
class MegaMoe {
public:
    template <int32_t QM>
    struct QuantTraits {
        using OutType = fp8_e4m3fn_t;
    };
    template <>
    struct QuantTraits<E5M2_QUANT> {
        using OutType = fp8_e5m2_t;
    };
    template <>
    struct QuantTraits<E2M1_QUANT> {
        using OutType = fp4x2_e2m1_t;
    };
    using QuantOutType = typename QuantTraits<QuantMode>::OutType;
    using ActivationType =
        typename std::conditional<Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value, uint8_t, QuantOutType>::type;
    using QuantScaleOutType = typename std::conditional<(QuantMode >= E5M2_QUANT), fp8_e8m0_t, float>::type;
    __aicore__ inline MegaMoe(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1,
                                GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
                                GM_ADDR scales, GM_ADDR sharedWeight1, GM_ADDR sharedWeight2,
                                GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2, GM_ADDR yOut,
                                GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData *tilingData);
    __aicore__ inline void Process();

private:
    using SendMaskBufferConfig = MegaMoeSendMaskBufferConfig;
    using UnpermuteBufferConfig = MegaMoeUnpermuteBufferConfig;

    __aicore__ inline SendMaskBufferConfig SendAndQuantBuffInit();
    __aicore__ inline UnpermuteBufferConfig InitTokenUnpermuteBuffers();
    __aicore__ inline void CrossRankSyncInWorldSize();

    __gm__ Mc2MoeContext *mc2Context_{nullptr};
    __gm__ int32_t *gmmToEpilogueFlag_{nullptr};
    Params params_{};
    ExpertWeightTensorListAddrs moeWeightTensorListAddrs_{};
    ExpertWeightTensorListAddrs sharedWeightTensorListAddrs_{};
    DispatchPrepareContext dispatchPrepareContext_;
    TokenDispatchContext tokenDispatchContext_;
    Gmm1SwigluContext gmm1Context_;
    SharedExpertGmm1SwigluContext sharedGmm1Context_;
    SendMaskArgs sendMaskArgs_;
    ResetWorkspaceArgs resetWorkspaceArgs_;
    QuantProcessArgs quantProcessArgs_;
    SharedExpertPrepareArgs sharedExpertPrepareArgs_;
    SharedExpertGmm1SwigluArgs sharedGmm1Args_;
    Gmm2Context gmm2Context_;
    Gmm2Args gmm2Args_;
    Gmm2Context sharedGmm2Context_;
    SharedGmm2Args sharedGmm2Args_;
    QuantCombineContext quantCombineContext_;
    QuantCombineBufferConfig quantCombineBufferConfig_;
    TokenUnpermuteContext tokenUnpermuteContext_;
    TokenUnpermuteArgs tokenUnpermuteArgs_;
    // A8W4 路径下 RunGmm1A8W4 会覆盖 V1 UB，导致 UB 上跨 expert 的状态
    // 无法保持。tokenDispatchScratch_ 中的 cumsumInfoGlobalTensor 作为 GM 持久备份：
    // ComputeExpertTokenCountAndNotify 中 Load → 计算 → Store；Dispatch/Export 从 GM 恢复。

    uint32_t m_ = 0;
    uint32_t k_ = 0;
    uint32_t aicNum_ = 0;
    uint32_t topK_ = 0;
    uint32_t rankId_ = 0;
    uint32_t worldSize_ = 0;
    int64_t hiddenDim_ = 0;
    uint64_t maxOutputSize_ = 0;
    uint32_t blockNumPerRank_ = 2;
    int32_t dispatchFlagSlotsPerExpert_ = 0;
    int32_t maxWavesPerExpert_ = 0;
    uint32_t blockNum_ = GetBlockNum();
    uint32_t blockAivNum_ = GetBlockNum() * 2;
    uint32_t blockIdx_ = GetBlockIdx() / GetTaskRation();
    uint32_t aivCoreIdx_ = GetBlockIdx();
    uint32_t subBlockIdx_ = GetSubBlockIdx();
    uint32_t mxQuantScaleNumAlignPerToken_ = 0;
    uint32_t mxQuantTokenAlignBytes_ = 0;
    uint32_t mxQuantScaleAlignBytes_ = 0;
    uint32_t mxQuantTokenScaleAlignBytes_ = 0;
    uint32_t weightAlignBytes_ = 0;
    uint16_t gmm2PingPongIdx_ = 0;
    uint64_t sendTotalNum_ = 0;
    uint32_t maskAlignSize_ = 0;
    uint32_t maskSlotSize_ = 0;   // 单个 win 槽位 = maskAlignSize_(mask) + 32B(count)
    uint64_t maskWinOffset_ = 0;  // maskRecvPtr 相对 win 基址(rankSyncInWorldPtr)的偏移
    uint64_t quantWinOffset_ = 0; // quantTokenScalePtr 相对 win 基址的偏移
    int32_t compareCount_ = 0;
    // 主线 shared-expert 特性成员
    uint32_t sharedExpertNum_ = 0;
    uint32_t moeExpertPerRank_ = 0;
    bool isPerExpertWeightTensor_ = false;

    // 大 BS reset batch 成员
    int32_t resetBatchElementCount_ = 0; // 每个 reset batch 清零的 int32 元素数（封顶到 DISPATCH_RESET_BATCH）

    static constexpr uint32_t A_ELEMS_PER_BYTE = PackedElementTraits<QuantOutType>::ELEMENTS_PER_BYTE;
    static constexpr uint32_t B_ELEMS_PER_BYTE = PackedElementTraits<Weight1Type>::ELEMENTS_PER_BYTE;
    // ENABLE_A8W4: A8W8 路径（fp8 act + fp4 w1），GMM1 使用 A8W4 prologue（W4→W8 + MMAD）。
    static constexpr bool ENABLE_A8W4 =
        Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value && Std::IsSame<QuantOutType, fp8_e4m3fn_t>::value;
    // ENABLE_A4W4: A4W4 路径（fp4 act + fp4 weight），GMM2 复用 A8W4 prologue。
    //             a4w4 场景下 GMM1 走 generic a4w4、GMM2 走 a8w4，避免两段都用 a4w4 导致精度损失过大。
    static constexpr bool ENABLE_A4W4 =
        Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value && Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value;
    static constexpr uint32_t GMM1_TILE_M = MegaMoeImpl::L1_TILE_M_256;
    static constexpr uint32_t EPILOGUE_TILE_M =
        TopkWeightsPrefetch ? MegaMoeImpl::L1_TILE_M_128 : MegaMoeImpl::L1_TILE_M_256;
    QuantProcessScratch<ActivationType> quantScratch_;
    SharedExpertPrepareScratch<ActivationType> sharedExpertPrepareScratch_;
    SendMaskScratch sendMaskScratch_;
    LocalTensor<int32_t> resetTensor_;

    // GMM2 走 A8W4 且 QuantMode 为 a4w4（E2M1）时，SwigluQuant 输出需提升为 fp8_e4m3fn_t。
    // 同时当 Weight2 非 fp4 但 QuantMode==E2M1 时（generic GMM2 路径），也需 promotion，
    // 否则会出现 A=QuantOutType(fp4) vs B=Weight1Type(fp8) 的类型不匹配。
    using SwigluQuantOutType = typename std::conditional<(QuantMode == E2M1_QUANT), fp8_e4m3fn_t, QuantOutType>::type;

    // SwigluQuant 输出的元素字节密度：fp4 时为 2elem/B，fp8 时为 1elem/B。
    static constexpr uint32_t C_ELEMS_PER_BYTE = PackedElementTraits<SwigluQuantOutType>::ELEMENTS_PER_BYTE;

    using BlockEpilogue =
        BlockEpilogueSwigluMxQuant<SwigluQuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, true,
                                   EPILOGUE_TILE_M, MegaMoeImpl::L1_TILE_N, TopkWeightsPrefetch>;
    using SharedBlockEpilogue =
        BlockEpilogueSwigluMxQuant<SwigluQuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, true,
                                   MegaMoeImpl::L1_TILE_M_256, MegaMoeImpl::L1_TILE_N, false>;
    BlockEpilogue epilogueOp_;
    SharedBlockEpilogue sharedEpilogueOp_;
    TokenDispatchArgs tokenDispatchArgs_;
    Gmm1SwigluArgs gmm1Args_;
    TokenDispatchScratch<ActivationType> tokenDispatchScratch_;
    Gmm1SwigluScratch gmm1SwigluScratch_;
    Gmm2Scratch gmm2Scratch_;
    TokenUnpermuteScratch tokenUnpermuteScratch_;
};

// ========================
// Init：初始化 & 偏移计算
// ========================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales, GM_ADDR sharedWeight1,
    GM_ADDR sharedWeight2, GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2, GM_ADDR yOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData *tilingData)
{
    m_ = tilingData->bs;
    k_ = tilingData->h;
    aicNum_ = tilingData->aicNum;
    topK_ = tilingData->topK;
    sendTotalNum_ = static_cast<uint64_t>(m_) * topK_;
    worldSize_ = tilingData->epWorldSize;
    moeExpertPerRank_ = tilingData->moeExpertPerRank;
    sharedExpertNum_ = tilingData->sharedExpertNum;
    isPerExpertWeightTensor_ = tilingData->isPerExpertWeightTensor;
    blockNumPerRank_ = tilingData->blockNumPerEP;
    maxOutputSize_ = tilingData->maxOutputSize;
    gmm2PingPongIdx_ = 0;
    // 与 WorkspaceInfo 构造里 flagDispatchToGmm1Ptr 的分配公式保持一致。
    maxWavesPerExpert_ = static_cast<int32_t>(
        Ops::Base::CeilDiv(static_cast<int64_t>(maxOutputSize_), static_cast<int64_t>(GMM1_TILE_M)));
    dispatchFlagSlotsPerExpert_ = maxWavesPerExpert_ * INT_CACHELINE;
    hiddenDim_ = tilingData->hiddenDim;
    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext *>(context);
    rankId_ = mc2Context_->epRankId;
    for (int i = 0; i < worldSize_; i++) {
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
    }
    params_.aGmAddr = x;
    params_.expertIdxGmAddr = topkIds;
    moeWeightTensorListAddrs_ = {
        .weight1 = weight1,
        .weightScales1 = weightScales1,
        .weight2 = weight2,
        .weightScales2 = weightScales2};
    sharedWeightTensorListAddrs_ = {
        .weight1 = sharedWeight1,
        .weightScales1 = sharedWeightScales1,
        .weight2 = sharedWeight2,
        .weightScales2 = sharedWeightScales2};
    params_.y2GmAddr = yOut;
    params_.expertTokenNumsOutGmAddr = expertTokenNumsOut;
    params_.probsGmAddr = topkWeights;
    params_.workspaceInfo = WorkspaceInfo(workspaceGM, tilingData);
    params_.peermemInfo = PeermemInfo(winRankAddr_[rankId_], tilingData, A_ELEMS_PER_BYTE);
    params_.tilingData = tilingData;
    epilogueOp_.Init({.yGmAddr = params_.workspaceInfo.swigluQuantDataPtr,
                      .yScaleGmAddr = params_.workspaceInfo.swigluQuantScalePtr,
                      .groupFlagListGmAddr = params_.workspaceInfo.flagSwiGluToGmm2Ptr,
                      .x2ScaleGmAddr = nullptr,
                      .x1ScaleGmAddr = nullptr,
                      .biasGmAddr = nullptr,
                      .metaInfoGmAddr = params_.workspaceInfo.metaInfoPtr,
                      .clampLimit = tilingData->clampLimit,
                      .actMode = tilingData->actMode,
                      .actSubMode = tilingData->actSubMode,
                      .activationAlpha = tilingData->activationAlpha,
                      .activationBeta = tilingData->activationBeta});
    // 各 win 区相对 win 基址(rankSyncInWorldPtr)的偏移; 所有卡 win 布局一致, 跨卡读写用同一偏移。
    maskWinOffset_ = static_cast<uint64_t>(params_.peermemInfo.maskRecvPtr - params_.peermemInfo.rankSyncInWorldPtr);
    quantWinOffset_ =
        static_cast<uint64_t>(params_.peermemInfo.quantTokenScalePtr - params_.peermemInfo.rankSyncInWorldPtr);
    // maskAlignSize_ 必与 PeermemInfo 中 maskAlignSize 公式数值一致。
    compareCount_ =
        Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_256)) /
        sizeof(int32_t);
    maskAlignSize_ = Ops::Base::CeilAlign(static_cast<int64_t>(compareCount_) / 8, static_cast<int64_t>(ALIGN_32));
    // 每个 win 槽位再追加 32B 存 count(源卡 GatherAndSendExpertMasks 同步算好), 须与 PeermemInfo 的 maskSlotSize 一致。
    maskSlotSize_ = maskAlignSize_ + static_cast<uint32_t>(ALIGN_32);
    mxQuantScaleNumAlignPerToken_ = Ops::Base::CeilDiv(k_, static_cast<uint32_t>(ALIGN_32));
    mxQuantTokenAlignBytes_ =
        Ops::Base::CeilAlign(static_cast<uint32_t>(k_ / A_ELEMS_PER_BYTE), static_cast<uint32_t>(ALIGN_256)) *
        sizeof(ActivationType);
    mxQuantScaleAlignBytes_ =
        Ops::Base::CeilAlign(mxQuantScaleNumAlignPerToken_ * static_cast<uint32_t>(sizeof(QuantScaleOutType)),
                             static_cast<uint32_t>(ALIGN_32));
    mxQuantTokenScaleAlignBytes_ = mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_;
    if constexpr (TopkWeightsPrefetch) {
        weightAlignBytes_ =
            Ops::Base::CeilAlign(static_cast<uint32_t>(topK_ * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
        mxQuantTokenScaleAlignBytes_ += weightAlignBytes_;
    }
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        gmmToEpilogueFlag_ = reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.flagGmmToEpiloguePtr) +
                             static_cast<uint64_t>(blockIdx_) * INT_CACHELINE;
    }

    dispatchPrepareContext_ = {
        .job = {.jobIndex = aivCoreIdx_, .totalJobs = blockAivNum_},
        .topology = {.rankId = rankId_, .worldSize = worldSize_, .expertPerRank = moeExpertPerRank_},
        .tokenShape = {.tokenNum = static_cast<int32_t>(m_), .topK = topK_, .tokenHiddenDim = k_}};
    tokenDispatchContext_ = {
        .blockJob = {.jobIndex = blockIdx_, .totalJobs = blockNum_},
        .countWorkspace = {.blockIdx = blockIdx_, .blockNum = aicNum_},
        .worldSize = worldSize_,
        .expertCapacityPerRank = moeExpertPerRank_,
        .tokenHiddenDim = k_,
        .topK = topK_,
        .blockNumPerRank = blockNumPerRank_,
        .maxOutputSize = maxOutputSize_,
        .sendTotalNum = sendTotalNum_,
        .bufferConfig = tilingData->dispatchBufferConfig,
        .maskAlignSize = maskAlignSize_,
        .maskSlotSize = maskSlotSize_,
        .quantWinOffset = quantWinOffset_,
        .quantTokenAlignBytes = mxQuantTokenAlignBytes_,
        .quantScaleAlignBytes = mxQuantScaleAlignBytes_,
        .quantTokenScaleAlignBytes = mxQuantTokenScaleAlignBytes_,
        .dispatchFlagSlotsPerExpert = dispatchFlagSlotsPerExpert_,
        .activationElementsPerByte = A_ELEMS_PER_BYTE};
    gmm1Context_ = {
        .blockJob = {.jobIndex = blockIdx_, .totalJobs = blockNum_},
        .countWorkspace = {.blockIdx = blockIdx_, .blockNum = aicNum_},
        .expertCapacityPerRank = moeExpertPerRank_,
        .expertPerRank = moeExpertPerRank_,
        .tokenHiddenDim = k_,
        .gmm1OutputDim = static_cast<uint32_t>(hiddenDim_),
        .dispatchFlagSlotsPerExpert = dispatchFlagSlotsPerExpert_,
        .swigluFlagSlotsPerExpert = INT_CACHELINE,
        .maxTilesPerExpert = tilingData->maxTilesPerExpert,
        .groupedMatmulMode = tilingData->groupedMatmulMode,
        .activationElementsPerByte = A_ELEMS_PER_BYTE,
        .isPerExpertWeightTensor = isPerExpertWeightTensor_};
    tokenDispatchArgs_ = {
        .winRankAddr = ::winRankAddr_,
        .maskRecvPtr = params_.peermemInfo.maskRecvPtr,
        .expertRevTokenNumsPtr = params_.workspaceInfo.expertRevTokenNumsPtr,
        .metaInfoPtr = params_.workspaceInfo.metaInfoPtr,
        .cumsumInfoPtr = params_.workspaceInfo.cumsumInfoPtr,
        .dispatchRevDataPtr = params_.workspaceInfo.dispatchRevDataPtr,
        .dispatchRevScalePtr = params_.workspaceInfo.dispatchRevScalePtr,
        .dispatchToGmm1FlagPtr = params_.workspaceInfo.flagDispatchToGmm1Ptr,
        .sendCntFlagPtr = params_.workspaceInfo.flagSendCntCalToUpdParamsPtr};
    gmm1Args_ = {
        .expertRevTokenNumsPtr = params_.workspaceInfo.expertRevTokenNumsPtr,
        .dispatchRevDataPtr = params_.workspaceInfo.dispatchRevDataPtr,
        .dispatchRevScalePtr = params_.workspaceInfo.dispatchRevScalePtr,
        .gmm1MmadResPtr = params_.workspaceInfo.gmm1MmadResPtr,
        .weight1TensorListAddr = moeWeightTensorListAddrs_.weight1,
        .weight1ScaleTensorListAddr = moeWeightTensorListAddrs_.weightScales1,
        .dispatchToGmm1FlagPtr = params_.workspaceInfo.flagDispatchToGmm1Ptr,
        .sendCntFlagPtr = params_.workspaceInfo.flagSendCntCalToUpdParamsPtr,
        .gmm1TileStatusPtr = params_.workspaceInfo.gmm1TileStatusPtr,
        .gmmToEpilogueFlagPtr = reinterpret_cast<GM_ADDR>(gmmToEpilogueFlag_)};
    sharedGmm1Context_ = {
        .blockJob = {.jobIndex = blockIdx_, .totalJobs = blockNum_},
        .sharedExpertNum = sharedExpertNum_,
        .tokenNum = m_,
        .tokenHiddenDim = k_,
        .gmm1OutputDim = static_cast<uint32_t>(hiddenDim_),
        .groupedMatmulMode = tilingData->groupedMatmulMode,
        .isPerExpertWeightTensor = isPerExpertWeightTensor_};
    sharedGmm1Args_ = {
        .inputDataPtr = params_.workspaceInfo.sharedExpertInputDataPtr,
        .inputScalePtr = params_.workspaceInfo.sharedExpertInputScalePtr,
        .gmm1MmadResultPtr = params_.workspaceInfo.sharedExpertGmm1OutPtr,
        .weight1TensorListAddr = sharedWeightTensorListAddrs_.weight1,
        .weight1ScaleTensorListAddr = sharedWeightTensorListAddrs_.weightScales1,
        .swigluOutputDataPtr = params_.workspaceInfo.sharedExpertSwigluDataPtr,
        .swigluOutputScalePtr = params_.workspaceInfo.sharedExpertSwigluScalePtr,
        .gmmToEpilogueFlagPtr = reinterpret_cast<GM_ADDR>(gmmToEpilogueFlag_),
        .clampLimit = tilingData->clampLimit,
        .actMode = tilingData->actMode,
        .actSubMode = tilingData->actSubMode,
        .activationAlpha = tilingData->activationAlpha,
        .activationBeta = tilingData->activationBeta};
    const SendMaskBufferConfig &sendMaskBufferConfig = aivCoreIdx_ < tilingData->sendMaskCoreCountWithExtraExpert ?
                                                           tilingData->sendMaskConfigForCoreWithExtraExpert :
                                                           tilingData->sendMaskConfigForCoreWithoutExtraExpert;
    sendMaskArgs_ = {
        .expertIdxGmAddr = params_.expertIdxGmAddr,
        .winRankAddr = ::winRankAddr_,
        .maskAlignSize = maskAlignSize_,
        .maskSlotSize = maskSlotSize_,
        .maskWinOffset = maskWinOffset_,
        .bufferConfig = sendMaskBufferConfig};
    int32_t resetFlagNum =
        static_cast<int32_t>(CalcMegaMoeFlagWorkspaceSize(params_.tilingData) / sizeof(int32_t));
    resetWorkspaceArgs_ = {
        .flagWorkspacePtr = params_.workspaceInfo.flagSwiGluToGmm2Ptr,
        .gmm1TileStatusPtr = params_.workspaceInfo.gmm1TileStatusPtr,
        .flagElementCount = resetFlagNum,
        .gmm1StatusExpertCapacity = static_cast<int32_t>(moeExpertPerRank_),
        .maxTilesPerExpert = static_cast<int32_t>(tilingData->maxTilesPerExpert),
        .resetBatchElementCount = 0};
    quantProcessArgs_ = {
        .xGmAddr = params_.aGmAddr,
        .topkWeightsGmAddr = params_.probsGmAddr,
        .quantTokenScalePtr = params_.peermemInfo.quantTokenScalePtr,
        .quantTokenAlignBytes = mxQuantTokenAlignBytes_,
        .quantScaleAlignBytes = mxQuantScaleAlignBytes_,
        .quantTokenScaleAlignBytes = mxQuantTokenScaleAlignBytes_,
        .quantScaleNumAlignPerToken = mxQuantScaleNumAlignPerToken_,
        .topK = topK_};
    sharedExpertPrepareArgs_ = {
        .quantTokenScalePtr = params_.peermemInfo.quantTokenScalePtr,
        .sharedExpertInputDataPtr = params_.workspaceInfo.sharedExpertInputDataPtr,
        .sharedExpertInputScalePtr = params_.workspaceInfo.sharedExpertInputScalePtr,
        .quantTokenAlignBytes = mxQuantTokenAlignBytes_,
        .quantScaleAlignBytes = mxQuantScaleAlignBytes_,
        .quantTokenScaleAlignBytes = mxQuantTokenScaleAlignBytes_,
        .activationElementsPerByte = A_ELEMS_PER_BYTE};
    gmm2Context_ = {
        .blockJob = {.jobIndex = blockIdx_, .totalJobs = blockNum_},
        .countWorkspace = {.blockIdx = blockIdx_, .blockNum = aicNum_},
        .tokenNum = m_,
        .expertPerRank = moeExpertPerRank_,
        .tokenHiddenDim = k_,
        .gmm1OutputDim = static_cast<uint32_t>(hiddenDim_),
        .swigluFlagSlotsPerExpert = INT_CACHELINE,
        .dispatchFlagSlotsPerExpert = dispatchFlagSlotsPerExpert_,
        .combineSyncSlotCountPerExpert = tilingData->combineSyncSlotCountPerExpert,
        .groupedMatmulMode = tilingData->groupedMatmulMode,
        .isPerExpertWeightTensor = isPerExpertWeightTensor_};
    gmm2Args_ = {
        .expertRevTokenNumsPtr = params_.workspaceInfo.expertRevTokenNumsPtr,
        .swigluDataPtr = params_.workspaceInfo.swigluQuantDataPtr,
        .swigluScalePtr = params_.workspaceInfo.swigluQuantScalePtr,
        .gmm2OutputPtr = params_.workspaceInfo.gmm2MmadResPtr,
        .metaInfoPtr = params_.workspaceInfo.metaInfoPtr,
        .weight2TensorListAddr = moeWeightTensorListAddrs_.weight2,
        .weight2ScaleTensorListAddr = moeWeightTensorListAddrs_.weightScales2,
        .swigluToGmm2FlagPtr = params_.workspaceInfo.flagSwiGluToGmm2Ptr,
        .dispatchToGmm1FlagPtr = params_.workspaceInfo.flagDispatchToGmm1Ptr,
        .gmm2CombineSyncCounterPtr = params_.workspaceInfo.gmm2CombineSyncCounterPtr,
        .gmmToEpilogueFlagPtr = reinterpret_cast<GM_ADDR>(gmmToEpilogueFlag_)};
    sharedGmm2Context_ = gmm2Context_;
    sharedGmm2Context_.expertPerRank = sharedExpertNum_;
    sharedGmm2Args_ = {
        .swigluDataPtr = params_.workspaceInfo.sharedExpertSwigluDataPtr,
        .swigluScalePtr = params_.workspaceInfo.sharedExpertSwigluScalePtr,
        .resultPtr = params_.workspaceInfo.sharedExpertResultPtr,
        .weight2TensorListAddr = sharedWeightTensorListAddrs_.weight2,
        .weight2ScaleTensorListAddr = sharedWeightTensorListAddrs_.weightScales2,
        .tileCounterPtr = params_.workspaceInfo.sharedExpertGmm2TileCounterPtr,
        .gmmToEpilogueFlagPtr = reinterpret_cast<GM_ADDR>(gmmToEpilogueFlag_)};
    constexpr bool pairedAivCombine = ENABLE_A8W4 || ENABLE_A4W4;
    uint32_t gmm2NTilesPerGroup = 0U;
    if constexpr (g_coreType == AIV) {
        if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
            gmm2NTilesPerGroup = Ops::Base::CeilDiv(k_, L1_TILE_N);
        } else if (sharedExpertNum_ > 0U) {
            gmm2NTilesPerGroup = Ops::Base::CeilDiv(k_, L1_TILE_N);
        }
    }
    quantCombineContext_ = {
        .job = {.jobIndex = pairedAivCombine ? blockIdx_ : aivCoreIdx_,
                .totalJobs = pairedAivCombine ? blockNum_ : blockAivNum_},
        .tokenHiddenDim = k_,
        .rankId = rankId_,
        .gmm2NTilesPerGroup = gmm2NTilesPerGroup,
        .participates = !pairedAivCombine || subBlockIdx_ == 1U};
    quantCombineBufferConfig_ = {.combineUbElementCount = 0, .quantTokenSizeBytes = 0U};
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV) {
        quantCombineBufferConfig_ = CreateQuantCombineBufferConfig(k_);
    }
    tokenUnpermuteContext_ = {
        .job = {.jobIndex = aivCoreIdx_, .totalJobs = blockAivNum_},
        .tokenNum = m_,
        .tokenHiddenDim = k_,
        .topK = topK_,
        .sharedExpertNum = sharedExpertNum_,
        .gmm2NTilesPerGroup = gmm2NTilesPerGroup,
        .quantTokenSizeBytes = quantCombineBufferConfig_.quantTokenSizeBytes,
        .fullTokenChunkJobCount = tilingData->unpermuteFullTokenChunkCoreCount,
        .fullTokenChunkConfig = tilingData->unpermuteConfigForFullTokenChunk,
        .tailTokenChunkConfig = tilingData->unpermuteConfigForTailTokenChunk};
    tokenUnpermuteArgs_ = {
        .topkWeightsPtr = params_.probsGmAddr,
        .expandedXPtr = params_.peermemInfo.combineSendPtr,
        .outputPtr = params_.y2GmAddr,
        .sharedExpertResultPtr = params_.workspaceInfo.sharedExpertResultPtr,
        .sharedExpertGmm2TileCounterPtr = params_.workspaceInfo.sharedExpertGmm2TileCounterPtr};
}

// ======================================================================================
// SendAndQuantBuffInit：单核 mask/reset/quant/shared-prepare 模块使用的 buffer 申请。
//   shared prepare 复用 quant 输出双 buffer；reset 封顶 DISPATCH_RESET_BATCH。
// ======================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline typename MegaMoe<TemplateMegaMoeTypeFunc>::SendMaskBufferConfig
MegaMoe<TemplateMegaMoeTypeFunc>::SendAndQuantBuffInit()
{
    SendMaskBufferConfig bufferConfig{};
    if constexpr (g_coreType == AIC) {
        return bufferConfig;
    }

    // 与 route batch 无关的固定占用
    uint64_t totalFlagInt32 = static_cast<uint64_t>(resetWorkspaceArgs_.flagElementCount);
    if constexpr (TopkWeightsPrefetch) {
        uint64_t statusElementCount =
            (static_cast<uint64_t>(resetWorkspaceArgs_.gmm1StatusExpertCapacity) *
                 static_cast<uint64_t>(resetWorkspaceArgs_.maxTilesPerExpert) +
             1U) *
            INT_CACHELINE;
        totalFlagInt32 = totalFlagInt32 > statusElementCount ? totalFlagInt32 : statusElementCount;
    }
    uint32_t resetElementCountPerCore = Ops::Base::CeilDiv(totalFlagInt32, static_cast<uint64_t>(blockAivNum_));
    resetBatchElementCount_ = resetElementCountPerCore < static_cast<uint32_t>(DISPATCH_RESET_BATCH) ?
                                  static_cast<int32_t>(resetElementCountPerCore) :
                                  DISPATCH_RESET_BATCH;
    uint32_t resetTensorSize =
        Ops::Base::CeilAlign(static_cast<uint64_t>(resetBatchElementCount_), static_cast<uint64_t>(INT32_PER_256B)) *
        sizeof(int32_t);

    uint32_t mxTempTensorSize = 2 * 1024;
    uint32_t xOutTensorSize = quantProcessArgs_.quantTokenScaleAlignBytes;
    uint32_t xInAlignSize = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_128)) * sizeof(bfloat16_t);
    uint32_t expertPerCoreMax = Ops::Base::CeilDiv(worldSize_ * moeExpertPerRank_, blockAivNum_);
    uint32_t sendCntAccSize =
        Ops::Base::CeilAlign(static_cast<int64_t>(expertPerCoreMax * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));

    // 必须与 host SetAdaptiveBufferConfigs 的 quotient/remainder 分核保持一致。GatherAndSendExpertMasks 按
    // expertId = aivCoreIdx_ + ownedIdx * blockAivNum_ 遍历，因此前 remainder 个 core 多处理一个 expert。
    bufferConfig = sendMaskArgs_.bufferConfig;
    int32_t routeItemsPerBatch = bufferConfig.routeItemsPerBatch;

    // 按既定顺序落地址。routeItemsPerBatch 按 256 个 item 对齐，因此两个 int32 tensor 均天然满足 256B 对齐。
    uint32_t topkIdsTensorAddr = 0;
    uint32_t topkIdsTensorSize = static_cast<uint32_t>(routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));
    sendMaskScratch_.topkIdsTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, topkIdsTensorAddr, topkIdsTensorSize / sizeof(int32_t));

    uint32_t resetAddrActual = topkIdsTensorAddr + topkIdsTensorSize;
    resetTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, resetAddrActual, resetTensorSize / sizeof(int32_t));
    Duplicate<int32_t>(resetTensor_, 0, (resetTensorSize / sizeof(int32_t)));
    resetWorkspaceArgs_.resetBatchElementCount = resetBatchElementCount_;

    uint32_t mxTempTensorAddr = resetAddrActual + resetTensorSize;
    quantScratch_.mxTempTensor =
        LocalTensor<uint16_t>(TPosition::VECCALC, mxTempTensorAddr, mxTempTensorSize / sizeof(uint16_t));

    uint32_t xOutTensorAddr1 = mxTempTensorAddr + mxTempTensorSize;
    quantScratch_.xOutTensor0 =
        LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr1, xOutTensorSize / sizeof(ActivationType));
    sharedExpertPrepareScratch_.copyBuffer0 = quantScratch_.xOutTensor0;
    uint32_t xOutTensorAddr2 = xOutTensorAddr1 + xOutTensorSize;
    quantScratch_.xOutTensor1 =
        LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr2, xOutTensorSize / sizeof(ActivationType));
    sharedExpertPrepareScratch_.copyBuffer1 = quantScratch_.xOutTensor1;

    uint32_t xInAlignAddr1 = xOutTensorAddr2 + xOutTensorSize;
    quantScratch_.xInTensor0 =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr1, xInAlignSize / sizeof(bfloat16_t));
    uint32_t xInAlignAddr2 = xInAlignAddr1 + xInAlignSize;
    quantScratch_.xInTensor1 =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr2, xInAlignSize / sizeof(bfloat16_t));

    uint32_t sendMaskAddr = xInAlignAddr2 + xInAlignSize;
    uint32_t sendGatherOutSize = static_cast<uint32_t>(routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));

    uint32_t sendMaskTotalBytes = static_cast<uint32_t>(bufferConfig.bufferCount) * bufferConfig.bufferBytes;
    sendMaskScratch_.sendMaskTensor = LocalTensor<uint8_t>(TPosition::VECCALC, sendMaskAddr, sendMaskTotalBytes);
    uint32_t sendGatherOutAddr = sendMaskAddr + sendMaskTotalBytes;
    sendMaskScratch_.sendGatherOutTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, sendGatherOutAddr, sendGatherOutSize / sizeof(int32_t));
    uint32_t sendCntAccAddr = sendGatherOutAddr + sendGatherOutSize;
    sendMaskScratch_.sendCntAccTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, sendCntAccAddr, sendCntAccSize / sizeof(int32_t));
    return bufferConfig;
}

// 在普通模板内构造 Unpermute 使用的 UB 视图，并返回当前 AIV 对应的 buffer 配置。
template <TemplateMegaMoeTypeClass>
__aicore__ inline typename MegaMoe<TemplateMegaMoeTypeFunc>::UnpermuteBufferConfig
MegaMoe<TemplateMegaMoeTypeFunc>::InitTokenUnpermuteBuffers()
{
    const TokenUnpermuteContext &context = tokenUnpermuteContext_;
    TokenUnpermuteScratch &scratch = tokenUnpermuteScratch_;
    UnpermuteBufferConfig bufferConfig =
        context.job.jobIndex < context.fullTokenChunkJobCount ? context.fullTokenChunkConfig :
                                                                  context.tailTokenChunkConfig;

    uint32_t bf16ScaleBufAlign = 0U;
    uint32_t fp32ScaleBufAlign = 0U;
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        uint32_t scaleNum = Ops::Base::CeilDiv(context.tokenHiddenDim, static_cast<uint32_t>(ALIGN_32));
        bf16ScaleBufAlign = Ops::Base::CeilAlign(
            scaleNum * static_cast<uint32_t>(sizeof(bfloat16_t)) *
                static_cast<uint32_t>(DEQUANT_BF16_SCALE_EXPANSION),
            static_cast<uint32_t>(ALIGN_32));
        fp32ScaleBufAlign = Ops::Base::CeilAlign(
            scaleNum * static_cast<uint32_t>(sizeof(float)) *
                static_cast<uint32_t>(DEQUANT_FP32_SCALE_EXPANSION),
            static_cast<uint32_t>(ALIGN_32));
    }

    uint32_t bf16SlotBytes = bufferConfig.bf16SlotElementCount * sizeof(bfloat16_t);
    uint32_t fp32SlotBytes = bufferConfig.fp32SlotElementCount * sizeof(float);
    uint32_t dataResBufBytes = (bufferConfig.inputBufferCount + 1) * bf16SlotBytes;
    uint32_t dataResFp32BufBytes = (bufferConfig.inputBufferCount + 1) * fp32SlotBytes;
    uint32_t dataResAddr = 0U;
    scratch.dataResTensor = LocalTensor<bfloat16_t>(
        TPosition::VECCALC, dataResAddr, dataResBufBytes / sizeof(bfloat16_t));
    uint32_t dataResFp32Addr = dataResAddr + dataResBufBytes;
    scratch.dataResFp32Tensor = LocalTensor<float>(
        TPosition::VECCALC, dataResFp32Addr, dataResFp32BufBytes / sizeof(float));
    uint32_t tempAddr = dataResFp32Addr + dataResFp32BufBytes;

    scratch.topKWeightsTensor = LocalTensor<float>(
        TPosition::VECCALC, tempAddr, bufferConfig.topKWeightsBufferBytes / sizeof(float));
    tempAddr += bufferConfig.topKWeightsBufferBytes;
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        scratch.topKWeightsBf16Tensor = LocalTensor<bfloat16_t>(
            TPosition::VECCALC, tempAddr,
            bufferConfig.topKWeightsConversionBufferBytes / sizeof(bfloat16_t));
        tempAddr += bufferConfig.topKWeightsConversionBufferBytes;
    }
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        scratch.bf16ScaleTensor = LocalTensor<bfloat16_t>(
            TPosition::VECCALC, tempAddr, bf16ScaleBufAlign / sizeof(bfloat16_t));
        tempAddr += bf16ScaleBufAlign;
        scratch.fp32ScaleTensor = LocalTensor<float>(
            TPosition::VECCALC, tempAddr, fp32ScaleBufAlign / sizeof(float));
    }
    return bufferConfig;
}

// 全卡同步：各 AIV 按物理核号分摊 rank 握手，末尾执行本卡 AIV 同步。
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::CrossRankSyncInWorldSize()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    __gm__ int32_t *syncRank = reinterpret_cast<__gm__ int32_t *>(params_.peermemInfo.rankSyncInWorldPtr);
    __gm__ int32_t *syncCount = reinterpret_cast<__gm__ int32_t *>(
        params_.peermemInfo.rankSyncInWorldPtr + 48U * 1024U + static_cast<uint64_t>(aivCoreIdx_) * 64U);
    int32_t count = ReadGmByPassDCache(syncCount) + 1;
    for (uint32_t rankIdx = aivCoreIdx_; rankIdx < worldSize_; rankIdx += blockAivNum_) {
        __gm__ int32_t *remoteSyncAddr =
            reinterpret_cast<__gm__ int32_t *>(::winRankAddr_[rankIdx]) + rankId_ * INT_CACHELINE;
        WriteGmByPassDCache(remoteSyncAddr, count);
        GmSignalWaitBarrier(syncRank + rankIdx * INT_CACHELINE, count);
    }
    WriteGmByPassDCache(syncCount, count);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Process()
{
    // 1.本卡数据处理
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    SendAndQuantBuffInit();

    // Phase 1: 量化 + 共享专家输入拆分 + mask/reset (AIV)
    QuantizeLocalTokens<QuantMode, QuantOutType, ActivationType, TopkWeightsType, TopkWeightsPrefetch>(
        dispatchPrepareContext_, quantProcessArgs_, quantScratch_);
    GatherAndSendExpertMasks(dispatchPrepareContext_, sendMaskArgs_, sendMaskScratch_);
    ResetDispatchWorkspace<TopkWeightsPrefetch>(
        dispatchPrepareContext_, resetWorkspaceArgs_, resetTensor_);
    if (sharedExpertNum_ > 0) {
        PrepareSharedExpertInput<ActivationType, QuantScaleOutType>(
            dispatchPrepareContext_, sharedExpertPrepareArgs_, sharedExpertPrepareScratch_);
    }
    if constexpr (g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
    }
    SyncAll<false>(); // aic需要等待flag位reset清理完成

    int32_t gmTileSequence = 0;
    if (sharedExpertNum_ > 0) {
        uint32_t sharedGmm1StartBlockIdx = 0U;
        int32_t sharedGmm1VecSetSyncCom = 0;
        uint16_t sharedGmm1PingPongIdx = 0U;
        SharedExpertGmm1SwigluState sharedGmm1State{
            sharedGmm1StartBlockIdx, sharedGmm1VecSetSyncCom, gmTileSequence, sharedGmm1PingPongIdx};
        SharedExpertGmm1SwigluPipeline<QuantOutType, ActivationType, Weight1Type, SwigluQuantOutType,
                                              QuantScaleOutType, ENABLE_A8W4, GMM1_TILE_M>(
            sharedGmm1Context_, sharedGmm1Args_, params_, sharedEpilogueOp_, sharedGmm1State);
    }

    CrossRankSyncInWorldSize();

    // 2. MoE 专家 Dispatch -> GMM1/SwiGLU -> GMM2/Combine。
    ProcessMoeExperts<CombineQuantMode, QuantOutType, ActivationType, Weight1Type,
                      SwigluQuantOutType, QuantScaleOutType, ENABLE_A8W4, ENABLE_A4W4,
                      GMM1_TILE_M, EPILOGUE_TILE_M, TopkWeightsPrefetch>(
        tokenDispatchContext_, tokenDispatchArgs_, gmm1Context_, gmm1Args_, gmm2Context_, gmm2Args_,
        quantCombineContext_, quantCombineBufferConfig_, params_, epilogueOp_, tokenDispatchScratch_,
        gmm1SwigluScratch_, gmm2Scratch_, params_.expertTokenNumsOutGmAddr, gmTileSequence, gmm2PingPongIdx_);

    if constexpr (g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
    }

    if (sharedExpertNum_ > 0) {
        uint32_t sharedGmm2StartBlockIdx = 0U;
        int32_t sharedGmm2VecSetSyncCom = 0;
        Gmm2RuntimeState sharedGmm2RuntimeState{
            sharedGmm2StartBlockIdx, sharedGmm2VecSetSyncCom, gmTileSequence, gmm2PingPongIdx_};
        ProcessSharedExpertGmm2<QuantOutType, SwigluQuantOutType, Weight1Type, QuantScaleOutType,
                                ENABLE_A8W4, ENABLE_A4W4, GMM1_TILE_M, false>(
            sharedGmm2Context_, sharedGmm2Args_, params_, sharedGmm2RuntimeState);
    }

    // 3. 本卡数据Unpermute
    if constexpr (g_coreType == AIV) {
        CrossRankSyncInWorldSize();
        UnpermuteBufferConfig unpermuteBufferConfig = InitTokenUnpermuteBuffers();
        UnpermuteTokens<CombineQuantMode, TopkWeightsType, TopkWeightsPrefetch, GMM1_TILE_M>(
            tokenUnpermuteContext_, tokenUnpermuteArgs_, tokenUnpermuteScratch_, unpermuteBufferConfig);
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

} // namespace MegaMoeImpl
#undef TemplateMegaMoeTypeClass
#undef TemplateMegaMoeTypeFunc
#endif
