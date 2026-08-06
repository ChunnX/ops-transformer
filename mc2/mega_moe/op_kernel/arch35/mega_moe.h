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
#include "mega_moe_base.h"
#include "mega_moe_workspace_info.h"
#include "block_epilogue_swiglu_mx_quant.h"
#include "mega_moe_impl.h"
#include "mega_moe_job_context.h"
#include "aiv_compute/mega_moe_quant_process.h"
#include "aiv_compute/mega_moe_shared_expert_prepare.h"
#include "aiv_compute/mega_moe_expert_token_count.h"
#include "aiv_comm/mega_moe_send_mask.h"
#include "mega_moe_workspace_reset.h"
#include "mega_moe_routed_expert_pipeline.h"
#include "mega_moe_shared_expert_gmm1_swiglu.h"
#include "mega_moe_unpermute.h"
#include "aiv_comm/mega_moe_rank_sync.h"
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

    __aicore__ inline SendMaskBufferConfig SendAndQuantBuffInit();

    __gm__ Mc2MoeContext *mc2Context_{nullptr};
    __gm__ int32_t *gmmToEpilogueFlag_{nullptr};
    Params params_{};
    ExpertWeightTensorListAddrs moeWeightTensorListAddrs_{};
    ExpertWeightTensorListAddrs sharedWeightTensorListAddrs_{};
    DispatchPrepareContext dispatchPrepareContext_;
    DispatchGmm1SwigluContext gmm1PipelineContext_;
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
    RankSyncContext rankSyncContext_;
    RankSyncArgs rankSyncArgs_;
    // A8W4 路径下 GroupMatmulSwigluQuant 会覆盖 V1 UB，导致 UB 上跨 expert 的状态
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
    DispatchGmm1SwigluArgs gmm1PipelineArgs_;
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
    moeWeightTensorListAddrs_ = {weight1, weightScales1, weight2, weightScales2};
    sharedWeightTensorListAddrs_ =
        {sharedWeight1, sharedWeightScales1, sharedWeight2, sharedWeightScales2};
    params_.y2GmAddr = yOut;
    params_.expertTokenNumsOutGmAddr = expertTokenNumsOut;
    params_.probsGmAddr = topkWeights;
    params_.workspaceInfo = WorkspaceInfo(workspaceGM, tilingData);
    params_.peermemInfo = PeermemInfo(winRankAddr_[rankId_], tilingData, A_ELEMS_PER_BYTE);
    params_.tilingData = tilingData;
    epilogueOp_.Init({params_.workspaceInfo.swigluQuantDataPtr, params_.workspaceInfo.swigluQuantScalePtr,
                      params_.workspaceInfo.flagSwiGluToGmm2Ptr, nullptr, nullptr, nullptr,
                      params_.workspaceInfo.metaInfoPtr, tilingData->clampLimit, tilingData->actMode,
                      tilingData->actSubMode, tilingData->activationAlpha, tilingData->activationBeta});
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
        {aivCoreIdx_, blockAivNum_}, {rankId_, worldSize_, moeExpertPerRank_}, {static_cast<int32_t>(m_), topK_, k_}};
    gmm1PipelineContext_ = {
        {{blockIdx_, blockNum_},
         {blockIdx_, aicNum_},
         worldSize_,
         moeExpertPerRank_,
         k_,
         topK_,
         blockNumPerRank_,
         maxOutputSize_,
         sendTotalNum_,
         tilingData->dispatchBufferConfig,
         maskAlignSize_,
         maskSlotSize_,
         quantWinOffset_,
         mxQuantTokenAlignBytes_,
         mxQuantScaleAlignBytes_,
         mxQuantTokenScaleAlignBytes_,
         dispatchFlagSlotsPerExpert_,
         A_ELEMS_PER_BYTE},
        {{blockIdx_, blockNum_},
         {blockIdx_, aicNum_},
         moeExpertPerRank_,
         moeExpertPerRank_,
         k_,
         static_cast<uint32_t>(hiddenDim_),
         dispatchFlagSlotsPerExpert_,
         INT_CACHELINE,
         tilingData->maxTilesPerExpert,
         tilingData->groupedMatmulMode,
         A_ELEMS_PER_BYTE,
         isPerExpertWeightTensor_}};
    gmm1PipelineArgs_ = {
        {::winRankAddr_,
         params_.peermemInfo.maskRecvPtr,
         params_.workspaceInfo.expertRevTokenNumsPtr,
         params_.workspaceInfo.metaInfoPtr,
         params_.workspaceInfo.cumsumInfoPtr,
         params_.workspaceInfo.dispatchRevDataPtr,
         params_.workspaceInfo.dispatchRevScalePtr,
         params_.workspaceInfo.flagDispatchToGmm1Ptr,
         params_.workspaceInfo.flagSendCntCalToUpdParamsPtr},
        {params_.workspaceInfo.expertRevTokenNumsPtr,
         params_.workspaceInfo.dispatchRevDataPtr,
         params_.workspaceInfo.dispatchRevScalePtr,
         params_.workspaceInfo.gmm1MmadResPtr,
         moeWeightTensorListAddrs_.weight1,
         moeWeightTensorListAddrs_.weightScales1,
         params_.workspaceInfo.flagDispatchToGmm1Ptr,
         params_.workspaceInfo.flagSendCntCalToUpdParamsPtr,
         params_.workspaceInfo.gmm1TileStatusPtr,
         reinterpret_cast<GM_ADDR>(gmmToEpilogueFlag_)}};
    sharedGmm1Context_ = {{blockIdx_, blockNum_},
                          sharedExpertNum_,
                          m_,
                          k_,
                          static_cast<uint32_t>(hiddenDim_),
                          tilingData->groupedMatmulMode,
                          isPerExpertWeightTensor_};
    sharedGmm1Args_ = {params_.workspaceInfo.sharedExpertInputDataPtr,
                       params_.workspaceInfo.sharedExpertInputScalePtr,
                       params_.workspaceInfo.sharedExpertGmm1OutPtr,
                       sharedWeightTensorListAddrs_.weight1,
                       sharedWeightTensorListAddrs_.weightScales1,
                       params_.workspaceInfo.sharedExpertSwigluDataPtr,
                       params_.workspaceInfo.sharedExpertSwigluScalePtr,
                       reinterpret_cast<GM_ADDR>(gmmToEpilogueFlag_),
                       tilingData->clampLimit,
                       tilingData->actMode,
                       tilingData->actSubMode,
                       tilingData->activationAlpha,
                       tilingData->activationBeta};
    const SendMaskBufferConfig &sendMaskBufferConfig = aivCoreIdx_ < tilingData->sendMaskCoreCountWithExtraExpert ?
                                                           tilingData->sendMaskConfigForCoreWithExtraExpert :
                                                           tilingData->sendMaskConfigForCoreWithoutExtraExpert;
    sendMaskArgs_ = {params_.expertIdxGmAddr, ::winRankAddr_, maskAlignSize_,
                     maskSlotSize_, maskWinOffset_, sendMaskBufferConfig};
    int32_t resetFlagNum =
        static_cast<int32_t>(CalcMegaMoeFlagWorkspaceSize(params_.tilingData) / sizeof(int32_t));
    int32_t gmm2CombineSyncCounterNum = 0;
    resetWorkspaceArgs_ = {params_.workspaceInfo.flagSwiGluToGmm2Ptr,
                           params_.workspaceInfo.gmm2CombineSyncCounterPtr,
                           params_.workspaceInfo.gmm1TileStatusPtr,
                           resetFlagNum,
                           gmm2CombineSyncCounterNum,
                           static_cast<int32_t>(moeExpertPerRank_),
                           static_cast<int32_t>(tilingData->maxTilesPerExpert),
                           0};
    quantProcessArgs_ = {params_.aGmAddr,
                         params_.probsGmAddr,
                         params_.peermemInfo.quantTokenScalePtr,
                         mxQuantTokenAlignBytes_,
                         mxQuantScaleAlignBytes_,
                         mxQuantTokenScaleAlignBytes_,
                         mxQuantScaleNumAlignPerToken_,
                         topK_};
    sharedExpertPrepareArgs_ = {params_.peermemInfo.quantTokenScalePtr,
                                params_.workspaceInfo.sharedExpertInputDataPtr,
                                params_.workspaceInfo.sharedExpertInputScalePtr,
                                mxQuantTokenAlignBytes_,
                                mxQuantScaleAlignBytes_,
                                mxQuantTokenScaleAlignBytes_,
                                A_ELEMS_PER_BYTE};
    gmm2Context_ = {{blockIdx_, blockNum_},
                    {blockIdx_, aicNum_},
                    m_,
                    moeExpertPerRank_,
                    k_,
                    static_cast<uint32_t>(hiddenDim_),
                    INT_CACHELINE,
                    dispatchFlagSlotsPerExpert_,
                    tilingData->combineSyncSlotCountPerExpert,
                    tilingData->groupedMatmulMode,
                    isPerExpertWeightTensor_};
    gmm2Args_ = {params_.workspaceInfo.expertRevTokenNumsPtr,
                 params_.workspaceInfo.swigluQuantDataPtr,
                 params_.workspaceInfo.swigluQuantScalePtr,
                 params_.workspaceInfo.gmm2MmadResPtr,
                 moeWeightTensorListAddrs_.weight2,
                 moeWeightTensorListAddrs_.weightScales2,
                 params_.workspaceInfo.flagSwiGluToGmm2Ptr,
                 params_.workspaceInfo.flagDispatchToGmm1Ptr,
                 params_.workspaceInfo.gmm2CombineSyncCounterPtr,
                 reinterpret_cast<GM_ADDR>(gmmToEpilogueFlag_)};
    sharedGmm2Context_ = gmm2Context_;
    sharedGmm2Context_.expertPerRank = sharedExpertNum_;
    sharedGmm2Args_ = {params_.workspaceInfo.sharedExpertSwigluDataPtr,
                       params_.workspaceInfo.sharedExpertSwigluScalePtr,
                       params_.workspaceInfo.sharedExpertResultPtr,
                       sharedWeightTensorListAddrs_.weight2,
                       sharedWeightTensorListAddrs_.weightScales2,
                       params_.workspaceInfo.sharedExpertGmm2TileCounterPtr,
                       reinterpret_cast<GM_ADDR>(gmmToEpilogueFlag_)};
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
        {pairedAivCombine ? blockIdx_ : aivCoreIdx_, pairedAivCombine ? blockNum_ : blockAivNum_},
        k_,
        rankId_,
        gmm2NTilesPerGroup,
        !pairedAivCombine || subBlockIdx_ == 1U};
    quantCombineBufferConfig_ = {0, 0U};
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV) {
        quantCombineBufferConfig_ = CreateQuantCombineBufferConfig(k_);
    }
    tokenUnpermuteContext_ = {{aivCoreIdx_, blockAivNum_},
                              m_,
                              k_,
                              topK_,
                              sharedExpertNum_,
                              gmm2NTilesPerGroup,
                              quantCombineBufferConfig_.quantTokenSizeBytes,
                              tilingData->unpermuteFullTokenChunkCoreCount,
                              tilingData->unpermuteConfigForFullTokenChunk,
                              tilingData->unpermuteConfigForTailTokenChunk};
    tokenUnpermuteArgs_ = {params_.probsGmAddr,
                           params_.peermemInfo.combineSendPtr,
                           params_.y2GmAddr,
                           params_.workspaceInfo.sharedExpertResultPtr,
                           params_.workspaceInfo.sharedExpertGmm2TileCounterPtr};
    rankSyncContext_ = {{aivCoreIdx_, blockAivNum_}, rankId_, worldSize_};
    rankSyncArgs_ = {::winRankAddr_, params_.peermemInfo.rankSyncInWorldPtr};
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
    uint64_t totalFlagInt32 = static_cast<uint64_t>(resetWorkspaceArgs_.flagNum);
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        uint64_t combineCounterNum = static_cast<uint64_t>(resetWorkspaceArgs_.gmm2CombineSyncCounterNum);
        totalFlagInt32 = totalFlagInt32 > combineCounterNum ? totalFlagInt32 : combineCounterNum;
    }
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
    ResetDispatchWorkspace<false, TopkWeightsPrefetch>(
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
        ControlSharedExpertGmm1SwigluPipeline<QuantOutType, ActivationType, Weight1Type, SwigluQuantOutType,
                                              QuantScaleOutType, ENABLE_A8W4, GMM1_TILE_M>(
            sharedGmm1Context_, sharedGmm1Args_, params_, sharedEpilogueOp_, sharedGmm1State);
    }

    SynchronizeRanksInWorld(rankSyncContext_, rankSyncArgs_);

    // 2. 路由专家 Dispatch -> GMM1/SwiGLU -> GMM2/Combine。
    ControlRoutedExpertPipeline<CombineQuantMode, QuantOutType, ActivationType, Weight1Type,
                                SwigluQuantOutType, QuantScaleOutType, ENABLE_A8W4, ENABLE_A4W4,
                                GMM1_TILE_M, EPILOGUE_TILE_M, TopkWeightsPrefetch>(
        gmm1PipelineContext_, gmm1PipelineArgs_, gmm2Context_, gmm2Args_, quantCombineContext_,
        quantCombineBufferConfig_, params_, epilogueOp_, tokenDispatchScratch_, gmm1SwigluScratch_,
        gmm2Scratch_, params_.expertTokenNumsOutGmAddr, gmTileSequence, gmm2PingPongIdx_);

    if constexpr (g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
    }

    if (sharedExpertNum_ > 0) {
        uint32_t sharedGmm2StartBlockIdx = 0U;
        int32_t sharedGmm2VecSetSyncCom = 0;
        Gmm2RuntimeState sharedGmm2RuntimeState{
            sharedGmm2StartBlockIdx, sharedGmm2VecSetSyncCom, gmTileSequence, gmm2PingPongIdx_};
        ControlSharedGmm2Pipeline<QuantOutType, SwigluQuantOutType, Weight1Type, QuantScaleOutType,
                                  ENABLE_A8W4, ENABLE_A4W4, GMM1_TILE_M, false>(
            sharedGmm2Context_, sharedGmm2Args_, params_, sharedGmm2RuntimeState);
    }

    // 3. 本卡数据Unpermute
    if constexpr (g_coreType == AIV) {
        SynchronizeRanksInWorld(rankSyncContext_, rankSyncArgs_);
        UnpermuteTokens<CombineQuantMode, TopkWeightsType, TopkWeightsPrefetch, GMM1_TILE_M>(
            tokenUnpermuteContext_, tokenUnpermuteArgs_, tokenUnpermuteScratch_);
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

} // namespace MegaMoeImpl
#undef TemplateMegaMoeTypeClass
#undef TemplateMegaMoeTypeFunc
#endif
