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
 * \file mega_moe_wave.h
 * \brief MegaMoe A8W8 wave 流水实现
 */

#ifndef MEGA_MOE_WAVE_H
#define MEGA_MOE_WAVE_H

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
#include "mega_moe_utils.h"
#include "mega_moe_job_context.h"
#include "aiv_compute/mega_moe_quant_process.h"
#include "aiv_compute/mega_moe_shared_expert_prepare.h"
#include "aiv_comm/mega_moe_send_mask.h"
#include "mega_moe_workspace_reset.h"
#include "mega_moe_shared_expert_gmm1_swiglu.h"
#include "mega_moe_wave_routed_expert_pipeline.h"
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
#define TemplateMegaMoeWaveTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode, \
        int32_t CombineQuantMode, bool TopkWeightsPrefetch, bool IsGmm1Interleaved
#define TemplateMegaMoeWaveTypeFunc \
    XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode, TopkWeightsPrefetch, \
        IsGmm1Interleaved

template <TemplateMegaMoeWaveTypeClass>
class MegaMoeWave {
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
    using QuantScaleOutType = typename std::conditional<(QuantMode >= E5M2_QUANT), fp8_e8m0_t, float>::type;
    using ActivationType = QuantOutType;
    __aicore__ inline MegaMoeWave(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1,
                                GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
                                GM_ADDR scales, GM_ADDR sharedWeight1, GM_ADDR sharedWeight2,
                                GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2, GM_ADDR yOut,
                                GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData *tilingData);
    __aicore__ inline void Process();

private:
    using SendMaskBufferConfig = MegaMoeSendMaskBufferConfig;
    using DispatchBufferConfig = MegaMoeDispatchBufferConfig;
    using CombineBufferConfig = WaveCombineBufferConfig;

    __aicore__ inline void DispatchBuffInit();
    __aicore__ inline void SendAndQuantBuffInit();
    __aicore__ inline void ProcessSharedExpertGmm1();

    __gm__ Mc2MoeContext *mc2Context_{nullptr};
    Params params_{};
    ExpertWeightTensorListAddrs moeWeightTensorListAddrs_{};
    ExpertWeightTensorListAddrs sharedWeightTensorListAddrs_{};
    DispatchPrepareContext dispatchPrepareContext_;
    SendMaskArgs sendMaskArgs_;
    ResetWorkspaceArgs resetWorkspaceArgs_;
    QuantProcessArgs quantProcessArgs_;
    SharedExpertPrepareArgs sharedExpertPrepareArgs_;
    TokenDispatchContext tokenDispatchContext_;
    TokenDispatchArgs tokenDispatchArgs_;
    Gmm1SwigluContext gmm1Context_;
    Gmm1SwigluArgs gmm1Args_;
    SharedExpertGmm1SwigluContext sharedGmm1Context_;
    SharedExpertGmm1SwigluArgs sharedGmm1Args_;
    ExpertTokenCountExportContext expertTokenCountExportContext_;
    ExpertTokenCountExportArgs expertTokenCountExportArgs_;
    Gmm2Context gmm2Context_;
    Gmm2Args gmm2Args_;
    Gmm2Context sharedGmm2Context_;
    SharedGmm2Args sharedGmm2Args_;
    WaveCombineContext waveCombineContext_;
    WaveCombineArgs waveCombineArgs_;
    WaveGmm2ReadyArgs waveGmm2ReadyArgs_;
    TokenUnpermuteContext tokenUnpermuteContext_;
    TokenUnpermuteArgs tokenUnpermuteArgs_;
    RankSyncContext rankSyncContext_;
    RankSyncArgs rankSyncArgs_;

    uint32_t m_ = 0;
    uint32_t k_ = 0;
    uint32_t aicNum_ = 0;
    uint32_t topK_ = 0;
    uint32_t rankId_ = 0;
    uint32_t worldSize_ = 0;
    uint32_t expertsPerBatch_ = 1;
    int64_t hiddenDim_ = 0;
    uint64_t maxOutputSize_ = 0;
    uint16_t gmm1PingPongIdx_ = 0;
    uint32_t startBlockIdx_ = 0;
    uint32_t blockNumPerRank_ = 2;
    int32_t dispatchFlagSlotsPerExpert_ = 0;
    int32_t swigluFlagSlotsPerExpert_ = 0;
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
    uint16_t gmm2PingPongIdx_ = 0;
    uint64_t sendTotalNum_ = 0;
    uint32_t maskAlignSize_ = 0;
    uint32_t maskSlotSize_ = 0;   // 单个 win 槽位 = maskAlignSize_(mask) + 32B(count)
    uint64_t maskWinOffset_ = 0;  // maskRecvPtr 相对 win 基址(rankSyncInWorldPtr)的偏移
    uint64_t quantWinOffset_ = 0; // quantTokenScalePtr 相对 win 基址的偏移
    int32_t compareCount_ = 0;
    // 共享专家相关成员
    uint32_t sharedExpertNum_ = 0;
    uint32_t moeExpertPerRank_ = 0;
    bool isPerExpertWeightTensor_ = false;

    static constexpr bool GMM1_INTERLEAVED = IsGmm1Interleaved;

    /*
     * AIV1 上 Dispatch 与 Combine 分阶段复用 UB，进入 Combine 前 Dispatch 的动态 ring 已经排空：
     *   [0, 64 KiB)       Dispatch 的 cumsum 等跨 wave 常驻状态；路由流水结束后复用其中最多 36 KiB，
     *                     从 GM 恢复并压紧最多 1024 个专家的 token count；
     *   [64, 160 KiB)     非量化 Combine 的 6 个 BF16 row buffer（H 最大 8 KiB）；
     *                     量化 Combine 使用 2 个 [BF16 row | FP8 data + scale] 槽及共享量化 scratch；
     *   [160, 184 KiB)    空闲；
     *   [184, 187.5 KiB)  GMM2-ready 的 GM 搬入、ReduceSum scratch 和最终 sum；
     *   [187.5, 200 KiB)  空闲；
     *   [200, 248 KiB)    Combine 共用的 meta-info，共 1536 token * 8 int32；
     *   [248, 256 KiB)    硬件保留，不使用。
     */
    LocalTensor<int32_t> resetTensor_;
    QuantProcessScratch<ActivationType> quantProcessScratch_;
    SharedExpertPrepareScratch<ActivationType> sharedExpertPrepareScratch_;
    SendMaskScratch sendMaskScratch_;
    TokenDispatchScratch<ActivationType> tokenDispatchScratch_;
    Gmm1SwigluScratch gmm1Scratch_;
    Gmm2Scratch gmm2Scratch_;
    ExpertTokenCountExportScratch expertTokenCountExportScratch_;
    WaveCombineScratch waveCombineScratch_;
    TokenUnpermuteScratch tokenUnpermuteScratch_;

    static constexpr uint32_t GMM1_TILE_M = MegaMoeImpl::L1_TILE_M_256;
    static constexpr uint32_t EPILOGUE_TILE_M =
        TopkWeightsPrefetch ? MegaMoeImpl::L1_TILE_M_128 : MegaMoeImpl::L1_TILE_M_256;

    using BlockEpilogue =
        BlockEpilogueSwigluMxQuant<ActivationType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, true,
                                   EPILOGUE_TILE_M, MegaMoeImpl::L1_TILE_N, TopkWeightsPrefetch, GMM1_INTERLEAVED,
                                   true>;
    using SharedBlockEpilogue =
        BlockEpilogueSwigluMxQuant<ActivationType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, true,
                                   MegaMoeImpl::L1_TILE_M_256, MegaMoeImpl::L1_TILE_N, false, GMM1_INTERLEAVED, true>;
    BlockEpilogue epilogueOp_;
    SharedBlockEpilogue sharedEpilogueOp_;
};

// ========================
// Init：初始化成员并计算地址偏移
// ========================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::Init(
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
    expertsPerBatch_ = tilingData->expertsPerBatch == 0U ? 1U : tilingData->expertsPerBatch;
    if (expertsPerBatch_ > moeExpertPerRank_) {
        expertsPerBatch_ = moeExpertPerRank_;
    }
    blockNumPerRank_ = tilingData->blockNumPerEP;
    maxOutputSize_ = tilingData->maxOutputSize;
    // 与 WorkspaceInfo 构造里 flagDispatchToGmm1Ptr 的分配公式保持一致。
    maxWavesPerExpert_ = static_cast<int32_t>(
        Ops::Base::CeilDiv(static_cast<int64_t>(maxOutputSize_), static_cast<int64_t>(GMM1_TILE_M)));
    dispatchFlagSlotsPerExpert_ = maxWavesPerExpert_ * INT_CACHELINE;
    swigluFlagSlotsPerExpert_ = dispatchFlagSlotsPerExpert_;
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
    params_.peermemInfo = PeermemInfo(winRankAddr_[rankId_], tilingData, 1U);
    params_.tilingData = tilingData;
    tokenDispatchScratch_.expertRevNumsGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.expertRevTokenNumsPtr));
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
    // 每个 win 槽位再追加 32B 存 count，由 GatherAndSendExpertMasks 计算并发送。
    maskSlotSize_ = maskAlignSize_ + static_cast<uint32_t>(ALIGN_32);
    mxQuantScaleNumAlignPerToken_ = Ops::Base::CeilDiv(k_, static_cast<uint32_t>(ALIGN_32));
    mxQuantTokenAlignBytes_ =
        Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_256)) * sizeof(ActivationType);
    mxQuantScaleAlignBytes_ =
        Ops::Base::CeilAlign(mxQuantScaleNumAlignPerToken_ * static_cast<uint32_t>(sizeof(QuantScaleOutType)),
                             static_cast<uint32_t>(ALIGN_32));
    mxQuantTokenScaleAlignBytes_ = mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_;
    if constexpr (TopkWeightsPrefetch) {
        uint32_t weightAlignBytes =
            Ops::Base::CeilAlign(static_cast<uint32_t>(topK_ * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
        mxQuantTokenScaleAlignBytes_ += weightAlignBytes;
    }

    dispatchPrepareContext_ = {
        {aivCoreIdx_, blockAivNum_}, {rankId_, worldSize_, moeExpertPerRank_}, {static_cast<int32_t>(m_), topK_, k_}};
    const SendMaskBufferConfig &sendMaskBufferConfig =
        aivCoreIdx_ < tilingData->sendMaskCoreCountWithExtraExpert ?
            tilingData->sendMaskConfigForCoreWithExtraExpert :
            tilingData->sendMaskConfigForCoreWithoutExtraExpert;
    sendMaskArgs_ = {params_.expertIdxGmAddr, ::winRankAddr_, maskAlignSize_, maskSlotSize_, maskWinOffset_,
                     sendMaskBufferConfig};
    int32_t resetFlagNum =
        static_cast<int32_t>(CalcMegaMoeFlagWorkspaceSize(params_.tilingData) / sizeof(int32_t));
    resetWorkspaceArgs_ = {params_.workspaceInfo.flagSwiGluToGmm2Ptr,
                           params_.workspaceInfo.gmm2CombineSyncCounterPtr,
                           params_.workspaceInfo.gmm1TileStatusPtr,
                           resetFlagNum,
                           0,
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
                                1U};
    tokenDispatchContext_ = {{blockIdx_, blockNum_},
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
                             1U};
    tokenDispatchArgs_ = {::winRankAddr_,
                          params_.peermemInfo.maskRecvPtr,
                          params_.workspaceInfo.expertRevTokenNumsPtr,
                          params_.workspaceInfo.metaInfoPtr,
                          params_.workspaceInfo.cumsumInfoPtr,
                          params_.workspaceInfo.dispatchRevDataPtr,
                          params_.workspaceInfo.dispatchRevScalePtr,
                          params_.workspaceInfo.flagDispatchToGmm1Ptr,
                          params_.workspaceInfo.flagSendCntCalToUpdParamsPtr};
    gmm1Context_ = {{blockIdx_, blockNum_},
                    {blockIdx_, aicNum_},
                    moeExpertPerRank_,
                    moeExpertPerRank_,
                    k_,
                    static_cast<uint32_t>(hiddenDim_),
                    dispatchFlagSlotsPerExpert_,
                    swigluFlagSlotsPerExpert_,
                    tilingData->maxTilesPerExpert,
                    tilingData->groupedMatmulMode,
                    1U,
                    isPerExpertWeightTensor_};
    gmm1Args_ = {params_.workspaceInfo.expertRevTokenNumsPtr,
                 params_.workspaceInfo.dispatchRevDataPtr,
                 params_.workspaceInfo.dispatchRevScalePtr,
                 params_.workspaceInfo.gmm1MmadResPtr,
                 moeWeightTensorListAddrs_.weight1,
                 moeWeightTensorListAddrs_.weightScales1,
                 params_.workspaceInfo.flagDispatchToGmm1Ptr,
                 params_.workspaceInfo.flagSendCntCalToUpdParamsPtr,
                 params_.workspaceInfo.gmm1TileStatusPtr,
                 nullptr};
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
                       nullptr,
                       tilingData->clampLimit,
                       tilingData->actMode,
                       tilingData->actSubMode,
                       tilingData->activationAlpha,
                       tilingData->activationBeta};
    gmm2Context_ = {{blockIdx_, blockNum_},
                    {blockIdx_, aicNum_},
                    m_,
                    moeExpertPerRank_,
                    k_,
                    static_cast<uint32_t>(hiddenDim_),
                    swigluFlagSlotsPerExpert_,
                    dispatchFlagSlotsPerExpert_,
                    0U,
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
                 nullptr,
                 nullptr};
    sharedGmm2Context_ = gmm2Context_;
    sharedGmm2Context_.expertPerRank = sharedExpertNum_;
    sharedGmm2Args_ = {params_.workspaceInfo.sharedExpertSwigluDataPtr,
                       params_.workspaceInfo.sharedExpertSwigluScalePtr,
                       params_.workspaceInfo.sharedExpertResultPtr,
                       sharedWeightTensorListAddrs_.weight2,
                       sharedWeightTensorListAddrs_.weightScales2,
                       params_.workspaceInfo.sharedExpertGmm2TileCounterPtr,
                       nullptr};
    expertTokenCountExportContext_ = {{0U, aicNum_}, moeExpertPerRank_, aivCoreIdx_ == 1U};
    expertTokenCountExportArgs_ = {
        params_.workspaceInfo.expertRevTokenNumsPtr, params_.expertTokenNumsOutGmAddr};
    waveCombineContext_ = {{blockIdx_, blockNum_}, k_, subBlockIdx_ == 1U};
    waveCombineArgs_ = {params_.workspaceInfo.metaInfoPtr};
    waveGmm2ReadyArgs_ = {params_.workspaceInfo.gmm2ReadyPtr};
    uint32_t gmm2NTilesPerGroup = 0U;
    if constexpr (g_coreType == AIV) {
        if (sharedExpertNum_ > 0U) {
            gmm2NTilesPerGroup = Ops::Base::CeilDiv(k_, L1_TILE_N);
        }
    }
    uint32_t quantTokenSizeBytes = 0U;
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV) {
        quantTokenSizeBytes = CreateQuantCombineBufferConfig(k_).quantTokenSizeBytes;
    }
    tokenUnpermuteContext_ = {{aivCoreIdx_, blockAivNum_},
                              m_,
                              k_,
                              topK_,
                              sharedExpertNum_,
                              gmm2NTilesPerGroup,
                              quantTokenSizeBytes,
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
    gmm1Scratch_.expertRevNumsGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.expertRevTokenNumsPtr));
    gmm2Scratch_.expertRevNumsGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.expertRevTokenNumsPtr));
}

// =================================================================================================
// DispatchBuffInit：申请公共 SendCount 和 Token Dispatch 函数使用的 buffer。
// =================================================================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::DispatchBuffInit()
{
    DispatchBufferConfig bufferConfig{};
    if constexpr (g_coreType == AIC) {
        return;
    }

    tokenDispatchScratch_.revTokenElemCnt = k_; // A8W8 输出 token 的元素数
    tokenDispatchScratch_.revScaleElemCnt =
        Ops::Base::CeilDiv(static_cast<int64_t>(k_), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE; // 输出 token-scale 元素数，紧密排列

    // 与 route batch 无关的固定占用
    uint32_t expertTokenCntTensorSize = ALIGN_32;
    uint32_t cumsumInfoTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(worldSize_ * moeExpertPerRank_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    // sendCntTensor：每 src rank 一个 burst(32B)，共 worldSize*32B。
    uint32_t sendCntTensorSize = worldSize_ * static_cast<uint32_t>(ALIGN_32);
    // Dispatch 的 UB 布局与 AIV 分核无关；对应 host CalcDispatchBufferConfig 的唯一配置。
    bufferConfig = params_.tilingData->dispatchBufferConfig;
    int32_t routeItemsPerBatch = bufferConfig.routeItemsPerBatch;

    // 按既定顺序落地址
    // Tensor 用途：ComputeExpertTokenCountAndNotify 记录本卡各专家收到的 token 总数；
    // Tensor 大小：仅记录 count 值且各专家之间复用，申请 32 字节；
    uint32_t expertTokenCntTensorAddr = 0;
    tokenDispatchScratch_.expertTokenCntTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, expertTokenCntTensorAddr, expertTokenCntTensorSize / sizeof(int32_t));
    // Tensor 用途：ComputeExpertTokenCountAndNotify 记录本卡专家收到 token count 的前缀和；
    // Tensor 大小：worldSize_ * moeExpertPerRank_ * sizeof(int32_t)，向上对齐至 32 字节；
    uint32_t cumsumInfoTensorAddr = expertTokenCntTensorAddr + expertTokenCntTensorSize;
    tokenDispatchScratch_.cumsumInfoTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, cumsumInfoTensorAddr, cumsumInfoTensorSize / sizeof(int32_t));
    // Tensor 用途：ComputeExpertTokenCountAndNotify 暂存各源 rank 的 count；
    // Tensor 大小：每个源 rank 占一个 32B burst，共 worldSize_ * 32B；
    uint32_t sendCntTensorAddr = cumsumInfoTensorAddr + cumsumInfoTensorSize;
    tokenDispatchScratch_.sendCntTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, sendCntTensorAddr, sendCntTensorSize / sizeof(int32_t));
    // Tensor 用途：DispatchExpertTokens 接收当前 batch 的 mask 切片；
    // Tensor 大小：routeItemsPerBatch / 8 字节，每个 bit 对应一个路由项；
    uint32_t maskBatchAddr = sendCntTensorAddr + sendCntTensorSize;
    uint32_t maskBatchSize =
        static_cast<uint32_t>(routeItemsPerBatch / 8) * static_cast<uint32_t>(sizeof(uint8_t));
    tokenDispatchScratch_.maskBatchTensor =
        LocalTensor<uint8_t>(TPosition::VECCALC, maskBatchAddr, maskBatchSize / sizeof(uint8_t));
    tokenDispatchScratch_.maskBatchU32Tensor =
        LocalTensor<uint32_t>(TPosition::VECCALC, maskBatchAddr, maskBatchSize / sizeof(uint32_t));
    // Tensor 用途：DispatchExpertTokens 中 GatherMask 的目标 Tensor；
    // Tensor 大小：routeItemsPerBatch * sizeof(int32_t)，向上对齐至 32 字节；
    uint32_t validTopkIndexTensorAddr = maskBatchAddr + maskBatchSize;
    uint32_t validTopkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(routeItemsPerBatch * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    tokenDispatchScratch_.validTopkIndexTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, validTopkIndexTensorAddr, validTopkIndexTensorSize / sizeof(int32_t));
    // Tensor 用途：DispatchExpertTokens 中 GatherMask 的源 Tensor，保存本 batch 的全局索引；
    // Tensor 大小：与 validTopkIndexTensor 一致，为 routeItemsPerBatch * sizeof(int32_t)，
    // 向上对齐至 32 字节；
    uint32_t topkIndexTensorAddr = validTopkIndexTensorAddr + validTopkIndexTensorSize;
    uint32_t topkIndexTensorSize = validTopkIndexTensorSize;
    tokenDispatchScratch_.topkIndexTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, topkIndexTensorAddr, topkIndexTensorSize / sizeof(int32_t));
    // 路由批次 Tensor 后依次放置 copyTmp 环形缓冲区和 32B metaInfo 环形缓冲区。
    // Tensor 用途：DispatchExpertTokens 中的动态 dispatch 环形缓冲区，配合
    // EVENT_ID0..EVENT_ID(bufferCount-1) 形成软流水；
    // 只记基址：槽视图在热路径由 GetDispatchCopyBuffer 现场构造，
    // Tensor 大小：bufferConfig.bufferCount 块（主线自适应 UB 预算给出的 2~6），
    // 每块 mxQuantTokenScaleAlignBytes_；
    // 该值即 Init() 算好的 Align256(token) + Align32(scale) + optional Align32(weight)，与 host
    // CalcDispatchBufferConfig 的 copyBufferBytes 恒相等，故连续 ring 中每个槽位均保持 32B 对齐。
    tokenDispatchScratch_.copyTmpBaseAddr = topkIndexTensorAddr + topkIndexTensorSize;
    uint32_t copyTmpTotalSize = static_cast<uint32_t>(bufferConfig.bufferCount) * mxQuantTokenScaleAlignBytes_;
    // Tensor 用途：CopyTokensAndMetaForDispatch 中的 metaInfo 环形缓冲区，逐 token 即时写入 GM；
    // Tensor 大小：bufferCount * 32B，与 copyTmp 槽位和事件编号一一对应。
    uint32_t metaInfoTensorAddr = tokenDispatchScratch_.copyTmpBaseAddr + copyTmpTotalSize;
    uint32_t metaInfoReserveSize =
        static_cast<uint32_t>(bufferConfig.bufferCount) * static_cast<uint32_t>(INT32_PER_256B) * sizeof(int32_t);
    tokenDispatchScratch_.metaInfoTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, metaInfoTensorAddr, metaInfoReserveSize / sizeof(int32_t));
    tokenDispatchScratch_.cumsumRevCntInRank = 0U;
}

// ======================================================================================
// SendAndQuantBuffInit：申请公共 mask、workspace reset 和 token quant 函数使用的 buffer。
// ======================================================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::SendAndQuantBuffInit()
{
    SendMaskBufferConfig bufferConfig{};
    if constexpr (g_coreType == AIC) {
        return;
    }

    // 与 route batch 无关的固定占用
    uint64_t totalFlagInt32 =
        static_cast<uint64_t>(CalcMegaMoeFlagWorkspaceSize(params_.tilingData) / sizeof(int32_t));
    uint32_t resetElementCountPerCore = Ops::Base::CeilDiv(totalFlagInt32, static_cast<uint64_t>(blockAivNum_));
    int32_t resetBatchElementCount = resetElementCountPerCore < static_cast<uint32_t>(DISPATCH_RESET_BATCH) ?
                                         static_cast<int32_t>(resetElementCountPerCore) :
                                         DISPATCH_RESET_BATCH;
    uint32_t resetTensorSize =
        Ops::Base::CeilAlign(static_cast<uint64_t>(resetBatchElementCount), static_cast<uint64_t>(INT32_PER_256B)) *
        sizeof(int32_t);

    uint32_t mxTempTensorSize = 2 * 1024;
    // 单个 xOutTensor 槽位与 dispatch 的 token-scale-weight 通信记录使用相同布局。
    uint32_t xOutTensorSize = mxQuantTokenScaleAlignBytes_;
    uint32_t xInAlignSize = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_128)) * sizeof(bfloat16_t);
    uint32_t expertPerCoreMax = Ops::Base::CeilDiv(worldSize_ * moeExpertPerRank_, blockAivNum_);
    uint32_t sendCntAccSize =
        Ops::Base::CeilAlign(static_cast<int64_t>(expertPerCoreMax * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));

    // 必须与 host SetAdaptiveBufferConfigs 的 quotient/remainder 分核保持一致。
    bufferConfig = aivCoreIdx_ < params_.tilingData->sendMaskCoreCountWithExtraExpert ?
                       params_.tilingData->sendMaskConfigForCoreWithExtraExpert :
                       params_.tilingData->sendMaskConfigForCoreWithoutExtraExpert;
    int32_t routeItemsPerBatch = bufferConfig.routeItemsPerBatch;

    // 按既定顺序落地址。routeItemsPerBatch 按 256 个 item 对齐，因此两个 int32 tensor 均天然满足 256B 对齐。
    uint32_t topkIdsTensorAddr = 0;
    uint32_t topkIdsTensorSize =
        static_cast<uint32_t>(routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));
    sendMaskScratch_.topkIdsTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, topkIdsTensorAddr, topkIdsTensorSize / sizeof(int32_t));

    uint32_t resetAddrActual = topkIdsTensorAddr + topkIdsTensorSize;
    resetTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, resetAddrActual, resetTensorSize / sizeof(int32_t));
    Duplicate<int32_t>(resetTensor_, 0, (resetTensorSize / sizeof(int32_t)));

    uint32_t mxTempTensorAddr = resetAddrActual + resetTensorSize;
    quantProcessScratch_.mxTempTensor =
        LocalTensor<uint16_t>(TPosition::VECCALC, mxTempTensorAddr, mxTempTensorSize / sizeof(uint16_t));

    uint32_t xOutTensorAddr1 = mxTempTensorAddr + mxTempTensorSize;
    quantProcessScratch_.xOutTensor0 =
        LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr1, xOutTensorSize / sizeof(ActivationType));
    uint32_t xOutTensorAddr2 = xOutTensorAddr1 + xOutTensorSize;
    quantProcessScratch_.xOutTensor1 =
        LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr2, xOutTensorSize / sizeof(ActivationType));

    uint32_t xInAlignAddr1 = xOutTensorAddr2 + xOutTensorSize;
    quantProcessScratch_.xInTensor0 =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr1, xInAlignSize / sizeof(bfloat16_t));
    uint32_t xInAlignAddr2 = xInAlignAddr1 + xInAlignSize;
    quantProcessScratch_.xInTensor1 =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr2, xInAlignSize / sizeof(bfloat16_t));

    uint32_t sendMaskAddr = xInAlignAddr2 + xInAlignSize;
    uint32_t sendGatherOutSize =
        static_cast<uint32_t>(routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));

    uint32_t sendMaskTotalBytes = static_cast<uint32_t>(bufferConfig.bufferCount) * bufferConfig.bufferBytes;
    sendMaskScratch_.sendMaskTensor = LocalTensor<uint8_t>(TPosition::VECCALC, sendMaskAddr, sendMaskTotalBytes);
    uint32_t sendGatherOutAddr = sendMaskAddr + sendMaskTotalBytes;
    sendMaskScratch_.sendGatherOutTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, sendGatherOutAddr, sendGatherOutSize / sizeof(int32_t));
    uint32_t sendCntAccAddr = sendGatherOutAddr + sendGatherOutSize;
    sendMaskScratch_.sendCntAccTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, sendCntAccAddr, sendCntAccSize / sizeof(int32_t));
    sharedExpertPrepareScratch_.copyBuffer0 = quantProcessScratch_.xOutTensor0;
    sharedExpertPrepareScratch_.copyBuffer1 = quantProcessScratch_.xOutTensor1;
    resetWorkspaceArgs_.resetBatchElementCount = resetBatchElementCount;
}

// ===============================================================
// 可选共享专家流程：依次执行共享专家 GMM1 与 SwiGLU，并在结束后重置 GMM 调度状态。
// ===============================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::ProcessSharedExpertGmm1()
{
    int32_t vecSetSyncCom = 0;
    int32_t gmTileSequence = 0;
    SharedExpertGmm1SwigluState runtimeState{
        startBlockIdx_, vecSetSyncCom, gmTileSequence, gmm1PingPongIdx_};
    ControlSharedExpertGmm1SwigluPipeline<QuantOutType, ActivationType, Weight1Type, ActivationType,
                                          QuantScaleOutType, false, GMM1_TILE_M, GMM1_INTERLEAVED, true>(
        sharedGmm1Context_, sharedGmm1Args_, params_, sharedEpilogueOp_, runtimeState);
    vecSetSyncCom = 0;
    gmm1PingPongIdx_ = 0;
    startBlockIdx_ = 0; // 共享专家 GMM1 修改了 startBlockIdx_，重置后供路由专家 GMM1 使用
}

// ===============================================================
// Wave 模板主流程。
// ===============================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::Process()
{
    // 保存入口时的溢出模式，并初始化输入准备阶段使用的 UB。
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    SendAndQuantBuffInit();

    // 阶段 1：AIV 完成本卡输入量化、路由 mask 推送和 flag 清零。
    QuantizeLocalTokens<QuantMode, QuantOutType, ActivationType, TopkWeightsType, TopkWeightsPrefetch>(
        dispatchPrepareContext_, quantProcessArgs_, quantProcessScratch_);
    GatherAndSendExpertMasks(dispatchPrepareContext_, sendMaskArgs_, sendMaskScratch_);
    ResetDispatchWorkspace<false, TopkWeightsPrefetch>(
        dispatchPrepareContext_, resetWorkspaceArgs_, resetTensor_);
    if (sharedExpertNum_ > 0) {
        // 可选：为共享专家拆分连续布局的输入数据与 scale。
        PrepareSharedExpertInput<ActivationType, QuantScaleOutType>(
            dispatchPrepareContext_, sharedExpertPrepareArgs_, sharedExpertPrepareScratch_);
    } else {
        if constexpr (g_coreType == AIV) {
            PipeBarrier<PIPE_ALL>();
        }
    }
    SyncAll<false>(); // AIC 等待 AIV 完成输入准备与 flag 清零后再进入计算

    // 可选：提前执行共享专家 GMM1 + SwiGLU，复用与路由专家相同的计算实现。
    if (sharedExpertNum_ > 0) {
        ProcessSharedExpertGmm1();
    }

    // 等待所有 rank 完成本轮输入准备，再读取远端 dispatch 数据。
    SynchronizeRanksInWorld(rankSyncContext_, rankSyncArgs_);

    // 阶段 2：按 wave 执行路由专家 Dispatch、GMM1、SwiGLU、GMM2 与 Combine。
    DispatchBuffInit();
    CombineBufferConfig combineBufferConfig =
        InitWaveCombineBuffers<CombineQuantMode>(waveCombineContext_, waveCombineScratch_);
    InitExpertTokenCountExportBuffers(expertTokenCountExportContext_, expertTokenCountExportScratch_);

    ControlWaveRoutedExpertPipeline<CombineQuantMode, QuantOutType, ActivationType, Weight1Type,
                                    QuantScaleOutType, GMM1_TILE_M, EPILOGUE_TILE_M,
                                    TopkWeightsPrefetch, GMM1_INTERLEAVED>(
        tokenDispatchContext_, tokenDispatchArgs_, gmm1Context_, gmm1Args_, gmm2Context_, gmm2Args_,
        waveCombineContext_, waveCombineArgs_, waveGmm2ReadyArgs_, expertTokenCountExportContext_,
        expertTokenCountExportArgs_, params_, epilogueOp_, tokenDispatchScratch_, gmm1Scratch_,
        gmm2Scratch_, waveCombineScratch_, expertTokenCountExportScratch_, combineBufferConfig,
        expertsPerBatch_, startBlockIdx_, gmm1PingPongIdx_, gmm2PingPongIdx_);

    if constexpr (g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
    }

    // 可选：路由专家流水结束后执行共享专家 GMM2。
    if (sharedExpertNum_ > 0) {
        int32_t sharedGmm2VecSetSyncCom = 0;
        int32_t sharedGmm2TileSequence = 0;
        Gmm2RuntimeState sharedGmm2RuntimeState{
            startBlockIdx_, sharedGmm2VecSetSyncCom, sharedGmm2TileSequence, gmm2PingPongIdx_};
        ControlSharedGmm2Pipeline<QuantOutType, ActivationType, Weight1Type, QuantScaleOutType,
                                  false, false, GMM1_TILE_M, TopkWeightsPrefetch,
                                  GMM1_INTERLEAVED, true>(
            sharedGmm2Context_, sharedGmm2Args_, params_, sharedGmm2RuntimeState);
    }

    // 阶段 3：等待所有 rank 的 Combine 发送完成，再执行本卡 Unpermute。
    if constexpr (g_coreType == AIV) {
        SynchronizeRanksInWorld(rankSyncContext_, rankSyncArgs_);
        UnpermuteTokens<CombineQuantMode, TopkWeightsType, TopkWeightsPrefetch, GMM1_TILE_M>(
            tokenUnpermuteContext_, tokenUnpermuteArgs_, tokenUnpermuteScratch_);
    }
    // 恢复入口时保存的溢出模式。
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

#undef TemplateMegaMoeWaveTypeClass
#undef TemplateMegaMoeWaveTypeFunc

} // namespace MegaMoeImpl
#endif
