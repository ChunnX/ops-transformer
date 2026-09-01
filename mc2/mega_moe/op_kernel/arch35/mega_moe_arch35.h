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
 * \file mega_moe_arch35.h
 * \brief
 */

#ifndef MEGA_MOE_ARCH35_H
#define MEGA_MOE_ARCH35_H

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
#include "common/mega_moe_exception_dump_policy.h"
#include "blaze/epilogue/block_epilogue_activation_mx_quant.h"
#include "stage/mega_moe_token_quant.h"
#include "stage/mega_moe_send_mask.h"
#include "stage/mega_moe_workspace_reset.h"
#include "stage/mega_moe_token_dispatch.h"
#include "stage/mega_moe_gmm1_activation.h"
#include "stage/mega_moe_gmm2_combine.h"
#include "stage/mega_moe_unpermute.h"
#if __has_include("../../common/quantize_functions.h")
#include "../../common/quantize_functions.h"
#else
#include "../../../common/op_kernel/quantize_functions.h"
#endif

namespace MegaMoeImpl {

using namespace AscendC;

#define TemplateMegaMoeTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename MoeWeightType, int32_t MoeQuantMode, \
        typename SharedWeightType, int32_t SharedQuantMode, int32_t MoeWeight1Format, int32_t MoeWeight2Format, \
        int32_t SharedWeight1Format, int32_t SharedWeight2Format, int32_t CombineQuantMode, bool TopkWeightsPrefetch
#define TemplateMegaMoeTypeFunc \
    XType, OutputType, TopkWeightsType, MoeWeightType, MoeQuantMode, SharedWeightType, SharedQuantMode, \
        MoeWeight1Format, MoeWeight2Format, SharedWeight1Format, SharedWeight2Format, CombineQuantMode, \
        TopkWeightsPrefetch

template <TemplateMegaMoeTypeClass>
class MegaMoe {
public:
    using MoeQuantConfig = QuantConfig<MoeWeightType, MoeQuantMode>;
    using SharedQuantConfig = QuantConfig<SharedWeightType, SharedQuantMode>;
    __aicore__ inline MegaMoe(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1,
                                GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
                                GM_ADDR scales, GM_ADDR sharedWeight1, GM_ADDR sharedWeight2,
                                GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2, GM_ADDR yOut,
                                GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData *tilingData,
                                GM_ADDR tilingGM);

private:
    using SendMaskBufferConfig = MegaMoeSendMaskBufferConfig;
    using UnpermuteBufferConfig = MegaMoeUnpermuteBufferConfig;

    __aicore__ inline void InitInputPrepareConfigs();
    __aicore__ inline void InitSyncWorkspaceConfigs(int32_t dispatchFlagSlotsPerExpert,
                                                    int32_t activationFlagSlotsPerExpert);
    __aicore__ inline void InitGmmConfigs();
    __aicore__ inline void InitTokenUnpermuteConfig();

protected:
    __aicore__ inline void SendAndQuantBuffInit();
    __aicore__ inline void DispatchBuffInit();
    __aicore__ inline void InitCombineBuffers();
    __aicore__ inline UnpermuteBufferConfig InitTokenUnpermuteBuffers();
    __aicore__ inline void ProcessInputPreparationStage();
    __aicore__ inline uint32_t DispatchMoeExpert(uint32_t expertIdx);
    template <bool WaitForTokenCountReady>
    __aicore__ inline void PrepareGmmExpertState(ExpertLoopState &state, uint32_t expertIdx);
    template <typename... LayoutArgs>
    __aicore__ inline void RunSharedExpertGmm1Activation(const GMMAddrInfo &gmmAddrInfo,
                                                         const ProblemShape &problemShape,
                                                         const GmmExecutionConfig &gmmConfig,
                                                         Gmm1RuntimeState &runtimeState, uint32_t sharedExpertIdx,
                                                         const LayoutArgs &...layoutArgs);
    __aicore__ inline void RunSharedExpertGmm2(const GMMAddrInfo &gmmAddrInfo, const ProblemShape &problemShape,
                                               uint32_t &startBlockIdx);
    __aicore__ inline void ProcessSharedExpertGmm1();
    __aicore__ inline void ProcessSharedExpertGmm2();
    template <typename Derived>
    __aicore__ inline void ProcessWave(Derived &derived);

    __gm__ Mc2MoeContext *mc2Context_{nullptr};
    Params params_{};
    ExpertWeightTensorListAddrs moeWeightTensorListAddrs_{};
    ExpertWeightTensorListAddrs sharedWeightTensorListAddrs_{};
    MoeStageCommonConfig commonConfig_{};
    GmmExecutionConfig gmmExecutionConfig_{};
    BlockWorkspaceContext countWorkspace_{};
    MoeSyncWorkspaceLayout syncWorkspaceLayout_{};
    // 单次 reset 批量元素数（与 syncWorkspaceLayout_ 描述的清零区域配套）。
    int32_t resetBatchElementCount_ = 0;
    // 输入准备各 stage（quant/mask/reset/shared-prepare/unpermute）共用的逐 AIV 任务分工。
    AivJobContext aivJob_{};
    TokenDispatchConfig tokenDispatchConfig_;
    SendMaskConfig sendMaskConfig_;
    QuantProcessConfig quantProcessConfig_;
    QuantProcessConfig sharedQuantProcessConfig_;
    QuantCombineConfig quantCombineConfig_;
    QuantCombineBufferConfig quantCombineBufferConfig_;
    TokenUnpermuteConfig tokenUnpermuteConfig_;
    // A8W4 GMM1 和 A4W4 Wave GMM2 会覆盖 V1 UB，导致 UB 上跨 expert 的状态无法保持。
    // tokenDispatchScratch_ 中的 cumsumInfoGlobalTensor 作为 GM 持久备份：
    // ComputeExpertTokenCountAndNotify 中 Load → 计算 → Store；Dispatch/Export 从 GM 恢复。

    uint32_t k_ = 0;
    uint32_t rankId_ = 0;
    uint32_t worldSize_ = 0;
    uint32_t blockNum_ = GetBlockNum();
    uint32_t blockAivNum_ = GetBlockNum() * 2;
    uint32_t blockIdx_ = GetBlockIdx() / GetTaskRation();
    uint32_t aivCoreIdx_ = GetBlockIdx();
    int32_t gmTileSequence_ = 0;
    // 主线 shared-expert 特性成员
    uint32_t sharedExpertNum_ = 0;
    uint32_t moeExpertPerRank_ = 0;
    uint32_t mGroupsPerWave_ = 1U;

    static constexpr bool SHARED_INPUT_REUSES_MOE_QUANT =
        Std::IsSame<typename MoeQuantConfig::QuantOutType, typename SharedQuantConfig::QuantOutType>::value;
    static constexpr uint32_t GMM1_TILE_M = L1_TILE_M_256;
    static constexpr uint32_t EPILOGUE_TILE_M = TopkWeightsPrefetch ? L1_TILE_M_128 : L1_TILE_M_256;
    QuantProcessScratch<typename MoeQuantConfig::QuantStorageType> quantScratch_;
    QuantProcessScratch<typename SharedQuantConfig::QuantStorageType> sharedQuantScratch_;
    SendMaskScratch sendMaskScratch_;
    LocalTensor<int32_t> resetTensor_;

    using BlockEpilogue =
        BlockEpilogueActivationMxQuant<typename MoeQuantConfig::ActivationQuantOutType, bfloat16_t,
                                       typename MoeQuantConfig::QuantScaleType, typename MoeQuantConfig::QuantScaleType,
                                       true, EPILOGUE_TILE_M, L1_TILE_N, TopkWeightsPrefetch>;
    using SharedBlockEpilogue = BlockEpilogueActivationMxQuant<
        typename SharedQuantConfig::ActivationQuantOutType, bfloat16_t, typename SharedQuantConfig::QuantScaleType,
        typename SharedQuantConfig::QuantScaleType, true, L1_TILE_M_256, L1_TILE_N, false>;
    BlockEpilogue epilogueOp_;
    SharedBlockEpilogue sharedEpilogueOp_;
    TokenDispatchScratch<typename MoeQuantConfig::QuantStorageType> tokenDispatchScratch_;
    TokenUnpermuteScratch tokenUnpermuteScratch_;
    MegaMoeImpl::ExceptionDumpEngine exceptionDump_;
    __gm__ MegaMoeImpl::GmmLoopCount *gmmLoopCount_{nullptr};
};

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitInputPrepareConfigs()
{
    aivJob_ = {.jobIndex = aivCoreIdx_, .totalJobs = blockAivNum_};
    quantProcessConfig_ =
        CreateQuantProcessConfig<typename MoeQuantConfig::QuantStorageType, typename MoeQuantConfig::QuantScaleType,
                                 TopkWeightsPrefetch, MoeQuantConfig::A_ELEMS_PER_BYTE>(k_, params_);
    sharedQuantProcessConfig_ = CreateQuantProcessConfig<typename SharedQuantConfig::QuantStorageType,
                                                         typename SharedQuantConfig::QuantScaleType, false,
                                                         SharedQuantConfig::A_ELEMS_PER_BYTE>(k_, params_);
    sendMaskConfig_ = CreateSendMaskConfig(params_, aivCoreIdx_);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitSyncWorkspaceConfigs(int32_t dispatchFlagSlotsPerExpert,
                                                                                  int32_t activationFlagSlotsPerExpert)
{
    countWorkspace_ = {.blockIdx = blockIdx_, .blockNum = params_.tilingData->aicNum};
    syncWorkspaceLayout_ = {.dispatchFlagSlotCountPerExpert = dispatchFlagSlotsPerExpert,
                            .activationFlagSlotCountPerExpert = activationFlagSlotsPerExpert,
                            .gmm1TileStatusCountPerExpert = params_.tilingData->maxTilesPerExpert,
                            .combineSyncSlotCountPerExpert = params_.tilingData->combineSyncSlotCountPerExpert};
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitGmmConfigs()
{
    gmmExecutionConfig_ = {.blockJob = {.jobIndex = blockIdx_, .totalJobs = blockNum_},
                           .isPerExpertWeightTensor = params_.tilingData->isPerExpertWeightTensor};
    constexpr bool useAiv1OnlyCombine =
        MoeQuantConfig::AXW_MODE == AxWMode::A8W4 || MoeQuantConfig::AXW_MODE == AxWMode::A4W4;
    quantCombineConfig_ = {.job = {.jobIndex = useAiv1OnlyCombine ? blockIdx_ : aivCoreIdx_,
                                   .totalJobs = useAiv1OnlyCombine ? blockNum_ : blockAivNum_},
                           .participates = !useAiv1OnlyCombine || GetSubBlockIdx() == 1U};
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitTokenUnpermuteConfig()
{
    tokenUnpermuteConfig_ = {.job = {.jobIndex = aivCoreIdx_, .totalJobs = blockAivNum_},
                             .quantTokenSizeBytes = quantCombineBufferConfig_.quantTokenSizeBytes,
                             .fullTokenChunkJobCount = params_.tilingData->unpermuteFullTokenChunkCoreCount,
                             .fullTokenChunkConfig = params_.tilingData->unpermuteConfigForFullTokenChunk,
                             .tailTokenChunkConfig = params_.tilingData->unpermuteConfigForTailTokenChunk};
}

// ========================
// Init：初始化 & 偏移计算
// ========================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales, GM_ADDR sharedWeight1,
    GM_ADDR sharedWeight2, GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2, GM_ADDR yOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData *tilingData, GM_ADDR tilingGM)
{
    k_ = tilingData->h;
    worldSize_ = tilingData->epWorldSize;
    moeExpertPerRank_ = tilingData->moeExpertPerRank;
    sharedExpertNum_ = tilingData->sharedExpertNum;
    mGroupsPerWave_ = tilingData->mGroupsPerWave;
    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext *>(context);
    rankId_ = mc2Context_->epRankId;
    GM_ADDR dumpBase = reinterpret_cast<GM_ADDR>(mc2Context_->epHcclBuffer_[rankId_]);
    for (int i = 0; i < worldSize_; i++) {
        // g_winRankAddr_从win区地址偏移60K开始用，前面60K是异常dump区
        g_winRankAddr_[i] = reinterpret_cast<GM_ADDR>(mc2Context_->epHcclBuffer_[i]) + EXCEPTION_DUMP_REGION_SIZE;
    }
    params_.aGmAddr = x;
    params_.expertIdxGmAddr = topkIds;
    moeWeightTensorListAddrs_ = {
        .weight1 = weight1, .weightScales1 = weightScales1, .weight2 = weight2, .weightScales2 = weightScales2};
    if (sharedExpertNum_ > 0U) {
        sharedWeightTensorListAddrs_ = {.weight1 = sharedWeight1,
                                        .weightScales1 = sharedWeightScales1,
                                        .weight2 = sharedWeight2,
                                        .weightScales2 = sharedWeightScales2};
    }
    params_.y2GmAddr = yOut;
    params_.expertTokenNumsOutGmAddr = expertTokenNumsOut;
    params_.probsGmAddr = topkWeights;
    params_.workspaceInfo = WorkspaceInfo(workspaceGM, tilingData);
    params_.peermemInfo = PeermemInfo(g_winRankAddr_[rankId_], tilingData, MoeQuantConfig::A_ELEMS_PER_BYTE);
    params_.tilingData = tilingData;
    epilogueOp_.Init({.yGmAddr = params_.workspaceInfo.activationQuantDataPtr,
                      .yScaleGmAddr = params_.workspaceInfo.activationQuantScalePtr,
                      .x2ScaleGmAddr = nullptr,
                      .x1ScaleGmAddr = nullptr,
                      .biasGmAddr = nullptr,
                      .clampLimit = tilingData->clampLimit,
                      .actMode = tilingData->actMode,
                      .actSubMode = tilingData->actSubMode,
                      .activationAlpha = tilingData->activationAlpha,
                      .activationBeta = tilingData->activationBeta});
    commonConfig_ = {.rankId = rankId_,
                     .worldSize = worldSize_,
                     .moeExpertPerRank = moeExpertPerRank_,
                     .sharedExpertNum = sharedExpertNum_,
                     .tokenNum = tilingData->bs,
                     .topK = tilingData->topK,
                     .tokenHiddenDim = k_,
                     .gmm1OutputDim = tilingData->hiddenDim};
    const int64_t maxOutput = static_cast<int64_t>(tilingData->maxOutputSize);
    const int64_t tileM = static_cast<int64_t>(GMM1_TILE_M);
    int32_t dispatchFlagSlotsPerExpert = static_cast<int32_t>(Ops::Base::CeilDiv(maxOutput, tileM)) * INT_CACHELINE;
    int32_t activationFlagSlotsPerExpert =
        static_cast<int32_t>(Ops::Base::CeilDiv(maxOutput, static_cast<int64_t>(L1_TILE_M_256))) * INT_CACHELINE;
    InitInputPrepareConfigs();
    tokenDispatchConfig_ = CreateTokenDispatchConfig(params_, quantProcessConfig_, sendMaskConfig_.maskAlignSize);
    InitSyncWorkspaceConfigs(dispatchFlagSlotsPerExpert, activationFlagSlotsPerExpert);
    InitGmmConfigs();
    InitCombineBuffers();
    InitTokenUnpermuteConfig();

    gmmLoopCount_ = MegaMoeImpl::RegisterMegaMoeExceptionDump(exceptionDump_, dumpBase, tilingGM, tilingData,
                                                              params_.peermemInfo, sendMaskConfig_.maskAlignSize,
                                                              reinterpret_cast<GM_ADDR>(&mc2Context_->epRankId));
}

// 普通模板 Token Dispatch 使用的 UB/GM 视图。
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::DispatchBuffInit()
{
    const TokenDispatchConfig &context = tokenDispatchConfig_;
    TokenDispatchScratch<typename MoeQuantConfig::QuantStorageType> &scratch = tokenDispatchScratch_;
    scratch.expertRevNumsGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.expertRevTokenNumsPtr));
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U) {
        return;
    }

    const MegaMoeDispatchBufferConfig &bufferConfig = context.bufferConfig;
    scratch.revTokenElemCnt = commonConfig_.tokenHiddenDim / MoeQuantConfig::A_ELEMS_PER_BYTE;
    scratch.revScaleElemCnt = Ops::Base::CeilDiv(static_cast<int64_t>(commonConfig_.tokenHiddenDim),
                                                 static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
                              MXFP_MULTI_BASE_SIZE;
    uint32_t cumsumInfoTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(commonConfig_.worldSize * commonConfig_.moeExpertPerRank * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_32));
    if constexpr (MoeQuantConfig::AXW_MODE == AxWMode::A8W4 || MoeQuantConfig::AXW_MODE == AxWMode::A4W4) {
        scratch.cumsumInfoGlobalTensor.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.cumsumInfoPtr +
                                               static_cast<uint64_t>(cumsumInfoTensorSize) * countWorkspace_.blockIdx));
    }
    scratch.cumsumRevCntInRank = 0U;

    uint32_t expertTokenCntTensorAddr = 0U;
    uint32_t expertTokenCntTensorSize = ALIGN_32;
    scratch.expertTokenCntTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, expertTokenCntTensorAddr, expertTokenCntTensorSize / sizeof(int32_t));
    uint32_t cumsumInfoTensorAddr = expertTokenCntTensorAddr + expertTokenCntTensorSize;
    scratch.cumsumInfoTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, cumsumInfoTensorAddr, cumsumInfoTensorSize / sizeof(int32_t));
    uint32_t sendCntTensorAddr = cumsumInfoTensorAddr + cumsumInfoTensorSize;
    uint32_t sendCntTensorSize = commonConfig_.worldSize * static_cast<uint32_t>(ALIGN_32);
    scratch.sendCntTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, sendCntTensorAddr, sendCntTensorSize / sizeof(int32_t));
    uint32_t maskBatchTensorAddr = sendCntTensorAddr + sendCntTensorSize;
    uint32_t maskBatchTensorSize = static_cast<uint32_t>(bufferConfig.routeItemsPerBatch / 8);
    scratch.maskBatchTensor = LocalTensor<uint8_t>(TPosition::VECCALC, maskBatchTensorAddr, maskBatchTensorSize);
    scratch.maskBatchU32Tensor =
        LocalTensor<uint32_t>(TPosition::VECCALC, maskBatchTensorAddr, maskBatchTensorSize / sizeof(uint32_t));
    uint32_t validTopkIndexTensorAddr = maskBatchTensorAddr + maskBatchTensorSize;
    uint32_t validTopkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(bufferConfig.routeItemsPerBatch * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    scratch.validTopkIndexTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, validTopkIndexTensorAddr, validTopkIndexTensorSize / sizeof(int32_t));
    uint32_t topkIndexTensorAddr = validTopkIndexTensorAddr + validTopkIndexTensorSize;
    scratch.topkIndexTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, topkIndexTensorAddr, validTopkIndexTensorSize / sizeof(int32_t));
    scratch.copyTmpBaseAddr = topkIndexTensorAddr + validTopkIndexTensorSize;
    uint32_t copyTmpTotalSize = static_cast<uint32_t>(bufferConfig.bufferCount) * context.quantTokenScaleAlignBytes;
    uint32_t expertTokenNumsOutTensorAddr = scratch.copyTmpBaseAddr + copyTmpTotalSize;
    uint32_t expertTokenNumsOutTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(commonConfig_.moeExpertPerRank * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    scratch.expertTokenNumsOutTensor = LocalTensor<int32_t>(TPosition::VECCALC, expertTokenNumsOutTensorAddr,
                                                            expertTokenNumsOutTensorSize / sizeof(int32_t));
    uint32_t metaInfoTensorAddr = expertTokenNumsOutTensorAddr + expertTokenNumsOutTensorSize;
    uint32_t metaInfoTensorSize = static_cast<uint32_t>(bufferConfig.bufferCount) * INT32_PER_256B * sizeof(int32_t);
    scratch.metaInfoTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, metaInfoTensorAddr, metaInfoTensorSize / sizeof(int32_t));
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitCombineBuffers()
{
    quantCombineBufferConfig_ = {.combineUbElementCount = 0, .quantTokenSizeBytes = 0U};
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV) {
        quantCombineBufferConfig_ = CreateQuantCombineBufferConfig(k_);
    }
}

// ======================================================================================
// SendAndQuantBuffInit：单核 mask/reset/quant/shared-prepare 模块使用的 buffer 申请。
//   shared prepare 复用 quant 输出双 buffer；reset 封顶 DISPATCH_RESET_BATCH。
// ======================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::SendAndQuantBuffInit()
{
    if constexpr (g_coreType == AIC) {
        return;
    }

    // 与 route batch 无关的固定占用
    uint64_t totalFlagInt32 = static_cast<uint64_t>(params_.workspaceInfo.flagResetElementCount);
    if constexpr (TopkWeightsPrefetch) {
        uint64_t statusElementCount = static_cast<uint64_t>(params_.workspaceInfo.gmm1TileStatusElementCount);
        totalFlagInt32 = totalFlagInt32 > statusElementCount ? totalFlagInt32 : statusElementCount;
    }
    uint32_t resetElementCountPerCore = Ops::Base::CeilDiv(totalFlagInt32, static_cast<uint64_t>(blockAivNum_));
    int32_t resetBatchElementCount = resetElementCountPerCore < static_cast<uint32_t>(DISPATCH_RESET_BATCH) ?
                                         static_cast<int32_t>(resetElementCountPerCore) :
                                         DISPATCH_RESET_BATCH;
    uint32_t resetTensorSize =
        Ops::Base::CeilAlign(static_cast<uint64_t>(resetBatchElementCount), static_cast<uint64_t>(INT32_PER_256B)) *
        sizeof(int32_t);

    uint32_t mxTempTensorSize = 2 * 1024;
    // MoE/shared 分时复用同一组 xOut UB；存在 shared 时按两侧最大记录预留，实际访问范围由各自 config 限定。
    uint32_t xOutTensorSize = quantProcessConfig_.quantTokenScaleAlignBytes;
    if (sharedExpertNum_ > 0U && xOutTensorSize < sharedQuantProcessConfig_.quantTokenScaleAlignBytes) {
        xOutTensorSize = sharedQuantProcessConfig_.quantTokenScaleAlignBytes;
    }
    uint32_t xInAlignSize = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_128)) * sizeof(bfloat16_t);
    uint32_t expertPerCoreMax = Ops::Base::CeilDiv(worldSize_ * moeExpertPerRank_, blockAivNum_);
    uint32_t sendCntAccSize =
        Ops::Base::CeilAlign(static_cast<int64_t>(expertPerCoreMax * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));

    // 必须与 host SetAdaptiveBufferConfigs 的 quotient/remainder 分核保持一致。GatherAndSendExpertMasks 按
    // expertId = aivCoreIdx_ + ownedIdx * blockAivNum_ 遍历，因此前 remainder 个 core 多处理一个 expert。
    const SendMaskBufferConfig &bufferConfig = sendMaskConfig_.bufferConfig;
    int32_t routeItemsPerBatch = bufferConfig.routeItemsPerBatch;

    // 按既定顺序落地址。routeItemsPerBatch 按 256 个 item 对齐，因此两个 int32 tensor 均天然满足 256B 对齐。
    uint32_t topkIdsTensorAddr = 0;
    uint32_t topkIdsTensorSize = static_cast<uint32_t>(routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));
    sendMaskScratch_.topkIdsTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, topkIdsTensorAddr, topkIdsTensorSize / sizeof(int32_t));

    uint32_t resetAddrActual = topkIdsTensorAddr + topkIdsTensorSize;
    resetTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, resetAddrActual, resetTensorSize / sizeof(int32_t));
    Duplicate<int32_t>(resetTensor_, 0, (resetTensorSize / sizeof(int32_t)));
    resetBatchElementCount_ = resetBatchElementCount;

    uint32_t mxTempTensorAddr = resetAddrActual + resetTensorSize;
    quantScratch_.mxTempTensor =
        LocalTensor<uint16_t>(TPosition::VECCALC, mxTempTensorAddr, mxTempTensorSize / sizeof(uint16_t));

    uint32_t xOutTensorAddr1 = mxTempTensorAddr + mxTempTensorSize;
    quantScratch_.xOutTensor0 = LocalTensor<typename MoeQuantConfig::QuantStorageType>(
        TPosition::VECCALC, xOutTensorAddr1, xOutTensorSize / sizeof(typename MoeQuantConfig::QuantStorageType));
    uint32_t xOutTensorAddr2 = xOutTensorAddr1 + xOutTensorSize;
    quantScratch_.xOutTensor1 = LocalTensor<typename MoeQuantConfig::QuantStorageType>(
        TPosition::VECCALC, xOutTensorAddr2, xOutTensorSize / sizeof(typename MoeQuantConfig::QuantStorageType));
    uint32_t xInAlignAddr1 = xOutTensorAddr2 + xOutTensorSize;
    quantScratch_.xInTensor0 =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr1, xInAlignSize / sizeof(bfloat16_t));
    uint32_t xInAlignAddr2 = xInAlignAddr1 + xInAlignSize;
    quantScratch_.xInTensor1 =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr2, xInAlignSize / sizeof(bfloat16_t));
    if (sharedExpertNum_ > 0U) {
        if constexpr (!SHARED_INPUT_REUSES_MOE_QUANT) {
            sharedQuantScratch_.mxTempTensor = quantScratch_.mxTempTensor;
            sharedQuantScratch_.xOutTensor0 = LocalTensor<typename SharedQuantConfig::QuantStorageType>(
                TPosition::VECCALC, xOutTensorAddr1,
                xOutTensorSize / sizeof(typename SharedQuantConfig::QuantStorageType));
            sharedQuantScratch_.xOutTensor1 = LocalTensor<typename SharedQuantConfig::QuantStorageType>(
                TPosition::VECCALC, xOutTensorAddr2,
                xOutTensorSize / sizeof(typename SharedQuantConfig::QuantStorageType));
            sharedQuantScratch_.xInTensor0 = quantScratch_.xInTensor0;
            sharedQuantScratch_.xInTensor1 = quantScratch_.xInTensor1;
        }
    }

    uint32_t sendMaskAddr = xInAlignAddr2 + xInAlignSize;
    /*
     * h%64==32（scale 组数为奇数）时，量化链路存在三处"计算不覆盖、却进入定长通信记录或参与
     * 计算"的跨 launch UB 残留：xIn 尾部（进 ComputeMaxExp 尾块 mask 内 lane）、xOut 记录的
     * scale 偶数补齐槽（ComputeScale 掩码写不到）、mxTemp 的 halfScale 补偶槽（被
     * ComputeFp8Data 尾块 E2B 广播进乘法，0×NaN 仍为 NaN）。残留呈 NaN/大指数位型时整行
     * GMM 输出被污染为 NaN，最终 combine 输出成块清零（首轮 UB 干净故仅多轮调用时显形）。
     * 此处对 [mxTempTensorAddr, sendMaskAddr) 连续 span（mxTemp/xOut0/xOut1/xIn0/xIn1 五段
     * 量化 scratch）一次性清零：span 边界取 sendMaskAddr、与本函数的地址推进公式同源，
     * 中间插入新 buffer 时范围自动跟随；有效区随后每 token 均被完整覆写，残留位恒为良性 0。
     * h%64==0 时不存在上述缝隙，本清零不改变任何可观测行为。
     */
    LocalTensor<int16_t> quantScratchSpan(TPosition::VECCALC, mxTempTensorAddr,
                                          (sendMaskAddr - mxTempTensorAddr) / sizeof(int16_t));
    Duplicate<int16_t>(quantScratchSpan, 0, static_cast<int32_t>((sendMaskAddr - mxTempTensorAddr) / sizeof(int16_t)));
    PipeBarrier<PIPE_V>();
    SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID2>();
    uint32_t sendGatherOutSize = static_cast<uint32_t>(routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));

    uint32_t sendMaskTotalBytes = static_cast<uint32_t>(bufferConfig.bufferCount) * bufferConfig.bufferBytes;
    sendMaskScratch_.sendMaskTensor = LocalTensor<uint8_t>(TPosition::VECCALC, sendMaskAddr, sendMaskTotalBytes);
    uint32_t sendGatherOutAddr = sendMaskAddr + sendMaskTotalBytes;
    sendMaskScratch_.sendGatherOutTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, sendGatherOutAddr, sendGatherOutSize / sizeof(int32_t));
    uint32_t sendCntAccAddr = sendGatherOutAddr + sendGatherOutSize;
    sendMaskScratch_.sendCntAccTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, sendCntAccAddr, sendCntAccSize / sizeof(int32_t));
}

// 在普通模板内构造 Unpermute 使用的 UB 视图，并返回当前 AIV 对应的 buffer 配置。
template <TemplateMegaMoeTypeClass>
__aicore__ inline typename MegaMoe<TemplateMegaMoeTypeFunc>::UnpermuteBufferConfig
MegaMoe<TemplateMegaMoeTypeFunc>::InitTokenUnpermuteBuffers()
{
    return CreateTokenUnpermuteBuffers<TopkWeightsType, CombineQuantMode>(
        tokenUnpermuteConfig_, commonConfig_.tokenHiddenDim, tokenUnpermuteScratch_);
}

template <TemplateMegaMoeTypeClass>
template <typename... LayoutArgs>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::RunSharedExpertGmm1Activation(
    const GMMAddrInfo &gmmAddrInfo, const ProblemShape &problemShape, const GmmExecutionConfig &gmmConfig,
    Gmm1RuntimeState &runtimeState, uint32_t sharedExpertIdx, const LayoutArgs &...layoutArgs)
{
    uint32_t expertBeforeCnt = sharedExpertIdx * commonConfig_.tokenNum;
    if constexpr (SharedQuantConfig::AXW_MODE == AxWMode::A8W4) {
        RunGmm1A8W4<typename SharedQuantConfig::QuantOutType, SharedWeightType, bfloat16_t,
                    typename SharedQuantConfig::QuantScaleType, typename SharedQuantConfig::QuantScaleType, GMM1_TILE_M,
                    L1_TILE_M_256, false, true>(sharedEpilogueOp_, params_, problemShape, gmmAddrInfo,
                                                runtimeState.startBlockIdx, gmTileSequence_, gmmConfig.blockJob,
                                                expertBeforeCnt, sharedExpertIdx, layoutArgs...);
    } else {
        constexpr bool isWeightNz = SharedWeight1Format == FORMAT_FRACTAL_NZ;
        RunGmm1Generic<typename SharedQuantConfig::QuantOutType, typename SharedQuantConfig::ActivationQuantOutType,
                       SharedWeightType, bfloat16_t, typename SharedQuantConfig::QuantScaleType,
                       typename SharedQuantConfig::QuantScaleType, isWeightNz, GMM1_TILE_M, L1_TILE_M_256, false, true>(
            sharedEpilogueOp_, params_, problemShape, gmmAddrInfo, runtimeState.startBlockIdx,
            runtimeState.vecSetSyncCom, gmmConfig.blockJob, expertBeforeCnt, sharedExpertIdx, runtimeState.pingpongIdx,
            nullptr, false, layoutArgs...);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessSharedExpertGmm1()
{
    if (gmmExecutionConfig_.blockJob.totalJobs == 0U ||
        gmmExecutionConfig_.blockJob.jobIndex >= gmmExecutionConfig_.blockJob.totalJobs) {
        return;
    }
    typename SharedBlockEpilogue::Params epilogueParams{
        .yGmAddr = params_.workspaceInfo.sharedExpertActivationDataPtr,
        .yScaleGmAddr = params_.workspaceInfo.sharedExpertActivationScalePtr,
        .x2ScaleGmAddr = nullptr,
        .x1ScaleGmAddr = nullptr,
        .biasGmAddr = nullptr,
        .clampLimit = params_.tilingData->clampLimit,
        .actMode = params_.tilingData->actMode,
        .actSubMode = params_.tilingData->actSubMode,
        .activationAlpha = params_.tilingData->activationAlpha,
        .activationBeta = params_.tilingData->activationBeta};
    sharedEpilogueOp_.Init(epilogueParams);

    ProblemShape problemShape;
    Get<M_VALUE>(problemShape) = commonConfig_.tokenNum;
    Get<N_VALUE>(problemShape) = commonConfig_.gmm1OutputDim;
    Get<K_VALUE>(problemShape) = commonConfig_.tokenHiddenDim;
    GMMAddrInfo gmmAddrInfo{};
    if constexpr (SHARED_INPUT_REUSES_MOE_QUANT) {
        // MoE 量化记录按 token 交织存放 [data | scale | optional topk]。shared GMM1 直接建立两个
        // 带行跨度的视图，避免先将 data/scale 拆分到连续 workspace。
        gmmAddrInfo.aGlobal = params_.peermemInfo.quantTokenScalePtr;
        gmmAddrInfo.aScaleGlobal = params_.peermemInfo.quantTokenScalePtr + quantProcessConfig_.quantTokenAlignBytes;
    } else {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.sharedExpertInputDataPtr;
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.sharedExpertInputScalePtr;
    }
    uint32_t startBlockIdx = 0U;
    int32_t vecSetSyncCom = 0;
    uint16_t pingpongIdx = 0U;
    Gmm1RuntimeState runtimeState{startBlockIdx, vecSetSyncCom, pingpongIdx};
    GmmExecutionConfig sharedGmmConfig = gmmExecutionConfig_;
    if constexpr (SHARED_INPUT_REUSES_MOE_QUANT) {
        sharedGmmConfig.inputLayout = {quantProcessConfig_.quantTokenScaleAlignBytes * MoeQuantConfig::A_ELEMS_PER_BYTE,
                                       quantProcessConfig_.quantTokenScaleAlignBytes /
                                           static_cast<uint32_t>(sizeof(typename MoeQuantConfig::QuantScaleType))};
    }
    constexpr bool isA8W4 = SharedQuantConfig::AXW_MODE == AxWMode::A8W4;
    for (uint32_t sharedExpertIdx = 0U; sharedExpertIdx < sharedExpertNum_; ++sharedExpertIdx) {
        UpdateSharedExpertGmm1GlobalBuffer<SharedWeightType, typename SharedQuantConfig::ActivationQuantOutType,
                                           typename SharedQuantConfig::QuantScaleType, isA8W4>(
            commonConfig_, sharedGmmConfig, params_.workspaceInfo, sharedWeightTensorListAddrs_, sharedEpilogueOp_,
            gmmAddrInfo, sharedExpertIdx);
        if constexpr (SHARED_INPUT_REUSES_MOE_QUANT) {
            RunSharedExpertGmm1Activation(gmmAddrInfo, problemShape, sharedGmmConfig, runtimeState, sharedExpertIdx,
                                          sharedGmmConfig.inputLayout);
        } else {
            RunSharedExpertGmm1Activation(gmmAddrInfo, problemShape, sharedGmmConfig, runtimeState, sharedExpertIdx);
        }
    }
    EndSync(runtimeState.vecSetSyncCom, runtimeState.pingpongIdx);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::RunSharedExpertGmm2(const GMMAddrInfo &gmmAddrInfo,
                                                                             const ProblemShape &problemShape,
                                                                             uint32_t &startBlockIdx)
{
    if constexpr (SharedQuantConfig::AXW_MODE == AxWMode::A8W4 || SharedQuantConfig::AXW_MODE == AxWMode::A4W4) {
        RunGmm2A8W4<COMBINE_NO_QUANT, typename SharedQuantConfig::ActivationQuantOutType, SharedWeightType, bfloat16_t,
                    typename SharedQuantConfig::QuantScaleType, typename SharedQuantConfig::QuantScaleType, GMM1_TILE_M,
                    false, true>(params_, problemShape, gmmAddrInfo, startBlockIdx, gmTileSequence_,
                                 gmmExecutionConfig_.blockJob, static_cast<uint32_t>(Get<M_VALUE>(problemShape)), 0U);
    } else {
        constexpr bool isWeightNz = SharedWeight2Format == FORMAT_FRACTAL_NZ;
        RunGmm2Generic<COMBINE_NO_QUANT, typename SharedQuantConfig::ActivationQuantOutType, SharedWeightType,
                       bfloat16_t, typename SharedQuantConfig::QuantScaleType,
                       typename SharedQuantConfig::QuantScaleType, isWeightNz, false, GMM1_TILE_M, false, true>(
            problemShape, gmmAddrInfo, startBlockIdx, gmmExecutionConfig_.blockJob);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessSharedExpertGmm2()
{
    ProblemShape problemShape;
    Get<M_VALUE>(problemShape) = commonConfig_.tokenNum;
    Get<N_VALUE>(problemShape) = commonConfig_.gmm1OutputDim;
    Get<K_VALUE>(problemShape) = commonConfig_.tokenHiddenDim;
    uint32_t startBlockIdx = 0U;
    GMMAddrInfo gmmAddrInfo{};
    constexpr bool isW4 = SharedQuantConfig::AXW_MODE == AxWMode::A8W4 || SharedQuantConfig::AXW_MODE == AxWMode::A4W4;
    for (uint32_t sharedExpertIdx = 0U; sharedExpertIdx < sharedExpertNum_; ++sharedExpertIdx) {
        UpdateSharedExpertGmm2GlobalBuffer<typename SharedQuantConfig::ActivationQuantOutType, SharedWeightType,
                                           typename SharedQuantConfig::QuantScaleType, GMM1_TILE_M, isW4>(
            commonConfig_, gmmExecutionConfig_, params_.workspaceInfo, sharedWeightTensorListAddrs_, gmmAddrInfo,
            sharedExpertIdx);
        RunSharedExpertGmm2(gmmAddrInfo, problemShape, startBlockIdx);
    }
}

// 计算当前专家的 token 数，并在非空时执行完整 Dispatch。
template <TemplateMegaMoeTypeClass>
__aicore__ inline uint32_t MegaMoe<TemplateMegaMoeTypeFunc>::DispatchMoeExpert(uint32_t expertIdx)
{
    uint32_t expertTokenCount = 0U;
    ComputeExpertTokenCountAndNotify<
        typename MoeQuantConfig::QuantStorageType,
        MoeQuantConfig::AXW_MODE == AxWMode::A8W4 || MoeQuantConfig::AXW_MODE == AxWMode::A4W4, TopkWeightsPrefetch>(
        tokenDispatchConfig_, commonConfig_, countWorkspace_, params_, tokenDispatchScratch_, expertIdx,
        expertTokenCount);
    if (expertTokenCount != 0U) {
        int32_t expertGlobalRowBegin =
            GetExpertGlobalRowBegin(tokenDispatchScratch_, expertIdx, commonConfig_.worldSize);
        DispatchExpertTokensByRankShard<typename MoeQuantConfig::QuantStorageType,
                                        typename MoeQuantConfig::QuantScaleType, GMM1_TILE_M, TopkWeightsPrefetch>(
            tokenDispatchConfig_, commonConfig_, gmmExecutionConfig_.blockJob, countWorkspace_, syncWorkspaceLayout_,
            params_, g_winRankAddr_, tokenDispatchScratch_, expertIdx, expertGlobalRowBegin);
    }
    return expertTokenCount;
}

// 从 workspace 读取 token 数并准备 GMM 专家状态；GMM1 可按需先等待 token count ready。
template <TemplateMegaMoeTypeClass>
template <bool WaitForTokenCountReady>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::PrepareGmmExpertState(ExpertLoopState &state,
                                                                               uint32_t expertIdx)
{
    if constexpr (WaitForTokenCountReady) {
        if (GetSubBlockIdx() == 0U) {
            WaitForMoeExpertTokenCountReady(params_.workspaceInfo.flagSendCntCalToUpdParamsPtr, countWorkspace_,
                                            expertIdx);
        }
    }
    uint32_t expertTokenCount =
        GetExpertTokenCountFromWorkspace(params_.workspaceInfo.expertRevTokenNumsPtr, countWorkspace_, expertIdx);
    UpdateExpertLoopState(state, expertIdx, expertTokenCount);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessInputPreparationStage()
{
    SendAndQuantBuffInit();
    InterleavedQuantOutput moeQuantOutput;
    moeQuantOutput.dataAndScale.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint8_t *>(params_.peermemInfo.quantTokenScalePtr));
    QuantizeLocalTokens<MoeQuantMode, typename MoeQuantConfig::QuantOutType, typename MoeQuantConfig::QuantStorageType,
                        TopkWeightsType, TopkWeightsPrefetch>(aivJob_, commonConfig_, params_, quantProcessConfig_,
                                                              moeQuantOutput, quantScratch_);
    if (sharedExpertNum_ > 0) {
        if constexpr (!SHARED_INPUT_REUSES_MOE_QUANT) {
            using SharedQuantStorageType = typename SharedQuantConfig::QuantStorageType;
            using SharedQuantScaleType = typename SharedQuantConfig::QuantScaleType;
            SeparatedQuantOutput<SharedQuantStorageType, SharedQuantScaleType> sharedQuantOutput;
            sharedQuantOutput.data.SetGlobalBuffer(
                reinterpret_cast<__gm__ SharedQuantStorageType *>(params_.workspaceInfo.sharedExpertInputDataPtr));
            sharedQuantOutput.scale.SetGlobalBuffer(
                reinterpret_cast<__gm__ SharedQuantScaleType *>(params_.workspaceInfo.sharedExpertInputScalePtr));
            QuantizeLocalTokens<SharedQuantMode, typename SharedQuantConfig::QuantOutType,
                                typename SharedQuantConfig::QuantStorageType, TopkWeightsType, false>(
                aivJob_, commonConfig_, params_, sharedQuantProcessConfig_, sharedQuantOutput, sharedQuantScratch_);
        }
    }
    GatherAndSendExpertMasks(aivJob_, commonConfig_, params_, g_winRankAddr_, sendMaskConfig_, sendMaskScratch_);
    ResetSyncStatus<TopkWeightsPrefetch>(aivJob_, params_, resetBatchElementCount_, resetTensor_);
}

/*
 * 所有 MTE Wave 路径共用相同的阶段边界，派生类只负责 MoE 专家 Wave 的具体编排。
 */
template <TemplateMegaMoeTypeClass>
template <typename Derived>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessWave(Derived &derived)
{
    // 保存入口时的溢出模式，计算期间关闭溢出检查。
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);

    // 阶段 1：量化本卡输入、推送路由 mask、清零同步状态，并准备共享专家输入。
    exceptionDump_.UpdateStage(MegaMoeImpl::Stage::INPUT_PREPARE);
    ProcessInputPreparationStage();
    if constexpr (g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
    }
    SyncAll<false>(); // AIC 等待 AIV 完成输入准备后再进入 GMM。

    if (sharedExpertNum_ > 0U) {
        // 阶段 2：可选的共享专家 GMM1 及 Activation。
        exceptionDump_.UpdateStage(MegaMoeImpl::Stage::SHARED_EXPERT_GMM1);
        ProcessSharedExpertGmm1();
    }

    // 等待所有 rank 完成输入准备，再消费远端 Dispatch 数据。
    exceptionDump_.UpdateStage(MegaMoeImpl::Stage::CROSS_RANK_SYNC_INPUT);
    CrossRankSyncInWorldSize(params_.peermemInfo.rankSyncInWorldPtr, rankId_, worldSize_, aivJob_);
    // 阶段 3：由派生类编排 MoE 专家的 Dispatch、GMM1/Activation 和 GMM2/Combine。
    derived.ProcessMoeExpertStages();
    if constexpr (g_coreType == AIV) {
        // 导出各 MoE 专家最终 token 数，并在进入后续阶段前完成核间同步。
        if (GetSubBlockIdx() == 1U && countWorkspace_.blockIdx == 0U) {
            constexpr bool reloadCumsumFromGm =
                MoeQuantConfig::AXW_MODE == AxWMode::A8W4 || MoeQuantConfig::AXW_MODE == AxWMode::A4W4;
            ExportExpertTokenCounts<typename MoeQuantConfig::QuantStorageType, reloadCumsumFromGm>(
                commonConfig_, tokenDispatchScratch_, params_);
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
    }
    if (sharedExpertNum_ > 0U) {
        // 阶段 4：可选的共享专家 GMM2，其结果由输出聚合阶段合并。
        exceptionDump_.UpdateStage(MegaMoeImpl::Stage::SHARED_EXPERT_GMM2);
        ProcessSharedExpertGmm2();
    }

    // 阶段 5：所有 rank 完成 Combine 结果发送后，AIV 执行输出聚合。
    if constexpr (g_coreType == AIV) {
        exceptionDump_.UpdateStage(MegaMoeImpl::Stage::CROSS_RANK_SYNC_OUTPUT);
        CrossRankSyncInWorldSize(params_.peermemInfo.rankSyncInWorldPtr, rankId_, worldSize_, aivJob_);
        exceptionDump_.UpdateStage(MegaMoeImpl::Stage::UNPERMUTE);
        MegaMoeUnpermuteBufferConfig unpermuteBufferConfig = InitTokenUnpermuteBuffers();
        UnpermuteTokens<CombineQuantMode, TopkWeightsType, TopkWeightsPrefetch, GMM1_TILE_M>(
            tokenUnpermuteConfig_, commonConfig_, params_, tokenUnpermuteScratch_, unpermuteBufferConfig);
    }

    // 恢复入口时的溢出模式。
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
    exceptionDump_.UpdateStage(MegaMoeImpl::Stage::COMPLETE);
}

} // namespace MegaMoeImpl
#undef TemplateMegaMoeTypeClass
#undef TemplateMegaMoeTypeFunc
#endif
