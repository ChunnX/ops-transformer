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
 * \brief Multi-core entry point: orchestrates single-core modules using blockIdx-derived TileInfo.
 *
 * This file is the multi-core dispatcher. It computes TileInfo from blockIdx and calls
 * the 7 single-core module files (TopKGather, TokenDispatch, GMM1, SwigluQuant,
 * GMM2, CombineSend, CombineReduce) located in the modules/ subdirectory.
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
#if __has_include("../../moe_distribute_dispatch_v2/quantize_functions.h")
#include "../../moe_distribute_dispatch_v2/quantize_functions.h"
#else
#include "../../../moe_distribute_dispatch_v2/op_kernel/quantize_functions.h"
#endif

#include "modules/mega_moe_tile_info.h"
#include "modules/topk_gather.h"
#include "modules/token_dispatch.h"
#include "modules/gmm1.h"
#include "modules/swiglu_quant.h"
#include "modules/gmm2.h"
#include "modules/combine_send.h"
#include "modules/combine_reduce.h"

using namespace AscendC;

namespace MegaMoeImpl {

#define TemplateMegaMoeTypeClass typename XType, typename OutputType, typename TopkWeightsType, \
                                  typename Weight1Type, int32_t QuantMode, int32_t CombineQuantMode
#define TemplateMegaMoeTypeFunc XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode

template <TemplateMegaMoeTypeClass>
class MegaMoe {
public:
    template <int32_t QM> struct QuantTraits { using OutType = fp8_e4m3fn_t; };
    template <> struct QuantTraits<E5M2_QUANT> { using OutType = fp8_e5m2_t; };
    template <> struct QuantTraits<E2M1_QUANT> { using OutType = fp4x2_e2m1_t; };
    using QuantOutType = typename QuantTraits<QuantMode>::OutType;
    using ActivationType = typename std::conditional<
        Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value, uint8_t, QuantOutType>::type;
    using QuantScaleOutType = typename std::conditional<(QuantMode >= E5M2_QUANT), fp8_e8m0_t, float>::type;

    using SwigluQuantOutType = typename std::conditional<
        (QuantMode == E2M1_QUANT), fp8_e4m3fn_t, QuantOutType>::type;

    static constexpr uint32_t A_ELEMS_PER_BYTE = Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value ? 2U : 1U;
    static constexpr uint32_t B_ELEMS_PER_BYTE = Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value ? 2U : 1U;
    static constexpr uint32_t C_ELEMS_PER_BYTE = Std::IsSame<SwigluQuantOutType, fp4x2_e2m1_t>::value ? 2U : 1U;
    static constexpr bool ENABLE_A8W4 = Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value &&
                                            Std::IsSame<QuantOutType, fp8_e4m3fn_t>::value;
    static constexpr bool ENABLE_A4W4 = Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value &&
                                            Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value;
    static constexpr int32_t DISPATCH_BUFFER_NUM = 6;

    using BlockEpilogue = BlockEpilogueSwigluMxQuant<SwigluQuantOutType, bfloat16_t,
        QuantScaleOutType, QuantScaleOutType, true>;

    __aicore__ inline MegaMoe() {};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights,
        GM_ADDR weight1, GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
        GM_ADDR scales, GM_ADDR yOut, GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM,
        MegaMoeTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void DispatchBuffInit();
    __aicore__ inline void SendAndQuantBuffInit();
    __aicore__ inline void UnpermuteBuffInit();

    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    Params params_{};

    GlobalTensor<int32_t> swigluToGmm2FlagGm_;
    GlobalTensor<int32_t> expertTokenNumsOut_;
    GlobalTensor<int32_t> tripleGlobalTensor_;
    GlobalTensor<int32_t> expertRevNumsGlobalTensor_;
    GlobalTensor<int32_t> cumsumInfoGlobalTensor_;

    uint32_t m_ = 0;
    uint32_t k_ = 0;
    uint32_t aicNum_ = 0;
    uint32_t topK_ = 0;
    uint32_t rankId_ = 0;
    uint32_t worldSize_ = 0;
    uint32_t expertPerRank_ = 0;
    int64_t hiddenDim_ = 0;
    uint64_t maxOutputSize_ = 0;
    int32_t vecSetSyncCom_ = 0;
    uint32_t startBlockIdx_ = 0;
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
    uint32_t ubBufferUsedAddr_ = 0;
    uint16_t gmm2PingPongIdx_ = 0;
    uint64_t sendTotalNum_ = 0;
    uint32_t maskAlignSize_ = 0;
    uint32_t maskSlotSize_ = 0;
    uint64_t maskWinOffset_ = 0;
    uint64_t quantWinOffset_ = 0;
    uint64_t cumsumRevCntInRank_ = 0;
    int32_t compareCount_ = 0;
    int64_t combineUbTensorSize_ = 0;

    LocalTensor<int32_t> expertTokenCntTensor_;
    LocalTensor<uint8_t> gatherMaskTensor_;
    LocalTensor<uint32_t> gatherMaskInt32Tensor_;
    LocalTensor<int32_t> cumsumInfoTensor_;
    LocalTensor<int32_t> validTopkIndexTensor_;
    LocalTensor<int32_t> topkIndexTensor_;
    LocalTensor<ActivationType> copyTmpTensors_[DISPATCH_BUFFER_NUM];
    LocalTensor<int32_t> tripleTensor_;
    LocalTensor<int32_t> expertTokenNumsOutTensor_;
    LocalTensor<bfloat16_t> dataResTensor_;
    LocalTensor<float> dataResFp32Tensor_;
    LocalTensor<float> topKWeightsTensor_;
    LocalTensor<float> fp32ScaleTensor_;
    LocalTensor<bfloat16_t> bf16ScaleTensor_;

    BlockEpilogue epilogueOp_;

    MegaMoeModules::TileInfo tile_;
};

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales, GM_ADDR yOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData *tilingData)
{
    m_ = tilingData->bs;
    k_ = tilingData->h;
    aicNum_ = tilingData->aicNum;
    topK_ = tilingData->topK;
    sendTotalNum_ = m_ * topK_;
    worldSize_ = tilingData->epWorldSize;
    expertPerRank_ = tilingData->expertPerRank;
    blockNumPerRank_ = tilingData->blockNumPerEP;
    maxOutputSize_ = tilingData->maxOutputSize;
    maxWavesPerExpert_ = static_cast<int32_t>(Ops::Base::CeilDiv(
        static_cast<int64_t>(maxOutputSize_), DISPATCH_WAVE_TILE_M));
    dispatchFlagSlotsPerExpert_ = static_cast<int32_t>(Ops::Base::CeilAlign(
        static_cast<int64_t>(maxWavesPerExpert_), static_cast<int64_t>(INT_CACHELINE)));
    hiddenDim_ = tilingData->hiddenDim;
    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext*>(context);
    rankId_ = mc2Context_->epRankId;
    for (int i = 0; i < worldSize_; i++) {
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
    }
    params_.aGmAddr = x;
    params_.expertIdxGmAddr = topkIds;
    params_.bGmAddr = GetTensorAddr(0, weight1);
    params_.b2GmAddr = GetTensorAddr(0, weight2);
    params_.bScaleGmAddr = GetTensorAddr(0, weightScales1);
    params_.b2ScaleGmAddr = GetTensorAddr(0, weightScales2);
    params_.y2GmAddr = yOut;
    params_.expertTokenNumsOutGmAddr = expertTokenNumsOut;
    params_.probsGmAddr = topkWeights;
    params_.workspaceInfo = WorkspaceInfo(workspaceGM, tilingData);
    params_.peermemInfo = PeermemInfo(winRankAddr_[rankId_], tilingData, A_ELEMS_PER_BYTE);
    params_.tilingData = tilingData;
    expertTokenNumsOut_.SetGlobalBuffer((__gm__ int32_t*)params_.expertTokenNumsOutGmAddr);
    expertRevNumsGlobalTensor_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.expertRevTokenNumsPtr);
    tripleGlobalTensor_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.tripleInfoPtr);
    uint64_t cumsumStride =
        Ops::Base::CeilAlign(static_cast<int64_t>(worldSize_ * expertPerRank_ * sizeof(int32_t)), ALIGN_32);
    cumsumInfoGlobalTensor_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.cumsumInfoPtr + cumsumStride * blockIdx_));
    epilogueOp_.Init({params_.workspaceInfo.swigluQuantDataPtr, params_.workspaceInfo.swigluQuantScalePtr,
        params_.workspaceInfo.flagSwiGluToGmm2Ptr, nullptr, nullptr, nullptr, ALIGN_256, ALIGN_256,
        tilingData->clampLimit});
    maskWinOffset_ = static_cast<uint64_t>(params_.peermemInfo.maskRecvPtr -
        params_.peermemInfo.rankSyncInWorldPtr);
    quantWinOffset_ = static_cast<uint64_t>(params_.peermemInfo.quantTokenScalePtr -
        params_.peermemInfo.rankSyncInWorldPtr);
    compareCount_ = Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_256)) / sizeof(int32_t);
    maskAlignSize_ = Ops::Base::CeilAlign(static_cast<int64_t>(compareCount_) / 8, static_cast<int64_t>(ALIGN_32));
    maskSlotSize_ = maskAlignSize_ + static_cast<uint32_t>(ALIGN_32);
    mxQuantScaleNumAlignPerToken_ = Ops::Base::CeilDiv(k_, static_cast<uint32_t>(ALIGN_32));
    mxQuantTokenAlignBytes_ = Ops::Base::CeilAlign(static_cast<uint32_t>(k_ / A_ELEMS_PER_BYTE),
        static_cast<uint32_t>(ALIGN_256)) * sizeof(ActivationType);
    mxQuantScaleAlignBytes_ = mxQuantScaleNumAlignPerToken_ * sizeof(uint8_t);
    mxQuantTokenScaleAlignBytes_ = Ops::Base::CeilAlign(mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_,
        static_cast<uint32_t>(ALIGN_32));

    // Compute TileInfo from blockIdx (multi-core dispatch)
    tile_.tileIdx = blockIdx_;
    tile_.totalTiles = blockNum_;
    tile_.aivTileIdx = aivCoreIdx_;
    tile_.aivTotalTiles = blockAivNum_;
    tile_.subTileIdx = subBlockIdx_;
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::DispatchBuffInit()
{
    if constexpr(g_coreType == AIC) {
        return;
    }

    uint32_t expertTokenCntTensorAddr = 0;
    uint32_t expertTokenCntTensorSize = ALIGN_32;
    expertTokenCntTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, expertTokenCntTensorAddr,
        expertTokenCntTensorSize / sizeof(int32_t));
    uint32_t cumsumInfoTensorAddr = expertTokenCntTensorAddr + expertTokenCntTensorSize;
    uint32_t cumsumInfoTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(worldSize_ * expertPerRank_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    cumsumInfoTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, cumsumInfoTensorAddr,
        cumsumInfoTensorSize / sizeof(int32_t));
    uint32_t gatherMaskTensorAddr = cumsumInfoTensorAddr + cumsumInfoTensorSize;
    uint32_t gatherMaskTensorSize = maskSlotSize_ * worldSize_;
    gatherMaskTensor_ = LocalTensor<uint8_t>(TPosition::VECCALC, gatherMaskTensorAddr,
        gatherMaskTensorSize / sizeof(uint8_t));
    gatherMaskInt32Tensor_ = LocalTensor<uint32_t>(TPosition::VECCALC, gatherMaskTensorAddr,
        gatherMaskTensorSize / sizeof(uint32_t));
    uint32_t validTopkIndexTensorAddr = gatherMaskTensorAddr + gatherMaskTensorSize;
    uint32_t validTopkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    validTopkIndexTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, validTopkIndexTensorAddr,
        validTopkIndexTensorSize / sizeof(int32_t));
    uint32_t topkIndexTensorAddr = validTopkIndexTensorAddr + validTopkIndexTensorSize;
    uint32_t topkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    topkIndexTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, topkIndexTensorAddr,
        topkIndexTensorSize / sizeof(int32_t));
    uint32_t tokenScaleSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_),
        static_cast<int64_t>(ALIGN_32));
    uint32_t COPY_TMP_BUFFER_SIZE = tokenScaleSize;
    uint32_t copyTmpBaseAddr = topkIndexTensorAddr + topkIndexTensorSize;
    for (int32_t index = 0; index < DISPATCH_BUFFER_NUM; ++index) {
        copyTmpTensors_[index] = LocalTensor<ActivationType>(TPosition::VECCALC,
            copyTmpBaseAddr + static_cast<uint32_t>(index) * COPY_TMP_BUFFER_SIZE,
            COPY_TMP_BUFFER_SIZE / sizeof(ActivationType));
    }
    uint32_t expertTokenNumsOutTensorAddr = copyTmpBaseAddr +
        static_cast<uint32_t>(DISPATCH_BUFFER_NUM) * COPY_TMP_BUFFER_SIZE;
    uint32_t expertTokenNumsOutTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(expertPerRank_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    expertTokenNumsOutTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, expertTokenNumsOutTensorAddr,
        expertTokenNumsOutTensorSize / sizeof(int32_t));
    ubBufferUsedAddr_ = expertTokenNumsOutTensorAddr + expertTokenNumsOutTensorSize;
    CreateVecIndex(topkIndexTensor_, 0, (topkIndexTensorSize / sizeof(int32_t)));
    Duplicate<int32_t>(cumsumInfoTensor_, 0, (cumsumInfoTensorSize / sizeof(int32_t)));
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::SendAndQuantBuffInit()
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    // BuffInit for TopKGather module
    MegaMoeModules::TopKGather::TopKGatherState topkGatherState;
    MegaMoeModules::TopKGather::BuffInit(topkGatherState, tile_, params_, m_, topK_,
        expertPerRank_, worldSize_, aicNum_, rankId_, blockAivNum_, dispatchFlagSlotsPerExpert_);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::UnpermuteBuffInit()
{
    MegaMoeModules::CombineReduce::CombineReduceState<CombineQuantMode> combineReduceState;
    GlobalTensor<float> topKWeightsGlobal;
    GlobalTensor<bfloat16_t> topkWeightsBf16Global;
    MegaMoeModules::CombineReduce::BuffInit(combineReduceState, tile_, params_, m_, k_, topK_,
        worldSize_, expertPerRank_, rankId_,
        topKWeightsGlobal, topkWeightsBf16Global,
        typename std::conditional<Std::IsSame<TopkWeightsType, float>::value, float, bfloat16_t>::type{});
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Process()
{
    // ============================
    // Multi-core orchestration: compute TileInfo from blockIdx, dispatch to single-core modules
    // ============================
    tile_.tileIdx = blockIdx_;
    tile_.totalTiles = blockNum_;
    tile_.aivTileIdx = aivCoreIdx_;
    tile_.aivTotalTiles = blockAivNum_;
    tile_.subTileIdx = subBlockIdx_;

    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);

    // ===== Phase 1: TopKGather =====
    SendAndQuantBuffInit();
    MegaMoeModules::TopKGather::TopKGatherState topkGatherState;
    MegaMoeModules::TopKGather::BuffInit(topkGatherState, tile_, params_, m_, topK_,
        expertPerRank_, worldSize_, aicNum_, rankId_, blockAivNum_, dispatchFlagSlotsPerExpert_);
    MegaMoeModules::TopKGather::SendMaskCal(topkGatherState, tile_);
    MegaMoeModules::TopKGather::ResetFlagList(topkGatherState, tile_, swigluToGmm2FlagGm_);

    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        MegaMoeModules::GMM2::GMM2State<QuantOutType, Weight1Type, SwigluQuantOutType,
            QuantScaleOutType, QuantMode, CombineQuantMode, ENABLE_A8W4, ENABLE_A4W4> gmm2StateEarly;
        gmm2StateEarly.params = params_;
        MegaMoeModules::GMM2::ResetGmm2CombineSyncCounters(gmm2StateEarly, tile_, topkGatherState.resetTensor_);
    }

    // ===== Phase 1b: TokenDispatch - QuantProcessInRank =====
    DispatchBuffInit();
    MegaMoeModules::TokenDispatch::TokenDispatchState<ActivationType, QuantOutType, QuantMode> dispatchState;
    dispatchState.params = params_;
    dispatchState.m = m_;
    dispatchState.k = k_;
    dispatchState.topK = topK_;
    dispatchState.sendTotalNum = sendTotalNum_;
    dispatchState.worldSize = worldSize_;
    dispatchState.expertPerRank = expertPerRank_;
    dispatchState.rankId = rankId_;
    dispatchState.blockNumPerRank = blockNumPerRank_;
    dispatchState.maxOutputSize = maxOutputSize_;
    dispatchState.mxQuantScaleNumAlignPerToken = mxQuantScaleNumAlignPerToken_;
    dispatchState.mxQuantTokenAlignBytes = mxQuantTokenAlignBytes_;
    dispatchState.mxQuantScaleAlignBytes = mxQuantScaleAlignBytes_;
    dispatchState.mxQuantTokenScaleAlignBytes = mxQuantTokenScaleAlignBytes_;
    dispatchState.quantWinOffset = quantWinOffset_;
    dispatchState.cumsumRevCntInRank = 0;
    dispatchState.compareCount = compareCount_;
    dispatchState.maskAlignSize = maskAlignSize_;
    dispatchState.maskSlotSize = maskSlotSize_;
    dispatchState.k = k_;
    dispatchState.topK = topK_;

    // Copy UB buffer references from class members to module state
    dispatchState.expertTokenCntTensor_ = expertTokenCntTensor_;
    dispatchState.cumsumInfoTensor_ = cumsumInfoTensor_;
    dispatchState.gatherMaskTensor_ = gatherMaskTensor_;
    dispatchState.gatherMaskInt32Tensor_ = gatherMaskInt32Tensor_;
    dispatchState.validTopkIndexTensor_ = validTopkIndexTensor_;
    dispatchState.topkIndexTensor_ = topkIndexTensor_;
    for (int32_t i = 0; i < DISPATCH_BUFFER_NUM; ++i) {
        dispatchState.copyTmpTensors_[i] = copyTmpTensors_[i];
    }
    dispatchState.expertTokenNumsOutTensor_ = expertTokenNumsOutTensor_;
    dispatchState.ubBufferUsedAddr = ubBufferUsedAddr_;

    MegaMoeModules::TokenDispatch::QuantProcessInRank(dispatchState, tile_);
    if constexpr(g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
    }
    SyncAll<false>();

    // ===== Cross-rank sync =====
    MegaMoeModules::CombineReduce::CombineReduceState<CombineQuantMode> combineReduceState;
    combineReduceState.params = params_;
    combineReduceState.m = m_;
    combineReduceState.k = k_;
    combineReduceState.topK = topK_;
    combineReduceState.worldSize = worldSize_;
    combineReduceState.expertPerRank = expertPerRank_;
    combineReduceState.rankId = rankId_;
    MegaMoeModules::CombineReduce::CrossRankSyncInWorldSize(combineReduceState, tile_);

    // ===== Phase 2: GMM1 + SwigluQuant =====
    GMMAddrInfo dispatchAddrInfo;
    GMMAddrInfo gmm1AddrInfo;
    TupleShape initShape;
    Get<N_VALUE>(initShape) = hiddenDim_;
    Get<K_VALUE>(initShape) = k_;
    BlockOffset initOffset{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    ExpertLoopState dispatchStateLoop{initShape, initOffset, 0};
    ExpertLoopState gmm1StateLoop{initShape, initOffset, 0};

    MegaMoeModules::GMM1::GMM1State<QuantOutType, Weight1Type, SwigluQuantOutType,
        QuantScaleOutType, QuantMode, ENABLE_A8W4, ENABLE_A4W4> gmm1State;
    gmm1State.epilogueOp = epilogueOp_;
    gmm1State.startBlockIdx = startBlockIdx_;
    gmm1State.vecSetSyncCom = vecSetSyncCom_;
    gmm1State.aicNum = aicNum_;
    gmm1State.params = params_;

    MegaMoeModules::TokenDispatch::TokenDispatchState<ActivationType, QuantOutType, QuantMode> tokenDispatchFull;
    tokenDispatchFull = dispatchState;

    uint64_t curSendCnt = 0;
    uint64_t nextSendCnt = 0;

    if constexpr(g_coreType == AIV) {
        if (subBlockIdx_ == 1) {
            MegaMoeModules::TopKGather::SendCntCal(topkGatherState, tile_, 0, nextSendCnt,
                expertTokenCntTensor_, gatherMaskInt32Tensor_, gatherMaskTensor_,
                cumsumInfoTensor_, expertRevNumsGlobalTensor_, cumsumInfoGlobalTensor_, cumsumRevCntInRank_);
            if (MegaMoeModules::GMM1::UpdateGroupParams(gmm1State, tile_, dispatchStateLoop, 0, nextSendCnt,
                expertRevNumsGlobalTensor_)) {
                MegaMoeModules::GMM1::UpdateGlobalBuffer(gmm1State, tile_, dispatchAddrInfo, dispatchStateLoop,
                    dispatchFlagSlotsPerExpert_);
                MegaMoeModules::TokenDispatch::TripleInfoCalAndDispatch<ActivationType, QuantOutType, QuantScaleOutType,
                    QuantMode, ENABLE_A8W4>(tokenDispatchFull, tile_, dispatchAddrInfo, 0);
            }
        }
    }

    for (int localExpertId = 0; localExpertId < expertPerRank_; localExpertId++) {
        curSendCnt = nextSendCnt;

        if constexpr(g_coreType == AIV) {
            if (subBlockIdx_ == 1 && localExpertId + 1 < expertPerRank_) {
                MegaMoeModules::TopKGather::SendCntCal(topkGatherState, tile_, localExpertId + 1, nextSendCnt,
                    expertTokenCntTensor_, gatherMaskInt32Tensor_, gatherMaskTensor_,
                    cumsumInfoTensor_, expertRevNumsGlobalTensor_, cumsumInfoGlobalTensor_, cumsumRevCntInRank_);
                if (MegaMoeModules::GMM1::UpdateGroupParams(gmm1State, tile_, dispatchStateLoop, localExpertId + 1,
                    nextSendCnt, expertRevNumsGlobalTensor_)) {
                    MegaMoeModules::GMM1::UpdateGlobalBuffer(gmm1State, tile_, dispatchAddrInfo, dispatchStateLoop,
                        dispatchFlagSlotsPerExpert_);
                    MegaMoeModules::TokenDispatch::TripleInfoCalAndDispatch<ActivationType, QuantOutType, QuantScaleOutType,
                        QuantMode, ENABLE_A8W4>(tokenDispatchFull, tile_, dispatchAddrInfo, localExpertId + 1);
                }
            }
        }

        if (!MegaMoeModules::GMM1::UpdateGroupParams(gmm1State, tile_, gmm1StateLoop, localExpertId,
            curSendCnt, expertRevNumsGlobalTensor_)) {
            continue;
        }
        MegaMoeModules::GMM1::UpdateGlobalBuffer(gmm1State, tile_, gmm1AddrInfo, gmm1StateLoop,
            dispatchFlagSlotsPerExpert_);
        MegaMoeModules::GMM1::GroupMatmulWithSwigluQuant(gmm1State, tile_, gmm1AddrInfo, gmm1StateLoop);
    }
    EndSync(gmm1State.vecSetSyncCom);
    startBlockIdx_ = gmm1State.startBlockIdx;
    vecSetSyncCom_ = gmm1State.vecSetSyncCom;

    if constexpr(g_coreType == AIV) {
        if (subBlockIdx_ == 1) {
            MegaMoeModules::TokenDispatch::ExpertTokenNumCopyOut<ActivationType, QuantOutType, QuantMode, ENABLE_A8W4>(
                tokenDispatchFull, tile_, expertTokenNumsOut_, cumsumInfoGlobalTensor_);
        }
    }

    // ===== Phase 3: GMM2 + Combine =====
    vecSetSyncCom_ = 0;
    GMMAddrInfo gmm2AddrInfo;
    ExpertLoopState gmm2StateLoop{initShape, initOffset, 0};

    MegaMoeModules::GMM2::GMM2State<QuantOutType, Weight1Type, SwigluQuantOutType,
        QuantScaleOutType, QuantMode, CombineQuantMode, ENABLE_A8W4, ENABLE_A4W4> gmm2State;
    gmm2State.startBlockIdx = startBlockIdx_;
    gmm2State.vecSetSyncCom = vecSetSyncCom_;
    gmm2State.gmm2PingPongIdx = gmm2PingPongIdx_;
    gmm2State.aicNum = aicNum_;
    gmm2State.combineUbTensorSize = combineUbTensorSize_;
    gmm2State.dispatchFlagSlotsPerExpert = dispatchFlagSlotsPerExpert_;
    gmm2State.params = params_;

    MegaMoeModules::GMM2::InitCombineBuffers(gmm2State, tile_, k_);

    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
        if (!MegaMoeModules::GMM2::UpdateGroupParams(gmm2State, tile_, gmm2StateLoop, expertIdx,
            expertRevNumsGlobalTensor_)) {
            continue;
        }
        MegaMoeModules::GMM2::UpdateGlobalBuffer(gmm2State, tile_, gmm2AddrInfo, gmm2StateLoop);
        MegaMoeModules::GMM2::GroupMatmulWithCombine(gmm2State, tile_, gmm2AddrInfo, gmm2StateLoop, expertIdx);
    }
    if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        EndGMM2Sync(gmm2State.vecSetSyncCom, gmm2State.gmm2PingPongIdx);
    }
    startBlockIdx_ = gmm2State.startBlockIdx;
    vecSetSyncCom_ = gmm2State.vecSetSyncCom;
    gmm2PingPongIdx_ = gmm2State.gmm2PingPongIdx;

    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();

    // ===== Phase 4: CombineReduce =====
    if constexpr(g_coreType == AIV) {
        UnpermuteBuffInit();
        MegaMoeModules::CombineReduce::CrossRankSyncInWorldSize(combineReduceState, tile_);
        MegaMoeModules::CombineReduce::Unpermute(combineReduceState, tile_);
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

}   // namespace MegaMoeImpl
#endif
