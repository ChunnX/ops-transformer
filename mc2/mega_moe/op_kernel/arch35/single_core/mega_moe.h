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
 * \brief Multi-core orchestrator: class MegaMoe retains all buffer ownership,
 *        computes TileInfo from hardware APIs, and calls single-core module
 *        free functions from single_core/ headers.
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

#include "single_core/mega_moe_tile_info.h"
#include "single_core/topk_gather.h"
#include "single_core/token_dispatch.h"
#include "single_core/gmm1.h"
#include "single_core/swiglu_quant.h"
#include "single_core/gmm2.h"
#include "single_core/combine_send.h"
#include "single_core/combine_reduce.h"

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

    using Gmm1Module = MegaMoeModules::GMM1::Gmm1Module<
        QuantOutType, Weight1Type, SwigluQuantOutType, QuantScaleOutType,
        QuantMode, ENABLE_A8W4, ENABLE_A4W4>;
    using Gmm2Module = GMM2::Gmm2Module<
        QuantOutType, Weight1Type, SwigluQuantOutType, QuantScaleOutType,
        QuantMode, CombineQuantMode, ENABLE_A8W4, ENABLE_A4W4>;
    using CombineReduceBuffers = CombineReduce::CombineReduceBuffers<CombineQuantMode>;

    __aicore__ inline MegaMoe() {};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights,
        GM_ADDR weight1, GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
        GM_ADDR scales, GM_ADDR yOut, GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM,
        MegaMoeTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void DispatchBuffInit();
    __aicore__ inline void SendAndQuantBuffInit();

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

    BlockEpilogue epilogueOp_;

    LocalTensor<int32_t> topkIndexTensor_;
    LocalTensor<uint8_t> gatherMaskTensor_;
    LocalTensor<uint32_t> gatherMaskInt32Tensor_;
    LocalTensor<int32_t> expertTokenCntTensor_;
    LocalTensor<int32_t> validTopkIndexTensor_;
    LocalTensor<int32_t> cumsumInfoTensor_;
    LocalTensor<ActivationType> copyTmpTensors_[DISPATCH_BUFFER_NUM];
    LocalTensor<int32_t> tripleTensor_;
    LocalTensor<bfloat16_t> xInTensor1_;
    LocalTensor<bfloat16_t> xInTensor2_;
    LocalTensor<ActivationType> xOutTensor1_;
    LocalTensor<ActivationType> xOutTensor2_;
    LocalTensor<uint16_t> mxTempTensor_;
    LocalTensor<int32_t> resetTensor_;
    LocalTensor<int32_t> topkIdsTensor_;
    LocalTensor<uint8_t> sendMaskTensor_[DOUBLE_BUFFER];
    LocalTensor<int32_t> sendGatherOutTensor_;
    LocalTensor<int32_t> expertTokenNumsOutTensor_;
    LocalTensor<bfloat16_t> dataResTensor_;
    LocalTensor<float> dataResFp32Tensor_;
    LocalTensor<float> topKWeightsTensor_;
    LocalTensor<float> fp32ScaleTensor_;
    LocalTensor<bfloat16_t> bf16ScaleTensor_;

    CombineReduceBuffers combineReduceBuf_;

    Gmm1Module gmm1Module_;
    Gmm2Module gmm2Module_;
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
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.cumsumInfoPtr +
            cumsumStride * (GetBlockIdx() / GetTaskRation())));
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
    uint32_t topkIdsTensorAddr = 0;
    uint32_t topkIdsTensorSize = Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_256));
    topkIdsTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, topkIdsTensorAddr, topkIdsTensorSize / sizeof(int32_t));
    uint32_t resetTensorAddr = topkIdsTensorAddr + topkIdsTensorSize;
    uint64_t totalFlagInt32 = static_cast<uint64_t>(expertPerRank_) *
        (static_cast<uint64_t>(INT_CACHELINE) + static_cast<uint64_t>(dispatchFlagSlotsPerExpert_) +
        static_cast<uint64_t>(INT_CACHELINE) * static_cast<uint64_t>(aicNum_));
    uint32_t blockAivNum = GetBlockNum() * 2;
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        int64_t tokenGroupResetSize = static_cast<int64_t>(expertPerRank_) * blockAivNum * INT_CACHELINE;
        totalFlagInt32 = (static_cast<int64_t>(totalFlagInt32) > tokenGroupResetSize)
                         ? static_cast<int64_t>(totalFlagInt32) : tokenGroupResetSize;
    }
    uint32_t resetNumPerCore = Ops::Base::CeilDiv(totalFlagInt32, static_cast<uint64_t>(blockAivNum));
    uint32_t resetTensorSize = Ops::Base::CeilAlign(static_cast<uint64_t>(resetNumPerCore),
        static_cast<uint64_t>(INT32_PER_256B)) * sizeof(int32_t);
    resetTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, resetTensorAddr, resetTensorSize / sizeof(int32_t));
    Duplicate<int32_t>(resetTensor_, 0, (resetTensorSize / sizeof(int32_t)));
    uint32_t mxTempTensorAddr = resetTensorAddr + resetTensorSize;
    uint32_t mxTempTensorSize = 2 * 1024;
    mxTempTensor_ = LocalTensor<uint16_t>(TPosition::VECCALC, mxTempTensorAddr, mxTempTensorSize / sizeof(uint16_t));
    uint32_t xOutTensorAddr1 = mxTempTensorAddr + mxTempTensorSize;
    uint32_t xOutTokenBytes = Ops::Base::CeilAlign(static_cast<uint32_t>(k_ / A_ELEMS_PER_BYTE),
        static_cast<uint32_t>(ALIGN_256));
    uint32_t xOutTensorSize1 = xOutTokenBytes +
        Ops::Base::CeilDiv(k_, static_cast<uint32_t>(ALIGN_32));
    xOutTensor1_ = LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr1,
        xOutTensorSize1 / sizeof(ActivationType));
    uint32_t xOutTensorAddr2 = xOutTensorAddr1 + xOutTensorSize1;
    xOutTensor2_ = LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr2,
        xOutTensorSize1 / sizeof(ActivationType));
    uint32_t xInAlignAddr1 = xOutTensorAddr2 + xOutTensorSize1;
    uint32_t xInAlignSize1 = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_128)) * sizeof(bfloat16_t);
    xInTensor1_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr1, xInAlignSize1 / sizeof(bfloat16_t));
    uint32_t xInAlignAddr2 = xInAlignAddr1 + xInAlignSize1;
    xInTensor2_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr2, xInAlignSize1 / sizeof(bfloat16_t));
    uint32_t sendMaskAddr = xInAlignAddr2 + xInAlignSize1;
    for (int32_t index = 0; index < DOUBLE_BUFFER; ++index) {
        sendMaskTensor_[index] = LocalTensor<uint8_t>(TPosition::VECCALC,
            sendMaskAddr + static_cast<uint32_t>(index) * maskSlotSize_, maskSlotSize_);
    }
    uint32_t sendGatherOutAddr = sendMaskAddr + static_cast<uint32_t>(DOUBLE_BUFFER) * maskSlotSize_;
    uint32_t sendGatherOutSize = Ops::Base::CeilAlign(static_cast<int64_t>(compareCount_ * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_256));
    sendGatherOutTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, sendGatherOutAddr,
        sendGatherOutSize / sizeof(int32_t));
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Process()
{
    MegaMoeModules::TileInfo tile;
    tile.tileIdx = GetBlockIdx() / GetTaskRation();
    tile.totalTiles = GetBlockNum();
    tile.aivTileIdx = GetBlockIdx();
    tile.aivTotalTiles = GetBlockNum() * 2;
    tile.subTileIdx = GetSubBlockIdx();

    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);

    SendAndQuantBuffInit();
    TopKGather::SendMaskCal(tile, rankId_, worldSize_, expertPerRank_,
        sendTotalNum_, compareCount_, maskAlignSize_, maskSlotSize_, maskWinOffset_,
        topkIdsTensor_, sendMaskTensor_, sendGatherOutTensor_, params_);
    TopKGather::ResetFlagList(tile, expertPerRank_, dispatchFlagSlotsPerExpert_, aicNum_,
        resetTensor_, swigluToGmm2FlagGm_,
        (CombineQuantMode != COMBINE_NO_QUANT), params_);

    TokenDispatch::QuantProcessInRank<ActivationType, QuantOutType, QuantScaleOutType, QuantMode, ENABLE_A8W4>(
        tile, m_, k_, mxQuantTokenAlignBytes_, mxQuantScaleAlignBytes_,
        mxQuantScaleNumAlignPerToken_, mxQuantTokenScaleAlignBytes_,
        xInTensor1_, xInTensor2_, xOutTensor1_, xOutTensor2_, mxTempTensor_, params_);
    if constexpr(g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
    }
    SyncAll<false>();
    CombineReduce::CrossRankSyncInWorldSize<CombineQuantMode>(tile, params_);

    DispatchBuffInit();
    GMMAddrInfo dispatchAddrInfo;
    GMMAddrInfo gmm1AddrInfo;
    MegaMoeModules::TupleShape initShape;
    Get<N_VALUE>(initShape) = hiddenDim_;
    Get<K_VALUE>(initShape) = k_;
    MegaMoeModules::BlockOffset initOffset{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    MegaMoeModules::ExpertLoopState dispatchState{initShape, initOffset, 0};
    MegaMoeModules::ExpertLoopState gmm1State{initShape, initOffset, 0};

    uint64_t curSendCnt = 0;
    uint64_t nextSendCnt = 0;

    if constexpr(g_coreType == AIV) {
        if (tile.subTileIdx == 1) {
            TopKGather::SendCntCal(tile, 0, nextSendCnt, cumsumRevCntInRank_,
                worldSize_, aicNum_, maskSlotSize_, maskAlignSize_,
                gatherMaskTensor_, gatherMaskInt32Tensor_, expertTokenCntTensor_,
                cumsumInfoTensor_, expertRevNumsGlobalTensor_, cumsumInfoGlobalTensor_,
                ENABLE_A8W4, params_);
            if (gmm1Module_.UpdateGroupParams<AddrUpdateMode::kGmm1>(
                dispatchState, 0, nextSendCnt, tile, aicNum_, params_, expertRevNumsGlobalTensor_)) {
                gmm1Module_.UpdateGlobalBuffer<AddrUpdateMode::kGmm1>(
                    dispatchAddrInfo, dispatchState, params_, epilogueOp_, dispatchFlagSlotsPerExpert_);
                TokenDispatch::TripleInfoCalAndDispatch<ActivationType, QuantOutType, QuantScaleOutType,
                    QuantMode, ENABLE_A8W4>(
                    dispatchAddrInfo, 0, blockNumPerRank_, tile, worldSize_, expertPerRank_,
                    maxOutputSize_, sendTotalNum_, topK_, k_, maskSlotSize_,
                    mxQuantTokenAlignBytes_, mxQuantScaleAlignBytes_, quantWinOffset_,
                    ubBufferUsedAddr_, cumsumInfoTensor_, gatherMaskInt32Tensor_,
                    validTopkIndexTensor_, topkIndexTensor_, copyTmpTensors_,
                    tripleGlobalTensor_, params_);
            }
        }
    }

    for (int localExpertId = 0; localExpertId < expertPerRank_; localExpertId++) {
        curSendCnt = nextSendCnt;

        if constexpr(g_coreType == AIV) {
            if (tile.subTileIdx == 1 && localExpertId + 1 < expertPerRank_) {
                TopKGather::SendCntCal(tile, localExpertId + 1, nextSendCnt, cumsumRevCntInRank_,
                    worldSize_, aicNum_, maskSlotSize_, maskAlignSize_,
                    gatherMaskTensor_, gatherMaskInt32Tensor_, expertTokenCntTensor_,
                    cumsumInfoTensor_, expertRevNumsGlobalTensor_, cumsumInfoGlobalTensor_,
                    ENABLE_A8W4, params_);
                if (gmm1Module_.UpdateGroupParams<AddrUpdateMode::kGmm1>(
                    dispatchState, localExpertId + 1, nextSendCnt, tile, aicNum_, params_, expertRevNumsGlobalTensor_)) {
                    gmm1Module_.UpdateGlobalBuffer<AddrUpdateMode::kGmm1>(
                        dispatchAddrInfo, dispatchState, params_, epilogueOp_, dispatchFlagSlotsPerExpert_);
                    TokenDispatch::TripleInfoCalAndDispatch<ActivationType, QuantOutType, QuantScaleOutType,
                        QuantMode, ENABLE_A8W4>(
                        dispatchAddrInfo, localExpertId + 1, blockNumPerRank_, tile, worldSize_, expertPerRank_,
                        maxOutputSize_, sendTotalNum_, topK_, k_, maskSlotSize_,
                        mxQuantTokenAlignBytes_, mxQuantScaleAlignBytes_, quantWinOffset_,
                        ubBufferUsedAddr_, cumsumInfoTensor_, gatherMaskInt32Tensor_,
                        validTopkIndexTensor_, topkIndexTensor_, copyTmpTensors_,
                        tripleGlobalTensor_, params_);
                }
            }
        }

        if (!gmm1Module_.UpdateGroupParams<AddrUpdateMode::kGmm1>(
            gmm1State, localExpertId, curSendCnt, tile, aicNum_, params_, expertRevNumsGlobalTensor_)) {
            continue;
        }
        gmm1Module_.UpdateGlobalBuffer<AddrUpdateMode::kGmm1>(
            gmm1AddrInfo, gmm1State, params_, epilogueOp_, dispatchFlagSlotsPerExpert_);
        gmm1Module_.GroupMatmulWithSwigluQuant(
            epilogueOp_, params_, gmm1State.problemShape, gmm1AddrInfo, startBlockIdx_, vecSetSyncCom_);
    }
    EndSync(vecSetSyncCom_);
    if constexpr(g_coreType == AIV) {
        if (tile.subTileIdx == 1) {
            TokenDispatch::ExpertTokenNumCopyOut<ActivationType, QuantOutType, QuantScaleOutType,
                QuantMode, ENABLE_A8W4>(
                worldSize_, expertPerRank_, cumsumInfoTensor_, cumsumInfoGlobalTensor_,
                expertTokenNumsOutTensor_, expertTokenNumsOut_);
        }
    }

    vecSetSyncCom_ = 0;
    GMMAddrInfo gmm2AddrInfo;
    MegaMoeModules::ExpertLoopState gmm2State{initShape, initOffset, 0};
    gmm2Module_.InitCombineBuffers(k_, combineUbTensorSize_);

    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
        if (!gmm2Module_.UpdateGroupParams(gmm2State, expertIdx, tile, params_, expertRevNumsGlobalTensor_)) {
            continue;
        }
        gmm2Module_.UpdateGlobalBuffer(gmm2AddrInfo, gmm2State, params_, dispatchFlagSlotsPerExpert_);
        gmm2Module_.GroupMatmulWithCombine(gmm2AddrInfo, gmm2State, expertIdx, tile, params_,
            startBlockIdx_, vecSetSyncCom_, gmm2PingPongIdx_, rankId_, k_, combineUbTensorSize_);
    }
    if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        EndGMM2Sync(vecSetSyncCom_, gmm2PingPongIdx_);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();

    if constexpr(g_coreType == AIV) {
        CombineReduce::BuffInit<CombineQuantMode, TopkWeightsType>(combineReduceBuf_, params_);
        CombineReduce::CrossRankSyncInWorldSize<CombineQuantMode>(tile, params_);
        CombineReduce::Unpermute<CombineQuantMode>(tile, combineReduceBuf_, params_);
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

}   // namespace MegaMoeImpl
#endif
