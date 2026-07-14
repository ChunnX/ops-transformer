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
#include "mega_moe_single_core.h"

using namespace AscendC;

namespace MegaMoeImpl {
using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
using BlockOffset = Shape<int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t, int64_t, int64_t>;

// 预留：XType OutputType TopkWeightsType Weight1Type
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
    struct ExpertLoopState {
        TupleShape problemShape;
        BlockOffset baseOffset;
        // Rows before the current expert, kept per cursor for dispatch/GMM prefetch state split.
        uint32_t expertBeforeCnt = 0;
    };
    __aicore__ inline MegaMoe() {};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights,
        GM_ADDR weight1, GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
        GM_ADDR scales, GM_ADDR yOut, GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM,
        MegaMoeTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SendAndQuantBuffInit(const AivThreadContext &thread);
    __aicore__ inline bool UpdateGmm2GroupParams(const BlockWorkspaceContext &countWorkspace,
        ExpertLoopState &state, uint32_t expertIdx);
    __aicore__ inline void UpdateGmm2GlobalBuffer(
        GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state);
    __aicore__ inline void Unpermute();
    __aicore__ inline int32_t UnpermuteBuffInit(int32_t coreLen);
    __aicore__ inline void UnpermuteLoadWeights(int32_t coreOffset, int32_t batchStart,
        int32_t curTokens, LocalTensor<bfloat16_t>& tempLocal);
    __aicore__ inline void UnpermuteProcessToken(int32_t tokenIdx, int32_t localIdx,
        const GlobalTensor<bfloat16_t>& expandedX);
    __aicore__ inline void InitCombineBuffers();
    __aicore__ inline void ProcessCombine(const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &gmm2State,
        uint32_t expertIdx);
    __aicore__ inline void CrossRankSyncInWorldSize();
    __aicore__ inline void ExpertTokenNumCopyOut();
    __aicore__ inline void GroupMatmulWithCombine(const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state,
        uint32_t expertIdx);

    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    Params params_{};
    DispatchPrepareContext dispatchPrepareContext_{};
    DispatchGmm1SwigluContext gmm1PipelineContext_{};
    SendMaskArgs sendMaskArgs_{};
    ResetWorkspaceArgs resetWorkspaceArgs_{};
    QuantProcessArgs quantProcessArgs_{};

    GlobalTensor<int32_t> expertTokenNumsOut_;
    GlobalTensor<int32_t> expertRevNumsGlobalTensor_;

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
    uint32_t gmm1StartBlockIdx_ = 0;
    uint32_t gmm2StartBlockIdx_ = 0;
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
    uint16_t gmm2PingPongIdx_ = 0;
    uint64_t sendTotalNum_ = 0;
    uint32_t maskAlignSize_ = 0;
    uint32_t maskSlotSize_ = 0;    // 单个 win 槽位 = maskAlignSize_(mask) + 32B(count)
    uint64_t maskWinOffset_ = 0;   // maskRecvPtr 相对 win 基址(rankSyncInWorldPtr)的偏移
    uint64_t quantWinOffset_ = 0;  // quantTokenScalePtr 相对 win 基址的偏移
    int32_t compareCount_ = 0;
    int64_t combineUbTensorSize_ = 0; // combineUbTensor 的大小（元素数）

    static constexpr uint32_t A_ELEMS_PER_BYTE = Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value ? 2U : 1U;
    static constexpr uint32_t B_ELEMS_PER_BYTE = Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value ? 2U : 1U;
    // ENABLE_A8W4: A8W8 路径（fp8 act + fp4 w1），GMM1 使用 A8W4 prologue（W4→W8 + MMAD）。
    static constexpr bool ENABLE_A8W4 = Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value &&
                                            Std::IsSame<QuantOutType, fp8_e4m3fn_t>::value;
    // ENABLE_A4W4: A4W4 路径（fp4 act + fp4 weight），GMM2 复用 A8W4 prologue。
    //             a4w4 场景下 GMM1 走 generic a4w4、GMM2 走 a8w4，避免两段都用 a4w4 导致精度损失过大。
    static constexpr bool ENABLE_A4W4 = Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value &&
                                            Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value;
    static constexpr int32_t UNPERMUTE_BATCH_LEN = 1024;  // k∈[1,8192], topK∈[2,8], 每 token topK×6B
    QuantProcessScratch<ActivationType> quantScratch_{};
    SendMaskScratch sendMaskScratch_{};
    LocalTensor<int32_t> resetTensor_;
    LocalTensor<bfloat16_t> dataResTensor_;
    LocalTensor<float> dataResFp32Tensor_;
    LocalTensor<float> topKWeightsTensor_;
    LocalTensor<float> fp32ScaleTensor_;
    LocalTensor<bfloat16_t> bf16ScaleTensor_;
    LocalTensor<bfloat16_t> topKWeightsBf16Tensor_;        // Unpermute bf16 weight 搬运中转

    // GMM2 走 A8W4 且 QuantMode 为 a4w4（E2M1）时，SwigluQuant 输出需提升为 fp8_e4m3fn_t。
    // 同时当 Weight2 非 fp4 但 QuantMode==E2M1 时（generic GMM2 路径），也需 promotion，
    // 否则会出现 A=QuantOutType(fp4) vs B=Weight1Type(fp8) 的类型不匹配。
    using SwigluQuantOutType = typename std::conditional<
        (QuantMode == E2M1_QUANT),
        fp8_e4m3fn_t, QuantOutType>::type;

    // SwigluQuant 输出的元素字节密度：fp4 时为 2elem/B，fp8 时为 1elem/B。
    static constexpr uint32_t C_ELEMS_PER_BYTE = Std::IsSame<SwigluQuantOutType, fp4x2_e2m1_t>::value ? 2U : 1U;

    using BlockEpilogue = BlockEpilogueSwigluMxQuant<SwigluQuantOutType, bfloat16_t,
        QuantScaleOutType, QuantScaleOutType, true>;
    BlockEpilogue epilogueOp_;
    DispatchGmm1SwigluArgs<BlockEpilogue> gmm1PipelineArgs_{};
    DispatchGmm1SwigluScratch<ActivationType> gmm1PipelineScratch_{};
};

// ========================
// Init：初始化 & 偏移计算
// ========================
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
    gmm1StartBlockIdx_ = 0;
    gmm2StartBlockIdx_ = 0;
    vecSetSyncCom_ = 0;
    gmm2PingPongIdx_ = 0;
    // 与 WorkspaceInfo 构造里 flagDispatchToGmm1Ptr 的分配公式保持一致。
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
    epilogueOp_.Init({params_.workspaceInfo.swigluQuantDataPtr, params_.workspaceInfo.swigluQuantScalePtr,
        params_.workspaceInfo.flagSwiGluToGmm2Ptr, nullptr, nullptr, nullptr, ALIGN_256, ALIGN_256,
        tilingData->clampLimit});
    // 各 win 区相对 win 基址(rankSyncInWorldPtr)的偏移; 所有卡 win 布局一致, 跨卡读写用同一偏移。
    maskWinOffset_ = static_cast<uint64_t>(params_.peermemInfo.maskRecvPtr -
        params_.peermemInfo.rankSyncInWorldPtr);
    quantWinOffset_ = static_cast<uint64_t>(params_.peermemInfo.quantTokenScalePtr -
        params_.peermemInfo.rankSyncInWorldPtr);
    // maskAlignSize_ 必与 PeermemInfo 中 maskAlignSize 公式数值一致。
    compareCount_ = Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_256)) / sizeof(int32_t);
    maskAlignSize_ = Ops::Base::CeilAlign(static_cast<int64_t>(compareCount_) / 8, static_cast<int64_t>(ALIGN_32));
    // 每个 win 槽位再追加 32B 存 count(源卡 SendMaskCalOneCore 同步算好), 须与 PeermemInfo 的 maskSlotSize 一致。
    maskSlotSize_ = maskAlignSize_ + static_cast<uint32_t>(ALIGN_32);
    mxQuantScaleNumAlignPerToken_ = Ops::Base::CeilDiv(k_, static_cast<uint32_t>(ALIGN_32));
    mxQuantTokenAlignBytes_ = Ops::Base::CeilAlign(static_cast<uint32_t>(k_ / A_ELEMS_PER_BYTE),
        static_cast<uint32_t>(ALIGN_256)) * sizeof(ActivationType);
    mxQuantScaleAlignBytes_ = mxQuantScaleNumAlignPerToken_ * sizeof(uint8_t);
    mxQuantTokenScaleAlignBytes_ = Ops::Base::CeilAlign(mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_,
        static_cast<uint32_t>(ALIGN_32));

    dispatchPrepareContext_ = {{aivCoreIdx_, blockAivNum_},
        {rankId_, worldSize_, expertPerRank_},
        {static_cast<int32_t>(m_), topK_, k_}};
    // MegaMoe keeps its original physical mapping. Reusers may provide another
    // zero-based logical threadBlock mapping to the standalone function.
    gmm1PipelineContext_ = {
        {blockIdx_, blockNum_},
        {blockIdx_, aicNum_},
        worldSize_, expertPerRank_, k_, static_cast<uint32_t>(hiddenDim_), topK_, blockNumPerRank_,
        maxOutputSize_, sendTotalNum_, maskAlignSize_, maskSlotSize_, quantWinOffset_,
        mxQuantTokenAlignBytes_, mxQuantScaleAlignBytes_, dispatchFlagSlotsPerExpert_,
        tilingData->groupedMatmulMode};
    gmm1PipelineArgs_ = {
        &params_, &epilogueOp_, ::winRankAddr_, params_.peermemInfo.maskRecvPtr,
        params_.workspaceInfo.expertRevTokenNumsPtr, params_.workspaceInfo.tripleInfoPtr,
        params_.workspaceInfo.cumsumInfoPtr, params_.workspaceInfo.dispatchRevDataPtr,
        params_.workspaceInfo.dispatchRevScalePtr, params_.workspaceInfo.gmm1MmadResPtr,
        params_.bGmAddr, params_.bScaleGmAddr, params_.workspaceInfo.flagSwiGluToGmm2Ptr,
        params_.workspaceInfo.flagDispatchToGmm1Ptr,
        params_.workspaceInfo.flagSendCntCalToUpdParamsPtr,
        &gmm1StartBlockIdx_, &vecSetSyncCom_};
    sendMaskArgs_ = {
        params_.expertIdxGmAddr, ::winRankAddr_, compareCount_, maskAlignSize_, maskSlotSize_, maskWinOffset_};
    int32_t resetFlagNum = static_cast<int32_t>(expertPerRank_) *
        (static_cast<int32_t>(INT_CACHELINE) + dispatchFlagSlotsPerExpert_ +
        static_cast<int32_t>(INT_CACHELINE) * static_cast<int32_t>(aicNum_));
    int32_t gmm2CombineSyncCounterNum = static_cast<int32_t>(
        static_cast<int64_t>(expertPerRank_) * blockAivNum_ * INT_CACHELINE);
    resetWorkspaceArgs_ = {params_.workspaceInfo.flagSwiGluToGmm2Ptr,
        params_.workspaceInfo.gmm2CombineSyncCounterPtr, resetFlagNum, gmm2CombineSyncCounterNum,
        CombineQuantMode != COMBINE_NO_QUANT};
    quantProcessArgs_ = {params_.aGmAddr, params_.peermemInfo.quantTokenScalePtr,
        mxQuantTokenAlignBytes_, mxQuantScaleAlignBytes_, mxQuantTokenScaleAlignBytes_,
        mxQuantScaleNumAlignPerToken_};
}

// ======================================================================================
// SendAndQuantBuffInit：SendMaskCalOneCore & ResetFlagListOneCore & QuantProcessInRankOneCore localTensor申请
// ======================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::SendAndQuantBuffInit(const AivThreadContext &thread)
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    // Tensor用处：SendMaskCalOneCore 函数中搬运本卡的 topkIds；
    // Tensor大小：该Tensor在进行CompareScalar时，compareCount_要求256B对齐，申请大小256字节对齐；
    uint32_t topkIdsTensorAddr = 0;
    uint32_t topkIdsTensorSize = Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_256));
    sendMaskScratch_.topkIdsTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, topkIdsTensorAddr, topkIdsTensorSize / sizeof(int32_t));
    // Tensor用处：ResetFlagList中对于workSpace上的Flag位置区域进行清理；
    // Tensor大小：大小为清理区域大小均分到所有参与单核模块调用的逻辑线程；
    // SwiGluToGmm2区 DispatchToGmm1区 SendCntCalToUpdParams区，三段在workSpace上连续；
    uint32_t resetTensorAddr = topkIdsTensorAddr + topkIdsTensorSize;
    uint64_t totalFlagInt32 = static_cast<uint64_t>(expertPerRank_) *
        (static_cast<uint64_t>(INT_CACHELINE) + static_cast<uint64_t>(dispatchFlagSlotsPerExpert_) +
        static_cast<uint64_t>(INT_CACHELINE) * static_cast<uint64_t>(aicNum_));
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        int64_t tokenGroupResetSize = static_cast<int64_t>(expertPerRank_) * blockAivNum_ * INT_CACHELINE;
        totalFlagInt32 = (static_cast<int64_t>(totalFlagInt32) > tokenGroupResetSize)
                         ? static_cast<int64_t>(totalFlagInt32) : tokenGroupResetSize;
    }
    uint32_t resetNumPerThread = Ops::Base::CeilDiv(totalFlagInt32, static_cast<uint64_t>(thread.threadNum));
    uint32_t resetTensorSize = Ops::Base::CeilAlign(static_cast<uint64_t>(resetNumPerThread),
        static_cast<uint64_t>(INT32_PER_256B)) * sizeof(int32_t);
    resetTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, resetTensorAddr, resetTensorSize / sizeof(int32_t));
    Duplicate<int32_t>(resetTensor_, 0, (resetTensorSize / sizeof(int32_t)));
    // Tensor用处：QuantProcessInRank函数中用于量化计算中间区域；
    // Tensor大小：当前申请2K；
    uint32_t mxTempTensorAddr = resetTensorAddr + resetTensorSize;
    uint32_t mxTempTensorSize = 2 * 1024;
    quantScratch_.mxTempTensor =
        LocalTensor<uint16_t>(TPosition::VECCALC, mxTempTensorAddr, mxTempTensorSize / sizeof(uint16_t));
    // Tensor用处：QuantProcessInRank函数中用于存储量化输出结果；
    // Tensor大小：token长度Align256字节对齐 + scale存储长度，xOutTensor0与xOutTensor1为双buffer；
    uint32_t xOutTensorAddr1 = mxTempTensorAddr + mxTempTensorSize;
    uint32_t xOutTokenBytes = Ops::Base::CeilAlign(static_cast<uint32_t>(k_ / A_ELEMS_PER_BYTE),
        static_cast<uint32_t>(ALIGN_256));
    uint32_t xOutTensorSize1 = xOutTokenBytes +
        Ops::Base::CeilDiv(k_, static_cast<uint32_t>(ALIGN_32));
    quantScratch_.xOutTensor0 = LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr1,
        xOutTensorSize1 / sizeof(ActivationType));
    uint32_t xOutTensorAddr2 = xOutTensorAddr1 + xOutTensorSize1;
    uint32_t xOutTensorSize2 = xOutTensorSize1;
    quantScratch_.xOutTensor1 = LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr2,
        xOutTensorSize2 / sizeof(ActivationType));
    // Tensor用处：QuantProcessInRank函数中用于存储输入token；
    // Tensor大小：token长度Align256字节对齐，xInTensor0与xInTensor1为双buffer；
    uint32_t xInAlignAddr1 = xOutTensorAddr2 + xOutTensorSize2;
    uint32_t xInAlignSize1 = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_128)) * sizeof(bfloat16_t);
    quantScratch_.xInTensor0 =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr1, xInAlignSize1 / sizeof(bfloat16_t));
    uint32_t xInAlignAddr2 = xInAlignAddr1 + xInAlignSize1;
    uint32_t xInAlignSize2 = xInAlignSize1;
    quantScratch_.xInTensor1 =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr2, xInAlignSize2 / sizeof(bfloat16_t));
    // Tensor用处：SendMaskCalOneCore函数中用于存储mask位；
    // Tensor大小：大小maskSlotSize_与保持一致，DOUBLE_BUFFER为2，开启双buffer；
    uint32_t sendMaskAddr = xInAlignAddr2 + xInAlignSize2;
    for (int32_t index = 0; index < DOUBLE_BUFFER; ++index) {
        sendMaskScratch_.sendMaskTensor[index] = LocalTensor<uint8_t>(TPosition::VECCALC,
            sendMaskAddr + static_cast<uint32_t>(index) * maskSlotSize_, maskSlotSize_);
    }
    // Tensor用处：SendMaskCalOneCore函数中用于GatherMask的dstTensor；
    // Tensor大小：大小为一次GatherMaksk长度compareCount_对齐到256；
    // SendMaskCalOneCore GatherMask 计 count 用的废弃输出 scratch(compareCount_ 个 int, 256B 对齐)。
    uint32_t sendGatherOutAddr = sendMaskAddr + static_cast<uint32_t>(DOUBLE_BUFFER) * maskSlotSize_;
    uint32_t sendGatherOutSize = Ops::Base::CeilAlign(static_cast<int64_t>(compareCount_ * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_256));
    sendMaskScratch_.sendGatherOutTensor = LocalTensor<int32_t>(TPosition::VECCALC, sendGatherOutAddr,
        sendGatherOutSize / sizeof(int32_t));
}

// ==================================================
// ExpertTokenNumCopyOut：本卡各专家收到的token总数输出
// ==================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ExpertTokenNumCopyOut()
{
    // A8W4 路径下 cumsum 被 SwigluQuant 覆盖，从 GM 恢复
    if constexpr (ENABLE_A8W4) {
        DataCopyPad(gmm1PipelineScratch_.cumsumInfoTensor,
                gmm1PipelineScratch_.cumsumInfoGlobalTensor,
                {1U, static_cast<uint32_t>(worldSize_ * expertPerRank_ * sizeof(int32_t)), 0U, 0U, 0U},
                {true, 0U, 0U, 0U});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    }
    int32_t lastRankIdx = static_cast<int32_t>(worldSize_ - 1);
    gmm1PipelineScratch_.expertTokenNumsOutTensor.SetValue(
        0, gmm1PipelineScratch_.cumsumInfoTensor.GetValue(lastRankIdx));
    for (int32_t expertIdx = 1; expertIdx < expertPerRank_; expertIdx++) {
        int32_t cur = gmm1PipelineScratch_.cumsumInfoTensor.GetValue(
            expertIdx * static_cast<int32_t>(worldSize_) + lastRankIdx);
        int32_t prev = gmm1PipelineScratch_.cumsumInfoTensor.GetValue(
            (expertIdx - 1) * static_cast<int32_t>(worldSize_) + lastRankIdx);
        gmm1PipelineScratch_.expertTokenNumsOutTensor.SetValue(expertIdx, cur - prev);
    }
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    DataCopyExtParams copyParams{1U, static_cast<uint32_t>(expertPerRank_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(expertTokenNumsOut_, gmm1PipelineScratch_.expertTokenNumsOutTensor, copyParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
}

// =====================================================================================================
// UpdateGroupParams：更新 GMM2 当前 expert 的 problemShape 与地址偏移。
// =====================================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline bool MegaMoe<TemplateMegaMoeTypeFunc>::UpdateGmm2GroupParams(
    const BlockWorkspaceContext &countWorkspace, ExpertLoopState &state, uint32_t expertIdx)
{
    if (expertIdx != 0) {
        uint64_t m = Get<M_VALUE>(state.problemShape);
        uint64_t n = Get<N_VALUE>(state.problemShape);
        uint64_t k = Get<K_VALUE>(state.problemShape);
        state.expertBeforeCnt += m;
        Get<IDX_A_OFFSET>(state.baseOffset) += m * k / A_ELEMS_PER_BYTE;
        Get<IDX_B_OFFSET>(state.baseOffset) += n * k / B_ELEMS_PER_BYTE;
        // only splitM
        auto scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_A_SCALE_OFFSET>(state.baseOffset) += m * scaleK;
        Get<IDX_B_SCALE_OFFSET>(state.baseOffset) += n * scaleK;
        Get<IDX_C_OFFSET>(state.baseOffset) += m * n / SWIGLU_N_HALF / C_ELEMS_PER_BYTE;
        Get<IDX_C_SCALE_OFFSET>(state.baseOffset) +=
            m * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_FLAG_OFFSET>(state.baseOffset) += 1;
        Get<IDX_B2_OFFSET>(state.baseOffset) += k * n / SWIGLU_N_HALF / B_ELEMS_PER_BYTE;
        Get<IDX_B2_SCALE_OFFSET>(state.baseOffset) +=
            k * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_Y2_OFFSET>(state.baseOffset) += m * k;
        Get<IDX_M_OFFSET>(state.baseOffset) += m;
        Get<IDX_GMM1_OFFSET>(state.baseOffset) += m * n;
        Get<IDX_GMM2_OFFSET>(state.baseOffset) += m * k;
    }

    uint64_t offsetInCnt = expertIdx * INT32_PER_256B * countWorkspace.blockNum +
        INT32_PER_256B * countWorkspace.blockIdx;
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
        expertRevNumsGlobalTensor_[offsetInCnt]);
    Get<M_VALUE>(state.problemShape) = expertRevNumsGlobalTensor_.GetValue(offsetInCnt);
    return Get<M_VALUE>(state.problemShape) != 0;
}

// ==================================================================================
// UpdateGmm2GlobalBuffer：更新当前 expert 的 GMM2 地址视图。
// ==================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::UpdateGmm2GlobalBuffer(
    GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state)
{
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4 || CombineQuantMode != COMBINE_NO_QUANT) {
        gmmAddrInfo.gmm2OutGlobal =
            params_.workspaceInfo.gmm2MmadResPtr + Get<IDX_GMM2_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
    }
    gmmAddrInfo.aGlobal =
        params_.workspaceInfo.swigluQuantDataPtr + Get<IDX_C_OFFSET>(state.baseOffset) * sizeof(ActivationType);
    gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.swigluQuantScalePtr +
        Get<IDX_C_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);
    gmmAddrInfo.bGlobal = params_.b2GmAddr + Get<IDX_B2_OFFSET>(state.baseOffset) * sizeof(ActivationType);
    gmmAddrInfo.bScaleGlobal =
        params_.b2ScaleGmAddr + Get<IDX_B2_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);
    gmmAddrInfo.swigluToGmm2Flag = (__gm__ int32_t*)params_.workspaceInfo.flagSwiGluToGmm2Ptr +
        Get<IDX_FLAG_OFFSET>(state.baseOffset) * INT_CACHELINE;
    gmmAddrInfo.dispatchToGmm1Flag = (__gm__ int32_t*)params_.workspaceInfo.flagDispatchToGmm1Ptr +
        Get<IDX_FLAG_OFFSET>(state.baseOffset) * dispatchFlagSlotsPerExpert_;
}

// =============================================
// InitCombineBuffers：初始化 Combine 所需的 buffer 大小
// =============================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitCombineBuffers()
{
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV) {
        uint32_t nAlign32 = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_32));
        uint32_t nScale = Ops::Base::CeilDiv(k_, uint32_t(MXFP_SCALE_GROUP_NUM));
        uint32_t quantTokenSizeBytes = Ops::Base::CeilAlign(k_ + nScale, static_cast<uint32_t>(ALIGN_32));
        uint32_t singleTokenBytes = nAlign32 * sizeof(bfloat16_t) + quantTokenSizeBytes;
        combineUbTensorSize_ = (singleTokenBytes * 2) / sizeof(bfloat16_t);
    }
}

// =============================================
// ProcessCombine：generic combine-quant 路径的 AIV 后处理。
//                 等待本 expert 的 row-group 计数满足后，读取 triple 和 GMM2 输出，
//                 再执行 row-group 级 CombineRowGroup。
// =============================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessCombine(
    const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &gmm2State, uint32_t expertIdx)
{
    uint32_t nTilesPerGroup = Ops::Base::CeilDiv(k_, L1_TILE_N);

    GlobalTensor<int32_t> gmm2CombineSyncCounter;
    gmm2CombineSyncCounter.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.gmm2CombineSyncCounterPtr);

    uint32_t nScale = Ops::Base::CeilDiv(k_, uint32_t(MXFP_SCALE_GROUP_NUM));
    uint32_t quantTokenSizeBytes = Ops::Base::CeilAlign(k_ + nScale, static_cast<uint32_t>(ALIGN_32));

    uint32_t mExpert = Get<M_VALUE>(gmm2State.problemShape);
    uint32_t tokenGroupsThisExpert = Ops::Base::CeilDiv(mExpert, L1_TILE_M_256);

    uint32_t coreIdForGrouping = aivCoreIdx_;
    uint32_t totalCoresForGrouping = blockAivNum_;
    if constexpr (ENABLE_A8W4) {
        if (subBlockIdx_ != 1) {
            return;
        }
        coreIdForGrouping = aivCoreIdx_ / 2;
        totalCoresForGrouping = blockAivNum_ / 2;
    }

    uint32_t myGroup, myIdxInGrp, myGrpSize;
    MegaMoeImpl::ComputeCoreGrouping(coreIdForGrouping, tokenGroupsThisExpert, totalCoresForGrouping,
        myGroup, myIdxInGrp, myGrpSize);

    if (myGroup >= tokenGroupsThisExpert) {
        return;
    }

    __gm__ int32_t* myCounterAddr = (__gm__ int32_t*)gmm2CombineSyncCounter.GetPhyAddr()
        + expertIdx * blockAivNum_ * INT_CACHELINE
        + aivCoreIdx_ * INT_CACHELINE;
    while (AscendC::ReadGmByPassDCache(myCounterAddr) != nTilesPerGroup)
    {
        int64_t st = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - st < 100) {
        };
    }
    uint32_t tokenStart = myGroup * L1_TILE_M_256;
    uint32_t tokenCount = (L1_TILE_M_256 < mExpert - tokenStart) ? L1_TILE_M_256 : mExpert - tokenStart;
    uint32_t tokensPerCore = Ops::Base::CeilDiv(tokenCount, myGrpSize);
    int32_t myTokenOffset = myIdxInGrp * tokensPerCore;
    int32_t myTokenCount = 0;
    if (myTokenOffset < (int32_t)tokenCount) {
        myTokenCount = (tokensPerCore < tokenCount - myTokenOffset) ? tokensPerCore : tokenCount - myTokenOffset;
    }
    if (myTokenCount > 0) {
        AscendC::SetCtrlSpr<60, 60>(0);
        int64_t offset = 0;
        LocalTensor<int32_t> tripleTensor = LocalTensor<int32_t>(TPosition::VECIN, offset, myTokenCount * TRIPLE_SIZE);
        offset += myTokenCount * TRIPLE_SIZE * sizeof(int32_t);
        AscendC::GlobalTensor<int32_t> tripleGm;
        tripleGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.tripleInfoPtr
            + (gmm2State.expertBeforeCnt + tokenStart + myTokenOffset) * TRIPLE_SIZE * sizeof(int32_t)));
        AscendC::DataCopy(tripleTensor, tripleGm, myTokenCount * TRIPLE_SIZE);
        PipeBarrier<PIPE_MTE2>();
        MegaMoeCombineImpl::CombineTokenGroup<CombineQuantMode, bfloat16_t>(
            tokenStart + myTokenOffset, myTokenCount, k_, expertIdx, rankId_,
            gmmAddrInfo.gmm2OutGlobal, params_, tripleTensor, combineUbTensorSize_,
            offset, quantTokenSizeBytes);
    }
}

// ===============================================================
// UnpermuteLoadWeights：加载一个 token batch 的权重到 UB
// ===============================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::UnpermuteLoadWeights(
    int32_t coreOffset, int32_t batchStart, int32_t curTokens, LocalTensor<bfloat16_t>& tempLocal)
{
    if constexpr (Std::IsSame<TopkWeightsType, float>::value) {
        GlobalTensor<float> topKWeightsGlobalTensor_;
        topKWeightsGlobalTensor_.SetGlobalBuffer((__gm__ float*)params_.probsGmAddr);
        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(curTokens * topK_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(topKWeightsTensor_, topKWeightsGlobalTensor_[(coreOffset + batchStart) * topK_],
            copyParams, copyPadParams);
        SetFlag<AscendC::HardEvent::MTE2_S>(0);
        WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    }
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        GlobalTensor<bfloat16_t> topkWeightsGlobalTensor;
        topkWeightsGlobalTensor.SetGlobalBuffer((__gm__ bfloat16_t*)params_.probsGmAddr);
        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(curTokens * topK_ * sizeof(bfloat16_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<bfloat16_t> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(tempLocal, topkWeightsGlobalTensor[(coreOffset + batchStart) * topK_],
            copyParams, copyPadParams);
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID2>();
        Cast(topKWeightsTensor_, tempLocal, AscendC::RoundMode::CAST_NONE, curTokens * topK_);
        SetFlag<AscendC::HardEvent::V_S>(0);
        WaitFlag<AscendC::HardEvent::V_S>(0);
    }
}

// ===============================================================
// UnpermuteProcessToken：单个 token 的 per-expert 累加
// ===============================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::UnpermuteProcessToken(
    int32_t tokenIdx, int32_t localIdx, const GlobalTensor<bfloat16_t>& expandedX)
{
    LocalTensor<bfloat16_t> dataIn0Bf16 = dataResTensor_[k_];
    LocalTensor<bfloat16_t> dataIn1Bf16 = dataResTensor_[k_ * 2];
    LocalTensor<float> dataIn0Fp32 = dataResFp32Tensor_[k_];
    LocalTensor<float> dataIn1Fp32 = dataResFp32Tensor_[k_ * 2];
    for (int32_t expId = 0; expId < topK_; ++expId) {
        float expScale = topKWeightsTensor_.GetValue(localIdx * topK_ + expId);
        auto event = (expId % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
        auto dataInBf16 = (expId % DOUBLE_BUFFER == 0) ? dataIn0Bf16 : dataIn1Bf16;
        auto dataInFp32 = (expId % DOUBLE_BUFFER == 0) ? dataIn0Fp32 : dataIn1Fp32;
        if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
            WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            DataCopy(dataInBf16, expandedX[(tokenIdx * topK_ + expId) * k_], k_);
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            SetFlag<AscendC::HardEvent::S_V>(event);
            WaitFlag<AscendC::HardEvent::S_V>(event);
            Cast(dataInFp32, dataInBf16, AscendC::RoundMode::CAST_NONE, k_);
        } else {
            uint32_t nScale = Ops::Base::CeilDiv(k_, uint32_t(MXFP_SCALE_GROUP_NUM));
            uint32_t quantTokenSize = k_ + nScale;
            uint32_t quantEleNum = quantTokenSize / sizeof(bfloat16_t);
            WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            DataCopy(dataInBf16, expandedX[(tokenIdx * topK_ + expId) * quantEleNum], quantEleNum);
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            using Fp8Type = typename std::conditional<CombineQuantMode == MXFP8_E4M3_COMM_QUANT,
                fp8_e4m3fn_t, fp8_e5m2_t>::type;
            MegaMoeCombineImpl::DeQuantMxFp8<Fp8Type, bfloat16_t>(dataInBf16, dataInFp32,
                bf16ScaleTensor_, fp32ScaleTensor_, nScale, k_);
        }
        PipeBarrier<PIPE_V>();
        if (expId == 0) {
            Muls(dataResFp32Tensor_, dataInFp32, expScale, k_);
        } else {
            Muls(dataInFp32, dataInFp32, expScale, k_);
            PipeBarrier<PIPE_V>();
            Add(dataResFp32Tensor_, dataResFp32Tensor_, dataInFp32, k_);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<AscendC::HardEvent::V_MTE2>(event);
    }
}

// ===============================================================
// UnpermuteBuffInit：分配 Unpermute 所需固定 buffer，返回 batchLen
// ===============================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline int32_t MegaMoe<TemplateMegaMoeTypeFunc>::UnpermuteBuffInit(int32_t coreLen)
{
    // ── 固定 buffer：dataRes + dataResFp32 (+ scales) ──
    uint32_t dataResBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(UNPERMUTE_LIST_NUM * k_ * sizeof(bfloat16_t)), static_cast<uint32_t>(ALIGN_32));
    uint32_t dataResFp32BufAlign = dataResBufAlign * HALF_TO_FP32;
    // Tensor用处：Unpermute 函数用于存储 mte2 搬入 token；
    // Tensor大小：3 * 单个 token 长度，2 块用于 mte2 搬运 doubleBuffer，1 块存储 Cast 输出结果用于搬出；
    uint32_t dataResAddr = 0;
    dataResTensor_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, dataResAddr,
        dataResBufAlign / sizeof(bfloat16_t));
    // Tensor用处：Unpermute 函数用于存储 token Cast 目的 Tensor；
    // Tensor大小：dataResTensor_ 开设大小 × HALF_TO_FP32；
    uint32_t dataResFp32Addr = dataResAddr + dataResBufAlign;
    dataResFp32Tensor_ = LocalTensor<float>(TPosition::VECCALC, dataResFp32Addr,
        dataResFp32BufAlign / sizeof(float));
    uint32_t tempAddr = dataResFp32Addr + dataResFp32BufAlign;

    // ── batchLen（后续以模板参数形式传入）──
    int32_t batchLen = UNPERMUTE_BATCH_LEN;
    if (batchLen > coreLen) { batchLen = coreLen; }

    // ── weight buffer（在 scale 之前，与 master 顺序一致）──
    // Tensor用处：用于存储 topKWeight；
    // Tensor大小：batchLen × topK_ × sizeof(float) align 到 32 字节对齐；
    uint32_t topKWeightsBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(batchLen * topK_ * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
    topKWeightsTensor_ = LocalTensor<float>(TPosition::VECCALC, tempAddr,
        topKWeightsBufAlign / sizeof(float));
    tempAddr += topKWeightsBufAlign;

    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        // Tensor用处：Unpermute 中 bf16 weight 搬运中转 buffer；
        // Tensor大小：batchLen × topK_ × sizeof(bfloat16_t) align 到 32 字节；
        uint32_t tempBufAlign = Ops::Base::CeilAlign(
            static_cast<uint32_t>(batchLen * topK_ * sizeof(bfloat16_t)), uint32_t(ALIGN_32));
        topKWeightsBf16Tensor_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, tempAddr,
            tempBufAlign / sizeof(bfloat16_t));
        tempAddr += tempBufAlign;
    }

    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        uint32_t scaleNum = Ops::Base::CeilDiv(static_cast<uint32_t>(k_), static_cast<uint32_t>(ALIGN_32));
        // Tensor用处：DeQuantMxFp8 中用于存储 bf16 格式的 scale（e8m0 转换后的中间结果）
        // Tensor大小：scaleNum × sizeof(bfloat16_t) × DOUBLE_BUFFER × HALF_TO_FP32
        uint32_t bf16ScaleBufAlign = Ops::Base::CeilAlign(static_cast<uint32_t>
            (scaleNum * sizeof(bfloat16_t) * DOUBLE_BUFFER * HALF_TO_FP32), static_cast<uint32_t>(ALIGN_32));
        bf16ScaleTensor_ = LocalTensor<bfloat16_t>(
            TPosition::VECCALC, tempAddr, bf16ScaleBufAlign / sizeof(bfloat16_t));
        tempAddr += bf16ScaleBufAlign;
        // Tensor用处：DeQuantMxFp8 中用于存储 fp32 格式的 scale（广播后的最终 scale）
        // Tensor大小：scaleNum × sizeof(float) × DOUBLE_BUFFER × HALF_TO_FP32
        uint32_t fp32ScaleBufAlign = Ops::Base::CeilAlign(static_cast<uint32_t>
            (scaleNum * sizeof(float) * DOUBLE_BUFFER * HALF_TO_FP32), static_cast<uint32_t>(ALIGN_32));
        fp32ScaleTensor_ = LocalTensor<float>(TPosition::VECCALC, tempAddr, fp32ScaleBufAlign / sizeof(float));
        tempAddr += fp32ScaleBufAlign;
    }

    return batchLen;
}

// ===============================================================
// Unpermute：主入口 — 初始化 buffer → 分批循环处理
// ===============================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Unpermute()
{
    int32_t coreLen, coreOffset;
    TilingByCore(m_, coreLen, coreOffset, 1);
    if (coreLen == 0) { return; }
    int32_t batchLen = UnpermuteBuffInit(coreLen);

    GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer((__gm__ bfloat16_t*)params_.peermemInfo.combineSendPtr);
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer((__gm__ bfloat16_t*)params_.y2GmAddr);

    // ═══════════════════ 外层：token batch ═══════════════════
    for (int32_t batchStart = 0; batchStart < coreLen; batchStart += batchLen) {
        int32_t curTokens = (batchStart + batchLen > coreLen) ? (coreLen - batchStart) : batchLen;

        UnpermuteLoadWeights(coreOffset, batchStart, curTokens, topKWeightsBf16Tensor_);

        // ── 内层：token 循环 ──
        SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        for (int32_t localIdx = 0; localIdx < curTokens; localIdx++) {
            int32_t tokenIdx = coreOffset + batchStart + localIdx;
            SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID2>();
            UnpermuteProcessToken(tokenIdx, localIdx, expandedX);
            Cast(dataResTensor_, dataResFp32Tensor_, AscendC::RoundMode::CAST_RINT, k_);
            SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID3>();
            DataCopy(output[tokenIdx * k_], dataResTensor_, k_);
        }
        WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    }
}

// ==============================================================================================
// CrossRankSyncInWorldSize：全卡同步，rankSyncInWorldPtr前48K用于同步，后面区域用于记录当前syncCnt值
// ==============================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::CrossRankSyncInWorldSize()
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    __gm__ int32_t* syncRank = (__gm__ int32_t*)(params_.peermemInfo.rankSyncInWorldPtr);
    __gm__ int32_t* syncCount = (__gm__ int32_t*)(params_.peermemInfo.rankSyncInWorldPtr +
        48 * 1024 + aivCoreIdx_ * 64);
    int count = ReadGmByPassDCache(syncCount) + 1;
    for (int i = aivCoreIdx_; i < worldSize_; i += blockAivNum_) {
        __gm__ int32_t* syncRemoteAddr = (__gm__ int32_t*)(winRankAddr_[i]) + rankId_ * 16;
        WriteGmByPassDCache(syncRemoteAddr, count);
        auto syncCheck = syncRank + i * 16;
        GmSignalWaitBarrier(syncCheck, count);
    }
    WriteGmByPassDCache(syncCount, count);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

// ===============================================================
// GroupMatmulWithCombine：先按实现路径分发，再按 combine 模式分发。
//                        A8W4/A4W4 走 A8W4 prologue（支持 combine-quant）；
//                        generic 路径同时承载非量化 combine 和 combine-quant 主线实现。
// ===============================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::GroupMatmulWithCombine(
    const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state, uint32_t expertIdx)
{
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        MegaMoeImpl::GroupMatmul2CombineA8W4<CombineQuantMode,
            SwigluQuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
            params_, state.problemShape, gmmAddrInfo, gmm2StartBlockIdx_, vecSetSyncCom_,
            state.expertBeforeCnt, gmm2PingPongIdx_, expertIdx);
    } else {
        // A8W8_NZ / Generic: both use the same GroupMatmul2 template, only LayoutB differs (ZN vs ND).
        if (params_.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ) {
            MegaMoeImpl::GroupMatmul2<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t,
                QuantScaleOutType, QuantScaleOutType, true>(
                params_, state.problemShape, gmmAddrInfo, gmm2StartBlockIdx_, vecSetSyncCom_,
                state.expertBeforeCnt, gmm2PingPongIdx_, expertIdx);
        } else {
            MegaMoeImpl::GroupMatmul2<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t,
                QuantScaleOutType, QuantScaleOutType>(
                params_, state.problemShape, gmmAddrInfo, gmm2StartBlockIdx_, vecSetSyncCom_,
                state.expertBeforeCnt, gmm2PingPongIdx_, expertIdx);
        }
    }
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV) {
        ProcessCombine(gmmAddrInfo, state, expertIdx);
    }
}

// Process：MegaMoe 的完整算子入口。
// 可复用单核模块的 Context/Args 已在 Init 中初始化，UB Scratch 由单核函数按阶段布局。
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Process()
{
    // 1.本卡数据处理
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    SendAndQuantBuffInit(dispatchPrepareContext_.thread);
    SendMaskCalOneCore(
        dispatchPrepareContext_, sendMaskArgs_, sendMaskScratch_); // 源卡按当前逻辑线程负责的全局专家算 mask 并推送
    ResetFlagListOneCore(
        dispatchPrepareContext_, resetWorkspaceArgs_, resetTensor_); // 清理当前逻辑线程负责的 workSpace flag

    QuantProcessInRankOneCore<QuantMode, QuantOutType, ActivationType>(
        dispatchPrepareContext_, quantProcessArgs_, quantScratch_); // 对当前逻辑线程负责的本卡 token 量化
    if constexpr(g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
    }
    SyncAll<false>();            // aic需要等待flag位reset清理完成
    CrossRankSyncInWorldSize();  // 全卡同步

    // 2.本卡专家接收数据dispatch & GroupMatmul1 & SwigluQuant
    DispatchGmm1SwigluOneCore<QuantOutType, ActivationType, Weight1Type,
        SwigluQuantOutType, QuantScaleOutType, ENABLE_A8W4>(
            gmm1PipelineContext_, gmm1PipelineArgs_, gmm1PipelineScratch_);
    if constexpr(g_coreType == AIV) {
        if (subBlockIdx_ == 1) {
            ExpertTokenNumCopyOut(); // 本卡专家接受的tokenCnt总数搬出
        }
    }

    // 3. 本卡专家接收数据GroupMatmul2 & Combine
    // Preserve the original cross-stage rotation when the block counts match,
    // and normalize it before GMM2 when GMM1 used another logical block count.
    gmm2StartBlockIdx_ = blockNum_ == 0U ? 0U : gmm1StartBlockIdx_ % blockNum_;
    vecSetSyncCom_ = 0;
    GMMAddrInfo gmm2AddrInfo;
    TupleShape initShape;
    Get<N_VALUE>(initShape) = hiddenDim_;
    Get<K_VALUE>(initShape) = k_;
    BlockOffset initOffset{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    ExpertLoopState gmm2State{initShape, initOffset, 0};
    BlockWorkspaceContext gmm2CountWorkspace{blockIdx_, aicNum_};
    InitCombineBuffers();

    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
        if (!UpdateGmm2GroupParams(gmm2CountWorkspace, gmm2State, expertIdx)) {
            continue;
        }
        UpdateGmm2GlobalBuffer(gmm2AddrInfo, gmm2State);
        GroupMatmulWithCombine(gmm2AddrInfo, gmm2State, expertIdx);
    }
    if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        EndGMM2Sync(vecSetSyncCom_, gmm2PingPongIdx_);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();

    // 4. 本卡数据Unpermute
    if constexpr(g_coreType == AIV) {
        CrossRankSyncInWorldSize(); // 全卡软同步，确认combine send完成
        Unpermute();
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

}   // namespace MegaMoeImpl
#endif
