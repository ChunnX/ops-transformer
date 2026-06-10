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
#include "../../common/op_kernel/mc2_kernel_utils.h"
#include "kernel_operator_list_tensor_intf.h"
#include "mega_moe_base.h"
#include "mega_moe_workspace_info.h"
#include "block_epilogue_swiglu_mx_quant.h"
#include "mega_moe_impl.h"
#include "moe_distribute_dispatch_v2/op_kernel/quantize_functions.h"

using namespace AscendC;

namespace MegaMoeImpl {
using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
using BlockOffset = Shape<int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t>;

// 预留：XType OutputType
#define TemplateMegaMoeTypeClass typename XType, typename OutputType, typename TopkWeightsType, int32_t QuantMode
#define TemplateMegaMoeTypeFunc XType, OutputType, TopkWeightsType, QuantMode

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
    };
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
    __aicore__ inline void ResetFlagList();
    __aicore__ inline void SendMaskCal();
    __aicore__ inline void SendCntCal(int32_t localExpertId);
    __aicore__ inline void TripleInfoCalAndDispatch(GMMAddrInfo &gmmAddrInfo, int32_t localExpertId);
    template <AddrUpdateMode Mode>
    __aicore__ inline bool UpdateGroupParams(ExpertLoopState &state, uint32_t expertIdx);
    template <AddrUpdateMode Mode>
    __aicore__ inline void UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state);
    __aicore__ inline void Unpermute();
    __aicore__ inline void CrossRankSyncInWorldSize();
    __aicore__ inline void ExpertTokenNumCopyOut();
    __aicore__ inline void CopyGMToGMPerToken(int32_t rowDstOffsetInCore, int32_t remoteRankIdx,
        int32_t copyStartIdx, int32_t copyNum);
    __aicore__ inline void QuantProcessInRank();

    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    Params params_{};

    GlobalTensor<int32_t> groupFlagListGm_;
    GlobalTensor<int32_t> expertTokenNumsOut_;
    GlobalTensor<int32_t> tripleGloablTensor_;
    GlobalTensor<int32_t> expertRevNumsGloablTensor_;

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
    uint32_t expertBeforeCnt_ = 0;
    uint32_t ubBufferUsedAddr_ = 0;
    uint16_t gmm2PingPongIdx_ = 0;
    uint64_t sendTotalNum_ = 0;
    uint32_t maskAlignSize_ = 0;
    uint32_t maskSlotSize_ = 0;    // 单个 win 槽位 = maskAlignSize_(mask) + 32B(count)
    uint64_t maskWinOffset_ = 0;   // maskRecvPtr 相对 win 基址(rankSyncInWorldPtr)的偏移
    uint64_t quantWinOffset_ = 0;  // quantTokenScalePtr 相对 win 基址的偏移
    uint64_t totalSendCntInExp_ = 0;
    uint64_t cumsumRevCntInRank_ = 0;
    int32_t compareCount_ = 0;

    static constexpr uint32_t ELEMS_PER_BYTE = (QuantMode == E2M1_QUANT) ? 2 : 1;
    static constexpr int32_t DISPATCH_BUFFER_NUM = 6;
    LocalTensor<int32_t> topkIndexTensor_;
    LocalTensor<uint8_t> gatherMaskTensor_;
    LocalTensor<uint32_t> gatherMaskInt32Tensor_;
    LocalTensor<int32_t> expertTokenCntTensor_;
    LocalTensor<int32_t> validTopkIndexTensor_;
    LocalTensor<int32_t> cumsumInfoTensor_;
    LocalTensor<ActivationType> copyTmpTensors_[DISPATCH_BUFFER_NUM]; // 6-buffer 软流水：占满 EVENT_ID0..EVENT_ID5。
    LocalTensor<int32_t> tripleTensor_;
    LocalTensor<bfloat16_t> xInTensor1_;
    LocalTensor<bfloat16_t> xInTensor2_;
    LocalTensor<ActivationType> xOutTensor1_;
    LocalTensor<ActivationType> xOutTensor2_;
    LocalTensor<uint16_t> mxTempTensor_;
    LocalTensor<int32_t> resetTensor_;
    LocalTensor<int32_t> topkIdsTensor_;
    LocalTensor<uint8_t> sendMaskTensor_[DOUBLE_BUFFER];  // SendMaskCal 源卡算 [mask|count] 的 ping-pong 缓冲
    LocalTensor<int32_t> sendGatherOutTensor_;            // SendMaskCal GatherMask 计 count 的废弃输出 scratch
    LocalTensor<int32_t> expertTokenNumsOutTensor_;
    LocalTensor<bfloat16_t> dataResTensor_;
    LocalTensor<float> dataResFp32Tensor_;
    LocalTensor<float> topKWeightsTensor_;

    using BlockEpilogue = BlockEpilogueSwigluMxQuant<QuantOutType, bfloat16_t,
        QuantScaleOutType, QuantScaleOutType, true>;
    BlockEpilogue epilogueOp_;
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
    params_.peermemInfo = PeermemInfo(winRankAddr_[rankId_], tilingData, ELEMS_PER_BYTE);
    params_.tilingData = tilingData;
    expertTokenNumsOut_.SetGlobalBuffer((__gm__ int32_t*)params_.expertTokenNumsOutGmAddr);
    expertRevNumsGloablTensor_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.expertRevTokenNumsPtr);
    tripleGloablTensor_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.tripleInfoPtr);
    epilogueOp_.Init({params_.workspaceInfo.swigluQuantDataPtr, params_.workspaceInfo.swigluQuantScalePtr,
        params_.workspaceInfo.flagSwiGluToGmm2Ptr, nullptr, nullptr, nullptr, ALIGN_256, ALIGN_256});
    // 各 win 区相对 win 基址(rankSyncInWorldPtr)的偏移; 所有卡 win 布局一致, 跨卡读写用同一偏移。
    maskWinOffset_ = static_cast<uint64_t>(params_.peermemInfo.maskRecvPtr -
        params_.peermemInfo.rankSyncInWorldPtr);
    quantWinOffset_ = static_cast<uint64_t>(params_.peermemInfo.quantTokenScalePtr -
        params_.peermemInfo.rankSyncInWorldPtr);
    // maskAlignSize_ 必与 PeermemInfo 中 maskAlignSize 公式数值一致。
    compareCount_ = Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_256)) / sizeof(int32_t);
    maskAlignSize_ = Ops::Base::CeilAlign(static_cast<int64_t>(compareCount_) / 8, static_cast<int64_t>(ALIGN_32));
    // 每个 win 槽位再追加 32B 存 count(源卡 SendMaskCal 同步算好), 须与 PeermemInfo 的 maskSlotSize 一致。
    maskSlotSize_ = maskAlignSize_ + static_cast<uint32_t>(ALIGN_32);
    mxQuantScaleNumAlignPerToken_ = Ops::Base::CeilDiv(k_, static_cast<uint32_t>(ALIGN_32));
    mxQuantTokenAlignBytes_ = Ops::Base::CeilAlign(static_cast<uint32_t>(k_ / ELEMS_PER_BYTE),
        static_cast<uint32_t>(ALIGN_256)) * sizeof(ActivationType);
    mxQuantScaleAlignBytes_ = mxQuantScaleNumAlignPerToken_ * sizeof(uint8_t);
    mxQuantTokenScaleAlignBytes_ = Ops::Base::CeilAlign(mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_,
        static_cast<uint32_t>(ALIGN_32));
}

// =================================================================================================
// DispatchBuffInit：SendCntCal & TripleInfoCalAndDispatch & ExpertTokenNumCopyOut 中使用的buffer申请
// =================================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::DispatchBuffInit()
{
    if constexpr(g_coreType == AIC) {
        return;
    }

    // Tensor用处：SendCntCal 函数中记录本卡各专家收到的token总数；
    // Tensor大小：仅记录count值，且各专家之间复用，申请大小为32字节；
    uint32_t expertTokenCntTensorAddr = 0;
    uint32_t expertTokenCntTensorSize = ALIGN_32;
    expertTokenCntTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, expertTokenCntTensorAddr,
        expertTokenCntTensorSize / sizeof(int32_t));
    // Tensor用处：SendCntCal 函数中记录本卡专家收到token count的cumsum累加值；
    // Tensor大小：worldSize_ * expertPerRank_ * sizeof(int32_t) align至32字节对齐；
    uint32_t cumsumInfoTensorAddr = expertTokenCntTensorAddr + expertTokenCntTensorSize;
    uint32_t cumsumInfoTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(worldSize_ * expertPerRank_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    cumsumInfoTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, cumsumInfoTensorAddr,
        cumsumInfoTensorSize / sizeof(int32_t));
    // Tensor用处：SendCntCal 函数中用来存储本卡上专家收到的mask+count位的buffer；
    // Tensor大小：maskSlotSize_ * worldSize_，每个maskSlotSize_中包含mask与count；
    uint32_t gatherMaskTensorAddr = cumsumInfoTensorAddr + cumsumInfoTensorSize;
    uint32_t gatherMaskTensorSize = maskSlotSize_ * worldSize_;
    gatherMaskTensor_ = LocalTensor<uint8_t>(TPosition::VECCALC, gatherMaskTensorAddr,
        gatherMaskTensorSize / sizeof(uint8_t));
    gatherMaskInt32Tensor_ = LocalTensor<uint32_t>(TPosition::VECCALC, gatherMaskTensorAddr,
        gatherMaskTensorSize / sizeof(uint32_t));
    // Tensor用处：TripleInfoCalAndDispatch 函数中GatherMask的dst Tensor；
    // Tensor大小：sendTotalNum_ * sizeof(int32_t) align至32字节对齐；
    uint32_t validTopkIndexTensorAddr = gatherMaskTensorAddr + gatherMaskTensorSize;
    uint32_t validTopkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    validTopkIndexTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, validTopkIndexTensorAddr,
        validTopkIndexTensorSize / sizeof(int32_t));
    // Tensor用处：TripleInfoCalAndDispatch 函数中GatherMask的src Tensor；
    // Tensor大小：sendTotalNum_ * sizeof(int32_t) align至32字节对齐；
    uint32_t topkIndexTensorAddr = validTopkIndexTensorAddr + validTopkIndexTensorSize;
    uint32_t topkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    topkIndexTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, topkIndexTensorAddr,
        topkIndexTensorSize / sizeof(int32_t));
    // Tensor用处：TripleInfoCalAndDispatch 函数中的6个dispatch buffer, 配合EVENT_ID0..EVENT_ID5做软流水
    // Tensor大小：每块大小为 TOKEN_SCALE_SIZE=8KB, 容纳token+scale，一共开设48K。
    constexpr uint32_t COPY_TMP_BUFFER_SIZE = TOKEN_SCALE_SIZE;
    uint32_t copyTmpBaseAddr = topkIndexTensorAddr + topkIndexTensorSize;
    for (int32_t index = 0; index < DISPATCH_BUFFER_NUM; ++index) {
        copyTmpTensors_[index] = LocalTensor<ActivationType>(TPosition::VECCALC,
            copyTmpBaseAddr + static_cast<uint32_t>(index) * COPY_TMP_BUFFER_SIZE,
            COPY_TMP_BUFFER_SIZE / sizeof(ActivationType));
    }
    // Tensor用处：ExpertTokenNumCopyOut 函数中本卡各专家收到的tokenCnt数；
    // Tensor大小：expertPerRank_ * sizeof(int32_t) 对齐至32字节；
    uint32_t expertTokenNumsOutTensorAddr = copyTmpBaseAddr +
        static_cast<uint32_t>(DISPATCH_BUFFER_NUM) * COPY_TMP_BUFFER_SIZE;
    uint32_t expertTokenNumsOutTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(expertPerRank_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    expertTokenNumsOutTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, expertTokenNumsOutTensorAddr,
        expertTokenNumsOutTensorSize / sizeof(int32_t));
    // 记录当前已被使用的ub地址，用于后续TripleInfoCalAndDispatch函数中分核后神申请tripleTensor_
    ubBufferUsedAddr_ = expertTokenNumsOutTensorAddr + expertTokenNumsOutTensorSize;
    CreateVecIndex(topkIndexTensor_, 0, (topkIndexTensorSize / sizeof(int32_t)));
    Duplicate<int32_t>(cumsumInfoTensor_, 0, (cumsumInfoTensorSize / sizeof(int32_t)));
    PipeBarrier<PIPE_ALL>();
}

// ======================================================================================
// SendAndQuantBuffInit：SendMaskCal & ResetFlagList & QuantProcessInRank localTensor申请
// ======================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::SendAndQuantBuffInit()
{
    if constexpr(g_coreType == AIC) {
        return;
    }

    // Tensor用处：SendMaskCal 函数中搬运本卡的 topkIds；
    // Tensor大小：该Tensor在进行CompareScalar时，compareCount_要求256B对齐，申请大小256字节对齐；
    uint32_t topkIdsTensorAddr = 0;
    uint32_t topkIdsTensorSize = Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_256));
    topkIdsTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, topkIdsTensorAddr, topkIdsTensorSize / sizeof(int32_t));
    // Tensor用处：ResetFlagList中对于workSpace上的Flag位置区域进行清理；
    // Tensor大小：大小为清理区域大小均分到所有的blockAivNum_；
    // SwiGluToGmm2区(expertPerRank * INT_CACHELINE)+DispatchToGmm1区(expertPerRank * dispatchFlagSlotsPerExpert_)；
    uint32_t resetTensorAddr = topkIdsTensorAddr + topkIdsTensorSize;
    uint64_t totalFlagInt32 = static_cast<uint64_t>(expertPerRank_) *
        (static_cast<uint64_t>(INT_CACHELINE) + static_cast<uint64_t>(dispatchFlagSlotsPerExpert_));
    uint32_t resetNumPerCore = Ops::Base::CeilDiv(totalFlagInt32, static_cast<uint64_t>(blockAivNum_));
    uint32_t resetTensorSize = Ops::Base::CeilAlign(static_cast<uint64_t>(resetNumPerCore),
        static_cast<uint64_t>(INT32_PER_256B)) * sizeof(int32_t);
    resetTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, resetTensorAddr, resetTensorSize / sizeof(int32_t));
    Duplicate<int32_t>(resetTensor_, 0, (resetTensorSize / sizeof(int32_t)));
    // Tensor用处：QuantProcessInRank函数中用于量化计算中间区域；
    // Tensor大小：当前申请2K；
    uint32_t mxTempTensorAddr = resetTensorAddr + resetTensorSize;
    uint32_t mxTempTensorSize = 2 * 1024;
    mxTempTensor_ = LocalTensor<uint16_t>(TPosition::VECCALC, mxTempTensorAddr, mxTempTensorSize / sizeof(uint16_t));
    // Tensor用处：QuantProcessInRank函数中用于存储量化输出结果；
    // Tensor大小：token长度Align256字节对齐 + scale存储长度，当前支持mxfp8量化，xOutTensor1_与xOutTensor2_为双buffer；
    uint32_t xOutTensorAddr1 = mxTempTensorAddr + mxTempTensorSize;
    uint32_t xOutTokenBytes = Ops::Base::CeilAlign(static_cast<uint32_t>(k_ / ELEMS_PER_BYTE),
        static_cast<uint32_t>(ALIGN_256));
    uint32_t xOutTensorSize1 = xOutTokenBytes +
        Ops::Base::CeilDiv(k_, static_cast<uint32_t>(ALIGN_32));
    xOutTensor1_ = LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr1,
        xOutTensorSize1 / sizeof(ActivationType));
    uint32_t xOutTensorAddr2 = xOutTensorAddr1 + xOutTensorSize1;
    uint32_t xOutTensorSize2 = xOutTensorSize1;
    xOutTensor2_ = LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr2,
        xOutTensorSize2 / sizeof(ActivationType));
    // Tensor用处：QuantProcessInRank函数中用于存储输入token；
    // Tensor大小：token长度Align256字节对齐，xInTensor1_与xInTensor2_为双buffer；
    uint32_t xInAlignAddr1 = xOutTensorAddr2 + xOutTensorSize2;
    uint32_t xInAlignSize1 = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_128)) * sizeof(bfloat16_t);
    xInTensor1_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr1, xInAlignSize1 / sizeof(bfloat16_t));
    uint32_t xInAlignAddr2 = xInAlignAddr1 + xInAlignSize1;
    uint32_t xInAlignSize2 = xInAlignSize1;
    xInTensor2_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr2, xInAlignSize2 / sizeof(bfloat16_t));
    // Tensor用处：SendMaskCal函数中用于存储mask位；
    // Tensor大小：大小maskSlotSize_与保持一致，DOUBLE_BUFFER为2，开启双buffer；
    uint32_t sendMaskAddr = xInAlignAddr2 + xInAlignSize2;
    for (int32_t index = 0; index < DOUBLE_BUFFER; ++index) {
        sendMaskTensor_[index] = LocalTensor<uint8_t>(TPosition::VECCALC,
            sendMaskAddr + static_cast<uint32_t>(index) * maskSlotSize_, maskSlotSize_);
    }
    // Tensor用处：SendMaskCal函数中用于GatherMask的dstTensor；
    // Tensor大小：大小为一次GatherMaksk长度compareCount_对齐到256；
    // SendMaskCal GatherMask 计 count 用的废弃输出 scratch(compareCount_ 个 int, 256B 对齐)。
    uint32_t sendGatherOutAddr = sendMaskAddr + static_cast<uint32_t>(DOUBLE_BUFFER) * maskSlotSize_;
    uint32_t sendGatherOutSize = Ops::Base::CeilAlign(static_cast<int64_t>(compareCount_ * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_256));
    sendGatherOutTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, sendGatherOutAddr,
        sendGatherOutSize / sizeof(int32_t));
}

// ===============================================================================================
// ResetFlagList：对本卡workSpace上对于Flag位的清理，包括flagSwiGluToGmm2Ptr & flagDispatchToGmm1Ptr
// ===============================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ResetFlagList()
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    // workSpace groupFlagList 清零（含 dispatch->gmm1 wave-grain 槽位区）。
    // 总数 = SwiGluToGmm2(expertPerRank * INT_CACHELINE) + DispatchToGmm1(expertPerRank * dispatchFlagSlotsPerExpert_)。
    groupFlagListGm_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.flagSwiGluToGmm2Ptr);
    int32_t flagNum = static_cast<int32_t>(expertPerRank_) *
        (static_cast<int32_t>(INT_CACHELINE) + dispatchFlagSlotsPerExpert_);
    int32_t coreLen, coreOffset;
    TilingByCore(flagNum, coreLen, coreOffset, 1);
    DataCopyExtParams rankSyncCopyParams{1U, static_cast<uint32_t>(coreLen * sizeof(int32_t)), 0U, 0U, 0U};
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    if (coreLen != 0) {
        DataCopyPad(groupFlagListGm_[coreOffset], resetTensor_, rankSyncCopyParams);
    }
}

// ==================================================
// ExpertTokenNumCopyOut：本卡各专家收到的token总数输出
// ==================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ExpertTokenNumCopyOut()
{
    int32_t lastRankIdx = static_cast<int32_t>(worldSize_ - 1);
    expertTokenNumsOutTensor_.SetValue(0, cumsumInfoTensor_.GetValue(lastRankIdx));
    for (int32_t expertIdx = 1; expertIdx < expertPerRank_; expertIdx++) {
        int32_t cur = cumsumInfoTensor_.GetValue(expertIdx * static_cast<int32_t>(worldSize_) + lastRankIdx);
        int32_t prev = cumsumInfoTensor_.GetValue((expertIdx - 1) * static_cast<int32_t>(worldSize_) + lastRankIdx);
        expertTokenNumsOutTensor_.SetValue(expertIdx, cur - prev);
    }
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    DataCopyExtParams copyParams{1U, static_cast<uint32_t>(expertPerRank_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(expertTokenNumsOut_, expertTokenNumsOutTensor_, copyParams);
}

// ======================================================================================================
// SendMaskCal：对本卡 topk 按通信域内所有专家id计算mask位，并发送至目标专家卡
// ------------------------------------------------------------------------------------------------------
//   Phase 1: 本卡 topk 的搬入；
//   Phase 2: 根据专家id进行CompareScalar & GatherMask，生成mask位与count总数，doubleBuffer进行计算并进行发送；
// ======================================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::SendMaskCal()
{
    if constexpr(g_coreType == AIC) {
        return;
    }

    // Phase 1: 加载本卡 topk 到 topkIdsTensor_ (compareCount_ 个 int, 尾部补 0)
    GlobalTensor<int32_t> srcGlobalTensor;
    srcGlobalTensor.SetGlobalBuffer((__gm__ int32_t*)params_.expertIdxGmAddr);
    Duplicate<int32_t>(topkIdsTensor_, 0, compareCount_);
    SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID1>();
    DataCopyExtParams loadParams{1U, static_cast<uint32_t>(sendTotalNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> loadPad{false, 0U, 0U, 0U};
    DataCopyPad(topkIdsTensor_, srcGlobalTensor, loadParams, loadPad);
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();

    // Phase 2: 逐个全局专家算 [mask|count] 推送 (ping-pong 双 buffer 软流水)
    // buffer内前 maskAlignSize_ 存mask, 末 32B 存 count
    constexpr TEventID kBufEvents[DOUBLE_BUFFER] = {EVENT_ID0, EVENT_ID1};
    for (int32_t index = 0; index < static_cast<int32_t>(DOUBLE_BUFFER); ++index) {
        SetFlag<AscendC::HardEvent::MTE3_V>(kBufEvents[index]);
    }
    int32_t totalExperts = static_cast<int32_t>(worldSize_ * expertPerRank_);
    uint32_t countWordIdx = static_cast<uint32_t>(maskAlignSize_) / sizeof(int32_t);  // count 在槽内 int32 偏移
    DataCopyExtParams maskCopyParams{1U, static_cast<uint32_t>(maskSlotSize_), 0U, 0U, 0U};
    int32_t iter = 0;
    GlobalTensor<uint8_t> dstGlobalTensor;
    for (int32_t curExpertId = aivCoreIdx_; curExpertId < totalExperts; curExpertId += blockAivNum_, ++iter) {
        int32_t dstRank = curExpertId / static_cast<int32_t>(expertPerRank_);
        int32_t expertNumPer = curExpertId % static_cast<int32_t>(expertPerRank_);
        TEventID eventId = kBufEvents[iter % static_cast<int32_t>(DOUBLE_BUFFER)];
        LocalTensor<uint8_t> maskBuf = sendMaskTensor_[iter % static_cast<int32_t>(DOUBLE_BUFFER)];
        LocalTensor<uint32_t> maskBufU32 = maskBuf.template ReinterpretCast<uint32_t>();
        LocalTensor<int32_t> maskBufI32 = maskBuf.template ReinterpretCast<int32_t>();

        WaitFlag<AscendC::HardEvent::MTE3_V>(eventId);  // 等本 buffer 上一轮 MTE3 推送完成
        CompareScalar(maskBuf, topkIdsTensor_, curExpertId, AscendC::CMPMODE::EQ, compareCount_);
        // 同步算 count: GatherMask 对 mask 取 set-bit 数(mask=sendTotalNum_ 跳过尾部 padding)。
        uint64_t sendCnt = 0;
        GatherMask(sendGatherOutTensor_, topkIdsTensor_, maskBufU32, true,
            static_cast<uint32_t>(sendTotalNum_), {1, 1, 0, 0}, sendCnt);
        SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();   // count 标量就绪
        maskBufI32.SetValue(countWordIdx, static_cast<int32_t>(sendCnt));  // 写入槽内 count
        SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>(); // count 对 MTE3 可见(V 已被 V_S 排空)

        uint64_t dstOffset = maskWinOffset_ +
            static_cast<uint64_t>(expertNumPer * static_cast<int32_t>(worldSize_) + static_cast<int32_t>(rankId_)) *
            static_cast<uint64_t>(maskSlotSize_);
        dstGlobalTensor.SetGlobalBuffer(
            (__gm__ uint8_t*)GetRankWinAddrWithOffset(dstRank, dstOffset));
        DataCopyPad(dstGlobalTensor, maskBuf, maskCopyParams);
        SetFlag<AscendC::HardEvent::MTE3_V>(eventId);
    }
    for (int32_t index = 0; index < static_cast<int32_t>(DOUBLE_BUFFER); ++index) {
        WaitFlag<AscendC::HardEvent::MTE3_V>(kBufEvents[index]);
    }
}

// ===================================
// QuantProcessInRank：本卡token的量化
// ===================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::QuantProcessInRank()
{
    if constexpr(g_coreType == AIC) {
        return;
    }

    // 分核，按照BS与aivCoreNum均分
    int32_t curentNum;
    int32_t curentOffset;
    TilingByCore(m_, curentNum, curentOffset, 1);
    uint32_t H = k_;
    GlobalTensor<bfloat16_t> srcGlobalTensor;
    GlobalTensor<uint8_t> dstGlobalTensor;
    DataCopyParams xCopyInParams = {1U, static_cast<uint16_t>(H * sizeof(bfloat16_t)), 0U, 0U};
    DataCopyPadParams xCopyInPadParams{true, 0, 0, 0};
    DataCopyExtParams xCopyOutParams = {1U,
        static_cast<uint32_t>(mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_), 0U, 0U, 0U};
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int32_t index = 0; index < curentNum; index++) {
        srcGlobalTensor.SetGlobalBuffer((__gm__ bfloat16_t*)(params_.aGmAddr +
            (curentOffset + index) * H * sizeof(bfloat16_t)));
        dstGlobalTensor.SetGlobalBuffer((__gm__ uint8_t*)(params_.peermemInfo.quantTokenScalePtr +
            (curentOffset + index) * mxQuantTokenScaleAlignBytes_));
        auto event = (index % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
        auto xInTensor = (index % DOUBLE_BUFFER == 0) ? xInTensor1_ : xInTensor2_;
        auto xOutTensor = (index % DOUBLE_BUFFER == 0) ? xOutTensor1_ : xOutTensor2_;
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        DataCopyPad(xInTensor, srcGlobalTensor, xCopyInParams, xCopyInPadParams);
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        __ubuf__ bfloat16_t* srcAddr = (__ubuf__ bfloat16_t*)xInTensor.GetPhyAddr();
        __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)mxTempTensor_.GetPhyAddr();
        __ubuf__ uint16_t* halfScaleAddr = (__ubuf__ uint16_t*)mxTempTensor_[Ops::Base::CeilAlign(
        mxQuantScaleNumAlignPerToken_, static_cast<uint32_t>(ALIGN_32))].GetPhyAddr();
        __ubuf__ int8_t* outDataAddr = (__ubuf__ int8_t*)xOutTensor.GetPhyAddr();
        __ubuf__ uint16_t* mxScaleAddr = (__ubuf__ uint16_t*)xOutTensor[mxQuantTokenAlignBytes_].GetPhyAddr();

        quant::ComputeMaxExp(srcAddr, maxExpAddr, H); // 计算最大Exp
        quant::ComputeScale<QuantOutType>(maxExpAddr, mxScaleAddr, halfScaleAddr,
        mxQuantScaleNumAlignPerToken_); // 计算scales并填充f
        if constexpr (QuantMode == E2M1_QUANT) {
            quant::ComputeFp4Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC,
                AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleAddr, outDataAddr, H);
        } else {
            quant::ComputeFp8Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC,
                AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleAddr, outDataAddr, H);
        }
        SetFlag<AscendC::HardEvent::V_MTE3>(event);
        WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        auto xOutBytesTensor = xOutTensor.template ReinterpretCast<uint8_t>();
        DataCopyPad(dstGlobalTensor, xOutBytesTensor, xCopyOutParams);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

// ==================================================================================================
// SendCntCal：目标专家卡读 count 计数，得到当前专家Id收到的token总数
// --------------------------------------------------------------------------------------------------
//   Phase 1: 从本卡 win 将当前 localExpertId 的 worldSize_ 个 [mask|count] 槽位读进 gatherMaskTensor_;
//   Phase 2: 逐卡读取count → 累加 totalSendCntInExp_/cumsumRevCntInRank_, 写 cumsumInfoTensor_;
//   Phase 3: 写 expertRevNumsGloablTensor_ + CrossCoreSetFlag 通知 AIC;
// ==================================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::SendCntCal(int32_t localExpertId)
{
    totalSendCntInExp_ = 0;
    uint32_t slotWordStride = static_cast<uint32_t>(maskSlotSize_) / sizeof(uint32_t);
    uint32_t countWordIdx = static_cast<uint32_t>(maskAlignSize_) / sizeof(uint32_t);

    // Phase 1: 从本卡 win 读本 localExpert 的全部 worldSize_ 个 [mask|count] 槽位
    GlobalTensor<uint8_t> maskSrcGlobal;
    maskSrcGlobal.SetGlobalBuffer((__gm__ uint8_t*)(params_.peermemInfo.maskRecvPtr +
        static_cast<uint64_t>(localExpertId) * worldSize_ * maskSlotSize_));
    DataCopy(gatherMaskTensor_, maskSrcGlobal, worldSize_ * maskSlotSize_);
    SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>();   // count 读取(标量)就绪
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();   // mask 供 Triple 的 GatherMask(V)就绪

    // Phase 2: 逐源卡直接读取槽内 count + cumsum
    for (int32_t calRankId = 0; calRankId < static_cast<int32_t>(worldSize_); ++calRankId) {
        int32_t sendCnt = gatherMaskInt32Tensor_.GetValue(calRankId * slotWordStride + countWordIdx);
        totalSendCntInExp_ += static_cast<uint64_t>(sendCnt);
        cumsumRevCntInRank_ += static_cast<uint64_t>(sendCnt);
        cumsumInfoTensor_.SetValue(localExpertId * worldSize_ + calRankId, static_cast<int32_t>(cumsumRevCntInRank_));
    }
    
    // Phase 3: 写到 gm 上，并通知 AIC
    expertTokenCntTensor_.SetValue(0, totalSendCntInExp_);
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    DataCopy<int32_t>(expertRevNumsGloablTensor_[localExpertId * INT32_PER_256B * aicNum_ + INT32_PER_256B * blockIdx_],
        expertTokenCntTensor_, INT32_PER_256B);
    PipeBarrier<PIPE_ALL>();
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_MTE3>(2);
}

// ============================================================================
// CopyGMToGMPerToken：6-buffer 软流水，搬运对端rank数据至本专家卡
// ----------------------------------------------------------------------------
//   Phase 1: 所有 token 的三元组 (rank, tokenIndex, topkIndex) 组装写入tripleTensor_
//   Phase 2 prime: 连发 6 个 MTE2,中间不插 WaitFlag<MTE2_MTE3>, 让 MTE2 引擎同时持有 6 个跨卡读请求。
//   Phase 2 steady: 每轮做 (MTE3_out[i] + MTE2_in[i+6]), 槽位循环复用。
//   Phase 2 drain: 收尾不再发新 MTE2,只等 MTE3。
//   Phase 3: triple 三元组搬出。
// ============================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::CopyGMToGMPerToken(
    int32_t rowDstOffsetInCore, int32_t remoteRankIdx, int32_t copyStartIdx, int32_t copyNum)
{
    if (copyNum <= 0) {
        return;
    }
    constexpr int32_t BufferNum = DISPATCH_BUFFER_NUM;
    constexpr TEventID kBufEvents[BufferNum] = {EVENT_ID0, EVENT_ID1, EVENT_ID2, EVENT_ID3, EVENT_ID4, EVENT_ID5};
    int64_t widthA = k_ / ELEMS_PER_BYTE;
    int64_t widthAScale = Ops::Base::CeilDiv(static_cast<int64_t>(k_), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE;  // 输出 token-scale 长度,紧密排列
    uint32_t copyInNum = Ops::Base::CeilAlign(static_cast<int64_t>(mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_),
        static_cast<int64_t>(ALIGN_32)); // 输入 token-scale 拼接,非紧密排列
    GlobalTensor<ActivationType> remoteRankGlobalTensor;
    GlobalTensor<ActivationType> tokenRevGlobalTensor;
    GlobalTensor<QuantScaleOutType> scaleRevGlobalTensor;
    tokenRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType*>(
        params_.workspaceInfo.dispatchRevDataPtr + rowDstOffsetInCore * widthA));
    scaleRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleOutType *>(
        params_.workspaceInfo.dispatchRevScalePtr + rowDstOffsetInCore * widthAScale));
    remoteRankGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType*>(
        GetRankWinAddrWithOffset(remoteRankIdx, quantWinOffset_)));

    // 预置 6 个 MTE3_MTE2 flag,Phase 2 的 prime/steady 的 WaitFlag 可立即通过。
    for (int32_t bufferIdx = 0; bufferIdx < BufferNum; ++bufferIdx) {
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(kBufEvents[bufferIdx]);
    }
    PipeBarrier<PIPE_ALL>();
    // Phase 1: token三元组信息组装(rank, tokenIndex, topkIndex)
    for (int32_t i = 0; i < copyNum; ++i) {
        int32_t topkIndex = validTopkIndexTensor_.GetValue(copyStartIdx + i);
        int32_t tokenIndex = topkIndex / topK_;
        tripleTensor_[i * INT32_PER_256B].SetValue(RANK_ID, remoteRankIdx);
        tripleTensor_[i * INT32_PER_256B].SetValue(TOKEN_ID, tokenIndex);
        tripleTensor_[i * INT32_PER_256B].SetValue(TOPK_INDEX, topkIndex % topK_);
    }

    // 6-buffer SW 流水 MTE2 + MTE3
    // Phase 2 prime: 发出前 BufferNum 个 token 的 MTE2
    int32_t primeCount = (copyNum < BufferNum) ? copyNum : BufferNum;
    for (int32_t primeIdx = 0; primeIdx < primeCount; ++primeIdx) {
        int32_t tokenIndex = tripleTensor_[primeIdx * INT32_PER_256B].GetValue(TOKEN_ID);
        uint64_t remoteCopyOffset = static_cast<uint64_t>(tokenIndex) * static_cast<uint64_t>(copyInNum);
        TEventID eventId = kBufEvents[primeIdx];
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        DataCopy(copyTmpTensors_[primeIdx], remoteRankGlobalTensor[remoteCopyOffset], copyInNum);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
    }
    // Phase 2 steady: MTE3[copyIdx] + issueMTE2[copyIdx + BufferNum]
    for (int32_t copyIdx = 0; copyIdx < copyNum; ++copyIdx) {
        int32_t outIdx = copyIdx % BufferNum;
        TEventID eventId = kBufEvents[outIdx];
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);

        LocalTensor<ActivationType> tokenScalebuf = copyTmpTensors_[outIdx];
        LocalTensor<QuantScaleOutType> bufScale =
            tokenScalebuf[mxQuantTokenAlignBytes_].template ReinterpretCast<QuantScaleOutType>();
        DataCopyPad(tokenRevGlobalTensor[copyIdx * widthA], tokenScalebuf,
            {1, static_cast<uint16_t>(widthA * sizeof(ActivationType)), 0U, 0U, 0U});
        DataCopyPad(scaleRevGlobalTensor[copyIdx * widthAScale], bufScale,
            {1, static_cast<uint16_t>(widthAScale * sizeof(QuantScaleOutType)), 0U, 0U, 0U});
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);

        // 发下一个槽的 MTE2 (copyIdx + BufferNum,复用 outIdx 槽)
        int32_t nextIdx = copyIdx + BufferNum;
        if (nextIdx < copyNum) {
            int32_t tokenIndex = tripleTensor_[nextIdx * INT32_PER_256B].GetValue(TOKEN_ID);
            uint64_t remoteCopyOffset = static_cast<uint64_t>(tokenIndex) * static_cast<uint64_t>(copyInNum);
            // WaitFlag 此处等待的是本轮刚发出的 SetFlag<MTE3_MTE2>(eventId),即等本槽 MTE3 完成。
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            DataCopy(copyTmpTensors_[outIdx], remoteRankGlobalTensor[remoteCopyOffset], copyInNum);
            SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        }
    }
    // Phase 2 drain: 6 个 buffer-free flag
    for (int32_t bufferIdx = 0; bufferIdx < BufferNum; ++bufferIdx) {
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kBufEvents[bufferIdx]);
    }
    // Phase 3: triple 三元组搬出
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();
    DataCopy(tripleGloablTensor_[rowDstOffsetInCore * INT32_PER_256B], tripleTensor_, copyNum * INT32_PER_256B);
}

// ====================================================================================================
// TripleInfoCalAndDispatch：专家接收token的三元组信息计算搬出 & token dispatch & 写Flag位
// ----------------------------------------------------------------------------------------------------
//   Phase 1: 按照blockNumPerRank_对aiv进行分组，一个rank归一个组aiv处理，组内根据该卡要发的token数量进行分核；
//   Phase 2: dispatch->gmm1 flag位AtomicAdd，每个 expert 有 maxWavesPerExpert_ 个槽位读写Flag；
// ====================================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::TripleInfoCalAndDispatch(
    GMMAddrInfo &gmmAddrInfo, int32_t localExpertId)
{
    // Phase 1: 分组分核进行三元组组装以及token dispatch
    constexpr int32_t L1_TILE_M_I32 = static_cast<int32_t>(MegaMoeImpl::L1_TILE_M_256);
    int32_t priorExpertCumsum = (localExpertId == 0) ? 0 : // 前面所有 expert 在本卡的总 token 数
        cumsumInfoTensor_.GetValue(localExpertId * worldSize_ - 1);
    for (uint32_t idx = blockIdx_; idx < worldSize_ * blockNumPerRank_; idx += blockNum_) {
        uint32_t dstRankIdx = idx / blockNumPerRank_;
        uint64_t sendToCurExpTokenCnt = 0;
        GatherMask(validTopkIndexTensor_, topkIndexTensor_,
            gatherMaskInt32Tensor_[dstRankIdx * (maskSlotSize_ / sizeof(uint32_t))],
            true, sendTotalNum_, {1, 1, 0, 0}, sendToCurExpTokenCnt);
        uint32_t aivIdxInGroup = idx % blockNumPerRank_;
        uint32_t rowStartIdxInDst = ((dstRankIdx == 0 && localExpertId == 0) ? 0 :
            cumsumInfoTensor_.GetValue(localExpertId * worldSize_ + dstRankIdx - 1));
        int32_t rowsThisCore = 0;
        int32_t rowDstOffsetInCore = 0;
        if (rowStartIdxInDst < maxOutputSize_) {
            int32_t rowsCopyCnt = sendToCurExpTokenCnt;
            if (rowStartIdxInDst + rowsCopyCnt > maxOutputSize_) {
                rowsCopyCnt = maxOutputSize_ - rowStartIdxInDst;
            }
            int32_t rowsNumInCore = Ops::Base::CeilDiv(rowsCopyCnt, static_cast<int32_t>(blockNumPerRank_));
            int32_t rowSrcIdxInCore = aivIdxInGroup * rowsNumInCore;
            rowDstOffsetInCore = rowStartIdxInDst + rowSrcIdxInCore;
            if (rowDstOffsetInCore + rowsNumInCore > rowStartIdxInDst + rowsCopyCnt) {
                rowsNumInCore = rowStartIdxInDst + rowsCopyCnt - rowDstOffsetInCore;
            }
            if (rowsNumInCore > 0) {
                uint32_t tripleTensorAddr = ubBufferUsedAddr_;
                uint32_t tripleTensorSize = rowsNumInCore * ALIGN_32;
                tripleTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, tripleTensorAddr,
                    tripleTensorSize / sizeof(int32_t));
                SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID4>();
                CopyGMToGMPerToken(rowDstOffsetInCore, dstRankIdx, rowSrcIdxInCore, rowsNumInCore);
                rowsThisCore = rowsNumInCore;
            }
        }

        // Phase 2: flag位AtomicAdd，确认当前token需要atomic的槽位
        if (rowsThisCore > 0) {
            SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID5>();
            // rowDstOffsetInCore 是跨expert累加的全局偏移(含前置 expert 的行数)，需要减掉前置专家行数;
            int32_t rowStartLocal = rowDstOffsetInCore - priorExpertCumsum;
            int32_t rowEndLocal = rowStartLocal + rowsThisCore;
            int32_t waveLo = rowStartLocal / L1_TILE_M_I32;
            int32_t waveHi = (rowEndLocal - 1) / L1_TILE_M_I32;
            __gm__ int32_t* flagBase = gmmAddrInfo.groupFlagList2;
            for (int32_t w = waveLo; w <= waveHi; ++w) {
                int32_t waveStartLocal = w * L1_TILE_M_I32;
                int32_t waveEndLocal = waveStartLocal + L1_TILE_M_I32;
                int32_t lo = rowStartLocal > waveStartLocal ? rowStartLocal : waveStartLocal;
                int32_t hi = rowEndLocal < waveEndLocal ? rowEndLocal : waveEndLocal;
                AtomicAdd(flagBase + w, int32_t(hi - lo));
            }
        }
    }
}

// =====================================================================================================
// UpdateGroupParams：更新当前expertIdx的problemShape_与baseOffset_，为后续GMM运算准备矩阵维度与偏移
// ----------------------------------------------------------------------------------------------------
// 整体流程：
//   1. 若不是第一个专家，则累加 baseOffset_：根据上一个专家的 M(收到的token数) 前移所有矩阵/量化scale/flag偏移，
//      使得当前专家在全局workspace中定位到属于自己的数据区域；
//   2. 更新 problemShape_ 的 M 维度（当前专家实际收到的token数）：
//      - gmmIndex==0 (第一轮GMM1+SwiGlu) 时，AIC/AIV 通过跨核同步读取 AIV(subBlockIdx_=1) 写入的
//        expertRevNumsGloablTensor_ 来获取 token 数；
//      - gmmIndex==1 (第二轮GMM2) 时，直接从 expertRevNumsGloablTensor_ 读取（此时数据已就绪，无需同步）；
//   3. 若当前专家收到的token数为0，返回false跳过该专家的计算。
//
// 参数说明：
//   expertIdx  : 当前本卡专家的局部索引(0 ~ expertPerRank_-1)
//   gmmIndex   : 当前处于哪一轮GMM：0=第一轮(GMM1+SwiGluQuant), 1=第二轮(GMM2+Combine)
//
// baseOffset_ 中各字段含义与增量公式（BlockOffset 是一个 Shape<int64_t,...,int64_t>，12个元素）：
//   [IDX_A_OFFSET      = 0] : dispatch接收到的量化token数据(A矩阵)偏移，增量 = m * k
//                               m=token数, k=hiddenDim，A是 [m, k] 的fp8矩阵
//   [IDX_B_OFFSET      = 1] : GMM1权重(B矩阵)偏移，增量 = n * k
//                               n=hiddenDim(输出维度), k=hiddenDim(输入维度)，B是 [n, k] 的fp8权重
//   [IDX_A_SCALE_OFFSET= 2] : A矩阵量化scale偏移，增量 = m * scaleK
//                               scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE
//                               即每32个元素一组scale(MXFP)，每组2字节(e8m0)，共 CeilDiv(k/64)*2 个scale元素
//   [IDX_B_SCALE_OFFSET= 3] : B矩阵量化scale偏移，增量 = n * scaleK
//                               同A scale的公式，B按列分组，每组64列对应2字节scale
//   [未使用            = 4] : baseOffset_第4位保留
//   [IDX_C_OFFSET      = 5] : GMM1输出/SwiGluQuant中间结果(C矩阵)偏移，增量 = m * n / SWIGLU_N_HALF
//                               SwiGlu将n维度减半(n/2)，C是 [m, n/2] 的fp8中间结果
//   [IDX_C_SCALE_OFFSET= 6] : C矩阵量化scale偏移，增量 = m * CeilDiv(n/2, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE
//                               对SwiGlu减半后的维度 n/2 按 MXFP 规则计算scale数量
//   [IDX_FLAG_OFFSET   = 7] : expert级flag偏移，增量 = 1
//                               每个专家独占一组flag(SwiGluToGmm2 INT_CACHELINE个 + DispatchToGmm1 dispatchFlagSlotsPerExpert_个)
//   [IDX_B2_OFFSET     = 8] : GMM2权重(B2矩阵)偏移，增量 = k * n / SWIGLU_N_HALF
//                               B2是 [k, n/2] 的fp8权重，因为GMM2的输入是SwiGlu减半后的n/2维
//   [IDX_B2_SCALE_OFFSET= 9] : B2矩阵量化scale偏移，增量 = k * CeilDiv(n/2, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE
//                               B2按列分组，列数=n/2，scale数量同理
//   [IDX_Y2_OFFSET     = 10] : GMM2输出(Y2矩阵)偏移，增量 = m * k
//                               Y2是 [m, k] 的最终Combine结果，维度回到原始k
//   [IDX_M_OFFSET      = 11] : 专家级M(行数/token数)累计偏移，增量 = m
//                               用于GMM2阶段定位当前专家在所有专家中的全局行偏移
// =====================================================================================================
template <TemplateMegaMoeTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline bool MegaMoe<TemplateMegaMoeTypeFunc>::UpdateGroupParams(ExpertLoopState &state, uint32_t expertIdx)
{
    // === Phase 1: 基偏移累加 ===
    // 仅在非首个专家时执行：根据上一个专家的M(收到的token数)前推所有偏移量，
    // 使得当前专家在workspace中找到属于自己的数据起始位置。
    // expertIdx==0 时 baseOffset_ 已在 Init 中初始化为全0，无需累加。
    if (expertIdx != 0) {
        // m: 上一个专家收到的token数（行数），来自 problemShape_ 的 M 维度
        // n: 输出维度（hiddenDim），来自 problemShape_ 的 N 维度，全局不变
        // k: 输入维度（hiddenDim），来自 problemShape_ 的 K 维度，全局不变
        uint64_t m = Get<M_VALUE>(problemShape_);
        uint64_t n = Get<N_VALUE>(problemShape_);
        uint64_t k = Get<K_VALUE>(problemShape_);

        // expertBeforeCnt_: 累计前面所有专家收到的token总数，
        // 用于GMM2阶段(Combine)定位当前专家在全局workspace中的行偏移
        expertBeforeCnt_ += m;

        // A矩阵偏移：dispatch接收的fp8量化token数据，形状[m, k]，增量=m*k个元素
        Get<IDX_A_OFFSET>(baseOffset_) += m * k;

        // B矩阵偏移：GMM1权重，形状[n, k]，增量=n*k个元素
        // 不同专家的权重在GM上紧密排列，按n*k步长前移
        Get<IDX_B_OFFSET>(baseOffset_) += n * k;

        // MXFP量化scale长度计算：
        //   scaleK = CeilDiv(k, 64) * 2
        //   每64个元素共享1个e8m0 scale(2字节)，向上取整后再乘MXFP_MULTI_BASE_SIZE(=2)
        auto scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

        // A矩阵scale偏移：每行有scaleK个scale元素，m行共m*scaleK，增量=m*scaleK
        Get<IDX_A_SCALE_OFFSET>(baseOffset_) += m * scaleK;

        // B矩阵scale偏移：每列有scaleK个scale元素，n列共n*scaleK，增量=n*scaleK
        Get<IDX_B_SCALE_OFFSET>(baseOffset_) += n * scaleK;

        // C矩阵偏移：GMM1输出经SwiGlu激活后的中间结果，形状[m, n/2]，增量=m*(n/2)个元素
        // SWIGLU_N_HALF=2: SwiGlu操作将输出维度n减半(n/2)
        Get<IDX_C_OFFSET>(baseOffset_) += m * n / SWIGLU_N_HALF;

        // C矩阵scale偏移：对SwiGlu减半后的n/2维度按MXFP规则计算scale数量，增量=m*CeilDiv(n/2,64)*2
        Get<IDX_C_SCALE_OFFSET>(baseOffset_) +=
            m * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

        // expert级flag偏移：每个专家独占一组flag区，增量=1（代表1个专家单位）
        // 实际字节偏移在UpdateGlobalBuffer中按 INT_CACHELINE 或 dispatchFlagSlotsPerExpert_ 展开
        Get<IDX_FLAG_OFFSET>(baseOffset_) += 1;

        // B2矩阵偏移：GMM2权重，形状[k, n/2]，增量=k*(n/2)个元素
        // GMM2的输入是SwiGlu的n/2维输出，所以B2列数=n/2
        Get<IDX_B2_OFFSET>(baseOffset_) += k * n / SWIGLU_N_HALF;

        // B2矩阵scale偏移：B2按列分组，列数=n/2，每列scaleK个scale，增量=k*CeilDiv(n/2,64)*2
        // 注意：B2是[k, n/2]，scale按列方向排列，每列k个元素，ceil(k/64)*2个scale
        Get<IDX_B2_SCALE_OFFSET>(baseOffset_) +=
            k * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

        // Y2矩阵偏移：GMM2最终输出(Combine结果)，形状[m, k]，增量=m*k个元素
        // Y2维度回到原始k(与输入同维度)
        Get<IDX_Y2_OFFSET>(baseOffset_) += m * k;

        // M累计偏移：记录前面所有专家的token数之和，用于GMM2阶段跨专家的行定位
        Get<IDX_M_OFFSET>(baseOffset_) += m;
    }

    // === Phase 2: 更新当前专家的M维度(收到的token数) ===
    // gmm1阶段(gmmIndex==0)：token数由AIV的subBlockIdx_=1在SendCntCal中计算并写入
    // expertRevNumsGloablTensor_，然后通过CrossCoreSetFlag通知AIC读取。
    // AIC需要先WaitFlag等AIV写完，AIV的subBlockIdx_=0也需要等AIC读完后再读(避免缓存不一致)。
    if (gmmIndex == 0) {
        // AIC路径：等待AIV(subBlockIdx_=1)通过CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_MTE3>(2)
        // 发出的信号。flag编号2+16=18，其中2是SendCntCal中设置的flag号，16是AIC侧的偏移基址。
        if constexpr (g_coreType == AscendC::AIC) {
            CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_S>(2 + 16);

            // 计算当前AIC核在expertRevNumsGloablTensor_中的偏移：
            // expertIdx * 8 * aicNum_ : 每个专家占8个int32(256B对齐)*aicNum_个AIC核的空间
            // + 8 * blockIdx_         : 当前AIC核在该专家区域内的偏移(每个AIC核占8个int32=256B)
            uint64_t offsetInCnt = expertIdx * 8 * aicNum_ + 8 * blockIdx_;

            // 刷新数据缓存：确保读到AIV最新写入的值，而非本地缓存中的旧值
            DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
                expertRevNumsGloablTensor_[offsetInCnt]);

            // 从全局内存读取当前专家收到的token数，更新problemShape_的M维度
            Get<M_VALUE>(problemShape_) = expertRevNumsGloablTensor_.GetValue(offsetInCnt);

            // 通知AIV(subBlockIdx_=0)可以读取：AIC读完后设置flag 3，
            // AIV的subBlockIdx_=0在下方WaitFlag(3)处等待此信号
            CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_S>(3);
        }
        // AIV路径：subBlockIdx_=0等待AIC读完并设置flag 3后，再从expertRevNumsGloablTensor_读取；
        // subBlockIdx_=1已经在SendCntCal中直接计算了totalSendCntInExp_，无需从GM读取。
        if constexpr (g_coreType == AscendC::AIV) {
            if (subBlockIdx_ == 0) {
                // 等待AIC读完并设置flag 3，确保AIC的DataCacheCleanAndInvalid已完成，
                // 避免AIV读到缓存中的脏数据
                CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_S>(3);

                // 与AIC相同的偏移计算方式，从expertRevNumsGloablTensor_读取token数
                uint64_t offsetInCnt = expertIdx * 8 * aicNum_ + 8 * blockIdx_;
                DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
                expertRevNumsGloablTensor_[offsetInCnt]);
                Get<M_VALUE>(state.problemShape) = expertRevNumsGloablTensor_.GetValue(offsetInCnt);
            } else {
                // subBlockIdx_=1是实际执行SendCntCal的核，已经计算出totalSendCntInExp_
                // (该专家从所有源卡收到的token总数)，直接使用本地值，避免GM读取开销
                Get<M_VALUE>(problemShape_) = totalSendCntInExp_;
            }
        }
    } else {
        // gmm2阶段(gmmIndex==1)：此时所有专家的token数已在gmm1阶段写入expertRevNumsGloablTensor_，
        // 数据已就绪无需跨核同步，直接刷新缓存并读取即可。
        uint64_t offsetInCnt = expertIdx * 8 * aicNum_ + 8 * blockIdx_;
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
                expertRevNumsGloablTensor_[offsetInCnt]);
        Get<M_VALUE>(state.problemShape) = expertRevNumsGloablTensor_.GetValue(offsetInCnt);
    }

    // 若当前专家收到的token数为0，则无需进行GMM计算，返回false让Process()跳过该专家
    if (Get<M_VALUE>(problemShape_) == 0) {
        return false;
    }
    return true;
}

// =====================================================================================================
// UpdateGlobalBuffer：根据 baseOffset_ 更新当前专家的 gmmAddrInfo 全局地址指针
// ----------------------------------------------------------------------------------------------------
// 整体流程：
//   根据当前处于第几轮GMM(gmmIndex)以及核类型(AIC/AIV)，将 baseOffset_ 中累计的元素偏移量
//   转换为字节偏移，并与workspace/权重基址相加，得到当前专家在全局内存(GM)中各矩阵的实际地址。
//
// 数据流与地址映射关系：
//
//   gmmIndex==0 (第一轮: GMM1 + SwiGluQuant)：
//     计算过程: A(token) x B(weight1) → C(中间) → SwiGlu(C) → C'(量化输出)
//     A来源 = dispatch接收的fp8量化token (dispatchRevDataPtr)
//     A scale来源 = dispatch接收的量化scale (dispatchRevScalePtr)
//     B来源 = 专家第一层权重 (weight1/bGmAddr)，仅AIC使用(矩阵乘法在AIC上执行)
//     B scale来源 = 专家第一层权重scale (weightScales1/bScaleGmAddr)，仅AIC使用
//     AIV侧不设bGlobal/bScaleGlobal，而是更新 epilogueOp_ (SwiGlu+量化后处理) 的输出地址：
//       - C偏移 → SwiGlu量化后的fp8数据地址(swigluQuantDataPtr)
//       - C scale偏移 → SwiGlu量化后的scale地址(swigluQuantScalePtr)
//       - FLAG偏移 → SwiGlu→GMM2的flag地址(flagSwiGluToGmm2Ptr)
//
//   gmmIndex==1 (第二轮: GMM2 + Combine)，仅AIC执行(else if constexpr(g_coreType == AIC))：
//     计算过程: C'(SwiGlu输出) x B2(weight2) → Y2(最终输出)
//     A来源 = SwiGlu量化输出 (swigluQuantDataPtr)，复用gmm1阶段的C偏移(IDX_C_OFFSET)
//     A scale来源 = SwiGlu量化scale (swigluQuantScalePtr)，复用C scale偏移(IDX_C_SCALE_OFFSET)
//     B来源 = 专家第二层权重 (weight2/b2GmAddr)
//     B scale来源 = 专家第二层权重scale (weightScales2/b2ScaleGmAddr)
//     AIV不参与GMM2计算，因此gmmIndex==1时AIV分支什么都不做
//
// 参数说明：
//   gmmAddrInfo  : 输出参数，GMM地址信息结构体，包含6个字段：
//                    aGlobal      - A矩阵(左矩阵/输入)的全局地址
//                    bGlobal      - B矩阵(右矩阵/权重)的全局地址
//                    aScaleGlobal - A矩阵量化scale的全局地址
//                    bScaleGlobal - B矩阵量化scale的全局地址
//                    groupFlagList  - SwiGlu→GMM2的expert级flag地址
//                    groupFlagList2 - Dispatch→GMM1的wave-grain flag地址
//   gmmIndex     : 0=第一轮(GMM1+SwiGluQuant), 1=第二轮(GMM2+Combine)
//   localExpertId: 当前本卡专家的局部索引(0 ~ expertPerRank_-1)
//
// 偏移转换规则：
//   baseOffset_ 中存储的是"元素个数"偏移(以QuantOutType/QuantScaleOutType为单位)，
//   转换为字节偏移时需乘 sizeof(QuantOutType)(fp8=1字节) 或 sizeof(QuantScaleOutType)(e8m0=1字节/fp32=4字节)。
// =====================================================================================================
template <TemplateMegaMoeTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo,
    const ExpertLoopState &state)
{
    // === gmmIndex==0: 第一轮GMM (GMM1 + SwiGluQuant) 的地址更新 ===
    if (gmmIndex == 0) {
        // A矩阵地址：dispatch接收的fp8量化token数据基址 + A元素偏移 * sizeof(fp8)
        // IDX_A_OFFSET 记录了前面所有专家的A矩阵元素总数(m*k累加)，乘以fp8元素大小(1字节)得到字节偏移
        gmmAddrInfo.aGlobal = params_.workspaceInfo.dispatchRevDataPtr +
            Get<IDX_A_OFFSET>(baseOffset_) * sizeof(QuantOutType);

        // A矩阵scale地址：dispatch接收的MXFP量化scale基址 + A scale元素偏移 * sizeof(scale)
        // IDX_A_SCALE_OFFSET 记录了前面所有专家的A scale元素总数(m*scaleK累加)
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.dispatchRevScalePtr +
        Get<IDX_A_SCALE_OFFSET>(baseOffset_) * sizeof(QuantScaleOutType);

        // AIC和AIV在gmm1阶段承担不同职责：
        //   AIC: 执行GMM1矩阵乘法 A x B → C，需要设置B矩阵地址
        //   AIV: 执行SwiGlu激活+量化后处理(epilogueOp_)，需要设置SwiGlu输出地址和flag地址
        if constexpr(g_coreType == AIC) {
            // B矩阵地址：专家第一层权重基址 + B元素偏移 * sizeof(fp8)
            // IDX_B_OFFSET 记录了前面所有专家的B矩阵元素总数(n*k累加)
            // 每个专家的权重按 [n, k] 形状紧密排列，AIC按顺序读取
            gmmAddrInfo.bGlobal = params_.bGmAddr + Get<IDX_B_OFFSET>(baseOffset_) * sizeof(QuantOutType);

            // B矩阵scale地址：专家第一层权重scale基址 + B scale元素偏移 * sizeof(scale)
            // IDX_B_SCALE_OFFSET 记录了前面所有专家的B scale元素总数(n*scaleK累加)
            gmmAddrInfo.bScaleGlobal = params_.bScaleGmAddr + Get<IDX_B_SCALE_OFFSET>(baseOffset_) *
                                        sizeof(QuantScaleOutType);
        } else {
            // AIV侧：更新 epilogueOp_ (BlockEpilogueSwigluMxQuant) 的全局地址
            // vecBaseOffset 包含6个字段：
            //   [0] IDX_C_OFFSET     : SwiGlu量化输出数据的元素偏移，用于定位 swigluQuantDataPtr 中当前专家的输出区
            //   [1] IDX_C_SCALE_OFFSET: SwiGlu量化输出scale的元素偏移，用于定位 swigluQuantScalePtr 中当前专家的scale区
            //   [2] IDX_FLAG_OFFSET   : expert级flag偏移，用于定位 flagSwiGluToGmm2Ptr 中当前专家的flag区
            //   [3-5] 0L              : 保留字段，epilogueOp_ 内部未使用
            // epilogueOp_.UpdateGlobalAddr 会将这些偏移转换为实际GM地址，
            // 供SwiGlu激活+MXFP量化后处理将结果写入正确的workspace区域
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                Get<IDX_C_OFFSET>(state.baseOffset), Get<IDX_C_SCALE_OFFSET>(state.baseOffset),
                Get<IDX_FLAG_OFFSET>(state.baseOffset), 0L, 0L, 0L};
            epilogueOp_.UpdateGlobalAddr(vecBaseOffset);
        }
    } else if constexpr(g_coreType == AIC) {
        // === gmmIndex==1: 第二轮GMM (GMM2 + Combine) 的地址更新，仅AIC执行 ===
        // AIV不参与GMM2计算（Combine阶段的权重乘法全部在AIC上完成），
        // 所以此分支用 `else if constexpr(g_coreType == AIC)` 确保AIV编译时直接跳过。

        // A矩阵地址：SwiGlu量化输出的fp8数据基址 + C元素偏移 * sizeof(fp8)
        // GMM2的A矩阵就是GMM1+SwiGlu的输出，复用 IDX_C_OFFSET (SwiGlu输出形状[m, n/2])
        // 基址是 swigluQuantDataPtr (workspace中SwiGlu量化结果的存储区)
        gmmAddrInfo.aGlobal = params_.workspaceInfo.swigluQuantDataPtr +
            Get<IDX_C_OFFSET>(baseOffset_) * sizeof(QuantOutType);

        // A矩阵scale地址：SwiGlu量化输出的scale基址 + C scale元素偏移 * sizeof(scale)
        // 复用 IDX_C_SCALE_OFFSET，基址是 swigluQuantScalePtr
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.swigluQuantScalePtr + Get<IDX_C_SCALE_OFFSET>(baseOffset_) *
                                    sizeof(QuantScaleOutType);

        // B矩阵地址：专家第二层权重基址 + B2元素偏移 * sizeof(fp8)
        // IDX_B2_OFFSET 记录了前面所有专家的B2矩阵元素总数(k*(n/2)累加)
        // B2形状为 [k, n/2]，GMM2计算 C'(n/2维) x B2 → Y2(k维)
        gmmAddrInfo.bGlobal = params_.b2GmAddr + Get<IDX_B2_OFFSET>(baseOffset_) * sizeof(QuantOutType);

        // B矩阵scale地址：专家第二层权重scale基址 + B2 scale元素偏移 * sizeof(scale)
        // IDX_B2_SCALE_OFFSET 记录了前面所有专家的B2 scale元素总数
        gmmAddrInfo.bScaleGlobal = params_.b2ScaleGmAddr + Get<IDX_B2_SCALE_OFFSET>(baseOffset_) *
                                    sizeof(QuantScaleOutType);
    }

    // === 公共部分：两种flag地址更新 ===
    // groupFlagList: SwiGlu→GMM2的flag，用于AIV(SwiGlu完成后)通知AIC(GMM2可以开始)
    // 每个专家占 INT_CACHELINE(=16) 个int32(即64字节=1条cache line)，保证cache line对齐避免false sharing
    // 偏移计算: 基址 + expert编号 * INT_CACHELINE
    gmmAddrInfo.groupFlagList = (__gm__ int32_t*)params_.workspaceInfo.flagSwiGluToGmm2Ptr +
                                Get<IDX_FLAG_OFFSET>(baseOffset_) * INT_CACHELINE;

    // groupFlagList2: Dispatch→GMM1的wave-grain flag，用于AIV(dispatch完成后)通知AIC(GMM1可以开始)
    // 每个专家占 dispatchFlagSlotsPerExpert_ 个int32槽位(每个wave一个槽位，L1_TILE_M=256行/wave)，
    // 步长远大于 INT_CACHELINE，因为GMM1按wave粒度(256行)推进，需要多个flag槽位记录各wave的dispatch完成状态
    gmmAddrInfo.groupFlagList2 = (__gm__ int32_t*)params_.workspaceInfo.flagDispatchToGmm1Ptr +
                                Get<IDX_FLAG_OFFSET>(baseOffset_) * dispatchFlagSlotsPerExpert_;

    // 将计算好的flag地址绑定到成员GlobalTensor，供后续 ResetFlagList/TripleInfoCalAndDispatch 等函数
    // 通过 groupFlagListGm_ / groupFlagListGm2_ 直接访问当前专家的flag区域
    groupFlagListGm_.SetGlobalBuffer(gmmAddrInfo.groupFlagList);
}

// =============================================
// UnpermuteBuffInit：Unpermute中使用的buffer申请
// =============================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::UnpermuteBuffInit()
{
    uint32_t dataResBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(UNPERMUTE_LIST_NUM * k_ * sizeof(bfloat16_t)), static_cast<uint32_t>(ALIGN_32));
    int32_t num = worldSize_ * Ops::Base::CeilAlign(
        static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128)) * sizeof(int32_t);
    uint32_t dataResFp32BufAlign = dataResBufAlign * HALF_TO_FP32;
    uint32_t topKWeightsBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(m_ * topK_ * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
    uint32_t tempBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(m_ * topK_ * sizeof(bfloat16_t)), uint32_t(ALIGN_32));
    
    // Tensor用处：Unpermute 函数用于存储mte2搬入token；
    // Tensor大小：大小为3 * 单个token长度，2块是用于mte2搬运的doubleBuffer，1块是用于存储累加计算Cast完的输出结果，用于搬出；
    uint32_t dataResAddr = 0;
    uint32_t dataResSize = dataResBufAlign / sizeof(bfloat16_t);
    dataResTensor_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, dataResAddr, dataResSize);
    // Tensor用处：Unpermute 函数用于存储token Cast 目的Tensor；
    // Tensor大小：dataResTensor_开设大小乘以BF16_TO_FP32；
    uint32_t dataResFp32Addr = dataResAddr + dataResBufAlign;
    uint32_t dataResFp32Size = dataResFp32BufAlign / sizeof(float);
    dataResFp32Tensor_ = LocalTensor<float>(TPosition::VECCALC, dataResFp32Addr, dataResFp32Size);
    // Tensor用处：用于存储topKWeight；
    // Tensor大小：m_ * topK_ * sizeof(float) align到32字节对齐；
    uint32_t topKWeightsAddr = dataResFp32Addr + dataResFp32BufAlign;
    uint32_t topKWeightsSize = topKWeightsBufAlign / sizeof(float);
    topKWeightsTensor_ = LocalTensor<float>(TPosition::VECCALC, topKWeightsAddr, topKWeightsSize);
    uint32_t tempAddr = topKWeightsAddr + topKWeightsBufAlign;
    if constexpr (Std::IsSame<TopkWeightsType, float>::value) {
        GlobalTensor<float> topKWeightsGlobalTensor_;
        topKWeightsGlobalTensor_.SetGlobalBuffer((__gm__ float*)params_.probsGmAddr);
        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(m_ * topK_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(topKWeightsTensor_, topKWeightsGlobalTensor_, copyParams, copyPadParams);
    }
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        uint32_t tempSize = tempBufAlign / sizeof(bfloat16_t);
        LocalTensor<bfloat16_t> tempLocal(TPosition::VECCALC, tempAddr, tempSize);
        GlobalTensor<bfloat16_t> topkWeightsGlobalTensor;
        topkWeightsGlobalTensor.SetGlobalBuffer((__gm__ bfloat16_t*)params_.probsGmAddr);
        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(m_ * topK_ * sizeof(bfloat16_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<bfloat16_t> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(tempLocal, topkWeightsGlobalTensor, copyParams, copyPadParams);
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID2>();
        Cast(topKWeightsTensor_, tempLocal, AscendC::RoundMode::CAST_NONE, m_ * topK_);
    }
}

// ===============================================================
// Unpermute：对于各个专家还回来token的后处理，进行对应scale相乘与累加
// ===============================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Unpermute()
{
    int32_t coreLen, coreOffset;
    TilingByCore(m_, coreLen, coreOffset, 1);
    GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer((__gm__ bfloat16_t*)params_.peermemInfo.combineSendPtr);
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer((__gm__ bfloat16_t*)params_.y2GmAddr);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    for (int32_t tokenIdx = coreOffset; tokenIdx < coreLen + coreOffset; tokenIdx++) {
        SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID2>();
        LocalTensor<bfloat16_t> dataIn0Bf16 = dataResTensor_[k_];
        LocalTensor<bfloat16_t> dataIn1Bf16 = dataResTensor_[k_ * 2];
        LocalTensor<float> dataIn0Fp32 = dataResFp32Tensor_[k_];
        LocalTensor<float> dataIn1Fp32 = dataResFp32Tensor_[k_ * 2];
        for (int32_t expId = 0; expId < topK_; ++expId) {
            float expScale = topKWeightsTensor_.GetValue(tokenIdx * topK_ + expId);
            auto event = (expId % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
            auto dataInBf16 = (expId % DOUBLE_BUFFER == 0) ? dataIn0Bf16 : dataIn1Bf16;
            auto dataInFp32 = (expId % DOUBLE_BUFFER == 0) ? dataIn0Fp32 : dataIn1Fp32;
            WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            DataCopy(dataInBf16, expandedX[(tokenIdx * topK_ + expId) * k_], k_);
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            SetFlag<AscendC::HardEvent::S_V>(event);
            WaitFlag<AscendC::HardEvent::S_V>(event);
            // bf16 -> fp32
            Cast(dataInFp32, dataInBf16, AscendC::RoundMode::CAST_NONE, k_);
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
        // fp32 -> bf16
        Cast(dataResTensor_, dataResFp32Tensor_, AscendC::RoundMode::CAST_RINT, k_);
        SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID3>();
        DataCopy(output[tokenIdx * k_], dataResTensor_, k_);
    }
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
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

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Process()
{
    // 1.本卡数据处理
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    SendAndQuantBuffInit();
    SendMaskCal();               // 源卡按所有全局专家算 mask 并推送到目标专家卡
    ResetFlagList();             // 清理workSpace空间上的flag位
    QuantProcessInRank();        // 对本卡token的量化
    if constexpr(g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
    }
    CrossRankSyncInWorldSize();  // 全卡同步

    // 2.本卡专家接收数据dispatch & GroupMatmul1 & SwigluQuant
    DispatchBuffInit();
    GMMAddrInfo gmmAddrInfo;
    TupleShape initShape;
    Get<N_VALUE>(initShape) = hiddenDim_;
    Get<K_VALUE>(initShape) = k_;
    BlockOffset initOffset{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    ExpertLoopState gmm1State{initShape, initOffset};
    for (int localExpertId = 0; localExpertId < expertPerRank_; localExpertId++) {
        if constexpr(g_coreType == AIV) {
            if (subBlockIdx_ == 1) {
                SendCntCal(localExpertId); // 计算当前localExpertId接受到的token值
            }
        }
        if (!UpdateGroupParams<AddrUpdateMode::kGmm1>(gmm1State, localExpertId)) {  // 更新偏移
            continue;
        }
        UpdateGlobalBuffer<AddrUpdateMode::kGmm1>(gmmAddrInfo, gmm1State); // 更新地址
        if constexpr(g_coreType == AIV) {
            if (subBlockIdx_ == 1) {
                TripleInfoCalAndDispatch(gmmAddrInfo, localExpertId); // 三元组组装以及token dispatch
            }
        }
        if (subBlockIdx_ == 0) {
            MegaMoeImpl::GroupMatmulSwigluQuant<
                QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
                epilogueOp_, params_, gmm1State.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_);
        }
    }
    EndSync(vecSetSyncCom_);
    if constexpr(g_coreType == AIV) {
        if (subBlockIdx_ == 1) {
            ExpertTokenNumCopyOut(); // 本卡专家接受的tokenCnt总数搬出
        }
    }

    // 3. 本卡专家接收数据GroupMatmul2 & Combine
    vecSetSyncCom_ = 0;
    expertBeforeCnt_ = 0;
    ExpertLoopState gmm2State{initShape, initOffset};
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
        if (!UpdateGroupParams<AddrUpdateMode::kGmm2>(gmm2State, expertIdx)) {
            continue;
        }
        UpdateGlobalBuffer<AddrUpdateMode::kGmm2>(gmmAddrInfo, gmm2State);
        if (subBlockIdx_ == 0) {
            MegaMoeImpl::GroupMatmul2Combine<QuantOutType, QuantOutType, bfloat16_t,
                QuantScaleOutType, QuantScaleOutType>(
                params_, gmm2State.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_,
                expertBeforeCnt_, gmm2PingPongIdx_);
        }
    }
    EndGMM2Sync(vecSetSyncCom_, gmm2PingPongIdx_);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();

    // 4. 本卡数据Unpermute
    if constexpr(g_coreType == AIV) {
        UnpermuteBuffInit();
        CrossRankSyncInWorldSize(); // 全卡软同步，确认combine send完成
        Unpermute();
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

}   // namespace MegaMoeImpl
#endif
