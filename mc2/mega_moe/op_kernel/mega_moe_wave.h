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
#include "../../moe_distribute_dispatch_v2/op_kernel/quantize_functions.h"

using namespace AscendC;

namespace MegaMoeImpl {
using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
using BlockOffset = Shape<int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t>;

#define TemplateMegaMoeTypeClass typename XType, typename OutputType, typename TopkWeightsType, int32_t QuantMode
#define TemplateMegaMoeTypeFunc XType, OutputType, TopkWeightsType, QuantMode

template <TemplateMegaMoeTypeClass>
class MegaMoe {
public:
    using QuantOutType = typename std::conditional<((QuantMode >= E5M2_QUANT) && (QuantMode != E4M3_QUANT)),
        fp8_e5m2_t, fp8_e4m3fn_t>::type;
    using QuantScaleOutType = typename std::conditional<(QuantMode >= E5M2_QUANT), fp8_e8m0_t, float>::type;
    __aicore__ inline MegaMoe() {};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights,
        GM_ADDR weight1, GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
        GM_ADDR scales, GM_ADDR yOut, GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM,
        MegaMoeTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void FirstBuffInit();
    __aicore__ inline void BuffInit();
    __aicore__ inline void SecondBuffInit();
    __aicore__ inline void ResetFlagList();
    __aicore__ inline void SendMaskCal();
    __aicore__ inline void SendInfoCal(int32_t localExpertId);
    __aicore__ inline void TripleInfoCalAndDispatch(GMMAddrInfo &gmmAddrInfo, int32_t localExpertId);
    __aicore__ inline bool UpdateGroupParams(uint32_t expertIdx, uint32_t gmmIndex);
    __aicore__ inline void UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo, uint32_t gmmIndex, uint32_t localExpertId);
    __aicore__ inline void Unpermute();
    __aicore__ inline void CrossRankSyncInWorldSize();
    __aicore__ inline void ExpertTokenNumCopyOut();
    __aicore__ inline void CopyGMToGMPerToken(int32_t rowDstOffsetInCore,
        int32_t remoteRankIdx, int32_t copyStartIdx, int32_t copyNum, int32_t& pingpongId);
    __aicore__ inline void QuantProcessInRank();

    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    ParamsWave params_{};
    TupleShape problemShape_{};
    BlockOffset baseOffset_{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    GlobalTensor<int32_t> cumsumInWorkSpace_;
    GlobalTensor<int32_t> groupFlagListGm_;
    GlobalTensor<int32_t> groupFlagListGm2_;
    GlobalTensor<int32_t> groupList_;
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
    uint32_t groupListType_ = 0;
    uint64_t maxOutputSize_ = 0;
    int32_t prevSum_ = 0;
    int32_t dispatchRows_ = 0;
    int32_t vecSetSyncCom_ = 0;
    uint32_t startBlockIdx_ = 0;
    uint32_t blockNumPerRank_ = 2;
    // wave-grain dispatch-gmm1 flag:每个 expert 分 dispatchFlagSlotsPerExpert_ 个 int32 槽,
    // 每槽对应一个 L1_TILE_M=256 行的 wave;dispatch 完成本 wave 行数后 AtomicAdd 到对应槽。
    // GMM1 在每个 wave 开始时只等本 wave 槽达到 wave 实际行数,不需要全 expert 完成。
    int32_t dispatchFlagSlotsPerExpert_ = 0;
    int32_t maxWavesPerExpert_ = 0;
    uint32_t blockNum_ = GetBlockNum();
    uint32_t blockAivNum_ = GetBlockNum() * 2;
    uint32_t blockIdx_ = GetBlockIdx() / GetTaskRation();
    uint32_t aivCoreIdx_ = GetBlockIdx();
    uint32_t subBlockIdx_ = GetSubBlockIdx();
    uint32_t mxFp8ScaleNumAlignPerToken_ = 0;
    uint32_t mxFp8TokenAlignBytes_ = 0;
    uint32_t mxFp8ScaleAlignBytes_ = 0;
    uint32_t mxFp8TokenScaleAlignBytes_ = 0;
    uint32_t expertBeforeCnt_ = 0;
    uint32_t dataResAddr_ = 0;
    uint32_t dataResSize_ = 0;
    uint32_t resetTokenAddr_ = 0;
    uint32_t resetTokenSize_ = 0;
    uint32_t dataResFp32Addr_ = 0;
    uint32_t dataResFp32Size_ = 0;
    uint32_t topKWeightsAddr_ = 0;
    uint32_t topKWeightsSize_ = 0;
    uint32_t ubBufferUsedAddr_ = 0;
    uint64_t preOffset_ = 0;
    uint16_t gmm2PingPongIdx_ = 0;
    uint64_t sendTotalNum_ = 0;
    uint32_t maskAlignSize_ = 0;
    uint64_t maskWinOffset_ = 0;   // maskRecvPtr 相对 win 基址(rankSyncInWorldPtr)的偏移
    uint64_t quantWinOffset_ = 0;  // quantTokenScalePtr 相对 win 基址的偏移
    uint64_t totalSendCntInExp_ = 0;
    uint64_t cumsumRevCntInRank_ = 0;
    int32_t compareCount_ = 0;

    // 6-buffer 软流水：占满 EVENT_ID0..EVENT_ID5。
    //   - CopyGMToGMPerToken 用 copyTmpTensors_[6] 让 winmem token+scale 读
    //     在引擎里同时排队 5+ 个 (配合 prime/steady/drain 软流水)。
    // 注: topkTensors_[6] 原供旧版 SendInfoCal 跨卡读 topk 用; mask 计算前移到源卡后已不再使用,
    //     仍保留 FirstBuffInit 中的分配以维持其后续 UB 地址链不变(UB 有余量)。
    static constexpr int32_t DISPATCH_BUFFER_NUM = 6;
    LocalTensor<int32_t> topkTensors_[DISPATCH_BUFFER_NUM];
    LocalTensor<int32_t> topkIndexTensor_;
    LocalTensor<uint8_t> gatherMaskTensor_;
    LocalTensor<uint32_t> gatherMaskInt32Tensor_;
    LocalTensor<int32_t> expertTokenCntTensor_;
    LocalTensor<int32_t> validTopkIndexTensor_;
    LocalTensor<int32_t> cumsumInfoTensor_;
    LocalTensor<QuantOutType> copyTmpTensors_[DISPATCH_BUFFER_NUM];
    LocalTensor<int32_t> tripleTensor_;
    LocalTensor<XType> xInTensor1_;
    LocalTensor<XType> xInTensor2_;
    LocalTensor<QuantOutType> xOutTensor1_;
    LocalTensor<QuantOutType> xOutTensor2_;
    LocalTensor<uint16_t> mxTempTensor_;
    LocalTensor<int32_t> resetTensor_;
    LocalTensor<int32_t> topkIdsTensor_;
    LocalTensor<uint8_t> sendMaskTensor_[DOUBLE_BUFFER];  // SendMaskCal 源卡算 mask 的 ping-pong 缓冲
    LocalTensor<int32_t> expertTokenNumsOutTensor_;
    LocalTensor<bfloat16_t> dataResTensor_;
    LocalTensor<int32_t> resetTokenTensor_;
    LocalTensor<float> dataResFp32Tensor_;
    LocalTensor<float> topKWeightsTensor_;

    using BlockEpilogue = BlockEpilogueSwigluMxQuant<QuantOutType, float, QuantScaleOutType, QuantScaleOutType, true>;
    BlockEpilogue epilogueOp_;
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
    dispatchRows_ = tilingData->dispatchRows;
    groupListType_ = tilingData->groupListType;
    maxOutputSize_ = tilingData->maxOutputSize;
    // 与 WorkspaceInfoWave 构造里 flagDispatchToGmm1Ptr 的分配公式保持一致。
    maxWavesPerExpert_ = static_cast<int32_t>(Ops::Base::CeilDiv(
        static_cast<int64_t>(maxOutputSize_), DISPATCH_WAVE_TILE_M));
    dispatchFlagSlotsPerExpert_ = static_cast<int32_t>(Ops::Base::CeilAlign(
        static_cast<int64_t>(maxWavesPerExpert_), static_cast<int64_t>(INT_CACHELINE)));
    Get<N_VALUE>(problemShape_) = tilingData->hiddenDim;
    Get<K_VALUE>(problemShape_) = k_;
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
    params_.workspaceInfo = WorkspaceInfoWave(workspaceGM, tilingData);
    params_.peermemInfo = PeermemInfoWave(winRankAddr_[rankId_], tilingData);
    params_.tilingData = tilingData;
    expertTokenNumsOut_.SetGlobalBuffer((__gm__ int32_t*)params_.expertTokenNumsOutGmAddr);
    expertRevNumsGloablTensor_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.expertRevTokenNumsPtr);
    tripleGloablTensor_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.tripleInfoPtr);
    epilogueOp_.Init({params_.workspaceInfo.swigluQuantDataPtr, params_.workspaceInfo.swigluQuantScalePtr,
        params_.workspaceInfo.flagSwiGluToGmm2Ptr, nullptr, nullptr, nullptr, ALIGN_128, ALIGN_256});
    // 各 win 区相对 win 基址(rankSyncInWorldPtr)的偏移; 所有卡 win 布局一致, 跨卡读写用同一偏移。
    maskWinOffset_ = static_cast<uint64_t>(params_.peermemInfo.maskRecvPtr - params_.peermemInfo.rankSyncInWorldPtr);
    quantWinOffset_ = static_cast<uint64_t>(params_.peermemInfo.quantTokenScalePtr - params_.peermemInfo.rankSyncInWorldPtr);
    // compareCount_/maskAlignSize_ 仅依赖 sendTotalNum_, 在 Init 算好以便 SendMaskCal(同步前) 使用。
    // maskAlignSize_ 必须与 PeermemInfoWave 中 maskAlignSize 公式数值一致。
    compareCount_ = Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_256)) / sizeof(int32_t);
    maskAlignSize_ = Ops::Base::CeilAlign(static_cast<int64_t>(compareCount_) / 8, static_cast<int64_t>(ALIGN_32));
    mxFp8ScaleNumAlignPerToken_ = Ops::Base::CeilDiv(k_, static_cast<uint32_t>(ALIGN_32));
    mxFp8TokenAlignBytes_ = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_256)) * sizeof(QuantOutType);
    mxFp8ScaleAlignBytes_ = mxFp8ScaleNumAlignPerToken_ * sizeof(uint8_t);
    mxFp8TokenScaleAlignBytes_ = Ops::Base::CeilAlign(mxFp8TokenAlignBytes_ + mxFp8ScaleAlignBytes_,
        static_cast<uint32_t>(ALIGN_32));
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::FirstBuffInit()
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    // compareCount_ / maskAlignSize_ 已在 Init 算好(SendMaskCal 同步前要用), 这里直接使用。
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
    uint32_t gatherMaskTensorSize = maskAlignSize_ * worldSize_;
    gatherMaskTensor_ = LocalTensor<uint8_t>(TPosition::VECCALC, gatherMaskTensorAddr,
        gatherMaskTensorSize / sizeof(uint8_t));
    gatherMaskInt32Tensor_ = LocalTensor<uint32_t>(TPosition::VECCALC, gatherMaskTensorAddr,
        gatherMaskTensorSize / sizeof(uint32_t));
    // 6 个 topkTensor buffer 给 SendInfoCal 用,跨卡 MTE2 同时排队多个读。
    uint32_t topkTensorBaseAddr = gatherMaskTensorAddr + gatherMaskTensorSize;
    uint32_t topkTensorPerBufSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_256));
    for (int32_t b = 0; b < DISPATCH_BUFFER_NUM; ++b) {
        topkTensors_[b] = LocalTensor<int32_t>(TPosition::VECCALC,
            topkTensorBaseAddr + static_cast<uint32_t>(b) * topkTensorPerBufSize,
            topkTensorPerBufSize / sizeof(int32_t));
    }
    uint32_t validTopkIndexTensorAddr = topkTensorBaseAddr +
        static_cast<uint32_t>(DISPATCH_BUFFER_NUM) * topkTensorPerBufSize;
    uint32_t validTopkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    validTopkIndexTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, validTopkIndexTensorAddr,
        validTopkIndexTensorSize / sizeof(int32_t));
    uint32_t topkIndexTensorAddr = validTopkIndexTensorAddr + validTopkIndexTensorSize;
    uint32_t topkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    topkIndexTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, topkIndexTensorAddr,
        topkIndexTensorSize / sizeof(int32_t));
    // 6 个 dispatch buffer,配合 EVENT_ID0..EVENT_ID5 做软流水。每槽 TOKEN_SCALE_SIZE=8KB,
    // 足以容纳一个 token+scale。6 × 8KB = 48KB,FirstBuffInit 总用量仍远低于 256KB UB 上限。
    constexpr uint32_t COPY_TMP_BUFFER_SIZE = TOKEN_SCALE_SIZE;
    uint32_t copyTmpBaseAddr = topkIndexTensorAddr + topkIndexTensorSize;
    for (int32_t b = 0; b < DISPATCH_BUFFER_NUM; ++b) {
        copyTmpTensors_[b] = LocalTensor<QuantOutType>(TPosition::VECCALC,
            copyTmpBaseAddr + static_cast<uint32_t>(b) * COPY_TMP_BUFFER_SIZE,
            COPY_TMP_BUFFER_SIZE / sizeof(QuantOutType));
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
    for (int32_t b = 0; b < DISPATCH_BUFFER_NUM; ++b) {
        Duplicate<int32_t>(topkTensors_[b], 0, (topkTensorPerBufSize / sizeof(int32_t)));
    }
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::BuffInit()
{
    if constexpr(g_coreType == AIC) {
        return;
    }

    uint32_t topkIdsTensorAddr = 0;
    // SendMaskCal 的 CompareScalar 处理 compareCount_ 个 int(256B 对齐), 故 topk 缓冲按 256B 对齐分配,
    // 尾部 [sendTotalNum_, compareCount_) 在 SendMaskCal 里 Duplicate 清 0。
    uint32_t topkIdsTensorSize = Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_256));
    topkIdsTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, topkIdsTensorAddr, topkIdsTensorSize / sizeof(int32_t));
    uint32_t resetTensorAddr = topkIdsTensorAddr + topkIdsTensorSize;
    // 总 int32 flag entries: SwiGluToGmm2 区 (expertPerRank * INT_CACHELINE)
    // + DispatchToGmm1 区 wave-grain (expertPerRank * dispatchFlagSlotsPerExpert_).
    uint64_t totalFlagInt32 = static_cast<uint64_t>(expertPerRank_) *
        (static_cast<uint64_t>(INT_CACHELINE) + static_cast<uint64_t>(dispatchFlagSlotsPerExpert_));
    uint32_t resetNumPerCore = Ops::Base::CeilDiv(totalFlagInt32, static_cast<uint64_t>(blockAivNum_));
    uint32_t resetTensorSize = Ops::Base::CeilAlign(static_cast<uint64_t>(resetNumPerCore),
        static_cast<uint64_t>(INT32_PER_256B)) * sizeof(int32_t);
    resetTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, resetTensorAddr, resetTensorSize / sizeof(int32_t));
    uint32_t mxTempTensorAddr = resetTensorAddr + resetTensorSize;
    uint32_t mxTempTensorSize = 2 * 1024;
    mxTempTensor_ = LocalTensor<uint16_t>(TPosition::VECCALC, mxTempTensorAddr, mxTempTensorSize / sizeof(uint16_t));
    uint32_t xOutTensorAddr1 = mxTempTensorAddr + mxTempTensorSize;
    uint32_t xOutTensorSize1 = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_256)) +
        Ops::Base::CeilDiv(k_, static_cast<uint32_t>(ALIGN_32));
    xOutTensor1_ = LocalTensor<QuantOutType>(TPosition::VECCALC, xOutTensorAddr1,
        xOutTensorSize1 / sizeof(QuantOutType));
    uint32_t xOutTensorAddr2 = xOutTensorAddr1 + xOutTensorSize1;
    uint32_t xOutTensorSize2 = xOutTensorSize1;
    xOutTensor2_ = LocalTensor<QuantOutType>(TPosition::VECCALC, xOutTensorAddr2,
        xOutTensorSize2 / sizeof(QuantOutType));
    uint32_t xInAlignAddr1 = xOutTensorAddr2 + xOutTensorSize2;
    uint32_t xInAlignSize1 = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_128)) * sizeof(XType);
    xInTensor1_ = LocalTensor<XType>(TPosition::VECCALC, xInAlignAddr1, xInAlignSize1 / sizeof(XType));
    uint32_t xInAlignAddr2 = xInAlignAddr1 + xInAlignSize1;
    uint32_t xInAlignSize2 = xInAlignSize1;
    xInTensor2_ = LocalTensor<XType>(TPosition::VECCALC, xInAlignAddr2, xInAlignSize2 / sizeof(XType));
    // SendMaskCal 源卡算 mask 的 ping-pong 缓冲(每个 maskAlignSize_ 字节, 配 V->MTE3 软流水)。
    // 与 topkIdsTensor_ / 量化缓冲不重叠, 同处 BuffInit 阶段共存。
    uint32_t sendMaskAddr = xInAlignAddr2 + xInAlignSize2;
    for (int32_t b = 0; b < DOUBLE_BUFFER; ++b) {
        sendMaskTensor_[b] = LocalTensor<uint8_t>(TPosition::VECCALC,
            sendMaskAddr + static_cast<uint32_t>(b) * maskAlignSize_, maskAlignSize_);
    }
    Duplicate<int32_t>(resetTensor_, 0, (resetTensorSize / sizeof(int32_t)));
}

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

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::SendMaskCal()
{
    // ============================================================================
    // 源卡侧:对本卡 topk 按通信域内【所有】全局专家号算 bit-packed mask, 只把 mask
    // 推送给对应的目标专家卡(替代旧的 TopKIdsAllGather 推送整份 topk)。
    //   - 扁平分核(Option B): 核 aivCoreIdx_ 处理全局专家 g ∈ {aivCoreIdx_, +blockAivNum_, ...},
    //     所有核参与运算, 负载均衡。
    //   - 目标卡 win 内 mask 布局 [localExpert][srcRank], 单份 maskAlignSize_ 字节:
    //     offset = maskWinOffset_ + (le*worldSize_ + rankId_) * maskAlignSize_。
    //   - 因推送 rank Y 的核与 CrossRankSyncInWorldSize 中 signal rank Y 的核不再同一个,
    //     Process 在跨卡 signal 前加了 PipeBarrier<PIPE_ALL>+SyncAll 保证可见性。
    // ============================================================================
    if constexpr(g_coreType == AIC) {
        return;
    }

    // ----- 1. 加载本卡 topk 到 topkIdsTensor_ (compareCount_ 个 int, 尾部补 0) -----
    GlobalTensor<int32_t> srcGlobalTensor;
    srcGlobalTensor.SetGlobalBuffer((__gm__ int32_t*)params_.expertIdxGmAddr);
    Duplicate<int32_t>(topkIdsTensor_, 0, compareCount_);
    SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID1>();
    DataCopyExtParams loadParams{1U, static_cast<uint32_t>(sendTotalNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> loadPad{false, 0U, 0U, 0U};
    DataCopyPad(topkIdsTensor_, srcGlobalTensor, loadParams, loadPad);
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();

    // ----- 2. 扁平分核, 逐个全局专家算 mask 推送 (ping-pong 双 buffer 软流水) -----
    constexpr TEventID kBufEvents[DOUBLE_BUFFER] = {EVENT_ID0, EVENT_ID1};
    for (int32_t b = 0; b < static_cast<int32_t>(DOUBLE_BUFFER); ++b) {
        SetFlag<AscendC::HardEvent::MTE3_V>(kBufEvents[b]);
    }
    int32_t totalExperts = static_cast<int32_t>(worldSize_ * expertPerRank_);
    DataCopyExtParams maskCopyParams{1U, static_cast<uint32_t>(maskAlignSize_), 0U, 0U, 0U};
    int32_t iter = 0;
    GlobalTensor<uint8_t> dstGlobalTensor;
    for (int32_t g = aivCoreIdx_; g < totalExperts; g += blockAivNum_, ++iter) {
        int32_t dstRank = g / static_cast<int32_t>(expertPerRank_);
        int32_t le = g % static_cast<int32_t>(expertPerRank_);
        TEventID ev = kBufEvents[iter % static_cast<int32_t>(DOUBLE_BUFFER)];
        LocalTensor<uint8_t> maskBuf = sendMaskTensor_[iter % static_cast<int32_t>(DOUBLE_BUFFER)];

        WaitFlag<AscendC::HardEvent::MTE3_V>(ev);  // 等本 buffer 上一轮 MTE3 推送完成
        CompareScalar(maskBuf, topkIdsTensor_, g, AscendC::CMPMODE::EQ, compareCount_);
        SetFlag<AscendC::HardEvent::V_MTE3>(ev);
        WaitFlag<AscendC::HardEvent::V_MTE3>(ev);

        uint64_t dstOffset = maskWinOffset_ +
            static_cast<uint64_t>(le * static_cast<int32_t>(worldSize_) + static_cast<int32_t>(rankId_)) *
            static_cast<uint64_t>(maskAlignSize_);
        dstGlobalTensor.SetGlobalBuffer(
            (__gm__ uint8_t*)GetRankWinAddrWithOffset(dstRank, dstOffset));
        DataCopyPad(dstGlobalTensor, maskBuf, maskCopyParams);
        SetFlag<AscendC::HardEvent::MTE3_V>(ev);
    }
    for (int32_t b = 0; b < static_cast<int32_t>(DOUBLE_BUFFER); ++b) {
        WaitFlag<AscendC::HardEvent::MTE3_V>(kBufEvents[b]);
    }
}

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
    GlobalTensor<XType> srcGlobalTensor;
    GlobalTensor<uint8_t> dstGlobalTensor;
    DataCopyParams xCopyInParams = {1U, static_cast<uint16_t>(H * sizeof(XType)), 0U, 0U};
    DataCopyPadParams xCopyInPadParams{true, 0, 0, 0};
    DataCopyExtParams xCopyOutParams = {1U,
        static_cast<uint32_t>(mxFp8TokenAlignBytes_ + mxFp8ScaleAlignBytes_), 0U, 0U, 0U};
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int32_t index = 0; index < curentNum; index++) {
        srcGlobalTensor.SetGlobalBuffer((__gm__ XType*)(params_.aGmAddr + (curentOffset + index) * H * sizeof(XType)));
        dstGlobalTensor.SetGlobalBuffer((__gm__ uint8_t*)(params_.peermemInfo.quantTokenScalePtr +
        (curentOffset + index) * mxFp8TokenScaleAlignBytes_));
        auto event = (index % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
        auto xInTensor = (index % DOUBLE_BUFFER == 0) ? xInTensor1_ : xInTensor2_;
        auto xOutTensor = (index % DOUBLE_BUFFER == 0) ? xOutTensor1_ : xOutTensor2_;
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        DataCopyPad(xInTensor, srcGlobalTensor, xCopyInParams, xCopyInPadParams);
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        __ubuf__ XType* srcAddr = (__ubuf__ XType*)xInTensor.GetPhyAddr();
        __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)mxTempTensor_.GetPhyAddr();
        __ubuf__ uint16_t* halfScaleAddr = (__ubuf__ uint16_t*)mxTempTensor_[Ops::Base::CeilAlign(
        mxFp8ScaleNumAlignPerToken_, static_cast<uint32_t>(ALIGN_32))].GetPhyAddr();
        __ubuf__ int8_t* outDataAddr = (__ubuf__ int8_t*)xOutTensor.GetPhyAddr();
        __ubuf__ uint16_t* mxScaleAddr = (__ubuf__ uint16_t*)xOutTensor[Ops::Base::CeilAlign(
        H, static_cast<uint32_t>(ALIGN_256))].GetPhyAddr();

        quant::ComputeMaxExp(srcAddr, maxExpAddr, H); // 计算最大Exp
        quant::ComputeScale<QuantOutType>(maxExpAddr, mxScaleAddr, halfScaleAddr,
        mxFp8ScaleNumAlignPerToken_); // 计算scales并填充f
        quant::ComputeFp8Data<XType, QuantOutType, AscendC::RoundMode::CAST_TRUNC,
            AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleAddr, outDataAddr, H);
        SetFlag<AscendC::HardEvent::V_MTE3>(event);
        WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        auto xOutBytesTensor = xOutTensor.template ReinterpretCast<uint8_t>();
        DataCopyPad(dstGlobalTensor, xOutBytesTensor, xCopyOutParams);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::SendInfoCal(int32_t localExpertId)
{
    // ============================================================================
    // 目标卡侧"读 mask 计数"。各源卡为本 localExpert 算好的 bit-packed mask 已由
    // SendMaskCal 推送到本卡 win(布局 [localExpert][srcRank]); 全卡同步后这里:
    //   1. 一次性从本卡 win 把本 localExpert 的 worldSize_ 份 mask 读进 gatherMaskTensor_;
    //   2. 逐源卡 GatherMask 计数 → 累加 totalSendCntInExp_/cumsumRevCntInRank_, 写 cumsumInfoTensor_;
    //   3. 写 expertRevNumsGloablTensor_ + CrossCoreSetFlag 通知 AIC。
    // 不再跨卡读 topk、不再 CompareScalar(已前移到源卡 SendMaskCal)。
    // gatherMaskTensor_ 在本函数填好后, 同一专家迭代里被 TripleInfoCalAndDispatch 复用。
    // ============================================================================
    totalSendCntInExp_ = 0;
    uint32_t mask = sendTotalNum_;

    // ----- 1. 从本卡 win 读本 localExpert 的全部 worldSize_ 份 mask -----
    GlobalTensor<uint8_t> maskSrcGlobal;
    maskSrcGlobal.SetGlobalBuffer((__gm__ uint8_t*)(params_.peermemInfo.maskRecvPtr +
        static_cast<uint64_t>(localExpertId) * worldSize_ * maskAlignSize_));
    DataCopy(gatherMaskTensor_, maskSrcGlobal, worldSize_ * maskAlignSize_);
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();

    // ----- 2. 逐源卡 GatherMask 计数 + cumsum -----
    for (int32_t calRankId = 0; calRankId < static_cast<int32_t>(worldSize_); ++calRankId) {
        uint64_t sendCnt = 0;
        GatherMask(validTopkIndexTensor_, topkIndexTensor_,
            gatherMaskInt32Tensor_[calRankId * (maskAlignSize_ / sizeof(uint32_t))],
            true, mask, {1, 1, 0, 0}, sendCnt);
        SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
        totalSendCntInExp_ += sendCnt;
        cumsumRevCntInRank_ += sendCnt;
        cumsumInfoTensor_.SetValue(localExpertId * worldSize_ + calRankId, static_cast<int32_t>(cumsumRevCntInRank_));
    }

    expertTokenCntTensor_.SetValue(0, totalSendCntInExp_);
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    DataCopy<int32_t>(expertRevNumsGloablTensor_[localExpertId * INT32_PER_256B * aicNum_ + INT32_PER_256B * blockIdx_],
        expertTokenCntTensor_, INT32_PER_256B);
    PipeBarrier<PIPE_ALL>();
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_MTE3>(2);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::CopyGMToGMPerToken(
    int32_t rowDstOffsetInCore, int32_t remoteRankIdx, int32_t copyStartIdx, int32_t copyNum, int32_t& pingpongId)
{
    // ============================================================================
    // 6-buffer 软流水版 CopyGMToGMPerToken
    // ----------------------------------------------------------------------------
    // 旧版（pingpong, BufferNum=2）的 loop 结构是：
    //   WaitFlag<MTE3_MTE2>(ev) → DataCopy MTE2 → SetFlag<MTE2_MTE3>(ev)
    //   → WaitFlag<MTE2_MTE3>(ev) ★scalar 在此阻塞,等本槽 MTE2 完成★
    //   → DataCopyPad MTE3 × 2 → SetFlag<MTE3_MTE2>(ev)
    // 由于 scalar 必须等本槽 MTE2 完成才能继续,跨卡 winmem 读在 MTE2 引擎上
    // 始终只有 1 个在途,HCCL/RDMA 延迟无法靠并发隐藏。
    //
    // 新版把整个循环重排成 prime → steady → drain:
    //   Phase 1: 一次性把所有 token 的 (rank, tokenIndex, topkPos) 写到 tripleTensor_,
    //            Phase 2 之后只读 tripleTensor_,不再算 topkIndex/tokenIndex。
    //   Phase 2 prime: 连发 6 个 MTE2,中间不插 WaitFlag<MTE2_MTE3>,
    //                  让 MTE2 引擎同时持有 6 个跨卡读请求。
    //   Phase 2 steady: 每轮做 (MTE3_out[i] + MTE2_in[i+6]),槽位循环复用。
    //   Phase 2 drain:  收尾不再发新 MTE2,只等 MTE3。
    //
    // 同一 expert 内多个 rank 的 CopyGMToGMPerToken 串行调用,每次调用都自包含
    // (prime + drain),drain 是 6 次 WaitFlag<MTE3_MTE2>,会丢失跨 rank 调用的
    // 流水机会但保证调用间清洁状态。
    // ============================================================================
    constexpr int32_t BufferNum = DISPATCH_BUFFER_NUM;
    constexpr TEventID kBufEvents[BufferNum] = {
        EVENT_ID0, EVENT_ID1, EVENT_ID2, EVENT_ID3, EVENT_ID4, EVENT_ID5
    };
    (void)pingpongId;   // 自包含 flag 管理,不再依赖跨调用 pingpongId

    int64_t widthA = Get<K_VALUE>(problemShape_); // 输出单 token 长度,紧密排列
    int64_t widthAScale = Ops::Base::CeilDiv(static_cast<int64_t>(widthA), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE;  // 输出 token-scale 长度,紧密排列
    uint32_t copyInNum = Ops::Base::CeilAlign(static_cast<int64_t>(mxFp8TokenAlignBytes_ + mxFp8ScaleAlignBytes_),
        static_cast<int64_t>(ALIGN_32)); // 输入 token-scale 拼接,非紧密排列
    GlobalTensor<QuantOutType> remoteRankGlobalTensor;
    GlobalTensor<QuantOutType> tokenRevGlobalTensor;
    GlobalTensor<QuantScaleOutType> scaleRevGlobalTensor;
    tokenRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(
        params_.workspaceInfo.dispatchRevDataPtr + rowDstOffsetInCore * widthA));
    scaleRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleOutType *>(
        params_.workspaceInfo.dispatchRevScalePtr + rowDstOffsetInCore * widthAScale));
    // 远端源卡量化 token 在其 win 内的偏移; 各卡 win 布局一致, 用统一的 quantWinOffset_
    // (mask 区缩小后该偏移随之变化, 故不再硬编码 worldSize_*topkOffsetPerRank_)。
    remoteRankGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(
        GetRankWinAddrWithOffset(remoteRankIdx, quantWinOffset_)));

    if (copyNum <= 0) {
        return;
    }

    // 预置 6 个 MTE3_MTE2 flag,Phase 2 的 prime/steady 的 WaitFlag 可立即通过。
    for (int32_t b = 0; b < BufferNum; ++b) {
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(kBufEvents[b]);
    }
    PipeBarrier<PIPE_ALL>();

    // ============================================================
    // Phase 1: 写满 tripleTensor_。
    //   tripleTensor_[i*8 + RANK_ID]   = remoteRankIdx
    //   tripleTensor_[i*8 + TOKEN_ID]  = topkIndex / topK_
    //   tripleTensor_[i*8 + TOPK_INDEX]= topkIndex % topK_
    // Phase 2 之后只通过 GetValue 读 tripleTensor_,不再访问 validTopkIndex。
    // ============================================================
    for (int32_t i = 0; i < copyNum; ++i) {
        int32_t topkIndex = validTopkIndexTensor_.GetValue(copyStartIdx + i);
        int32_t tokenIndex = topkIndex / topK_;
        tripleTensor_[i * INT32_PER_256B].SetValue(RANK_ID, remoteRankIdx);
        tripleTensor_[i * INT32_PER_256B].SetValue(TOKEN_ID, tokenIndex);
        tripleTensor_[i * INT32_PER_256B].SetValue(TOPK_INDEX, topkIndex % topK_);
    }

    // ============================================================
    // Phase 2: 6-buffer SW 流水 MTE2 + MTE3
    // ============================================================

    // ----- Prime: 发出前 BufferNum 个 token 的 MTE2 -----
    int32_t primeCount = (copyNum < BufferNum) ? copyNum : BufferNum;
    for (int32_t p = 0; p < primeCount; ++p) {
        int32_t tokenIndex_p = tripleTensor_[p * INT32_PER_256B].GetValue(TOKEN_ID);
        uint64_t remoteCopyOffset_p = static_cast<uint64_t>(tokenIndex_p) * static_cast<uint64_t>(copyInNum);
        TEventID ev = kBufEvents[p];
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ev);
        DataCopy(copyTmpTensors_[p], remoteRankGlobalTensor[remoteCopyOffset_p], copyInNum);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(ev);
    }

    // ----- Steady: MTE3[i_out] + issueMTE2[i_out + BufferNum] -----
    for (int32_t i_out = 0; i_out < copyNum; ++i_out) {
        int32_t buf_out = i_out % BufferNum;
        TEventID ev_out = kBufEvents[buf_out];

        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(ev_out);

        LocalTensor<QuantOutType> tokenScalebuf = copyTmpTensors_[buf_out];
        LocalTensor<QuantScaleOutType> bufScale =
            tokenScalebuf[mxFp8TokenAlignBytes_].template ReinterpretCast<QuantScaleOutType>();
        DataCopyPad(tokenRevGlobalTensor[i_out * widthA], tokenScalebuf,
            {1, static_cast<uint16_t>(widthA * sizeof(QuantOutType)), 0U, 0U, 0U});
        DataCopyPad(scaleRevGlobalTensor[i_out * widthAScale], bufScale,
            {1, static_cast<uint16_t>(widthAScale * sizeof(QuantScaleOutType)), 0U, 0U, 0U});
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(ev_out);

        // 发下一个槽的 MTE2 (i_out + BufferNum,复用 buf_out 槽)
        int32_t i_in_next = i_out + BufferNum;
        if (i_in_next < copyNum) {
            int32_t tokenIndex_n = tripleTensor_[i_in_next * INT32_PER_256B].GetValue(TOKEN_ID);
            uint64_t remoteCopyOffset_n = static_cast<uint64_t>(tokenIndex_n) * static_cast<uint64_t>(copyInNum);
            // WaitFlag 此处等待的是本轮刚发出的 SetFlag<MTE3_MTE2>(ev_out),即等本槽 MTE3 完成。
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ev_out);
            DataCopy(copyTmpTensors_[buf_out], remoteRankGlobalTensor[remoteCopyOffset_n], copyInNum);
            SetFlag<AscendC::HardEvent::MTE2_MTE3>(ev_out);
        }
    }

    // ----- Drain: 6 个 buffer-free flag -----
    for (int32_t b = 0; b < BufferNum; ++b) {
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kBufEvents[b]);
    }

    // ============================================================
    // Phase 3: triple 三元组搬出
    // ============================================================
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();
    DataCopy(tripleGloablTensor_[rowDstOffsetInCore * INT32_PER_256B], tripleTensor_, copyNum * INT32_PER_256B);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::TripleInfoCalAndDispatch(
    GMMAddrInfo &gmmAddrInfo, int32_t localExpertId)
{
    // wave-grain dispatch-gmm1 flag:每个 expert 有 maxWavesPerExpert_ 个槽位,
    // 每槽对应一个 L1_TILE_M 行的 wave。dispatch 完成的行数按 wave 分桶 AtomicAdd
    // 到 groupFlagListGm2_[waveIdx]。GMM1 在每个 wave 启动时只等本 wave 槽达到 wave 实际
    // 行数,不需要全 expert 完成才能启动 → GMM1 提前启动。
    //
    // 注 1: pingpongId 保留用于 ABI 兼容,CopyGMToGMPerToken 自包含 flag 管理,实际不用。
    // 注 2: rowDstOffsetInCore 是【跨 expert 累加】的全局偏移(含前置 expert 的行数);
    //       而 groupFlagListGm2_ 已经指向当前 expert 的槽起点 + GMM1 端 mLoc 是【expert 内 local】
    //       行偏移,所以这里算 waveIdx 必须减去前置 expert 的累加,才能跟 GMM1 对齐。
    int32_t pingpongId = 0;
    constexpr int32_t L1_TILE_M_I32 = static_cast<int32_t>(MegaMoeImpl::L1_TILE_M);
    // priorExpertCumsum = 前面所有 expert 在本卡的总 token 数 = 当前 expert 在 dispatchRevData
    // 里的起始 row。等价于 cumsumInfoTensor_[localExpertId * worldSize_ - 1]。
    int32_t priorExpertCumsum = (localExpertId == 0) ? 0 :
        cumsumInfoTensor_.GetValue(localExpertId * worldSize_ - 1);

    for (uint32_t idx = blockIdx_; idx < worldSize_ * blockNumPerRank_; idx += blockNum_) {
        uint32_t dstRankIdx = idx / blockNumPerRank_;
        uint64_t sendToCurExpTokenCnt = 0;
        GatherMask(validTopkIndexTensor_, topkIndexTensor_,
            gatherMaskInt32Tensor_[dstRankIdx * (maskAlignSize_ / sizeof(uint32_t))],
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
                CopyGMToGMPerToken(rowDstOffsetInCore, dstRankIdx, rowSrcIdxInCore, rowsNumInCore, pingpongId);
                rowsThisCore = rowsNumInCore;
            }
        }
        // wave-grain AtomicAdd:本次 CopyGMToGMPerToken 写到了【expert 内】行区间
        // [rowStartLocal, rowEndLocal) = [rowDstOffsetInCore - priorExpertCumsum, +rowsThisCore)。
        // 按 wave (L1_TILE_M 行) 边界拆分,对每个跨越的 wave 槽位 AtomicAdd 该 wave 内实际行数。
        if (rowsThisCore > 0) {
            SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID5>();
            int32_t rowStartLocal = rowDstOffsetInCore - priorExpertCumsum;
            int32_t rowEndLocal = rowStartLocal + rowsThisCore;
            int32_t waveLo = rowStartLocal / L1_TILE_M_I32;
            int32_t waveHi = (rowEndLocal - 1) / L1_TILE_M_I32;
            __gm__ int32_t* flagBase = (__gm__ int32_t*)(groupFlagListGm2_.GetPhyAddr());
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

template <TemplateMegaMoeTypeClass>
__aicore__ inline bool MegaMoe<TemplateMegaMoeTypeFunc>::UpdateGroupParams(uint32_t expertIdx, uint32_t gmmIndex)
{
    if (expertIdx != 0) {
        uint64_t m = Get<M_VALUE>(problemShape_);
        uint64_t n = Get<N_VALUE>(problemShape_);
        uint64_t k = Get<K_VALUE>(problemShape_);
        expertBeforeCnt_ += m;
        Get<IDX_A_OFFSET>(baseOffset_) += m * k;
        Get<IDX_B_OFFSET>(baseOffset_) += n * k;
        // only splitM
        auto scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_A_SCALE_OFFSET>(baseOffset_) += m * scaleK;
        Get<IDX_B_SCALE_OFFSET>(baseOffset_) += n * scaleK;
        Get<IDX_C_OFFSET>(baseOffset_) += m * n / SWIGLU_N_HALF;
        Get<IDX_C_SCALE_OFFSET>(baseOffset_) +=
            m * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_FLAG_OFFSET>(baseOffset_) += 1;
        Get<IDX_B2_OFFSET>(baseOffset_) += k * n / SWIGLU_N_HALF;
        Get<IDX_B2_SCALE_OFFSET>(baseOffset_) +=
            k * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_Y2_OFFSET>(baseOffset_) += m * k;
        Get<IDX_M_OFFSET>(baseOffset_) += m;
    }

    if (gmmIndex == 0) {
        if constexpr (g_coreType == AscendC::AIC) {
            CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_S>(2 + 16);
            uint64_t offsetInCnt = expertIdx * 8 * aicNum_ + 8 * blockIdx_;
            DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
                expertRevNumsGloablTensor_[offsetInCnt]);
            Get<M_VALUE>(problemShape_) = expertRevNumsGloablTensor_.GetValue(offsetInCnt);
            CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_S>(3);
        }
        if constexpr (g_coreType == AscendC::AIV) {
            if (subBlockIdx_ == 0) {
                CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_S>(3);
                uint64_t offsetInCnt = expertIdx * 8 * aicNum_ + 8 * blockIdx_;
                DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
                expertRevNumsGloablTensor_[offsetInCnt]);
                Get<M_VALUE>(problemShape_) = expertRevNumsGloablTensor_.GetValue(offsetInCnt);
            } else {
                Get<M_VALUE>(problemShape_) = totalSendCntInExp_;
            }
        }
    } else {
        uint64_t offsetInCnt = expertIdx * 8 * aicNum_ + 8 * blockIdx_;
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
                expertRevNumsGloablTensor_[offsetInCnt]);
        Get<M_VALUE>(problemShape_) = expertRevNumsGloablTensor_.GetValue(offsetInCnt);
    }

    if (Get<M_VALUE>(problemShape_) == 0) {
        return false;
    }
    return true;
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo, uint32_t gmmIndex,
    uint32_t localExpertId)
{
    if (gmmIndex == 0) {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.dispatchRevDataPtr +
            Get<IDX_A_OFFSET>(baseOffset_) * sizeof(QuantOutType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.dispatchRevScalePtr +
        Get<IDX_A_SCALE_OFFSET>(baseOffset_) * sizeof(QuantScaleOutType);
        if constexpr(g_coreType == AIC) {
            gmmAddrInfo.bGlobal = params_.bGmAddr + Get<IDX_B_OFFSET>(baseOffset_) * sizeof(QuantOutType);
            gmmAddrInfo.bScaleGlobal = params_.bScaleGmAddr + Get<IDX_B_SCALE_OFFSET>(baseOffset_) *
                                        sizeof(QuantScaleOutType);
        } else {
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                Get<IDX_C_OFFSET>(baseOffset_), Get<IDX_C_SCALE_OFFSET>(baseOffset_),
                Get<IDX_FLAG_OFFSET>(baseOffset_), 0L, 0L, 0L};
            epilogueOp_.UpdateGlobalAddr(vecBaseOffset);
        }
    } else if constexpr(g_coreType == AIC) {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.swigluQuantDataPtr +
            Get<IDX_C_OFFSET>(baseOffset_) * sizeof(QuantOutType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.swigluQuantScalePtr + Get<IDX_C_SCALE_OFFSET>(baseOffset_) *
                                    sizeof(QuantScaleOutType);
        gmmAddrInfo.bGlobal = params_.b2GmAddr + Get<IDX_B2_OFFSET>(baseOffset_) * sizeof(QuantOutType);
        gmmAddrInfo.bScaleGlobal = params_.b2ScaleGmAddr + Get<IDX_B2_SCALE_OFFSET>(baseOffset_) *
                                    sizeof(QuantScaleOutType);
    }
    gmmAddrInfo.groupFlagList = (__gm__ int32_t*)params_.workspaceInfo.flagSwiGluToGmm2Ptr +
                                Get<IDX_FLAG_OFFSET>(baseOffset_) * INT_CACHELINE;
    // wave-grain dispatch-gmm1 flag: per-expert 步长是 dispatchFlagSlotsPerExpert_,而不是 INT_CACHELINE。
    gmmAddrInfo.groupFlagList2 = (__gm__ int32_t*)params_.workspaceInfo.flagDispatchToGmm1Ptr +
                                Get<IDX_FLAG_OFFSET>(baseOffset_) * dispatchFlagSlotsPerExpert_;
    groupFlagListGm_.SetGlobalBuffer(gmmAddrInfo.groupFlagList);
    groupFlagListGm2_.SetGlobalBuffer(gmmAddrInfo.groupFlagList2);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::SecondBuffInit()
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

    dataResAddr_ = 0;
    dataResSize_ = dataResBufAlign / sizeof(bfloat16_t);
    dataResTensor_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, dataResAddr_, dataResSize_);
    resetTokenAddr_ = dataResAddr_ + dataResBufAlign;
    resetTokenSize_ = num / sizeof(int32_t);
    resetTokenTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, resetTokenAddr_, resetTokenSize_);
    dataResFp32Addr_ = resetTokenAddr_ + num;
    dataResFp32Size_ = dataResFp32BufAlign / sizeof(float);
    dataResFp32Tensor_ = LocalTensor<float>(TPosition::VECCALC, dataResFp32Addr_, dataResFp32Size_);
    topKWeightsAddr_ = dataResFp32Addr_ + dataResFp32BufAlign;
    topKWeightsSize_ = topKWeightsBufAlign / sizeof(float);
    topKWeightsTensor_ = LocalTensor<float>(TPosition::VECCALC, topKWeightsAddr_, topKWeightsSize_);
    uint32_t tempAddr = topKWeightsAddr_ + topKWeightsBufAlign;
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
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    BuffInit();
    SendMaskCal();   // 源卡按所有全局专家算 mask 并推送到目标专家卡(替代 TopKIdsAllGather)
    ResetFlagList();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    QuantProcessInRank();

    // 2. 全卡同步
    // Option B: SendMaskCal 扁平分核后, 推送 rank Y 的核与 CrossRankSyncInWorldSize 中 signal rank Y
    // 的核不再同一个, 同核 push+signal 顺序保证失效。这里先做一次本地 barrier+SyncAll, 确保所有核的
    // mask 推送(及 Quant 写本卡 win)都已 issue/drain, 远端可见后再做跨卡 signal 握手。
    if constexpr(g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
    }
    CrossRankSyncInWorldSize();

    // 3. dispatch & GroupMatmulSwigluQuant
    FirstBuffInit();
    GMMAddrInfo gmmAddrInfo;
    for (int localExpertId = 0; localExpertId < expertPerRank_; localExpertId++) {
        if constexpr(g_coreType == AIV) {
            if (subBlockIdx_ == 1) {
                SendInfoCal(localExpertId);
            }
        }
        if (!UpdateGroupParams(localExpertId, 0)) {
            continue;
        }
        UpdateGlobalBuffer(gmmAddrInfo, 0, localExpertId);
        if constexpr(g_coreType == AIV) {
            if (subBlockIdx_ == 1) {
                TripleInfoCalAndDispatch(gmmAddrInfo, localExpertId);
            }
        }
        if (subBlockIdx_ == 0) {
            MegaMoeImpl::GroupMatmulSwigluQuant<
                QuantOutType, QuantOutType, float, QuantScaleOutType, QuantScaleOutType>(
                epilogueOp_, params_, problemShape_, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_);
        }
    }
    EndSync(vecSetSyncCom_);
    if constexpr(g_coreType == AIV) {
        if (subBlockIdx_ == 1) {
            ExpertTokenNumCopyOut();
        }
    }

    // 5. for循环按照本卡专家id进行gmm2 & combine，内部gmm2与combine之间按照基本块进行流水
    preOffset_ = 0;
    vecSetSyncCom_ = 0;
    expertBeforeCnt_ = 0;
    baseOffset_ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
        if (!UpdateGroupParams(expertIdx, 1)) {
            continue;
        }
        UpdateGlobalBuffer(gmmAddrInfo, 1, expertIdx);
        if (subBlockIdx_ == 0) {
            MegaMoeImpl::GroupMatmul2Wave<QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                QuantScaleOutType>(
                params_, problemShape_, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_, expertBeforeCnt_, expertIdx,
                gmm2PingPongIdx_,
                rankId_);
        }
    }
    EndGMM2Sync(vecSetSyncCom_, gmm2PingPongIdx_);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
    // 6. unpermute & 清理win区上放置token的位置
    if constexpr(g_coreType == AIV) {
        SecondBuffInit();
        CrossRankSyncInWorldSize(); // 全卡软同步，确认combine send完成
        Unpermute();
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

}   // namespace MegaMoeImpl
#endif
