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

#ifndef MEGA_MOE_LAYERED_H
#define MEGA_MOE_LAYERED_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#include "kernel_operator_list_tensor_intf.h"
#include "mega_moe_base.h"
#include "mega_moe_workspace_info.h"
#include "block_epilogue_swiglu_mx_quant.h"
#include "mega_moe_impl.h"
#include "moe_distribute_dispatch_v2/op_kernel/quantize_functions.h"
// #include "adv_api/hccl/hccl.h"
#include "adv_api/hcomm/hcomm.h"
#include "mega_moe_dispatch_send.h"

using namespace AscendC;

namespace MegaMoeImpl {
using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
using BlockOffset = Shape<int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t>;

#define TemplateMegaMoeTypeClass typename XType, typename OutputType, typename TopkWeightsType, \
                                  int32_t QuantMode, int32_t CombineQuantMode
#define TemplateMegaMoeTypeFunc XType, OutputType, TopkWeightsType, QuantMode, CombineQuantMode

template <TemplateMegaMoeTypeClass>
class MegaMoeLayered {
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
    __aicore__ inline MegaMoeLayered() {};
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
    __aicore__ inline void ResetRowGroupCompleteFlags();
    __aicore__ inline void SendMaskCal();
    __aicore__ inline void SendCntCal(int32_t localExpertId);
    __aicore__ inline void TripleInfoCalAndDispatch(GMMAddrInfo &gmmAddrInfo, int32_t localExpertId);
    template <AddrUpdateMode Mode>
    __aicore__ inline bool UpdateGroupParams(ExpertLoopState &state, uint32_t expertIdx);
    template <AddrUpdateMode Mode>
    __aicore__ inline void UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state);
    __aicore__ inline void Unpermute();
    __aicore__ inline void InitCombineBuffers();
    __aicore__ inline void ProcessCombine(GMMAddrInfo &gmmAddrInfo, ExpertLoopState &gmm2State, uint32_t expertIdx);
    __aicore__ inline void ProcessCombineByUrma(GMMAddrInfo &gmmAddrInfo, ExpertLoopState &gmm2State, uint32_t expertIdx);
    __aicore__ inline void CrossRankSyncInWorldSize();
    __aicore__ inline void ExpertTokenNumCopyOut();
    __aicore__ inline void CopyGMToGMPerToken(int32_t rowDstOffsetInCore, int32_t remoteRankIdx,
        int32_t copyStartIdx, int32_t copyNum);
    __aicore__ inline void QuantProcessInRank();
    __aicore__ inline void DispatchLevel1(int32_t localExpertId);       // L1 dispatch: 跨server URMA 发送
    __aicore__ inline void DispatchLevel2(GMMAddrInfo &gmmAddrInfo, int32_t localExpertId);  // L2 dispatch: server内 ReadNbi 读

    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
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
    int64_t maxRowGroupsStride_ = 0;  // rowGroupComplete 计数器中每个 expert 的 stride
    int64_t combineUbTensorSize_ = 0; // combineUbTensor 的大小（元素数）
    uint64_t l1RelayPerServerSize_ = 0;   // L1 dispatch: 单个 server 中转区大小
    uint64_t l1RecvPerServerSize_ = 0;    // L1 dispatch: 单个 server 接收区大小 (URMA 窗口内偏移)
    uint32_t l1SlotDataBytes_ = 0;        // L1 dispatch: slot data 部分字节数 (=mxQuantTokenScaleAlignBytes_)
    uint32_t l1SlotSize_ = 0;             // L1 dispatch: slot 总大小 (data + 32B metadata)

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
    LocalTensor<float> fp32ScaleTensor_;
    LocalTensor<bfloat16_t> bf16ScaleTensor_;

    using BlockEpilogue = BlockEpilogueSwigluMxQuant<QuantOutType, bfloat16_t,
        QuantScaleOutType, QuantScaleOutType, true>;
    BlockEpilogue epilogueOp_;
};

// ========================
// Init：初始化 & 偏移计算
// ========================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::Init(
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
    params_.epGmAddr = (GM_ADDR)mc2Context_->urmacclBuffer[rankId_];
    params_.rankId = rankId_;
    params_.aGmAddr = x;
    params_.expertIdxGmAddr = topkIds;
    params_.bGmAddr = GetTensorAddr(0, weight1);
    params_.b2GmAddr = GetTensorAddr(0, weight2);
    params_.bScaleGmAddr = GetTensorAddr(0, weightScales1);
    params_.b2ScaleGmAddr = GetTensorAddr(0, weightScales2);

    params_.y2GmAddr = yOut;
    params_.expertTokenNumsOutGmAddr = expertTokenNumsOut;
    params_.probsGmAddr = topkWeights;
    params_.workspaceInfo = WorkspaceInfo(workspaceGM, tilingData, ELEMS_PER_BYTE);
    params_.peermemInfo = PeermemInfo(winRankAddr_[rankId_], tilingData, ELEMS_PER_BYTE);
    params_.tilingData = tilingData;
    params_.peermemInfo.hcomm = &hcomm_;
    params_.peermemInfo.mc2Context = mc2Context_;
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
    // L1 dispatch: slot/中转区/接收区 参数
    // l1SlotDataBytes_ = tokendata + scaledata (32B 对齐)
    // l1SlotSize_ = tokendata + scaledata + 32B metadata, 再 512B 对齐 (单个 slot 总大小)
    l1SlotDataBytes_ = mxQuantTokenScaleAlignBytes_;
    l1SlotSize_ = static_cast<uint32_t>(Ops::Base::CeilAlign(
        static_cast<int64_t>(l1SlotDataBytes_) + static_cast<int64_t>(ALIGN_32),
        static_cast<int64_t>(ALIGN_512)));
    l1RelayPerServerSize_ = MegaMoeDispatchImpl::CalcRelayAreaPerServer(m_, l1SlotSize_);
    l1RecvPerServerSize_ = MegaMoeDispatchImpl::CalcL1RecvPerServer(m_, l1SlotSize_);
}

// =================================================================================================
// DispatchBuffInit：SendCntCal & TripleInfoCalAndDispatch & ExpertTokenNumCopyOut 中使用的buffer申请
// =================================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::DispatchBuffInit()
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
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::SendAndQuantBuffInit()
{
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        maxRowGroupsStride_ = MegaMoeImpl::ComputeMaxRowGroupsStride(
            m_, worldSize_, blockAivNum_);
    }
    if constexpr(g_coreType == AIC) {
        return;
    }
    // Tensor用处：SendMaskCal 函数中搬运本卡的 topkIds；
    // Tensor大小：该Tensor在进行CompareScalar时，compareCount_要求256B对齐，申请大小256字节对齐；
    LocalTensor<uint8_t> hcommTensor_=LocalTensor<uint8_t>(TPosition::VECCALC, 0, ALIGN_512 / sizeof(uint8_t));
    hcomm_.Init(hcommTensor_, ALIGN_512);
    uint32_t topkIdsTensorAddr = ALIGN_512;
    uint32_t topkIdsTensorSize = Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_256));
    topkIdsTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, topkIdsTensorAddr, topkIdsTensorSize / sizeof(int32_t));
    // Tensor用处：ResetFlagList中对于workSpace上的Flag位置区域进行清理；
    // Tensor大小：大小为清理区域大小均分到所有的blockAivNum_；
    // SwiGluToGmm2区(expertPerRank * INT_CACHELINE)+DispatchToGmm1区(expertPerRank * dispatchFlagSlotsPerExpert_)；
    uint32_t resetTensorAddr = topkIdsTensorAddr + topkIdsTensorSize;
    uint64_t totalFlagInt32 = static_cast<uint64_t>(expertPerRank_) *
        (static_cast<uint64_t>(INT_CACHELINE) + static_cast<uint64_t>(dispatchFlagSlotsPerExpert_));
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        int64_t rowGroupResetSize = static_cast<int64_t>(expertPerRank_) * maxRowGroupsStride_;
        totalFlagInt32 = (static_cast<int64_t>(totalFlagInt32) > rowGroupResetSize)
                         ? static_cast<int64_t>(totalFlagInt32)
                         : rowGroupResetSize;
    }
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
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::ResetFlagList()
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

    // 仅清零 combineSendNotifyPtr 的 worldSize 个 per-rank 计数器, 以及每个 token 槽位
    // triple(32B) 中的 flag(offset 7)。转发者(CombineTokensWithUrmaSend) 轮询 notify 计数与
    // flag==1 判断数据就绪, 残留旧值会让转发者扫描到陈旧槽位, 发送垃圾数据造成精度全错。
    // 槽位布局与 CombineTokensWithUrma 一致: slotStride = maxDataLengthPerBlock + ALIGN_32,
    // triple(32B) 位于 slotBase + maxDataLengthPerBlock。flag 在 triple byte 偏移 28(非 32B 对齐),
    // 无法单独按 32B 对齐写出; triple 整体 32B 对齐且生产者置 flag=1 前会重写整个 triple,
    // 故清整个 32B triple 即覆盖 flag, 同时满足 UB->GM 拷贝 GM 起始地址 32B 对齐的要求。
    // if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
    //     // 1) 清零 combineSendNotifyPtr 的 worldSize 个计数器。
    //     //    该区域位于 workspaceSize 末尾(workspaceSize 未计入其大小), 不可越界, 用标量写精确清零。
    //     int32_t notifyCoreLen, notifyCoreOffset;
    //     TilingByCore(static_cast<int32_t>(worldSize_), notifyCoreLen, notifyCoreOffset, 1);
    //     if (notifyCoreLen > 0) {
    //         GlobalTensor<int32_t> combineSendNotifyGm;
    //         combineSendNotifyGm.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.combineSendNotifyPtr);
    //         for (int32_t i = 0; i < notifyCoreLen; i++) {
    //             combineSendNotifyGm[notifyCoreOffset + i].SetValue(0, 0);
    //         }
    //     }

    //     // 2) 清零每个 token 槽位的 triple(32B)。槽位总数由区域大小与 slotStride 推得, 与分配一致。
    //     int32_t maxDataLengthPerBlock = static_cast<int32_t>(Ops::Base::CeilAlign(
    //         static_cast<int64_t>(L1_TILE_N) * static_cast<int64_t>(sizeof(bfloat16_t)), ALIGN_32));
    //     int32_t slotStrideBytes = maxDataLengthPerBlock + static_cast<int32_t>(ALIGN_32);
    //     int64_t totalDataBytes = static_cast<int64_t>(params_.workspaceInfo.combineSendNotifyPtr -
    //         params_.workspaceInfo.combineSendDataPtr);
    //     int32_t totalSlots = static_cast<int32_t>(totalDataBytes / slotStrideBytes);
    //     int32_t slotCoreLen, slotCoreOffset;
    //     TilingByCore(totalSlots, slotCoreLen, slotCoreOffset, 1);
    //     if (slotCoreLen > 0) {
    //         GlobalTensor<int32_t> combineSendDataGm;
    //         combineSendDataGm.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.combineSendDataPtr);
    //         int32_t tripleOffsetInt32Base = maxDataLengthPerBlock / static_cast<int32_t>(sizeof(int32_t));
    //         int32_t slotStrideInt32 = slotStrideBytes / static_cast<int32_t>(sizeof(int32_t));
    //         // 相邻 triple 间隔(32B 单位): (slotStride - tripleSize) / 32
    //         uint16_t dstStride = static_cast<uint16_t>((slotStrideBytes - static_cast<int32_t>(ALIGN_32)) /
    //             static_cast<int32_t>(ALIGN_32));
    //         // resetTensor_ 容量有限, 每个 32B triple 对应 8 个 int32, 按其容量分批(stride 拷贝批量清零)
    //         uint32_t resetTensorBlocks = resetTensor_.GetSize() /
    //             (static_cast<uint32_t>(ALIGN_32) / static_cast<uint32_t>(sizeof(int32_t)));
    //         SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    //         for (int32_t s = 0; s < slotCoreLen; ) {
    //             int32_t batch = slotCoreLen - s;
    //             if (batch > 4095) { batch = 4095; }
    //             if (static_cast<uint32_t>(batch) > resetTensorBlocks) { batch = static_cast<int32_t>(resetTensorBlocks); }
    //             if (batch <= 0) { break; }
    //             int32_t slotIdx = slotCoreOffset + s;
    //             int32_t tripleOffsetInt32 = slotIdx * slotStrideInt32 + tripleOffsetInt32Base;
    //             // blockLen=32B(triple 大小), dstStride 为相邻 triple 间隔(32B 单位), 仅清 triple 不动 data
    //             DataCopyExtParams tripleClearParams{static_cast<uint16_t>(batch),
    //                 static_cast<uint32_t>(ALIGN_32), 0U, dstStride, 0U};
    //             DataCopyPad(combineSendDataGm[tripleOffsetInt32], resetTensor_, tripleClearParams);
    //             s += batch;
    //         }
    //     }
    // }
}

// ==================================================
// ExpertTokenNumCopyOut：本卡各专家收到的token总数输出
// ==================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::ExpertTokenNumCopyOut()
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
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::SendMaskCal()
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
    int32_t totalExperts = static_cast<int32_t>(worldSize_);
    uint32_t countWordIdx = static_cast<uint32_t>(maskAlignSize_) / sizeof(int32_t);  // count 在槽内 int32 偏移
    DataCopyExtParams maskCopyParams{1U, static_cast<uint32_t>(maskSlotSize_), 0U, 0U, 0U};
    int32_t iter = 0;
    GlobalTensor<uint8_t> dstGlobalTensor;
    for (uint32_t curRankId = aivCoreIdx_; curRankId < totalExperts; curRankId += blockAivNum_) {
        for(uint32_t expertIdIndex = 0; expertIdIndex < expertPerRank_; ++expertIdIndex, ++iter) {
            int32_t curExpertId = curRankId * expertPerRank_ + expertIdIndex;
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

            uint64_t srcOffset = static_cast<uint64_t>(expertIdIndex * static_cast<int32_t>(worldSize_) + 
                                 static_cast<int32_t>(curRankId)) * static_cast<uint64_t>(maskSlotSize_);
            uint64_t dstOffset = static_cast<uint64_t>(expertIdIndex * static_cast<int32_t>(worldSize_) + static_cast<int32_t>(rankId_)) *
                            static_cast<uint64_t>(maskSlotSize_);
            dstGlobalTensor.SetGlobalBuffer(
                (__gm__ uint8_t*)(params_.workspaceInfo.maskSlotPtr + srcOffset));
            if (curRankId == rankId_) {
                dstGlobalTensor.SetGlobalBuffer(
                    (__gm__ uint8_t*)(GetUrmaAddrByRankId(mc2Context_, curRankId) + dstOffset));
            }
            DataCopyPad(dstGlobalTensor, maskBuf, maskCopyParams);
            SetFlag<AscendC::HardEvent::MTE3_V>(eventId);
            PipeBarrier<PIPE_ALL>(); //todo:后续优化
            if (curRankId != rankId_) {
                GM_ADDR remoteDataAddr = GetUrmaAddrByRankId(mc2Context_, curRankId) + dstOffset;
                GM_ADDR localGmAddr = params_.workspaceInfo.maskSlotPtr + srcOffset;
                // GM_ADDR remoteNotifyAddr = ;//todo
                hcomm_.WriteNbi(GetUrmaCommHandle(mc2Context_, curRankId, rankId_), remoteDataAddr,localGmAddr, maskSlotSize_);
                hcomm_.Drain(GetUrmaCommHandle(mc2Context_, curRankId, rankId_)); // todo:后续优化
                PipeBarrier<PIPE_ALL>();
            }
        }
    }
    for (int32_t index = 0; index < static_cast<int32_t>(DOUBLE_BUFFER); ++index) {
        WaitFlag<AscendC::HardEvent::MTE3_V>(kBufEvents[index]);
    }
}

// ===================================
// QuantProcessInRank：本卡token的量化
// ===================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::QuantProcessInRank()
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
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::SendCntCal(int32_t localExpertId)
{
    totalSendCntInExp_ = 0;
    uint32_t slotWordStride = static_cast<uint32_t>(maskSlotSize_) / sizeof(uint32_t);
    uint32_t countWordIdx = static_cast<uint32_t>(maskAlignSize_) / sizeof(uint32_t);

    // Phase 1: 从本卡 win 读本 localExpert 的全部 worldSize_ 个 [mask|count] 槽位
    GlobalTensor<uint8_t> maskSrcGlobal;
    maskSrcGlobal.SetGlobalBuffer((__gm__ uint8_t*)(GetUrmaAddrByRankId(mc2Context_, rankId_) +
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
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::CopyGMToGMPerToken(
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
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::TripleInfoCalAndDispatch(
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
// UpdateGroupParams：更新当前expertIdx的problemShape，偏移掉本卡前侧专家收到的cnt数
// ----------------------------------------------------------------------------------------------------
//   Phase 1: 根据problemShape中的M(前一个专家收到的count数)，偏移计算baseOffset中gmm1与gmm2的左右矩阵偏移；
//   Phase 2: 更新当前专家id收到的count数;
// =====================================================================================================
template <TemplateMegaMoeTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline bool MegaMoeLayered<TemplateMegaMoeTypeFunc>::UpdateGroupParams(ExpertLoopState &state, uint32_t expertIdx)
{
    if (expertIdx != 0) {
        uint64_t m = Get<M_VALUE>(state.problemShape);
        uint64_t n = Get<N_VALUE>(state.problemShape);
        uint64_t k = Get<K_VALUE>(state.problemShape);
        expertBeforeCnt_ += m;
        Get<IDX_A_OFFSET>(state.baseOffset) += m * k / ELEMS_PER_BYTE;
        Get<IDX_B_OFFSET>(state.baseOffset) += n * k / ELEMS_PER_BYTE;
        // only splitM
        auto scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_A_SCALE_OFFSET>(state.baseOffset) += m * scaleK;
        Get<IDX_B_SCALE_OFFSET>(state.baseOffset) += n * scaleK;
        Get<IDX_C_OFFSET>(state.baseOffset) += m * n / SWIGLU_N_HALF / ELEMS_PER_BYTE;
        Get<IDX_C_SCALE_OFFSET>(state.baseOffset) +=
            m * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_FLAG_OFFSET>(state.baseOffset) += 1;
        Get<IDX_B2_OFFSET>(state.baseOffset) += k * n / SWIGLU_N_HALF / ELEMS_PER_BYTE;
        Get<IDX_B2_SCALE_OFFSET>(state.baseOffset) +=
            k * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_Y2_OFFSET>(state.baseOffset) += m * k;
        Get<IDX_M_OFFSET>(state.baseOffset) += m;
    }

    // gmm1中当前专家收到的count数是由subBlockIdx_=1的aiv计算出并写入expertRevNumsGloablTensor_，通知后续aic读取该值
    if constexpr (Mode == AddrUpdateMode::kGmm1) {
        if constexpr (g_coreType == AscendC::AIC) {
            CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_S>(2 + 16);
            uint64_t offsetInCnt = expertIdx * 8 * aicNum_ + 8 * blockIdx_;
            DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
                expertRevNumsGloablTensor_[offsetInCnt]);
            Get<M_VALUE>(state.problemShape) = expertRevNumsGloablTensor_.GetValue(offsetInCnt);
            CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_S>(3);
        }
        if constexpr (g_coreType == AscendC::AIV) {
            if (subBlockIdx_ == 0) {
                CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_S>(3);
                uint64_t offsetInCnt = expertIdx * 8 * aicNum_ + 8 * blockIdx_;
                DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
                expertRevNumsGloablTensor_[offsetInCnt]);
                Get<M_VALUE>(state.problemShape) = expertRevNumsGloablTensor_.GetValue(offsetInCnt);
            } else {
                Get<M_VALUE>(state.problemShape) = totalSendCntInExp_;
            }
        }
    } else if constexpr (Mode == AddrUpdateMode::kGmm2) {
        uint64_t offsetInCnt = expertIdx * 8 * aicNum_ + 8 * blockIdx_;
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
                expertRevNumsGloablTensor_[offsetInCnt]);
        Get<M_VALUE>(state.problemShape) = expertRevNumsGloablTensor_.GetValue(offsetInCnt);
    }

    if (Get<M_VALUE>(state.problemShape) == 0) {
        return false;
    }
    return true;
}

// ==================================================================================
// UpdateGlobalBuffer：更新当前expertIdx gmmAddrInfo中的地址，
//                     该地址用于后续GroupMatmulSwigluQuant 与 GroupMatmul2运算；
// ==================================================================================
template <TemplateMegaMoeTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo,
    const ExpertLoopState &state)
{
    if constexpr (Mode == AddrUpdateMode::kGmm1) {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.dispatchRevDataPtr +
            Get<IDX_A_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.dispatchRevScalePtr +
        Get<IDX_A_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);
        if constexpr(g_coreType == AIC) {
            gmmAddrInfo.bGlobal = params_.bGmAddr + Get<IDX_B_OFFSET>(state.baseOffset) * sizeof(ActivationType);
            gmmAddrInfo.bScaleGlobal = params_.bScaleGmAddr + Get<IDX_B_SCALE_OFFSET>(state.baseOffset) *
                                        sizeof(QuantScaleOutType);
        } else {
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                Get<IDX_C_OFFSET>(state.baseOffset), Get<IDX_C_SCALE_OFFSET>(state.baseOffset),
                Get<IDX_FLAG_OFFSET>(state.baseOffset), 0L, 0L, 0L};
            epilogueOp_.UpdateGlobalAddr(vecBaseOffset);
        }
    } else if constexpr(Mode == AddrUpdateMode::kGmm2 && g_coreType == AIC) {
        gmmAddrInfo.aGlobal =
            params_.workspaceInfo.swigluQuantDataPtr + Get<IDX_C_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.swigluQuantScalePtr +
                                   Get<IDX_C_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);
        gmmAddrInfo.bGlobal = params_.b2GmAddr + Get<IDX_B2_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.bScaleGlobal =
            params_.b2ScaleGmAddr + Get<IDX_B2_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);
    }
    if constexpr (Mode == AddrUpdateMode::kGmm2 && CombineQuantMode != COMBINE_NO_QUANT) {
        gmmAddrInfo.gmm2OutGlobal = params_.workspaceInfo.gmmOutPtr +
                                    Get<IDX_Y2_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
    }
    gmmAddrInfo.groupFlagList = (__gm__ int32_t*)params_.workspaceInfo.flagSwiGluToGmm2Ptr +
                                Get<IDX_FLAG_OFFSET>(state.baseOffset) * INT_CACHELINE;
    // wave-grain dispatch-gmm1 flag: per-expert 步长是 dispatchFlagSlotsPerExpert_,而不是 INT_CACHELINE。
    gmmAddrInfo.groupFlagList2 = (__gm__ int32_t*)params_.workspaceInfo.flagDispatchToGmm1Ptr +
                                Get<IDX_FLAG_OFFSET>(state.baseOffset) * dispatchFlagSlotsPerExpert_;
    groupFlagListGm_.SetGlobalBuffer(gmmAddrInfo.groupFlagList);
}

// =============================================
// ResetRowGroupCompleteFlags：重置 row group 完成标志计数器
// =============================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::ResetRowGroupCompleteFlags()
{
    if constexpr(g_coreType == AIV) {
        int32_t totalCounters = static_cast<int32_t>(static_cast<int64_t>(expertPerRank_) * maxRowGroupsStride_);
        int32_t coreLen, coreOffset;
        TilingByCore(totalCounters, coreLen, coreOffset);
        GlobalTensor<int32_t> rowGroupCompleteGm;
        rowGroupCompleteGm.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.rowGroupCompletePtr);
        if (coreLen > 0) {
            Duplicate(resetTensor_, 0, coreLen);
            SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
            DataCopy(rowGroupCompleteGm[coreOffset], resetTensor_, coreLen);
        }
    }
}

// =============================================
// InitCombineBuffers：初始化 Combine 所需的 buffer 大小
// =============================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::InitCombineBuffers()
{
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV) {
        uint32_t nAlign32 = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_32));
        uint32_t nScale = Ops::Base::CeilDiv(k_, uint32_t(MXFP_SCALE_GROUP_NUM));
        uint32_t quantRowSizeBytes = Ops::Base::CeilAlign(k_ + nScale, static_cast<uint32_t>(ALIGN_32));
        uint32_t singleTokenBytes = nAlign32 * sizeof(bfloat16_t) + quantRowSizeBytes;
        combineUbTensorSize_ = (singleTokenBytes * 2) / sizeof(bfloat16_t);
    }
}


// =============================================
// ProcessCombineByUrma：量化路径的 Combine 处理（URMA 通信），按 rank 分核
// =============================================
// 与 ProcessCombine 的区别：
//   - 量化模式下所有 AIV 核参与（AIV 不参与计算），按 rank 分核而非按 row group 分核
//   - 需等待该专家所有 row group 完成（因为按 rank 分核后需扫描所有 token）
//   - 通过 ComputeGroupRange 定位每个 row group 的计数器进行轮询
//   - 量化+发送逻辑复用 CombineRowGroup
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::ProcessCombineByUrma(
    GMMAddrInfo &gmmAddrInfo, ExpertLoopState &gmm2State, uint32_t expertIdx)
{
    // 量化模式下所有 AIV 核参与，按 rank 分核
    uint32_t aivIdx = aivCoreIdx_;
    uint32_t aivNum = blockAivNum_;

    // 按 rank 分核: 每个核负责一段连续的目标 rank
    uint32_t rankPerCore = worldSize_ / aivNum;
    uint32_t remainder = worldSize_ % aivNum;
    uint32_t startRankId = rankPerCore * aivIdx + (aivIdx < remainder ? aivIdx : remainder);
    uint32_t processRankNum = rankPerCore + (aivIdx < remainder ? 1 : 0);
    uint32_t endRankId = startRankId + processRankNum;

    if (processRankNum == 0) {
        return;
    }

    int64_t offset = ALIGN_512;
    uint32_t m_expert = Get<M_VALUE>(gmm2State.problemShape);
    uint32_t nTilesPerGroup = Ops::Base::CeilDiv(k_, L1_TILE_N);
    uint32_t rowGroupsThisExpert = Ops::Base::CeilDiv(m_expert, L1_TILE_M_256);
    uint32_t nScale = Ops::Base::CeilDiv(k_, uint32_t(MXFP_SCALE_GROUP_NUM));
    uint32_t quantRowSizeBytes = Ops::Base::CeilAlign(k_ + nScale, static_cast<uint32_t>(ALIGN_32));

    GlobalTensor<int32_t> rowGroupComplete;
    rowGroupComplete.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.rowGroupCompletePtr);

    // Phase 1: 从本地 URMA mask slot 读取每个源 rank 的 token count, 计算每 rank 的行范围
    // token 按源 rank 紧密排布, 利用 mask slot 中的 count 可精确计算每 rank 的起止行
    __gm__ uint8_t* maskSlotBase = (__gm__ uint8_t*)(GetUrmaAddrByRankId(mc2Context_, rankId_) +
        static_cast<uint64_t>(expertIdx) * worldSize_ * maskSlotSize_);

    // 计算 rank 0..startRankId-1 的 cumsum (作为本核处理范围之前的行偏移)
    uint32_t cumsumBeforeRange = 0;
    for (uint32_t r = 0; r < startRankId; r++) {
        __gm__ int32_t* cntAddr = reinterpret_cast<__gm__ int32_t*>(maskSlotBase + r * maskSlotSize_ + maskAlignSize_);
        cumsumBeforeRange += AscendC::ReadGmByPassDCache(cntAddr);
    }

    // UB 布局 (VECCALC/VECIN 共享物理内存):
    //   0 ~ ALIGN_512:                      hcomm_ (VECCALC)
    //   ALIGN_512 ~ ALIGN_512+rankInfoSize:  rankInfoTensor (VECCALC, 跨 Phase 1/2 持续读写)
    //   ALIGN_512+rankInfoSize ~:            tripleTensor + combineUbTensor (VECIN, 每次 batch 短暂使用)
    // Layout: [startRow_0..N-1 | tokenCnt_0..N-1 | processedCnt_0..N-1]
    uint32_t rankInfoSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(processRankNum * 3 * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    LocalTensor<int32_t> rankInfoTensor = LocalTensor<int32_t>(TPosition::VECCALC, offset,
        rankInfoSize / sizeof(int32_t));
    offset  += rankInfoSize;

    uint32_t cumsumInRange = 0;
    uint32_t totalTokensToProcess = 0;
    for (uint32_t i = 0; i < processRankNum; i++) {
        uint32_t r = startRankId + i;
        __gm__ int32_t* cntAddr = reinterpret_cast<__gm__ int32_t*>(maskSlotBase + r * maskSlotSize_ + maskAlignSize_);
        int32_t cnt = AscendC::ReadGmByPassDCache(cntAddr);
        rankInfoTensor.SetValue(i, cumsumBeforeRange + cumsumInRange);  // startRow
        rankInfoTensor.SetValue(processRankNum + i, cnt);               // tokenCnt
        rankInfoTensor.SetValue(2 * processRankNum + i, 0);             // processedCnt
        cumsumInRange += cnt;
        totalTokensToProcess += cnt;
    }

    if (totalTokensToProcess == 0) {
        return;
    }

    // Phase 2: 按 rank 轮询, 只检查每 rank 需要的 row group, 不阻塞等待
    // 当 group 完成时, 批量处理该 rank 在该 group 内的所有 token
    // batchBaseOffset 是每轮 batch 的 UB 基地址 (VECIN), 每次 loop 重置避免线性增长溢出
    uint32_t batchBaseOffset = static_cast<uint32_t>(offset);
    uint32_t rankIndex = 0;
    uint32_t totalProcessed = 0;
    while (totalProcessed < totalTokensToProcess) {
        // 跳过已处理完毕 (processedCnt >= tokenCnt) 的 rank
        while (rankInfoTensor.GetValue(2 * processRankNum + rankIndex) >=
               rankInfoTensor.GetValue(processRankNum + rankIndex)) {
            rankIndex = (rankIndex + 1) % processRankNum;
        }

        uint32_t startRow = rankInfoTensor.GetValue(rankIndex);
        uint32_t tokenCnt = rankInfoTensor.GetValue(processRankNum + rankIndex);
        uint32_t processedCnt = rankInfoTensor.GetValue(2 * processRankNum + rankIndex);
        uint32_t currentRow = startRow + processedCnt;
        uint32_t targetGroup = currentRow / L1_TILE_M_256;

        // 检查该 row group 是否完成 (非阻塞, 未完成则跳到下一个 rank)
        uint32_t groupCoreStart, groupCoreSize;
        MegaMoeImpl::ComputeGroupRange(targetGroup, rowGroupsThisExpert, blockAivNum_, groupCoreStart, groupCoreSize);
        __gm__ int32_t* counterAddr = (__gm__ int32_t*)rowGroupComplete.GetPhyAddr()
            + expertIdx * maxRowGroupsStride_ + targetGroup * blockAivNum_ * INT_CACHELINE
            + groupCoreStart * INT_CACHELINE;
        int32_t counterVal = AscendC::ReadGmByPassDCache(counterAddr);

        if (counterVal == static_cast<int32_t>(nTilesPerGroup)) {
            // Group 已完成, 计算该 rank 在此 group 内的 batch token 数
            uint32_t groupEndRow = (targetGroup + 1) * L1_TILE_M_256;
            if (groupEndRow > m_expert) {
                groupEndRow = m_expert;
            }
            uint32_t remainingTokens = tokenCnt - processedCnt;
            uint32_t batchCount = groupEndRow - currentRow;
            if (batchCount > remainingTokens) {
                batchCount = remainingTokens;
            }

            // 每次 batch 重置 offset, 避免 linear growth 导致 UB 溢出
            offset = batchBaseOffset;

            // 加载该 batch 的 triple info 到 VECIN (从 rankInfoTensor 末尾之后开始, 避免覆盖)
            LocalTensor<int32_t> tripleTensor = LocalTensor<int32_t>(TPosition::VECIN, offset, batchCount * TRIPLE_SIZE);
            offset += batchCount * TRIPLE_SIZE * sizeof(int32_t);
            AscendC::GlobalTensor<int32_t> tripleGm;
            tripleGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.tripleInfoPtr
                + (expertBeforeCnt_ + currentRow) * TRIPLE_SIZE * sizeof(int32_t)));
            AscendC::DataCopy(tripleTensor, tripleGm, batchCount * TRIPLE_SIZE);
            PipeBarrier<PIPE_MTE2>();

            // 调用 CombineRowGroup 批量处理该 rank 在此 group 内的所有 token
            MegaMoeCombineImpl::CombineRowGroup<CombineQuantMode, bfloat16_t>(
                currentRow, batchCount, k_, expertIdx, rankId_,
                gmmAddrInfo.gmm2OutGlobal, params_, tripleTensor, combineUbTensorSize_,
                offset, quantRowSizeBytes);

            rankInfoTensor.SetValue(2 * processRankNum + rankIndex, processedCnt + batchCount);
            totalProcessed += batchCount;
        } else {
            // Group 未完成, 短暂等待后轮询下一个 rank
            AscendC::Nop<200>();
        }

        // 轮询: 移到下一个 rank
        rankIndex = (rankIndex + 1) % processRankNum;
    }

    // drain 本核负责的远程 rank 的 URMA channel
    for (uint32_t dr = startRankId; dr < endRankId; ++dr) {
        if (dr == rankId_) {
            continue;
        }
        hcomm_.Drain(GetUrmaCommHandle(mc2Context_, dr, rankId_));
        PipeBarrier<PIPE_ALL>();
    }
}

// =============================================
// ProcessCombine：量化路径的 Combine 处理，等待 GMM2 完成，读取结果，量化后发送到目标 rank
// =============================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::ProcessCombine(
    GMMAddrInfo &gmmAddrInfo, ExpertLoopState &gmm2State, uint32_t expertIdx)
{
    uint32_t nTilesPerGroup = Ops::Base::CeilDiv(k_, L1_TILE_N);

    GlobalTensor<int32_t> rowGroupComplete;
    rowGroupComplete.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.rowGroupCompletePtr);

    uint32_t nScale = Ops::Base::CeilDiv(k_, uint32_t(MXFP_SCALE_GROUP_NUM));
    uint32_t quantRowSizeBytes = Ops::Base::CeilAlign(k_ + nScale, static_cast<uint32_t>(ALIGN_32));

    uint32_t m_expert = Get<M_VALUE>(gmm2State.problemShape);
    uint32_t rowGroupsThisExpert = Ops::Base::CeilDiv(m_expert, L1_TILE_M_256);

    uint32_t myGroup, myIdxInGrp, myGrpSize;
    MegaMoeImpl::ComputeCoreGrouping(aivCoreIdx_, rowGroupsThisExpert, blockAivNum_,
        myGroup, myIdxInGrp, myGrpSize);

    if (myGroup >= rowGroupsThisExpert) {
        return;
    }

    __gm__ int32_t* myCounterAddr = (__gm__ int32_t*)rowGroupComplete.GetPhyAddr()
        + expertIdx * maxRowGroupsStride_ + myGroup * blockAivNum_ * INT_CACHELINE
        + aivCoreIdx_ * INT_CACHELINE;
    while (AscendC::ReadGmByPassDCache(myCounterAddr) != nTilesPerGroup)
    {
        AscendC::Nop<200>();
    }
    uint32_t rowStart = myGroup * L1_TILE_M_256;
    uint32_t rowCount = (L1_TILE_M_256 < m_expert - rowStart) ? L1_TILE_M_256 : m_expert - rowStart;
    uint32_t rowsPerCore = Ops::Base::CeilDiv(rowCount, myGrpSize);
    int32_t myRowOffset = myIdxInGrp * rowsPerCore;
    int32_t myRowCount = 0;
    if (myRowOffset < (int32_t)rowCount) {
        myRowCount = (rowsPerCore < rowCount - myRowOffset) ? rowsPerCore : rowCount - myRowOffset;
    }
    if (myRowCount > 0) {
        AscendC::SetCtrlSpr<60, 60>(0);
        int64_t offset = 0;
        LocalTensor<int32_t> tripleTensor = LocalTensor<int32_t>(TPosition::VECIN, offset, myRowCount * TRIPLE_SIZE);
        offset += myRowCount * TRIPLE_SIZE * sizeof(int32_t);
        AscendC::GlobalTensor<int32_t> tripleGm;
        tripleGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.tripleInfoPtr
            + (expertBeforeCnt_ + rowStart + myRowOffset) * TRIPLE_SIZE * sizeof(int32_t)));
        AscendC::DataCopy(tripleTensor, tripleGm, myRowCount * TRIPLE_SIZE);
        PipeBarrier<PIPE_MTE2>();
        MegaMoeCombineImpl::CombineRowGroup<CombineQuantMode, bfloat16_t>(
            rowStart + myRowOffset, myRowCount, k_, expertIdx, rankId_,
            gmmAddrInfo.gmm2OutGlobal, params_, tripleTensor, combineUbTensorSize_,
            offset, quantRowSizeBytes);
    }
}

// =============================================
// UnpermuteBuffInit：Unpermute中使用的buffer申请
// =============================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::UnpermuteBuffInit()
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
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        uint32_t scaleNum = Ops::Base::CeilAlign(static_cast<uint32_t>(k_), static_cast<uint32_t>(ALIGN_32));
        // Tensor用处：DeQuantMxFp8 中用于存储 bf16 格式的 scale（e8m0 转换后的中间结果）
        // Tensor大小：scaleNum * sizeof(bfloat16_t) * DOUBLE_BUFFER * HALF_TO_FP32，双缓冲 + scale 扩展
        uint32_t bf16ScaleBufAlign = Ops::Base::CeilAlign(static_cast<uint32_t>
            (scaleNum * sizeof(bfloat16_t) * DOUBLE_BUFFER * HALF_TO_FP32), static_cast<uint32_t>(ALIGN_32));
        bf16ScaleTensor_ = LocalTensor<bfloat16_t>(
            TPosition::VECCALC, tempAddr, bf16ScaleBufAlign / sizeof(bfloat16_t));
        tempAddr += bf16ScaleBufAlign;
        // Tensor用处：DeQuantMxFp8 中用于存储 fp32 格式的 scale（广播后的最终 scale）
        // Tensor大小：scaleNum * sizeof(float) * DOUBLE_BUFFER * HALF_TO_FP32，双缓冲 + scale 扩展
        uint32_t fp32ScaleBufAlign = Ops::Base::CeilAlign(static_cast<uint32_t>
            (scaleNum * sizeof(float) * DOUBLE_BUFFER * HALF_TO_FP32), static_cast<uint32_t>(ALIGN_32));
        fp32ScaleTensor_ = LocalTensor<float>(TPosition::VECCALC, tempAddr, fp32ScaleBufAlign / sizeof(float));
        tempAddr += fp32ScaleBufAlign;
    }
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
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::Unpermute()
{
    int32_t coreLen, coreOffset;
    TilingByCore(m_, coreLen, coreOffset, 1);
    GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer((__gm__ bfloat16_t*)(params_.peermemInfo.URMACombineWinPtr + params_.epGmAddr));
    //expandedX.SetGlobalBuffer((__gm__ bfloat16_t*)params_.peermemInfo.combineSendPtr);
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
        // fp32 -> bf16
        Cast(dataResTensor_, dataResFp32Tensor_, AscendC::RoundMode::CAST_RINT, k_);
        SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID3>();
        DataCopy(output[tokenIdx * k_], dataResTensor_, k_);
    }
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
}

// =====================================================================================================
// DispatchLevel1：跨 server 一级 URMA 通信
// ----------------------------------------------------------------------------------------------------
//   核心分工 (发送核用最后几个核, 计算核用前面的核):
//     - 发送核 (aivCoreIdx >= blockNum_ - numSendCores): 轮询中转区, WriteWithNotifyNbi 发到对端接收区
//     - 计算核 (aivCoreIdx < blockNum_ - numSendCores): 量化 token → 写入中转区/二级区域 → 写完成标志
//
//   通信模式: 同层跨 server 直连 (server0 卡0 ↔ server1 卡0)
//   每轮按 localExpertId 遍历, 已发送的 token 跳过
//   本 server 内的 token 直接写入二级通信读区域 (L2 实现后填地址)
//   中转区 layout: [32B count header][slot_0..slot_m-1]  per target server
//   每个 slot = tokendata + scaledata + 32B, 总大小 512B 对齐 (l1SlotSize_)
//   对端接收区 layout: sparse by tokenIdx, 整个 slot 的最后 2 个 int (slotSize-8B) 为 WriteWithNotifyNbi notify
// =====================================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::DispatchLevel1(int32_t localExpertId)
{
    // ---------- Step 0: Server 拓扑 ----------
    uint32_t rankSizePerServer = mc2Context_->rankSizePerServer;
    bool hasMultipleServers = (rankSizePerServer > 0 && rankSizePerServer < worldSize_);
    uint32_t totalServers = hasMultipleServers
        ? MegaMoeDispatchImpl::GetTotalServers(worldSize_, rankSizePerServer) : 1;
    uint32_t numOtherServers = hasMultipleServers ? (totalServers - 1) : 0;
    uint32_t myServerId = hasMultipleServers
        ? MegaMoeDispatchImpl::GetServerId(rankId_, rankSizePerServer) : 0;
    uint32_t myLocalRank = hasMultipleServers
        ? MegaMoeDispatchImpl::GetLocalRankInServer(rankId_, rankSizePerServer) : 0;

    // ---------- Step 1: 核角色分配 ----------
    // 发送核用最后 numSendCores 个核, 计算核用前面 blockNum_ - numSendCores 个核
    uint32_t numSendCores = 0;
    if (hasMultipleServers) {
        numSendCores = (numOtherServers < MegaMoeDispatchImpl::MAX_L1_SEND_CORES)
            ? numOtherServers : MegaMoeDispatchImpl::MAX_L1_SEND_CORES;
    }
    uint32_t numComputeCores = blockNum_ - numSendCores;
    bool isSendCore = (aivCoreIdx_ >= numComputeCores);
    uint32_t sendCoreIdx = aivCoreIdx_ - numComputeCores;  // 发送核在发送核组内的索引 (0-based)
    uint32_t computeCoreLocalIdx = aivCoreIdx_;              // 计算核在计算核组内的索引 (0-based)

    if (isSendCore && numSendCores == 0) {
        // 无跨 server 通信需求, 发送核空闲, 直接退出 (完成标志由计算核写)
        return;
    }

    if (isSendCore) {
        // ========== 发送核路径 (最后几个核) ==========
        // 分配: 每个 发送核 负责一段连续的 otherServer
        uint32_t serversPerSendCore = numOtherServers / numSendCores;
        uint32_t sendRemainder = numOtherServers % numSendCores;
        uint32_t myStartOtherIdx = serversPerSendCore * sendCoreIdx +
            (sendCoreIdx < sendRemainder ? sendCoreIdx : sendRemainder);
        uint32_t myNumServers = serversPerSendCore + (sendCoreIdx < sendRemainder ? 1 : 0);
        if (myNumServers == 0) {
            // 无分配 server, 直接退出 (完成标志由计算核写)
            return;
        }

        // UB 布局: metadata 读缓冲(32B) + 发送计数(每 otherServer 1 int)
        uint32_t ubOffset = ubBufferUsedAddr_;
        LocalTensor<int32_t> metaReadTensor = LocalTensor<int32_t>(TPosition::VECCALC, ubOffset,
            MegaMoeDispatchImpl::L1_META_BYTES / sizeof(int32_t));
        ubOffset += MegaMoeDispatchImpl::L1_META_BYTES;
        uint32_t sentCntSize = Ops::Base::CeilAlign(static_cast<int64_t>(myNumServers * sizeof(int32_t)),
            static_cast<int64_t>(ALIGN_32));
        LocalTensor<int32_t> sentCntTensor = LocalTensor<int32_t>(TPosition::VECCALC, ubOffset,
            sentCntSize / sizeof(int32_t));
        ubOffset += sentCntSize;
        Duplicate<int32_t>(sentCntTensor, 0, myNumServers);
        PipeBarrier<PIPE_ALL>();

        // 轮询中转区: 逐 otherServer 检查 count, 发现新数据后读取 metadata flag → WriteWithNotifyNbi
        // 退出条件: 所有 server 的数据已发完 且 所有计算核已完成 (完成标志由计算核写)
        AscendC::GlobalTensor<int32_t> gmRelayMeta;
        uint32_t processedServers = 0;
        uint32_t serverPollIdx = 0;

        while (processedServers < myNumServers) {
            uint32_t otherIdx = myStartOtherIdx + serverPollIdx;
            uint32_t targetServerId = MegaMoeDispatchImpl::OtherServerIdxToServerId(otherIdx, myServerId);
            uint32_t targetRankId = MegaMoeDispatchImpl::GetL1TargetRankId(targetServerId, myLocalRank, rankSizePerServer);

            // 读中转区 count
            GM_ADDR relayBase = MegaMoeDispatchImpl::GetRelayAreaBase(
                params_.workspaceInfo.dispatchL1RelayDataPtr, otherIdx, l1RelayPerServerSize_);
            __gm__ int32_t* countAddr = MegaMoeDispatchImpl::GetRelayCountAddr(relayBase);
            uint32_t curCnt = AscendC::ReadGmByPassDCache(countAddr);
            uint32_t localSentCnt = sentCntTensor.GetValue(serverPollIdx);

            // 有新数据 → 逐 slot 读取 metadata → 检查 flag → 发送
            if (curCnt > localSentCnt) {
                uint32_t needCnt = curCnt - localSentCnt;
                uint32_t slotIdx = localSentCnt;
                while (needCnt > 0) {
                    GM_ADDR slotMetaAddr = MegaMoeDispatchImpl::GetRelaySlotMetaAddr(
                        relayBase, slotIdx, l1SlotSize_, l1SlotDataBytes_);
                    gmRelayMeta.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(slotMetaAddr));
                    AscendC::Nop<800>();  // 等 GM 写入完成
                    AscendC::DataCopy(metaReadTensor, gmRelayMeta,
                        MegaMoeDispatchImpl::L1_META_BYTES / sizeof(int32_t));
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);

                    if (metaReadTensor.GetValue(0) == 1) {  // flag = ready
                        uint32_t tokenIdx = metaReadTensor.GetValue(1);

                        // 计算远端地址: URMA 窗口 + section offset + token slot offset
                        uint64_t channelHandle = GetUrmaCommHandle(mc2Context_, targetRankId, rankId_);
                        GM_ADDR remoteBase = GetUrmaAddrByRankId(mc2Context_, targetRankId) +
                            params_.peermemInfo.URMADispatchL1RecvPtr;
                        uint64_t recvSectionOffset = MegaMoeDispatchImpl::GetL1RecvSectionOffset(
                            myServerId, l1RecvPerServerSize_);
                        uint64_t recvDataOffset = recvSectionOffset +
                            MegaMoeDispatchImpl::GetL1RecvDataOffset(tokenIdx, l1SlotSize_);
                        uint64_t recvNotifyOffset = recvSectionOffset +
                            MegaMoeDispatchImpl::GetL1RecvNotifyOffset(tokenIdx, l1SlotSize_);
                        GM_ADDR remoteDataAddr = remoteBase + recvDataOffset;
                        GM_ADDR remoteNotifyAddr = remoteBase + recvNotifyOffset;

                        // 本地中转区 data 地址
                        GM_ADDR localDataAddr = MegaMoeDispatchImpl::GetRelaySlotDataAddr(
                            relayBase, slotIdx, l1SlotSize_);

                        // WriteWithNotifyNbi: 发送 tokendata+scaledata (data 部分),
                        // 同时把 notify(flag=1) 写到对端整个 slot 的最后 2 个 int (slotSize-8B),
                        // 二级通信读 slot 末尾 8B 即可判断该 token 就绪
                        hcomm_.WriteWithNotifyNbi(channelHandle, remoteDataAddr, localDataAddr,
                            l1SlotDataBytes_, remoteNotifyAddr, 1);
                        // 标记已处理
                        metaReadTensor.SetValue(0, 0);
                        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
                        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
                        AscendC::DataCopy(gmRelayMeta, metaReadTensor,
                            MegaMoeDispatchImpl::L1_META_BYTES / sizeof(int32_t));
                        PipeBarrier<PIPE_ALL>();

                        needCnt--;
                    }
                    // 修正: 轮询所有槽位, 跳过已处理的(flag=0), 只发送 flag=1 的
                    slotIdx = (slotIdx + 1 < curCnt) ? slotIdx + 1 : localSentCnt;
                }
                sentCntTensor.SetValue(serverPollIdx, curCnt);
            }

            // 判断当前 server 是否已完成: 数据已全部发送 且 计算核已完成
            // 完成标志只由计算核写, 发送核只读不写
            // 注意: curCnt 是循环顶部读取的旧值, 计算核完成后 count 可能已继续增长,
            //       因此必须在确认计算核完成后重新读 count, 否则会漏发最后一批数据
            if (curCnt == sentCntTensor.GetValue(serverPollIdx)) {
                bool allComputeDone = true;
                for (uint32_t coreIdx = 0; coreIdx < numComputeCores; coreIdx++) {
                    __gm__ int32_t* flagAddr = MegaMoeDispatchImpl::GetCompleteListAddr(
                        params_.workspaceInfo.dispatchL1CompleteListPtr, coreIdx, expertPerRank_);
                    if (AscendC::ReadGmByPassDCache(flagAddr + localExpertId) != 1) {
                        allComputeDone = false;
                        break;
                    }
                }
                if (allComputeDone) {
                    // 计算核已全部完成, 重新读取最终 count 确认无遗漏数据未发送
                    uint32_t finalCnt = AscendC::ReadGmByPassDCache(countAddr);
                    if (finalCnt == sentCntTensor.GetValue(serverPollIdx)) {
                        processedServers++;
                    }
                }
            }
            serverPollIdx = (serverPollIdx + 1) % myNumServers;
        }

    } else {
        // ========== 计算核路径 (前面 numComputeCores 个核) ==========
        // 按 token 分核: tokenIdx % numComputeCores == computeCoreLocalIdx
        // 复用 SendAndQuantBuffInit 中已有的 xInTensor1_/xOutTensor1_/mxTempTensor_

        // UB: metadata 写缓冲 + sent 跟踪 (跨 server)
        uint32_t ubOffset = ubBufferUsedAddr_;
        LocalTensor<int32_t> metaWriteTensor = LocalTensor<int32_t>(TPosition::VECCALC, ubOffset,
            MegaMoeDispatchImpl::L1_META_BYTES / sizeof(int32_t));
        ubOffset += MegaMoeDispatchImpl::L1_META_BYTES;
        uint32_t sentFlagsSize = hasMultipleServers
            ? Ops::Base::CeilAlign(static_cast<int64_t>(m_ * numOtherServers), static_cast<int64_t>(ALIGN_32))
            : ALIGN_32;  // 单 server 时无跨 server 发送, 但仍分配最小空间避免 UB 溢出
        LocalTensor<uint8_t> sentFlags = LocalTensor<uint8_t>(TPosition::VECCALC, ubOffset,
            sentFlagsSize / sizeof(uint8_t));
        ubOffset += sentFlagsSize;

        // 首轮初始化 sentFlags 为 0 (后续 round 保持)
        if (localExpertId == 0) {
            Duplicate<uint8_t>(sentFlags, 0, sentFlagsSize / sizeof(uint8_t));
            PipeBarrier<PIPE_ALL>();
        }

        // 读本卡 topkIds (从 GM)
        AscendC::GlobalTensor<int32_t> topkIdsGlobal;
        topkIdsGlobal.SetGlobalBuffer((__gm__ int32_t*)params_.expertIdxGmAddr);

        // 量化输出写 GM 的 DataCopyExtParams
        AscendC::DataCopyExtParams dataOutParams{1, static_cast<uint32_t>(l1SlotDataBytes_), 0U, 0U, 0U};
        AscendC::DataCopyParams xCopyInParams = {1U, static_cast<uint16_t>(k_ * sizeof(bfloat16_t)), 0U, 0U};
        AscendC::DataCopyPadParams xCopyInPadParams{true, 0, 0, 0};

        // 预置 MTE3_MTE2 事件: 使首次 WaitFlag<MTE3_MTE2> 不死锁 (参考 QuantProcessInRank 的双 buffer 软流水模式)
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

        // // 逐 token: 检查 topkIds → 确定目标 server → 量化 → 写中转区或二级区域
        for (uint32_t tokenIdx = 0; tokenIdx < m_; tokenIdx++) {
            // 分核: 只处理属于本计算核的 token
            if (tokenIdx % numComputeCores != computeCoreLocalIdx) {
                continue;
            }
            
            for (uint32_t topkIdx = 0; topkIdx < topK_; topkIdx++) {
                int32_t globalExpertId = topkIdsGlobal.GetValue(tokenIdx * topK_ + topkIdx);
                uint32_t targetRank = globalExpertId / expertPerRank_;
                uint32_t targetServer = hasMultipleServers
                    ? MegaMoeDispatchImpl::GetServerId(targetRank, rankSizePerServer) : 0;
                uint32_t localExpertOnTarget = globalExpertId % expertPerRank_;

                // 只处理当前 round 的 expert
                if (localExpertOnTarget != static_cast<uint32_t>(localExpertId)) {
                    continue;
                }

                // -------- 量化 token --------
                AscendC::GlobalTensor<bfloat16_t> srcGlobal;
                srcGlobal.SetGlobalBuffer((__gm__ bfloat16_t*)(params_.aGmAddr +
                    static_cast<uint64_t>(tokenIdx) * k_ * sizeof(bfloat16_t)));
                auto xInTensor = (tokenIdx % DOUBLE_BUFFER == 0) ? xInTensor1_ : xInTensor2_;
                auto xOutTensor = (tokenIdx % DOUBLE_BUFFER == 0) ? xOutTensor1_ : xOutTensor2_;
                auto event = (tokenIdx % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
                AscendC::DataCopyPad(xInTensor, srcGlobal, xCopyInParams, xCopyInPadParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);

                __ubuf__ bfloat16_t* srcAddr = (__ubuf__ bfloat16_t*)xInTensor.GetPhyAddr();
                __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)mxTempTensor_.GetPhyAddr();
                __ubuf__ uint16_t* halfScaleAddr = (__ubuf__ uint16_t*)mxTempTensor_[
                    Ops::Base::CeilAlign(mxQuantScaleNumAlignPerToken_, static_cast<uint32_t>(ALIGN_32))].GetPhyAddr();
                __ubuf__ int8_t* outDataAddr = (__ubuf__ int8_t*)xOutTensor.GetPhyAddr();
                __ubuf__ uint16_t* mxScaleAddr = (__ubuf__ uint16_t*)xOutTensor[mxQuantTokenAlignBytes_].GetPhyAddr();

                quant::ComputeMaxExp(srcAddr, maxExpAddr, k_);
                quant::ComputeScale<QuantOutType>(maxExpAddr, mxScaleAddr, halfScaleAddr,
                    mxQuantScaleNumAlignPerToken_);
                if constexpr (QuantMode == E2M1_QUANT) {
                    quant::ComputeFp4Data<bfloat16_t, QuantOutType,
                        AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
                        srcAddr, halfScaleAddr, outDataAddr, k_);
                } else {
                    quant::ComputeFp8Data<bfloat16_t, QuantOutType,
                        AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
                        srcAddr, halfScaleAddr, outDataAddr, k_);
                }

                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);

                if (hasMultipleServers && targetServer != myServerId) {
                    // ===== 跨 server: 写入中转区 =====
                    uint32_t otherIdx = MegaMoeDispatchImpl::ServerIdToOtherServerIdx(targetServer, myServerId);

                    // 已发送跳过
                    if (sentFlags.GetValue(tokenIdx * numOtherServers + otherIdx) != 0) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
                        continue;
                    }

                    // AtomicAdd 获取 slot index
                    GM_ADDR relayBase = MegaMoeDispatchImpl::GetRelayAreaBase(
                        params_.workspaceInfo.dispatchL1RelayDataPtr, otherIdx, l1RelayPerServerSize_);
                    __gm__ int32_t* countAddr = MegaMoeDispatchImpl::GetRelayCountAddr(relayBase);
                    int32_t slotIdx = AscendC::AtomicAdd(countAddr, 1);
                    PipeBarrier<PIPE_ALL>();

                    // 写中转区 data
                    GM_ADDR slotDataAddr = MegaMoeDispatchImpl::GetRelaySlotDataAddr(
                        relayBase, slotIdx, l1SlotSize_);
                    AscendC::GlobalTensor<uint8_t> dstDataGlobal;
                    dstDataGlobal.SetGlobalBuffer((__gm__ uint8_t*)(slotDataAddr));
                    auto xOutBytesTensor = xOutTensor.template ReinterpretCast<uint8_t>();
                    AscendC::DataCopyPad(dstDataGlobal, xOutBytesTensor, dataOutParams);
                    PipeBarrier<PIPE_ALL>();

                    // 写中转区 metadata (先写 data 再写 metadata, 保证发送核看到 flag=1 时 data 已就绪)
                    metaWriteTensor.SetValue(0, 1);          // flag = ready
                    metaWriteTensor.SetValue(1, tokenIdx);    // token_id
                    AscendC::GlobalTensor<int32_t> dstMetaGlobal;
                    dstMetaGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
                        MegaMoeDispatchImpl::GetRelaySlotMetaAddr(
                            relayBase, slotIdx, l1SlotSize_, l1SlotDataBytes_)));
                    AscendC::DataCopy(dstMetaGlobal, metaWriteTensor,
                        MegaMoeDispatchImpl::L1_META_BYTES / sizeof(int32_t));
                    PipeBarrier<PIPE_ALL>();

                    // 标记已发送
                    sentFlags.SetValue(tokenIdx * numOtherServers + otherIdx, 1);
                } else {
                    // ===== 本 server 内: 直接写入本卡 URMA 窗口 L1 接收区 (myServerId section) =====
                    // L2 通信时, 本 server 其他卡通过 ReadNbi 从本卡该位置读数据
                    GM_ADDR localUrmaBase = GetUrmaAddrByRankId(mc2Context_, rankId_) +
                        params_.peermemInfo.URMADispatchL1RecvPtr;
                    uint64_t recvSectionOffset = MegaMoeDispatchImpl::GetL1RecvSectionOffset(
                        myServerId, l1RecvPerServerSize_);
                    uint64_t dataOffset = recvSectionOffset +
                        MegaMoeDispatchImpl::GetL1RecvDataOffset(tokenIdx, l1SlotSize_);
                    uint64_t notifyOffset = recvSectionOffset +
                        MegaMoeDispatchImpl::GetL1RecvNotifyOffset(tokenIdx, l1SlotSize_);

                    // 先写 data
                    AscendC::GlobalTensor<uint8_t> dstDataGlobal;
                    dstDataGlobal.SetGlobalBuffer((__gm__ uint8_t*)(localUrmaBase + dataOffset));
                    auto xOutBytesTensor = xOutTensor.template ReinterpretCast<uint8_t>();
                    AscendC::DataCopyPad(dstDataGlobal, xOutBytesTensor, dataOutParams);
                    PipeBarrier<PIPE_ALL>();

                    // 再写 notify flag (int32 = 1), 保证 L2 读 flag=1 时 data 已就绪
                    __gm__ int32_t* notifyAddr = reinterpret_cast<__gm__ int32_t*>(
                        localUrmaBase + notifyOffset);
                    WriteGmByPassDCache(notifyAddr, 1);
                    PipeBarrier<PIPE_ALL>();
                }

                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
            }
        }

        // 排空最后两轮 MTE3 (GM 写) 完成, 保证后续写完成标志前所有数据已落盘
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

        // 写完成标志: 本计算核完成当前 expert round
        __gm__ int32_t* myCompleteAddr = MegaMoeDispatchImpl::GetCompleteListAddr(
            params_.workspaceInfo.dispatchL1CompleteListPtr, aivCoreIdx_, expertPerRank_);
        WriteGmByPassDCache(myCompleteAddr + localExpertId, 1);
    }
}

// =====================================================================================================
// DispatchLevel2：server 内二级通信 (ReadNbi)
// ----------------------------------------------------------------------------------------------------
//   数据流: 本 server 其他卡的 L1 接收区 → ReadNbi → 本卡 dispatchRevDataPtr/dispatchRevScalePtr
//
//   分核: 按 src rank 分核 (参考 combine), 每核负责一段连续 src rank
//   基于 mask 算出当前专家需要读哪些 token (GatherMask 提取 validTopkIndexTensor_)
//   逐 token: 先读 flag (WriteWithNotifyNbi 写的) 确认就绪 → ReadNbi 读 data
//   写三元组到 tripleGloablTensor_ + flag 位 AtomicAdd 通知 AIC (复用原 Phase 2 逻辑)
//
//   注意: ReadNbi dst=本地workspace, src=远端卡URMA窗口 (与 Write 相反)
//   一级二级通信间不需同步, 谁先做完一级就开始做二级
// =====================================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::DispatchLevel2(
    GMMAddrInfo &gmmAddrInfo, int32_t localExpertId)
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    if (subBlockIdx_ != 1) {
        return;
    }

    // ---------- Step 0: Server 拓扑 ----------
    uint32_t rankSizePerServer = mc2Context_->rankSizePerServer;
    bool hasMultipleServers = (rankSizePerServer > 0 && rankSizePerServer < worldSize_);
    uint32_t myServerId = hasMultipleServers
        ? MegaMoeDispatchImpl::GetServerId(rankId_, rankSizePerServer) : 0;

    // ---------- Step 1: 按 src rank 分核 (参考 combine CombineTokensWithUrmaSend) ----------
    // 每个 AIV 核 (subBlockIdx_==1) 负责一段连续的 src rank
    uint32_t aivIdx = blockIdx_;  // block 级别索引
    uint32_t aivNum = GetBlockNum();
    uint32_t rankPerCore = worldSize_ / aivNum;
    uint32_t remainder = worldSize_ % aivNum;
    uint32_t startRankId = rankPerCore * aivIdx + (aivIdx < remainder ? aivIdx : remainder);
    uint32_t processRankNum = rankPerCore + (aivIdx < remainder ? 1 : 0);
    uint32_t endRankId = startRankId + processRankNum;
    if (processRankNum == 0) {
        return;
    }

    constexpr int32_t L1_TILE_M_I32 = static_cast<int32_t>(MegaMoeImpl::L1_TILE_M_256);
    int32_t priorExpertCumsum = (localExpertId == 0) ? 0 :
        cumsumInfoTensor_.GetValue(localExpertId * worldSize_ - 1);

    int64_t widthA = k_ / ELEMS_PER_BYTE;
    int64_t widthAScale = Ops::Base::CeilDiv(static_cast<int64_t>(k_),
        static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    uint32_t slotSize = l1SlotDataBytes_ + static_cast<uint32_t>(ALIGN_32);

    // ---------- Step 2: 逐 src rank, 基于 mask 读 token ----------
    for (uint32_t rankIdx = startRankId; rankIdx < endRankId; rankIdx++) {
        // GatherMask: 从 mask 提取当前 rank 发给本专家的 valid topkIndex 列表
        uint64_t sendToCurExpTokenCnt = 0;
        GatherMask(validTopkIndexTensor_, topkIndexTensor_,
            gatherMaskInt32Tensor_[rankIdx * (maskSlotSize_ / sizeof(uint32_t))],
            true, sendTotalNum_, {1, 1, 0, 0}, sendToCurExpTokenCnt);

        // 本 rank 数据在本专家接收区的起始行
        uint32_t rowStartIdxInDst = ((rankIdx == 0 && localExpertId == 0) ? 0 :
            cumsumInfoTensor_.GetValue(localExpertId * worldSize_ + rankIdx - 1));
        if (rowStartIdxInDst >= maxOutputSize_ || sendToCurExpTokenCnt == 0) {
            continue;
        }
        uint32_t rowsCopyCnt = sendToCurExpTokenCnt;
        if (rowStartIdxInDst + rowsCopyCnt > maxOutputSize_) {
            rowsCopyCnt = maxOutputSize_ - rowStartIdxInDst;
        }
        int32_t rowDstOffsetInCore = static_cast<int32_t>(rowStartIdxInDst);
        int32_t rowsThisCore = static_cast<int32_t>(rowsCopyCnt);

        // 申请 tripleTensor_ 用于三元组组装
        uint32_t tripleTensorAddr = ubBufferUsedAddr_;
        uint32_t tripleTensorSize = rowsThisCore * ALIGN_32;
        tripleTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, tripleTensorAddr,
            tripleTensorSize / sizeof(int32_t));
        SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID4>();

        // 目标地址: dispatchRevDataPtr / dispatchRevScalePtr (给 GMM 用)
        GlobalTensor<ActivationType> tokenRevGlobalTensor;
        GlobalTensor<QuantScaleOutType> scaleRevGlobalTensor;
        tokenRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType*>(
            params_.workspaceInfo.dispatchRevDataPtr + rowDstOffsetInCore * widthA));
        scaleRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleOutType*>(
            params_.workspaceInfo.dispatchRevScalePtr + rowDstOffsetInCore * widthAScale));

        // 逐 token: 读 flag → ReadNbi/DataCopy 读 data → 拆分写 token+scale → 组装三元组
        // 统一逻辑: 确定源卡 → 从源卡 URMA 窗口 L1 接收区读
        //   sourceServerId == myServerId: 源卡 = rankIdx (发送方本身, 本 server 内通信)
        //   sourceServerId != myServerId: 源卡 = 本 server 内同 localRank 的卡 (一级通信接收卡)
        //   单 server: 源卡 = rankIdx
        for (int32_t i = 0; i < rowsThisCore; i++) {
            int32_t topkIndex = validTopkIndexTensor_.GetValue(i);
            int32_t tokenIndex = topkIndex / topK_;
            int32_t topkIdxInToken = topkIndex % topK_;

            // 三元组组装 (rank, tokenIndex, topkIndex)
            tripleTensor_[i * INT32_PER_256B].SetValue(RANK_ID, rankIdx);
            tripleTensor_[i * INT32_PER_256B].SetValue(TOKEN_ID, tokenIndex);
            tripleTensor_[i * INT32_PER_256B].SetValue(TOPK_INDEX, topkIdxInToken);

            // 确定源卡和 sourceServerId
            uint32_t sourceServerId;
            uint32_t srcRankId;
            if (hasMultipleServers) {
                sourceServerId = MegaMoeDispatchImpl::GetServerId(rankIdx, rankSizePerServer);
                if (sourceServerId == myServerId) {
                    // 本 server 内通信: 源卡 = 发送方本身
                    srcRankId = rankIdx;
                } else {
                    // 跨 server: 源卡 = 本 server 内同 localRank 的卡 (一级通信接收卡)
                    uint32_t myLocalRank = MegaMoeDispatchImpl::GetLocalRankInServer(
                        rankId_, rankSizePerServer);
                    srcRankId = MegaMoeDispatchImpl::GetL1TargetRankId(
                        myServerId, myLocalRank, rankSizePerServer);
                }
            } else {
                // 单 server: 源卡 = 发送方本身
                sourceServerId = 0;
                srcRankId = rankIdx;
            }

            // 源卡 URMA 窗口 L1 接收区基址
            GM_ADDR srcUrmaBase = GetUrmaAddrByRankId(mc2Context_, srcRankId) +
                params_.peermemInfo.URMADispatchL1RecvPtr;
            uint64_t recvSectionOffset = MegaMoeDispatchImpl::GetL1RecvSectionOffset(
                sourceServerId, l1RecvPerServerSize_);
            uint64_t srcDataOffset = recvSectionOffset +
                MegaMoeDispatchImpl::GetL1RecvDataOffset(tokenIndex, l1SlotSize_);
            uint64_t srcNotifyOffset = recvSectionOffset +
                MegaMoeDispatchImpl::GetL1RecvNotifyOffset(tokenIndex, l1SlotSize_);

            // 轮询读 flag, 确认数据就绪
            // 本卡 (srcRankId == rankId_): notifyAddr 在本卡 URMA 窗口 (本地 GM), 可直接 ReadGmByPassDCache
            // 跨卡 (srcRankId != rankId_): notifyAddr 在远端卡 URMA 窗口 (远端内存), 必须用 ReadNbi 读到本地 GM 再判断
            if (srcRankId == rankId_) {
                __gm__ int32_t* notifyAddr = reinterpret_cast<__gm__ int32_t*>(
                    srcUrmaBase + srcNotifyOffset);
                while (AscendC::ReadGmByPassDCache(notifyAddr) != 1) {
                    AscendC::Nop<200>();
                }
            } else {
                uint64_t channelHandle = GetUrmaCommHandle(mc2Context_, srcRankId, rankId_);
                // flag 占 4B, 复用 data 中转区地址 (flag 轮询完成后即被 data ReadNbi 覆盖, 不再使用)
                GM_ADDR gmFlagAddr = params_.workspaceInfo.dispatchRevDataPtr +
                    static_cast<uint64_t>(maxOutputSize_) * widthA * sizeof(ActivationType) +
                    static_cast<uint64_t>(i) * l1SlotDataBytes_;
                __gm__ int32_t* localFlagAddr = reinterpret_cast<__gm__ int32_t*>(gmFlagAddr);
                while (true) {
                    hcomm_.ReadNbi(channelHandle, gmFlagAddr,
                        srcUrmaBase + srcNotifyOffset, sizeof(int32_t));
                    hcomm_.Drain(channelHandle);
                    PipeBarrier<PIPE_ALL>();
                    if (AscendC::ReadGmByPassDCache(localFlagAddr) == 1) {
                        break;
                    }
                    AscendC::Nop<200>();
                }
            }

            // 读 data: 源卡 L1 接收区 → 本卡 dispatchRevDataPtr/dispatchRevScalePtr
            // 用 UB 中转 (copyTmpTensors_), 拆分 token 和 scale 后分别写 GM
            LocalTensor<ActivationType> tmpBuf = copyTmpTensors_[i % DISPATCH_BUFFER_NUM];
            TEventID bufEvent = static_cast<TEventID>(i % DISPATCH_BUFFER_NUM);

            if (srcRankId == rankId_) {
                // 本卡: 直接 DataCopy 从本卡 URMA 窗口读
                AscendC::GlobalTensor<ActivationType> srcGlobal;
                srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType*>(
                    srcUrmaBase + srcDataOffset));
                AscendC::DataCopy(tmpBuf, srcGlobal, l1SlotDataBytes_ / sizeof(ActivationType));
            } else {
                // 跨卡: 用 ReadNbi 从远端卡 URMA 窗口读
                // ReadNbi dst 需 GM 地址, 先读到 GM 中转区
                uint64_t channelHandle = GetUrmaCommHandle(mc2Context_, srcRankId, rankId_);
                GM_ADDR gmTmpAddr = params_.workspaceInfo.dispatchRevDataPtr +
                    static_cast<uint64_t>(maxOutputSize_) * widthA * sizeof(ActivationType) +
                    static_cast<uint64_t>(i) * l1SlotDataBytes_;
                hcomm_.ReadNbi(channelHandle, gmTmpAddr,
                    srcUrmaBase + srcDataOffset, l1SlotDataBytes_);
                // ReadNbi 是异步的, 必须 Drain 等待数据到达 GM 中转区后才能 DataCopy
                hcomm_.Drain(channelHandle);
                PipeBarrier<PIPE_ALL>();
                // 从 GM 中转区 DataCopy 到 UB
                AscendC::GlobalTensor<ActivationType> srcGlobal;
                srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType*>(gmTmpAddr));
                AscendC::DataCopy(tmpBuf, srcGlobal, l1SlotDataBytes_ / sizeof(ActivationType));
            }

            // 等 MTE2 (DataCopy GM→UB) 完成后再拆分写
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(bufEvent);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(bufEvent);

            // 拆分写 token 和 scale 到 GM (给 GMM 用)
            LocalTensor<QuantScaleOutType> bufScale =
                tmpBuf[mxQuantTokenAlignBytes_].template ReinterpretCast<QuantScaleOutType>();
            AscendC::DataCopyPad(tokenRevGlobalTensor[i * widthA], tmpBuf,
                {1, static_cast<uint16_t>(widthA * sizeof(ActivationType)), 0U, 0U, 0U});
            AscendC::DataCopyPad(scaleRevGlobalTensor[i * widthAScale], bufScale,
                {1, static_cast<uint16_t>(widthAScale * sizeof(QuantScaleOutType)), 0U, 0U, 0U});
        }

        // Phase 2: 三元组搬出 + flag 位 AtomicAdd (通知 AIC), 复用原逻辑
        if (rowsThisCore > 0) {
            SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID5>();
            // 三元组搬出
            SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();
            DataCopy(tripleGloablTensor_[rowDstOffsetInCore * INT32_PER_256B],
                tripleTensor_, rowsThisCore * INT32_PER_256B);

            // flag 位 AtomicAdd
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

// ==============================================================================================
// CrossRankSyncInWorldSize：全卡同步，rankSyncInWorldPtr前48K用于同步，后面区域用于记录当前syncCnt值
// ==============================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::CrossRankSyncInWorldSize()
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
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::Process()
{
    // 1.本卡数据处理
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    SendAndQuantBuffInit();
    SendMaskCal();               // 源卡按所有全局专家算 mask 并推送到目标专家卡
    ResetFlagList();             // 清理workSpace空间上的flag位
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        ResetRowGroupCompleteFlags();
    }
    // QuantProcessInRank();  // L1 dispatch 模式: 量化在发送时完成, 不再全量预量化
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
                // L1 dispatch: 跨server URMA 发送 (计算核量化+写入中转区, 发送核轮询+WriteWithNotifyNbi)
                DispatchLevel1(localExpertId);
            }
        }
        if (!UpdateGroupParams<AddrUpdateMode::kGmm1>(gmm1State, localExpertId)) {  // 更新偏移
            continue;
        }
        UpdateGlobalBuffer<AddrUpdateMode::kGmm1>(gmmAddrInfo, gmm1State); // 更新地址
        if constexpr(g_coreType == AIV) {
            if (subBlockIdx_ == 1) {
                DispatchLevel2(gmmAddrInfo, localExpertId); // L2 dispatch: server内 ReadNbi 读数据 + 三元组 + flag
            }
        }
        if (subBlockIdx_ == 0) {
            MegaMoeImpl::GroupMatmulSwigluQuant<
                QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
                epilogueOp_, params_, gmm1State.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_);
        }
    }
    // EndSync(vecSetSyncCom_);
    // if constexpr(g_coreType == AIV) {
    //     if (subBlockIdx_ == 1) {
    //         ExpertTokenNumCopyOut(); // 本卡专家接受的tokenCnt总数搬出
    //     }
    // }

    // // 3. 本卡专家接收数据GroupMatmul2 & Combine
    // vecSetSyncCom_ = 0;
    // expertBeforeCnt_ = 0;
    // ExpertLoopState gmm2State{initShape, initOffset};
    // InitCombineBuffers();

    // if constexpr (g_coreType == AIV) {
    //     if (CombineQuantMode != COMBINE_NO_QUANT || subBlockIdx_ == 1) {
    //         LocalTensor<uint8_t> hcommTensor_=LocalTensor<uint8_t>(TPosition::VECCALC, 0, ALIGN_512 / sizeof(uint8_t));
    //         hcomm_.Init(hcommTensor_, ALIGN_512);
    //         if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
    //             MegaMoeCombineImpl::CombineTokensWithUrmaSend<bfloat16_t>(params_, mc2Context_, hcomm_);
    //         }
    //     }
    // }

    // for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
    //     if (!UpdateGroupParams<AddrUpdateMode::kGmm2>(gmm2State, expertIdx)) {
    //         continue;
    //     }
    //     UpdateGlobalBuffer<AddrUpdateMode::kGmm2>(gmmAddrInfo, gmm2State);
        
    //     MegaMoeImpl::GroupMatmul2<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t,
    //         QuantScaleOutType, QuantScaleOutType>(
    //         params_, gmm2State.problemShape, gmmAddrInfo, startBlockIdx_,
    //         vecSetSyncCom_, expertBeforeCnt_, gmm2PingPongIdx_, expertIdx, maxRowGroupsStride_);
        
    //     if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV) {
    //         //ProcessCombine(gmmAddrInfo, gmm2State, expertIdx);
    //         ProcessCombineByUrma(gmmAddrInfo, gmm2State, expertIdx);
    //     }
    // }
    // if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
    //     EndGMM2Sync(vecSetSyncCom_, gmm2PingPongIdx_);
    // }
    // PipeBarrier<PIPE_ALL>();
    // SyncAll<true>();

    // // 4. 本卡数据Unpermute
    // if constexpr(g_coreType == AIV) {
    //     UnpermuteBuffInit();
    //     CrossRankSyncInWorldSize(); // 全卡软同步，确认combine send完成
    //     Unpermute();
    // }
    // SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

}   // namespace MegaMoeImpl
#endif
