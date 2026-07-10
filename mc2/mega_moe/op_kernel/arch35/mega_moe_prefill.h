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
 * \file mega_moe_prefill.h
 * \brief MegaMoe 大 bs Prefill 场景适配实现。
 *        对 dispatch 路径的 UB 做分块流式处理（route batch / triple tile / reset batch /
 *        unpermute token batch），将 UB 峰值从 O(bs*topK) 降为 O(batchLen)，与大 bs 解耦。
 *        其余 GMM/combine 等路径保持与主线 mega_moe.h 一致。
 */

#ifndef MEGA_MOE_PREFILL_H
#define MEGA_MOE_PREFILL_H

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
                            int64_t, int64_t, int64_t, int64_t>;

#define TemplateMegaMoePrefillTypeClass typename XType, typename OutputType, typename TopkWeightsType, \
                                         typename Weight1Type, int32_t QuantMode, int32_t CombineQuantMode
#define TemplateMegaMoePrefillTypeFunc XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode

template <TemplateMegaMoePrefillTypeClass>
class MegaMoePrefill {
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
        uint32_t expertBeforeCnt = 0;
    };
    __aicore__ inline MegaMoePrefill() {};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights,
        GM_ADDR weight1, GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
        GM_ADDR scales, GM_ADDR yOut, GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM,
        MegaMoeTilingData *tilingData);
    __aicore__ inline void Process();

private:
    // ── 大 bs 适配：变更的函数 ──
    __aicore__ inline void DispatchBuffInit();
    __aicore__ inline void SendAndQuantBuffInit();
    __aicore__ inline void ResetFlagList();
    __aicore__ inline void SendMaskCal();
    __aicore__ inline void SendCntCal(int32_t localExpertId, uint64_t& sendCnt);
    __aicore__ inline void TripleInfoCalAndDispatch(GMMAddrInfo &gmmAddrInfo, int32_t localExpertId);
    __aicore__ inline void Unpermute();
    __aicore__ inline void UnpermuteLoadWeights(int32_t coreOffset, int32_t batchStart, int32_t curTokens,
        LocalTensor<bfloat16_t>& tempLocal);
    __aicore__ inline void UnpermuteProcessToken(int32_t tokenIdx, int32_t localIdx,
        const GlobalTensor<bfloat16_t>& expandedX);

    // ── 不变函数（与主线 mega_moe.h 一致）──
    __aicore__ inline void ResetGmm2CombineSyncCounters();
    template <AddrUpdateMode Mode>
    __aicore__ inline bool UpdateGroupParams(ExpertLoopState &state, uint32_t expertIdx,
        uint64_t sendCnt = 0);
    template <AddrUpdateMode Mode>
    __aicore__ inline void UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state);
    __aicore__ inline void InitCombineBuffers();
    __aicore__ inline void ProcessCombine(const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &gmm2State,
        uint32_t expertIdx);
    __aicore__ inline void CrossRankSyncInWorldSize();
    __aicore__ inline void ExpertTokenNumCopyOut();
    __aicore__ inline void CopyGMToGMPerToken(int32_t rowDstOffsetInCore, int32_t remoteRankIdx,
        int32_t copyStartIdx, int32_t copyNum);
    __aicore__ inline void QuantProcessInRank();
    __aicore__ inline void GroupMatmulWithSwigluQuant(const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state);
    __aicore__ inline void GroupMatmulWithCombine(const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state,
        uint32_t expertIdx);

    // ── 成员变量 ──
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

    // ── 大 bs 专用成员 ──
    // 分块流式 dispatch: 按 route batch 分批处理，UB 峰值从 O(bs*topK) 降为 O(batchLen)。
    int32_t sendRouteBatchLen_ = 0;   // 发侧每 batch 的 route item 数
    int32_t sendRouteBatchNum_ = 0;   // 发侧 batch 总数
    int32_t recvRouteBatchLen_ = 0;   // 收侧每 batch 的 route item 数
    int32_t recvRouteBatchNum_ = 0;   // 收侧 batch 总数
    int32_t resetBatchNum_ = 0;       // reset 分批粒度（封顶到 DISPATCH_RESET_BATCH）

    static constexpr uint32_t A_ELEMS_PER_BYTE = Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value ? 2U : 1U;
    static constexpr uint32_t B_ELEMS_PER_BYTE = Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value ? 2U : 1U;
    static constexpr bool ENABLE_A8W4 = Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value &&
                                            Std::IsSame<QuantOutType, fp8_e4m3fn_t>::value;
    static constexpr bool ENABLE_A4W4 = Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value &&
                                            Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value;
    static constexpr int32_t DISPATCH_BUFFER_NUM = 6;
    static constexpr int32_t UNPERMUTE_BATCH_LEN = 1024;  // k∈[1,8192], topK∈[2,8], 每 token topK×6B, 1024 安全
    static constexpr int32_t DISPATCH_RECV_BATCH_LEN = 12288;  // 收侧每 batch route item 数，最坏 ws=1024 时 UB 占用 75%

    // ── UB tensor（与主线共用名，但大 bs 路径下大小按 batch 级而非全量 routeItemNum）──
    LocalTensor<int32_t> topkIndexTensor_;          // 收侧: 每 batch 重建的全局 route item index
    LocalTensor<int32_t> sendCntTensor_;            // 收侧: SendCntCal stride 只读 ws 个 count（替代 gatherMaskTensor_）
    LocalTensor<uint8_t> maskBatchTensor_;          // 收侧: 当前 batch 的 mask 切片
    LocalTensor<uint32_t> maskBatchU32Tensor_;      // maskBatchTensor_ 的 u32 视图，供 GatherMask
    LocalTensor<int32_t> expertTokenCntTensor_;
    LocalTensor<int32_t> validTopkIndexTensor_;     // 收侧: 每 batch 命中列表
    LocalTensor<int32_t> cumsumInfoTensor_;
    LocalTensor<ActivationType> copyTmpTensors_[DISPATCH_BUFFER_NUM];
    LocalTensor<int32_t> tripleTensor_;
    LocalTensor<bfloat16_t> xInTensor1_;
    LocalTensor<bfloat16_t> xInTensor2_;
    LocalTensor<ActivationType> xOutTensor1_;
    LocalTensor<ActivationType> xOutTensor2_;
    LocalTensor<uint16_t> mxTempTensor_;
    LocalTensor<int32_t> resetTensor_;
    LocalTensor<int32_t> topkIdsTensor_;            // 发侧: route batch 级 topk（大 bs 下按 sendRouteBatchLen_ 分配）
    LocalTensor<uint8_t> sendMaskTensor_[DOUBLE_BUFFER];  // 发侧: batch 级 mask 切片 + 32B count 尾
    LocalTensor<int32_t> sendGatherOutTensor_;       // 发侧: batch 级 GatherMask scratch
    LocalTensor<int32_t> sendCntAccTensor_;          // 发侧: per-owned-expert 跨 batch count 累加器
    LocalTensor<int32_t> expertTokenNumsOutTensor_;
    LocalTensor<bfloat16_t> dataResTensor_;
    LocalTensor<float> dataResFp32Tensor_;
    LocalTensor<float> topKWeightsTensor_;
    LocalTensor<float> fp32ScaleTensor_;
    LocalTensor<bfloat16_t> bf16ScaleTensor_;

    using SwigluQuantOutType = typename std::conditional<
        (QuantMode == E2M1_QUANT),
        fp8_e4m3fn_t, QuantOutType>::type;
    static constexpr uint32_t C_ELEMS_PER_BYTE = Std::IsSame<SwigluQuantOutType, fp4x2_e2m1_t>::value ? 2U : 1U;

    using BlockEpilogue = BlockEpilogueSwigluMxQuant<SwigluQuantOutType, bfloat16_t,
        QuantScaleOutType, QuantScaleOutType, true>;
    BlockEpilogue epilogueOp_;
};

// ========================
// Init：初始化 & 偏移计算（与主线一致）
// ========================
template <TemplateMegaMoePrefillTypeClass>
__aicore__ inline void MegaMoePrefill<TemplateMegaMoePrefillTypeFunc>::Init(
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
    // compareCount_ = CeilAlign(sendTotalNum_, 64)         — 按 64 个 int32(256B) 对齐，供 CompareScalar
    // maskAlignSize_ = CeilAlign(sendTotalNum_ / 8, 32)    — mask 区字节数，按 32B burst 对齐
    // batch (sendRouteBatchLen_/recvRouteBatchLen_) 强制 256 对齐，保证:
    //   每 batch mask slice = batchLen/8 = 32B 的整数倍 = 刚好对齐 maskAlignSize_ 的 32B burst 边界
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

// ======================================================================================
// SendAndQuantBuffInit：发侧 UB 布局 + 反推 sendRouteBatchLen_
// ======================================================================================
template <TemplateMegaMoePrefillTypeClass>
__aicore__ inline void MegaMoePrefill<TemplateMegaMoePrefillTypeFunc>::SendAndQuantBuffInit()
{
    if constexpr(g_coreType == AIC) {
        return;
    }

    // ── 与 route batch 无关的固定占用 ──
    uint32_t resetTensorAddr = 0;  // 占位，等下面算完 batchLen 再落
    uint64_t totalFlagInt32 = static_cast<uint64_t>(expertPerRank_) *
        (static_cast<uint64_t>(INT_CACHELINE) + static_cast<uint64_t>(dispatchFlagSlotsPerExpert_) +
        static_cast<uint64_t>(INT_CACHELINE) * static_cast<uint64_t>(aicNum_));
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        int64_t tokenGroupResetSize = static_cast<int64_t>(expertPerRank_) * blockAivNum_ * INT_CACHELINE;
        totalFlagInt32 = (static_cast<int64_t>(totalFlagInt32) > tokenGroupResetSize)
                         ? static_cast<int64_t>(totalFlagInt32) : tokenGroupResetSize;
    }
    uint32_t resetPerCore = Ops::Base::CeilDiv(totalFlagInt32, static_cast<uint64_t>(blockAivNum_));
    resetBatchNum_ = resetPerCore < static_cast<uint32_t>(DISPATCH_RESET_BATCH) ?
        static_cast<int32_t>(resetPerCore) : DISPATCH_RESET_BATCH;
    uint32_t resetTensorSize = Ops::Base::CeilAlign(static_cast<uint64_t>(resetBatchNum_),
        static_cast<uint64_t>(INT32_PER_256B)) * sizeof(int32_t);

    uint32_t mxTempTensorSize = 2 * 1024;
    uint32_t xOutTokenBytes = Ops::Base::CeilAlign(static_cast<uint32_t>(k_ / A_ELEMS_PER_BYTE),
        static_cast<uint32_t>(ALIGN_256));
    uint32_t xOutTensorSize = xOutTokenBytes + Ops::Base::CeilDiv(k_, static_cast<uint32_t>(ALIGN_32));
    uint32_t xInAlignSize = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_128)) * sizeof(bfloat16_t);
    // sendCntAccTensor_: per-owned-expert 跨 batch count 累加器
    uint32_t expertPerCoreMax = Ops::Base::CeilDiv(worldSize_ * expertPerRank_, blockAivNum_);
    uint32_t sendCntAccSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(expertPerCoreMax * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    uint32_t sendMaskFixedBytes = static_cast<uint32_t>(DOUBLE_BUFFER) * static_cast<uint32_t>(ALIGN_32);
    uint32_t fixedBytes = resetTensorSize + mxTempTensorSize + 2 * xOutTensorSize + 2 * xInAlignSize +
        sendCntAccSize + sendMaskFixedBytes;

    // ── 用 UB 余量反推 sendRouteBatchLen_ ──
    int64_t ubAvailable = static_cast<int64_t>(params_.tilingData->ubSize) - DISPATCH_UB_SAFETY;
    int64_t ubForBatch = ubAvailable - static_cast<int64_t>(fixedBytes);
    // 随 batch 线性增长: topk(4B) + sendGatherOut(4B) + sendMask 双buf(DOUBLE_BUFFER*1/8B)
    // batchLen * ((4+4) + DOUBLE_BUFFER/8) ≤ ubForBatch
    int64_t batchLenByUb = ubForBatch * 8 / ((4 + 4) * 8 + static_cast<int64_t>(DOUBLE_BUFFER));
    int64_t batchLenCap = Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_), static_cast<int64_t>(ALIGN_256));
    int64_t batchLen = batchLenByUb < batchLenCap ? batchLenByUb : batchLenCap;
    batchLen = batchLen / ALIGN_256 * ALIGN_256;
    batchLen = batchLen < ALIGN_256 ? ALIGN_256 : batchLen;
    sendRouteBatchLen_ = static_cast<int32_t>(batchLen);
    sendRouteBatchNum_ = static_cast<int32_t>(
        Ops::Base::CeilDiv(sendTotalNum_, static_cast<uint64_t>(sendRouteBatchLen_)));

    // ── 按既定顺序落地址 ──
    // Tensor用处：SendMaskCal 函数中搬运本卡当前 batch 的 topkIds；
    // Tensor大小：sendRouteBatchLen_ * sizeof(int32_t) align 到 256 字节对齐；
    uint32_t topkIdsTensorAddr = 0;
    uint32_t topkIdsTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(sendRouteBatchLen_) * static_cast<int64_t>(sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_256));
    topkIdsTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, topkIdsTensorAddr, topkIdsTensorSize / sizeof(int32_t));

    // Tensor用处：ResetFlagList 中对于 workSpace 上的 Flag 位置区域进行清理；
    // Tensor大小：大小为清理区域大小均分到所有的 blockAivNum_，封顶到 DISPATCH_RESET_BATCH；
    uint32_t resetAddrActual = topkIdsTensorAddr + topkIdsTensorSize;
    resetTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, resetAddrActual, resetTensorSize / sizeof(int32_t));
    Duplicate<int32_t>(resetTensor_, 0, (resetTensorSize / sizeof(int32_t)));

    // Tensor用处：QuantProcessInRank 函数中用于量化计算中间区域；
    // Tensor大小：当前申请 2K；
    uint32_t mxTempTensorAddr = resetAddrActual + resetTensorSize;
    mxTempTensor_ = LocalTensor<uint16_t>(TPosition::VECCALC, mxTempTensorAddr, mxTempTensorSize / sizeof(uint16_t));
    // Tensor用处：QuantProcessInRank 函数中用于存储量化输出结果；
    // Tensor大小：token 长度 Align256 字节对齐 + scale 存储长度，xOutTensor1_ 与 xOutTensor2_ 为双 buffer；
    uint32_t xOutTensorAddr1 = mxTempTensorAddr + mxTempTensorSize;
    xOutTensor1_ = LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr1,
        xOutTensorSize / sizeof(ActivationType));
    uint32_t xOutTensorAddr2 = xOutTensorAddr1 + xOutTensorSize;
    xOutTensor2_ = LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr2,
        xOutTensorSize / sizeof(ActivationType));
    // Tensor用处：QuantProcessInRank 函数中用于存储输入 token；
    // Tensor大小：k_ Align128 字节对齐，xInTensor1_ 与 xInTensor2_ 为双 buffer；
    uint32_t xInAlignAddr1 = xOutTensorAddr2 + xOutTensorSize;
    xInTensor1_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr1, xInAlignSize / sizeof(bfloat16_t));
    uint32_t xInAlignAddr2 = xInAlignAddr1 + xInAlignSize;
    xInTensor2_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr2, xInAlignSize / sizeof(bfloat16_t));

    // Tensor用处：SendMaskCal 函数中用于存储 mask 位；
    // Tensor大小：各 batch/8 mask 切片 + 32B（末块折叠 count），DOUBLE_BUFFER 为 2，开启双 buffer；
    uint32_t maskBufBytes = static_cast<uint32_t>(sendRouteBatchLen_ / 8) + static_cast<uint32_t>(ALIGN_32);
    uint32_t sendMaskAddr = xInAlignAddr2 + xInAlignSize;
    for (int32_t index = 0; index < DOUBLE_BUFFER; ++index) {
        sendMaskTensor_[index] = LocalTensor<uint8_t>(TPosition::VECCALC,
            sendMaskAddr + static_cast<uint32_t>(index) * maskBufBytes, maskBufBytes);
    }
    // Tensor用处：SendMaskCal 函数中用于 GatherMask 的 dst Tensor；
    // Tensor大小：sendRouteBatchLen_ * sizeof(int32_t) align 到 256 字节对齐；
    uint32_t sendGatherOutAddr = sendMaskAddr + static_cast<uint32_t>(DOUBLE_BUFFER) * maskBufBytes;
    uint32_t sendGatherOutSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(sendRouteBatchLen_) * static_cast<int64_t>(sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_256));
    sendGatherOutTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, sendGatherOutAddr,
        sendGatherOutSize / sizeof(int32_t));
    // Tensor用处：SendMaskCal 函数中 per-owned-expert 跨 batch count 累加器；
    // Tensor大小：expertPerCoreMax * sizeof(int32_t) align 到 32 字节对齐；
    uint32_t sendCntAccAddr = sendGatherOutAddr + sendGatherOutSize;
    sendCntAccTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, sendCntAccAddr,
        sendCntAccSize / sizeof(int32_t));
}

// ======================================================================================================
// SendMaskCal：发侧 mask 生成 & 推送到各 dst rank 的 peer window。
//
// 核心思路：大 bs 下 topk 全量常驻 UB 不可行，改为 route batch 流式处理。
//
// 循环结构（两层）：
//   外层 batchIdx — 按 sendRouteBatchLen_ 分块遍历 topkIds（每个 batch 只加载一批 topk 进 UB）
//   内层 ownedIdx — 对本 core 负责的每个 owned expert，生成该 expert 在此 batch 上的 mask
//
// 流水（双缓冲 mask push）：
//   batch e 的 topk 加载 (V_MTE2→MTE2_V) 与 batch e-1 的 per-expert mask 生成+推送 (V_S→...→MTE3_V) 重叠。
//   kBufEvents[DOUBLE_BUFFER] 控制 MTE3 write 完成事件，保证双 buf 交替使用不冲突。
//
// 关键细节：
//   - 非末 batch: pushBytes = sliceBytes（纯 mask 切片）
//   - 末 batch:   pushBytes = sliceBytes + 4B；末尾多写一个 int32 是该 expert 跨 batch 的累计 count
//                 （SendCntCal 收侧通过 maskSlotSize 跳过 mask 区直接读 count，无需再翻 mask）
//   - sendCntAccTensor_[ownedIdx]: per-expert 跨 batch 累加计数，末 batch 折叠进 mask 尾部
//   - peer window 地址: maskWinOffset_ + expert*srcRank*(mask+count slot) + batchStart/8 偏移
// ======================================================================================================
template <TemplateMegaMoePrefillTypeClass>
__aicore__ inline void MegaMoePrefill<TemplateMegaMoePrefillTypeFunc>::SendMaskCal()
{
    if constexpr(g_coreType == AIC) {
        return;
    }

    // ── owned expert: 本 AIV core 负责的 global expert 子集 ──
    int32_t totalExperts = static_cast<int32_t>(worldSize_ * expertPerRank_);
    int32_t coreIdx = static_cast<int32_t>(aivCoreIdx_);
    int32_t ownedExpertNum = (coreIdx < totalExperts) ?
        Ops::Base::CeilDiv(totalExperts - coreIdx, static_cast<int32_t>(blockAivNum_)) : 0;
    if (ownedExpertNum <= 0) {
        return;
    }

    // ── 准备 GM 读写句柄 ──
    GlobalTensor<int32_t> srcGlobalTensor;
    srcGlobalTensor.SetGlobalBuffer((__gm__ int32_t*)params_.expertIdxGmAddr);  // topkIds GM
    GlobalTensor<uint8_t> dstGlobalTensor;  // peer window 写目标（按 rank 切换）
    int32_t maskSliceBytesFull = sendRouteBatchLen_ / 8;          // 每 batch bitmask 字节数（1bit/item）
    DataCopyPadExtParams<int32_t> loadPad{true, 0U, 0U, -1};   // 加载 topk 时 pad -1

    // ── per-expert 跨 batch 累加器清零 ──
    Duplicate<int32_t>(sendCntAccTensor_, 0, ownedExpertNum);
    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();

    // ── 双缓冲 mask push 事件：初始全部 set，表示两个 buf 都空闲可写入 ──
    constexpr TEventID kBufEvents[DOUBLE_BUFFER] = {EVENT_ID0, EVENT_ID1};
    for (int32_t bufIdx = 0; bufIdx < static_cast<int32_t>(DOUBLE_BUFFER); ++bufIdx) {
        SetFlag<AscendC::HardEvent::MTE3_V>(kBufEvents[bufIdx]);
    }

    int32_t iter = 0;  // 全局迭代计数，用于 ping-pong 选 buf 和事件
    // ═══════════════════ 外层：route batch 循环 ═══════════════════
    for (int32_t batchIdx = 0; batchIdx < sendRouteBatchNum_; ++batchIdx) {
        int32_t batchStart = batchIdx * sendRouteBatchLen_;    // 当前 batch 在 topkIds 中的起始位置
        bool isLastBatch = (batchIdx == sendRouteBatchNum_ - 1);
        int32_t validLen = sendRouteBatchLen_;                  // 本 batch 有效 route item 数
        int32_t sliceBytes = maskSliceBytesFull;                // 本 batch mask 切片字节数
        int32_t pushBytes = sliceBytes;                         // 实际推送到 peer window 的字节数

        if (isLastBatch) {
            // 末 batch 可能不满 sendRouteBatchLen_，调整有效长度和 mask 切片大小
            validLen = static_cast<int32_t>(sendTotalNum_) - batchStart;
            if (batchStart / 8 + sliceBytes > static_cast<int32_t>(maskAlignSize_)) {
                sliceBytes = static_cast<int32_t>(maskAlignSize_) - batchStart / 8;
            }
            // 末 batch 多推一个 int32：该 expert 的最终累计 count，收侧 SendCntCal 直接读
            pushBytes = sliceBytes + static_cast<int32_t>(sizeof(int32_t));
            // 尾部的 topkIdsTensor_ 填充 -1，避免 CompareScalar 误匹配
            Duplicate<int32_t>(topkIdsTensor_, -1, sendRouteBatchLen_);
        }
        DataCopyExtParams maskCopyParams{1U, static_cast<uint32_t>(pushBytes), 0U, 0U, 0U};

        // ── 加载本 batch 的 topk（V_MTE2 → MTE2_V = topk load 与 mask push 流水重叠）──
        SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID1>();
        DataCopyExtParams loadParams{1U, static_cast<uint32_t>(validLen * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(topkIdsTensor_, srcGlobalTensor[batchStart], loadParams, loadPad);
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();  // topk 加载完成

        // ═══════════════════ 内层：per-expert 循环 ═══════════════════
        for (int32_t ownedIdx = 0; ownedIdx < ownedExpertNum; ++ownedIdx, ++iter) {
            int32_t globalExpertId = coreIdx + ownedIdx * static_cast<int32_t>(blockAivNum_);
            int32_t dstRank = globalExpertId / static_cast<int32_t>(expertPerRank_);      // 目标卡
            int32_t localExpertId = globalExpertId % static_cast<int32_t>(expertPerRank_); // 目标卡上第几个 expert

            // ping-pong 选 buf：iter % DOUBLE_BUFFER
            TEventID ev = kBufEvents[iter % static_cast<int32_t>(DOUBLE_BUFFER)];
            LocalTensor<uint8_t> maskBuf = sendMaskTensor_[iter % static_cast<int32_t>(DOUBLE_BUFFER)];
            LocalTensor<uint32_t> maskBufU32 = maskBuf.template ReinterpretCast<uint32_t>();

            // ── Step 1: 等上次 push 完成 → 生成 mask ──
            WaitFlag<AscendC::HardEvent::MTE3_V>(ev);
            // CompareScalar: topkIdsTensor_[i] == globalExpertId → maskBuf bit[i]=1
            CompareScalar(maskBuf, topkIdsTensor_, globalExpertId, AscendC::CMPMODE::EQ, sendRouteBatchLen_);
            uint64_t sendCnt = 0;
            // GatherMask: 从 mask 中提取命中位置到 sendGatherOutTensor_，返回命中数 sendCnt
            GatherMask(sendGatherOutTensor_, topkIdsTensor_, maskBufU32, true,
                static_cast<uint32_t>(validLen), {1, 1, 0, 0}, sendCnt);

            // ── Step 2: 累加 count → 末 batch 折叠进 mask 尾部 ──
            SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
            int32_t acc = sendCntAccTensor_.GetValue(ownedIdx) + static_cast<int32_t>(sendCnt);
            sendCntAccTensor_.SetValue(ownedIdx, acc);
            if (isLastBatch) {
                // mask 末尾的 int32 位置写入最终累计 count（收侧 SendCntCal 在此读取）
                maskBuf.template ReinterpretCast<int32_t>().SetValue(sliceBytes / sizeof(int32_t), acc);
            }
            SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();

            // ── Step 3: 推送 mask 切片到 dst rank 的 peer window ──
            // peer window 布局: maskWinOffset_
            //   + localExpertId * worldSize_ * maskSlotSize_  (expert 偏移)
            //   + rankId_ * maskSlotSize_                       (本 src rank 的 slot)
            //   + batchStart / 8                                (mask 字节级偏移)
            uint64_t dstOffset = maskWinOffset_ + static_cast<uint64_t>(
                localExpertId * static_cast<int32_t>(worldSize_) + static_cast<int32_t>(rankId_)) *
                static_cast<uint64_t>(maskSlotSize_) + static_cast<uint64_t>(batchStart / 8);
            dstGlobalTensor.SetGlobalBuffer(
                (__gm__ uint8_t*)GetRankWinAddrWithOffset(dstRank, dstOffset));
            DataCopyPad(dstGlobalTensor, maskBuf, maskCopyParams);
            SetFlag<AscendC::HardEvent::MTE3_V>(ev);  // 标记该 buf push 完成，可复用
        }
    }

    // ── 等待最后一批 push 全部完成 ──
    for (int32_t bufIdx = 0; bufIdx < static_cast<int32_t>(DOUBLE_BUFFER); ++bufIdx) {
        WaitFlag<AscendC::HardEvent::MTE3_V>(kBufEvents[bufIdx]);
    }
}

// ===============================================================================================
// ResetFlagList：大 bs 分批清零（封顶到 DISPATCH_RESET_BATCH）
// ===============================================================================================
template <TemplateMegaMoePrefillTypeClass>
__aicore__ inline void MegaMoePrefill<TemplateMegaMoePrefillTypeFunc>::ResetFlagList()
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    swigluToGmm2FlagGm_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.flagSwiGluToGmm2Ptr);
    int32_t flagNum = static_cast<int32_t>(expertPerRank_) *
        (static_cast<int32_t>(INT_CACHELINE) + dispatchFlagSlotsPerExpert_ +
        static_cast<int32_t>(INT_CACHELINE) * static_cast<int32_t>(aicNum_));
    int32_t coreLen, coreOffset;
    TilingByCore(flagNum, coreLen, coreOffset, 1);
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();

    for (int32_t off = 0; off < coreLen; off += resetBatchNum_) {
        int32_t cur = coreLen - off < resetBatchNum_ ? coreLen - off : resetBatchNum_;
        DataCopyExtParams rankSyncCopyParams{1U, static_cast<uint32_t>(cur * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(swigluToGmm2FlagGm_[coreOffset + off], resetTensor_, rankSyncCopyParams);
    }

    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        ResetGmm2CombineSyncCounters();
    }
}

// =============================================
// ResetGmm2CombineSyncCounters（分批清零，与 ResetFlagList 一致）
// =============================================
template <TemplateMegaMoePrefillTypeClass>
__aicore__ inline void MegaMoePrefill<TemplateMegaMoePrefillTypeFunc>::ResetGmm2CombineSyncCounters()
{
    if constexpr(g_coreType == AIV) {
        int32_t totalCounters = static_cast<int32_t>(
            static_cast<int64_t>(expertPerRank_) * blockAivNum_ * INT_CACHELINE);
        int32_t coreLen, coreOffset;
        TilingByCore(totalCounters, coreLen, coreOffset);
        GlobalTensor<int32_t> gmm2CombineSyncCounterGm;
        gmm2CombineSyncCounterGm.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.gmm2CombineSyncCounterPtr);
        if (coreLen > 0) {
            Duplicate(resetTensor_, 0, resetBatchNum_);
            SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
            for (int32_t off = 0; off < coreLen; off += resetBatchNum_) {
                int32_t cur = coreLen - off < resetBatchNum_ ? coreLen - off : resetBatchNum_;
                DataCopy(gmm2CombineSyncCounterGm[coreOffset + off], resetTensor_, cur);
            }
        }
    }
}

// ===================================
// QuantProcessInRank（与主线一致）
// ===================================
template <TemplateMegaMoePrefillTypeClass>
__aicore__ inline void MegaMoePrefill<TemplateMegaMoePrefillTypeFunc>::QuantProcessInRank()
{
    if constexpr(g_coreType == AIC) { return; }
    int32_t curentNum; int32_t curentOffset;
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
        quant::ComputeMaxExp(srcAddr, maxExpAddr, H);
        quant::ComputeScale<QuantOutType>(maxExpAddr, mxScaleAddr, halfScaleAddr, mxQuantScaleNumAlignPerToken_);
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

// ==============================================================================================
// CrossRankSyncInWorldSize（与主线一致）
// ==============================================================================================
template <TemplateMegaMoePrefillTypeClass>
__aicore__ inline void MegaMoePrefill<TemplateMegaMoePrefillTypeFunc>::CrossRankSyncInWorldSize()
{
    if constexpr(g_coreType == AIC) { return; }
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

// =================================================================================================
// DispatchBuffInit：收侧 UB 布局 + 反推 recvRouteBatchLen_
// =================================================================================================
template <TemplateMegaMoePrefillTypeClass>
__aicore__ inline void MegaMoePrefill<TemplateMegaMoePrefillTypeFunc>::DispatchBuffInit()
{
    if constexpr(g_coreType == AIC) {
        return;
    }

    // ── 与 route batch 无关的固定占用 ──
    uint32_t expertTokenCntTensorSize = ALIGN_32;
    uint32_t cumsumInfoTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(worldSize_ * expertPerRank_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    // sendCntTensor_: 每 src rank 一个 burst(32B), 共 ws*32B
    uint32_t sendCntTensorSize = worldSize_ * static_cast<uint32_t>(ALIGN_32);
    // copyTmpTensors_ 6 个 dispatch buffer, 各 tokenScaleSize
    uint32_t tokenScaleSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_),
        static_cast<int64_t>(ALIGN_32));
    uint32_t COPY_TMP_BUFFER_SIZE = tokenScaleSize;
    uint32_t copyTmpTotalSize = static_cast<uint32_t>(DISPATCH_BUFFER_NUM) * COPY_TMP_BUFFER_SIZE;
    uint32_t expertTokenNumsOutTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(expertPerRank_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    // triple ring buffer: DISPATCH_BUFFER_NUM 条 * 32B，逐 token 即时写 GM
    uint32_t tripleReserveSize = static_cast<uint32_t>(DISPATCH_BUFFER_NUM) *
        static_cast<uint32_t>(INT32_PER_256B) * sizeof(int32_t);

    // ── 固定 batchLen（后续以模板参数形式传入）──
    recvRouteBatchLen_ = DISPATCH_RECV_BATCH_LEN;
    if (recvRouteBatchLen_ > static_cast<int32_t>(sendTotalNum_)) {
        recvRouteBatchLen_ = static_cast<int32_t>(sendTotalNum_);
    }
    recvRouteBatchLen_ = static_cast<int32_t>(Ops::Base::CeilAlign(
        static_cast<int64_t>(recvRouteBatchLen_), static_cast<int64_t>(ALIGN_256)));
    recvRouteBatchNum_ = static_cast<int32_t>(
        Ops::Base::CeilDiv(sendTotalNum_, static_cast<uint64_t>(recvRouteBatchLen_)));

    // ── 按既定顺序落地址 ──
    // Tensor用处：SendCntCal 函数中记录本卡各专家收到的 token 总数；
    // Tensor大小：仅记录 count 值且各专家之间复用，申请大小为 32 字节；
    uint32_t expertTokenCntTensorAddr = 0;
    expertTokenCntTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, expertTokenCntTensorAddr,
        expertTokenCntTensorSize / sizeof(int32_t));
    // Tensor用处：SendCntCal 函数中记录本卡专家收到 token count 的 cumsum 累加值；
    // Tensor大小：worldSize_ * expertPerRank_ * sizeof(int32_t) align 至 32 字节对齐；
    uint32_t cumsumInfoTensorAddr = expertTokenCntTensorAddr + expertTokenCntTensorSize;
    cumsumInfoTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, cumsumInfoTensorAddr,
        cumsumInfoTensorSize / sizeof(int32_t));
    // Tensor用处：SendCntCal 函数中 stride 只读 count 跳过 mask 区时，暂存各 src rank 的 count；
    // Tensor大小：每 src rank 一个 burst(32B)，共 ws*32B；
    uint32_t sendCntTensorAddr = cumsumInfoTensorAddr + cumsumInfoTensorSize;
    sendCntTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, sendCntTensorAddr,
        sendCntTensorSize / sizeof(int32_t));
    // Tensor用处：TripleInfoCalAndDispatch 函数中接收当前 batch 的 mask 切片；
    // Tensor大小：recvRouteBatchLen_ / 8 字节，每 bit 对应一个 route item；
    uint32_t maskBatchAddr = sendCntTensorAddr + sendCntTensorSize;
    uint32_t maskBatchSize = static_cast<uint32_t>(recvRouteBatchLen_ / 8) * static_cast<uint32_t>(sizeof(uint8_t));
    maskBatchTensor_ = LocalTensor<uint8_t>(TPosition::VECCALC, maskBatchAddr, maskBatchSize / sizeof(uint8_t));
    maskBatchU32Tensor_ = LocalTensor<uint32_t>(TPosition::VECCALC, maskBatchAddr, maskBatchSize / sizeof(uint32_t));
    // Tensor用处：TripleInfoCalAndDispatch 函数中 GatherMask 的 dst Tensor；
    // Tensor大小：recvRouteBatchLen_ * sizeof(int32_t) align 至 32 字节对齐；
    uint32_t validTopkIndexTensorAddr = maskBatchAddr + maskBatchSize;
    uint32_t validTopkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(recvRouteBatchLen_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    validTopkIndexTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, validTopkIndexTensorAddr,
        validTopkIndexTensorSize / sizeof(int32_t));
    // Tensor用处：TripleInfoCalAndDispatch 函数中 GatherMask 的 src Tensor（本 batch 的全局 index）；
    // Tensor大小：与 validTopkIndexTensor_ 一致，recvRouteBatchLen_ * sizeof(int32_t) align 至 32 字节对齐；
    uint32_t topkIndexTensorAddr = validTopkIndexTensorAddr + validTopkIndexTensorSize;
    uint32_t topkIndexTensorSize = validTopkIndexTensorSize;
    topkIndexTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, topkIndexTensorAddr,
        topkIndexTensorSize / sizeof(int32_t));
    // Tensor用处：TripleInfoCalAndDispatch 函数中的 6 个 dispatch buffer，配合 EVENT_ID0..EVENT_ID5 做软流水；
    // Tensor大小：每块容纳 token+scale，动态计算以节省 UB 空间；
    uint32_t copyTmpBaseAddr = topkIndexTensorAddr + topkIndexTensorSize;
    for (int32_t index = 0; index < DISPATCH_BUFFER_NUM; ++index) {
        copyTmpTensors_[index] = LocalTensor<ActivationType>(TPosition::VECCALC,
            copyTmpBaseAddr + static_cast<uint32_t>(index) * COPY_TMP_BUFFER_SIZE,
            COPY_TMP_BUFFER_SIZE / sizeof(ActivationType));
    }
    // Tensor用处：ExpertTokenNumCopyOut 函数中本卡各专家收到的 tokenCnt 数；
    // Tensor大小：expertPerRank_ * sizeof(int32_t) 对齐至 32 字节；
    uint32_t expertTokenNumsOutTensorAddr = copyTmpBaseAddr + copyTmpTotalSize;
    expertTokenNumsOutTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, expertTokenNumsOutTensorAddr,
        expertTokenNumsOutTensorSize / sizeof(int32_t));
    // Tensor用处：CopyGMToGMPerToken 函数中的 triple ring buffer，逐 token 即时写 GM；
    // Tensor大小：DISPATCH_BUFFER_NUM 条 * 32B；
    uint32_t tripleTensorAddr = expertTokenNumsOutTensorAddr + expertTokenNumsOutTensorSize;
    tripleTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, tripleTensorAddr,
        tripleReserveSize / sizeof(int32_t));
    ubBufferUsedAddr_ = expertTokenNumsOutTensorAddr + expertTokenNumsOutTensorSize;
    Duplicate<int32_t>(cumsumInfoTensor_, 0, (cumsumInfoTensorSize / sizeof(int32_t)));
    PipeBarrier<PIPE_ALL>();
}

// ======================================================================================================
// SendCntCal：stride 只读 count（跳过 mask 区）
// ======================================================================================================
template <TemplateMegaMoePrefillTypeClass>
__aicore__ inline void MegaMoePrefill<TemplateMegaMoePrefillTypeFunc>::SendCntCal(int32_t localExpertId,
    uint64_t& sendCnt)
{
    sendCnt = 0;

    // Phase 1: stride 读本 localExpert 的 ws 个 count
    GlobalTensor<int32_t> cntSrcGlobal;
    cntSrcGlobal.SetGlobalBuffer((__gm__ int32_t*)(params_.peermemInfo.maskRecvPtr +
        static_cast<uint64_t>(localExpertId) * worldSize_ * maskSlotSize_ + maskAlignSize_));
    DataCopyExtParams cntCopyParams{static_cast<uint16_t>(worldSize_), static_cast<uint32_t>(sizeof(int32_t)),
        static_cast<uint32_t>(maskSlotSize_ - sizeof(int32_t)), 0U, 0U};
    DataCopyPadExtParams<int32_t> cntPad{true, 0U, 0U, 0U};
    DataCopyPad(sendCntTensor_, cntSrcGlobal, cntCopyParams, cntPad);

    if constexpr (ENABLE_A8W4) {
        if (localExpertId != 0) {
            DataCopyPad(cumsumInfoTensor_, cumsumInfoGlobalTensor_,
                        {1U, static_cast<uint32_t>(worldSize_ * localExpertId * sizeof(int32_t)), 0U, 0U, 0U},
                        {true, 0U, 0U, 0U});
        }
    }

    SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>();

    // Phase 2: 逐 rank 读 count + cumsum（4B count 按 32B burst 落位 → 下标 rank*8）
    constexpr int32_t CNT_STRIDE_I32 = ALIGN_32 / sizeof(int32_t);
    for (int32_t calRankId = 0; calRankId < static_cast<int32_t>(worldSize_); ++calRankId) {
        int32_t perRankCnt = sendCntTensor_.GetValue(calRankId * CNT_STRIDE_I32);
        sendCnt += static_cast<uint64_t>(perRankCnt);
        cumsumRevCntInRank_ += static_cast<uint64_t>(perRankCnt);
        cumsumInfoTensor_.SetValue(localExpertId * worldSize_ + calRankId, static_cast<int32_t>(cumsumRevCntInRank_));
    }

    // Phase 3: 写 GM + 通知 AIC
    expertTokenCntTensor_.SetValue(0, sendCnt);
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    DataCopy<int32_t>(expertRevNumsGlobalTensor_[localExpertId * INT32_PER_256B * aicNum_ + INT32_PER_256B * blockIdx_],
        expertTokenCntTensor_, INT32_PER_256B);
    if constexpr (ENABLE_A8W4) {
        DataCopyPad(cumsumInfoGlobalTensor_, cumsumInfoTensor_,
                    {1U, static_cast<uint32_t>(worldSize_ * (localExpertId + 1) * sizeof(int32_t)), 0U, 0U, 0U});
    }
    PipeBarrier<PIPE_ALL>();

    __gm__ int32_t* sendCntFlag = (__gm__ int32_t*)params_.workspaceInfo.flagSendCntCalToUpdParamsPtr +
        static_cast<uint64_t>(localExpertId) * aicNum_ * INT_CACHELINE +
        static_cast<uint64_t>(blockIdx_) * INT_CACHELINE;
    AscendC::AtomicAdd(sendCntFlag, static_cast<int32_t>(1));
}

// ====================================================================================================
// TripleInfoCalAndDispatch：route batch 扫描 + matchedTotal(batch间累计命中数) + triple tile
// ====================================================================================================
template <TemplateMegaMoePrefillTypeClass>
__aicore__ inline void MegaMoePrefill<TemplateMegaMoePrefillTypeFunc>::TripleInfoCalAndDispatch(
    GMMAddrInfo &gmmAddrInfo, int32_t localExpertId)
{
    constexpr int32_t L1_TILE_M_I32 = static_cast<int32_t>(MegaMoeImpl::L1_TILE_M_256);
    int32_t priorExpertCumsum = (localExpertId == 0) ? 0 :
        cumsumInfoTensor_.GetValue(localExpertId * worldSize_ - 1);

    for (uint32_t idx = blockIdx_; idx < worldSize_ * blockNumPerRank_; idx += blockNum_) {
        uint32_t dstRankIdx = idx / blockNumPerRank_;
        uint32_t aivIdxInGroup = idx % blockNumPerRank_;
        uint32_t rowStartIdxInDst = ((dstRankIdx == 0 && localExpertId == 0) ? 0 :
            cumsumInfoTensor_.GetValue(localExpertId * worldSize_ + dstRankIdx - 1));
        int32_t srcRowLo = 0;
        int32_t srcRowHi = 0;
        int32_t rowDstOffsetInCore = 0;
        if (rowStartIdxInDst < maxOutputSize_) {
            int32_t sendToCurExpTokenCnt = cumsumInfoTensor_.GetValue(localExpertId * worldSize_ + dstRankIdx) -
                static_cast<int32_t>(rowStartIdxInDst);
            int32_t rowsCopyCnt = (rowStartIdxInDst + sendToCurExpTokenCnt > maxOutputSize_) ?
                static_cast<int32_t>(maxOutputSize_ - rowStartIdxInDst) : sendToCurExpTokenCnt;
            int32_t rowsNumInCore = Ops::Base::CeilDiv(rowsCopyCnt, static_cast<int32_t>(blockNumPerRank_));
            int32_t rowSrcIdxInCore = aivIdxInGroup * rowsNumInCore;
            rowDstOffsetInCore = rowStartIdxInDst + rowSrcIdxInCore;
            rowsNumInCore = (rowDstOffsetInCore + rowsNumInCore > rowStartIdxInDst + rowsCopyCnt) ?
                static_cast<int32_t>(rowStartIdxInDst + rowsCopyCnt - rowDstOffsetInCore) : rowsNumInCore;
            if (rowsNumInCore > 0) {
                srcRowLo = rowSrcIdxInCore;
                srcRowHi = rowSrcIdxInCore + rowsNumInCore;
            }
        }

        GlobalTensor<uint8_t> maskSrcGlobal;
        int32_t matchedTotal = 0;  // 跨 batch 累计已命中数量，作为下一 batch 的全局命中偏移
        int32_t rowsThisCore = 0;
        for (int32_t batchIdx = 0; batchIdx < recvRouteBatchNum_ && matchedTotal < srcRowHi; ++batchIdx) {
            int32_t batchStart = batchIdx * recvRouteBatchLen_;
            bool isLastBatch = (batchIdx == recvRouteBatchNum_ - 1);
            int32_t validLen = recvRouteBatchLen_;
            int32_t sliceBytes = recvRouteBatchLen_ / 8;
            if (isLastBatch) {
                validLen = static_cast<int32_t>(sendTotalNum_ - static_cast<uint64_t>(batchStart));
                if (batchStart / 8 + sliceBytes > static_cast<int32_t>(maskAlignSize_)) {
                    sliceBytes = static_cast<int32_t>(maskAlignSize_) - batchStart / 8;
                }
            }
            maskSrcGlobal.SetGlobalBuffer((__gm__ uint8_t*)(params_.peermemInfo.maskRecvPtr +
                (static_cast<uint64_t>(localExpertId) * worldSize_ + dstRankIdx) * maskSlotSize_ +
                static_cast<uint64_t>(batchStart / 8)));
            DataCopy(maskBatchTensor_, maskSrcGlobal, static_cast<uint32_t>(sliceBytes));
            SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();
            CreateVecIndex(topkIndexTensor_, batchStart, recvRouteBatchLen_);
            uint64_t matchedInBatch = 0;
            GatherMask(validTopkIndexTensor_, topkIndexTensor_, maskBatchU32Tensor_, true,
                static_cast<uint32_t>(validLen), {1, 1, 0, 0}, matchedInBatch);
            SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID4>();
            // 本 batch 的命中在全局命中序列中的范围 [batchLo, batchHi)
            int32_t batchLo = matchedTotal;
            int32_t batchHi = matchedTotal + static_cast<int32_t>(matchedInBatch);
            // 取本 core 源行区间 [srcRowLo, srcRowHi) 与 [batchLo, batchHi) 的交集
            int32_t hitLo = batchLo > srcRowLo ? batchLo : srcRowLo;
            int32_t hitHi = batchHi < srcRowHi ? batchHi : srcRowHi;
            if (hitHi > hitLo) {
                int32_t batchHitStart = hitLo - matchedTotal;
                int32_t batchHitCount = hitHi - hitLo;
                int32_t globalDstRow = static_cast<int32_t>(rowStartIdxInDst) + hitLo;
                CopyGMToGMPerToken(globalDstRow, dstRankIdx, batchHitStart, batchHitCount);
                rowsThisCore += batchHitCount;
            }
            matchedTotal = batchHi;  // 推进全局偏移
        }

        if (rowsThisCore > 0) {
            SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID5>();
            int32_t rowStartLocal = rowDstOffsetInCore - priorExpertCumsum;
            int32_t rowEndLocal = rowStartLocal + rowsThisCore;
            int32_t waveLo = rowStartLocal / L1_TILE_M_I32;
            int32_t waveHi = (rowEndLocal - 1) / L1_TILE_M_I32;
            __gm__ int32_t* flagBase = gmmAddrInfo.dispatchToGmm1Flag;
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

// ============================================================================
// CopyGMToGMPerToken（与主线一致）
// ============================================================================
template <TemplateMegaMoePrefillTypeClass>
__aicore__ inline void MegaMoePrefill<TemplateMegaMoePrefillTypeFunc>::CopyGMToGMPerToken(
    int32_t rowDstOffsetInCore, int32_t remoteRankIdx, int32_t copyStartIdx, int32_t copyNum)
{
    if (copyNum <= 0) { return; }
    constexpr int32_t BufferNum = DISPATCH_BUFFER_NUM;
    constexpr TEventID kBufEvents[BufferNum] = {EVENT_ID0, EVENT_ID1, EVENT_ID2, EVENT_ID3, EVENT_ID4, EVENT_ID5};
    int64_t widthA = k_ / A_ELEMS_PER_BYTE;
    int64_t widthAScale = Ops::Base::CeilDiv(static_cast<int64_t>(k_), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE;
    uint32_t copyInNum = Ops::Base::CeilAlign(static_cast<int64_t>(mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_),
        static_cast<int64_t>(ALIGN_32));
    GlobalTensor<ActivationType> remoteRankGlobalTensor;
    GlobalTensor<ActivationType> tokenRevGlobalTensor;
    GlobalTensor<QuantScaleOutType> scaleRevGlobalTensor;
    tokenRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType*>(
        params_.workspaceInfo.dispatchRevDataPtr + rowDstOffsetInCore * widthA));
    scaleRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleOutType *>(
        params_.workspaceInfo.dispatchRevScalePtr + rowDstOffsetInCore * widthAScale));
    remoteRankGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType*>(
        GetRankWinAddrWithOffset(remoteRankIdx, quantWinOffset_)));
    PipeBarrier<PIPE_ALL>();
    // Phase 1: prime count 条 triple 组装 + 发 MTE2（合并为一轮）
    int32_t primeCount = (copyNum < BufferNum) ? copyNum : BufferNum;
    for (int32_t primeIdx = 0; primeIdx < primeCount; ++primeIdx) {
        int32_t topkIndex = validTopkIndexTensor_.GetValue(copyStartIdx + primeIdx);
        int32_t tokenIndex = topkIndex / topK_;
        TEventID eventId = kBufEvents[primeIdx];
        uint64_t remoteCopyOffset = static_cast<uint64_t>(tokenIndex) * static_cast<uint64_t>(copyInNum);
        DataCopy(copyTmpTensors_[primeIdx], remoteRankGlobalTensor[remoteCopyOffset], copyInNum);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        tripleTensor_[primeIdx * INT32_PER_256B].SetValue(RANK_ID, remoteRankIdx);
        tripleTensor_[primeIdx * INT32_PER_256B].SetValue(TOKEN_ID, tokenIndex);
        tripleTensor_[primeIdx * INT32_PER_256B].SetValue(TOPK_INDEX, topkIndex % topK_);
        SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
    }
    // Phase 2: steady 6-buffer 流水（MTE3 写 token+scale+triple，MTE2 预取下一条）
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
        // triple 即时写 GM（S_MTE3 保证 SetValue 对 tripleTensor_ 的写入已完成）
        WaitFlag<AscendC::HardEvent::S_MTE3>(eventId);
        DataCopy(tripleGlobalTensor_[(rowDstOffsetInCore + copyIdx) * INT32_PER_256B],
            tripleTensor_[outIdx * INT32_PER_256B], INT32_PER_256B);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        SetFlag<AscendC::HardEvent::MTE3_S>(eventId);   // MTE3 读完 triple，S 可覆盖
        int32_t nextIdx = copyIdx + BufferNum;
        if (nextIdx < copyNum) {
            // 组装下一条 triple：先发 MTE2(慢) → 再写 triple(快)，MTE2 在后台与 S 重叠
            int32_t nextTopkIndex = validTopkIndexTensor_.GetValue(copyStartIdx + nextIdx);
            int32_t nextTokenIndex = nextTopkIndex / topK_;
            uint64_t remoteCopyOffset = static_cast<uint64_t>(nextTokenIndex) * static_cast<uint64_t>(copyInNum);
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            DataCopy(copyTmpTensors_[outIdx], remoteRankGlobalTensor[remoteCopyOffset], copyInNum);
            SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
            WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
            tripleTensor_[outIdx * INT32_PER_256B].SetValue(RANK_ID, remoteRankIdx);
            tripleTensor_[outIdx * INT32_PER_256B].SetValue(TOKEN_ID, nextTokenIndex);
            tripleTensor_[outIdx * INT32_PER_256B].SetValue(TOPK_INDEX, nextTopkIndex % topK_);
            SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
        }
    }
    // Phase 2 drain: 只等实际 prime 过的槽位，避免死等 + SetFlag/SetFlag 配对
    for (int32_t bufferIdx = 0; bufferIdx < primeCount; ++bufferIdx) {
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kBufEvents[bufferIdx]);
        WaitFlag<AscendC::HardEvent::MTE3_S>(kBufEvents[bufferIdx]);
    }
}

// ==================================================
// ExpertTokenNumCopyOut（与主线一致）
// ==================================================
template <TemplateMegaMoePrefillTypeClass>
__aicore__ inline void MegaMoePrefill<TemplateMegaMoePrefillTypeFunc>::ExpertTokenNumCopyOut()
{
    if constexpr (ENABLE_A8W4) {
        DataCopyPad(cumsumInfoTensor_, cumsumInfoGlobalTensor_,
                {1U, static_cast<uint32_t>(worldSize_ * expertPerRank_ * sizeof(int32_t)), 0U, 0U, 0U},
                {true, 0U, 0U, 0U});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    }
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
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
}

// =====================================================================================================
// UpdateGroupParams（与主线一致）
// =====================================================================================================
template <TemplateMegaMoePrefillTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline bool MegaMoePrefill<TemplateMegaMoePrefillTypeFunc>::UpdateGroupParams(ExpertLoopState &state,
    uint32_t expertIdx, uint64_t sendCnt)
{
    if (expertIdx != 0) {
        uint64_t m = Get<M_VALUE>(state.problemShape);
        uint64_t n = Get<N_VALUE>(state.problemShape);
        uint64_t k = Get<K_VALUE>(state.problemShape);
        state.expertBeforeCnt += m;
        Get<IDX_A_OFFSET>(state.baseOffset) += m * k / A_ELEMS_PER_BYTE;
        Get<IDX_B_OFFSET>(state.baseOffset) += n * k / B_ELEMS_PER_BYTE;
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
    if constexpr (Mode == AddrUpdateMode::kGmm1) {
        if (subBlockIdx_ == 0) {
            __gm__ int32_t* sendCntFlag = (__gm__ int32_t*)params_.workspaceInfo.flagSendCntCalToUpdParamsPtr +
                static_cast<uint64_t>(expertIdx) * aicNum_ * INT_CACHELINE +
                static_cast<uint64_t>(blockIdx_) * INT_CACHELINE;
            while (AscendC::ReadGmByPassDCache(sendCntFlag) == 0) {
                int64_t st = AscendC::GetSystemCycle();
                while (AscendC::GetSystemCycle() - st < 100) { }
            }
            uint64_t offsetInCnt = expertIdx * 8 * aicNum_ + 8 * blockIdx_;
            DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
                expertRevNumsGlobalTensor_[offsetInCnt]);
            Get<M_VALUE>(state.problemShape) = expertRevNumsGlobalTensor_.GetValue(offsetInCnt);
        } else {
            Get<M_VALUE>(state.problemShape) = sendCnt;
        }
    } else if constexpr (Mode == AddrUpdateMode::kGmm2) {
        uint64_t offsetInCnt = expertIdx * 8 * aicNum_ + 8 * blockIdx_;
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
                expertRevNumsGlobalTensor_[offsetInCnt]);
        Get<M_VALUE>(state.problemShape) = expertRevNumsGlobalTensor_.GetValue(offsetInCnt);
    }
    if (Get<M_VALUE>(state.problemShape) == 0) { return false; }
    return true;
}

// ==================================================================================
// UpdateGlobalBuffer（与主线一致）
// ==================================================================================
template <TemplateMegaMoePrefillTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline void MegaMoePrefill<TemplateMegaMoePrefillTypeFunc>::UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo,
    const ExpertLoopState &state)
{
    if constexpr (Mode == AddrUpdateMode::kGmm1) {
        if constexpr (ENABLE_A8W4) {
            gmmAddrInfo.gmm1OutGlobal =
                params_.workspaceInfo.gmm1MmadResPtr + Get<IDX_GMM1_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
        }
        gmmAddrInfo.aGlobal = params_.workspaceInfo.dispatchRevDataPtr +
            Get<IDX_A_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.dispatchRevScalePtr +
            Get<IDX_A_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);
        gmmAddrInfo.bGlobal = params_.bGmAddr + Get<IDX_B_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.bScaleGlobal = params_.bScaleGmAddr + Get<IDX_B_SCALE_OFFSET>(state.baseOffset) *
                                    sizeof(QuantScaleOutType);
        if constexpr(g_coreType == AIV) {
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                Get<IDX_C_OFFSET>(state.baseOffset), Get<IDX_C_SCALE_OFFSET>(state.baseOffset),
                Get<IDX_FLAG_OFFSET>(state.baseOffset), 0L, 0L, 0L};
            epilogueOp_.UpdateGlobalAddr(vecBaseOffset);
        }
    } else if constexpr(Mode == AddrUpdateMode::kGmm2) {
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
    }
    gmmAddrInfo.swigluToGmm2Flag = (__gm__ int32_t*)params_.workspaceInfo.flagSwiGluToGmm2Ptr +
                                Get<IDX_FLAG_OFFSET>(state.baseOffset) * INT_CACHELINE;
    gmmAddrInfo.dispatchToGmm1Flag = (__gm__ int32_t*)params_.workspaceInfo.flagDispatchToGmm1Ptr +
                                Get<IDX_FLAG_OFFSET>(state.baseOffset) * dispatchFlagSlotsPerExpert_;
}

// ===============================================================
// GroupMatmulWithSwigluQuant（与主线一致）
// ===============================================================
template <TemplateMegaMoePrefillTypeClass>
__aicore__ inline void MegaMoePrefill<TemplateMegaMoePrefillTypeFunc>::GroupMatmulWithSwigluQuant(
    const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state)
{
    if constexpr (ENABLE_A8W4) {
        MegaMoeImpl::GroupMatmulSwigluQuantA8W4<
            QuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
            epilogueOp_, params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_);
    } else {
        if (params_.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ||
            params_.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
            MegaMoeImpl::GroupMatmulSwigluQuant<
                QuantOutType, SwigluQuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, true>(
                epilogueOp_, params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_);
        } else {
            MegaMoeImpl::GroupMatmulSwigluQuant<
                QuantOutType, SwigluQuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
                epilogueOp_, params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_);
        }
    }
}

// =============================================
// InitCombineBuffers（与主线一致）
// =============================================
template <TemplateMegaMoePrefillTypeClass>
__aicore__ inline void MegaMoePrefill<TemplateMegaMoePrefillTypeFunc>::InitCombineBuffers()
{
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV) {
        uint32_t nAlign32 = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_32));
        uint32_t nScale = Ops::Base::CeilDiv(k_, uint32_t(MXFP_SCALE_GROUP_NUM));
        uint32_t quantTokenSizeBytes = Ops::Base::CeilAlign(k_ + nScale, static_cast<uint32_t>(ALIGN_32));
        uint32_t singleTokenBytes = nAlign32 * sizeof(bfloat16_t) + quantTokenSizeBytes;
        combineUbTensorSize_ = (singleTokenBytes * 2) / sizeof(bfloat16_t);
    }
}

// ===============================================================
// GroupMatmulWithCombine（与主线一致）
// ===============================================================
template <TemplateMegaMoePrefillTypeClass>
__aicore__ inline void MegaMoePrefill<TemplateMegaMoePrefillTypeFunc>::GroupMatmulWithCombine(
    const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state, uint32_t expertIdx)
{
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        MegaMoeImpl::GroupMatmul2CombineA8W4<CombineQuantMode,
            SwigluQuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
            params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_,
            state.expertBeforeCnt, gmm2PingPongIdx_, expertIdx);
    } else {
        if (params_.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ) {
            MegaMoeImpl::GroupMatmul2<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t,
                QuantScaleOutType, QuantScaleOutType, true>(
                params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_,
                state.expertBeforeCnt, gmm2PingPongIdx_, expertIdx);
        } else {
            MegaMoeImpl::GroupMatmul2<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t,
                QuantScaleOutType, QuantScaleOutType>(
                params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_,
                state.expertBeforeCnt, gmm2PingPongIdx_, expertIdx);
        }
    }
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV) {
        ProcessCombine(gmmAddrInfo, state, expertIdx);
    }
}

// =============================================
// ProcessCombine（与主线一致）
// =============================================
template <TemplateMegaMoePrefillTypeClass>
__aicore__ inline void MegaMoePrefill<TemplateMegaMoePrefillTypeFunc>::ProcessCombine(
    const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &gmm2State, uint32_t expertIdx)
{
    uint32_t nTilesPerGroup = Ops::Base::CeilDiv(k_, L1_TILE_N);
    GlobalTensor<int32_t> gmm2CombineSyncCounter;
    gmm2CombineSyncCounter.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.gmm2CombineSyncCounterPtr);
    uint32_t nScale = Ops::Base::CeilDiv(k_, uint32_t(MXFP_SCALE_GROUP_NUM));
    uint32_t quantTokenSizeBytes = Ops::Base::CeilAlign(k_ + nScale, static_cast<uint32_t>(ALIGN_32));
    uint32_t m_expert = Get<M_VALUE>(gmm2State.problemShape);
    uint32_t tokenGroupsThisExpert = Ops::Base::CeilDiv(m_expert, L1_TILE_M_256);
    uint32_t coreIdForGrouping = aivCoreIdx_;
    uint32_t totalCoresForGrouping = blockAivNum_;
    if constexpr (ENABLE_A8W4) {
        if (subBlockIdx_ != 1) { return; }
        coreIdForGrouping = aivCoreIdx_ / 2;
        totalCoresForGrouping = blockAivNum_ / 2;
    }
    uint32_t myGroup, myIdxInGrp, myGrpSize;
    MegaMoeImpl::ComputeCoreGrouping(coreIdForGrouping, tokenGroupsThisExpert, totalCoresForGrouping,
        myGroup, myIdxInGrp, myGrpSize);
    if (myGroup >= tokenGroupsThisExpert) { return; }
    __gm__ int32_t* myCounterAddr = (__gm__ int32_t*)gmm2CombineSyncCounter.GetPhyAddr()
        + expertIdx * blockAivNum_ * INT_CACHELINE + aivCoreIdx_ * INT_CACHELINE;
    while (AscendC::ReadGmByPassDCache(myCounterAddr) != nTilesPerGroup) {
        int64_t st = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - st < 100) { };
    }
    uint32_t tokenStart = myGroup * L1_TILE_M_256;
    uint32_t tokenCount = (L1_TILE_M_256 < m_expert - tokenStart) ? L1_TILE_M_256 : m_expert - tokenStart;
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
// Unpermute：固定 buffer → 反推 batchLen → 分批循环
// ===============================================================
template <TemplateMegaMoePrefillTypeClass>
__aicore__ inline void MegaMoePrefill<TemplateMegaMoePrefillTypeFunc>::UnpermuteLoadWeights(
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

template <TemplateMegaMoePrefillTypeClass>
__aicore__ inline void MegaMoePrefill<TemplateMegaMoePrefillTypeFunc>::UnpermuteProcessToken(
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
// Unpermute：主入口 — 分配固定 buffer → 反推 batchLen → 分批循环
// ===============================================================
template <TemplateMegaMoePrefillTypeClass>
__aicore__ inline void MegaMoePrefill<TemplateMegaMoePrefillTypeFunc>::Unpermute()
{
    int32_t coreLen, coreOffset;
    TilingByCore(m_, coreLen, coreOffset, 1);

    // ── 固定 buffer：dataRes + dataResFp32 (+ scales) ──
    // Tensor用处：Unpermute 函数用于存储 mte2 搬入 token；
    // Tensor大小：UNPERMUTE_LIST_NUM * 单个 token 长度，2 块是 mte2 搬运的 doubleBuffer，1 块存储累加计算 Cast 完的输出结果用于搬出；
    uint32_t dataResBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(UNPERMUTE_LIST_NUM * k_ * sizeof(bfloat16_t)), static_cast<uint32_t>(ALIGN_32));
    uint32_t dataResFp32BufAlign = dataResBufAlign * HALF_TO_FP32;
    uint32_t dataResAddr = 0;
    dataResTensor_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, dataResAddr,
        dataResBufAlign / sizeof(bfloat16_t));
    // Tensor用处：Unpermute 函数用于存储 token Cast 目的 Tensor；
    // Tensor大小：dataResTensor_ 开设大小乘以 HALF_TO_FP32；
    uint32_t dataResFp32Addr = dataResAddr + dataResBufAlign;
    dataResFp32Tensor_ = LocalTensor<float>(TPosition::VECCALC, dataResFp32Addr,
        dataResFp32BufAlign / sizeof(float));
    uint32_t tempAddr = dataResFp32Addr + dataResFp32BufAlign;
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        uint32_t scaleNum = Ops::Base::CeilAlign(static_cast<uint32_t>(k_), static_cast<uint32_t>(ALIGN_32));
        // Tensor用处：DeQuantMxFp8 中用于存储 bf16 格式的 scale（e8m0 转换后的中间结果）；
        // Tensor大小：scaleNum * sizeof(bfloat16_t) * DOUBLE_BUFFER * HALF_TO_FP32，双缓冲 + scale 扩展；
        uint32_t bf16ScaleBufAlign = Ops::Base::CeilAlign(static_cast<uint32_t>
            (scaleNum * sizeof(bfloat16_t) * DOUBLE_BUFFER * HALF_TO_FP32), static_cast<uint32_t>(ALIGN_32));
        bf16ScaleTensor_ = LocalTensor<bfloat16_t>(
            TPosition::VECCALC, tempAddr, bf16ScaleBufAlign / sizeof(bfloat16_t));
        tempAddr += bf16ScaleBufAlign;
        // Tensor用处：DeQuantMxFp8 中用于存储 fp32 格式的 scale（广播后的最终 scale）；
        // Tensor大小：scaleNum * sizeof(float) * DOUBLE_BUFFER * HALF_TO_FP32，双缓冲 + scale 扩展；
        uint32_t fp32ScaleBufAlign = Ops::Base::CeilAlign(static_cast<uint32_t>
            (scaleNum * sizeof(float) * DOUBLE_BUFFER * HALF_TO_FP32), static_cast<uint32_t>(ALIGN_32));
        fp32ScaleTensor_ = LocalTensor<float>(TPosition::VECCALC, tempAddr, fp32ScaleBufAlign / sizeof(float));
        tempAddr += fp32ScaleBufAlign;
    }

    // ── 固定 batchLen（后续以模板参数形式传入）──
    int32_t batchLen = UNPERMUTE_BATCH_LEN;
    if (batchLen > coreLen) { batchLen = coreLen; }
    int32_t batchNum = Ops::Base::CeilDiv(coreLen, batchLen);

    // ── weight buffer（at tempAddr，每批复用）──
    // Tensor用处：用于存储 topKWeight；
    // Tensor大小：batchLen * topK_ * sizeof(float) align 到 32 字节对齐；
    uint32_t topKWeightsBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(batchLen * topK_ * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
    topKWeightsTensor_ = LocalTensor<float>(TPosition::VECCALC, tempAddr,
        topKWeightsBufAlign / sizeof(float));
    LocalTensor<bfloat16_t> tempLocal;
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        uint32_t tempBufAlign = Ops::Base::CeilAlign(
            static_cast<uint32_t>(batchLen * topK_ * sizeof(bfloat16_t)), uint32_t(ALIGN_32));
        tempLocal = LocalTensor<bfloat16_t>(TPosition::VECCALC, tempAddr + topKWeightsBufAlign,
            tempBufAlign / sizeof(bfloat16_t));
    }

    // ── Global tensors ──
    GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer((__gm__ bfloat16_t*)params_.peermemInfo.combineSendPtr);
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer((__gm__ bfloat16_t*)params_.y2GmAddr);

    // ═══════════════════ 外层：token batch ═══════════════════
    for (int32_t batchIdx = 0; batchIdx < batchNum; ++batchIdx) {
        int32_t batchStart = batchIdx * batchLen;
        int32_t curTokens = (batchIdx == batchNum - 1) ? (coreLen - batchStart) : batchLen;

        UnpermuteLoadWeights(coreOffset, batchStart, curTokens, tempLocal);

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
// Process：大 bs 流水编排（dispatch prefetch，与主线 Process 结构一致）
// ==============================================================================================
template <TemplateMegaMoePrefillTypeClass>
__aicore__ inline void MegaMoePrefill<TemplateMegaMoePrefillTypeFunc>::Process()
{
    // 1. 本卡数据处理（发 mask + 量化 + reset）
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    SendAndQuantBuffInit();
    SendMaskCal();
    ResetFlagList();
    QuantProcessInRank();
    if constexpr(g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
    }
    SyncAll<false>();
    CrossRankSyncInWorldSize();

    // 2. dispatch + GMM1
    DispatchBuffInit();
    GMMAddrInfo dispatchAddrInfo;
    GMMAddrInfo gmm1AddrInfo;
    TupleShape initShape;
    Get<N_VALUE>(initShape) = hiddenDim_;
    Get<K_VALUE>(initShape) = k_;
    BlockOffset initOffset{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    ExpertLoopState dispatchState{initShape, initOffset, 0};
    ExpertLoopState gmm1State{initShape, initOffset, 0};

    uint64_t curSendCnt = 0;
    uint64_t nextSendCnt = 0;

    if constexpr(g_coreType == AIV) {
        if (subBlockIdx_ == 1) {
            SendCntCal(0, nextSendCnt);
            if (UpdateGroupParams<AddrUpdateMode::kGmm1>(dispatchState, 0, nextSendCnt)) {
                UpdateGlobalBuffer<AddrUpdateMode::kGmm1>(dispatchAddrInfo, dispatchState);
                TripleInfoCalAndDispatch(dispatchAddrInfo, 0);
            }
        }
    }

    for (int localExpertId = 0; localExpertId < expertPerRank_; localExpertId++) {
        curSendCnt = nextSendCnt;
        if constexpr(g_coreType == AIV) {
            if (subBlockIdx_ == 1 && localExpertId + 1 < expertPerRank_) {
                SendCntCal(localExpertId + 1, nextSendCnt);
                if (UpdateGroupParams<AddrUpdateMode::kGmm1>(dispatchState, localExpertId + 1, nextSendCnt)) {
                    UpdateGlobalBuffer<AddrUpdateMode::kGmm1>(dispatchAddrInfo, dispatchState);
                    TripleInfoCalAndDispatch(dispatchAddrInfo, localExpertId + 1);
                }
            }
        }
        if (!UpdateGroupParams<AddrUpdateMode::kGmm1>(gmm1State, localExpertId, curSendCnt)) {
            continue;
        }
        UpdateGlobalBuffer<AddrUpdateMode::kGmm1>(gmm1AddrInfo, gmm1State);
        GroupMatmulWithSwigluQuant(gmm1AddrInfo, gmm1State);
    }
    EndSync(vecSetSyncCom_);
    if constexpr(g_coreType == AIV) {
        if (subBlockIdx_ == 1) {
            ExpertTokenNumCopyOut();
        }
    }

    // 3. GMM2 + Combine
    vecSetSyncCom_ = 0;
    GMMAddrInfo gmm2AddrInfo;
    ExpertLoopState gmm2State{initShape, initOffset, 0};
    InitCombineBuffers();
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
        if (!UpdateGroupParams<AddrUpdateMode::kGmm2>(gmm2State, expertIdx)) {
            continue;
        }
        UpdateGlobalBuffer<AddrUpdateMode::kGmm2>(gmm2AddrInfo, gmm2State);
        GroupMatmulWithCombine(gmm2AddrInfo, gmm2State, expertIdx);
    }
    if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        EndGMM2Sync(vecSetSyncCom_, gmm2PingPongIdx_);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();

    // 4. Unpermute
    if constexpr(g_coreType == AIV) {
        CrossRankSyncInWorldSize();
        Unpermute();
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

}   // namespace MegaMoeImpl
#endif
