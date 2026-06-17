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

// GMM1 dispatch(sb1)->matmul(AIC) 反压: aic↔aiv1 信用额度(credit)flag。
//   循环前 AIC 预发 GMM1_DISPATCH_CREDITS 个信用; sb1 每专家开头 WaitFlag 消费 1 个再 SendCntCal;
//   AIC 每处理完一专家 SetFlag 归还 1 个。⇒ sb1 领先 AIC 恒 ≤ 信用数, 故 flag2 与本 ack 两个计数器都恒
//   ≤ 信用数 < 15, 与 epr 解耦(epr=1024 也不会触顶溢出)。取 8, 远离 15 上限, 又够 sb1 提前 8 个专家做
//   dispatch 保流水。aiv1↔aic 约定: aiv1 用通道 N(0-10), aic 用 N+16; 本 ack 为 aic->aiv1, 故 aic 侧
//   SetFlag(GMM1_CREDIT_FLAG+16), sb1 侧 WaitFlag(GMM1_CREDIT_FLAG)。通道 8 与 matmul 的 4/6(+value) 无冲突。
constexpr int32_t GMM1_CREDIT_FLAG = 8;
constexpr int32_t GMM1_DISPATCH_CREDITS = 8;

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
    // 分块流式 dispatch: 所有随 ST=bs*topK 增长的 UB 缓冲都按 chunk 个 token 分批处理, 使 UB 与 bs 解耦
    // (win/HBM 仍按全 ST 分配)。发/收两段时间隔离、UB 各自从 0 铺、win mask 是整条 ST 的 bit 数组(写/读切法
    // 无耦合) ⇒ 各按自己段的 UB 余量独立定 chunk(不取 min, 见设计文档 §6.3), 在对应 BuffInit 里算并 cap 到
    // CeilAlign(ST,256) ⇒ 中小 bs 各自单块零开销; 恒为 256 倍数 ⇒ chunk/8 mask 切片天然 32B 对齐。
    int32_t sendChunkTokenNum_ = 0;   // 发侧每 chunk 的 token 数(SendAndQuantBuffInit + SendMaskCal)
    int32_t sendChunkNum_ = 0;        // 发侧 chunk 总数
    int32_t recvChunkTokenNum_ = 0;   // 收侧每 chunk 的 token 数(DispatchBuffInit + TripleInfoCalAndDispatch)
    int32_t recvChunkNum_ = 0;        // 收侧 chunk 总数
    // resetTensor_ 封顶份额(int32 个数) = min(本核 flag 份额, DISPATCH_RESET_BATCH); 超额时 ResetFlagList 分批推 GM。
    int32_t resetBatchNum_ = 0;
    int32_t wCoreOffset_ = 0;       // 本核 Unpermute token 段起点(与 Unpermute 的 TilingByCore 一致)

    static constexpr uint32_t ELEMS_PER_BYTE = (QuantMode == E2M1_QUANT) ? 2 : 1;
    static constexpr int32_t DISPATCH_BUFFER_NUM = 6;
    LocalTensor<int32_t> topkIndexTensor_;                // 每 chunk 重建的全局 token 位置 [chunkStart, +CHUNK)
    LocalTensor<uint8_t> maskChunkTensor_;                      // 接收端: 当前 dstRank 的一个 CHUNK/8 mask 切片
    LocalTensor<uint32_t> maskChunkU32Tensor_;                  // maskChunkTensor_ 的 u32 视图(同址别名), 供 GatherMask
    LocalTensor<int32_t> sendCntTensor_;                  // 接收端: 本 expert 的 ws 个 srcRank count
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
    LocalTensor<int32_t> topkChunkTensor_;                      // SendMaskCal 源卡: 当前 chunk 的 topk(尾部填无效位 -1)
    LocalTensor<uint8_t> sendMaskTensor_[DOUBLE_BUFFER];  // SendMaskCal 源卡算 mask 切片的 ping-pong 缓冲
    LocalTensor<int32_t> sendGatherOutTensor_;            // SendMaskCal GatherMask 计 count 的废弃输出 scratch
    LocalTensor<int32_t> sendCntAccumulateTensor_;        // SendMaskCal 源卡每核 per-owned-expert 跨 chunk count 累加
    LocalTensor<int32_t> expertTokenNumsOutTensor_;
    LocalTensor<bfloat16_t> dataResTensor_;
    LocalTensor<float> dataResFp32Tensor_;
    LocalTensor<float> topKWeightsTensor_;    // 仅存本核 token 段的 weights(大 bs 适配)

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
    // 发/收侧 chunk(sendChunkTokenNum_/recvChunkTokenNum_)在各自的 SendAndQuantBuffInit/DispatchBuffInit
    // 里就地计算: 那里本就要逐项算各 tensor 大小, 顺带用 UB 余量反推 chunk, 避免在此重复一遍固定占用计算。
    // maskAlignSize_/maskSlotSize_ 仅依赖 sendTotalNum_, 公式与改前一致(分块不改 win 槽位大小):
    // 源按 chunk 分片填同一 mask 区, 末片 clamp 到 maskAlignSize_。须与 PeermemInfo(mega_moe_base.h)一致。
    uint32_t compareCount = Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_256)) / sizeof(int32_t);
    maskAlignSize_ = Ops::Base::CeilAlign(static_cast<int64_t>(compareCount) / 8, static_cast<int64_t>(ALIGN_32));
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

    // 时段 B 布局从 UB 地址 0 起, 整体覆盖时段 A(SendAndQuantBuffInit)的区域 —— 跨时段重叠是有意复用。
    // 分块流式后所有 O(ST) 缓冲降为 O(收侧 chunk)(maskChunkTensor_/validTopkIndex_/topkIndex_), 与 bs 无关。

    // —— 与 chunk 无关的固定占用, 先逐项算大小 ——
    // Tensor用处：expertTokenCntTensor_ —— SendCntCal 写 GM 用的单 expert 收到 token 数暂存。
    // Tensor大小：32B。
    uint32_t expertTokenCntTensorSize = ALIGN_32;
    // Tensor用处：cumsumInfoTensor_ —— 全局 [expert][srcRank] 收 token cumsum 表, 全程驻留(GMM offset/输出用)。
    // Tensor大小：ws*epr*4 对齐 32B。
    uint32_t cumsumInfoTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(worldSize_ * expertPerRank_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    // Tensor用处：sendCntTensor_ —— SendCntCal stride 读本 expert 的 ws 个 srcRank count(只读 count, 不读 mask)。
    // Tensor大小：每 rank 一个 burst 占 32B, 共 ws*32B。
    uint32_t sendCntTensorSize = worldSize_ * static_cast<uint32_t>(ALIGN_32);
    // Tensor用处：copyTmpTensors_ —— TripleInfoCalAndDispatch 的 6 个 dispatch buffer(EVENT_ID0..5 软流水)。
    // Tensor大小：每块 TOKEN_SCALE_SIZE=8KB 容纳 token+scale, 共 6 块 48KB。
    constexpr uint32_t COPY_TMP_BUFFER_SIZE = TOKEN_SCALE_SIZE;
    uint32_t copyTmpTotalSize = static_cast<uint32_t>(DISPATCH_BUFFER_NUM) * COPY_TMP_BUFFER_SIZE;
    // Tensor用处：expertTokenNumsOutTensor_ —— ExpertTokenNumCopyOut 用, 本卡各专家收到的 tokenCnt 数。
    // Tensor大小：epr*4 对齐 32B。
    uint32_t expertTokenNumsOutTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(expertPerRank_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    // Tensor用处：tripleTensor_ 尾部预留区 —— TripleInfoCalAndDispatch 每 sub-slice 动态重建(每行 32B), 与 chunk 解耦。
    // Tensor大小：DISPATCH_TRIPLE_TILE 行 * 32B。
    uint32_t tripleReserveSize = static_cast<uint32_t>(DISPATCH_TRIPLE_TILE) *
        static_cast<uint32_t>(INT32_PER_256B) * sizeof(int32_t);
    uint32_t fixedBytes = expertTokenCntTensorSize + cumsumInfoTensorSize + sendCntTensorSize +
        copyTmpTotalSize + expertTokenNumsOutTensorSize + tripleReserveSize;

    // —— 用 UB 余量反推收侧 chunk(与发侧无耦合, 各算各的) ——
    // chunk 随 token 线性增长的 UB: validTopkIndex(4B/tok) + topkIndex(4B/tok) + maskChunk(1/8 B/tok)。
    // 解 chunk*(4+4 + 1/8)B ≤ 余量 ⇒ chunk ≤ 余量*8 / ((4+4)*8 + 1)(分母 65 即 8.125 B/token)。
    int64_t ubBudget = static_cast<int64_t>(params_.tilingData->ubSize) - DISPATCH_UB_SAFETY;
    int64_t avail = ubBudget - static_cast<int64_t>(fixedBytes);
    int64_t ckLimit = avail * 8 / ((4 + 4) * 8 + 1);
    int64_t ckCap = Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_), static_cast<int64_t>(ALIGN_256));
    int64_t ck = ckLimit < ckCap ? ckLimit : ckCap;
    ck = ck / ALIGN_256 * ALIGN_256;                  // 向下取整到 256 倍数 ⇒ chunk/8 mask 切片天然 32B 对齐
    ck = ck < ALIGN_256 ? ALIGN_256 : ck;             // 兜底下限(极端大 H 时固定占用已近满)
    recvChunkTokenNum_ = static_cast<int32_t>(ck);
    recvChunkNum_ = static_cast<int32_t>(
        Ops::Base::CeilDiv(sendTotalNum_, static_cast<uint64_t>(recvChunkTokenNum_)));

    // —— 按既定顺序落地址(复用上面的 size) ——
    uint32_t expertTokenCntTensorAddr = 0;
    expertTokenCntTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, expertTokenCntTensorAddr,
        expertTokenCntTensorSize / sizeof(int32_t));
    uint32_t cumsumInfoTensorAddr = expertTokenCntTensorAddr + expertTokenCntTensorSize;
    cumsumInfoTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, cumsumInfoTensorAddr,
        cumsumInfoTensorSize / sizeof(int32_t));
    // SendCntCal 用 stride 一次性读本 expert 的 ws 个 srcRank count(只读 count, 不读 mask): 每个 burst(4B count)
    // 在 UB 内 32B 对齐落位, 故每 rank 占 32B, 读取下标 = rank*8。
    uint32_t sendCntTensorAddr = cumsumInfoTensorAddr + cumsumInfoTensorSize;
    sendCntTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, sendCntTensorAddr,
        sendCntTensorSize / sizeof(int32_t));
    // Tensor用处：maskChunkTensor_(u8)/maskChunkU32Tensor_(u32) 同址别名 —— 接收端单个 (dstRank,chunk) 的 mask 切片(u8 读 win, u32 给 GatherMask)。
    // Tensor大小：chunk/8(chunk 为 256 倍数 ⇒ 32B 对齐)。
    uint32_t maskChunkAddr = sendCntTensorAddr + sendCntTensorSize;
    uint32_t maskChunkSize = static_cast<uint32_t>(recvChunkTokenNum_ / 8);
    maskChunkTensor_ = LocalTensor<uint8_t>(TPosition::VECCALC, maskChunkAddr, maskChunkSize / sizeof(uint8_t));
    maskChunkU32Tensor_ = LocalTensor<uint32_t>(TPosition::VECCALC, maskChunkAddr, maskChunkSize / sizeof(uint32_t));
    // Tensor用处：validTopkIndexTensor_ —— GatherMask 输出, 命中当前 (dstRank,chunk) 的 chunk-local token 位置(每 chunk 覆写)。
    // Tensor大小：chunk*4 对齐 32B。
    uint32_t validTopkIndexTensorAddr = maskChunkAddr + maskChunkSize;
    uint32_t validTopkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(recvChunkTokenNum_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    validTopkIndexTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, validTopkIndexTensorAddr,
        validTopkIndexTensorSize / sizeof(int32_t));
    // Tensor用处：topkIndexTensor_ —— 当前 chunk 的全局 token 位置 [chunkStart,+chunk)(每 chunk 重 CreateVecIndex, 作 GatherMask src)。
    // Tensor大小：同 validTopkIndexTensor_(chunk*4 对齐 32B)。
    uint32_t topkIndexTensorAddr = validTopkIndexTensorAddr + validTopkIndexTensorSize;
    uint32_t topkIndexTensorSize = validTopkIndexTensorSize;
    topkIndexTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, topkIndexTensorAddr,
        topkIndexTensorSize / sizeof(int32_t));
    // copyTmpTensors_: TripleInfoCalAndDispatch 的 6 个 dispatch buffer(EVENT_ID0..EVENT_ID5 软流水), 各 8KB 容纳 token+scale。
    uint32_t copyTmpBaseAddr = topkIndexTensorAddr + topkIndexTensorSize;
    for (int32_t index = 0; index < DISPATCH_BUFFER_NUM; ++index) {
        copyTmpTensors_[index] = LocalTensor<ActivationType>(TPosition::VECCALC,
            copyTmpBaseAddr + static_cast<uint32_t>(index) * COPY_TMP_BUFFER_SIZE,
            COPY_TMP_BUFFER_SIZE / sizeof(ActivationType));
    }
    // expertTokenNumsOutTensor_: ExpertTokenNumCopyOut 用, 本卡各专家收到的 tokenCnt 数。
    uint32_t expertTokenNumsOutTensorAddr = copyTmpBaseAddr + copyTmpTotalSize;
    expertTokenNumsOutTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, expertTokenNumsOutTensorAddr,
        expertTokenNumsOutTensorSize / sizeof(int32_t));
    // [ubBufferUsedAddr_, ...): 剩余空间留给 TripleInfoCalAndDispatch 每 sub-slice 动态重建的 tripleTensor_,
    // 单批 ≤ DISPATCH_TRIPLE_TILE 行 * 32B(已在 fixedBytes 里预留)。
    ubBufferUsedAddr_ = expertTokenNumsOutTensorAddr + expertTokenNumsOutTensorSize;
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

    // 时段 A 布局从 UB 地址 0 起(跨卡同步后整体被时段 B DispatchBuffInit 覆盖复用)。
    // 先按各 tensor 实际大小算出与 chunk 无关的固定占用, 再用 UB 余量反推发侧 chunk, 最后按既定顺序落地址。

    // —— 与 chunk 无关的固定占用, 先逐项算大小 ——
    // Tensor用处：resetTensor_ —— groupFlagList 清零用的零缓冲(封顶到 resetBatchNum_, 超额时 ResetFlagList 分批推 GM, §6.7)。
    // Tensor大小：resetBatchNum_(=min(本核 flag 份额, DISPATCH_RESET_BATCH))个 int32, 对齐 256B。
    uint64_t flagInt32 = static_cast<uint64_t>(expertPerRank_) *
        (static_cast<uint64_t>(INT_CACHELINE) + static_cast<uint64_t>(dispatchFlagSlotsPerExpert_));
    uint32_t resetPerCore = Ops::Base::CeilDiv(flagInt32, static_cast<uint64_t>(blockAivNum_));
    resetBatchNum_ = resetPerCore < static_cast<uint32_t>(DISPATCH_RESET_BATCH) ?
        static_cast<int32_t>(resetPerCore) : DISPATCH_RESET_BATCH;
    uint32_t resetTensorSize = Ops::Base::CeilAlign(static_cast<uint64_t>(resetBatchNum_),
        static_cast<uint64_t>(INT32_PER_256B)) * sizeof(int32_t);
    // Tensor用处：mxTempTensor_ —— QuantProcessInRank 量化计算中间区域。
    // Tensor大小：2KB。
    uint32_t mxTempTensorSize = 2 * 1024;
    // Tensor用处：xOutTensor1_/2_(双 buffer) —— QuantProcessInRank 量化输出结果(token+scale)。
    // Tensor大小：token 长度 Align256 + scale 存储长度(sizeof(ActivationType)=1)。
    uint32_t xOutTokenBytes = Ops::Base::CeilAlign(static_cast<uint32_t>(k_ / ELEMS_PER_BYTE),
        static_cast<uint32_t>(ALIGN_256));
    uint32_t xOutTensorSize = xOutTokenBytes + Ops::Base::CeilDiv(k_, static_cast<uint32_t>(ALIGN_32));
    // Tensor用处：xInTensor1_/2_(双 buffer) —— QuantProcessInRank 搬入的输入 token。
    // Tensor大小：H 长度 Align128 * sizeof(bf16)。
    uint32_t xInAlignSize = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_128)) * sizeof(bfloat16_t);
    // Tensor用处：sendCntAccumulateTensor_ —— SendMaskCal 每核 per-owned-expert 跨 chunk count 累加器。
    // Tensor大小：每核 owned 全局专家数上界 ceil(ws*epr/核)个 int32, 对齐 32B; 与 bs 无关。
    uint32_t expertPerCoreMax = Ops::Base::CeilDiv(worldSize_ * expertPerRank_, blockAivNum_);
    uint32_t sendCntAccSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(expertPerCoreMax * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    // sendMaskTensor_ 双 buffer 各含 32B count 尾(末块把累计 count 折叠进切片尾一起 push); 这 32B 部分与 chunk 无关。
    uint32_t sendMaskFixedBytes = static_cast<uint32_t>(DOUBLE_BUFFER) * static_cast<uint32_t>(ALIGN_32);
    uint32_t fixedBytes = resetTensorSize + mxTempTensorSize + 2 * xOutTensorSize + 2 * xInAlignSize +
        sendCntAccSize + sendMaskFixedBytes;

    // —— 用 UB 余量反推发侧 chunk(与收侧无耦合, 各算各的) ——
    // chunk 随 token 线性增长的 UB: topkChunk(4B/tok) + sendGatherOut(4B/tok) + sendMask 双buf(DOUBLE_BUFFER * 1/8 B/tok)。
    // 解 chunk*((4+4)B + DOUBLE_BUFFER/8 B) ≤ 余量 ⇒ chunk ≤ 余量*8 / ((4+4)*8 + DOUBLE_BUFFER)(分母 66 即 8.25 B/token)。
    int64_t ubBudget = static_cast<int64_t>(params_.tilingData->ubSize) - DISPATCH_UB_SAFETY;
    int64_t avail = ubBudget - static_cast<int64_t>(fixedBytes);
    int64_t ckLimit = avail * 8 / ((4 + 4) * 8 + static_cast<int64_t>(DOUBLE_BUFFER));
    int64_t ckCap = Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_), static_cast<int64_t>(ALIGN_256));
    int64_t ck = ckLimit < ckCap ? ckLimit : ckCap;
    ck = ck / ALIGN_256 * ALIGN_256;                  // 向下取整到 256 倍数 ⇒ chunk/8 mask 切片天然 32B 对齐
    ck = ck < ALIGN_256 ? ALIGN_256 : ck;             // 兜底下限(极端大 H 时固定占用已近满)
    sendChunkTokenNum_ = static_cast<int32_t>(ck);
    sendChunkNum_ = static_cast<int32_t>(
        Ops::Base::CeilDiv(sendTotalNum_, static_cast<uint64_t>(sendChunkTokenNum_)));

    // —— 按既定顺序落地址(复用上面的 size) ——
    // Tensor用处：topkChunkTensor_ —— 当前 chunk 的 topk(CompareScalar 处理 chunk 个 int)。SendMaskCal 末块先 Duplicate(-1)
    //   填无效位、再 DataCopyPad 有效段(尾部保持 -1, 不会误命中专家 0)。
    // Tensor大小：chunk*4 对齐 256B。
    uint32_t topkChunkAddr = 0;
    uint32_t topkChunkSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(sendChunkTokenNum_) * static_cast<int64_t>(sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_256));
    topkChunkTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, topkChunkAddr, topkChunkSize / sizeof(int32_t));
    uint32_t resetTensorAddr = topkChunkAddr + topkChunkSize;
    resetTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, resetTensorAddr, resetTensorSize / sizeof(int32_t));
    uint32_t mxTempTensorAddr = resetTensorAddr + resetTensorSize;
    mxTempTensor_ = LocalTensor<uint16_t>(TPosition::VECCALC, mxTempTensorAddr, mxTempTensorSize / sizeof(uint16_t));
    uint32_t xOutTensorAddr1 = mxTempTensorAddr + mxTempTensorSize;
    xOutTensor1_ = LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr1,
        xOutTensorSize / sizeof(ActivationType));
    uint32_t xOutTensorAddr2 = xOutTensorAddr1 + xOutTensorSize;
    xOutTensor2_ = LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr2,
        xOutTensorSize / sizeof(ActivationType));
    uint32_t xInAlignAddr1 = xOutTensorAddr2 + xOutTensorSize;
    xInTensor1_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr1, xInAlignSize / sizeof(bfloat16_t));
    uint32_t xInAlignAddr2 = xInAlignAddr1 + xInAlignSize;
    xInTensor2_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr2, xInAlignSize / sizeof(bfloat16_t));
    // Tensor用处：sendMaskTensor_ 双 buffer(配 V->MTE3 软流水) —— 源卡每 (chunk,expert) 算出的 mask 切片。
    // Tensor大小：各 = chunk/8 mask 切片 + 32B(末块折叠 count); chunk/8 天然 32B 对齐。
    uint32_t maskBufBytes = static_cast<uint32_t>(sendChunkTokenNum_ / 8) + static_cast<uint32_t>(ALIGN_32);
    uint32_t sendMaskAddr = xInAlignAddr2 + xInAlignSize;
    for (int32_t index = 0; index < DOUBLE_BUFFER; ++index) {
        sendMaskTensor_[index] = LocalTensor<uint8_t>(TPosition::VECCALC,
            sendMaskAddr + static_cast<uint32_t>(index) * maskBufBytes, maskBufBytes);
    }
    // Tensor用处：sendGatherOutTensor_ —— SendMaskCal GatherMask 计 count 用的废弃输出 scratch。
    // Tensor大小：chunk*4 对齐 256B。
    uint32_t sendGatherOutAddr = sendMaskAddr + static_cast<uint32_t>(DOUBLE_BUFFER) * maskBufBytes;
    uint32_t sendGatherOutSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(sendChunkTokenNum_) * static_cast<int64_t>(sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_256));
    sendGatherOutTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, sendGatherOutAddr,
        sendGatherOutSize / sizeof(int32_t));
    uint32_t sendCntAccAddr = sendGatherOutAddr + sendGatherOutSize;
    sendCntAccumulateTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, sendCntAccAddr,
        sendCntAccSize / sizeof(int32_t));
    Duplicate<int32_t>(resetTensor_, 0, (resetTensorSize / sizeof(int32_t)));
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
    // workSpace groupFlagList 清零。总数 = SwiGluToGmm2(expertPerRank * INT_CACHELINE, 与 bs 无关) +
    // DispatchToGmm1(expertPerRank * dispatchFlagSlotsPerExpert_, ∝ maxOutputSize ∝ bs)。仅后者随 bs 膨胀,
    // 故按 resetBatchNum_ 分批只为它兜底; swiglu->gmm2 那段恒定很小, 与本批不会超 UB。
    groupFlagListGm_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.flagSwiGluToGmm2Ptr);
    int32_t flagNum = static_cast<int32_t>(expertPerRank_) *
        (static_cast<int32_t>(INT_CACHELINE) + dispatchFlagSlotsPerExpert_);
    int32_t coreLen, coreOffset;
    TilingByCore(flagNum, coreLen, coreOffset, 1);
    // resetTensor_ 已在 SendAndQuantBuffInit Duplicate 清零; 一次 V->MTE3 同步即覆盖后续所有批(零 UB 只读, 批间无写冒险)。
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    // 本核份额 coreLen 按 resetBatchNum_ 分批推 GM: coreLen ≤ resetBatchNum_ 时单批, 完全退化为改前一次拷贝;
    // 仅大 bs(flag 份额超封顶)才多批, 用同一片零 UB 重复推 ⇒ reset UB 与 bs 解耦(§6.7)。
    for (int32_t off = 0; off < coreLen; off += resetBatchNum_) {
        int32_t cur = coreLen - off < resetBatchNum_ ? coreLen - off : resetBatchNum_;
        DataCopyExtParams rankSyncCopyParams{1U, static_cast<uint32_t>(cur * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(groupFlagListGm_[coreOffset + off], resetTensor_, rankSyncCopyParams);
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

    // 分块流式: outer = token-chunk(topk 只载一次/块), inner = 本核 owned 全局专家。
    // 每核 owned 专家数 ownedExpertNum = ceil((totalExperts-aivCoreIdx_)/blockAivNum_), 与 bs 无关。
    // 每 (chunk,expert) 把一片 mask(满块 chunk/8, 末块 clamp 到 maskAlignSize_)推到 win 槽内偏移 chunkStart/8;
    // count 跨 chunk 累加到 sendCntAccumulateTensor_[ownedIdx], 末块把累计 count 折叠进切片尾部一起 push。
    int32_t totalExperts = static_cast<int32_t>(worldSize_ * expertPerRank_);
    int32_t coreIdx = static_cast<int32_t>(aivCoreIdx_);
    int32_t ownedExpertNum = (coreIdx < totalExperts) ?
        Ops::Base::CeilDiv(totalExperts - coreIdx, static_cast<int32_t>(blockAivNum_)) : 0;
    if (ownedExpertNum <= 0) {
        return;   // 本核不负责任何全局专家(核数 > 专家数时尾部核)
    }

    GlobalTensor<int32_t> srcGlobalTensor;
    srcGlobalTensor.SetGlobalBuffer((__gm__ int32_t*)params_.expertIdxGmAddr);
    GlobalTensor<uint8_t> dstGlobalTensor;
    int32_t maskSliceBytesFull = sendChunkTokenNum_ / 8;  // 满块 mask 切片字节数(chunk 为 256 倍数 ⇒ 32B 对齐)
    DataCopyPadExtParams<int32_t> loadPad{false, 0U, 0U, 0U};

    // 每核 per-owned-expert 跨 chunk count 累加器清零(scalar 访问前需 V->S)。
    Duplicate<int32_t>(sendCntAccumulateTensor_, 0, ownedExpertNum);
    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();

    constexpr TEventID kBufEvents[DOUBLE_BUFFER] = {EVENT_ID0, EVENT_ID1};
    for (int32_t bufIdx = 0; bufIdx < static_cast<int32_t>(DOUBLE_BUFFER); ++bufIdx) {
        SetFlag<AscendC::HardEvent::MTE3_V>(kBufEvents[bufIdx]);
    }
    int32_t iter = 0;
    for (int32_t chunkIdx = 0; chunkIdx < sendChunkNum_; ++chunkIdx) {
        int32_t chunkStart = chunkIdx * sendChunkTokenNum_;
        // 绝大多数块都是满块: validLen/sliceBytes/pushBytes 直接取满块值, 只有末块才收口(走下面的 if)。
        bool isLastChunk = (chunkIdx == sendChunkNum_ - 1);
        int32_t validLen = sendChunkTokenNum_;
        int32_t sliceBytes = maskSliceBytesFull;
        int32_t pushBytes = sliceBytes;
        if (isLastChunk) {
            validLen = static_cast<int32_t>(sendTotalNum_) - chunkStart;
            // 末块 mask 字节数 clamp 到 maskAlignSize_(不越界 count 字段); chunkStart/8 与 maskAlignSize_ 均 32 对齐。
            if (chunkStart / 8 + sliceBytes > static_cast<int32_t>(maskAlignSize_)) {
                sliceBytes = static_cast<int32_t>(maskAlignSize_) - chunkStart / 8;
            }
            // count 回折: 末块把累计 count 紧跟 mask 切片一起 push(切片末尾恰好是槽的 count 字段偏移 maskAlignSize_)。
            pushBytes = sliceBytes + static_cast<int32_t>(sizeof(int32_t));
            // 仅末块有无效尾: Duplicate(-1) 填满, 下面 DataCopyPad 只覆盖有效段, 余下保持 -1(不会误命中专家0)。
            // 非末块整块都是有效 topk, DataCopyPad 全覆盖, 不必填 -1, 省一次 Duplicate。
            Duplicate<int32_t>(topkChunkTensor_, -1, sendChunkTokenNum_);
        }
        DataCopyExtParams maskCopyParams{1U, static_cast<uint32_t>(pushBytes), 0U, 0U, 0U};
        // V->MTE2: 保护 topkChunkTensor_ 复用(等上一轮 GatherMask 读完)兼使末块 Duplicate 对下面的 DataCopyPad 可见。
        SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID1>();
        DataCopyExtParams loadParams{1U, static_cast<uint32_t>(validLen * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(topkChunkTensor_, srcGlobalTensor[chunkStart], loadParams, loadPad);
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();

        for (int32_t ownedIdx = 0; ownedIdx < ownedExpertNum; ++ownedIdx, ++iter) {
            int32_t globalExpertId = coreIdx + ownedIdx * static_cast<int32_t>(blockAivNum_);
            int32_t dstRank = globalExpertId / static_cast<int32_t>(expertPerRank_);
            int32_t localExpertId = globalExpertId % static_cast<int32_t>(expertPerRank_);
            TEventID ev = kBufEvents[iter % static_cast<int32_t>(DOUBLE_BUFFER)];
            LocalTensor<uint8_t> maskBuf = sendMaskTensor_[iter % static_cast<int32_t>(DOUBLE_BUFFER)];
            LocalTensor<uint32_t> maskBufU32 = maskBuf.template ReinterpretCast<uint32_t>();

            WaitFlag<AscendC::HardEvent::MTE3_V>(ev);  // 等本 buffer 上一轮 MTE3 推送完成
            CompareScalar(maskBuf, topkChunkTensor_, globalExpertId, AscendC::CMPMODE::EQ, sendChunkTokenNum_);
            // 本 chunk 命中数: GatherMask 取 set-bit 数(mask=validLen 跳过尾部无效位)。
            uint64_t sendCnt = 0;
            GatherMask(sendGatherOutTensor_, topkChunkTensor_, maskBufU32, true,
                static_cast<uint32_t>(validLen), {1, 1, 0, 0}, sendCnt);
            SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();   // count 标量就绪(V 已排空, 覆盖 mask 的 V->MTE3)
            int32_t acc = sendCntAccumulateTensor_.GetValue(ownedIdx) + static_cast<int32_t>(sendCnt);
            sendCntAccumulateTensor_.SetValue(ownedIdx, acc);
            if (isLastChunk) {
                // 把最终 count 折叠进切片尾部(buffer 字节偏移 sliceBytes → win 偏移 maskAlignSize_)。
                maskBuf.template ReinterpretCast<int32_t>().SetValue(sliceBytes / sizeof(int32_t), acc);
            }
            SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>(); // mask(V)/count(S) 对 MTE3 可见

            uint64_t dstOffset = maskWinOffset_ + static_cast<uint64_t>(
                localExpertId * static_cast<int32_t>(worldSize_) + static_cast<int32_t>(rankId_)) *
                static_cast<uint64_t>(maskSlotSize_) + static_cast<uint64_t>(chunkStart / 8);
            dstGlobalTensor.SetGlobalBuffer(
                (__gm__ uint8_t*)GetRankWinAddrWithOffset(dstRank, dstOffset));
            DataCopyPad(dstGlobalTensor, maskBuf, maskCopyParams);
            SetFlag<AscendC::HardEvent::MTE3_V>(ev);
        }
    }
    for (int32_t bufIdx = 0; bufIdx < static_cast<int32_t>(DOUBLE_BUFFER); ++bufIdx) {
        WaitFlag<AscendC::HardEvent::MTE3_V>(kBufEvents[bufIdx]);
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
            static_cast<uint64_t>(curentOffset + index) * H * sizeof(bfloat16_t)));
        dstGlobalTensor.SetGlobalBuffer((__gm__ uint8_t*)(params_.peermemInfo.quantTokenScalePtr +
            static_cast<uint64_t>(curentOffset + index) * mxQuantTokenScaleAlignBytes_));
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
//   Phase 1: 用 stride 从本卡 win 只读当前 localExpertId 的 worldSize_ 个 count 字段进 sendCntTensor_(跳过 mask);
//   Phase 2: 逐卡读取count → 累加 totalSendCntInExp_/cumsumRevCntInRank_, 写 cumsumInfoTensor_;
//   Phase 3: 写 expertRevNumsGloablTensor_ + CrossCoreSetFlag 通知 AIC;
// ==================================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::SendCntCal(int32_t localExpertId)
{
    // 分块流式后 mask 不再常驻 UB: TripleInfoCalAndDispatch 自行按 (dstRank,chunk) 直接读 win mask 切片,
    // 这里只 stride 读各槽的 count 字段(跳过 mask 区), 消除整槽 mask 的常驻读。
    totalSendCntInExp_ = 0;

    // Phase 1: 用 stride 从本卡 win 只读本 localExpert 的 ws 个 count 字段(各槽 count 偏移 maskAlignSize_)
    GlobalTensor<int32_t> cntSrcGlobal;
    cntSrcGlobal.SetGlobalBuffer((__gm__ int32_t*)(params_.peermemInfo.maskRecvPtr +
        static_cast<uint64_t>(localExpertId) * worldSize_ * maskSlotSize_ + maskAlignSize_));
    DataCopyExtParams cntCopyParams{static_cast<uint16_t>(worldSize_), static_cast<uint32_t>(sizeof(int32_t)),
        static_cast<uint32_t>(maskSlotSize_ - sizeof(int32_t)), 0U, 0U};   // src 每块跳过 (slot-4) 字节
    DataCopyPadExtParams<int32_t> cntPad{false, 0U, 0U, 0U};
    DataCopyPad(sendCntTensor_, cntSrcGlobal, cntCopyParams, cntPad);
    SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>();   // count 读取(标量)就绪

    // Phase 2: 逐源卡读取 count + cumsum (每 burst 在 UB 内 32B 对齐落位 → 下标 = rank*8)
    constexpr int32_t CNT_STRIDE_I32 = ALIGN_32 / sizeof(int32_t);
    for (int32_t calRankId = 0; calRankId < static_cast<int32_t>(worldSize_); ++calRankId) {
        int32_t sendCnt = sendCntTensor_.GetValue(calRankId * CNT_STRIDE_I32);
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
        uint32_t aivIdxInGroup = idx % blockNumPerRank_;
        // 本 (expert,dstRank) 命中总数直接由 cumsum 表推出, 无需扫全 ST 的 GatherMask:
        //   sendToCurExpTokenCnt = cumsum[le*ws+dstRank] - cumsum[le*ws+dstRank-1]。
        uint32_t rowStartIdxInDst = ((dstRankIdx == 0 && localExpertId == 0) ? 0 :
            cumsumInfoTensor_.GetValue(localExpertId * worldSize_ + dstRankIdx - 1));
        // 本核负责的【命中索引】区间 [coreLo, coreHi) 与目标起始行 rowDstOffsetInCore, 与原逐字一致。
        int32_t coreLo = 0;
        int32_t coreHi = 0;
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
                coreLo = rowSrcIdxInCore;
                coreHi = rowSrcIdxInCore + rowsNumInCore;
            }
        }

        // 单趟分块: 升序扫 chunk, 逐块读本 dstRank 的 mask 切片 → GatherMask 出本块命中(全局 token 位置),
        // 与本核命中区间 [coreLo, coreHi) 求交, 把命中的 sub-slice 即时 CopyGMToGMPerToken 派发。
        // 升序扫 + GatherMask 升序压缩 ⇒ 命中 token 落到与改前全 ST 顺序相同的 dispatchRevData 行。
        GlobalTensor<uint8_t> maskSrcGlobal;
        int32_t matchedSoFar = 0;
        int32_t rowsThisCore = 0;
        for (int32_t chunkIdx = 0; chunkIdx < recvChunkNum_ && matchedSoFar < coreHi; ++chunkIdx) {
            int32_t chunkStart = chunkIdx * recvChunkTokenNum_;
            // 绝大多数块都是满块: validLen/sliceBytes 直接取满块值, 只有末块才收口(走下面的 if)。
            bool isLastChunk = (chunkIdx == recvChunkNum_ - 1);
            int32_t validLen = recvChunkTokenNum_;
            int32_t sliceBytes = recvChunkTokenNum_ / 8;
            if (isLastChunk) {
                validLen = static_cast<int32_t>(sendTotalNum_ - static_cast<uint64_t>(chunkStart));
                // 末块 mask 读取字节数 clamp 到 maskAlignSize_(与源写入对称, 不读到 count 字段)。
                if (chunkStart / 8 + sliceBytes > static_cast<int32_t>(maskAlignSize_)) {
                    sliceBytes = static_cast<int32_t>(maskAlignSize_) - chunkStart / 8;
                }
            }
            maskSrcGlobal.SetGlobalBuffer((__gm__ uint8_t*)(params_.peermemInfo.maskRecvPtr +
                (static_cast<uint64_t>(localExpertId) * worldSize_ + dstRankIdx) * maskSlotSize_ +
                static_cast<uint64_t>(chunkStart / 8)));
            DataCopy(maskChunkTensor_, maskSrcGlobal, static_cast<uint32_t>(sliceBytes));
            SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();   // mask 切片供 GatherMask(V)
            CreateVecIndex(topkIndexTensor_, chunkStart, recvChunkTokenNum_);   // 本块全局 token 位置
            uint64_t matchedInChunk = 0;
            GatherMask(validTopkIndexTensor_, topkIndexTensor_, maskChunkU32Tensor_, true,
                static_cast<uint32_t>(validLen), {1, 1, 0, 0}, matchedInChunk);
            // V_S 排空 GatherMask: (a) matchedInChunk 标量就绪; (b) validTopkIndex 供 CopyGMToGM 标量读;
            // (c) 作 V->下一轮 DataCopy(MTE2) 的程序序屏障, 保护 maskChunkTensor_/validTopkIndex 复用。
            SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID4>();
            int32_t gLo = matchedSoFar;
            int32_t gHi = matchedSoFar + static_cast<int32_t>(matchedInChunk);
            int32_t lo = gLo > coreLo ? gLo : coreLo;
            int32_t hi = gHi < coreHi ? gHi : coreHi;
            if (hi > lo) {
                int32_t subStart = lo - matchedSoFar;             // chunk-local 索引(validTopkIndex_ 内偏移)
                int32_t subNum = hi - lo;
                int32_t subDst = static_cast<int32_t>(rowStartIdxInDst) + lo;  // 绝对 dispatchRevData 行
                // triple-tile 解耦: subNum 可达整个 chunk(很大), 按 DISPATCH_TRIPLE_TILE 行分批,
                // tripleTensor_ 每行 32B(INT32_PER_256B 个 int32), 每批 ≤ TT*32B, 与 chunk 解耦, 保证 UB 不爆。
                for (int32_t off = 0; off < subNum; off += DISPATCH_TRIPLE_TILE) {
                    int32_t n = (subNum - off > DISPATCH_TRIPLE_TILE) ? DISPATCH_TRIPLE_TILE : (subNum - off);
                    tripleTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, ubBufferUsedAddr_,
                        static_cast<uint32_t>(n) * INT32_PER_256B);
                    CopyGMToGMPerToken(subDst + off, dstRankIdx, subStart + off, n);
                }
                rowsThisCore += subNum;
            }
            matchedSoFar = gHi;
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
__aicore__ inline bool MegaMoe<TemplateMegaMoeTypeFunc>::UpdateGroupParams(ExpertLoopState &state, uint32_t expertIdx)
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
            // 处理完本专家, 归还 1 个 dispatch 信用给 sb1(aic->aiv1 反压, 见 GMM1_CREDIT_FLAG 注释)。
            // 放在 M==0 早退判断之前 ⇒ 每个专家恒归还 1 个, 与 sb1 每专家消费 1 个配平, 不会死锁。
            CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_S>(GMM1_CREDIT_FLAG + 16);
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
//                     该地址用于后续GroupMatmulSwigluQuant 与 GroupMatmul2Combine运算；
// ==================================================================================
template <TemplateMegaMoeTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo,
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
    gmmAddrInfo.groupFlagList = (__gm__ int32_t*)params_.workspaceInfo.flagSwiGluToGmm2Ptr +
                                Get<IDX_FLAG_OFFSET>(state.baseOffset) * INT_CACHELINE;
    // wave-grain dispatch-gmm1 flag: per-expert 步长是 dispatchFlagSlotsPerExpert_,而不是 INT_CACHELINE。
    gmmAddrInfo.groupFlagList2 = (__gm__ int32_t*)params_.workspaceInfo.flagDispatchToGmm1Ptr +
                                Get<IDX_FLAG_OFFSET>(state.baseOffset) * dispatchFlagSlotsPerExpert_;
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
    // 大 bs 适配: weights 只载本核负责的 token 段(Unpermute 按 TilingByCore 分核, 每核只读自己
    // [wCoreOffset_, wCoreOffset_+wCoreLen) 的 weights), UB 占用 O(bs/coreNum) 而非 O(bs)。
    int32_t wCoreLen = 0;
    TilingByCore(m_, wCoreLen, wCoreOffset_, 1);
    wCoreLen = wCoreLen < 0 ? 0 : wCoreLen;
    uint32_t topKWeightsBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(wCoreLen * topK_ * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
    uint32_t tempBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(wCoreLen * topK_ * sizeof(bfloat16_t)), uint32_t(ALIGN_32));
    
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
    // 本核 token 段的 topkWeights(下标 = (token-wCoreOffset_)*topK), Unpermute 驻留只读。
    uint32_t topKWeightsAddr = dataResFp32Addr + dataResFp32BufAlign;
    uint32_t topKWeightsSize = topKWeightsBufAlign / sizeof(float);
    topKWeightsTensor_ = LocalTensor<float>(TPosition::VECCALC, topKWeightsAddr, topKWeightsSize);
    uint32_t tempAddr = topKWeightsAddr + topKWeightsBufAlign;
    if (wCoreLen <= 0) {
        return;   // 本核无 token(核数 > bs), Unpermute 循环也是空
    }
    if constexpr (Std::IsSame<TopkWeightsType, float>::value) {
        GlobalTensor<float> topKWeightsGlobalTensor_;
        topKWeightsGlobalTensor_.SetGlobalBuffer(
            (__gm__ float*)params_.probsGmAddr + static_cast<uint64_t>(wCoreOffset_) * topK_);
        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(wCoreLen * topK_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(topKWeightsTensor_, topKWeightsGlobalTensor_, copyParams, copyPadParams);
    }
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        uint32_t tempSize = tempBufAlign / sizeof(bfloat16_t);
        LocalTensor<bfloat16_t> tempLocal(TPosition::VECCALC, tempAddr, tempSize);
        GlobalTensor<bfloat16_t> topkWeightsGlobalTensor;
        topkWeightsGlobalTensor.SetGlobalBuffer(
            (__gm__ bfloat16_t*)params_.probsGmAddr + static_cast<uint64_t>(wCoreOffset_) * topK_);
        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(wCoreLen * topK_ * sizeof(bfloat16_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<bfloat16_t> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(tempLocal, topkWeightsGlobalTensor, copyParams, copyPadParams);
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID2>();
        Cast(topKWeightsTensor_, tempLocal, AscendC::RoundMode::CAST_NONE, wCoreLen * topK_);
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
            float expScale = topKWeightsTensor_.GetValue((tokenIdx - wCoreOffset_) * topK_ + expId);
            auto event = (expId % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
            auto dataInBf16 = (expId % DOUBLE_BUFFER == 0) ? dataIn0Bf16 : dataIn1Bf16;
            auto dataInFp32 = (expId % DOUBLE_BUFFER == 0) ? dataIn0Fp32 : dataIn1Fp32;
            WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            DataCopy(dataInBf16, expandedX[(static_cast<uint64_t>(tokenIdx) * topK_ + expId) * k_], k_);
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
        DataCopy(output[static_cast<uint64_t>(tokenIdx) * k_], dataResTensor_, k_);
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
    // dispatch 反压预发信用: AIC 先放 GMM1_DISPATCH_CREDITS 个信用, 允许 sb1 提前这么多个专家做 dispatch;
    // 之后每处理完一专家归还 1 个(UpdateGroupParams 里), sb1 每专家消费 1 个 → sb1 领先 AIC 恒 ≤ 信用数,
    // flag2/credit 两个计数器都恒 <15, 与 epr 无关(epr=1024 也不溢出)。预发的这 N 个比消费多出 N 个,
    // 通道 8 在 GMM1 之后不再被复用, 故留到 kernel 末尾由 sb1 一次性排空(见 Process 结尾 drain),
    // 只需保证退出前归零、不污染后续 kernel。
    if constexpr (g_coreType == AscendC::AIC) {
        for (int c = 0; c < GMM1_DISPATCH_CREDITS; c++) {
            CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_S>(GMM1_CREDIT_FLAG + 16);
        }
    }
    for (int localExpertId = 0; localExpertId < expertPerRank_; localExpertId++) {
        if constexpr(g_coreType == AIV) {
            if (subBlockIdx_ == 1) {
                CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_S>(GMM1_CREDIT_FLAG);  // 消费 1 信用: 领先 AIC ≤ 信用数
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
    // kernel 收尾: sb1 排空 AIC 在 GMM1 预发但未被循环消费的 GMM1_DISPATCH_CREDITS 个信用(每专家归还 1, 预发的 N 个恒多出)。
    // 通道 8 在 GMM1 之后无人复用, AIC 早已发齐全部归还, 这 N 个必定可取; 取空使计数器归零, 不污染后续 kernel。
    if constexpr(g_coreType == AIV) {
        if (subBlockIdx_ == 1) {
            for (int c = 0; c < GMM1_DISPATCH_CREDITS; c++) {
                CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_S>(GMM1_CREDIT_FLAG);
            }
        }
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

}   // namespace MegaMoeImpl
#endif
