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
    Params params_{};
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
    uint32_t maskSlotSize_ = 0;    // 单个 win 槽位 = maskAlignSize_(mask) + 32B(count)
    uint64_t maskWinOffset_ = 0;   // maskRecvPtr 相对 win 基址(rankSyncInWorldPtr)的偏移
    uint64_t quantWinOffset_ = 0;  // quantTokenScalePtr 相对 win 基址的偏移
    uint64_t totalSendCntInExp_ = 0;
    uint64_t cumsumRevCntInRank_ = 0;
    // 分块流式 dispatch: 所有随 ST=bs*topK 增长的 UB 缓冲都按 DISPATCH_TOKEN_CHUNK(见 mega_moe_base.h)
    // 个 token 分批, 循环 numChunks_=ceil(ST/CHUNK) 次, 使 UB 占用与 bs 解耦(win/HBM 仍按全 ST 分配)。
    // CHUNK=2048(=8*256): CHUNK/8=256B mask 切片天然 32B 对齐, 段 B 峰值 ≈134KB 安全。
    int32_t numChunks_ = 0;

    // 6-buffer 软流水：占满 EVENT_ID0..EVENT_ID5。
    //   - CopyGMToGMPerToken 用 copyTmpTensors_[6] 让 winmem token+scale 读
    //     在引擎里同时排队 5+ 个 (配合 prime/steady/drain 软流水)。
    static constexpr int32_t DISPATCH_BUFFER_NUM = 6;
    LocalTensor<int32_t> topkIndexTensor_;                // 每 chunk 重建的全局 token 位置 [chunkStart, +CHUNK)
    LocalTensor<uint8_t> maskChunk_;                      // 接收端: 当前 dstRank 的一个 CHUNK/8 mask 切片
    LocalTensor<uint32_t> maskChunkU32_;                  // maskChunk_ 的 u32 视图(同址别名), 供 GatherMask
    LocalTensor<int32_t> sendCntTensor_;                  // 接收端: 本 expert 的 ws 个 srcRank count
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
    LocalTensor<int32_t> topkChunk_;                      // SendMaskCal 源卡: 当前 chunk 的 topk(尾部哨兵 -1)
    LocalTensor<uint8_t> sendMaskTensor_[DOUBLE_BUFFER];  // SendMaskCal 源卡算 mask 切片的 ping-pong 缓冲
    LocalTensor<int32_t> sendGatherOutTensor_;            // SendMaskCal GatherMask 计 count 的废弃输出 scratch
    LocalTensor<int32_t> sendCntAcc_;                     // SendMaskCal 源卡每核 per-owned-expert 跨 chunk count 累加
    LocalTensor<int32_t> expertTokenNumsOutTensor_;
    LocalTensor<bfloat16_t> dataResTensor_;
    LocalTensor<int32_t> resetTokenTensor_;
    LocalTensor<float> dataResFp32Tensor_;
    LocalTensor<float> topKWeightsTensor_;

    using BlockEpilogue = BlockEpilogueSwigluMxQuant<QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, true>;
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
    // 与 WorkspaceInfo 构造里 flagDispatchToGmm1Ptr 的分配公式保持一致。
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
    params_.workspaceInfo = WorkspaceInfo(workspaceGM, tilingData);
    params_.peermemInfo = PeermemInfo(winRankAddr_[rankId_], tilingData);
    params_.tilingData = tilingData;
    expertTokenNumsOut_.SetGlobalBuffer((__gm__ int32_t*)params_.expertTokenNumsOutGmAddr);
    expertRevNumsGloablTensor_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.expertRevTokenNumsPtr);
    tripleGloablTensor_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.tripleInfoPtr);
    epilogueOp_.Init({params_.workspaceInfo.swigluQuantDataPtr, params_.workspaceInfo.swigluQuantScalePtr,
        params_.workspaceInfo.flagSwiGluToGmm2Ptr, nullptr, nullptr, nullptr, ALIGN_256, ALIGN_256});
    // 各 win 区相对 win 基址(rankSyncInWorldPtr)的偏移; 所有卡 win 布局一致, 跨卡读写用同一偏移。
    maskWinOffset_ = static_cast<uint64_t>(params_.peermemInfo.maskRecvPtr - params_.peermemInfo.rankSyncInWorldPtr);
    quantWinOffset_ = static_cast<uint64_t>(params_.peermemInfo.quantTokenScalePtr - params_.peermemInfo.rankSyncInWorldPtr);
    // 分块流式: 整条 dispatch 流水按 DISPATCH_TOKEN_CHUNK 个 token 分批, 共 numChunks_ 批。
    numChunks_ = static_cast<int32_t>(Ops::Base::CeilDiv(sendTotalNum_,
        static_cast<uint64_t>(DISPATCH_TOKEN_CHUNK)));
    // maskAlignSize_/maskSlotSize_ 仅依赖 sendTotalNum_, 公式与改前一致(分块不改 win 槽位大小):
    // 源按 chunk 分片填同一 mask 区, 末片 clamp 到 maskAlignSize_。须与 PeermemInfo(mega_moe_base.h) 一致。
    uint32_t compareCount = Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_256)) / sizeof(int32_t);
    maskAlignSize_ = Ops::Base::CeilAlign(static_cast<int64_t>(compareCount) / 8, static_cast<int64_t>(ALIGN_32));
    maskSlotSize_ = maskAlignSize_ + static_cast<uint32_t>(ALIGN_32);
    mxFp8ScaleNumAlignPerToken_ = Ops::Base::CeilDiv(k_, static_cast<uint32_t>(ALIGN_32));
    mxFp8TokenAlignBytes_ = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_256)) * sizeof(QuantOutType);
    mxFp8ScaleAlignBytes_ = mxFp8ScaleNumAlignPerToken_ * sizeof(uint8_t);
    mxFp8TokenScaleAlignBytes_ = Ops::Base::CeilAlign(mxFp8TokenAlignBytes_ + mxFp8ScaleAlignBytes_,
        static_cast<uint32_t>(ALIGN_32));
}

// ============================================================================
// UB(unified buffer, 每个 AIV 独立 256KB) 总体划分 —— 按"时段"整块复用:
// 三个 BuffInit 都从 UB 地址 0 重新铺平, 时段之间有 PipeBarrier/SyncAll 隔离,
// 所以【跨时段同地址重叠是有意复用】; 同一时段内部各 tensor 顺序排布、互不重叠。
// 记号: ST = sendTotalNum_ = bs*topK, CK = DISPATCH_TOKEN_CHUNK = 8192(分批 token 数),
//       TT = DISPATCH_TRIPLE_TILE = 2048, slot = maskSlotSize_ = align32(cc/8)+32B(win 仍存全 ST mask, 大小不变),
//       ws = worldSize_, epr = expertPerRank_, H = k_。
// ★ 分块流式后所有 O(ST) 缓冲都降为 O(CK), UB 占用与 bs 解耦; 循环 numChunks_=ceil(ST/CK) 次。
//
// 时段 A: BuffInit (SendMaskCal / ResetFlagList / QuantProcessInRank, 跨卡同步前)
//   [0)                topkChunk_          4*CK          当前 chunk 的 topk(尾部哨兵 -1)
//   [+4CK)             resetTensor_        ~flag清零区/核
//   [+...)             mxTempTensor_       2KB           mx 量化中间 exp/scale
//   [+...)             xOutTensor1_/2_     2×(align256(H)+H/32)  量化输出 token+scale ping-pong
//   [+...)             xInTensor1_/2_      2×align128(H)*sizeof(XType)  bf16 输入 ping-pong
//   [+...)             sendMaskTensor_[2]  2×(CK/8+32)   mask 切片(+末块折叠 count)推送 ping-pong
//   [+...)             sendGatherOutTensor_ 4*CK         GatherMask 计数废弃输出
//   [+...)             sendCntAcc_         numMyExperts*4  每核 owned-expert 跨 chunk count 累加(O(ws*epr/核))
//   合计 ≈ 8.25*CK + 2KB + ~6.5*H + flag区 ≈ 118KB @ CK=8192,H=7168 —— 与 bs 无关。
//
// 时段 B: FirstBuffInit (dispatch + GMM1 循环, 覆盖时段 A 同一片 UB)
//   [0)                expertTokenCntTensor_   32B
//   [+32)              cumsumInfoTensor_       align32(ws*epr*4)   全程驻留
//   [+...)             sendCntTensor_          align32(ws*4)       接收端 count-only 读
//   [+...)             maskChunk_/U32_         CK/8     ★两 tensor 同地址别名(u8/u32 视图), 单 dstRank mask 切片
//   [+...)             validTopkIndexTensor_   align32(4CK)
//   [+...)             topkIndexTensor_        align32(4CK)  每 chunk 重 CreateVecIndex
//   [+...)             copyTmpTensors_[6]      6×8KB=48KB  token+scale 软流水
//   [+...)             expertTokenNumsOutTensor_ align32(epr*4)
//   [ubBufferUsedAddr_) tripleTensor_(动态)    ≤TT*32B  尾部剩余区, triple-tile 分批重建(与 CK 解耦)
//   合计 ≈ 8*CK + 48KB + TT*32 + 小项 ≈ 179KB @ CK=8192,TT=2048 —— 与 bs 无关。
//
// 时段 C: SecondBuffInit (Unpermute, 再次覆盖地址 0 起整片)
//   [0)                dataResTensor_      3*H*2B        bf16 累加 ping-pong(偏移 k_/2k_ 复用)
//   [+...)             resetTokenTensor_   ws*align128(ws*epr)*4
//   [+...)             dataResFp32Tensor_  3*H*4B
//   [+...)             topKWeightsTensor_  bs*topK*4B
//   [+...)             tempLocal(bf16 时)  bs*topK*2B    Cast 临时
// ============================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::FirstBuffInit()
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    // maskAlignSize_/maskSlotSize_ 已在 Init 算好。
    // 时段 B 布局从 UB 地址 0 起, 整体覆盖时段 A(BuffInit)的区域 —— 跨时段重叠是有意复用。
    // 分块流式后所有 O(ST) 缓冲降为 O(CK)(maskChunk_/validTopkIndex_/topkIndex_), 与 bs 无关。
    constexpr int32_t CK = DISPATCH_TOKEN_CHUNK;
    // [0, 32B): 单个 expert 收到 token 数(SendInfoCal 写 GM 用的 32B 暂存)。
    uint32_t expertTokenCntTensorAddr = 0;
    uint32_t expertTokenCntTensorSize = ALIGN_32;
    expertTokenCntTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, expertTokenCntTensorAddr,
        expertTokenCntTensorSize / sizeof(int32_t));
    // [32B, +ws*epr*4): 全局 [expert][srcRank] 收 token cumsum 表, 全程驻留(GMM offset/ExpertTokenNumCopyOut 用)。
    uint32_t cumsumInfoTensorAddr = expertTokenCntTensorAddr + expertTokenCntTensorSize;
    uint32_t cumsumInfoTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(worldSize_ * expertPerRank_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    cumsumInfoTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, cumsumInfoTensorAddr,
        cumsumInfoTensorSize / sizeof(int32_t));
    // [+ws*4): SendInfoCal 用 stride 一次性读本 expert 的 ws 个 srcRank count(只读 count, 不读 mask)。
    uint32_t sendCntTensorAddr = cumsumInfoTensorAddr + cumsumInfoTensorSize;
    uint32_t sendCntTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(worldSize_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    sendCntTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, sendCntTensorAddr,
        sendCntTensorSize / sizeof(int32_t));
    // [+CK/8): 接收端单个 (dstRank,chunk) 的 mask 切片。maskChunk_(u8)/maskChunkU32_(u32) 同址别名:
    //   u8 视图给 DataCopy 读 win, u32 视图给 GatherMask。
    uint32_t maskChunkAddr = sendCntTensorAddr + sendCntTensorSize;
    uint32_t maskChunkSize = static_cast<uint32_t>(DISPATCH_MASK_SLICE_BYTES);
    maskChunk_ = LocalTensor<uint8_t>(TPosition::VECCALC, maskChunkAddr, maskChunkSize / sizeof(uint8_t));
    maskChunkU32_ = LocalTensor<uint32_t>(TPosition::VECCALC, maskChunkAddr, maskChunkSize / sizeof(uint32_t));
    // [+4CK): GatherMask 输出 —— 命中当前 (dstRank,chunk) 的 chunk-local token 位置(每 chunk 覆写)。
    uint32_t validTopkIndexTensorAddr = maskChunkAddr + maskChunkSize;
    uint32_t validTopkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(CK * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    validTopkIndexTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, validTopkIndexTensorAddr,
        validTopkIndexTensorSize / sizeof(int32_t));
    // [+4CK): 当前 chunk 的全局 token 位置 [chunkStart, +CK)(每 chunk 重 CreateVecIndex, 作 GatherMask 的 src)。
    uint32_t topkIndexTensorAddr = validTopkIndexTensorAddr + validTopkIndexTensorSize;
    uint32_t topkIndexTensorSize = validTopkIndexTensorSize;
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
    // [+epr*4): per-expert 收 token 数输出暂存(ExpertTokenNumCopyOut 用)。
    uint32_t expertTokenNumsOutTensorAddr = copyTmpBaseAddr +
        static_cast<uint32_t>(DISPATCH_BUFFER_NUM) * COPY_TMP_BUFFER_SIZE;
    uint32_t expertTokenNumsOutTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(expertPerRank_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    expertTokenNumsOutTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, expertTokenNumsOutTensorAddr,
        expertTokenNumsOutTensorSize / sizeof(int32_t));
    // [ubBufferUsedAddr_, 256KB): 时段 B 剩余空间, 留给 TripleInfoCalAndDispatch 每 sub-slice 动态重建的
    // tripleTensor_(subNum*32B ≤ CK*32B), 是隐式上限: CK*32 ≤ 256KB - ubBufferUsedAddr_。
    ubBufferUsedAddr_ = expertTokenNumsOutTensorAddr + expertTokenNumsOutTensorSize;
    // topkIndexTensor_ 改为每 chunk CreateVecIndex(chunkStart, CK), 此处不再一次性生成。
    Duplicate<int32_t>(cumsumInfoTensor_, 0, (cumsumInfoTensorSize / sizeof(int32_t)));
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::BuffInit()
{
    if constexpr(g_coreType == AIC) {
        return;
    }

    // 时段 A 布局从 UB 地址 0 起(跨卡同步后整体被时段 B FirstBuffInit 覆盖复用)。
    // [0, 4*CK): 当前 chunk 的 topk(CompareScalar 处理 CK 个 int, 256B 对齐)。每 chunk 在 SendMaskCal
    // 里先 Duplicate(-1) 哨兵再 DataCopyPad 有效段(尾 chunk 不足 CK 的部分保持 -1, 不会误命中专家 0)。
    uint32_t topkChunkAddr = 0;
    uint32_t topkChunkSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(DISPATCH_TOKEN_CHUNK * sizeof(int32_t)), static_cast<int64_t>(ALIGN_256));
    topkChunk_ = LocalTensor<int32_t>(TPosition::VECCALC, topkChunkAddr, topkChunkSize / sizeof(int32_t));
    uint32_t resetTensorAddr = topkChunkAddr + topkChunkSize;
    // 总 int32 flag entries: SwiGluToGmm2 区 (expertPerRank * INT_CACHELINE)
    // + DispatchToGmm1 区 wave-grain (expertPerRank * dispatchFlagSlotsPerExpert_).
    uint64_t totalFlagInt32 = static_cast<uint64_t>(expertPerRank_) *
        (static_cast<uint64_t>(INT_CACHELINE) + static_cast<uint64_t>(dispatchFlagSlotsPerExpert_));
    uint32_t resetNumPerCore = Ops::Base::CeilDiv(totalFlagInt32, static_cast<uint64_t>(blockAivNum_));
    uint32_t resetTensorSize = Ops::Base::CeilAlign(static_cast<uint64_t>(resetNumPerCore),
        static_cast<uint64_t>(INT32_PER_256B)) * sizeof(int32_t);
    resetTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, resetTensorAddr, resetTensorSize / sizeof(int32_t));
    // [+2KB): mx 量化中间结果(maxExp + halfScale 两段, QuantProcessInRank 用)。
    uint32_t mxTempTensorAddr = resetTensorAddr + resetTensorSize;
    uint32_t mxTempTensorSize = 2 * 1024;
    mxTempTensor_ = LocalTensor<uint16_t>(TPosition::VECCALC, mxTempTensorAddr, mxTempTensorSize / sizeof(uint16_t));
    // [+2×(align256(H)+H/32)): 量化输出 [fp8 token | e8m0 scale] ping-pong;
    // 后接 [+2×align128(H)*sizeof(XType)): bf16 输入 ping-pong。
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
    // SendMaskCal 源卡算 mask 切片的 ping-pong 缓冲(配 V->MTE3 软流水)。每个 = CK/8 字节 mask + 32B,
    // 末块把累计 count 折叠进切片尾部一起 push(省掉单独 count 写), 故多留 32B。CK/8 天然 32B 对齐。
    constexpr uint32_t maskSliceBytes = static_cast<uint32_t>(DISPATCH_MASK_SLICE_BYTES);
    constexpr uint32_t maskBufBytes = maskSliceBytes + static_cast<uint32_t>(ALIGN_32);  // 切片 + 折叠 count
    uint32_t sendMaskAddr = xInAlignAddr2 + xInAlignSize2;
    for (int32_t b = 0; b < DOUBLE_BUFFER; ++b) {
        sendMaskTensor_[b] = LocalTensor<uint8_t>(TPosition::VECCALC,
            sendMaskAddr + static_cast<uint32_t>(b) * maskBufBytes, maskBufBytes);
    }
    // SendMaskCal GatherMask 计 count 用的废弃输出 scratch(CK 个 int, 256B 对齐)。
    uint32_t sendGatherOutAddr = sendMaskAddr + static_cast<uint32_t>(DOUBLE_BUFFER) * maskBufBytes;
    uint32_t sendGatherOutSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(DISPATCH_TOKEN_CHUNK * sizeof(int32_t)), static_cast<int64_t>(ALIGN_256));
    sendGatherOutTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, sendGatherOutAddr,
        sendGatherOutSize / sizeof(int32_t));
    // 每核 owned 全局专家数的上界 = ceil(ws*epr / blockAivNum_); count 跨 chunk 累加, O(ws*epr/核), 与 bs 无关。
    uint32_t sendCntAccAddr = sendGatherOutAddr + sendGatherOutSize;
    uint32_t numMyExpertsMax = Ops::Base::CeilDiv(worldSize_ * expertPerRank_, blockAivNum_);
    uint32_t sendCntAccSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(numMyExpertsMax * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    sendCntAcc_ = LocalTensor<int32_t>(TPosition::VECCALC, sendCntAccAddr, sendCntAccSize / sizeof(int32_t));
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
    //   - 目标卡 win 内布局 [localExpert][srcRank], 单槽 maskSlotSize_ 字节([mask|count]):
    //     slotBase = maskWinOffset_ + (le*worldSize_ + rankId_) * maskSlotSize_。mask 区大小不变(align32(cc/8)),
    //     第 c 块切片落在 slotBase + chunkStart(c)/8, 末块 clamp 到 maskAlignSize_。
    //   - 分块流式: 见下方 outer=chunk/inner=expert 结构, UB 占用与 bs 无关。
    //   - 因推送 rank Y 的核与 CrossRankSyncInWorldSize 中 signal rank Y 的核不再同一个,
    //     Process 在跨卡 signal 前加了 PipeBarrier<PIPE_ALL>+SyncAll 保证可见性。
    // ============================================================================
    if constexpr(g_coreType == AIC) {
        return;
    }

    // 分块流式: outer = token-chunk(topk 只载一次/块), inner = 本核 owned 全局专家。
    // 每核 owned 专家数 numMyExperts = ceil((totalExperts-aivCoreIdx_)/blockAivNum_), 与 bs 无关。
    // 每 (chunk,expert) 把一片 mask(满块 CK/8, 末块 clamp 到 maskAlignSize_)推到 win 槽内偏移 chunkStart/8;
    // count 跨 chunk 累加到 sendCntAcc_[e], 全部 chunk 后再单独写各槽的 count 字段。
    int32_t totalExperts = static_cast<int32_t>(worldSize_ * expertPerRank_);
    int32_t coreIdx = static_cast<int32_t>(aivCoreIdx_);
    int32_t numMyExperts = (coreIdx < totalExperts) ?
        Ops::Base::CeilDiv(totalExperts - coreIdx, static_cast<int32_t>(blockAivNum_)) : 0;
    if (numMyExperts <= 0) {
        return;   // 本核不负责任何全局专家(核数 > 专家数时尾部核)
    }

    GlobalTensor<int32_t> srcGlobalTensor;
    srcGlobalTensor.SetGlobalBuffer((__gm__ int32_t*)params_.expertIdxGmAddr);
    GlobalTensor<uint8_t> dstGlobalTensor;
    constexpr int32_t CK = DISPATCH_TOKEN_CHUNK;
    constexpr int32_t maskSliceBytesFull = DISPATCH_MASK_SLICE_BYTES;  // 满块 CK/8 = 256B
    DataCopyPadExtParams<int32_t> loadPad{false, 0U, 0U, 0U};

    // 每核 per-owned-expert 跨 chunk count 累加器清零(scalar 访问前需 V->S)。
    Duplicate<int32_t>(sendCntAcc_, 0, numMyExperts);
    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();

    constexpr TEventID kBufEvents[DOUBLE_BUFFER] = {EVENT_ID0, EVENT_ID1};
    for (int32_t b = 0; b < static_cast<int32_t>(DOUBLE_BUFFER); ++b) {
        SetFlag<AscendC::HardEvent::MTE3_V>(kBufEvents[b]);
    }
    int32_t iter = 0;
    for (int32_t c = 0; c < numChunks_; ++c) {
        int32_t chunkStart = c * CK;
        int32_t validLen = static_cast<int32_t>(sendTotalNum_) - chunkStart;
        if (validLen > CK) { validLen = CK; }
        // 本块写入 win 的 mask 字节数: 末块 clamp 到 maskAlignSize_(不越界 count 字段); 其余块满 CK/8。
        // 32B 对齐成立: chunkStart/8 与 maskAlignSize_ 均 32 对齐, 故 clamp 差值也 32 对齐。
        int32_t sliceBytes = maskSliceBytesFull;
        if (chunkStart / 8 + sliceBytes > static_cast<int32_t>(maskAlignSize_)) {
            sliceBytes = static_cast<int32_t>(maskAlignSize_) - chunkStart / 8;
        }
        // count 回折: 末块把累计 count 紧跟 mask 切片一起 push(切片末尾恰好是槽的 count 字段偏移
        // maskAlignSize_, 见 clamp 不变量 H), 省掉每专家单独的 count 写。非末块只 push 切片。
        bool isLastChunk = (c == numChunks_ - 1);
        int32_t pushBytes = isLastChunk ? (sliceBytes + static_cast<int32_t>(sizeof(int32_t))) : sliceBytes;
        DataCopyExtParams maskCopyParams{1U, static_cast<uint32_t>(pushBytes), 0U, 0U, 0U};
        // 载本 chunk topk: 先 Duplicate(-1) 哨兵, 再 DataCopyPad 有效段; 尾 chunk 余下保持 -1(不会误命中专家0)。
        Duplicate<int32_t>(topkChunk_, -1, CK);
        SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID1>();
        DataCopyExtParams loadParams{1U, static_cast<uint32_t>(validLen * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(topkChunk_, srcGlobalTensor[chunkStart], loadParams, loadPad);
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();

        for (int32_t e = 0; e < numMyExperts; ++e, ++iter) {
            int32_t g = coreIdx + e * static_cast<int32_t>(blockAivNum_);
            int32_t dstRank = g / static_cast<int32_t>(expertPerRank_);
            int32_t le = g % static_cast<int32_t>(expertPerRank_);
            TEventID ev = kBufEvents[iter % static_cast<int32_t>(DOUBLE_BUFFER)];
            LocalTensor<uint8_t> maskBuf = sendMaskTensor_[iter % static_cast<int32_t>(DOUBLE_BUFFER)];
            LocalTensor<uint32_t> maskBufU32 = maskBuf.template ReinterpretCast<uint32_t>();

            WaitFlag<AscendC::HardEvent::MTE3_V>(ev);  // 等本 buffer 上一轮 MTE3 推送完成
            CompareScalar(maskBuf, topkChunk_, g, AscendC::CMPMODE::EQ, CK);   // 本 chunk 的 CK bit mask 切片
            // 本 chunk 命中数: GatherMask 取 set-bit 数(mask=validLen 跳过尾部哨兵)。
            uint64_t sendCnt = 0;
            GatherMask(sendGatherOutTensor_, topkChunk_, maskBufU32, true,
                static_cast<uint32_t>(validLen), {1, 1, 0, 0}, sendCnt);
            SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();   // count 标量就绪(V 已排空, 覆盖 mask 的 V->MTE3)
            int32_t acc = sendCntAcc_.GetValue(e) + static_cast<int32_t>(sendCnt);
            sendCntAcc_.SetValue(e, acc);
            if (isLastChunk) {
                // 把最终 count 折叠进切片尾部(buffer 字节偏移 sliceBytes → win 偏移 maskAlignSize_)。
                maskBuf.template ReinterpretCast<int32_t>().SetValue(sliceBytes / sizeof(int32_t), acc);
            }
            SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>(); // mask(V)/count(S) 对 MTE3 可见

            uint64_t dstOffset = maskWinOffset_ +
                static_cast<uint64_t>(le * static_cast<int32_t>(worldSize_) + static_cast<int32_t>(rankId_)) *
                static_cast<uint64_t>(maskSlotSize_) + static_cast<uint64_t>(chunkStart / 8);
            dstGlobalTensor.SetGlobalBuffer(
                (__gm__ uint8_t*)GetRankWinAddrWithOffset(dstRank, dstOffset));
            DataCopyPad(dstGlobalTensor, maskBuf, maskCopyParams);
            SetFlag<AscendC::HardEvent::MTE3_V>(ev);
        }
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
    // 目标卡侧"读 count 计数"。各源卡为本 localExpert 算好的 [mask|count] 槽位已由 SendMaskCal
    // 推送到本卡 win(布局 [localExpert][srcRank], 槽 = align32(cc/8) mask + 32B count); 全卡同步后这里:
    //   1. 用 stride 一次性【只读 count 字段】(ws 个 int32, 跳过 mask 区)进 sendCntTensor_;
    //   2. 逐源卡累加 totalSendCntInExp_/cumsumRevCntInRank_, 写 cumsumInfoTensor_;
    //   3. 写 expertRevNumsGloablTensor_ + CrossCoreSetFlag 通知 AIC。
    // 分块流式后 mask 不再常驻: TripleInfoCalAndDispatch 自行按 (dstRank,chunk) 直接读 win mask 切片。
    // ============================================================================
    totalSendCntInExp_ = 0;

    // ----- 1. 用 stride 从本卡 win 只读本 localExpert 的 ws 个 count 字段(各槽 count 偏移 maskAlignSize_) -----
    GlobalTensor<int32_t> cntSrcGlobal;
    cntSrcGlobal.SetGlobalBuffer((__gm__ int32_t*)(params_.peermemInfo.maskRecvPtr +
        static_cast<uint64_t>(localExpertId) * worldSize_ * maskSlotSize_ + maskAlignSize_));
    DataCopyExtParams cntCopyParams{static_cast<uint16_t>(worldSize_), static_cast<uint32_t>(sizeof(int32_t)),
        static_cast<uint32_t>(maskSlotSize_ - sizeof(int32_t)), 0U, 0U};   // src 每块跳过 (slot-4) 字节
    DataCopyPadExtParams<int32_t> cntPad{false, 0U, 0U, 0U};
    DataCopyPad(sendCntTensor_, cntSrcGlobal, cntCopyParams, cntPad);
    SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>();   // count 读取(标量)就绪

    // ----- 2. 逐源卡读取 count + cumsum -----
    for (int32_t calRankId = 0; calRankId < static_cast<int32_t>(worldSize_); ++calRankId) {
        int32_t sendCnt = sendCntTensor_.GetValue(calRankId);
        totalSendCntInExp_ += static_cast<uint64_t>(sendCnt);
        cumsumRevCntInRank_ += static_cast<uint64_t>(sendCnt);
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
    // 始终只有 1 个在途,跨卡访存延迟无法靠并发隐藏。
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
    constexpr int32_t L1_TILE_M_I32 = static_cast<int32_t>(MegaMoeImpl::L1_TILE_M_256);
    constexpr int32_t CK = DISPATCH_TOKEN_CHUNK;
    // priorExpertCumsum = 前面所有 expert 在本卡的总 token 数 = 当前 expert 在 dispatchRevData
    // 里的起始 row。等价于 cumsumInfoTensor_[localExpertId * worldSize_ - 1]。
    int32_t priorExpertCumsum = (localExpertId == 0) ? 0 :
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
        for (int32_t c = 0; c < numChunks_ && matchedSoFar < coreHi; ++c) {
            int32_t chunkStart = c * CK;
            int32_t validLen = static_cast<int32_t>(sendTotalNum_ - static_cast<uint64_t>(chunkStart));
            if (validLen > CK) { validLen = CK; }
            // 末块 clamp 读取字节数到 maskAlignSize_(与源写入对称, 不读到 count 字段); 满块 CK/8。
            int32_t sliceBytes = DISPATCH_MASK_SLICE_BYTES;
            if (chunkStart / 8 + sliceBytes > static_cast<int32_t>(maskAlignSize_)) {
                sliceBytes = static_cast<int32_t>(maskAlignSize_) - chunkStart / 8;
            }
            maskSrcGlobal.SetGlobalBuffer((__gm__ uint8_t*)(params_.peermemInfo.maskRecvPtr +
                (static_cast<uint64_t>(localExpertId) * worldSize_ + dstRankIdx) * maskSlotSize_ +
                static_cast<uint64_t>(chunkStart / 8)));
            DataCopy(maskChunk_, maskSrcGlobal, static_cast<uint32_t>(sliceBytes));
            SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();   // mask 切片供 GatherMask(V)
            CreateVecIndex(topkIndexTensor_, chunkStart, CK);              // 本块全局 token 位置
            uint64_t matchedInChunk = 0;
            GatherMask(validTopkIndexTensor_, topkIndexTensor_, maskChunkU32_, true,
                static_cast<uint32_t>(validLen), {1, 1, 0, 0}, matchedInChunk);
            // V_S 排空 GatherMask: (a) matchedInChunk 标量就绪; (b) validTopkIndex 供 CopyGMToGM 标量读;
            // (c) 作 V->下一轮 DataCopy(MTE2) 的程序序屏障, 保护 maskChunk_/validTopkIndex 复用(含未派发的前导块)。
            SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID4>();
            int32_t gLo = matchedSoFar;
            int32_t gHi = matchedSoFar + static_cast<int32_t>(matchedInChunk);
            int32_t lo = gLo > coreLo ? gLo : coreLo;
            int32_t hi = gHi < coreHi ? gHi : coreHi;
            if (hi > lo) {
                int32_t subStart = lo - matchedSoFar;             // chunk-local 索引(validTopkIndex_ 内偏移)
                int32_t subNum = hi - lo;
                int32_t subDst = static_cast<int32_t>(rowStartIdxInDst) + lo;  // 绝对 dispatchRevData 行
                // triple-tile 解耦: subNum 可达 CK(调大后很大), 按 DISPATCH_TRIPLE_TILE 行分批,
                // tripleTensor_ 每批 ≤ DISPATCH_TRIPLE_TILE*32B, 与 CK 解耦, 保证 UB 不爆。
                for (int32_t off = 0; off < subNum; off += DISPATCH_TRIPLE_TILE) {
                    int32_t n = subNum - off;
                    if (n > DISPATCH_TRIPLE_TILE) { n = DISPATCH_TRIPLE_TILE; }
                    tripleTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, ubBufferUsedAddr_,
                        static_cast<uint32_t>(n) * ALIGN_32 / sizeof(int32_t));
                    CopyGMToGMPerToken(subDst + off, dstRankIdx, subStart + off, n, pingpongId);
                }
                rowsThisCore += subNum;
            }
            matchedSoFar = gHi;
        }

        // wave-grain AtomicAdd:本核派发的行在 dispatchRevData 内是连续区间
        // [rowStartLocal, rowEndLocal) = [rowDstOffsetInCore - priorExpertCumsum, +rowsThisCore)(与改前同)。
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

    // 时段 C 布局从 UB 地址 0 起, 整体覆盖时段 B(FirstBuffInit)区域(GMM2/combine 已完成, 有 SyncAll 隔离)。
    // [0, 3H*2B): bf16 区 —— [0,H) 最终输出, [H,2H)/[2H,3H) 是 topK 累加的搬入 ping-pong。
    dataResAddr_ = 0;
    dataResSize_ = dataResBufAlign / sizeof(bfloat16_t);
    dataResTensor_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, dataResAddr_, dataResSize_);
    // [+ws*align128(ws*epr)*4): win 区 token 槽位清理用零缓冲。
    resetTokenAddr_ = dataResAddr_ + dataResBufAlign;
    resetTokenSize_ = num / sizeof(int32_t);
    resetTokenTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, resetTokenAddr_, resetTokenSize_);
    // [+3H*4B): fp32 区, 与 bf16 区同切分([0,H) 累加器 + 两块 ping-pong)。
    dataResFp32Addr_ = resetTokenAddr_ + num;
    dataResFp32Size_ = dataResFp32BufAlign / sizeof(float);
    dataResFp32Tensor_ = LocalTensor<float>(TPosition::VECCALC, dataResFp32Addr_, dataResFp32Size_);
    // [+bs*topK*4B): 全量 topkWeights(fp32), Unpermute 全程驻留只读。
    topKWeightsAddr_ = dataResFp32Addr_ + dataResFp32BufAlign;
    topKWeightsSize_ = topKWeightsBufAlign / sizeof(float);
    topKWeightsTensor_ = LocalTensor<float>(TPosition::VECCALC, topKWeightsAddr_, topKWeightsSize_);
    // [+bs*topK*2B): bf16 topkWeights Cast 前临时区(仅 bf16 输入时使用)。
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
                QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
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
            MegaMoeImpl::GroupMatmul2Combine<QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                QuantScaleOutType>(
                params_, problemShape_, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_, expertBeforeCnt_, 
                gmm2PingPongIdx_);
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
