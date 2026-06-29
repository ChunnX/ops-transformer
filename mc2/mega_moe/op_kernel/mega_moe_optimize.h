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
// #include "../../moe_distribute_dispatch_v2/op_kernel/quantize_functions.h"
#include "quantize_functions.h"

using namespace AscendC;

namespace MegaMoeImpl {
using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
using BlockOffset = Shape<int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t>;

#define TemplateMegaMoeTypeClass typename XType, typename OutputType, typename TopkWeightsType, int32_t QuantMode, bool IsInterleaved
#define TemplateMegaMoeTypeFunc XType, OutputType, TopkWeightsType, QuantMode, IsInterleaved

template <typename XType, typename OutputType, typename TopkWeightsType, int32_t QuantMode, bool IsInterleaved = true>
class MegaMoe {
public:
    using QuantOutType = typename std::conditional<((QuantMode >= 3) && (QuantMode != 4)),
        fp8_e5m2_t, fp8_e4m3fn_t>::type;
    using QuantScaleOutType = typename std::conditional<(QuantMode >= 3), fp8_e8m0_t, float>::type;
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
    __aicore__ inline void ResetGmm2RowReadyFlags();
    __aicore__ inline void SendMaskCal();
    __aicore__ inline void SendInfoCal(int32_t localExpertId);
    __aicore__ inline void TripleInfoCalAndDispatch(int32_t totalTokensInExpert, int32_t localExpertId);
    __aicore__ inline bool UpdateGroupParams(uint32_t expertIdx, uint32_t gmmIndex);
    __aicore__ inline void UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo, uint32_t gmmIndex,
        uint32_t localExpertId, uint64_t waveIdx = 0);
    __aicore__ inline void Unpermute();
    __aicore__ inline void EndSync();
    __aicore__ inline void EndGMM2Sync();
    __aicore__ inline void CrossRankSyncInWorldSize();
    __aicore__ inline void ExpertTokenNumCopyOut();
    __aicore__ inline void CopyGMToGMPerToken(int32_t rowDstOffsetInCore,
        int32_t remoteRankIdx, int32_t copyStartIdx, int32_t copyNum);
    __aicore__ inline void QuantProcessInRank();

    __aicore__ inline void TilingByCore(int32_t totalLen, int32_t &coreLen,
        int32_t& coreOffset, int32_t align = ALIGN_32)
    {
        int32_t coreIdx = GetBlockIdx();
        int32_t coreNum = GetBlockNum() * 2; // 取到vec核总数
        int32_t lenPerCore = Ops::Base::CeilDiv(static_cast<uint32_t>(totalLen), static_cast<uint32_t>(coreNum));
        int32_t lenPerCoreAlign = Ops::Base::CeilAlign(static_cast<uint32_t>(lenPerCore), static_cast<uint32_t>(align));
        coreLen = lenPerCoreAlign;
        coreOffset = coreIdx * lenPerCoreAlign;
        if (coreOffset + coreLen >= totalLen) {
            coreLen = totalLen - coreOffset;
        }
        if (coreOffset >= totalLen) {
            coreLen = 0;
        }
    }

    __aicore__ inline void GmSignalWaitBarrier(__gm__ int32_t *sig_addr, int32_t cmp_val)
    {
        do {
            if (ReadGmByPassDCache(sig_addr) == cmp_val) {
                return;
            }
        } while (true);
    }

    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    ParamsOptimize params_{};
    TupleShape problemShape_{};
    BlockOffset baseOffset_{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    GlobalTensor<int32_t> cumsumInWorkSpace_;
    GlobalTensor<int32_t> groupFlagListGm_;
    GlobalTensor<int32_t> groupFlagListGm2_;
    GlobalTensor<int32_t> groupList_;
    GlobalTensor<int32_t> expertTokenNumsOut_;
    GlobalTensor<int32_t> tripleGloablTensor_;
    GlobalTensor<int32_t> expertRevNumsGloablTensor_;
    LocalTensor<float> topKWeightsTensor_;

    uint32_t m_ = 0;
    uint32_t k_ = 0;
    uint32_t aicNum_ = 0;
    uint32_t topK_ = 0;
    uint32_t rankId_ = 0;
    uint32_t worldSize_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t groupListType_ = 0;
    uint64_t maxOutputSize_ = 0;
    uint64_t maxWavesPerExpert_ = 0;
    // wave 并行分组信息（每张 expert 复用，由 Process 计算后填入）
    uint32_t coresPerWave_ = 0;
    uint32_t wavesParallel_ = 1;
    uint32_t coreInGroup_ = 0;
    uint32_t coreGroupId_ = 0;
    bool coreActive_ = true;
    int32_t prevSum_ = 0;
    int32_t dispatchRows_ = 0;
    int32_t vecSetSyncCom_ = 0;
    uint32_t startBlockIdx_ = 0;
    uint32_t blockNumPerRank_ = 2;
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
    uint32_t resetTokenAddr_ = 0;
    uint32_t dataResFp32Addr_ = 0;
    uint32_t topKWeightsAddr_ = 0;
    uint32_t crossRankSyncAddr_ = 0;
    uint32_t dataResSize_ = 0;
    uint32_t resetTokenSize_ = 0;
    uint32_t dataResFp32Size_ = 0;
    uint32_t topKWeightsSize_ = 0;
    uint32_t crossRankSyncSize_ = 0;
    uint32_t ubBufferUsedAddr_ = 0;

    uint64_t preOffset_ = 0;
    // GMM2 ping-pong：跨 wave/expert 持久的 tile 计数。pp = gmm2SyncCom_&1 选 buf0/buf1，
    // 滞后一拍复用机制跨 GMM2 调用无缝；最后两个在飞 buffer 在 expert loop 后由 EndGMM2Sync drain。
    int32_t gmm2SyncCom_ = 0;
    // GMM1 pingpong：跨 wave 持久化的 ping/pong 状态。
    // tileNum 通常 = blockNum/wavesParallel = coresPerWave，意味着每核每 wave 只 1 个 tile，
    // 单次 GroupMatmulSwigluQuant 调用内 pingpong 无法翻转。把它提到类成员后，
    // 同核 batch 内连续两个 wave (wave0/wave2 或 wave1/wave3) 分别落在 UbFirst/UbSecond，
    // 实现 GMM1(N) mmad 与 swiglu(N-1) 在 ping/pong 半区真正并行。
    uint16_t gmm1PingPongIdx_ = 0;
    uint64_t sendTotalNum_ = 0;
    uint32_t maskAlignSize_ = 0;
    uint32_t maskSlotSize_ = 0;    // 单个 win 槽位 = maskAlignSize_(mask) + 32B(count)
    uint64_t maskWinOffset_ = 0;   // maskRecvPtr 相对 win 基址(rankSyncInWorldPtr)的偏移
    uint64_t quantWinOffset_ = 0;  // quantTokenScalePtr 相对 win 基址的偏移
    uint64_t totalSendCntInExp_ = 0;
    uint64_t cumsumRevCntInRank_ = 0;
    int32_t compareCount_ = 0;

    // 6-buffer 软流水：占满 EVENT_ID0..EVENT_ID5。
    //   - TripleInfoCalAndDispatch 用 copyTmpTensors_[6] 让 winmem token+scale 读
    //     在引擎里同时排队 5+ 个(配合 prime/steady/drain 软流水)。
    static constexpr int32_t DISPATCH_BUFFER_NUM = 6;
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
    LocalTensor<uint8_t> sendMaskTensor_[2];  // SendMaskCal 源卡算 [mask|count] 的 ping-pong 缓冲
    LocalTensor<int32_t> sendGatherOutTensor_;  // SendMaskCal GatherMask 计 count 的废弃输出 scratch

    using BlockEpilogue = BlockEpilogueSwigluMxQuant<
        QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, true, IsInterleaved>;
    BlockEpilogue epilogueOp_;

    // 本核连续累计的 wave iteration 序号（trace 用，trace 关时无副作用）
    int waveTraceIdx_ = 0;

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
    maxWavesPerExpert_ = (maxOutputSize_ + MegaMoeImpl::L1_TILE_M - 1) / MegaMoeImpl::L1_TILE_M;
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
    params_.workspaceInfo = WorkspaceInfoOptimize(workspaceGM, tilingData);
    params_.peermemInfo = PeermemInfoOptimize(winRankAddr_[rankId_], tilingData);
    params_.tilingData = tilingData;
    expertTokenNumsOut_.SetGlobalBuffer((__gm__ int32_t*)params_.expertTokenNumsOutGmAddr);
    expertRevNumsGloablTensor_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.expertRevTokenNumsPtr);
    tripleGloablTensor_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.tripleInfoPtr);
    epilogueOp_.Init({params_.workspaceInfo.swigluQuantDataPtr, params_.workspaceInfo.swigluQuantScalePtr,
        params_.workspaceInfo.flagSwiGluToGmm2Ptr, nullptr, nullptr, nullptr, ALIGN_256, ALIGN_256});
    // 各 win 区相对 win 基址(rankSyncInWorldPtr)的偏移; 所有卡 win 布局一致, 跨卡读写用同一偏移。
    maskWinOffset_ = static_cast<uint64_t>(params_.peermemInfo.maskRecvPtr - params_.peermemInfo.rankSyncInWorldPtr);
    quantWinOffset_ = static_cast<uint64_t>(params_.peermemInfo.quantTokenScalePtr - params_.peermemInfo.rankSyncInWorldPtr);
    // compareCount_/maskAlignSize_ 仅依赖 sendTotalNum_, 在 Init 算好以便 SendMaskCal(同步前) 使用。
    // maskAlignSize_ 必须与 PeermemInfoOptimize 中 maskAlignSize 公式数值一致。
    compareCount_ = Ops::Base::CeilAlign((int64_t)(sendTotalNum_ * sizeof(int32_t)),
        (int64_t)(ALIGN_256)) / sizeof(int32_t);
    maskAlignSize_ = Ops::Base::CeilAlign((int64_t)(compareCount_) / 8, (int64_t)(ALIGN_32));
    // 每个 win 槽位再追加 32B 存 count(源卡 SendMaskCal 同步算好), 须与 PeermemInfoOptimize 的 maskSlotSize 一致。
    maskSlotSize_ = maskAlignSize_ + static_cast<uint32_t>(ALIGN_32);
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
        (int64_t)(worldSize_ * expertPerRank_ * sizeof(int32_t)), (int64_t)ALIGN_32);
    cumsumInfoTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, cumsumInfoTensorAddr,
        cumsumInfoTensorSize / sizeof(int32_t));
    uint32_t topkIndexTensorAddr = cumsumInfoTensorAddr + cumsumInfoTensorSize;
    uint32_t topkIndexTensorSize = Ops::Base::CeilAlign(
        (int64_t)(sendTotalNum_ * sizeof(int32_t)), (int64_t)(ALIGN_32));
    topkIndexTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, topkIndexTensorAddr,
        topkIndexTensorSize / sizeof(int32_t));
    uint32_t gatherMaskTensorAddr = topkIndexTensorAddr + topkIndexTensorSize;
    uint32_t gatherMaskTensorSize = maskSlotSize_ * worldSize_;  // 每槽 [mask|count]
    gatherMaskTensor_ = LocalTensor<uint8_t>(TPosition::VECCALC, gatherMaskTensorAddr,
        gatherMaskTensorSize / sizeof(uint8_t));
    gatherMaskInt32Tensor_ = LocalTensor<uint32_t>(TPosition::VECCALC, gatherMaskTensorAddr,
        gatherMaskTensorSize / sizeof(uint32_t));
    // topkTensors_[6] from the old cross-rank topk read path was removed here.
    // Current dispatch uses the source-rank precomputed mask in gatherMaskTensor_ and
    // materializes selected indices directly into validTopkIndexTensor_.
    uint32_t validTopkIndexTensorAddr = gatherMaskTensorAddr + gatherMaskTensorSize;
    uint32_t validTopkIndexTensorSize = Ops::Base::CeilAlign(
        (int64_t)(sendTotalNum_ * sizeof(int32_t)), (int64_t)(ALIGN_32));
    validTopkIndexTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, validTopkIndexTensorAddr,
        validTopkIndexTensorSize / sizeof(int32_t));
    // 6 个 dispatch buffer，配合 EVENT_ID0..EVENT_ID5 做软流水。
    // 每槽 8KB，足以容纳一个 token+scale (mxFp8TokenAlignBytes_ + mxFp8ScaleAlignBytes_ 通常 < 7.5KB)。
    // 6 × 8KB = 48KB；tripleTensor_ follows these buffers at ubBufferUsedAddr_.
    // GMM2 AIV1 uses fixed ping-pong buffers [64K,128K) and [128K,192K);
    // the current dispatch/GMM2 ordering is still protected by the dispatch->GMM2 flag.
    constexpr uint32_t COPY_TMP_BUFFER_SIZE = 8 * 1024;
    uint32_t copyTmpBaseAddr = validTopkIndexTensorAddr + validTopkIndexTensorSize;
    for (int32_t b = 0; b < DISPATCH_BUFFER_NUM; ++b) {
        copyTmpTensors_[b] = LocalTensor<QuantOutType>(TPosition::VECCALC,
            copyTmpBaseAddr + static_cast<uint32_t>(b) * COPY_TMP_BUFFER_SIZE,
            COPY_TMP_BUFFER_SIZE / sizeof(QuantOutType));
    }
    ubBufferUsedAddr_ = copyTmpBaseAddr +
        static_cast<uint32_t>(DISPATCH_BUFFER_NUM) * COPY_TMP_BUFFER_SIZE;
    CreateVecIndex(topkIndexTensor_, 0, (topkIndexTensorSize / sizeof(int32_t)));
    Duplicate<int32_t>(cumsumInfoTensor_, 0, (cumsumInfoTensorSize / sizeof(int32_t)));
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
    uint32_t topkIdsTensorSize = Ops::Base::CeilAlign((int64_t)(sendTotalNum_ * sizeof(int32_t)), (int64_t)(ALIGN_256));
    topkIdsTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, topkIdsTensorAddr, topkIdsTensorSize / sizeof(int32_t));
    // resetTensor_ 用作两个 zero-fill 场景的 UB 缓存：
    //   (1) ResetFlagList    : flag block 总元素数 = expertPerRank * maxWavesPerExpert * 2
    //                          (swiglu->gmm2 per-wave + dispatch->gmm1 per-wave) 对齐到 8
    //                          按 ALIGN_32 分核（每核 ≤ ALIGN_32 个 int32）
    //   (2) ResetGmm2RowReadyFlags: 单 AIV0 写 align(aicNum, 8) 个 int32
    // 大小取两者上限的较大者，并按 8 个 int32 对齐
    uint32_t resetTensorAddr = topkIdsTensorAddr + topkIdsTensorSize;
    int32_t flagListTotal = static_cast<int32_t>(
        Ops::Base::CeilAlign(static_cast<uint32_t>(expertPerRank_ * maxWavesPerExpert_ * 2U),
                             static_cast<uint32_t>(8)));
    int32_t flagPerCoreMax = static_cast<int32_t>(Ops::Base::CeilAlign(
        Ops::Base::CeilDiv(static_cast<uint32_t>(flagListTotal), static_cast<uint32_t>(blockAivNum_)),
        static_cast<uint32_t>(ALIGN_32)));
    int32_t rowReadyMax = static_cast<int32_t>(
        Ops::Base::CeilAlign(static_cast<uint32_t>(blockNum_), static_cast<uint32_t>(8)));
    int32_t resetMax = flagPerCoreMax > rowReadyMax ? flagPerCoreMax : rowReadyMax;
    uint32_t resetTensorSize = Ops::Base::CeilAlign((uint64_t)resetMax, (uint64_t)8) * sizeof(int32_t);
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
    // SendMaskCal 源卡算 [mask|count] 的 ping-pong 缓冲(每个 maskSlotSize_ 字节, 配 V->MTE3 软流水)。
    // 与 topkIdsTensor_ / 量化缓冲不重叠, 同处 BuffInit 阶段共存; combine 阶段(SecondBuffInit)这块地址不再使用。
    uint32_t sendMaskAddr = xInAlignAddr2 + xInAlignSize2;
    for (int32_t b = 0; b < 2; ++b) {
        sendMaskTensor_[b] = LocalTensor<uint8_t>(TPosition::VECCALC,
            sendMaskAddr + static_cast<uint32_t>(b) * maskSlotSize_, maskSlotSize_);
    }
    // SendMaskCal GatherMask 计 count 用的废弃输出 scratch(compareCount_ 个 int, 256B 对齐)。
    uint32_t sendGatherOutAddr = sendMaskAddr + 2U * maskSlotSize_;
    uint32_t sendGatherOutSize = Ops::Base::CeilAlign((int64_t)(compareCount_ * sizeof(int32_t)),
        (int64_t)(ALIGN_256));
    sendGatherOutTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, sendGatherOutAddr,
        sendGatherOutSize / sizeof(int32_t));
    Duplicate<int32_t>(resetTensor_, 0, (resetTensorSize / sizeof(int32_t)));
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ResetFlagList()
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    // workSpace groupFlagList 清零（覆盖 swiglu->gmm2 段 + dispatch->gmm1 段，共一整块）：
    //   总元素数 = expertPerRank * maxWavesPerExpert (swiglu, per-wave) + expertPerRank * maxWavesPerExpert (dispatch, per-wave)
    //   workspace 里按 CeilAlign(., 8) 分配，这里 flagNum 也按 8 对齐保证整段覆盖
    //   分核走 ALIGN_32 的粒度（默认）：每核一次写一个 32 个 int32 的块，DataCopy 直发，避免 DataCopyPad 的非对齐小块路径
    groupFlagListGm_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.flagSwiGluToGmm2Ptr);
    int32_t swigluElems = static_cast<int32_t>(expertPerRank_ * maxWavesPerExpert_);
    int32_t dispatchElems = static_cast<int32_t>(expertPerRank_ * maxWavesPerExpert_);
    int32_t flagNum = Ops::Base::CeilAlign(
        static_cast<uint32_t>(swigluElems + dispatchElems), static_cast<uint32_t>(8));
    int32_t coreLen, coreOffset;
    TilingByCore(flagNum, coreLen, coreOffset);  // 默认 align=ALIGN_32
    if (coreLen != 0) {
        Duplicate<int32_t>(resetTensor_, 0, coreLen);
        SyncFuncStatic<AscendC::HardEvent::V_MTE3, 0>();
        DataCopy(groupFlagListGm_[coreOffset], resetTensor_, coreLen);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ResetGmm2RowReadyFlags()
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    if (aivCoreIdx_ != 0) {
        return;
    }
    // ptrGmm2RowReady 分配大小 = align(aicNum, ALIGN_128) 个 int32，足够覆盖 align(blockNum_, 8)。
    // 单 AIV0 写 align(blockNum_, 8) 个 int32（按 8 对齐够 DataCopy）。
    GlobalTensor<int32_t> rowReadyFlags;
    rowReadyFlags.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrGmm2RowReady);
    int32_t blockNumAlign = Ops::Base::CeilAlign(static_cast<uint32_t>(blockNum_), static_cast<uint32_t>(8));
    Duplicate<int32_t>(resetTensor_, 0, blockNumAlign);
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, 1>();
    DataCopy(rowReadyFlags, resetTensor_, blockNumAlign);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ExpertTokenNumCopyOut()
{
    int32_t copyNum = Ops::Base::CeilAlign(static_cast<uint32_t>(expertPerRank_), static_cast<uint32_t>(ALIGN_32));
    LocalTensor<int32_t> tmpCopyBuffer(TPosition::VECCALC, crossRankSyncAddr_, crossRankSyncSize_);

    DataCopyExtParams copyParams{static_cast<uint16_t>(expertPerRank_), static_cast<uint32_t>(sizeof(int32_t)), 
                                 static_cast<uint32_t>((8 * aicNum_ - 1) * sizeof(int32_t)), 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(tmpCopyBuffer, expertRevNumsGloablTensor_, copyParams, padParams);
    SyncFuncStatic<AscendC::HardEvent::MTE2_MTE3, 0>();
    DataCopy(expertTokenNumsOut_, tmpCopyBuffer, copyNum);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::EndSync()
{
    if (vecSetSyncCom_ == 0) {
        return;
    }
    if constexpr(g_coreType == AIC) {
        // vecSetSyncCom_ == 1：单 pending（非 pingpong 始终 1，或 pingpong 仅 1 个 iter 全程），落在 ch0。
        // vecSetSyncCom_ >= 2：必为 pingpong 模式且已进入稳态，ch0/ch1 各 1 个 pending，drain 双通道。
        // 顺序：先 drain 当前 gmm1PingPongIdx_ 指向的通道（最近被 flip 到的、即下一个本该用的那个，
        // 其上一个未消费的 NotifyCube 在等），再 drain 另一个。两个 ch 必然各有 pending，不会卡死。
        if (vecSetSyncCom_ == 1) {
            WaitForVector();
        } else {
            WaitForVector(gmm1PingPongIdx_);
            WaitForVector(1 - gmm1PingPongIdx_);
        }
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::EndGMM2Sync()
{
    if constexpr(g_coreType == AIV) {
        return;
    }
    // 所有 GMM2 调用完成后，drain 最后两个仍在飞的 ping-pong buffer（各一次）。
    // gmm2SyncCom_ 是全局 tile 计数；倒数第二、倒数第一两个 tile 的 pp 必一奇一偶，覆盖两条通道。
    // 通道与 GMM2 内部一致：AIC WaitForVector(16+pp)，对应 AIV1 NotifyCube(pp)（flag 22/23）。
    int32_t total = gmm2SyncCom_;
    if (total >= 2) {
        WaitForVector(16 + ((total - 2) & 1));
    }
    if (total >= 1) {
        WaitForVector(16 + ((total - 1) & 1));
    }
    gmm2SyncCom_ = 0;
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
    //     offset = maskWinOffset_ + (le*worldSize_ + rankId_) * maskSlotSize_。
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
    SyncFuncStatic<AscendC::HardEvent::V_MTE2, 1>();
    DataCopyExtParams loadParams{1U, static_cast<uint32_t>(sendTotalNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> loadPad{false, 0U, 0U, 0U};
    DataCopyPad(topkIdsTensor_, srcGlobalTensor, loadParams, loadPad);
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, 1>();

    // ----- 2. 扁平分核, 逐个全局专家算 [mask|count] 推送 (ping-pong 双 buffer 软流水) -----
    // 每个槽位 maskSlotSize_ 字节: 前 maskAlignSize_ 是 bit-packed mask, 末 32B 存 count。
    // count 在源卡用 GatherMask(mask=sendTotalNum_, 忽略尾部 padding)同步算出, 接收端直接读取。
    constexpr TEventID kBufEvents[2] = {EVENT_ID0, EVENT_ID1};
    for (int32_t b = 0; b < 2; ++b) {
        SetFlag<AscendC::HardEvent::MTE3_V>(kBufEvents[b]);
    }
    int32_t totalExperts = static_cast<int32_t>(worldSize_ * expertPerRank_);
    uint32_t countWordIdx = static_cast<uint32_t>(maskAlignSize_) / sizeof(int32_t);  // count 在槽内 int32 偏移
    DataCopyExtParams maskCopyParams{1U, static_cast<uint32_t>(maskSlotSize_), 0U, 0U, 0U};
    int32_t iter = 0;
    GlobalTensor<uint8_t> dstGlobalTensor;
    for (int32_t g = aivCoreIdx_; g < totalExperts; g += blockAivNum_, ++iter) {
        int32_t dstRank = g / static_cast<int32_t>(expertPerRank_);
        int32_t le = g % static_cast<int32_t>(expertPerRank_);
        TEventID ev = kBufEvents[iter % 2];
        LocalTensor<uint8_t> maskBuf = sendMaskTensor_[iter % 2];
        LocalTensor<uint32_t> maskBufU32 = maskBuf.template ReinterpretCast<uint32_t>();
        LocalTensor<int32_t> maskBufI32 = maskBuf.template ReinterpretCast<int32_t>();

        WaitFlag<AscendC::HardEvent::MTE3_V>(ev);  // 等本 buffer 上一轮 MTE3 推送完成
        CompareScalar(maskBuf, topkIdsTensor_, g, AscendC::CMPMODE::EQ, compareCount_);
        // 同步算 count: GatherMask 对 mask 取 set-bit 数(mask=sendTotalNum_ 跳过尾部 padding)。
        uint64_t sendCnt = 0;
        GatherMask(sendGatherOutTensor_, topkIdsTensor_, maskBufU32, true,
            static_cast<uint32_t>(sendTotalNum_), {1, 1, 0, 0}, sendCnt);
        SyncFuncStatic<AscendC::HardEvent::V_S, 2>();   // count 标量就绪
        maskBufI32.SetValue(countWordIdx, static_cast<int32_t>(sendCnt));  // 写入槽内 count
        SyncFuncStatic<AscendC::HardEvent::S_MTE3, 3>(); // count 对 MTE3 可见(V 已被 V_S 排空)

        uint64_t dstOffset = maskWinOffset_ +
            static_cast<uint64_t>(le * static_cast<int32_t>(worldSize_) + static_cast<int32_t>(rankId_)) *
            static_cast<uint64_t>(maskSlotSize_);
        dstGlobalTensor.SetGlobalBuffer(
            (__gm__ uint8_t*)GetRankWinAddrWithOffset(dstRank, dstOffset));
        DataCopyPad(dstGlobalTensor, maskBuf, maskCopyParams);
        SetFlag<AscendC::HardEvent::MTE3_V>(ev);
    }
    for (int32_t b = 0; b < 2; ++b) {
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
        auto event = (index % 2 == 0) ? EVENT_ID0 : EVENT_ID1;
        auto xInTensor = (index % 2 == 0) ? xInTensor1_ : xInTensor2_;
        auto xOutTensor = (index % 2 == 0) ? xOutTensor1_ : xOutTensor2_;
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
    // 目标卡侧"读 count 计数"。各源卡为本 localExpert 算好的 [mask|count] 槽位已由
    // SendMaskCal 推送到本卡 win(布局 [localExpert][srcRank]); 全卡同步后这里:
    //   1. 一次性从本卡 win 把本 localExpert 的 worldSize_ 个 [mask|count] 槽位读进 gatherMaskTensor_;
    //   2. 逐源卡【直接读取槽内 count】→ 累加 totalSendCntInExp_/cumsumRevCntInRank_, 写 cumsumInfoTensor_;
    //   3. 写 expertRevNumsGloablTensor_ + CrossCoreSetFlag 通知 AIC。
    // 不再 GatherMask 计数(count 已在源卡 SendMaskCal 同步算好), 消除与 TripleInfoCalAndDispatch 的重复计算。
    // mask 部分在本函数填好后, 同一专家迭代里被 TripleInfoCalAndDispatch 的 GatherMask 复用取 token 索引。
    // ============================================================================
    totalSendCntInExp_ = 0;
    uint32_t slotWordStride = static_cast<uint32_t>(maskSlotSize_) / sizeof(uint32_t);   // 槽内 int32 步长
    uint32_t countWordIdx = static_cast<uint32_t>(maskAlignSize_) / sizeof(uint32_t);    // count 在槽内 int32 偏移

    // ----- 1. 从本卡 win 读本 localExpert 的全部 worldSize_ 个 [mask|count] 槽位 -----
    GlobalTensor<uint8_t> maskSrcGlobal;
    maskSrcGlobal.SetGlobalBuffer((__gm__ uint8_t*)(params_.peermemInfo.maskRecvPtr +
        static_cast<uint64_t>(localExpertId) * worldSize_ * maskSlotSize_));
    DataCopy(gatherMaskTensor_, maskSrcGlobal, worldSize_ * maskSlotSize_);
    SyncFuncStatic<AscendC::HardEvent::MTE2_S, 2>();   // count 读取(标量)就绪
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, 1>();   // mask 供 Triple 的 GatherMask(V)就绪

    // ----- 2. 逐源卡直接读取槽内 count + cumsum -----
    for (int32_t calRankId = 0; calRankId < static_cast<int32_t>(worldSize_); ++calRankId) {
        int32_t sendCnt = gatherMaskInt32Tensor_.GetValue(calRankId * slotWordStride + countWordIdx);
        totalSendCntInExp_ += static_cast<uint64_t>(sendCnt);
        cumsumRevCntInRank_ += static_cast<uint64_t>(sendCnt);
        cumsumInfoTensor_.SetValue(localExpertId * worldSize_ + calRankId, (int32_t)cumsumRevCntInRank_);
    }

    expertTokenCntTensor_.SetValue(0, totalSendCntInExp_);
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, 0>();
    DataCopy<int32_t>(expertRevNumsGloablTensor_[localExpertId * 8 * aicNum_ + 8 * blockIdx_],
        expertTokenCntTensor_, 8);
    PipeBarrier<PIPE_ALL>();
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_MTE3>(2);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::CopyGMToGMPerToken(
    int32_t rowDstOffsetInCore, int32_t remoteRankIdx, int32_t copyStartIdx, int32_t copyNum)
{
    // 6-buffer 软流水：搬运对端 rank 的 token+scale 至本专家卡。整段 token 同属一个 remoteRankIdx
    //（rank-grouped 分核：一个源 rank 归一组核处理），故 remoteRankGlobalTensor 只需设一次。
    //   Phase 1: 组装三元组 (rank, tokenIndex, topkInRank) → tripleTensor_
    //   Phase 2 prime : 连发 BufferNum 个 MTE2，让跨卡 winmem 读在 MTE2 引擎里同时排队
    //   Phase 2 steady: 每轮 MTE3[i] + issueMTE2[i+BufferNum]，槽位循环复用
    //   Phase 2 drain : 收尾只等 MTE3
    //   Phase 3: 三元组按全局偏移一次性搬出
    if (copyNum <= 0) {
        return;
    }
    constexpr int32_t BufferNum = DISPATCH_BUFFER_NUM;
    constexpr TEventID kBufEvents[BufferNum] = {
        EVENT_ID0, EVENT_ID1, EVENT_ID2, EVENT_ID3, EVENT_ID4, EVENT_ID5
    };
    int64_t widthA = Get<K_VALUE>(problemShape_); // 输出单token长度，紧密排列
    int64_t widthAScale = Ops::Base::CeilDiv((int64_t)widthA, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE;  // 输出token-scale长度，紧密排列
    uint32_t copyInNum = Ops::Base::CeilAlign((int64_t)(mxFp8TokenAlignBytes_ + mxFp8ScaleAlignBytes_),
        (int64_t)ALIGN_32); // 输入token-scale拼接，非紧密排列
    GlobalTensor<QuantOutType> remoteRankGlobalTensor;
    GlobalTensor<QuantOutType> tokenRevGlobalTensor;
    GlobalTensor<QuantScaleOutType> scaleRevGlobalTensor;
    tokenRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(
        params_.workspaceInfo.dispatchRevDataPtr + rowDstOffsetInCore * widthA)); // rowDstOffsetInCore 是当前专家全部的累加偏移
    scaleRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleOutType *>(
        params_.workspaceInfo.dispatchRevScalePtr + rowDstOffsetInCore * widthAScale));
    remoteRankGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(
        GetRankWinAddrWithOffset(remoteRankIdx,
        quantWinOffset_)));

    // 预置 6 个 MTE3_MTE2 flag，prime/steady 的 WaitFlag 可立即通过。
    for (int32_t b = 0; b < BufferNum; ++b) {
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(kBufEvents[b]);
    }
    PipeBarrier<PIPE_ALL>();

    // Phase 1: 三元组组装
    for (int32_t i = 0; i < copyNum; ++i) {
        int32_t topkIndex = validTopkIndexTensor_.GetValue(copyStartIdx + i);
        int32_t tokenIndex = topkIndex / topK_;
        tripleTensor_[i * 8].SetValue(0, remoteRankIdx);
        tripleTensor_[i * 8].SetValue(1, tokenIndex);
        tripleTensor_[i * 8].SetValue(2, topkIndex % topK_);
    }

    // Phase 2 prime
    int32_t primeCount = (copyNum < BufferNum) ? copyNum : BufferNum;
    for (int32_t p = 0; p < primeCount; ++p) {
        int32_t tokenIndex = tripleTensor_[p * 8].GetValue(1);
        uint64_t remoteCopyOffset = static_cast<uint64_t>(tokenIndex) * static_cast<uint64_t>(copyInNum);
        TEventID ev = kBufEvents[p];
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ev);
        DataCopy(copyTmpTensors_[p], remoteRankGlobalTensor[remoteCopyOffset], copyInNum);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(ev);
    }

    // Phase 2 steady: MTE3[i] + issueMTE2[i + BufferNum]
    for (int32_t i = 0; i < copyNum; ++i) {
        int32_t buf = i % BufferNum;
        TEventID ev = kBufEvents[buf];
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(ev);

        LocalTensor<QuantOutType> tokenScalebuf = copyTmpTensors_[buf];
        LocalTensor<QuantScaleOutType> bufScale = tokenScalebuf[mxFp8TokenAlignBytes_].template
            ReinterpretCast<QuantScaleOutType>();
        DataCopyPad(tokenRevGlobalTensor[i * widthA], tokenScalebuf,
            {1, static_cast<uint16_t>(widthA * sizeof(QuantOutType)), 0U, 0U, 0U});
        DataCopyPad(scaleRevGlobalTensor[i * widthAScale], bufScale,
            {1, static_cast<uint16_t>(widthAScale * sizeof(QuantScaleOutType)), 0U, 0U, 0U});
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(ev);

        int32_t nextIdx = i + BufferNum;
        if (nextIdx < copyNum) {
            int32_t tokenIndex = tripleTensor_[nextIdx * 8].GetValue(1);
            uint64_t remoteCopyOffset = static_cast<uint64_t>(tokenIndex) * static_cast<uint64_t>(copyInNum);
            // 等本槽这次 MTE3 完成、buffer 释放后再发下一个 MTE2。
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ev);
            DataCopy(copyTmpTensors_[buf], remoteRankGlobalTensor[remoteCopyOffset], copyInNum);
            SetFlag<AscendC::HardEvent::MTE2_MTE3>(ev);
        }
    }

    // Phase 2 drain: 6 个 buffer-free flag
    for (int32_t b = 0; b < BufferNum; ++b) {
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kBufEvents[b]);
    }

    // Phase 3: 三元组搬出（rowDstOffsetInCore 为跨 expert 全局偏移）
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, 1>();
    DataCopy(tripleGloablTensor_[rowDstOffsetInCore * 8], tripleTensor_, copyNum * 8);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::TripleInfoCalAndDispatch(
    int32_t totalTokensInExpert, int32_t localExpertId)
{
    // ============================================================================
    // rank-grouped 分核版 dispatch（回退 stripe 负载均衡）
    // ----------------------------------------------------------------------------
    // 按 blockNumPerRank_ 对 AIV 分组：一个源 rank 归一组核处理，组内按该 rank 要发的
    // token 数连续切分给各核。每个 (rank, 组内核) 的连续 token 段交给 CopyGMToGMPerToken
    // 做 6-buffer 软流水搬运；段内同属一个源 rank，故无需逐 token 切 rank、也无需二分查找。
    //
    // 注意：totalTokensInExpert 在 rank-grouped 下不再需要（每 rank 的发送数由 GatherMask
    // 现算）；保留入参与调用点签名不变，此处显式忽略。
    // ============================================================================
    (void)totalTokensInExpert;
    constexpr int32_t L1_TILE_M_I32 = static_cast<int32_t>(MegaMoeImpl::L1_TILE_M);
    int32_t priorExpertCumsum = (localExpertId == 0) ? 0 :
        cumsumInfoTensor_.GetValue(localExpertId * worldSize_ - 1);
    int32_t maxOut = static_cast<int32_t>(maxOutputSize_);

    for (uint32_t idx = blockIdx_; idx < worldSize_ * blockNumPerRank_; idx += blockNum_) {
        uint32_t dstRankIdx = idx / blockNumPerRank_;
        uint64_t sendToCurExpTokenCnt = 0;
        GatherMask(validTopkIndexTensor_, topkIndexTensor_,
            gatherMaskInt32Tensor_[dstRankIdx * (maskSlotSize_ / sizeof(uint32_t))],
            true, sendTotalNum_, {1, 1, 0, 0}, sendToCurExpTokenCnt);
        uint32_t aivIdxInGroup = idx % blockNumPerRank_;
        int32_t rowStartIdxInDst = ((dstRankIdx == 0 && localExpertId == 0) ? 0 :
            cumsumInfoTensor_.GetValue(localExpertId * worldSize_ + dstRankIdx - 1));

        int32_t rowsThisCore = 0;
        int32_t rowDstOffsetInCore = 0;
        if (rowStartIdxInDst < maxOut) {
            int32_t rowsCopyCnt = static_cast<int32_t>(sendToCurExpTokenCnt);
            if (rowStartIdxInDst + rowsCopyCnt > maxOut) {
                rowsCopyCnt = maxOut - rowStartIdxInDst;
            }
            int32_t rowsNumInCore = Ops::Base::CeilDiv(rowsCopyCnt, static_cast<int32_t>(blockNumPerRank_));
            int32_t rowSrcIdxInCore = static_cast<int32_t>(aivIdxInGroup) * rowsNumInCore;
            rowDstOffsetInCore = rowStartIdxInDst + rowSrcIdxInCore;
            if (rowDstOffsetInCore + rowsNumInCore > rowStartIdxInDst + rowsCopyCnt) {
                rowsNumInCore = rowStartIdxInDst + rowsCopyCnt - rowDstOffsetInCore;
            }
            if (rowsNumInCore > 0) {
                uint32_t tripleTensorAddr = ubBufferUsedAddr_;
                uint32_t tripleTensorSize = static_cast<uint32_t>(rowsNumInCore) * ALIGN_32;
                tripleTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, tripleTensorAddr,
                    tripleTensorSize / sizeof(int32_t));
                SyncFuncStatic<AscendC::HardEvent::V_S, 0>();
                CopyGMToGMPerToken(rowDstOffsetInCore, static_cast<int32_t>(dstRankIdx),
                    rowSrcIdxInCore, rowsNumInCore);
                rowsThisCore = rowsNumInCore;
            }
        }

        // wave 粒度 dispatch->gmm1 flag：对本核处理的 [rowStartLocal, rowEndLocal) 行，
        // 按 wave 边界分别 AtomicAdd。groupFlagListGm2_ 已由调用点指向本 expert 的 wave-0 槽。
        if (rowsThisCore > 0) {
            SyncFuncStatic<AscendC::HardEvent::MTE3_S, 0>();
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

    // 通知 AIC：dispatch 已完成，AIV1 的 UB 已释放，GMM2 可以开始 fixpipe 写 UB
    PipeBarrier<PIPE_ALL>();
    // dispatch(e+1) 用 AIV→AIC 通道通知 cube：本轮 prefetch 已写完 UB[128K+]，
    // 可以开始 GMM2(e) fixpipe 写 AIV1 UB。flag 号必须避开 GMM1/GMM2 的
    // NotifyVector/Cube(base 4/6 → 占 4,5,6,7,20,22)与 dispatch→gmm1(2,3,18)，
    // 这里取独占的 8（AIC 侧 wait 8+16=24）。
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_MTE3>(8);
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
        // 每个 expert 占 maxWavesPerExpert 个 flag 槽（每 wave 1 个 swiglu->gmm2 flag）
        Get<IDX_FLAG_OFFSET>(baseOffset_) += maxWavesPerExpert_;
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
    uint32_t localExpertId, uint64_t waveIdx)
{
    uint64_t k = Get<K_VALUE>(problemShape_);
    uint64_t n = Get<N_VALUE>(problemShape_);
    uint64_t outputN = n / SWIGLU_N_HALF;
    auto scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    auto scaleOutputN =
        Ops::Base::CeilDiv(outputN, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    int64_t waveRowOffset = static_cast<int64_t>(waveIdx * MegaMoeImpl::L1_TILE_M);

    if (gmmIndex == 0) {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.dispatchRevDataPtr +
            (Get<IDX_A_OFFSET>(baseOffset_) + waveRowOffset * static_cast<int64_t>(k)) * sizeof(QuantOutType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.dispatchRevScalePtr +
            (Get<IDX_A_SCALE_OFFSET>(baseOffset_) + waveRowOffset * static_cast<int64_t>(scaleK)) *
                sizeof(QuantScaleOutType);
        if constexpr(g_coreType == AIC) {
            gmmAddrInfo.bGlobal = params_.bGmAddr + Get<IDX_B_OFFSET>(baseOffset_) * sizeof(QuantOutType);
            gmmAddrInfo.bScaleGlobal = params_.bScaleGmAddr + Get<IDX_B_SCALE_OFFSET>(baseOffset_) *
                                        sizeof(QuantScaleOutType);
        } else {
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                Get<IDX_C_OFFSET>(baseOffset_) + waveRowOffset * static_cast<int64_t>(outputN),
                Get<IDX_C_SCALE_OFFSET>(baseOffset_) + waveRowOffset * static_cast<int64_t>(scaleOutputN),
                Get<IDX_FLAG_OFFSET>(baseOffset_) + static_cast<int64_t>(waveIdx), 0L, 0L, 0L};
            epilogueOp_.UpdateGlobalAddr(vecBaseOffset);
        }
    } else if constexpr(g_coreType == AIC) {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.swigluQuantDataPtr +
            (Get<IDX_C_OFFSET>(baseOffset_) + waveRowOffset * static_cast<int64_t>(outputN)) * sizeof(QuantOutType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.swigluQuantScalePtr +
            (Get<IDX_C_SCALE_OFFSET>(baseOffset_) + waveRowOffset * static_cast<int64_t>(scaleOutputN)) *
                sizeof(QuantScaleOutType);
        gmmAddrInfo.bGlobal = params_.b2GmAddr + Get<IDX_B2_OFFSET>(baseOffset_) * sizeof(QuantOutType);
        gmmAddrInfo.bScaleGlobal = params_.b2ScaleGmAddr + Get<IDX_B2_SCALE_OFFSET>(baseOffset_) *
                                    sizeof(QuantScaleOutType);
    }
    // 按 wave 切分的 swiglu->gmm2 flag：每个 expert 占 maxWavesPerExpert 个 int32，每 wave 1 个
    gmmAddrInfo.groupFlagList = (__gm__ int32_t*)params_.workspaceInfo.flagSwiGluToGmm2Ptr +
                                Get<IDX_FLAG_OFFSET>(baseOffset_) + static_cast<int64_t>(waveIdx);
    // dispatch->gmm1 flag：每个 expert 占 maxWavesPerExpert 个 int32，每 wave 1 个。
    // cube 在每个 wave 启动时只等本 wave 的槽位，目标值 = waveM。
    gmmAddrInfo.groupFlagList2 = (__gm__ int32_t*)params_.workspaceInfo.flagDispatchToGmm1Ptr +
                                static_cast<int64_t>(localExpertId) * static_cast<int64_t>(maxWavesPerExpert_) +
                                static_cast<int64_t>(waveIdx);
    // GMM2 直接写 GM 的目的地址（per-wave row offset）
    if (gmmIndex == 1) {
        gmmAddrInfo.gmm2OutGlobal = params_.workspaceInfo.ptrGmmOut +
            (Get<IDX_Y2_OFFSET>(baseOffset_) + waveRowOffset * static_cast<int64_t>(k)) * sizeof(bfloat16_t);
    }
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
    uint32_t dataResFp32BufAlign = dataResBufAlign * 2;
    uint32_t topKWeightsBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(m_ * topK_ * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
    uint32_t tempBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(m_ * topK_ * sizeof(bfloat16_t)), uint32_t(ALIGN_32));
    uint32_t crossRankSyncBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128)) * sizeof(int32_t);

    dataResAddr_ = 0;
    dataResSize_ = dataResBufAlign / sizeof(bfloat16_t);
    resetTokenAddr_ = dataResAddr_ + dataResBufAlign;
    resetTokenSize_ = num / sizeof(int32_t);
    dataResFp32Addr_ = resetTokenAddr_ + num;
    dataResFp32Size_ = dataResFp32BufAlign / sizeof(float);
    topKWeightsAddr_ = dataResFp32Addr_ + dataResFp32BufAlign;
    topKWeightsSize_ = topKWeightsBufAlign / sizeof(float);
    uint32_t tempAddr = topKWeightsAddr_ + topKWeightsBufAlign;
    crossRankSyncAddr_ = tempAddr + tempBufAlign;
    crossRankSyncSize_ = crossRankSyncBufAlign / sizeof(int32_t);

    topKWeightsTensor_ = LocalTensor<float>(TPosition::VECCALC, topKWeightsAddr_, topKWeightsSize_);
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
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, 0>();
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
    // 当 MegaMoeTrace 打开时，y2GmAddr 前 traceBytes 个字节存放时间戳；
    // Unpermute 这里直接跳过那些 token 不写回，避免覆盖打点。tokens 数 << bs，影响很小。
#ifdef MEGA_MOE_TRACE_ENABLE
    int32_t traceBytes = static_cast<int32_t>(MegaMoeTrace::POINTS_PER_CORE) *
                         static_cast<int32_t>(MegaMoeTrace::MAX_AIC_CORES +
                                              MegaMoeTrace::MAX_AIV_CORES) *
                         static_cast<int32_t>(sizeof(uint64_t));
    int32_t bytesPerToken = static_cast<int32_t>(k_) * static_cast<int32_t>(sizeof(bfloat16_t));
    int32_t traceSkipTokens = (traceBytes + bytesPerToken - 1) / bytesPerToken;
#else
    int32_t traceSkipTokens = 0;
#endif
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    for (int32_t tokenIdx = coreOffset; tokenIdx < coreLen + coreOffset; tokenIdx++) {
        if (tokenIdx < traceSkipTokens) {
            continue;
        }
        SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, 2>();
        LocalTensor<bfloat16_t> dataResBf16(TPosition::VECCALC, dataResAddr_, dataResSize_);
        LocalTensor<bfloat16_t> dataIn0Bf16 = dataResBf16[k_];
        LocalTensor<bfloat16_t> dataIn1Bf16 = dataResBf16[k_ * 2];
        LocalTensor<float> dataResFp32(TPosition::VECCALC, dataResFp32Addr_, dataResFp32Size_);
        LocalTensor<float> dataIn0Fp32 = dataResFp32[k_];
        LocalTensor<float> dataIn1Fp32 = dataResFp32[k_ * 2];
        for (int32_t expId = 0; expId < topK_; ++expId) {
            float expScale = topKWeightsTensor_.GetValue(tokenIdx * topK_ + expId);
            auto event = (expId % 2 == 0) ? EVENT_ID0 : EVENT_ID1;
            auto dataInBf16 = (expId % 2 == 0) ? dataIn0Bf16 : dataIn1Bf16;
            auto dataInFp32 = (expId % 2 == 0) ? dataIn0Fp32 : dataIn1Fp32;
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
                Muls(dataResFp32, dataInFp32, expScale, k_);
            } else {
                Muls(dataInFp32, dataInFp32, expScale, k_);
                PipeBarrier<PIPE_V>();
                Add(dataResFp32, dataResFp32, dataInFp32, k_);
                PipeBarrier<PIPE_V>();
            }
            SetFlag<AscendC::HardEvent::V_MTE2>(event);
        }
        // fp32 -> bf16
        Cast(dataResBf16, dataResFp32, AscendC::RoundMode::CAST_RINT, k_);
        SyncFuncStatic<AscendC::HardEvent::V_MTE3, 2>();
        DataCopy(output[tokenIdx * k_], dataResBf16, k_);
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
    MegaMoeTrace::RecordTimestampAt(params_.y2GmAddr, MegaMoeTrace::P_KERNEL_ENTRY);
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    BuffInit();
    SendMaskCal();   // 源卡按所有全局专家算 mask 并推送到目标专家卡(替代 TopKIdsAllGather)
    ResetFlagList();
    ResetGmm2RowReadyFlags();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    MegaMoeTrace::RecordTimestampAt(params_.y2GmAddr, MegaMoeTrace::P_ROUTING_DONE);
    QuantProcessInRank();

    // 全卡同步
    // Option B: SendMaskCal 扁平分核后, 推送 rank Y 的核与 CrossRankSyncInWorldSize 中 signal rank Y
    // 的核不再同一个, 同核 push+signal 顺序保证失效。这里先做一次本地 barrier+SyncAll, 确保所有核的
    // mask 推送(及 Quant 写本卡 win)都已 issue/drain, 远端可见后再做跨卡 signal 握手。
    if constexpr(g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
    }
    CrossRankSyncInWorldSize();
    MegaMoeTrace::RecordTimestampAt(params_.y2GmAddr, MegaMoeTrace::P_CROSSRANK_DONE);

    FirstBuffInit();

    // === prefetch 版本（参考 commit e9f093d）：dispatch 提前一轮 ===
    // pre-loop：AIV1 sub1 先 dispatch expert 0；进入 expert loop 后，每轮在
    // UpdateGroupParams(e, 0) 把 totalSendCntInExp_ 读入 problemShape_.M_VALUE
    // 之后再 prefetch expert e+1。这样 cube 等到 flag(2) 立即跑 GMM1，sub1 在
    // 准备下一个 SendInfoCal/Dispatch 时与 sub0/AIC 的 epi/GMM2 并行。
    //
    // 跨轮共用变量隔离分析：
    //   - totalSendCntInExp_：UpdateGroupParams sub1 在 prefetch 之前先把它读入
    //                         problemShape_.M_VALUE，再用 m_i = Get<M_VALUE>(...)
    //                         捕获到栈上。prefetch 改写 totalSendCntInExp_ 不会
    //                         影响本轮已捕获的 m_i / waveCnt。
    //   - cumsumRevCntInRank_：全局累加器，SendInfoCal(0..N-1) 仍按序在 AIV1 sub1
    //                          上串行执行（pre-loop + body 内 prefetch 单链），顺序成立。
    //   - cumsumInfoTensor_：按 (localExpertId * worldSize_ + rank) 寻址，每个
    //                        expert 独立槽位，prefetch 不会踩本轮。
    //   - validTopkIndexTensor_/topkIndexTensor_/gatherMaskInt32Tensor_/
    //     expertTokenCntTensor_：仅在同一对 (SendInfoCal(e) → TripleInfoCalAndDispatch(e))
    //                            内部被写后读，prefetch 把这对调用作为一个原子段一次跑完。
    //   - GM flag(2) / expertRevNumsGloablTensor_[e*8*aicNum+8*blockIdx]：按 expertIdx
    //                          隔离槽位 + flag 计数器自然排队，AIC 端 1:1 消费。
    //   - AIV0 与 AIV1 各自 256KB UB 独立 (见之前 UB 串扰分析)；prefetch (e+1)
    //                          的 UB 写入与 AIV0 epi(e) 互不可见，无串扰。
    //
    // 交织：GMM1 scheduler 走全 B 宽 n（每 tile 1 次 mmad）→ tileNum = n / L1_TILE_N
    // 非交织：GMM1 scheduler 走 swiglu 输出 N = n/2（每 tile 2 次 mmad）→ tileNum = (n/2) / L1_TILE_N
    uint64_t schedulerN_const;
    if constexpr (IsInterleaved) {
        schedulerN_const = Get<N_VALUE>(problemShape_);
    } else {
        schedulerN_const = Get<N_VALUE>(problemShape_) / SWIGLU_N_HALF;
    }
    uint32_t tileNumGmm1 = static_cast<uint32_t>(
        (schedulerN_const + static_cast<uint64_t>(MegaMoeImpl::L1_TILE_N) - 1UL) /
            static_cast<uint64_t>(MegaMoeImpl::L1_TILE_N));
    if (tileNumGmm1 < 1U) {
        tileNumGmm1 = 1U;
    }
    // 跨 wave 分核优化：MAX_TILES_PER_CORE_GMM1 = 1 把 "N 方向多 tile 占满一个 wave" 换成
    // "N 方向 1 tile + 同一个核串多个 wave"。这样:
    //   wavesParallel = blockNum/tileNumGmm1，coresPerWave = tileNumGmm1
    constexpr uint32_t MAX_TILES_PER_CORE_GMM1 = 1U;
    uint32_t minCoresPerWaveGmm1 = (tileNumGmm1 + MAX_TILES_PER_CORE_GMM1 - 1U) / MAX_TILES_PER_CORE_GMM1;
    if (minCoresPerWaveGmm1 < 1U) {
        minCoresPerWaveGmm1 = 1U;
    }

    // pre-loop: 把 expert 0 的 dispatch 先发出去，启动整条流水线。
    // groupFlagListGm2_ 指向 expert 0 的 wave-0 槽起点；TripleInfoCalAndDispatch
    // 在 wave 边界用 (basePtr + curWave) 偏移做 AtomicAdd。
    if constexpr (g_coreType == AscendC::AIV) {
        if (subBlockIdx_ == 1 && expertPerRank_ > 0U) {
            MegaMoeTrace::RecordDispatchPoint(params_.y2GmAddr, 0, 0);
            SendInfoCal(0);
            MegaMoeTrace::RecordDispatchPoint(params_.y2GmAddr, 0, 1);
            groupFlagListGm2_.SetGlobalBuffer(
                (__gm__ int32_t*)params_.workspaceInfo.flagDispatchToGmm1Ptr);
            TripleInfoCalAndDispatch(static_cast<int32_t>(totalSendCntInExp_), 0);
            MegaMoeTrace::RecordDispatchPoint(params_.y2GmAddr, 0, 2);
        }
    }

    // ── dispatch↔GMM2 UB 同步的计数模型（AIV1 写 triple@UB[128K+]，AIC fixpipe 也写同区）──
    // SetFlag(8) 由 AIV1 在每次 TripleInfoCalAndDispatch 末尾发出，次数 = expert 粒度：
    //   pre-loop dispatch(0) 1 次 + 主循环 dispatch(e+1)（e+1<N，含 invalid expert）N-1 次 = N 次。
    // 为严格配平，AIC 侧也按 expert 粒度恰好 wait N 次：循环前先消费 pre-loop 的 dispatch(0) 这一发，
    // 之后每个 expert（valid/invalid 都执行，与 set 一致）在 dispatch 调用之后消费 dispatch(e+1) 这一发。
    // 这样 GMM2(e) 等到的恰是「与它并行抢 UB 的」dispatch(e+1)，而非早已完成的 dispatch(e)。
    if constexpr (g_coreType == AscendC::AIC) {
        if (expertPerRank_ > 0U) {
            CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_M>(8 + 16);  // 消费 pre-loop dispatch(0)
        }
    }

    // for 循环: per-expert 并行 wave GMM1→GMM2 流水 + prefetch (expertIdx+1) 的 dispatch
    GMMAddrInfo gmmAddrInfo;
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
        // 当前 expert 的 SendInfoCal 已经在 pre-loop 或上一轮 prefetch 中跑完，
        // UpdateGroupParams sub1 读到的 totalSendCntInExp_ 即为 expert e 的值。
        bool valid = UpdateGroupParams(expertIdx, 0);

        // prefetch: 在本轮 UpdateGroupParams 之后，立刻把下一个 expert 的 dispatch 发出去。
        // 注意必须在 UpdateGroupParams 之后调用，否则 totalSendCntInExp_ 会被 (e+1) 提前覆盖。
        // groupFlagListGm2_ 指向 expert (e+1) 的 wave-0 槽起点，AtomicAdd 内部按 wave 偏移。
        if constexpr (g_coreType == AscendC::AIV) {
            if (subBlockIdx_ == 1 && expertIdx + 1U < expertPerRank_) {
                MegaMoeTrace::RecordDispatchPoint(params_.y2GmAddr,
                                                  static_cast<int>(expertIdx + 1U), 0);
                SendInfoCal(expertIdx + 1U);
                MegaMoeTrace::RecordDispatchPoint(params_.y2GmAddr,
                                                  static_cast<int>(expertIdx + 1U), 1);
                groupFlagListGm2_.SetGlobalBuffer(
                    (__gm__ int32_t*)params_.workspaceInfo.flagDispatchToGmm1Ptr +
                    static_cast<int64_t>(expertIdx + 1U) * static_cast<int64_t>(maxWavesPerExpert_));
                TripleInfoCalAndDispatch(static_cast<int32_t>(totalSendCntInExp_),
                                         static_cast<int32_t>(expertIdx + 1U));
                MegaMoeTrace::RecordDispatchPoint(params_.y2GmAddr,
                                                  static_cast<int>(expertIdx + 1U), 2);
            }
        }

        // 本轮 dispatch(e+1) 发了一次 SetFlag(8)（e+1<N 时；含 invalid expert）。AIC 必须恰好
        // 消费一次，否则 per-core 计数漂移。消费点理想是「GMM2(e) 第一个 tile 之前」——这样
        // GMM1(e) 仍能与 dispatch(e+1) 并行，只在 fixpipe 写 UB[128K+] 前才挡。但有三类核到不了
        // 那个点：!valid（下方 continue）、!coreActive_（batch loop 跑空）。用 needDispatchWait
        // 标记本轮待消费，正常核在首个 GMM2 前消费，漏网核在轮尾兜底，保证每核每轮恰好一次。
        bool needDispatchWait = (g_coreType == AscendC::AIC) && (expertIdx + 1U < expertPerRank_);

        if (!valid) {
            if (needDispatchWait) {
                CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_M>(8 + 16);
            }
            continue;
        }
        uint64_t m_i = Get<M_VALUE>(problemShape_);
        uint64_t waveCnt = (m_i + MegaMoeImpl::L1_TILE_M - 1) / MegaMoeImpl::L1_TILE_M;

        // 按本 expert 的 waveCnt 决定分组：让 wavesParallel 尽量等于 waveCnt
        uint32_t maxGroups = blockNum_ / minCoresPerWaveGmm1;
        if (maxGroups < 1U) {
            maxGroups = 1U;
        }
        wavesParallel_ = (static_cast<uint32_t>(waveCnt) < maxGroups) ? static_cast<uint32_t>(waveCnt) : maxGroups;
        if (wavesParallel_ < 1U) {
            wavesParallel_ = 1U;
        }
        coresPerWave_ = blockNum_ / wavesParallel_;
        coreInGroup_ = blockIdx_ % coresPerWave_;
        coreGroupId_ = blockIdx_ / coresPerWave_;
        coreActive_ = (coreGroupId_ < wavesParallel_);

        // 为当前 expert 配置 cube 用的 gmmAddrInfo（dispatch 不在这里发了，已经被预走了）。
        // groupFlagList2 在 wave loop 中由 UpdateGlobalBuffer(waveGmmInfo, 0, expertIdx, waveIdx)
        // 改写成 per-wave 槽位，这里先指向 expert 的 wave-0 起点作占位。
        UpdateGlobalBuffer(gmmAddrInfo, 0, expertIdx, 0);
        gmmAddrInfo.groupFlagList2 =
            (__gm__ int32_t*)params_.workspaceInfo.flagDispatchToGmm1Ptr +
            static_cast<int64_t>(expertIdx) * static_cast<int64_t>(maxWavesPerExpert_);
        groupFlagListGm2_.SetGlobalBuffer(gmmAddrInfo.groupFlagList2);

        // ★ Wave loop: 跨 wave 分核 batch 结构 ★
        // 本核所在组负责 waveIdx ∈ {coreGroupId_, coreGroupId_ + wavesParallel_, ...}
        // batch 长度 MAX_CROSS_WAVE_GMM1 = 2：同核连续把 batch 内 N 个 wave 的 GMM1+epi 全部跑完，
        // 再统一跑这 N 个 wave 的 GMM2+combine。GMM2 走 GM-direct + rowReadyFlags，
        // 与 GMM1 cube/vec 通道不相干，相邻 batch 之间无需 EndSync drain。
        constexpr uint32_t MAX_CROSS_WAVE_GMM1 = 2U;
        uint64_t waveStart = coreActive_ ? static_cast<uint64_t>(coreGroupId_) : waveCnt;
        uint64_t batchedWaves[MAX_CROSS_WAVE_GMM1];

        for (uint64_t batchAnchor = waveStart; batchAnchor < waveCnt;
             batchAnchor += static_cast<uint64_t>(wavesParallel_) * static_cast<uint64_t>(MAX_CROSS_WAVE_GMM1)) {
            // 收集本 batch 实际包含的 wave id
            uint32_t batchSize = 0;
            for (uint32_t bi = 0; bi < MAX_CROSS_WAVE_GMM1; ++bi) {
                uint64_t w = batchAnchor + static_cast<uint64_t>(bi) * static_cast<uint64_t>(wavesParallel_);
                if (w >= waveCnt) {
                    break;
                }
                batchedWaves[batchSize++] = w;
            }

            // ===== Phase 1: 把 batch 内每个 wave 的 GMM1 + epi 跑完 =====
            //   batch 内不调 EndSync：下一个 GMM1 首 tile 的 WaitForVector 自动 drain。
            for (uint32_t bi = 0; bi < batchSize; ++bi) {
                uint64_t waveIdx = batchedWaves[bi];
                int curWaveTrace = waveTraceIdx_ + static_cast<int>(bi);

                MegaMoeTrace::RecordWavePoint(params_.y2GmAddr, curWaveTrace,
                                              MegaMoeTrace::SP_WAVE_ENTRY);

                uint64_t waveM = (waveIdx == waveCnt - 1 && m_i % MegaMoeImpl::L1_TILE_M != 0) ?
                                 m_i % MegaMoeImpl::L1_TILE_M : MegaMoeImpl::L1_TILE_M;
                Get<M_VALUE>(problemShape_) = waveM;

                GMMAddrInfo waveGmmInfo = gmmAddrInfo;
                UpdateGlobalBuffer(waveGmmInfo, 0, expertIdx, waveIdx);
                groupFlagListGm_.SetGlobalBuffer(waveGmmInfo.groupFlagList);

                MegaMoeTrace::RecordWavePoint(params_.y2GmAddr, curWaveTrace,
                                              MegaMoeTrace::SP_GMM1_BEGIN);
                if (subBlockIdx_ == 0) {
                    MegaMoeImpl::GroupMatmulSwigluQuant<
                        QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, IsInterleaved>(
                        epilogueOp_, params_, problemShape_, waveGmmInfo, coreInGroup_, coresPerWave_,
                        vecSetSyncCom_, gmm1PingPongIdx_, curWaveTrace);
                }
                MegaMoeTrace::RecordWavePoint(params_.y2GmAddr, curWaveTrace,
                                              MegaMoeTrace::SP_GMM1_END);
            }

            // ===== GMM1→GMM2 切换（不在此 drain GMM1 的 AIV0 信号）=====
            // swiglu(w)→GMM2(w) 的数据依赖已由逐 wave 的 GM flag 单独保证：
            //   epilogue 在 CopyOutputFromUb2Gm 后做 MTE3→V/MTE3→S 屏障，再 AtomicAdd
            //   flagSwiGluToGmm2Ptr[wave]；GMM2 首 tile 用 ReadGmByPassDCache 自旋等到该 flag 达标
            //   (= 本 wave 的 swiglu tile 数) 才开算。故 GMM2(w0) 只依赖 swiglu(w0)，可与还在跑的
            //   swiglu(w2) 重叠，不该被「等齐 batch 内全部 swiglu」的 EndSync 卡住。
            // 因此把 AIV0→AIC 的 pending NotifyCube 惰性延后到 Phase2 之后再 drain（见 batch 末尾）：
            //   GMM2 用 AIV1 UB 与 flags 16-26，和 AIV0 的 UB/flags 0-10 物理独立，Phase2 期间互不干扰；
            //   swiglu 在 Phase2 期间已全部完成，届时 drain 不阻塞，且仍在「下个 batch GMM1 覆写 AIV0 UB」
            //   之前，保证 AIV0 UB 复用安全。

            // dispatch↔GMM2 的 UB 同步：在「本 expert 第一个 GMM2 tile 之前」消费一次本轮
            // dispatch(e+1) 的 SetFlag(8)。放这里而非 GMM1 前，是为了保住 GMM1(e) ∥ dispatch(e+1)
            // 的并行——只在 fixpipe 真正要写 UB[128K+] 前才挡。needDispatchWait 跨 batch 只消费一次。
            if (needDispatchWait) {
                CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_M>(8 + 16);
                needDispatchWait = false;
            }

            // ===== Phase 2: GMM2 (toAiv1, UB 128×256 单 buffer) + combine =====
            for (uint32_t bi = 0; bi < batchSize; ++bi) {
                uint64_t waveIdx = batchedWaves[bi];
                int curWaveTrace = waveTraceIdx_ + static_cast<int>(bi);

                uint64_t waveM = (waveIdx == waveCnt - 1 && m_i % MegaMoeImpl::L1_TILE_M != 0) ?
                                 m_i % MegaMoeImpl::L1_TILE_M : MegaMoeImpl::L1_TILE_M;
                Get<M_VALUE>(problemShape_) = waveM;

                GMMAddrInfo waveGmmInfo = gmmAddrInfo;
                UpdateGlobalBuffer(waveGmmInfo, 1, expertIdx, waveIdx);
                groupFlagListGm_.SetGlobalBuffer(waveGmmInfo.groupFlagList);

                uint64_t waveRowOffset = static_cast<uint64_t>(waveIdx) *
                                         static_cast<uint64_t>(MegaMoeImpl::L1_TILE_M);

                MegaMoeTrace::RecordWavePoint(params_.y2GmAddr, curWaveTrace,
                                              MegaMoeTrace::SP_GMM2_BEGIN);
                if constexpr(g_coreType == AscendC::AIC) {
                    MegaMoeImpl::GroupMatmul2Optimize<
                        QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, IsInterleaved>(
                        params_, problemShape_, waveGmmInfo, coreInGroup_, coresPerWave_,
                        expertBeforeCnt_, expertIdx, rankId_, waveRowOffset, curWaveTrace, gmm2SyncCom_);
                } else if (subBlockIdx_ == 1) {
                    MegaMoeImpl::GroupMatmul2Optimize<
                        QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, IsInterleaved>(
                        params_, problemShape_, waveGmmInfo, coreInGroup_, coresPerWave_,
                        expertBeforeCnt_, expertIdx, rankId_, waveRowOffset, curWaveTrace, gmm2SyncCom_);
                }
                MegaMoeTrace::RecordWavePoint(params_.y2GmAddr, curWaveTrace,
                                              MegaMoeTrace::SP_GMM2_END);
                MegaMoeTrace::RecordWavePoint(params_.y2GmAddr, curWaveTrace,
                                              MegaMoeTrace::SP_WAVE_END);
            }

            // ===== Phase2 之后惰性 drain GMM1 的 AIV0→AIC NotifyCube =====
            // swiglu 在 Phase2 期间(与 GMM2 重叠)已全部完成，此处 drain 不再阻塞 cube；仅为把
            // AIV0 的 pending NotifyCube 收掉、复位 vecSetSyncCom_/gmm1PingPongIdx_，保证下一个
            // batch/expert 的 GMM1 覆写 AIV0 UbFirst/UbSecond 之前，本 batch swiglu 已读完该 UB。
            // EndSync 内部按 vecSetSyncCom_ 决定 drain 1/2 个通道，并用 gmm1PingPongIdx_ 定 drain 顺序，
            // 二者此刻仍是 Phase1 结束时的值（Phase2 未改动），语义与原先 Phase2 前 drain 完全一致。
            EndSync();
            vecSetSyncCom_ = 0;
            gmm1PingPongIdx_ = 0;

            // 确保 GMM2 的 MTE3 combine 全部完成，AIV1 UB 释放，再让下一轮 prefetch dispatch 用
            if constexpr (g_coreType == AscendC::AIV) {
                if (subBlockIdx_ == 1) {
                    PipeBarrier<PIPE_ALL>();
                }
            }

            waveTraceIdx_ += static_cast<int>(batchSize);
        }

        // 兜底：!coreActive_ 的核 batch loop 跑空，没走到首个 GMM2 前的消费点，
        // needDispatchWait 仍为 true，在此补消费，保证每 AIC core 每轮恰好消费一次。
        if (needDispatchWait) {
            CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_M>(8 + 16);
        }

        Get<M_VALUE>(problemShape_) = m_i;
    }

    MegaMoeTrace::RecordTimestampAt(params_.y2GmAddr, MegaMoeTrace::P_EXPERT_LOOP_END);

    // ★ 全部 expert 跑完后，drain 最后一次 GMM1 留在 cube/vec flag 槽里的 pending NotifyCube。
    EndSync();
    vecSetSyncCom_ = 0;
    gmm1PingPongIdx_ = 0;

    // ★ GMM2 ping-pong 的延迟 drain：所有 GMM2 都跑完了，此时才 drain 最后两个在飞的 buffer。
    // 这之前的每一轮 GMM2 都不 drain，使 combine(e) 与 GMM1(e+1) 重叠。GMM2 用 AIV1 flag 22/23，
    // 与 EndSync 的 GMM1 AIV0 flag 6/7 不冲突，故两者先后调用互不干扰。
    EndGMM2Sync();

    if constexpr(g_coreType == AIV) {
        if (aivCoreIdx_ == 1) {
            ExpertTokenNumCopyOut();
        }
    }

    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();

    // unpermute & 清理 win 区上放置 token 的位置
    if constexpr(g_coreType == AIV) {
        SecondBuffInit();
        CrossRankSyncInWorldSize(); // 全卡软同步，确认 combine send 完成
        MegaMoeTrace::RecordTimestampAt(params_.y2GmAddr, MegaMoeTrace::P_SYNC_END);
        Unpermute();
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
    MegaMoeTrace::RecordTimestampAt(params_.y2GmAddr, MegaMoeTrace::P_PROCESS_END);
}

}   // namespace MegaMoeImpl
#endif
