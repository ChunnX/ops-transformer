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
    __aicore__ inline void TopKIdsAllGather();
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
        int32_t remoteRankIdx, int32_t copyStartIdx, int32_t copyNum, int32_t& pingpongId);
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

    __aicore__ inline int64_t TokenPerExpertLayout(int64_t dim0, int64_t dim1, int64_t dim2)
    {
        return dim0 * perRankStridesInTokenPerExpert_ + dim1 * expertPerRank_ + dim2;
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

    GM_ADDR tokenPerExpertCurRankAddr_;

    GlobalTensor<int32_t> tokenPerExpertWin_;
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
    int32_t perRankStridesInTokenPerExpert_ = 0;
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
    uint16_t gmm2PingPongIdx_ = 0;
    // GMM1 pingpong：跨 wave 持久化的 ping/pong 状态。
    // tileNum 通常 = blockNum/wavesParallel = coresPerWave，意味着每核每 wave 只 1 个 tile，
    // 单次 GroupMatmulSwigluQuant 调用内 pingpong 无法翻转。把它提到类成员后，
    // 同核 batch 内连续两个 wave (wave0/wave2 或 wave1/wave3) 分别落在 UbFirst/UbSecond，
    // 实现 GMM1(N) mmad 与 swiglu(N-1) 在 ping/pong 半区真正并行。
    uint16_t gmm1PingPongIdx_ = 0;
    uint64_t sendTotalNum_ = 0;
    uint32_t maskAlignSize_ = 0;
    uint32_t topkOffsetPerRank_ = 0;
    uint64_t totalSendCntInExp_ = 0;
    uint64_t cumsumRevCntInRank_ = 0;
    int32_t compareCount_ = 0;

    // 6-buffer 软流水：把 BufferNum 从 2 提到 6,占满 EVENT_ID0..EVENT_ID5。
    //   - SendInfoCal 用 topkTensors_[6] 让跨卡 MTE2 读在引擎里同时排队 5+ 个;
    //     V engine (CompareScalar + GatherMask) 仍是单 buffer (validTopkIndexTensor_) 串行,
    //     但 MTE2 端的延迟可被并发隐藏。
    //   - TripleInfoCalAndDispatch 用 copyTmpTensors_[6] 让 winmem token+scale 读
    //     在引擎里同时排队 5+ 个(配合 prime/steady/drain 软流水)。
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
    tokenPerExpertWin_.SetGlobalBuffer((__gm__ int32_t*)params_.peermemInfo.topKInfoInWorldSizePtr);
    expertTokenNumsOut_.SetGlobalBuffer((__gm__ int32_t*)params_.expertTokenNumsOutGmAddr);
    expertRevNumsGloablTensor_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.expertRevTokenNumsPtr);
    tripleGloablTensor_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.tripleInfoPtr);
    epilogueOp_.Init({params_.workspaceInfo.swigluQuantDataPtr, params_.workspaceInfo.swigluQuantScalePtr,
        params_.workspaceInfo.flagSwiGluToGmm2Ptr, nullptr, nullptr, nullptr, ALIGN_256, ALIGN_256});
    perRankStridesInTokenPerExpert_ = Ops::Base::CeilAlign(
        static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128));
    tokenPerExpertCurRankAddr_ = params_.peermemInfo.topKInfoInWorldSizePtr +
        TokenPerExpertLayout(rankId_, 0, 0) * sizeof(int32_t);
    topkOffsetPerRank_ = Ops::Base::CeilAlign((int64_t)(sendTotalNum_ * sizeof(int32_t)), (int64_t)ALIGN_512);
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
    compareCount_ = Ops::Base::CeilAlign((int64_t)(sendTotalNum_ * sizeof(int32_t)),
        (int64_t)(ALIGN_256)) / sizeof(int32_t);
    maskAlignSize_ = Ops::Base::CeilAlign(Ops::Base::CeilAlign((int64_t)(compareCount_), (int64_t)(8)),
        (int64_t)(ALIGN_32)); 
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
    uint32_t gatherMaskTensorSize = maskAlignSize_ * worldSize_;
    gatherMaskTensor_ = LocalTensor<uint8_t>(TPosition::VECCALC, gatherMaskTensorAddr,
        gatherMaskTensorSize / sizeof(uint8_t));
    gatherMaskInt32Tensor_ = LocalTensor<uint32_t>(TPosition::VECCALC, gatherMaskTensorAddr,
        gatherMaskTensorSize / sizeof(uint32_t));
    // 6 个 topkTensor buffer 给 SendInfoCal 用,跨卡 MTE2 同时排队多个读。
    uint32_t topkTensorBaseAddr = gatherMaskTensorAddr + gatherMaskTensorSize;
    uint32_t topkTensorPerBufSize = Ops::Base::CeilAlign(
        (int64_t)(sendTotalNum_ * sizeof(int32_t)), (int64_t)(ALIGN_256));
    for (int32_t b = 0; b < DISPATCH_BUFFER_NUM; ++b) {
        topkTensors_[b] = LocalTensor<int32_t>(TPosition::VECCALC,
            topkTensorBaseAddr + static_cast<uint32_t>(b) * topkTensorPerBufSize,
            topkTensorPerBufSize / sizeof(int32_t));
    }
    uint32_t validTopkIndexTensorAddr = topkTensorBaseAddr +
        static_cast<uint32_t>(DISPATCH_BUFFER_NUM) * topkTensorPerBufSize;
    uint32_t validTopkIndexTensorSize = Ops::Base::CeilAlign(
        (int64_t)(sendTotalNum_ * sizeof(int32_t)), (int64_t)(ALIGN_32));
    validTopkIndexTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, validTopkIndexTensorAddr,
        validTopkIndexTensorSize / sizeof(int32_t));
    // 6 个 dispatch buffer，配合 EVENT_ID0..EVENT_ID5 做软流水。
    // 每槽 8KB，足以容纳一个 token+scale (mxFp8TokenAlignBytes_ + mxFp8ScaleAlignBytes_ 通常 < 7.5KB)。
    // 6 × 8KB = 48KB；FirstBuffInit 总用量从 ~48KB 提到 ~96KB，仍远低于 256KB UB 上限。
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
    uint32_t topkIdsTensorSize = Ops::Base::CeilAlign((int64_t)(sendTotalNum_ * sizeof(int32_t)), (int64_t)(ALIGN_32));
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
    if (vecSetSyncCom_ <= 0) {
        return;
    } else if (vecSetSyncCom_ == 1) {
        WaitForVector();
    } else {
        WaitForVector(gmm2PingPongIdx_);
        WaitForVector(1 - gmm2PingPongIdx_);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::TopKIdsAllGather()
{
    // topKInfo给对端发送，当前分核逻辑为一个核负责一张卡的发送
    if constexpr(g_coreType == AIC) {
        return;
    }

    GlobalTensor<int32_t> srcGlobalTensor;
    GlobalTensor<int32_t> dstGlobalTensor;
    srcGlobalTensor.SetGlobalBuffer((__gm__ int32_t*)params_.expertIdxGmAddr);
    DataCopyExtParams dataCopyParams{1U, static_cast<uint32_t>(sendTotalNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(topkIdsTensor_, srcGlobalTensor, dataCopyParams, copyPadParams);
    SyncFuncStatic<AscendC::HardEvent::MTE2_MTE3, 1>();
    for (int32_t dstRankIdx = aivCoreIdx_; dstRankIdx < worldSize_; dstRankIdx += blockAivNum_) {
        uint64_t offsetInDstWinAddr = PEERMEM_DATA_OFFSET + rankId_ * Ops::Base::CeilAlign(
        (int64_t)(sendTotalNum_ * sizeof(int32_t)), (int64_t)ALIGN_512);
        dstGlobalTensor.SetGlobalBuffer((__gm__ int32_t*)GetRankWinAddrWithOffset(dstRankIdx, offsetInDstWinAddr));
        DataCopyPad(dstGlobalTensor, topkIdsTensor_, dataCopyParams);
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
    // 6-buffer 软流水版 SendInfoCal
    // ----------------------------------------------------------------------------
    // 旧版（pingpong, 2 buffer）每轮结构：
    //   WaitFlag<S_MTE2>(ev)  [MTE2 pipe 等,scalar 继续]
    //   DataCopyPad → SetFlag<MTE2_V>(ev) → WaitFlag<MTE2_V>(ev) [V pipe 等,scalar 继续]
    //   CompareScalar → GatherMask(sendCnt) → SetFlag<V_S>(ev)
    //   WaitFlag<V_S>(ev)  ★scalar 在此阻塞,等本轮 V 完成才能读 sendCnt★
    //   totalSend += sendCnt;  cumsumInfo.SetValue(...);  SetFlag<S_MTE2>(ev)
    //
    // 由于 WaitFlag<V_S> 是 S pipe 等 V pipe,scalar 必须等本轮 V 完成,
    // 下轮 MTE2 issue 才能发出。pingpong 下 MTE2 引擎单次只有 1 个跨卡读在途。
    //
    // 新版把"发 MTE2"和"读 sendCnt"解耦:
    //   Prime:    一次性 issue 前 6 个 rank 的 MTE2,让跨卡读在引擎里并行。
    //   Steady:   每轮在 V_S 阻塞之前的状态,MTE2 引擎已有 5+ 个跨卡读在途;
    //             阻塞之后读 sendCnt 时把 (calRankId + 6) 的 MTE2 接着发。
    //
    // V engine 仍然单 buffer(validTopkIndexTensor_)+单线程,串行没改;
    // 多 buffer 只是隐藏 MTE2 端跨卡延迟。
    // ============================================================================
    constexpr int32_t BufferNum = DISPATCH_BUFFER_NUM;
    constexpr TEventID kBufEvents[BufferNum] = {
        EVENT_ID0, EVENT_ID1, EVENT_ID2, EVENT_ID3, EVENT_ID4, EVENT_ID5
    };

    totalSendCntInExp_ = 0;
    uint32_t mask = sendTotalNum_;
    int32_t compareExpertId = localExpertId + expertPerRank_ * rankId_;
    DataCopyExtParams dataCopyParams{1U, static_cast<uint32_t>(sendTotalNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadParams{true, 0U,
        uint8_t(Ops::Base::CeilAlign((int64_t)(sendTotalNum_), (int64_t)(8)) - sendTotalNum_), 0U}; // 填充数据为 0
    Duplicate<uint32_t>(gatherMaskInt32Tensor_, 0, (worldSize_ * maskAlignSize_) / sizeof(uint32_t)); // mask 位置清理

    // 预置 6 个 S_MTE2 flag,prime 阶段的 WaitFlag<S_MTE2>(MTE2 pipe 等)立即放行。
    for (int32_t b = 0; b < BufferNum; ++b) {
        SetFlag<AscendC::HardEvent::S_MTE2>(kBufEvents[b]);
    }

    int64_t srcStride = Ops::Base::CeilAlign(
        (int64_t)(sendTotalNum_ * sizeof(int32_t)), (int64_t)ALIGN_512);
    GlobalTensor<int32_t> srcGlobalTensor;

    // ----- Prime: issue 前 BufferNum 个 rank 的 MTE2 -----
    int32_t primeCount = (static_cast<int32_t>(worldSize_) < BufferNum) ?
        static_cast<int32_t>(worldSize_) : BufferNum;
    for (int32_t p = 0; p < primeCount; ++p) {
        uint64_t offsetInDstWinAddr = PEERMEM_DATA_OFFSET + static_cast<uint64_t>(p) * srcStride;
        srcGlobalTensor.SetGlobalBuffer((__gm__ int32_t*)GetRankWinAddrWithOffset(p, offsetInDstWinAddr));
        TEventID ev = kBufEvents[p];
        WaitFlag<AscendC::HardEvent::S_MTE2>(ev);
        DataCopyPad(topkTensors_[p], srcGlobalTensor, dataCopyParams, copyPadParams);
        SetFlag<AscendC::HardEvent::MTE2_V>(ev);
    }

    // ----- Steady: 每轮 V/S 完成本 rank 的 sendCnt 后,issue (rank + BufferNum) 的 MTE2 -----
    for (int32_t calRankId = 0; calRankId < static_cast<int32_t>(worldSize_); ++calRankId) {
        uint64_t sendCnt = 0;
        int32_t buf = calRankId % BufferNum;
        TEventID ev = kBufEvents[buf];
        LocalTensor<int32_t> topkTensor = topkTensors_[buf];

        WaitFlag<AscendC::HardEvent::MTE2_V>(ev);
        CompareScalar(gatherMaskTensor_[calRankId * maskAlignSize_], topkTensor, compareExpertId,
            AscendC::CMPMODE::EQ, compareCount_);
        GatherMask(validTopkIndexTensor_, topkIndexTensor_,
            gatherMaskInt32Tensor_[calRankId * (maskAlignSize_ / sizeof(uint32_t))],
            true, mask, {1, 1, 0, 0}, sendCnt);
        SetFlag<AscendC::HardEvent::V_S>(ev);
        WaitFlag<AscendC::HardEvent::V_S>(ev);

        totalSendCntInExp_ += sendCnt;
        cumsumRevCntInRank_ += sendCnt;
        cumsumInfoTensor_.SetValue(localExpertId * worldSize_ + calRankId, (int32_t)cumsumRevCntInRank_);
        SetFlag<AscendC::HardEvent::S_MTE2>(ev);

        // 接着发 (rank + BufferNum) 的 MTE2,复用 buf 槽。
        int32_t rankNext = calRankId + BufferNum;
        if (rankNext < static_cast<int32_t>(worldSize_)) {
            uint64_t offsetInDstWinAddr = PEERMEM_DATA_OFFSET +
                static_cast<uint64_t>(rankNext) * srcStride;
            srcGlobalTensor.SetGlobalBuffer(
                (__gm__ int32_t*)GetRankWinAddrWithOffset(rankNext, offsetInDstWinAddr));
            WaitFlag<AscendC::HardEvent::S_MTE2>(ev);
            DataCopyPad(topkTensors_[buf], srcGlobalTensor, dataCopyParams, copyPadParams);
            SetFlag<AscendC::HardEvent::MTE2_V>(ev);
        }
    }

    // ----- Drain: WaitFlag<S_MTE2> × BufferNum -----
    // - 没被 Steady 用到的槽（worldSize < BufferNum 时）保留 pre-set。
    // - 用过的槽在最后一轮 Steady 的 phase B 里 SetFlag<S_MTE2> 但没人 Wait（rankNext 越界）。
    for (int32_t b = 0; b < BufferNum; ++b) {
        WaitFlag<AscendC::HardEvent::S_MTE2>(kBufEvents[b]);
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
    int32_t rowDstOffsetInCore, int32_t remoteRankIdx, int32_t copyStartIdx, int32_t copyNum, int32_t& pingpongId)
{
    constexpr int32_t BufferNum = 2;
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
        PEERMEM_DATA_OFFSET + worldSize_ * topkOffsetPerRank_)));
    for (uint32_t copyIndex = 0; copyIndex < copyNum; ++copyIndex) {
        pingpongId = (pingpongId + 1) % BufferNum;
        TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<QuantOutType> tokenScalebuf = copyTmpTensors_[pingpongId];
        LocalTensor<QuantScaleOutType> bufScale = tokenScalebuf[mxFp8TokenAlignBytes_].template
        ReinterpretCast<QuantScaleOutType>();
        int32_t topkIndex = validTopkIndexTensor_.GetValue(copyStartIdx + copyIndex) ;
        int32_t tokenIndex = topkIndex / topK_;
        tripleTensor_[copyIndex * 8].SetValue(0, remoteRankIdx);
        tripleTensor_[copyIndex * 8].SetValue(1, tokenIndex);
        tripleTensor_[copyIndex * 8].SetValue(2, topkIndex % topK_);
        auto remoteCopyOffset = tokenIndex * copyInNum;

        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        DataCopy(tokenScalebuf, remoteRankGlobalTensor[remoteCopyOffset], copyInNum);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        DataCopyPad(tokenRevGlobalTensor[copyIndex * widthA], tokenScalebuf,
        {1, static_cast<uint16_t>(widthA * sizeof(QuantOutType)), 0U, 0U, 0U});
        DataCopyPad(scaleRevGlobalTensor[copyIndex * widthAScale], bufScale,
        {1, static_cast<uint16_t>(widthAScale * sizeof(QuantScaleOutType)), 0U, 0U, 0U});
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
    }
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, 1>();
    DataCopy(tripleGloablTensor_[rowDstOffsetInCore * 8], tripleTensor_, copyNum * 8); // 三元组搬出
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::TripleInfoCalAndDispatch(
    int32_t totalTokensInExpert, int32_t localExpertId)
{
    // ============================================================================
    // 6-buffer 软流水版 dispatch
    // ----------------------------------------------------------------------------
    // 旧版（pingpong, BufferNum=2）的 loop 结构是：
    //   WaitFlag<MTE3_MTE2>(ev) → DataCopy MTE2 → SetFlag<MTE2_MTE3>(ev)
    //   → WaitFlag<MTE2_MTE3>(ev) ★scalar 在此阻塞,等本槽 MTE2 完成★
    //   → DataCopyPad MTE3 × 2 → SetFlag<MTE3_MTE2>(ev)
    // 由于 scalar 必须等本槽 MTE2 完成才能继续,跨卡 winmem 读在 MTE2 引擎上
    // 始终只有 1 个在途,HCCL/RDMA 延迟无法靠并发隐藏。
    //
    // 新版把整个循环重排成 prime → steady → drain：
    //   Phase 1 (纯 S/V): 一次性把所有 token 的 (rank, tokenIndex, topkInRank) 写到
    //                     tripleTensor_,Phase 2 只读 tripleTensor_ 不再做 V GatherMask。
    //   Phase 2 prime:    连发 6 个 MTE2,中间不插 WaitFlag<MTE2_MTE3>,
    //                     让 MTE2 引擎同时持有 6 个跨卡读请求。
    //   Phase 2 steady:   每轮做 (MTE3_out[i] + MTE2_in[i+6]),槽位循环复用。
    //   Phase 2 drain:    收尾不再发新 MTE2,只等 MTE3。
    //
    // wave AtomicAdd 仍然按 i_out 走（stripe 模式下 waveIdx 单调）；
    // tripleTensor_ 由于 Phase 1 全部预填,只需 Phase 3 一次性按 stripe 顺序搬出。
    // ============================================================================
    constexpr int32_t BufferNum = DISPATCH_BUFFER_NUM;
    constexpr TEventID kBufEvents[BufferNum] = {
        EVENT_ID0, EVENT_ID1, EVENT_ID2, EVENT_ID3, EVENT_ID4, EVENT_ID5
    };

    // 预置 6 个 MTE3_MTE2 flag,prime 阶段的 WaitFlag 可立即通过。
    for (int32_t b = 0; b < BufferNum; ++b) {
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(kBufEvents[b]);
    }
    PipeBarrier<PIPE_ALL>();

    int32_t priorExpertCumsum = (localExpertId == 0) ? 0 :
        cumsumInfoTensor_.GetValue(localExpertId * worldSize_ - 1);
    if (priorExpertCumsum >= static_cast<int32_t>(maxOutputSize_)) {
        totalTokensInExpert = 0;
    } else if (priorExpertCumsum + totalTokensInExpert > static_cast<int32_t>(maxOutputSize_)) {
        totalTokensInExpert = static_cast<int32_t>(maxOutputSize_) - priorExpertCumsum;
    }

    int32_t totalCores = static_cast<int32_t>(blockNum_);
    int32_t coreIdx = static_cast<int32_t>(blockIdx_);
    // stripe 分核：本核处理专家内全局 idx ∈ {coreIdx, coreIdx + totalCores, ...} 的 token。
    int32_t tokensThisCore = (totalTokensInExpert - coreIdx + totalCores - 1) / totalCores;
    if (tokensThisCore < 0) {
        tokensThisCore = 0;
    }

    // wave-粒度 dispatch flag：在 Phase 2 的 MTE3 端按 wave 边界批量 AtomicAdd。
    // stripe 分核下 globalIdxInExpert = coreIdx + i_out * totalCores 单调递增,
    // 所以 waveIdx_token = globalIdxInExpert / L1_TILE_M 单调,边界检测 O(1)。
    int32_t curWave = -1;
    int32_t curWaveCnt = 0;
    constexpr int32_t L1_TILE_M_I32 = static_cast<int32_t>(MegaMoeImpl::L1_TILE_M);

    if (tokensThisCore > 0) {
        // 二分查找：在当前专家内,找到首个 token (expert 内全局 idx = coreIdx) 所在的卡号
        int32_t firstGlobalIdxInExpert = coreIdx;
        int32_t lo = 0;
        int32_t hi = static_cast<int32_t>(worldSize_) - 1;
        int32_t startRank = hi;
        while (lo <= hi) {
            int32_t mid = (lo + hi) >> 1;
            int32_t cumsumAtMid = cumsumInfoTensor_.GetValue(localExpertId * worldSize_ + mid) - priorExpertCumsum;
            if (cumsumAtMid > firstGlobalIdxInExpert) {
                startRank = mid;
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }

        uint32_t tripleTensorAddr = ubBufferUsedAddr_;
        uint32_t tripleTensorSize = tokensThisCore * ALIGN_32;
        tripleTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, tripleTensorAddr,
            tripleTensorSize / sizeof(int32_t));

        SyncFuncStatic<AscendC::HardEvent::V_S, 0>();

        int64_t widthA = Get<K_VALUE>(problemShape_);
        int64_t widthAScale = Ops::Base::CeilDiv((int64_t)widthA, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        uint32_t copyInNum = Ops::Base::CeilAlign((int64_t)(mxFp8TokenAlignBytes_ + mxFp8ScaleAlignBytes_), (int64_t)ALIGN_32);

        // 本 expert dispatch 区基址
        GlobalTensor<QuantOutType> tokenRevGlobalTensor;
        GlobalTensor<QuantScaleOutType> scaleRevGlobalTensor;
        tokenRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(
            params_.workspaceInfo.dispatchRevDataPtr + priorExpertCumsum * widthA));
        scaleRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleOutType*>(
            params_.workspaceInfo.dispatchRevScalePtr + priorExpertCumsum * widthAScale));

        // ============================================================
        // Phase 1: 纯 S/V — 把每个 token 的 (rank, tokenIndex, topkInRank)
        //          写满 tripleTensor_。Phase 2 之后只读 tripleTensor_,
        //          完全不再触发 V GatherMask 或 cumsumInfo S 读。
        // ============================================================
        {
            int32_t curRank = startRank;
            int32_t curRankExprStart = (startRank == 0) ? 0 :
                (cumsumInfoTensor_.GetValue(localExpertId * worldSize_ + startRank - 1) - priorExpertCumsum);
            int32_t curRankExprEnd = cumsumInfoTensor_.GetValue(localExpertId * worldSize_ + startRank) - priorExpertCumsum;

            uint64_t sendCntCurRank = 0;
            GatherMask(validTopkIndexTensor_, topkIndexTensor_,
                gatherMaskInt32Tensor_[curRank * (maskAlignSize_ / sizeof(uint32_t))],
                true, sendTotalNum_, {1, 1, 0, 0}, sendCntCurRank);
            SyncFuncStatic<AscendC::HardEvent::V_S, 1>();

            for (int32_t i = 0; i < tokensThisCore; ++i) {
                int32_t globalIdxInExpert = coreIdx + i * totalCores;

                bool rankChanged = false;
                while (globalIdxInExpert >= curRankExprEnd) {
                    ++curRank;
                    curRankExprStart = curRankExprEnd;
                    curRankExprEnd = cumsumInfoTensor_.GetValue(localExpertId * worldSize_ + curRank) - priorExpertCumsum;
                    rankChanged = true;
                }
                if (rankChanged) {
                    GatherMask(validTopkIndexTensor_, topkIndexTensor_,
                        gatherMaskInt32Tensor_[curRank * (maskAlignSize_ / sizeof(uint32_t))],
                        true, sendTotalNum_, {1, 1, 0, 0}, sendCntCurRank);
                    SyncFuncStatic<AscendC::HardEvent::V_S, 2>();
                }

                int32_t withinRankIdx = globalIdxInExpert - curRankExprStart;
                int32_t topkIndex = validTopkIndexTensor_.GetValue(withinRankIdx);
                int32_t tokenIndex = topkIndex / topK_;

                tripleTensor_[i * 8].SetValue(0, curRank);
                tripleTensor_[i * 8].SetValue(1, tokenIndex);
                tripleTensor_[i * 8].SetValue(2, topkIndex % topK_);
            }
        }

        // ============================================================
        // Phase 2: 6-buffer 软流水 dispatch
        //   Prime: 连发 min(BufferNum, tokensThisCore) 个 MTE2,不等 MTE2 完成,
        //          让跨卡 winmem 读在 MTE2 引擎里同时排队。
        //   Steady (per i_out):
        //     1) wave 边界检查 + 上一 wave AtomicAdd (SyncFuncStatic<MTE3_S>)
        //     2) WaitFlag<MTE2_MTE3>(ev_out) 等本槽 MTE2 就绪
        //     3) MTE3 token + MTE3 scale
        //     4) SetFlag<MTE3_MTE2>(ev_out)
        //     5) 若 i_out + BufferNum < tokensThisCore：issueMTE2(i_out+BufferNum, 同槽)
        //          → WaitFlag<MTE3_MTE2>(ev_out) (上一步刚 Set 的,等 MTE3 完成)
        //          → DataCopy MTE2 → SetFlag<MTE2_MTE3>(ev_out)
        //   Drain: 收尾不再发 MTE2,只有 MTE3 在跑。
        // ============================================================
        int32_t lastRankIn = -1;
        GlobalTensor<QuantOutType> remoteRankGlobalTensor;

        // ----- Prime: 发出前 BufferNum 个 token 的 MTE2 -----
        int32_t primeCount = (tokensThisCore < BufferNum) ? tokensThisCore : BufferNum;
        for (int32_t p = 0; p < primeCount; ++p) {
            int32_t rank = tripleTensor_[p * 8].GetValue(0);
            int32_t tokenIndex = tripleTensor_[p * 8].GetValue(1);
            if (rank != lastRankIn) {
                remoteRankGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(
                    GetRankWinAddrWithOffset(rank,
                        PEERMEM_DATA_OFFSET + worldSize_ * topkOffsetPerRank_)));
                lastRankIn = rank;
            }
            uint64_t remoteCopyOffset = static_cast<uint64_t>(tokenIndex) *
                                        static_cast<uint64_t>(copyInNum);
            TEventID ev = kBufEvents[p];
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ev);
            DataCopy(copyTmpTensors_[p], remoteRankGlobalTensor[remoteCopyOffset], copyInNum);
            SetFlag<AscendC::HardEvent::MTE2_MTE3>(ev);
        }

        // ----- Steady: MTE3[i_out] + issueMTE2[i_out + BufferNum] -----
        for (int32_t i_out = 0; i_out < tokensThisCore; ++i_out) {
            int32_t buf_out = i_out % BufferNum;
            TEventID ev_out = kBufEvents[buf_out];

            int32_t globalIdxInExpert = coreIdx + i_out * totalCores;

            // wave 边界：drain MTE3 后 AtomicAdd 前一 wave 的累积计数
            int32_t waveIdx_token = globalIdxInExpert / L1_TILE_M_I32;
            if (waveIdx_token != curWave) {
                if (curWaveCnt > 0) {
                    SyncFuncStatic<AscendC::HardEvent::MTE3_S, 0>();
                    AtomicAdd((__gm__ int32_t*)(groupFlagListGm2_.GetPhyAddr()) + curWave,
                              int32_t(curWaveCnt));
                }
                curWave = waveIdx_token;
                curWaveCnt = 0;
            }

            WaitFlag<AscendC::HardEvent::MTE2_MTE3>(ev_out);

            int32_t dstOffsetInExpert = globalIdxInExpert;
            LocalTensor<QuantOutType> tokenScalebuf = copyTmpTensors_[buf_out];
            LocalTensor<QuantScaleOutType> bufScale =
                tokenScalebuf[mxFp8TokenAlignBytes_].template ReinterpretCast<QuantScaleOutType>();
            DataCopyPad(tokenRevGlobalTensor[dstOffsetInExpert * widthA], tokenScalebuf,
                {1, static_cast<uint16_t>(widthA * sizeof(QuantOutType)), 0U, 0U, 0U});
            DataCopyPad(scaleRevGlobalTensor[dstOffsetInExpert * widthAScale], bufScale,
                {1, static_cast<uint16_t>(widthAScale * sizeof(QuantScaleOutType)), 0U, 0U, 0U});
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(ev_out);

            ++curWaveCnt;

            // 发下一个槽的 MTE2 (i_out + BufferNum,复用 buf_out 槽)
            int32_t i_in_next = i_out + BufferNum;
            if (i_in_next < tokensThisCore) {
                int32_t rank = tripleTensor_[i_in_next * 8].GetValue(0);
                int32_t tokenIndex = tripleTensor_[i_in_next * 8].GetValue(1);
                if (rank != lastRankIn) {
                    remoteRankGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(
                        GetRankWinAddrWithOffset(rank,
                            PEERMEM_DATA_OFFSET + worldSize_ * topkOffsetPerRank_)));
                    lastRankIn = rank;
                }
                uint64_t remoteCopyOffset = static_cast<uint64_t>(tokenIndex) *
                                            static_cast<uint64_t>(copyInNum);
                // WaitFlag 此处等待的是本轮刚发出的 SetFlag<MTE3_MTE2>(ev_out),
                // 即等本槽这次 MTE3 完成、buffer 释放。
                WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ev_out);
                DataCopy(copyTmpTensors_[buf_out], remoteRankGlobalTensor[remoteCopyOffset], copyInNum);
                SetFlag<AscendC::HardEvent::MTE2_MTE3>(ev_out);
            }
        }

        // ============================================================
        // Phase 3: triple 三元组按 stripe 顺序搬出
        // ============================================================
        SyncFuncStatic<AscendC::HardEvent::S_MTE3, 2>();
        for (int32_t i = 0; i < tokensThisCore; ++i) {
            int32_t dstAbsIdx = priorExpertCumsum + coreIdx + i * totalCores;
            DataCopy(tripleGloablTensor_[dstAbsIdx * 8], tripleTensor_[i * 8], 8);
        }
    }

    // flush 最后一个 wave 累积计数（tokensThisCore==0 时 curWaveCnt 始终为 0,自动跳过）
    if (curWaveCnt > 0) {
        SyncFuncStatic<AscendC::HardEvent::MTE3_S, 0>();
        AtomicAdd((__gm__ int32_t*)(groupFlagListGm2_.GetPhyAddr()) + curWave,
                  int32_t(curWaveCnt));
    }

    // Drain 6 个 buffer-free flag
    //   - 槽 b 在 Steady 中最后一次出现在 i_out = (tokensThisCore - 1) - ((tokensThisCore - 1 - b) % BufferNum)
    //     位置 SetFlag<MTE3_MTE2>(b);若没被 issueMTE2 消耗,则在此 drain。
    //   - 没被 Steady 用到的槽（tokensThisCore < BufferNum 时）仍持有函数开头的 pre-set flag,也由此 drain。
    for (int32_t b = 0; b < BufferNum; ++b) {
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kBufEvents[b]);
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
    TopKIdsAllGather();
    ResetFlagList();
    ResetGmm2RowReadyFlags();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    MegaMoeTrace::RecordTimestampAt(params_.y2GmAddr, MegaMoeTrace::P_ROUTING_DONE);
    QuantProcessInRank();

    // 全卡同步
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

        if (!valid) {
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

            // ===== Phase 2: 把 batch 内每个 wave 的 GMM2 + combine 跑完 =====
            //   GMM2 直接写 GM，AIV1 通过 rowReadyFlags 等数据并 combine，不走 cube/vec 通道。
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
                        expertBeforeCnt_, expertIdx, rankId_, waveRowOffset, curWaveTrace);
                } else if (subBlockIdx_ == 1) {
                    MegaMoeImpl::GroupMatmul2Optimize<
                        QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, IsInterleaved>(
                        params_, problemShape_, waveGmmInfo, coreInGroup_, coresPerWave_,
                        expertBeforeCnt_, expertIdx, rankId_, waveRowOffset, curWaveTrace);
                }
                MegaMoeTrace::RecordWavePoint(params_.y2GmAddr, curWaveTrace,
                                              MegaMoeTrace::SP_GMM2_END);
                MegaMoeTrace::RecordWavePoint(params_.y2GmAddr, curWaveTrace,
                                              MegaMoeTrace::SP_WAVE_END);
            }

            waveTraceIdx_ += static_cast<int>(batchSize);
        }

        Get<M_VALUE>(problemShape_) = m_i;
    }

    MegaMoeTrace::RecordTimestampAt(params_.y2GmAddr, MegaMoeTrace::P_EXPERT_LOOP_END);

    // ★ 全部 expert 跑完后，drain 最后一次 GMM1 留在 cube/vec flag 槽里的 pending NotifyCube。
    EndSync();
    vecSetSyncCom_ = 0;
    gmm1PingPongIdx_ = 0;

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
        Unpermute();
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
    MegaMoeTrace::RecordTimestampAt(params_.y2GmAddr, MegaMoeTrace::P_PROCESS_END);
}

}   // namespace MegaMoeImpl
#endif