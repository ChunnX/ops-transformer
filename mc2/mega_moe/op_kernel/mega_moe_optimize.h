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
    uint64_t sendTotalNum_ = 0;
    uint32_t maskAlignSize_ = 0;
    uint32_t topkOffsetPerRank_ = 0;
    uint64_t totalSendCntInExp_ = 0;
    uint64_t cumsumRevCntInRank_ = 0;
    int32_t compareCount_ = 0;

    LocalTensor<int32_t> topkTensor1_;
    LocalTensor<int32_t> topkTensor2_;
    LocalTensor<int32_t> topkIndexTensor_;
    LocalTensor<uint8_t> gatherMaskTensor_;
    LocalTensor<uint32_t> gatherMaskInt32Tensor_;
    LocalTensor<int32_t> expertTokenCntTensor_;
    LocalTensor<int32_t> validTopkIndexTensor_;
    LocalTensor<int32_t> cumsumInfoTensor_;
    LocalTensor<QuantOutType> copyTmpTensor1_;
    LocalTensor<QuantOutType> copyTmpTensor2_;
    LocalTensor<int32_t> tripleTensor_;
    LocalTensor<XType> xInTensor1_;
    LocalTensor<XType> xInTensor2_;
    LocalTensor<QuantOutType> xOutTensor1_;
    LocalTensor<QuantOutType> xOutTensor2_;
    LocalTensor<uint16_t> mxTempTensor_;
    LocalTensor<int32_t> resetTensor_;
    LocalTensor<int32_t> topkIdsTensor_;

    using BlockEpilogue = BlockEpilogueSwigluMxQuant<
        QuantOutType, float, QuantScaleOutType, QuantScaleOutType, true, IsInterleaved>;
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
        params_.workspaceInfo.flagSwiGluToGmm2Ptr, nullptr, nullptr, nullptr, ALIGN_128, ALIGN_256});
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
    uint32_t topkTensorAddr1 = gatherMaskTensorAddr + gatherMaskTensorSize;
    uint32_t topkTensorSize1 = Ops::Base::CeilAlign((int64_t)(sendTotalNum_ * sizeof(int32_t)), (int64_t)(ALIGN_256));
    topkTensor1_ = LocalTensor<int32_t>(TPosition::VECCALC, topkTensorAddr1, topkTensorSize1 / sizeof(int32_t));
    uint32_t topkTensorAddr2 = topkTensorAddr1 + topkTensorSize1;
    uint32_t topkTensorSize2 = topkTensorSize1;
    topkTensor2_ = LocalTensor<int32_t>(TPosition::VECCALC, topkTensorAddr2, topkTensorSize2 / sizeof(int32_t));
    uint32_t validTopkIndexTensorAddr = topkTensorAddr2 + topkTensorSize2;
    uint32_t validTopkIndexTensorSize = Ops::Base::CeilAlign(
        (int64_t)(sendTotalNum_ * sizeof(int32_t)), (int64_t)(ALIGN_32));
    validTopkIndexTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, validTopkIndexTensorAddr,
        validTopkIndexTensorSize / sizeof(int32_t));
    uint32_t copyTmpTensor1Addr = validTopkIndexTensorAddr + validTopkIndexTensorSize;
    uint32_t copyTmpTensor1Size = 8 * 1024;
    copyTmpTensor1_ = LocalTensor<QuantOutType>(TPosition::VECCALC, copyTmpTensor1Addr,
        copyTmpTensor1Size / sizeof(QuantOutType));
    uint32_t copyTmpTensor2Addr = copyTmpTensor1Addr + copyTmpTensor1Size;
    uint32_t copyTmpTensor2Size = 8 * 1024;
    copyTmpTensor2_ = LocalTensor<QuantOutType>(TPosition::VECCALC, copyTmpTensor2Addr,
        copyTmpTensor2Size / sizeof(QuantOutType));
    ubBufferUsedAddr_ = copyTmpTensor2Addr + copyTmpTensor2Size;
    CreateVecIndex(topkIndexTensor_, 0, (topkIndexTensorSize / sizeof(int32_t)));
    Duplicate<int32_t>(cumsumInfoTensor_, 0, (cumsumInfoTensorSize / sizeof(int32_t)));
    Duplicate<int32_t>(topkTensor1_, 0, (topkTensorSize1 / sizeof(int32_t)));
    Duplicate<int32_t>(topkTensor2_, 0, (topkTensorSize2 / sizeof(int32_t)));
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
        WaitForVector();
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
    totalSendCntInExp_ = 0;
    uint32_t mask = sendTotalNum_;
    int32_t compareExpertId = localExpertId + expertPerRank_ * rankId_;
    GlobalTensor<int32_t> srcGlobalTensor;
    DataCopyExtParams dataCopyParams{1U, static_cast<uint32_t>(sendTotalNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadParams{true, 0U,
        uint8_t(Ops::Base::CeilAlign((int64_t)(sendTotalNum_), (int64_t)(8)) - sendTotalNum_), 0U}; // 填充数据为0
    Duplicate<uint32_t>(gatherMaskInt32Tensor_, 0, (worldSize_ * maskAlignSize_) / sizeof(uint32_t)); // mask位置清理
    SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID1);
    for (int32_t calRankId = 0; calRankId < worldSize_; calRankId++) {
        uint64_t sendCnt = 0;
        uint64_t offsetInDstWinAddr = PEERMEM_DATA_OFFSET + calRankId * Ops::Base::CeilAlign(
        (int64_t)(sendTotalNum_ * sizeof(int32_t)), (int64_t)ALIGN_512);
        srcGlobalTensor.SetGlobalBuffer((__gm__ int32_t*)GetRankWinAddrWithOffset(calRankId, offsetInDstWinAddr));
        auto event = (calRankId % 2 == 0) ? EVENT_ID0 : EVENT_ID1;
        auto topkTensor = (calRankId % 2 == 0) ? topkTensor1_ : topkTensor2_;
        WaitFlag<AscendC::HardEvent::S_MTE2>(event);
        DataCopyPad(topkTensor, srcGlobalTensor, dataCopyParams, copyPadParams);
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        CompareScalar(gatherMaskTensor_[calRankId * maskAlignSize_], topkTensor, compareExpertId,
        AscendC::CMPMODE::EQ, compareCount_); // mask位置直接存下来
        GatherMask(validTopkIndexTensor_, topkIndexTensor_,
        gatherMaskInt32Tensor_[calRankId * (maskAlignSize_ / sizeof(uint32_t))], true, mask, {1, 1, 0, 0}, sendCnt);
        SetFlag<AscendC::HardEvent::V_S>(event);
        WaitFlag<AscendC::HardEvent::V_S>(event);
        totalSendCntInExp_ += sendCnt;
        cumsumRevCntInRank_ += sendCnt;
        cumsumInfoTensor_.SetValue(localExpertId * worldSize_ + calRankId, (int32_t)cumsumRevCntInRank_); // 先专家 - 再rank
        SetFlag<AscendC::HardEvent::S_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID1);
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
        LocalTensor<QuantOutType> tokenScalebuf = pingpongId == 0 ? copyTmpTensor1_ : copyTmpTensor2_;
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
    int32_t pingpongId = 0;
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
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

    // wave-粒度 dispatch flag：本核在 dispatch loop 中按 wave 边界批量 AtomicAdd。
    // 槽位布局：groupFlagListGm2_ 指向当前 expert 的 wave-0 起点，+waveIdx 即对应槽。
    // stripe 分核下 globalIdxInExpert = coreIdx + i_in_core * totalCores 是单调递增的，
    // 所以 waveIdx_token = globalIdxInExpert / L1_TILE_M 也是单调递增的，边界检测 O(1)。
    int32_t curWave = -1;
    int32_t curWaveCnt = 0;
    constexpr int32_t L1_TILE_M_I32 = static_cast<int32_t>(MegaMoeImpl::L1_TILE_M);

    if (tokensThisCore > 0) {
        // 二分查找：在当前专家内，找到首个token (expert内全局idx = coreIdx) 所在的卡号
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

        constexpr int32_t BufferNum = 2;
        int64_t widthA = Get<K_VALUE>(problemShape_);
        int64_t widthAScale = Ops::Base::CeilDiv((int64_t)widthA, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        uint32_t copyInNum = Ops::Base::CeilAlign((int64_t)(mxFp8TokenAlignBytes_ + mxFp8ScaleAlignBytes_), (int64_t)ALIGN_32);

        // 目的地址基址按照当前专家在dispatchRevDataPtr中的起始位置
        GlobalTensor<QuantOutType> tokenRevGlobalTensor;
        GlobalTensor<QuantScaleOutType> scaleRevGlobalTensor;
        tokenRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(
            params_.workspaceInfo.dispatchRevDataPtr + priorExpertCumsum * widthA));
        scaleRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleOutType*>(
            params_.workspaceInfo.dispatchRevScalePtr + priorExpertCumsum * widthAScale));

        int32_t curRank = startRank;
        int32_t curRankExprStart = (startRank == 0) ? 0 :
            (cumsumInfoTensor_.GetValue(localExpertId * worldSize_ + startRank - 1) - priorExpertCumsum);
        int32_t curRankExprEnd = cumsumInfoTensor_.GetValue(localExpertId * worldSize_ + startRank) - priorExpertCumsum;

        uint64_t sendCntCurRank = 0;
        GatherMask(validTopkIndexTensor_, topkIndexTensor_,
            gatherMaskInt32Tensor_[curRank * (maskAlignSize_ / sizeof(uint32_t))],
            true, sendTotalNum_, {1, 1, 0, 0}, sendCntCurRank);
        SyncFuncStatic<AscendC::HardEvent::V_S, 1>();

        GlobalTensor<QuantOutType> remoteRankGlobalTensor;
        remoteRankGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(
            GetRankWinAddrWithOffset(curRank, PEERMEM_DATA_OFFSET + worldSize_ * topkOffsetPerRank_)));

        for (int32_t i_in_core = 0; i_in_core < tokensThisCore; ++i_in_core) {
            int32_t globalIdxInExpert = coreIdx + i_in_core * totalCores;

            // wave 边界检测：当前 token 属于的 wave 与 curWave 不同时，
            // 先 drain MTE3 把前一个 wave 全部 dispatch 写入 GM 可见，再 AtomicAdd 前一个 wave 的累积计数。
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
                remoteRankGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(
                    GetRankWinAddrWithOffset(curRank, PEERMEM_DATA_OFFSET + worldSize_ * topkOffsetPerRank_)));
            }

            int32_t withinRankIdx = globalIdxInExpert - curRankExprStart;
            int32_t topkIndex = validTopkIndexTensor_.GetValue(withinRankIdx);
            int32_t tokenIndex = topkIndex / topK_;

            tripleTensor_[i_in_core * 8].SetValue(0, curRank);
            tripleTensor_[i_in_core * 8].SetValue(1, tokenIndex);
            tripleTensor_[i_in_core * 8].SetValue(2, topkIndex % topK_);

            pingpongId = (pingpongId + 1) % BufferNum;
            TEventID EVENT_ID_CUR = (pingpongId == 0) ? EVENT_ID0 : EVENT_ID1;
            LocalTensor<QuantOutType> tokenScalebuf = (pingpongId == 0) ? copyTmpTensor1_ : copyTmpTensor2_;
            LocalTensor<QuantScaleOutType> bufScale =
                tokenScalebuf[mxFp8TokenAlignBytes_].template ReinterpretCast<QuantScaleOutType>();

            auto remoteCopyOffset = tokenIndex * copyInNum;

            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID_CUR);
            DataCopy(tokenScalebuf, remoteRankGlobalTensor[remoteCopyOffset], copyInNum);
            SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID_CUR);
            WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID_CUR);

            // stripe 分核：核 c 的第 i_in_core 个 token 写到专家内偏移 coreIdx + i_in_core * totalCores
            int32_t dstOffsetInExpert = globalIdxInExpert;
            DataCopyPad(tokenRevGlobalTensor[dstOffsetInExpert * widthA], tokenScalebuf,
                {1, static_cast<uint16_t>(widthA * sizeof(QuantOutType)), 0U, 0U, 0U});
            DataCopyPad(scaleRevGlobalTensor[dstOffsetInExpert * widthAScale], bufScale,
                {1, static_cast<uint16_t>(widthAScale * sizeof(QuantScaleOutType)), 0U, 0U, 0U});
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID_CUR);

            ++curWaveCnt;  // 本 token 已记账到 curWave，在下一个 wave 边界 / loop 末尾统一 AtomicAdd
        }

        // 三元组按 stripe 排布搬出（与数据同顺序）。和 dispatch flag 通路独立：cube 不读 triple。
        SyncFuncStatic<AscendC::HardEvent::S_MTE3, 2>();
        for (int32_t i = 0; i < tokensThisCore; ++i) {
            int32_t dstAbsIdx = priorExpertCumsum + coreIdx + i * totalCores;
            DataCopy(tripleGloablTensor_[dstAbsIdx * 8], tripleTensor_[i * 8], 8);
        }
    }

    // flush 最后一个 wave 累积计数（tokensThisCore==0 时 curWaveCnt 始终为 0，自动跳过）
    if (curWaveCnt > 0) {
        SyncFuncStatic<AscendC::HardEvent::MTE3_S, 0>();
        AtomicAdd((__gm__ int32_t*)(groupFlagListGm2_.GetPhyAddr()) + curWave,
                  int32_t(curWaveCnt));
    }

    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

// template <TemplateMegaMoeTypeClass>
// __aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::TripleInfoCalAndDispatch(
//     GMMAddrInfo &gmmAddrInfo, int32_t localExpertId)
// {
//     int32_t pingpongId = 0;
//     SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
//     SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
//     PipeBarrier<PIPE_ALL>();
//     for (uint32_t idx = blockIdx_; idx < worldSize_ * blockNumPerRank_; idx += blockNum_) {
//         // 相比对原来流程需要添加本卡的cnt对应到原始卡上的哪个bs
//         uint32_t dstRankIdx = idx / blockNumPerRank_;
//         uint64_t sendToCurExpTokenCnt = 0;
//         GatherMask(validTopkIndexTensor_, topkIndexTensor_,
//             gatherMaskInt32Tensor_[dstRankIdx * (maskAlignSize_ / sizeof(uint32_t))],
//             true, sendTotalNum_, {1, 1, 0, 0}, sendToCurExpTokenCnt);
//         uint32_t aivIdxInGroup = idx % blockNumPerRank_;
//         uint32_t rowStartIdxInDst = ((dstRankIdx == 0 && localExpertId == 0) ? 0 :
//             cumsumInfoTensor_.GetValue(localExpertId * worldSize_ + dstRankIdx - 1));  // 填到本端时的目的地址起始值
//         if (rowStartIdxInDst < maxOutputSize_) {
//             int32_t rowsCopyCnt = sendToCurExpTokenCnt;
//             if (rowStartIdxInDst + rowsCopyCnt > maxOutputSize_) {
//                 rowsCopyCnt = maxOutputSize_ - rowStartIdxInDst;
//             }
//             int32_t rowsNumInCore = Ops::Base::CeilDiv(rowsCopyCnt, static_cast<int32_t>(blockNumPerRank_));
//             int32_t rowSrcIdxInCore = aivIdxInGroup * rowsNumInCore;
//             int32_t rowDstOffsetInCore = rowStartIdxInDst + rowSrcIdxInCore;
//             if (rowDstOffsetInCore + rowsNumInCore > rowStartIdxInDst + rowsCopyCnt) {
//                 rowsNumInCore = rowStartIdxInDst + rowsCopyCnt - rowDstOffsetInCore;
//             }
//             if (rowsNumInCore > 0) {
//                 uint32_t tripleTensorAddr = ubBufferUsedAddr_;
//                 uint32_t tripleTensorSize = rowsNumInCore * ALIGN_32;
//                 tripleTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, tripleTensorAddr,
//                     tripleTensorSize / sizeof(int32_t));
//                 SyncFunc<AscendC::HardEvent::V_S>();
//                 CopyGMToGMPerToken(rowDstOffsetInCore, dstRankIdx, rowSrcIdxInCore, rowsNumInCore, pingpongId);
//             }
//         }
//         SyncFunc<AscendC::HardEvent::MTE3_S>();
//         AtomicAdd((__gm__ int32_t*)(groupFlagListGm2_.GetPhyAddr()), int32_t(1));
//     }
//     WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0); 
//     WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
// }

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
            SendInfoCal(0);
            groupFlagListGm2_.SetGlobalBuffer(
                (__gm__ int32_t*)params_.workspaceInfo.flagDispatchToGmm1Ptr);
            MegaMoeTrace::RecordDispatchPoint(params_.y2GmAddr, 0, false);
            TripleInfoCalAndDispatch(static_cast<int32_t>(totalSendCntInExp_), 0);
            MegaMoeTrace::RecordDispatchPoint(params_.y2GmAddr, 0, true);
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
                SendInfoCal(expertIdx + 1U);
                groupFlagListGm2_.SetGlobalBuffer(
                    (__gm__ int32_t*)params_.workspaceInfo.flagDispatchToGmm1Ptr +
                    static_cast<int64_t>(expertIdx + 1U) * static_cast<int64_t>(maxWavesPerExpert_));
                MegaMoeTrace::RecordDispatchPoint(params_.y2GmAddr,
                                                  static_cast<int>(expertIdx + 1U), false);
                TripleInfoCalAndDispatch(static_cast<int32_t>(totalSendCntInExp_),
                                         static_cast<int32_t>(expertIdx + 1U));
                MegaMoeTrace::RecordDispatchPoint(params_.y2GmAddr,
                                                  static_cast<int>(expertIdx + 1U), true);
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
                        QuantOutType, QuantOutType, float, QuantScaleOutType, QuantScaleOutType, IsInterleaved>(
                        epilogueOp_, params_, problemShape_, waveGmmInfo, coreInGroup_, coresPerWave_,
                        vecSetSyncCom_, curWaveTrace);
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