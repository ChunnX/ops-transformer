/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* MegaMoeV2: wave-based dispatch+GMM1+GMM2+combine single-kernel pipeline (DESIGN §3.3) */

#ifndef MEGA_MOE_V2_H
#define MEGA_MOE_V2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"               // AscendC::GetL0DataType for blaze
#include "lib/std/tuple.h"                  // AscendC::Std for blaze

// tensor_api / blaze MUST come before V2 own headers (which do file-scope
// `using namespace AscendC`). Otherwise c_api/sync.h's unqualified
// SYNC_MODE_SHIFT_VALUE becomes ambiguous against AscendC::SYNC_MODE_SHIFT_VALUE.
// tensor_api declares AscendC::Te; we add namespace aliases for Te/Std at the
// global scope so blaze's unqualified references resolve, without the broad
// `using namespace AscendC` that re-introduces the ambiguity.
#include "include/tensor_api/tensor.h"
namespace Std = AscendC::Std;
namespace Te = AscendC::Te;
#include "blaze/block/block_mmad_mx.h"
#include "blaze/block/block_scheduler_swizzle.h"

#include "mega_moe_v2_base.h"
#include "mega_moe_v2_workspace_info.h"
#include "mega_moe_v2_block_epilogue_swiglu_mx_quant.h"
#include "mega_moe_v2_helpers.h"
#include "mega_moe_v2_wave_scheduler.h"
#include "mega_moe_v2_impl.h"

using namespace AscendC;

namespace MegaMoeV2Impl {

#define TemplateMegaMoeV2TypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, int32_t QuantMode
#define TemplateMegaMoeV2TypeFunc XType, OutputType, TopkWeightsType, QuantMode

template <TemplateMegaMoeV2TypeClass>
class MegaMoeV2 {
public:
    using QuantOutType = typename std::conditional<((QuantMode >= 3) && (QuantMode != 4)),
        fp8_e5m2_t, fp8_e4m3fn_t>::type;
    using QuantScaleOutType = typename std::conditional<(QuantMode >= 3), fp8_e8m0_t, float>::type;
    using BlockEpilogue =
        BlockEpilogueSwigluMxQuant<QuantOutType, float, QuantScaleOutType, QuantScaleOutType, true>;

    __aicore__ inline MegaMoeV2() {}

    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights,
        GM_ADDR weight1, GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
        GM_ADDR scales, GM_ADDR yOut, GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM,
        MegaMoeV2TilingData *tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void FirstBuffInit(TPipe *tPipe);
    __aicore__ inline void SecondBuffInit(TPipe *tPipe);
    __aicore__ inline void ResetWorkspaceFlags();
    __aicore__ inline void DispatchPullLoop();          // vec1: per-chunk pull + AtomicAdd dispatchArrival
    __aicore__ inline void EpilogueWaveLoop();          // vec0: wait cube fixpipe, run epilogue, signal mask/combine
    __aicore__ inline void ScheduleWaveLoop();          // AIC: persistent wave scheduler driving GMM1/GMM2
    __aicore__ inline void CombineWaveLoop();           // vec1: drain gmm2ToCombineFlag, push combine
    __aicore__ inline void ArgSortExpandedRowIdx();
    __aicore__ inline void ResetTokenPerExpert();
    __aicore__ inline void CrossRankSyncInWorldSize();
    __aicore__ inline void ExpertTokenNumCopyOut();

    __aicore__ inline int64_t TokenPerExpertLayout(int64_t dim0, int64_t dim1, int64_t dim2)
    {
        return dim0 * perRankStridesInTokenPerExpert_ + dim1 * expertPerRank_ + dim2;
    }

    // sync helpers (kept identical to v1 names for code reuse readability)
    __aicore__ inline void GmSignalWaitBarrier(__gm__ int32_t *sig_addr, int32_t cmp_val)
    {
        do {
            if (ReadGmByPassDCache(sig_addr) == cmp_val) return;
        } while (true);
    }

    // ---- TBuf pool (vec only) ----
    TBuf<> CrossRankSyncBuf_;
    TBuf<> PrevSumBuf_;
    TBuf<> CopyTempBuf_;
    TBuf<> CumsumBuf_;
    TBuf<> ResetFlagBuf_;
    TBuf<> DataResBuf_;
    TBuf<> DataResFp32Buf_;
    TBuf<> ResetTokenBuf_;
    TBuf<> topKWeightsBuf_;

    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    Params params_{};
    GM_ADDR tokenPerExpertCurRankAddr_{nullptr};

    GlobalTensor<int32_t> preSumBeforeRankWorkSpace_;
    GlobalTensor<int32_t> tokenPerExpertWin_;
    GlobalTensor<int32_t> cumsumInWorkSpace_;
    GlobalTensor<int32_t> dispatchArrivalGm_;       // GM uint32 atomic counters
    GlobalTensor<int32_t> gmm2ToCombineGm_;         // GM uint32 atomic counters
    GlobalTensor<int32_t> expertTokenNumsOut_;
    GlobalTensor<int32_t> groupList_;
    LocalTensor<float> topKWeightsTensor_;

    uint32_t m_{0};
    uint32_t k_{0};
    uint32_t topK_{0};
    uint32_t rankId_{0};
    uint32_t worldSize_{0};
    uint32_t expertPerRank_{0};
    uint32_t groupListType_{0};
    uint32_t kExpertsPerWave_{0};
    uint32_t kNumWaves_{0};
    uint32_t chunkDepth_{0};
    uint32_t chunkM_{0};
    uint32_t numChunksPerWave_{0};
    uint32_t commVecCount_{0};
    uint32_t epiVecCount_{0};
    uint32_t dispatchVecPerRank_{0};
    int32_t  dispatchRows_{0};
    int32_t  prevSum_{0};
    int32_t  perRankStridesInTokenPerExpert_{0};

    uint32_t blockNum_{static_cast<uint32_t>(GetBlockNum())};
    uint32_t blockAivNum_{static_cast<uint32_t>(GetBlockNum() * 2)};
    uint32_t blockIdx_{static_cast<uint32_t>(GetBlockIdx() / GetTaskRation())};
    uint32_t aivCoreIdx_{static_cast<uint32_t>(GetBlockIdx())};
    uint32_t subBlockIdx_{static_cast<uint32_t>(GetSubBlockIdx())};

    BlockEpilogue epilogueOp_;
    WaveScheduler scheduler_;
};

template <TemplateMegaMoeV2TypeClass>
__aicore__ inline void MegaMoeV2<TemplateMegaMoeV2TypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales, GM_ADDR yOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeV2TilingData *tilingData)
{
    m_ = tilingData->m;
    k_ = tilingData->k;
    topK_ = tilingData->topK;
    worldSize_ = tilingData->epWorldSize;
    expertPerRank_ = tilingData->expertPerRank;
    dispatchRows_ = tilingData->dispatchRows;
    groupListType_ = tilingData->groupListType;
    kExpertsPerWave_ = tilingData->kExpertsPerWave;
    kNumWaves_ = tilingData->kNumWaves;
    chunkDepth_ = tilingData->chunkDepth;
    chunkM_ = tilingData->chunkM;
    numChunksPerWave_ = tilingData->numChunksPerWave;
    commVecCount_ = tilingData->commVecCount;
    epiVecCount_ = tilingData->epiVecCount;
    dispatchVecPerRank_ = tilingData->dispatchVecPerRank;

    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext*>(context);
    rankId_ = mc2Context_->epRankId;
    for (uint32_t i = 0; i < worldSize_; i++) {
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

    preSumBeforeRankWorkSpace_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrSumBeforeRank);
    tokenPerExpertWin_.SetGlobalBuffer((__gm__ int32_t*)params_.peermemInfo.ptrTokenPerExpert);
    cumsumInWorkSpace_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrcumsumMM);
    dispatchArrivalGm_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrDispatchArrival);
    gmm2ToCombineGm_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrGmm2ToCombine);
    expertTokenNumsOut_.SetGlobalBuffer((__gm__ int32_t*)params_.expertTokenNumsOutGmAddr);
    groupList_ = cumsumInWorkSpace_[(worldSize_ - 1) * expertPerRank_];

    epilogueOp_.Init({
        params_.workspaceInfo.ptrA2, params_.workspaceInfo.ptrA2Scale,
        params_.workspaceInfo.ptrL2ArrivalMask,   // epilogue AtomicAdd target = mask buffer (used as flag)
        nullptr, nullptr, nullptr, ALIGN_128, ALIGN_256});
    perRankStridesInTokenPerExpert_ = Ops::Base::CeilAlign(
        static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128));
    tokenPerExpertCurRankAddr_ = params_.peermemInfo.ptrTokenPerExpert +
        TokenPerExpertLayout(rankId_, 0, 0) * sizeof(int32_t);

    scheduler_.Init(tilingData, params_.workspaceInfo.ptrcumsumMM);
}

template <TemplateMegaMoeV2TypeClass>
__aicore__ inline void MegaMoeV2<TemplateMegaMoeV2TypeFunc>::FirstBuffInit(TPipe *tPipe)
{
    uint32_t syncBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128)) * sizeof(int32_t);
    uint32_t prevSumBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128))
        * sizeof(int32_t) * expertPerRank_;
    tPipe->InitBuffer(CrossRankSyncBuf_, syncBufAlign);
    tPipe->InitBuffer(PrevSumBuf_, prevSumBufAlign);
    uint32_t getCumsumBufAlign = (Ops::Base::CeilAlign(
        worldSize_ * expertPerRank_, (uint32_t)ALIGN_128)) * worldSize_ * sizeof(int32_t);
    tPipe->InitBuffer(CumsumBuf_, getCumsumBufAlign);
    tPipe->InitBuffer(CopyTempBuf_, BUFFER_ALIGN * sizeof(QuantOutType));
    uint32_t maxResetBufAlign = Ops::Base::CeilDiv(
        static_cast<int32_t>(expertPerRank_ * INT_CACHELINE * sizeof(int32_t) * 2),
        static_cast<int32_t>(ALIGN_32));
    tPipe->InitBuffer(ResetFlagBuf_, maxResetBufAlign);
}

template <TemplateMegaMoeV2TypeClass>
__aicore__ inline void MegaMoeV2<TemplateMegaMoeV2TypeFunc>::SecondBuffInit(TPipe *tPipe)
{
    uint32_t dataResBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(UNPERMUTE_LIST_NUM * k_ * sizeof(bfloat16_t)), static_cast<uint32_t>(ALIGN_32));
    tPipe->InitBuffer(DataResBuf_, dataResBufAlign);
    int32_t num = worldSize_ * Ops::Base::CeilAlign(
            static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128)) * sizeof(int32_t);
    tPipe->InitBuffer(ResetTokenBuf_, num);
    tPipe->InitBuffer(DataResFp32Buf_, dataResBufAlign * 2);
    tPipe->InitBuffer(topKWeightsBuf_, Ops::Base::CeilAlign(
        static_cast<uint32_t>(m_ * topK_ * sizeof(float)), static_cast<uint32_t>(ALIGN_32)));
    topKWeightsTensor_ = topKWeightsBuf_.template Get<float>();
    if constexpr (Std::IsSame<TopkWeightsType, float>::value) {
        GlobalTensor<float> topKWeightsGlobalTensor;
        topKWeightsGlobalTensor.SetGlobalBuffer((__gm__ float*)params_.probsGmAddr);
        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(m_ * topK_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(topKWeightsTensor_, topKWeightsGlobalTensor, copyParams, copyPadParams);
    }
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        TBuf<> tempBuf;
        tPipe->InitBuffer(tempBuf, Ops::Base::CeilAlign(
            static_cast<uint32_t>(m_ * topK_ * sizeof(bfloat16_t)), uint32_t(ALIGN_32)));
        LocalTensor<bfloat16_t> tempLocal = tempBuf.Get<bfloat16_t>();
        GlobalTensor<bfloat16_t> topkWeightsGlobalTensor;
        topkWeightsGlobalTensor.SetGlobalBuffer((__gm__ bfloat16_t*)params_.probsGmAddr);
        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(m_ * topK_ * sizeof(bfloat16_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<bfloat16_t> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(tempLocal, topkWeightsGlobalTensor, copyParams, copyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Cast(topKWeightsTensor_, tempLocal, AscendC::RoundMode::CAST_NONE, m_ * topK_);
    }
}

// reset all GM stage flags (dispatchArrival, l2ArrivalMask, gmm2ToCombine)
template <TemplateMegaMoeV2TypeClass>
__aicore__ inline void MegaMoeV2<TemplateMegaMoeV2TypeFunc>::ResetWorkspaceFlags()
{
    if constexpr(g_coreType != AIV) return;
    LocalTensor<int32_t> initTensor = ResetFlagBuf_.Get<int32_t>();
    uint32_t totalFlags = params_.tilingData->dispatchArrivalLen +
                          params_.tilingData->l2ArrivalMaskLen * 2 +  // uint64 = 2*uint32
                          params_.tilingData->gmm2ToCombineLen;
    int32_t coreNum = blockAivNum_;
    int32_t lenPerCore = Ops::Base::CeilDiv(totalFlags, static_cast<uint32_t>(coreNum));
    int32_t lenPerCoreAlign = Ops::Base::CeilAlign(lenPerCore, (int32_t)ALIGN_32);
    int32_t coreOffset = aivCoreIdx_ * lenPerCoreAlign;
    int32_t coreLen = lenPerCoreAlign;
    if (coreOffset + coreLen > (int32_t)totalFlags) coreLen = totalFlags - coreOffset;
    if (coreOffset >= (int32_t)totalFlags) coreLen = 0;
    if (coreLen > 0) {
        Duplicate(initTensor, 0, coreLen);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        // write into dispatchArrival region (linear memory; downstream regions follow contiguously)
        GlobalTensor<int32_t> dst;
        dst.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrDispatchArrival);
        DataCopy(dst[coreOffset], initTensor, coreLen);
    }
}

// vec1: full dispatch lifecycle. Loop over local experts; for each expert, pull tokens
// in chunkDepth-sized batches, AtomicAdd dispatchArrival[chunkId] when each batch ready.
template <TemplateMegaMoeV2TypeClass>
__aicore__ inline void MegaMoeV2<TemplateMegaMoeV2TypeFunc>::DispatchPullLoop()
{
    if (subBlockIdx_ != 1) return;
    int64_t widthA = k_;
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; ++expertIdx) {
        // chunkId encodes (waveIdx, expertOffsetInWave, chunkInExpert) — flatten linearly
        uint32_t waveIdx = expertIdx / kExpertsPerWave_;
        uint32_t exprOffInWave = expertIdx % kExpertsPerWave_;
        uint32_t chunkBase = waveIdx * numChunksPerWave_ + exprOffInWave * chunkDepth_;

        MegaMoeV2Helpers::DispatchPullChunk<QuantOutType, QuantScaleOutType>(
            CopyTempBuf_,
            params_.workspaceInfo.ptrA, params_.workspaceInfo.ptrAScale,
            tokenPerExpertWin_, cumsumInWorkSpace_, dispatchArrivalGm_,
            expertIdx, rankId_, worldSize_, expertPerRank_,
            blockIdx_, blockNum_, dispatchVecPerRank_, dispatchRows_,
            prevSum_, widthA, chunkBase, params_.tilingData->maxOutputSize,
            perRankStridesInTokenPerExpert_);
    }
}

// AIC: persistent wave loop. For each wave: drain Linear1 (all blocks), then drain Linear2.
template <TemplateMegaMoeV2TypeClass>
__aicore__ inline void MegaMoeV2<TemplateMegaMoeV2TypeFunc>::ScheduleWaveLoop()
{
    if constexpr(g_coreType != AIC) return;
    uint32_t intermediate = params_.tilingData->n / SWIGLU_N_HALF;
    uint32_t hidden = k_;

    GMMAddrInfo gmmInfo1{};
    GMMAddrInfo gmmInfo2{};
    __gm__ int32_t* dispArrAddr = (__gm__ int32_t*)params_.workspaceInfo.ptrDispatchArrival;
    __gm__ uint64_t* l2MaskAddr = (__gm__ uint64_t*)params_.workspaceInfo.ptrL2ArrivalMask;
    __gm__ int32_t* gmm2CombAddr = (__gm__ int32_t*)params_.workspaceInfo.ptrGmm2ToCombine;

    // dispatch target = numbers of vec1 cores per chunk (commVecCount distributed across worldSize)
    int32_t targetDispatchCount =
        static_cast<int32_t>(worldSize_ * dispatchVecPerRank_);

    for (uint32_t waveIdx = 0; waveIdx < kNumWaves_; ++waveIdx) {
        if (!scheduler_.BeginWave(waveIdx)) break;

        // ---- Linear1 phase ----
        {
            WaveBlockIter it1;
            scheduler_.BeginIter(BlockPhase::Linear1, it1);
            WaveBlock wb;
            while (scheduler_.NextBlock(it1, wb)) {
                uint32_t expertM = scheduler_.GetExpertTokenCount(wb.expertIdx);
                uint32_t expertN_l1 = intermediate;
                uint32_t expertK_l1 = hidden;
                uint32_t mLoc = wb.mBlockIdx * L1_TILE_M;
                uint32_t nLoc = wb.nBlockIdx * L1_TILE_N;
                uint32_t actualM = (mLoc + L1_TILE_M <= expertM) ? L1_TILE_M : (expertM - mLoc);
                uint32_t actualN = (nLoc + L1_TILE_N <= expertN_l1) ? L1_TILE_N : (expertN_l1 - nLoc);
                uint32_t actualK = expertK_l1;

                // GM offsets into ring buffer for this expert/chunk
                uint32_t chunkId = scheduler_.GetChunkId(waveIdx, wb.poolBlockOffset);
                uint32_t mBlockGlobal = scheduler_.GetGlobalMBlockId(waveIdx, wb.poolBlockOffset);
                uint32_t ringSlot = chunkId % chunkDepth_;
                uint64_t aRingOff = (uint64_t)ringSlot * chunkM_ * hidden;
                uint64_t aScaleRingOff = (uint64_t)ringSlot * chunkM_ * (hidden / MXFP_SCALE_GROUP_NUM);
                uint64_t bExpertOff = (uint64_t)wb.expertIdx * intermediate * hidden;          // SWIGLU_N_HALF folded into outer loop
                uint64_t bScaleExpertOff = (uint64_t)wb.expertIdx * intermediate *
                                           Ops::Base::CeilDiv(hidden, (uint32_t)MXFP_DIVISOR_SIZE) *
                                           MXFP_MULTI_BASE_SIZE;
                gmmInfo1.aGlobal = params_.workspaceInfo.ptrA + aRingOff * sizeof(QuantOutType);
                gmmInfo1.aScaleGlobal = params_.workspaceInfo.ptrAScale + aScaleRingOff * sizeof(QuantScaleOutType);
                gmmInfo1.bGlobal = params_.bGmAddr + bExpertOff * sizeof(QuantOutType) * SWIGLU_N_HALF;
                gmmInfo1.bScaleGlobal = params_.bScaleGmAddr + bScaleExpertOff * sizeof(QuantScaleOutType) * SWIGLU_N_HALF;
                gmmInfo1.groupFlagList = (__gm__ int32_t*)params_.workspaceInfo.ptrL2ArrivalMask + mBlockGlobal;
                gmmInfo1.groupFlagList2 = dispArrAddr + chunkId;

                uint16_t pingpongSlot = static_cast<uint16_t>(wb.poolBlockOffset % WAVE_PINGPONG_DEPTH);
                DoLinear1Block<QuantOutType, QuantOutType, float, QuantScaleOutType, QuantScaleOutType>(
                    epilogueOp_, params_, gmmInfo1,
                    expertM, expertN_l1, expertK_l1,
                    mLoc, nLoc, actualM, actualN, actualK,
                    chunkId, mBlockGlobal,
                    dispArrAddr, l2MaskAddr,
                    targetDispatchCount, pingpongSlot);
            }
        }

        // ---- Linear2 phase ----
        uint32_t numNBlocksL1 = scheduler_.GetNumNBlocks(BlockPhase::Linear1);
        {
            WaveBlockIter it2;
            scheduler_.BeginIter(BlockPhase::Linear2, it2);
            WaveBlock wb;
            while (scheduler_.NextBlock(it2, wb)) {
                uint32_t expertM = scheduler_.GetExpertTokenCount(wb.expertIdx);
                uint32_t expertN_l2 = hidden;
                uint32_t expertK_l2 = intermediate;
                uint32_t mLoc = wb.mBlockIdx * L1_TILE_M;
                uint32_t nLoc = wb.nBlockIdx * L1_TILE_N;
                uint32_t actualM = (mLoc + L1_TILE_M <= expertM) ? L1_TILE_M : (expertM - mLoc);
                uint32_t actualN = (nLoc + L1_TILE_N <= expertN_l2) ? L1_TILE_N : (expertN_l2 - nLoc);
                uint32_t actualK = expertK_l2;

                uint32_t chunkId = scheduler_.GetChunkId(waveIdx, wb.poolBlockOffset);
                uint32_t mBlockGlobal = scheduler_.GetGlobalMBlockId(waveIdx, wb.poolBlockOffset);
                uint32_t ringSlot = chunkId % chunkDepth_;
                uint64_t a2Off = (uint64_t)ringSlot * chunkM_ * intermediate;
                uint64_t a2ScaleOff = (uint64_t)ringSlot * chunkM_ * (intermediate / MXFP_SCALE_GROUP_NUM);
                uint64_t b2ExpertOff = (uint64_t)wb.expertIdx * hidden * intermediate;
                uint64_t b2ScaleExpertOff = (uint64_t)wb.expertIdx * hidden *
                                            Ops::Base::CeilDiv(intermediate, (uint32_t)MXFP_DIVISOR_SIZE) *
                                            MXFP_MULTI_BASE_SIZE;
                gmmInfo2.aGlobal = params_.workspaceInfo.ptrA2 + a2Off * sizeof(QuantOutType);
                gmmInfo2.aScaleGlobal = params_.workspaceInfo.ptrA2Scale + a2ScaleOff * sizeof(QuantScaleOutType);
                gmmInfo2.bGlobal = params_.b2GmAddr + b2ExpertOff * sizeof(QuantOutType);
                gmmInfo2.bScaleGlobal = params_.b2ScaleGmAddr + b2ScaleExpertOff * sizeof(QuantScaleOutType);
                gmmInfo2.groupFlagList = gmm2CombAddr + chunkId;
                gmmInfo2.groupFlagList2 = (__gm__ int32_t*)params_.workspaceInfo.ptrL2ArrivalMask + mBlockGlobal;

                uint16_t pingpongSlot = static_cast<uint16_t>(wb.poolBlockOffset % WAVE_PINGPONG_DEPTH);
                DoLinear2Block<QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
                    params_, gmmInfo2,
                    expertM, expertN_l2, expertK_l2,
                    mLoc, nLoc, actualM, actualN, actualK,
                    chunkId, mBlockGlobal, numNBlocksL1,
                    l2MaskAddr, gmm2CombAddr,
                    wb.expertIdx, rankId_, pingpongSlot);
            }
        }
    }
}

// vec0: epilogue side mirrors AIC scheduler — same iteration order, same sync slots.
template <TemplateMegaMoeV2TypeClass>
__aicore__ inline void MegaMoeV2<TemplateMegaMoeV2TypeFunc>::EpilogueWaveLoop()
{
    if constexpr(g_coreType != AIV) return;
    if (subBlockIdx_ != 0) return;
    uint32_t intermediate = params_.tilingData->n / SWIGLU_N_HALF;
    uint32_t hidden = k_;
    GMMAddrInfo gmmInfo1{}, gmmInfo2{};
    __gm__ int32_t* dispArrAddr = (__gm__ int32_t*)params_.workspaceInfo.ptrDispatchArrival;
    __gm__ uint64_t* l2MaskAddr = (__gm__ uint64_t*)params_.workspaceInfo.ptrL2ArrivalMask;
    __gm__ int32_t* gmm2CombAddr = (__gm__ int32_t*)params_.workspaceInfo.ptrGmm2ToCombine;
    int32_t targetDispatchCount = static_cast<int32_t>(worldSize_ * dispatchVecPerRank_);

    for (uint32_t waveIdx = 0; waveIdx < kNumWaves_; ++waveIdx) {
        if (!scheduler_.BeginWave(waveIdx)) break;

        // Linear1 epilogue
        {
            WaveBlockIter it1;
            scheduler_.BeginIter(BlockPhase::Linear1, it1);
            WaveBlock wb;
            while (scheduler_.NextBlock(it1, wb)) {
                uint32_t expertM = scheduler_.GetExpertTokenCount(wb.expertIdx);
                uint32_t expertN_l1 = intermediate;
                uint32_t mLoc = wb.mBlockIdx * L1_TILE_M;
                uint32_t nLoc = wb.nBlockIdx * L1_TILE_N;
                uint32_t actualM = (mLoc + L1_TILE_M <= expertM) ? L1_TILE_M : (expertM - mLoc);
                uint32_t actualN = (nLoc + L1_TILE_N <= expertN_l1) ? L1_TILE_N : (expertN_l1 - nLoc);

                uint32_t chunkId = scheduler_.GetChunkId(waveIdx, wb.poolBlockOffset);
                uint32_t mBlockGlobal = scheduler_.GetGlobalMBlockId(waveIdx, wb.poolBlockOffset);
                uint32_t ringSlot = chunkId % chunkDepth_;
                uint64_t a2Off = (uint64_t)ringSlot * chunkM_ * intermediate;
                uint64_t a2ScaleOff = (uint64_t)ringSlot * chunkM_ * (intermediate / MXFP_SCALE_GROUP_NUM);

                // refresh epilogue's GM addresses for this chunk slot
                AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                    (int64_t)a2Off, (int64_t)a2ScaleOff, (int64_t)mBlockGlobal, 0L, 0L, 0L};
                epilogueOp_.UpdateGlobalAddr(vecBaseOffset);
                epilogueOp_.UpdateNextProblem({(int64_t)expertM, (int64_t)expertN_l1, (int64_t)hidden, 0L});

                uint16_t pingpongSlot = static_cast<uint16_t>(wb.poolBlockOffset % WAVE_PINGPONG_DEPTH);
                DoLinear1Block<QuantOutType, QuantOutType, float, QuantScaleOutType, QuantScaleOutType>(
                    epilogueOp_, params_, gmmInfo1,
                    expertM, expertN_l1, hidden,
                    mLoc, nLoc, actualM, actualN, hidden,
                    chunkId, mBlockGlobal,
                    dispArrAddr, l2MaskAddr,
                    targetDispatchCount, pingpongSlot);
            }
        }

        // Linear2 epilogue (BF16 cast + combine)
        uint32_t numNBlocksL1 = scheduler_.GetNumNBlocks(BlockPhase::Linear1);
        {
            WaveBlockIter it2;
            scheduler_.BeginIter(BlockPhase::Linear2, it2);
            WaveBlock wb;
            while (scheduler_.NextBlock(it2, wb)) {
                uint32_t expertM = scheduler_.GetExpertTokenCount(wb.expertIdx);
                uint32_t expertN_l2 = hidden;
                uint32_t mLoc = wb.mBlockIdx * L1_TILE_M;
                uint32_t nLoc = wb.nBlockIdx * L1_TILE_N;
                uint32_t actualM = (mLoc + L1_TILE_M <= expertM) ? L1_TILE_M : (expertM - mLoc);
                uint32_t actualN = (nLoc + L1_TILE_N <= expertN_l2) ? L1_TILE_N : (expertN_l2 - nLoc);

                uint32_t chunkId = scheduler_.GetChunkId(waveIdx, wb.poolBlockOffset);
                uint32_t mBlockGlobal = scheduler_.GetGlobalMBlockId(waveIdx, wb.poolBlockOffset);
                uint16_t pingpongSlot = static_cast<uint16_t>(wb.poolBlockOffset % WAVE_PINGPONG_DEPTH);
                DoLinear2Block<QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
                    params_, gmmInfo2,
                    expertM, expertN_l2, intermediate,
                    mLoc, nLoc, actualM, actualN, intermediate,
                    chunkId, mBlockGlobal, numNBlocksL1,
                    l2MaskAddr, gmm2CombAddr,
                    wb.expertIdx, rankId_, pingpongSlot);
            }
        }
    }
}

// vec1: combine push side. After AIC's GMM2 epilogue lands in remote win, vec1 just spins on
// gmm2ToCombineFlag[chunk_id] -> drives Unpermute later. (Combine push DataCopyPad is done
// inside DoLinear2Block via CombineTokens — vec1 only needs to ensure ordering until Unpermute.)
template <TemplateMegaMoeV2TypeClass>
__aicore__ inline void MegaMoeV2<TemplateMegaMoeV2TypeFunc>::CombineWaveLoop()
{
    if (subBlockIdx_ != 1) return;
    // combine push body is performed by vec0 inside DoLinear2Block (CombineTokens).
    // vec1 awaits chunk completion to gate any per-chunk post-processing before Unpermute.
    __gm__ int32_t* gmm2CombAddr = (__gm__ int32_t*)params_.workspaceInfo.ptrGmm2ToCombine;
    uint32_t totalChunks = numChunksPerWave_ * kNumWaves_;
    uint32_t target = scheduler_.GetNumNBlocks(BlockPhase::Linear2); // each chunk produces this many bumps
    for (uint32_t chunkId = aivCoreIdx_; chunkId < totalChunks; chunkId += blockAivNum_) {
        do {
            int32_t v = ReadGmByPassDCache(gmm2CombAddr + chunkId);
            if (v >= (int32_t)target) break;
            int64_t st = GetSystemCycle();
            while (GetSystemCycle() - st < 100) {}
        } while (true);
    }
}

template <TemplateMegaMoeV2TypeClass>
__aicore__ inline void MegaMoeV2<TemplateMegaMoeV2TypeFunc>::ArgSortExpandedRowIdx()
{
    int32_t coreNum = blockAivNum_;
    int32_t lenPerCore = Ops::Base::CeilDiv(m_ * topK_, static_cast<uint32_t>(coreNum));
    int32_t coreOffset = aivCoreIdx_ * lenPerCore;
    int32_t coreLen = lenPerCore;
    if (coreOffset + coreLen >= (int32_t)(m_ * topK_)) coreLen = m_ * topK_ - coreOffset;
    if (coreOffset >= (int32_t)(m_ * topK_)) coreLen = 0;
    __gm__ int32_t *src = (__gm__ int32_t *)params_.workspaceInfo.expandedRowIdx;
    __gm__ int32_t *dst = (__gm__ int32_t *)params_.workspaceInfo.expandedRowIdxGather;
    Simt::VF_CALL<ArgsortSimt>(Simt::Dim3{MoeInitRoutingV3::SIMT_THREAD_NUM, 1, 1}, coreLen, coreOffset, src, dst);
}

template <TemplateMegaMoeV2TypeClass>
__aicore__ inline void MegaMoeV2<TemplateMegaMoeV2TypeFunc>::ResetTokenPerExpert()
{
    if (aivCoreIdx_ != blockAivNum_ - 1) return;
    int32_t num = worldSize_ * Ops::Base::CeilAlign(
            static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128));
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    LocalTensor<int32_t> tmp = ResetTokenBuf_.Get<int32_t>();
    Duplicate(tmp, 0, num);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(tokenPerExpertWin_, tmp, num);
}

template <TemplateMegaMoeV2TypeClass>
__aicore__ inline void MegaMoeV2<TemplateMegaMoeV2TypeFunc>::CrossRankSyncInWorldSize()
{
    __gm__ int32_t* syncRank = (__gm__ int32_t*)(params_.peermemInfo.ptrBase);
    __gm__ int32_t* syncCount = (__gm__ int32_t*)(params_.peermemInfo.ptrBase + 48 * 1024);
    int count = *(syncCount + aivCoreIdx_ * 16) + 1;
    for (uint32_t i = aivCoreIdx_; i < worldSize_; i += blockNum_) {
        __gm__ int32_t* syncRemoteAddr = (__gm__ int32_t*)(winRankAddr_[i]) + rankId_ * 16;
        WriteGmByPassDCache(syncRemoteAddr, count);
        auto syncCheck = syncRank + i * 16;
        GmSignalWaitBarrier(syncCheck, count);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
    *(syncCount + aivCoreIdx_ * 16) = count;
}

template <TemplateMegaMoeV2TypeClass>
__aicore__ inline void MegaMoeV2<TemplateMegaMoeV2TypeFunc>::ExpertTokenNumCopyOut()
{
    int32_t copyNum = Ops::Base::CeilAlign(
        static_cast<uint32_t>(expertPerRank_), static_cast<uint32_t>(ALIGN_32));
    LocalTensor<int32_t> tmpCopyBuffer = CrossRankSyncBuf_.Get<int32_t>();
    DataCopy(tmpCopyBuffer, groupList_, copyNum);
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    DataCopy(expertTokenNumsOut_, tmpCopyBuffer, copyNum);
}

template <TemplateMegaMoeV2TypeClass>
__aicore__ inline void MegaMoeV2<TemplateMegaMoeV2TypeFunc>::Process()
{
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();

    // 1. routing + per-token MX-quant -> peermem ptrA0 + workspace ring A
    moe_init_routing_v3<bfloat16_t, QuantOutType>(
        params_.aGmAddr, params_.expertIdxGmAddr, nullptr, nullptr,
        params_.peermemInfo.ptrA0, params_.workspaceInfo.expandedRowIdx,
        tokenPerExpertCurRankAddr_, nullptr,
        params_.workspaceInfo.ptrA, &params_.tilingData->moeInitRoutingTilingData,
        params_.tilingData->moeInitRoutingTilingData.tilingKey);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();

    // 2. buffer init + reset stage flags
    TPipe firstPipe;
    FirstBuffInit(&firstPipe);
    ResetWorkspaceFlags();

    // 3. cross-rank cnt sync (single global barrier)
    MegaMoeV2Helpers::CrossRankSyncCntPerExpert(
        CrossRankSyncBuf_, PrevSumBuf_, CumsumBuf_,
        tokenPerExpertWin_, preSumBeforeRankWorkSpace_, cumsumInWorkSpace_,
        rankId_, worldSize_, expertPerRank_, perRankStridesInTokenPerExpert_,
        tokenPerExpertCurRankAddr_, params_.peermemInfo.ptrBase,
        aivCoreIdx_, blockNum_, blockIdx_, dispatchVecPerRank_, prevSum_);

    if (subBlockIdx_ == 0 && g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
        firstPipe.Destroy();
    }
    SyncAll<false>();

    // 4. wave-based pipeline: AIC schedules; vec0 epilogues; vec1 dispatch+combine.
    if constexpr (g_coreType == AscendC::AIV) {
        if (subBlockIdx_ == 1) {
            DispatchPullLoop();
            CombineWaveLoop();
        } else {
            EpilogueWaveLoop();
        }
    } else { // AIC
        ScheduleWaveLoop();
    }
    SyncAll<true>();

    if constexpr (g_coreType == AscendC::AIV) {
        if (aivCoreIdx_ == 1) {
            ExpertTokenNumCopyOut();
        }
        if (subBlockIdx_ == 1) {
            PipeBarrier<PIPE_ALL>();
            firstPipe.Destroy();
        }
    }

    // 5. Unpermute (recv-side topk weighted sum) on vec
    if constexpr(g_coreType == AIV) {
        TPipe secondPipe;
        SecondBuffInit(&secondPipe);
        ArgSortExpandedRowIdx();
        ResetTokenPerExpert();
        CrossRankSyncInWorldSize();
        MegaMoeV2Helpers::Unpermute<TopkWeightsType>(
            DataResBuf_, DataResFp32Buf_, topKWeightsTensor_,
            params_.workspaceInfo.expandedRowIdxGather, params_.peermemInfo.ptrD,
            params_.y2GmAddr, m_, k_, aivCoreIdx_, blockAivNum_);
        PipeBarrier<PIPE_ALL>();
        secondPipe.Destroy();
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

} // namespace MegaMoeV2Impl

#endif // MEGA_MOE_V2_H
