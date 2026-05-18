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


using namespace AscendC;

namespace MegaMoeImpl {
using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
using BlockOffset = Shape<int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t>;

// IsInterleaved 默认 false 走老路径；为 true 时 GMM1 的右矩阵 B 是预先交织的
// （每 256 列 = 128 列 gate + 128 列 up），GMM1 内每 tile 只做 1 次 mmad。
// 默认值只放在类声明，成员函数定义用不带默认的 macro，避免 C++ "default argument redefinition" 报错。
#define TemplateMegaMoeTypeClass typename XType, typename OutputType, typename TopkWeightsType, \
                                 int32_t QuantMode, bool IsInterleaved
#define TemplateMegaMoeTypeFunc XType, OutputType, TopkWeightsType, QuantMode, IsInterleaved

template <typename XType, typename OutputType, typename TopkWeightsType, int32_t QuantMode,
          bool IsInterleaved = false>
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
    __aicore__ inline void FirstBuffInit(TPipe *tPipe);
    __aicore__ inline void SecondBuffInit(TPipe *tPipe);
    __aicore__ inline void ResetWorkSpaceFlagList();
    __aicore__ inline void ResetGmm2RowReadyFlags();
    __aicore__ inline void GetCumsumForMMAIV();
    __aicore__ inline void CrossRankSyncCntPerExpert();
    __aicore__ inline void ExpertTokenNumCopyOut();
    __aicore__ inline void DispatchTokens(GMMAddrInfo &gmmAddrInfo, uint32_t expertIdx);
    __aicore__ inline bool UpdateGroupParams(uint32_t expertIdx);
    __aicore__ inline void UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo, uint32_t gmmIndex, uint64_t waveIdx = 0);
    // Dispatch 流水专用：操作的是 dispatchBaseOffset_ / dispatchProblemShape_ / dispatchPreOffset_，
    // 不动 baseOffset_ / problemShape_ / preOffset_ 那条主线。
    __aicore__ inline bool UpdateGroupParamsForDispatch(uint32_t expertIdx);
    __aicore__ inline void UpdateGlobalBufferForDispatch(GMMAddrInfo &gmmAddrInfo, uint32_t expertIdx);
    __aicore__ inline void ArgSortExpandedRowIdx();
    __aicore__ inline void ResetTokenPerExpert();
    __aicore__ inline void Unpermute();
    __aicore__ inline void EndSync();
    __aicore__ inline void CrossRankSyncInWorldSize();
    __aicore__ inline void CopyGMToGMPerTokenV2(GlobalTensor<QuantOutType> dst,
                GlobalTensor<QuantScaleOutType> dstScale, GlobalTensor<QuantOutType> src,
                int32_t rows, int32_t hiddenSize, int32_t& pingpongId);

    __aicore__ inline void TilingByCore(int32_t totalLen, int32_t &coreLen,  //这玩意尾块不是32对齐的 注意！
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

    __aicore__ inline void GmSignalWait(__gm__ int32_t *sig_addr, int32_t cmp_val)
    {
        do {
            if (ReadGmByPassDCache(sig_addr) != cmp_val) {
                return;
            }
        } while (true);
    }

    __aicore__ inline void GmSignalWaitBarrier(__gm__ int32_t *sig_addr, int32_t cmp_val)
    {
        do {
            if (ReadGmByPassDCache(sig_addr) == cmp_val) {
                return;
            }
        } while (true);
    }

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
    TupleShape problemShape_{};
    BlockOffset baseOffset_{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // === Dispatch 自己独立用的 state，与主线 baseOffset_/problemShape_/preOffset_ 完全隔离 ===
    // 只有 AIV1 sub1 在 dispatch 流水（预 dispatch + per-expert prefetch）里使用，
    // 避免污染 cube/sub0 主线 wave 用的 baseOffset_ 等成员。
    BlockOffset dispatchBaseOffset_{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    TupleShape dispatchProblemShape_{};
    uint64_t dispatchPreOffset_ = 0;

    GM_ADDR tokenPerExpertCurRankAddr_;

    GlobalTensor<int32_t> preSumBeforeRankWorkSpace_;
    GlobalTensor<int32_t> tokenPerExpertWin_;
    GlobalTensor<int32_t> cumsumInWorkSpace_;
    GlobalTensor<int32_t> groupFlagListWorkSpace_;
    GlobalTensor<int32_t> groupFlagListGm_;
    GlobalTensor<int32_t> groupFlagListGm2_;
    GlobalTensor<int32_t> groupList_;
    GlobalTensor<int32_t> expertTokenNumsOut_;
    LocalTensor<float> topKWeightsTensor_;

    uint32_t m_ = 0;
    uint32_t k_ = 0;
    uint32_t topK_ = 0;
    uint32_t rankId_ = 0;
    uint32_t worldSize_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t groupListType_ = 0;
    uint64_t maxOutputSize_ = 0;
    uint64_t maxWavesPerExpert_ = 0;
    int32_t prevSum_ = 0;
    int32_t dispatchRows_ = 0;
    int32_t vecSetSyncCom_ = 0;
    uint32_t dispatchBlockNumPerEP_ = 2;
    int32_t perRankStridesInTokenPerExpert_ = 0;
    uint32_t blockNum_ = GetBlockNum();
    uint32_t blockAivNum_ = GetBlockNum() * 2;
    uint32_t blockIdx_ = GetBlockIdx() / GetTaskRation();
    uint32_t aivCoreIdx_ = GetBlockIdx();
    uint32_t subBlockIdx_ = GetSubBlockIdx();

    uint64_t preOffset_ = 0;

    // 当前 expert 的 wave 并行分组信息（每张 expert 复用，由 Process 计算后填入）
    uint32_t coresPerWave_ = 0;
    uint32_t wavesParallel_ = 1;
    uint32_t coreInGroup_ = 0;
    uint32_t coreGroupId_ = 0;
    bool coreActive_ = true;

    // [TRACE] 本核连续累计的 wave iteration 序号（跨 expert 累加），用于 RecordWavePoint。
    int waveTraceIdx_ = 0;

    // 把 IsInterleaved 透传给 epilogue：true 时 epilogue 在同 UB 内读 gate / up 半，输出宽 = singleN_
    using BlockEpilogue = BlockEpilogueSwigluMxQuant<QuantOutType, float, QuantScaleOutType, QuantScaleOutType,
                                                    true, IsInterleaved>;
    BlockEpilogue epilogueOp_;
};

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales, GM_ADDR yOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData *tilingData)
{
    m_ = tilingData->m;
    k_ = tilingData->k;
    topK_ = tilingData->topK;
    worldSize_ = tilingData->epWorldSize;
    expertPerRank_ = tilingData->expertPerRank;
    dispatchBlockNumPerEP_ = tilingData->blockNumPerEP;
    dispatchRows_ = tilingData->dispatchRows;
    groupListType_ = tilingData->groupListType;
    maxOutputSize_ = tilingData->maxOutputSize;
    maxWavesPerExpert_ = (tilingData->maxOutputSize + L1_TILE_M - 1) / L1_TILE_M;
    Get<N_VALUE>(problemShape_) = tilingData->n;
    Get<K_VALUE>(problemShape_) = k_;
    // dispatch 流水的独立 problem shape，N/K 与主线一致
    Get<N_VALUE>(dispatchProblemShape_) = tilingData->n;
    Get<K_VALUE>(dispatchProblemShape_) = k_;
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
    preSumBeforeRankWorkSpace_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrSumBeforeRank);
    tokenPerExpertWin_.SetGlobalBuffer((__gm__ int32_t*)params_.peermemInfo.ptrTokenPerExpert);
    cumsumInWorkSpace_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.ptrcumsumMM));
    expertTokenNumsOut_.SetGlobalBuffer((__gm__ int32_t*)params_.expertTokenNumsOutGmAddr);
    groupList_ = cumsumInWorkSpace_[(worldSize_ - 1) * expertPerRank_];
    groupFlagListWorkSpace_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrFlagSwiGluToGmm2);
    epilogueOp_.Init({params_.workspaceInfo.ptrA2, params_.workspaceInfo.ptrA2Scale,
        params_.workspaceInfo.ptrFlagSwiGluToGmm2, nullptr, nullptr, nullptr, ALIGN_128, ALIGN_256});
    perRankStridesInTokenPerExpert_ = Ops::Base::CeilAlign(
        static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128));
    tokenPerExpertCurRankAddr_ = params_.peermemInfo.ptrTokenPerExpert +
        TokenPerExpertLayout(rankId_, 0, 0) * sizeof(int32_t);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::FirstBuffInit(TPipe *tPipe)
{
    // 对CrossRankSync、Dispatch、GetCumSum过程涉及到的LocalTensor初始化管理
    uint32_t syncBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128)) * sizeof(int32_t);
    uint32_t prevSumBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128))
        * sizeof(int32_t) * expertPerRank_;
    tPipe->InitBuffer(CrossRankSyncBuf_, syncBufAlign);
    tPipe->InitBuffer(PrevSumBuf_, prevSumBufAlign);
    uint32_t getCumsumBufAlign = (AlignUp(worldSize_ * expertPerRank_, ALIGN_128)) * worldSize_ * sizeof(int32_t);
    tPipe->InitBuffer(CumsumBuf_, getCumsumBufAlign);
    tPipe->InitBuffer(CopyTempBuf_, BUFFER_ALIGN * sizeof(QuantOutType));
    uint32_t swigluCount = expertPerRank_ * maxWavesPerExpert_;
    uint32_t dispatchCount = expertPerRank_;
    uint32_t totalFlagCount = Ops::Base::CeilAlign(swigluCount + dispatchCount, 8U);
    uint32_t maxResetBufAlign = Ops::Base::CeilDiv(
        static_cast<int32_t>(totalFlagCount * sizeof(int32_t)), static_cast<int32_t>(ALIGN_32));
    // 确保ResetFlagBuf_足够容纳blockNumAlign个int32_t（用于ResetGmm2RowReadyFlags）
    uint32_t blockNumAlignBytes = Ops::Base::CeilAlign(static_cast<int32_t>(blockNum_) * static_cast<int32_t>(sizeof(int32_t)),
        static_cast<int32_t>(ALIGN_32));
    maxResetBufAlign = max(maxResetBufAlign, blockNumAlignBytes);
    tPipe->InitBuffer(ResetFlagBuf_, maxResetBufAlign);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::SecondBuffInit(TPipe *tPipe)
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
    topKWeightsTensor_ = topKWeightsBuf_.Get<float>();
    if constexpr (Std::IsSame<TopkWeightsType, float>::value) {
        GlobalTensor<float> topKWeightsGlobalTensor_;
        topKWeightsGlobalTensor_.SetGlobalBuffer((__gm__ float*)params_.probsGmAddr);
        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(m_ * topK_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(topKWeightsTensor_, topKWeightsGlobalTensor_, copyParams, copyPadParams);
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

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ResetWorkSpaceFlagList()
{
    // workSpace groupFlagList清零
    groupFlagListGm_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrFlagSwiGluToGmm2);
    if constexpr(g_coreType == AIV) {
        int32_t swigluCount = expertPerRank_ * maxWavesPerExpert_;
        int32_t dispatchCount = expertPerRank_;
        int32_t flagNum = Ops::Base::CeilAlign(static_cast<uint32_t>(swigluCount + dispatchCount), 8U);
        int32_t coreLen, coreOffset;
        TilingByCore(flagNum, coreLen, coreOffset);
        LocalTensor<int32_t> initTensor = ResetFlagBuf_.Get<int32_t>();
        if (coreLen != 0) {
            Duplicate(initTensor, 0, coreLen);
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            DataCopy(groupFlagListGm_[coreOffset], initTensor, coreLen);
        }
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ResetGmm2RowReadyFlags()
{
    if constexpr(g_coreType == AIV) {
        if (aivCoreIdx_ != 0) {
            return;
        }
        GlobalTensor<int32_t> rowReadyFlags;
        rowReadyFlags.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrGmm2RowReady);

        // 对齐到32B（8个int32_t）
        int32_t blockNumAlign = Ops::Base::CeilAlign(static_cast<uint32_t>(blockNum_), static_cast<uint32_t>(8));
        LocalTensor<int32_t> initTensor = ResetFlagBuf_.Get<int32_t>();
        Duplicate(initTensor, 0, blockNumAlign);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(rowReadyFlags, initTensor, blockNumAlign);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::GetCumsumForMMAIV()
{
    int32_t expertPerRankAligned = (expertPerRank_ + 8 - 1) / 8 * 8;
    LocalTensor<int32_t> tmpBuffer1 = CumsumBuf_.Get<int32_t>();
    DataCopyPad(tmpBuffer1, tokenPerExpertWin_[rankId_ * expertPerRank_],
        {static_cast<uint16_t>(worldSize_), static_cast<uint16_t>(expertPerRank_ * sizeof(int32_t)),
         static_cast<uint16_t>((AlignUp(worldSize_ * expertPerRank_, ALIGN_128)
             - expertPerRank_) * sizeof(int32_t)), 0},
        {});
    SyncFunc<AscendC::HardEvent::MTE2_V>();

    for (uint32_t i = 1; i < worldSize_; ++i) {
        Add(tmpBuffer1[i * expertPerRankAligned], tmpBuffer1[i * expertPerRankAligned],
            tmpBuffer1[(i - 1) * expertPerRankAligned], expertPerRank_);
        PipeBarrier<PIPE_V>();
    }
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(cumsumInWorkSpace_, tmpBuffer1,
        {static_cast<uint16_t>(worldSize_), static_cast<uint16_t>((expertPerRank_) * sizeof(int32_t)), 0, 0});
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ExpertTokenNumCopyOut()
{
    int32_t copyNum = Ops::Base::CeilAlign(static_cast<uint32_t>(expertPerRank_), static_cast<uint32_t>(ALIGN_32));
    LocalTensor<int32_t> tmpCopyBuffer = CrossRankSyncBuf_.Get<int32_t>();
    DataCopy(tmpCopyBuffer, groupList_, copyNum);
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    DataCopy(expertTokenNumsOut_, tmpCopyBuffer, copyNum);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::EndSync()
{
    if (vecSetSyncCom_ == 0) {
        return;
    }
    // 与 impl.h GMM1 中的 kGmm1Pingpong 判定一致：UB 能塞下两块才走 pingpong drain
    constexpr bool kGmm1Pingpong = IsInterleaved &&
        (MegaMoeImpl::MAX_SINGLE_MN_ALIGN32_NUM * sizeof(float) * 2U <= MegaMoeImpl::UB_CAPACITY_BYTES);
    if constexpr(g_coreType == AIC) {
        if constexpr (kGmm1Pingpong) {
            // 交织+pingpong 末尾仍有未消费 NotifyCube：
            //   vecSetSyncCom_ == 1 → 仅 flag 0 留一发
            //   vecSetSyncCom_ >= 2 → 两块 flag 各留一发，FIFO head 的 flag = vecSetSyncCom_ % 2
            // CrossCoreWaitFlag 底层 FIFO，wait 顺序必须按 NotifyCube 发出顺序。
            if (vecSetSyncCom_ == 1) {
                WaitForVector(0);
            } else {
                uint16_t firstPendingIdx = static_cast<uint16_t>(vecSetSyncCom_ & 1U);
                WaitForVector(firstPendingIdx);
                WaitForVector(static_cast<uint16_t>(1U - firstPendingIdx));
            }
        } else {
            // 单 UB（非交织 或 交织+L1_TILE_M=256）：单通道，drain 一发
            WaitForVector();
        }
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::CrossRankSyncCntPerExpert()
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    uint32_t numPerCore = Ops::Base::CeilAlign(static_cast<uint32_t>(worldSize_ * expertPerRank_),
            static_cast<uint32_t>(ALIGN_128));
    LocalTensor<int32_t> tmpBuffer = CrossRankSyncBuf_.Get<int32_t>();
    LocalTensor<int32_t> prevSumBuf = PrevSumBuf_.Get<int32_t>();
    // 给对端发送
    uint64_t offsetInDstWinAddr = perRankStridesInTokenPerExpert_ * sizeof(int32_t) * rankId_ + PEERMEM_DATA_OFFSET;
    for (int32_t dstRankIdx = aivCoreIdx_; dstRankIdx < worldSize_; dstRankIdx += blockAivNum_) {
        if (dstRankIdx == rankId_) {
            continue;
        }
        GlobalTensor<int32_t> srcAddress;
        srcAddress.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tokenPerExpertCurRankAddr_));
        GlobalTensor<int32_t> dstAddress;
        __gm__ void* dstPeermemPtr = GetRankWinAddrWithOffset(dstRankIdx, offsetInDstWinAddr);
        dstAddress.SetGlobalBuffer((__gm__ int32_t*)dstPeermemPtr);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        DataCopy(tmpBuffer, srcAddress, numPerCore);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Adds(tmpBuffer, tmpBuffer, 0x800000, numPerCore);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(dstAddress, tmpBuffer, numPerCore);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    }
    // 本端接收
    int32_t intPer512 = ALIGN_512 / sizeof(int);
    for (int32_t dstRankIdx = aivCoreIdx_; dstRankIdx < worldSize_; dstRankIdx += blockAivNum_) {
        if (dstRankIdx != rankId_) {
            for (int32_t checkIdx = 0; checkIdx < numPerCore; checkIdx += intPer512) {
                __gm__ int32_t* syncCheck = reinterpret_cast<__gm__ int32_t*>(params_.peermemInfo.ptrTokenPerExpert)
            + TokenPerExpertLayout(dstRankIdx, 0, checkIdx);
                GmSignalWait(syncCheck, 0);
            }
            DataCopy(tmpBuffer, tokenPerExpertWin_[TokenPerExpertLayout(dstRankIdx, 0, 0)], numPerCore);
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            Adds(tmpBuffer, tmpBuffer, -0x800000, numPerCore);
            PipeBarrier<PIPE_V>();
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            DataCopy(tokenPerExpertWin_[TokenPerExpertLayout(dstRankIdx, 0, 0)], tmpBuffer, numPerCore);
        } else {
            DataCopy(tmpBuffer, tokenPerExpertWin_[TokenPerExpertLayout(dstRankIdx, 0, 0)], numPerCore);
            SyncFunc<AscendC::HardEvent::MTE2_V>();
        }
        PipeBarrier<PIPE_ALL>();
        int32_t tmpSum = 0;
        int32_t j = 0;
        for (int32_t i = 0; i < (rankId_ + 1) * expertPerRank_; i++) {
            if (i >= rankId_ * expertPerRank_) {
                prevSumBuf(j) = tmpSum;
                j++;
            }
            tmpSum += tmpBuffer(i);
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(preSumBeforeRankWorkSpace_[dstRankIdx * expertPerRank_], prevSumBuf,
                    DataCopyParams{1, static_cast<uint16_t>(expertPerRank_ * sizeof(int32_t)), 0, 0});
    }
    SyncAll<true>();
    if (aivCoreIdx_ == 0) {
        GetCumsumForMMAIV();
    }
    if (blockIdx_ < worldSize_ * dispatchBlockNumPerEP_) {
        prevSum_ = preSumBeforeRankWorkSpace_(blockIdx_ / dispatchBlockNumPerEP_ * expertPerRank_);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::CopyGMToGMPerTokenV2(
    GlobalTensor<QuantOutType> dst, GlobalTensor<QuantScaleOutType> dstScale, GlobalTensor<QuantOutType> src,
    int32_t rows, int32_t hiddenSize, int32_t& pingpongId)
{
    constexpr int32_t BufferNum = 2;
    LocalTensor<QuantOutType> tmpBuffer1 = CopyTempBuf_.Get<QuantOutType>();
    LocalTensor<QuantOutType> tmpBuffer2 = tmpBuffer1[96 * 1024];
    uint32_t scaleLen = Ops::Base::CeilDiv(static_cast<int32_t>(hiddenSize),
        static_cast<int32_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    uint32_t copyInNum = hiddenSize + scaleLen;
    auto processCount = Ops::Base::CeilDiv(rows, dispatchRows_);
    for (uint32_t processIndex = 0; processIndex < processCount; ++processIndex) {
        pingpongId = (pingpongId + 1) % BufferNum;
        TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<QuantOutType> buf = pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;
        LocalTensor<QuantScaleOutType> bufScale = buf[hiddenSize].template ReinterpretCast<QuantScaleOutType>();

        auto inputOffset = processIndex * dispatchRows_ * copyInNum;

        int32_t rowNum = dispatchRows_;
        if (processIndex == processCount - 1) {
            rowNum = rows - processIndex * dispatchRows_;
        }
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        int64_t dataLen = rowNum * copyInNum;
        DataCopy(buf, src[inputOffset], dataLen);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        auto outputOffset = processIndex * dispatchRows_ * hiddenSize;
        DataCopyPad(dst[outputOffset], buf,
            {static_cast<uint16_t>(rowNum), static_cast<uint16_t>(hiddenSize),
            static_cast<uint16_t>(scaleLen / ALIGN_32), 0, 0});
        DataCopyPad(dstScale[processIndex * dispatchRows_ * scaleLen], bufScale,
            {static_cast<uint16_t>(rowNum), static_cast<uint16_t>(scaleLen),
             static_cast<uint16_t>(hiddenSize / ALIGN_32), 0, 0});
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::DispatchTokens(GMMAddrInfo &gmmAddrInfo, uint32_t expertIdx)
{
    int64_t widthA = Get<K_VALUE>(problemShape_); // H 长度
    int64_t widthAScale = Ops::Base::CeilDiv(widthA, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    int32_t pingpongId = 0;
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (uint32_t idx = blockIdx_; idx < worldSize_ * dispatchBlockNumPerEP_; idx += blockNum_) {
        uint32_t dstRankIdx = idx / dispatchBlockNumPerEP_;
        uint32_t aivIdxInGroup = idx % dispatchBlockNumPerEP_;
        uint32_t rowStartInDst = (dstRankIdx == 0 ? 0 :
            cumsumInWorkSpace_((dstRankIdx - 1) * expertPerRank_ + expertIdx));
        if (rowStartInDst < maxOutputSize_) {
            int32_t rowsCopyCnt = tokenPerExpertWin_(TokenPerExpertLayout(dstRankIdx, rankId_, expertIdx));
            if (rowStartInDst + rowsCopyCnt > maxOutputSize_) {
                rowsCopyCnt = maxOutputSize_ - rowStartInDst;
            }
            int32_t rowSrcOffset = prevSum_;
            prevSum_ += rowsCopyCnt;
            int32_t rowsPerCore = Ops::Base::CeilDiv(rowsCopyCnt, static_cast<int32_t>(dispatchBlockNumPerEP_));
            int32_t rowDstOffsetPerCore = rowStartInDst + aivIdxInGroup * rowsPerCore;
            int32_t rowSrcOffsetPerCore = rowSrcOffset + aivIdxInGroup * rowsPerCore;
            if (rowDstOffsetPerCore + rowsPerCore > rowStartInDst + rowsCopyCnt) {
                rowsPerCore = rowStartInDst + rowsCopyCnt - rowDstOffsetPerCore;
            }
            if (rowsPerCore > 0) {
                GlobalTensor <QuantOutType> gmRemoteA, aGlobalTensor;
                GlobalTensor <QuantScaleOutType> aScaleGlobalTensor;
                int64_t gmOffsetA = rowDstOffsetPerCore * widthA;
                int64_t gmOffsetAScale = rowDstOffsetPerCore * widthAScale;
                int64_t gmOffsetPeer = rowSrcOffsetPerCore * (widthA + widthAScale);
                gmRemoteA.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(GetRankWinAddrWithOffset(dstRankIdx,
                    PEERMEM_DATA_OFFSET + perRankStridesInTokenPerExpert_ * worldSize_ * sizeof(uint32_t))));
                aGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantOutType*>(gmmAddrInfo.aGlobal));
                aScaleGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleOutType *>(
                    gmmAddrInfo.aScaleGlobal));
                CopyGMToGMPerTokenV2(aGlobalTensor[gmOffsetA], aScaleGlobalTensor[gmOffsetAScale],
                    gmRemoteA[gmOffsetPeer], rowsPerCore, widthA, pingpongId);
            }
        }
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        AtomicAdd((__gm__ int32_t*)(groupFlagListGm2_.GetPhyAddr()), int32_t(1));
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline bool MegaMoe<TemplateMegaMoeTypeFunc>::UpdateGroupParams(uint32_t expertIdx)
{
    // baseOffset_ is 0 when groupIdx = 0
    if (expertIdx != 0) {
        uint64_t m = Get<M_VALUE>(problemShape_);
        uint64_t n = Get<N_VALUE>(problemShape_);
        uint64_t k = Get<K_VALUE>(problemShape_);
        Get<IDX_A_OFFSET>(baseOffset_) += m * k;
        Get<IDX_B_OFFSET>(baseOffset_) += n * k;
        // only splitM
        auto scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_A_SCALE_OFFSET>(baseOffset_) += m * scaleK;
        Get<IDX_B_SCALE_OFFSET>(baseOffset_) += n * scaleK;
        Get<IDX_C_OFFSET>(baseOffset_) += m * n / SWIGLU_N_HALF;
        Get<IDX_C_SCALE_OFFSET>(baseOffset_) +=
            m * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_FLAG_OFFSET>(baseOffset_) += maxWavesPerExpert_;
        Get<IDX_B2_OFFSET>(baseOffset_) += k * n / SWIGLU_N_HALF;
        Get<IDX_B2_SCALE_OFFSET>(baseOffset_) +=
            k * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_Y2_OFFSET>(baseOffset_) += m * k;
        Get<IDX_M_OFFSET>(baseOffset_) += m;
    }

    int32_t splitValue = 0;
    if (groupListType_ == 0) {
        int32_t offset = static_cast<int32_t>(groupList_.GetValue(expertIdx));
        splitValue = offset - preOffset_;
        preOffset_ = offset;
    } else {
        splitValue = static_cast<uint64_t>(groupList_.GetValue(expertIdx));
    }
    Get<M_VALUE>(problemShape_) = splitValue;
    // split_m，when m=0, skip
    if (Get<M_VALUE>(problemShape_) == 0) {
        return false;
    }
    return true;
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo, uint32_t gmmIndex, uint64_t waveIdx)
{
    uint64_t k = Get<K_VALUE>(problemShape_);
    uint64_t n = Get<N_VALUE>(problemShape_);
    uint64_t outputN = n / SWIGLU_N_HALF;
    auto scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    auto scaleOutputN = Ops::Base::CeilDiv(outputN, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    int64_t waveRowOffset = static_cast<int64_t>(waveIdx * L1_TILE_M);
    if (gmmIndex == 0) {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.ptrA +
            (Get<IDX_A_OFFSET>(baseOffset_) + waveRowOffset * static_cast<int64_t>(k)) * sizeof(QuantOutType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.ptrAScale +
            (Get<IDX_A_SCALE_OFFSET>(baseOffset_) + waveRowOffset * static_cast<int64_t>(scaleK)) *
            sizeof(QuantScaleOutType);
        AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
            Get<IDX_C_OFFSET>(baseOffset_) + waveRowOffset * static_cast<int64_t>(outputN),
            Get<IDX_C_SCALE_OFFSET>(baseOffset_) + waveRowOffset * static_cast<int64_t>(scaleOutputN),
            Get<IDX_FLAG_OFFSET>(baseOffset_) + static_cast<int64_t>(waveIdx), 0L, 0L, 0L};
        epilogueOp_.UpdateGlobalAddr(vecBaseOffset);
        // ★ GMM1 HBM 输出地址（float）：每行 n 个 float
        //   不同 expert 通过 IDX_M_OFFSET 隔开，同 expert 内不同 wave 通过 waveRowOffset 隔开。
        //   走 (m, n) float 空间，按 row 维 (IDX_M_OFFSET + waveRowOffset) * n * 4 算偏移。
        int64_t gmm1RowBase = Get<IDX_M_OFFSET>(baseOffset_) + waveRowOffset;
        gmmAddrInfo.gmm1OutGlobal = params_.workspaceInfo.ptrGmm1Out +
            gmm1RowBase * static_cast<int64_t>(n) * sizeof(float);
        if constexpr(g_coreType == AIC) {
            gmmAddrInfo.bGlobal = params_.bGmAddr + Get<IDX_B_OFFSET>(baseOffset_) * sizeof(QuantOutType);
            gmmAddrInfo.bScaleGlobal = params_.bScaleGmAddr + Get<IDX_B_SCALE_OFFSET>(baseOffset_) *
                                        sizeof(QuantScaleOutType);
        }
    } else if constexpr(g_coreType == AIC) {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.ptrA2 +
            (Get<IDX_C_OFFSET>(baseOffset_) + waveRowOffset * static_cast<int64_t>(outputN)) * sizeof(QuantOutType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.ptrA2Scale +
            (Get<IDX_C_SCALE_OFFSET>(baseOffset_) + waveRowOffset * static_cast<int64_t>(scaleOutputN)) *
            sizeof(QuantScaleOutType);
        gmmAddrInfo.bGlobal = params_.b2GmAddr + Get<IDX_B2_OFFSET>(baseOffset_) * sizeof(QuantOutType);
        gmmAddrInfo.bScaleGlobal = params_.b2ScaleGmAddr + Get<IDX_B2_SCALE_OFFSET>(baseOffset_) *
                                    sizeof(QuantScaleOutType);
    }
    gmmAddrInfo.groupFlagList = (__gm__ int32_t*)params_.workspaceInfo.ptrFlagSwiGluToGmm2 +
                                Get<IDX_FLAG_OFFSET>(baseOffset_) + static_cast<int64_t>(waveIdx);
    if (gmmIndex == 1) {
        gmmAddrInfo.gmm2OutGlobal = params_.workspaceInfo.ptrGmmOut +
            (Get<IDX_Y2_OFFSET>(baseOffset_) + waveRowOffset * static_cast<int64_t>(k)) * sizeof(bfloat16_t);
    }
    groupFlagListGm_.SetGlobalBuffer(gmmAddrInfo.groupFlagList);
}

// Dispatch 流水专用：用 dispatchBaseOffset_ / dispatchProblemShape_ / dispatchPreOffset_，
// 与主线 baseOffset_ 等成员完全隔离。只 sub1 调用。
// 只维护 A / AScale 的 offset（dispatch 只写这两个），B / C / Y2 等不动。
template <TemplateMegaMoeTypeClass>
__aicore__ inline bool MegaMoe<TemplateMegaMoeTypeFunc>::UpdateGroupParamsForDispatch(uint32_t expertIdx)
{
    if (expertIdx != 0) {
        uint64_t m = Get<M_VALUE>(dispatchProblemShape_);
        uint64_t k = Get<K_VALUE>(dispatchProblemShape_);
        auto scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_A_OFFSET>(dispatchBaseOffset_) += m * k;
        Get<IDX_A_SCALE_OFFSET>(dispatchBaseOffset_) += m * scaleK;
    }
    int32_t splitValue = 0;
    if (groupListType_ == 0) {
        int32_t offset = static_cast<int32_t>(groupList_.GetValue(expertIdx));
        splitValue = offset - static_cast<int32_t>(dispatchPreOffset_);
        dispatchPreOffset_ = static_cast<uint64_t>(offset);
    } else {
        splitValue = static_cast<int32_t>(groupList_.GetValue(expertIdx));
    }
    Get<M_VALUE>(dispatchProblemShape_) = splitValue;
    return (splitValue != 0);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::UpdateGlobalBufferForDispatch(
    GMMAddrInfo &gmmAddrInfo, uint32_t expertIdx)
{
    gmmAddrInfo.aGlobal = params_.workspaceInfo.ptrA +
        Get<IDX_A_OFFSET>(dispatchBaseOffset_) * sizeof(QuantOutType);
    gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.ptrAScale +
        Get<IDX_A_SCALE_OFFSET>(dispatchBaseOffset_) * sizeof(QuantScaleOutType);
    gmmAddrInfo.groupFlagList2 =
        (__gm__ int32_t*)params_.workspaceInfo.ptrFlagDispatchToGmm1 + expertIdx;
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ArgSortExpandedRowIdx()
{
    int32_t coreLen, coreOffset;
    TilingByCore(m_ * topK_, coreLen, coreOffset, 1);
    __gm__ int32_t *src = (__gm__ int32_t *)params_.workspaceInfo.expandedRowIdx;
    __gm__ int32_t *dst = (__gm__ int32_t *)params_.workspaceInfo.expandedRowIdxGather;
    Simt::VF_CALL<ArgsortSimt>(Simt::Dim3{MoeInitRoutingV3::SIMT_THREAD_NUM, 1, 1}, coreLen, coreOffset, src, dst);
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::DDR>();
    GlobalTensor<int32_t> rowIndexGatherGm;
    rowIndexGatherGm.SetGlobalBuffer((__gm__ int32_t *)params_.workspaceInfo.expandedRowIdxGather);
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(rowIndexGatherGm);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ResetTokenPerExpert()
{
    if (aivCoreIdx_ != blockAivNum_ - 1) {
        return;
    }
    int32_t num = worldSize_ * Ops::Base::CeilAlign(
            static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128));
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    LocalTensor<int32_t> tmp = ResetTokenBuf_.Get<int32_t>();
    Duplicate(tmp, 0, num);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(tokenPerExpertWin_, tmp, num);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Unpermute()
{
    int32_t coreLen, coreOffset;
    TilingByCore(m_, coreLen, coreOffset, 1);
    GlobalTensor<int32_t> expandedRowIdx;
    expandedRowIdx.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.expandedRowIdxGather);
    GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer((__gm__ bfloat16_t*)params_.peermemInfo.ptrD);
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer((__gm__ bfloat16_t*)params_.y2GmAddr);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    for (int32_t tokenIdx = coreOffset; tokenIdx < coreLen + coreOffset; tokenIdx++) {
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        LocalTensor<bfloat16_t> dataResBf16 = DataResBuf_.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> dataIn0Bf16 = dataResBf16[k_];
        LocalTensor<bfloat16_t> dataIn1Bf16 = dataResBf16[k_ * 2];
        LocalTensor<float> dataResFp32 = DataResFp32Buf_.Get<float>();
        LocalTensor<float> dataIn0Fp32 = dataResFp32[k_];
        LocalTensor<float> dataIn1Fp32 = dataResFp32[k_ * 2];
        for (int32_t expId = 0; expId < topK_; ++expId) {
            int32_t rowIdx = expandedRowIdx(tokenIdx * topK_ + expId);
            float expScale = topKWeightsTensor_.GetValue(tokenIdx * topK_ + expId);
            auto event = (expId % 2 == 0) ? EVENT_ID0 : EVENT_ID1;
            auto dataInBf16 = (expId % 2 == 0) ? dataIn0Bf16 : dataIn1Bf16;
            auto dataInFp32 = (expId % 2 == 0) ? dataIn0Fp32 : dataIn1Fp32;
            WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            DataCopy(dataInBf16, expandedX[rowIdx * k_], k_);
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
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(output[tokenIdx * k_], dataResBf16, k_);
    }
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::CrossRankSyncInWorldSize()
{
    __gm__ int32_t* syncRank = (__gm__ int32_t*)(params_.peermemInfo.ptrBase);
    __gm__ int32_t* syncCount = (__gm__ int32_t*)(params_.peermemInfo.ptrBase + 48 * 1024);
    int count = *(syncCount + aivCoreIdx_ * 16) + 1;
    for (int i = aivCoreIdx_; i < worldSize_; i += blockAivNum_) {
        __gm__ int32_t* syncRemoteAddr = (__gm__ int32_t*)(winRankAddr_[i]) + rankId_ * 16;
        WriteGmByPassDCache(syncRemoteAddr, count);
        auto syncCheck = syncRank + i * 16;
        GmSignalWaitBarrier(syncCheck, count);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
    *(syncCount + aivCoreIdx_ * 16) = count;
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Process()
{
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();

    // [TRACE] Process 入口（最早可打的点，y2GmAddr 是输入 yOut 指针，未被使用前）
    MegaMoeTrace::RecordTimestampAt(params_.y2GmAddr, MegaMoeTrace::P_KERNEL_ENTRY);

    // 1. moe_init_routing_v3 + 量化
    moe_init_routing_v3<bfloat16_t, QuantOutType>(params_.aGmAddr, params_.expertIdxGmAddr, nullptr, nullptr,
            params_.peermemInfo.ptrA0, params_.workspaceInfo.expandedRowIdx, tokenPerExpertCurRankAddr_, nullptr,
            params_.workspaceInfo.ptrA, &params_.tilingData->moeInitRoutingTilingData,
            params_.tilingData->moeInitRoutingTilingData.tilingKey);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();

    // [TRACE] init_routing + SyncAll 完成
    MegaMoeTrace::RecordTimestampAt(params_.y2GmAddr, MegaMoeTrace::P_ROUTING_DONE);

    // 2. buffer申请 & workspace Init
    TPipe firstPipe;
    FirstBuffInit(&firstPipe);
    ResetWorkSpaceFlagList();
    ResetGmm2RowReadyFlags();

    // 3. cnt值全ep域内的发送以及本端接收整理
    CrossRankSyncCntPerExpert();
    // if (subBlockIdx_ == 0) { // 挪到后面去再destroy
    //     PipeBarrier<PIPE_ALL>();
    //     firstPipe.Destroy();
    // }
    SyncAll<false>();

    // [TRACE] CrossRankSync + buffer init 完成
    MegaMoeTrace::RecordTimestampAt(params_.y2GmAddr, MegaMoeTrace::P_CROSSRANK_DONE);

    // 4. for循环: per-expert Dispatch + 并行 wave GMM1→GMM2 流水
    //    分核策略（每 expert 重新决定一次，因 waveCnt 取决于 m_i）：
    //    - tileNumGmm1 = ceil(outputN / L1_TILE_N)            // GMM1 在 N 方向需多少 tile
    //    - wavesParallel = min(waveCnt, blockNum / tileNumGmm1) // 并行 wave 数，最多 waveCnt
    //    - coresPerWave  = blockNum / wavesParallel             // 每组分到的核数
    //    用 waveCnt 当上限保证不会出现"空 group"，coresPerWave 经常会 >= tileNumGmm1。
    //    GMM1 多出来的核（coreInGroup >= tileNumGmm1）在 GMM1 阶段空闲；
    //    但 GMM2 的 tileNum (k_orig / L1_TILE_N) 通常远大于 tileNumGmm1，那些核能在 GMM2 阶段干活。
    //    例 (blockNum 来自 GetBlockNum()，不写死)：
    //      BS=512, outputN=1536, waveCnt=4, tileNumGmm1=6
    //      blockNum_=32: maxGroups=5, wavesParallel=4, coresPerWave=8
    //                    GMM1 每 wave 6 核干活 / 2 核 idle，4 wave 共 24 核活跃；
    //                    GMM2 每 wave 8 核分 28 tile，4 wave 共 32 核全部满载。
    //      blockNum_=36: maxGroups=6, wavesParallel=4, coresPerWave=9
    //                    GMM1 每 wave 6 核干活 / 3 核 idle，4 wave 共 24 核活跃；
    //                    GMM2 每 wave 9 核分 28 tile，4 wave 共 36 核全部满载。
    //    cube/vec 信号是 per-pair（同 blockIdx 的 AIC/AIV 共享一个 flag 通道），不同 pair
    //    之间天然隔离，多组并行不需要再按 group 区分 flag value。
    //    真正需要的同步是同核相邻 wave 之间的 UB 屏障：上一 wave AIV combine 的 MTE3 未
    //    结束时 AIC 已经开始下一 wave GMM1 fixpipe 写 UB，会破坏 combine 读出的数据。
    //    见下面 wave 末的 SyncFunc<MTE3_V> + NotifyCube。
    // 交织：GMM1 scheduler 走全 B 宽 n（每 tile 1 次 mmad）→ tileNum = n / L1_TILE_N
    // 非交织：GMM1 scheduler 走 swiglu 输出 N = n/2（每 tile 2 次 mmad）→ tileNum = (n/2) / L1_TILE_N
    uint64_t schedulerN_const;
    if constexpr (IsInterleaved) {
        schedulerN_const = Get<N_VALUE>(problemShape_);
    } else {
        schedulerN_const = Get<N_VALUE>(problemShape_) / SWIGLU_N_HALF;
    }
    uint32_t tileNumGmm1 = static_cast<uint32_t>(
        (schedulerN_const + static_cast<uint64_t>(L1_TILE_N) - 1UL) / static_cast<uint64_t>(L1_TILE_N));
    if (tileNumGmm1 < 1U) {
        tileNumGmm1 = 1U;
    }
    // 旧版策略 maxGroups = blockNum/tileNumGmm1 强制 "每 wave 内每核只跑 1 个 GMM1 tile"，
    // 在 tileNumGmm1 大、waveCnt 也大时把 wavesParallel 卡死。改成允许 GMM1 内每核跑多 tile，
    // 让 wavesParallel 提到 blockNum/(tileNumGmm1/MAX_TILES_PER_CORE_GMM1)。cube mmad 在 wave 内
    // 多 tile 是 back-to-back 流水的，per-iter 时间不会线性放大。
    // ★ 跨 wave 分核优化：MAX_TILES_PER_CORE_GMM1 = 1 把"N 方向多 tile 占满一个 wave"换成
    //   "N 方向 1 tile + 同一个核串多个 wave"。这样:
    //     wavesParallel = blockNum/tileNumGmm1，coresPerWave = tileNumGmm1
    //   单个 wave 的 GMM1 在一轮 mmad 内全部 cube 都收尾，wave A 的 swiglu 完成时间也更早；
    //   配合下方 batch loop（Phase1: 同核连续跑 N 个 wave 的 GMM1+epi；Phase2: 再统一跑这 N 个
    //   wave 的 GMM2+combine），就能把 GMM2(wave A) 的 mmad 与 sw(wave B) 的 epi 在 AIC/AIV
    //   上交错——sw(wave B) 不再被同 wave 的 GMM2 卡死。
    constexpr uint32_t MAX_TILES_PER_CORE_GMM1 = 1U;
    uint32_t minCoresPerWaveGmm1 = (tileNumGmm1 + MAX_TILES_PER_CORE_GMM1 - 1U) / MAX_TILES_PER_CORE_GMM1;
    if (minCoresPerWaveGmm1 < 1U) {
        minCoresPerWaveGmm1 = 1U;
    }

    // === Dispatch 流水化 ===
    // sub1 用独立的 dispatchBaseOffset_/dispatchProblemShape_/dispatchPreOffset_ 维护自己的
    // expert 0..N-1 推进进度，与主线 baseOffset_ 等成员完全隔离。
    //   - expert loop 之前预 dispatch expert 0
    //   - expert k loop body 内 (在主线 UpdateGroupParams(k) 之后) 预 dispatch expert k+1
    // 这样 cube 进 expert k+1 wave 0 时 dispatch flag 已 ready，wDsp ≈ 0us。
    // sub1 自身的 groupFlagListGm2_ 被 prefetch 改但 sub1 后续不读它（DispatchTokens 内才用）。
    if constexpr(g_coreType == AscendC::AIV) {
        if (subBlockIdx_ == 1 && expertPerRank_ > 0U) {
            if (UpdateGroupParamsForDispatch(0)) {
                GMMAddrInfo prefetchInfo;
                UpdateGlobalBufferForDispatch(prefetchInfo, 0);
                groupFlagListGm2_.SetGlobalBuffer(prefetchInfo.groupFlagList2);
                MegaMoeTrace::RecordDispatchPoint(params_.y2GmAddr, 0, false);
                DispatchTokens(prefetchInfo, 0);
                MegaMoeTrace::RecordDispatchPoint(params_.y2GmAddr, 0, true);
            }
        }
    }

    GMMAddrInfo gmmAddrInfo;
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
        bool valid = UpdateGroupParams(expertIdx);

        // sub1 在当前 expert k loop body 内预 dispatch expert k+1：
        // 用独立的 dispatch 状态，与上面 UpdateGroupParams(k) 改的主线 baseOffset_ 等无关。
        if constexpr(g_coreType == AscendC::AIV) {
            if (subBlockIdx_ == 1 && expertIdx + 1U < expertPerRank_) {
                if (UpdateGroupParamsForDispatch(expertIdx + 1U)) {
                    GMMAddrInfo prefetchInfo;
                    UpdateGlobalBufferForDispatch(prefetchInfo, expertIdx + 1U);
                    groupFlagListGm2_.SetGlobalBuffer(prefetchInfo.groupFlagList2);
                    MegaMoeTrace::RecordDispatchPoint(params_.y2GmAddr, expertIdx + 1, false);
                    DispatchTokens(prefetchInfo, expertIdx + 1U);
                    MegaMoeTrace::RecordDispatchPoint(params_.y2GmAddr, expertIdx + 1, true);
                }
            }
        }

        if (!valid) {
            continue;
        }
        uint64_t m_i = Get<M_VALUE>(problemShape_);
        uint64_t waveCnt = (m_i + L1_TILE_M - 1) / L1_TILE_M;

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

        // 为当前 expert 配置 cube 用的 gmmAddrInfo（dispatch 不在这里发了，已经被前一轮预 dispatch 了）
        UpdateGlobalBuffer(gmmAddrInfo, 0);
        gmmAddrInfo.groupFlagList2 = (__gm__ int32_t*)params_.workspaceInfo.ptrFlagDispatchToGmm1 + expertIdx;
        groupFlagListGm2_.SetGlobalBuffer(gmmAddrInfo.groupFlagList2);

        // ★ Wave loop: 跨 wave 分核 batch 结构 ★
        // 本核所在组负责 waveIdx ∈ {coreGroupId_, coreGroupId_ + wavesParallel_, ...}
        // batch 长度 MAX_CROSS_WAVE_GMM1 = 2：同核连续把 batch 内 N 个 wave 的 GMM1+epi 全部跑完，
        // 再统一跑这 N 个 wave 的 GMM2+combine。
        //   AIC 时间线：mmad(wA) → mmad(wB) → GMM2(wA) → GMM2(wB)
        //   AIV(sub0) 时间线：sw(wA) → sw(wB)；AIV(sub1)：combine(wA) → combine(wB)
        // 关键 overlap：AIC 跑 GMM2(wA) 的 mmad 时，AIV(sub0) 正在跑 sw(wB)——swiglu 不再
        // 被同 wave 的 GMM2 cube mmad 卡到下一轮，sw(wB) 完整藏在 GMM2(wA) cube 时间里。
        //
        // 子块分工不变：
        //   - AIC      ：GMM1 mmad，GMM2 mmad
        //   - subBlock 0：GMM1 epilogue (swiglu+quant)
        //   - subBlock 1：dispatch（每 expert 一次，wave loop 之前）+ GMM2 combine（每 wave）
        //
        // EndSync 处理：batch 内 GMM1 之间、batch 之间都不调 EndSync——
        //   下一个 GMM1 的首个 tile 的 `if (vecSetSyncCom) WaitForVector` 会自然吃掉上一发
        //   AIV NotifyCube；vecSetSyncCom_ 也跨 batch / 跨 expert 保留。expert 循环结束后
        //   统一 drain 一次最末尾的 pending NotifyCube。
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

                // [TRACE] sp0: wave loop iteration 入口
                MegaMoeTrace::RecordWavePoint(params_.y2GmAddr, curWaveTrace, MegaMoeTrace::SP_WAVE_ENTRY);

                uint64_t waveM = (waveIdx == waveCnt - 1 && m_i % L1_TILE_M != 0) ?
                                 m_i % L1_TILE_M : L1_TILE_M;
                Get<M_VALUE>(problemShape_) = waveM;

                GMMAddrInfo waveGmmInfo = gmmAddrInfo;
                UpdateGlobalBuffer(waveGmmInfo, 0, waveIdx);
                groupFlagListGm_.SetGlobalBuffer(waveGmmInfo.groupFlagList);

                // [TRACE] sp1
                MegaMoeTrace::RecordWavePoint(params_.y2GmAddr, curWaveTrace, MegaMoeTrace::SP_GMM1_BEGIN);

                if (subBlockIdx_ == 0) {
                    // ★ flagIdBase = bi & 1：跨 batch wave 用 0/1 轮换，
                    //   避免 CrossCoreSetFlag 单槽阻塞 cube 的第二个 NotifyVector。
                    MegaMoeImpl::GroupMatmulSwigluQuant<
                        QuantOutType, QuantOutType, float, QuantScaleOutType, QuantScaleOutType, IsInterleaved>(
                        epilogueOp_, params_, problemShape_, waveGmmInfo, coreInGroup_, coresPerWave_,
                        vecSetSyncCom_, curWaveTrace, static_cast<uint16_t>(bi & 1U));
                }

                // [TRACE] sp5: GMM1 完成（注意此处不 EndSync，AIV sw 可能还在跑）
                MegaMoeTrace::RecordWavePoint(params_.y2GmAddr, curWaveTrace, MegaMoeTrace::SP_GMM1_END);
            }

            // ===== Phase 2: 把 batch 内每个 wave 的 GMM2 + combine 跑完 =====
            //   GMM2 的 cube mmad 由 groupFlagListGm（swiglu→gmm2 flag）保护；AIV combine 由
            //   rowReadyFlags 保护。都不依赖 cube/vec NotifyVector/WaitForCube 通道。
            //   GMM2(wA) cube mmad 与 sw(wB) AIV epi 自然在不同核上 overlap。
            for (uint32_t bi = 0; bi < batchSize; ++bi) {
                uint64_t waveIdx = batchedWaves[bi];
                int curWaveTrace = waveTraceIdx_ + static_cast<int>(bi);

                uint64_t waveM = (waveIdx == waveCnt - 1 && m_i % L1_TILE_M != 0) ?
                                 m_i % L1_TILE_M : L1_TILE_M;
                Get<M_VALUE>(problemShape_) = waveM;

                GMMAddrInfo waveGmmInfo = gmmAddrInfo;
                UpdateGlobalBuffer(waveGmmInfo, 1, waveIdx);
                groupFlagListGm_.SetGlobalBuffer(waveGmmInfo.groupFlagList);

                // [TRACE] sp6
                MegaMoeTrace::RecordWavePoint(params_.y2GmAddr, curWaveTrace, MegaMoeTrace::SP_GMM2_BEGIN);

                // GMM2 + Combine for this wave（GMM2 不用 cube/vec 信号，靠 rowReadyFlags 每核计数器）
                if constexpr(g_coreType == AscendC::AIC) {
                    MegaMoeImpl::GroupMatmul2<QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType,
                                              IsInterleaved>(
                        params_, problemShape_, waveGmmInfo, coreInGroup_, coresPerWave_, expertIdx, rankId_,
                        static_cast<uint64_t>(waveIdx) * static_cast<uint64_t>(L1_TILE_M), curWaveTrace);
                } else if (subBlockIdx_ == 1) {
                    MegaMoeImpl::GroupMatmul2<QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType,
                                              IsInterleaved>(
                        params_, problemShape_, waveGmmInfo, coreInGroup_, coresPerWave_, expertIdx, rankId_,
                        static_cast<uint64_t>(waveIdx) * static_cast<uint64_t>(L1_TILE_M), curWaveTrace);
                }

                // [TRACE] sp10 / sp11
                MegaMoeTrace::RecordWavePoint(params_.y2GmAddr, curWaveTrace, MegaMoeTrace::SP_GMM2_END);
                MegaMoeTrace::RecordWavePoint(params_.y2GmAddr, curWaveTrace, MegaMoeTrace::SP_WAVE_END);
            }

            waveTraceIdx_ += static_cast<int>(batchSize);
        }

        Get<M_VALUE>(problemShape_) = m_i;
    }

    // ★ 全部 expert 跑完后，drain 最后一次 GMM1 留在 cube/vec flag 槽里的 pending NotifyCube。
    //   batch 内 / batch 间 / expert 间都没调 EndSync，最末尾这一发要在 kernel 退出前清掉，
    //   避免 HW flag 状态对后续 kernel 启动产生意外影响。
    EndSync();
    vecSetSyncCom_ = 0;

    // [TRACE] expert loop 整体结束
    MegaMoeTrace::RecordTimestampAt(params_.y2GmAddr, MegaMoeTrace::P_EXPERT_LOOP_END);

    if (subBlockIdx_ == 0) {
        PipeBarrier<PIPE_ALL>();
        firstPipe.Destroy();
    }
    if constexpr(g_coreType == AscendC::AIV) {
        if (aivCoreIdx_ == 1) {
            ExpertTokenNumCopyOut();
        }
        if (subBlockIdx_ == 1) {
            PipeBarrier<PIPE_ALL>();
            firstPipe.Destroy();
        }
    }
    SyncAll<true>();

    // 最终 SyncAll<true> 之后打 P_PROCESS_END（trace 关时 inline 函数体为空，无开销）
    MegaMoeTrace::RecordTimestampAt(params_.y2GmAddr, MegaMoeTrace::P_PROCESS_END);

    // unpermute & 清理 win 区。开 trace 时 Unpermute 会覆盖 y 上的时间戳，所以跳过。
    if constexpr(g_coreType == AIV) {
        ArgSortExpandedRowIdx();
        TPipe secondPipe;
        SecondBuffInit(&secondPipe);
        ResetTokenPerExpert();
        CrossRankSyncInWorldSize(); // 全卡软同步，确认 combine send 完成
#ifndef MEGA_MOE_TRACE_ENABLE
        Unpermute();
#endif
        PipeBarrier<PIPE_ALL>();
        secondPipe.Destroy();
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

}   // namespace MegaMoeImpl
#endif