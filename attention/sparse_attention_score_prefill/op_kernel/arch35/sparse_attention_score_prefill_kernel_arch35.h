/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPARSE_ATTENTION_SCORE_PREFILL_KERNEL_ARCH35_H
#define SPARSE_ATTENTION_SCORE_PREFILL_KERNEL_ARCH35_H

#include "kernel_utils.hpp"
#include "../kernel_common.hpp"

using namespace NpuArch;
using namespace tla;

namespace SasaPrefillKernelArch35 {

template <
    class BlockMmadQK,
    class EpilogueOnlineSoftmax,
    class BlockMmadPV,
    class EpilogueRescaleO>
class SasaPrefillRegularKernelArch35 {
public:
    using ArchTag = typename BlockMmadPV::ArchTag;

    using ElementQ = typename BlockMmadQK::ElementQ;
    using ElementK = typename BlockMmadQK::ElementK;
    using ElementS = float;
    using ElementP = typename BlockMmadPV::ElementP;
    using ElementV = typename BlockMmadPV::ElementV;
    using ElementO = typename BlockMmadQK::ElementQ;
    using ElementOTmp = float;
    using ElementWorkspaceO = float;

    using LayoutK = layout::ColumnMajor;
    using LayoutV = layout::RowMajor;
    using LayoutQ = layout::RowMajor;
    using LayoutS = layout::RowMajor;
    using LayoutP = layout::RowMajor;
    using LayoutO = layout::RowMajor;

    static constexpr uint32_t PRE_LAUNCH = 2U;
    static constexpr uint32_t MAX_CROSS_CORE_BUF_STAGES = PRE_LAUNCH + 1U;
    static constexpr uint32_t UB_S_OTMP_BUF_STAGES = 2U;
    static constexpr uint32_t P_L1_BUF_NUM = PRE_LAUNCH + 1U;

    static constexpr uint32_t FP32_ONE_BLOCK_SIZE = 8U;

    __aicore__ inline
    uint64_t TaskLinearIdx(uint32_t qToken, uint32_t kvHeadIdx) const
    {
        return static_cast<uint64_t>(qToken) * kvHeads_ + kvHeadIdx;
    }

    __aicore__ inline
    uint64_t SlotOOffset(uint32_t qToken, uint32_t kvHeadIdx, uint32_t slotK) const
    {
        return TaskLinearIdx(qToken, kvHeadIdx) * topK_ * slotOElems_
            + static_cast<uint64_t>(slotK) * slotOElems_;
    }

    __aicore__ inline
    uint64_t SlotStatOffset(uint32_t qToken, uint32_t kvHeadIdx, uint32_t slotK, uint32_t rowGh) const
    {
        return TaskLinearIdx(qToken, kvHeadIdx) * topK_ * slotStatElems_
            + static_cast<uint64_t>(slotK) * slotStatElems_
            + static_cast<uint64_t>(rowGh);
    }

    __aicore__ inline
    void InitSyncFlags()
    {
#ifdef __DAV_CUBE__
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);
        AscendC::CrossCoreSetFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_MTE1>(2);
        AscendC::CrossCoreSetFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_MTE1>(18);
        AscendC::CrossCoreSetFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_MTE1>(3);
        AscendC::CrossCoreSetFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_MTE1>(19);
        AscendC::CrossCoreSetFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_MTE1>(4);
        AscendC::CrossCoreSetFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_MTE1>(20);
#endif
#ifdef __DAV_VEC__
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        AscendC::CrossCoreSetFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_V>(0);
        AscendC::CrossCoreSetFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_V>(1);
#endif
    }

    __aicore__ inline
    void ReleaseSyncFlags()
    {
#ifdef __DAV_CUBE__
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);
        AscendC::CrossCoreWaitFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_FIX>(0);
        AscendC::CrossCoreWaitFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_FIX>(1);
        AscendC::CrossCoreWaitFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_FIX>(16);
        AscendC::CrossCoreWaitFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_FIX>(17);
#endif
#ifdef __DAV_VEC__
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        AscendC::CrossCoreWaitFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_MTE3>(2);
        AscendC::CrossCoreWaitFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_MTE3>(3);
        AscendC::CrossCoreWaitFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_MTE3>(4);
#endif
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline
    static uint32_t VecRowSplit(uint32_t groupSize)
    {
        uint32_t half = RoundUp(groupSize, 8U) / 2U;
        return (half > groupSize) ? groupSize : half;
    }

    __aicore__ inline
    static uint32_t VecNumRows(uint32_t groupSize)
    {
        if (groupSize <= 1U) {
            return (AscendC::GetSubBlockIdx() == 0U) ? groupSize : 0U;
        }
        uint32_t split = VecRowSplit(groupSize);
        return (AscendC::GetSubBlockIdx() == 0U) ? split : (groupSize - split);
    }

    __aicore__ inline
    static uint32_t VecGlobalRowOffset(uint32_t groupSize)
    {
        return (AscendC::GetSubBlockIdx() == 0U) ? 0U : VecRowSplit(groupSize);
    }

    __aicore__ inline
    SasaPrefillRegularKernelArch35() {}

    __aicore__ inline
    void operator()(SasaPrefillKernelParamsArch35 const &params)
    {
        __gm__ SparseAttnPrefill::SparseAttentionScorePrefillTilingData *tilingData =
            reinterpret_cast<__gm__ SparseAttnPrefill::SparseAttentionScorePrefillTilingData *>(params.tiling);
        FetchBaseShapeInfo(tilingData);

        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK *)params.k);
        AscendC::GlobalTensor<ElementV> gV;
        gV.SetGlobalBuffer((__gm__ ElementV *)params.v);
        AscendC::GlobalTensor<int32_t> gK2qRowPtr;
        gK2qRowPtr.SetGlobalBuffer((__gm__ int32_t *)params.k2qRowPtr);
        AscendC::GlobalTensor<int32_t> gK2qQIndices;
        gK2qQIndices.SetGlobalBuffer((__gm__ int32_t *)params.k2qQIndices);
        AscendC::GlobalTensor<int32_t> gK2qSlotIndices;
        gK2qSlotIndices.SetGlobalBuffer((__gm__ int32_t *)params.k2qSlotIndices);
        AscendC::GlobalTensor<int32_t> gBlockTable;
        gBlockTable.SetGlobalBuffer((__gm__ int32_t *)params.blockTable);
        gActualQseqlen_.SetGlobalBuffer((__gm__ int32_t *)params.actualQseqlen);
        gActualKvseqlen_.SetGlobalBuffer((__gm__ int32_t *)params.actualKvseqlen);
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO *)params.o);

        uint64_t wsOffset = 0U;
        AscendC::GlobalTensor<ElementWorkspaceO> gAccumOut;
        gAccumOut.SetGlobalBuffer((__gm__ ElementWorkspaceO *)(params.workSpace + wsOffset));
        wsOffset += accumOutSize_ * sizeof(float);
        AscendC::GlobalTensor<float> gSoftmaxMax;
        gSoftmaxMax.SetGlobalBuffer((__gm__ float *)(params.workSpace + wsOffset));
        wsOffset += lseStatSize_ * sizeof(float);
        AscendC::GlobalTensor<float> gSoftmaxSum;
        gSoftmaxSum.SetGlobalBuffer((__gm__ float *)(params.workSpace + wsOffset));

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        InitSyncFlags();
        AscendC::SyncAll<false>();
#ifdef __DAV_CUBE__
        coreIdx = AscendC::GetBlockIdx();
#endif
#ifdef __DAV_VEC__
        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
#endif
        Phase1KvCentricCompute(
            coreIdx, coreNum, gQ, gK, gV, gK2qRowPtr, gK2qQIndices,
            gK2qSlotIndices, gBlockTable, gAccumOut, gSoftmaxMax, gSoftmaxSum);
        AscendC::SyncAll();
        DumpPhase1OPartial(gAccumOut, gSoftmaxMax, gSoftmaxSum, gO);
#ifdef __DAV_VEC__
        Phase2CombineScale(AscendC::GetBlockIdx(), gO, gAccumOut, gSoftmaxMax, gSoftmaxSum);
#endif
        ReleaseSyncFlags();
    }

private:
    __aicore__ inline
    void FetchBaseShapeInfo(
        __gm__ SparseAttnPrefill::SparseAttentionScorePrefillTilingData *tilingData)
    {
        batch_ = tilingData->batch;
        qHeads_ = tilingData->numHeads;
        kvHeads_ = tilingData->kvHeads;
        groupSize_ = tilingData->groupSize;
        embed_ = tilingData->embeddingSize;
        blockSize_ = tilingData->blockSize;
        topK_ = tilingData->topK;
        totalKvRows_ = tilingData->numKvBlocks;
        maxBlocksPerBatch_ = tilingData->maxBlocksPerBatch;
        k2qNnzUpperBound_ = tilingData->k2qNnzUpperBound;
        totalTaskNumP1_ = tilingData->totalTaskNumP1;
        totalTaskNumP2_ = tilingData->totalTaskNumP2;
        scaleValue_ = tilingData->scaleValue;
        accumOutSize_ = tilingData->accumOutSize;
        lseStatSize_ = tilingData->lseStatSize;
        slotOElems_ = static_cast<uint64_t>(groupSize_) * embed_;
        slotStatElems_ = static_cast<uint64_t>(groupSize_);
    }

    __aicore__ inline
    uint32_t KvRowsPerBatch(uint32_t batchIdx)
    {
        return CeilDiv(static_cast<uint32_t>(gActualKvseqlen_.GetValue(batchIdx)), blockSize_);
    }

    __aicore__ inline
    void InitPackedRowCoord(uint32_t &batchIdx, uint32_t &kvBlockIdx)
    {
        kvBlockIdx = 0;
        batchIdx = 0;
        while (batchIdx < batch_ && kvBlockIdx >= KvRowsPerBatch(batchIdx)) {
            batchIdx++;
        }
    }

    __aicore__ inline
    void AdvancePackedRowCoord(uint32_t &batchIdx, uint32_t &kvBlockIdx)
    {
        batchIdx++;
        while (batchIdx < batch_ && kvBlockIdx >= KvRowsPerBatch(batchIdx)) {
            batchIdx++;
        }
        if (batchIdx >= batch_) {
            kvBlockIdx++;
            batchIdx = 0;
            while (batchIdx < batch_ && kvBlockIdx >= KvRowsPerBatch(batchIdx)) {
                batchIdx++;
            }
        }
    }

    __aicore__ inline
    void FindBatchAndLocalQ(uint32_t qToken, uint32_t &qBatch, uint32_t &localQIdx)
    {
        uint32_t accum = 0;
        for (uint32_t b = 0; b < batch_; b++) {
            uint32_t batchLen = static_cast<uint32_t>(gActualQseqlen_.GetValue(b));
            if (qToken < accum + batchLen) {
                qBatch = b;
                localQIdx = qToken - accum;
                return;
            }
            accum += batchLen;
        }
        qBatch = batch_ - 1;
        localQIdx = qToken - accum;
    }

    __aicore__ inline
    uint32_t CalcKvBlockValidSize(uint32_t batchIdx, uint32_t localBlockIdx)
    {
        uint32_t kvSeqlenBatch = static_cast<uint32_t>(gActualKvseqlen_.GetValue(batchIdx));
        uint32_t numBlocksB = CeilDiv(kvSeqlenBatch, blockSize_);
        if (localBlockIdx == numBlocksB - 1) {
            return kvSeqlenBatch - localBlockIdx * blockSize_;
        }
        return blockSize_;
    }

    __aicore__ inline
    uint32_t CalcCausalValidLen(
        uint32_t qToken, uint32_t validSize, uint32_t kvStartPos)
    {
        uint32_t qBatch = 0;
        uint32_t localQIdx = 0;
        FindBatchAndLocalQ(qToken, qBatch, localQIdx);
        uint32_t kvSeqlen = static_cast<uint32_t>(gActualKvseqlen_.GetValue(qBatch));
        uint32_t qSeqlen = static_cast<uint32_t>(gActualQseqlen_.GetValue(qBatch));
        uint32_t qPosition = kvSeqlen - qSeqlen + localQIdx;
        if (qPosition < kvStartPos) {
            return 0U;
        }
        uint32_t maxLen = qPosition - kvStartPos + 1;
        return (maxLen < validSize) ? maxLen : validSize;
    }

    __aicore__ inline
    void CopyPartialStatsToGm(AscendC::GlobalTensor<float> &gSoftmaxMax,
                              AscendC::GlobalTensor<float> &gSoftmaxSum,
                              AscendC::LocalTensor<float> &rowMaxLocal,
                              AscendC::LocalTensor<float> &rowSumLocal,
                              uint64_t lseOffset,
                              uint32_t rowOffset,
                              uint32_t rowCount)
    {
        if (rowCount == 0U) {
            return;
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
        AscendC::DataCopyPad(
            gSoftmaxMax[lseOffset + rowOffset], rowMaxLocal,
            {1, static_cast<uint16_t>(rowCount * sizeof(float)), 0, 0});
        AscendC::DataCopyPad(
            gSoftmaxSum[lseOffset + rowOffset], rowSumLocal,
            {1, static_cast<uint16_t>(rowCount * sizeof(float)), 0, 0});
    }

    // Debug: copy O_partial / max / sum after Phase1 + SyncAll.
    __aicore__ inline
    void DumpPhase1OPartial(AscendC::GlobalTensor<ElementWorkspaceO> &gAccumOut,
                            AscendC::GlobalTensor<float> &gSoftmaxMax,
                            AscendC::GlobalTensor<float> &gSoftmaxSum,
                            AscendC::GlobalTensor<ElementO> &gO)
    {
        (void)gAccumOut;
#ifdef __DAV_VEC__
        if (AscendC::GetBlockIdx() != 0U) {
            return;
        }
        if (lseStatSize_ == 0U) {
            return;
        }
        static constexpr uint32_t PHASE1_O_DUMP_ELEMS = 8U;
        static constexpr uint32_t PHASE1_STAT_DUMP_ELEMS = 8U;
        AscendC::LocalTensor<float> ubDump =
            resource.ubBuf.template GetBufferByByte<float>(0);
        uint32_t totalQTokens = totalTaskNumP2_ / kvHeads_;
        for (uint32_t qToken = 0U; qToken < totalQTokens; ++qToken) {
            for (uint32_t kvHeadIdx = 0U; kvHeadIdx < kvHeads_; ++kvHeadIdx) {
                for (uint32_t slotK = 0U; slotK < topK_; ++slotK) {
                    uint64_t oOff = SlotOOffset(qToken, kvHeadIdx, slotK);
                    uint64_t lseOff = SlotStatOffset(qToken, kvHeadIdx, slotK, 0U);
                    //AscendC::DumpTensor(gAccumOut[oOff], 123, PHASE1_O_DUMP_ELEMS);
                    /*AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::DumpTensor(ubDump, 456U, PHASE1_O_DUMP_ELEMS);
                    AscendC::DumpTensor(gSoftmaxMax[lseOff], 123U, PHASE1_STAT_DUMP_ELEMS);
                    AscendC::DumpTensor(gSoftmaxSum[lseOff], 456U, PHASE1_STAT_DUMP_ELEMS);*/
                }
            }
        }
#endif
    }

    // Phase 1: KV-centric partial compute (Cube QK/PV + Vector softmax).
    __aicore__ inline
    void Phase1KvCentricCompute(
        uint32_t coreIdx, uint32_t coreNum,
        AscendC::GlobalTensor<ElementQ> const &gQ,
        AscendC::GlobalTensor<ElementK> const &gK,
        AscendC::GlobalTensor<ElementV> const &gV,
        AscendC::GlobalTensor<int32_t> const &gK2qRowPtr,
        AscendC::GlobalTensor<int32_t> const &gK2qQIndices,
        AscendC::GlobalTensor<int32_t> const &gK2qSlotIndices,
        AscendC::GlobalTensor<int32_t> const &gBlockTable,
        AscendC::GlobalTensor<ElementWorkspaceO> &gAccumOut,
        AscendC::GlobalTensor<float> &gSoftmaxMax,
        AscendC::GlobalTensor<float> &gSoftmaxSum)
    {
        uint32_t mPerSubCore = VecRowSplit(groupSize_);
        uint32_t sColMax = blockSize_;

#ifdef __DAV_CUBE__
        uint32_t mAligned = RoundUp(mPerSubCore, 16u);
        uint32_t ubSStageElems = mAligned * sColMax;

        BlockMmadQK blockMmadQK;
        BlockMmadPV blockMmadPV;

        blockMmadQK.Init(resource, blockSize_, embed_);
        uint32_t qkL1Used = blockSize_ * embed_ * sizeof(ElementK)
            + BlockMmadQK::L0_TILE_M * embed_ * sizeof(ElementQ);
        blockMmadPV.Init(resource, qkL1Used, blockSize_, embed_, groupSize_);

        AscendC::LocalTensor<float> ubSTensor[UB_S_OTMP_BUF_STAGES];
        for (uint32_t i = 0; i < UB_S_OTMP_BUF_STAGES; i++) {
            ubSTensor[i] = resource.ubBuf.template GetBufferByByte<float>(ubSStageElems * sizeof(float) * i);
        }

        int64_t strideKVRow = static_cast<int64_t>(kvHeads_) * embed_;
        auto ubSLayout = tla::MakeLayout<ElementS, LayoutS>(mAligned, sColMax);
#endif
#ifdef __DAV_VEC__
        uint32_t vecRows = VecNumRows(groupSize_);
        uint32_t vecRowOffset = VecGlobalRowOffset(groupSize_);
        uint32_t mAligned = RoundUp(VecRowSplit(groupSize_), 16u);
        uint32_t ubSStageElems = mAligned * sColMax;
        uint32_t vecUbPStageElems = (RoundUp(vecRows, 16U) + 1U) * sColMax;

        AscendC::LocalTensor<float> ubSTensor[UB_S_OTMP_BUF_STAGES];
        AscendC::LocalTensor<ElementP> ubPBuf[UB_S_OTMP_BUF_STAGES];
        for (uint32_t i = 0; i < UB_S_OTMP_BUF_STAGES; i++) {
            ubSTensor[i] = resource.ubBuf.template GetBufferByByte<float>(ubSStageElems * sizeof(float) * i);
            ubPBuf[i] = resource.ubBuf.template GetBufferByByte<ElementP>(
                ubSStageElems * sizeof(float) * UB_S_OTMP_BUF_STAGES
                + vecUbPStageElems * sizeof(ElementP) * i);
        }
        uint32_t ubWsMetaOffset = ubSStageElems * sizeof(float) * UB_S_OTMP_BUF_STAGES
            + vecUbPStageElems * sizeof(ElementP) * UB_S_OTMP_BUF_STAGES;
        AscendC::LocalTensor<float> ubRowMaxStage[P_L1_BUF_NUM];
        AscendC::LocalTensor<float> ubRowSumStage[P_L1_BUF_NUM];
        AscendC::LocalTensor<float> ubTmpBuf =
            resource.ubBuf.template GetBufferByByte<float>(
                ubWsMetaOffset + RoundUp(vecRows, 16U) * sizeof(float) * 2U * P_L1_BUF_NUM);
        for (uint32_t i = 0; i < P_L1_BUF_NUM; i++) {
            ubRowMaxStage[i] = resource.ubBuf.template GetBufferByByte<float>(
                ubWsMetaOffset + RoundUp(vecRows, 16U) * sizeof(float) * 2U * i);
            ubRowSumStage[i] = ubRowMaxStage[i][RoundUp(vecRows, 16U)];
        }

        EpilogueOnlineSoftmax epilogueSoftmax;
        epilogueSoftmax.SetScale(scaleValue_);

        uint32_t qkL1Used = blockSize_ * embed_ * sizeof(ElementK)
            + BlockMmadQK::L0_TILE_M * embed_ * sizeof(ElementQ);
        uint32_t vBufBytes = blockSize_ * embed_ * sizeof(ElementV);
        uint32_t l1PStageBytes = RoundUp(groupSize_, 16U) * RoundUp(blockSize_, 16U) * sizeof(ElementP);
        AscendC::LocalTensor<ElementP> l1PBuf[P_L1_BUF_NUM];
        for (uint32_t i = 0; i < P_L1_BUF_NUM; i++) {
            l1PBuf[i] = resource.l1Buf.template GetBufferByByte<ElementP>(
                qkL1Used + vBufBytes + l1PStageBytes * i);
        }
#endif

        int32_t curPackedRow = -1;
        uint32_t batchIdx = 0;
        uint32_t kvBlockIdx = 0;

        for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNumP1_; taskIdx += coreNum) {
            uint32_t packedRow = taskIdx / kvHeads_;
            uint32_t kvHeadIdx = taskIdx % kvHeads_;
            if (packedRow >= totalKvRows_) {
                continue;
            }

            if (static_cast<int32_t>(packedRow) != curPackedRow) {
                if (curPackedRow < 0) {
                    InitPackedRowCoord(batchIdx, kvBlockIdx);
                    curPackedRow = 0;
                }
                while (curPackedRow < static_cast<int32_t>(packedRow)) {
                    AdvancePackedRowCoord(batchIdx, kvBlockIdx);
                    curPackedRow++;
                }
            }
            uint32_t localBlockIdx = kvBlockIdx;

            int64_t btOffset =
                static_cast<int64_t>(batchIdx) * maxBlocksPerBatch_ + localBlockIdx;
            int32_t physicalBlockId = gBlockTable.GetValue(btOffset);

            uint32_t validSize = CalcKvBlockValidSize(batchIdx, localBlockIdx);
            if (validSize == 0) {
                continue;
            }

            uint64_t rowPtrBase = static_cast<uint64_t>(kvHeadIdx) * (totalKvRows_ + 1);
            uint32_t csrStart = static_cast<uint32_t>(
                gK2qRowPtr.GetValue(rowPtrBase + packedRow));
            uint32_t csrEnd = static_cast<uint32_t>(
                gK2qRowPtr.GetValue(rowPtrBase + packedRow + 1));
            uint32_t numQTokens = csrEnd - csrStart;
            if (numQTokens == 0) {
                continue;
            }
            
            uint64_t csrDataBase = static_cast<uint64_t>(kvHeadIdx) * k2qNnzUpperBound_;
            uint32_t sColsAligned = sColMax;
            uint32_t kvStartPos = localBlockIdx * blockSize_;

#ifdef __DAV_CUBE__
            uint64_t kvBlockBase =
                static_cast<uint64_t>(physicalBlockId) * blockSize_ * kvHeads_ * embed_
                + static_cast<uint64_t>(kvHeadIdx) * embed_;

            auto gmKLayout = tla::MakeLayout<ElementK, LayoutK>(strideKVRow, blockSize_);
            auto gmKTensor = tla::MakeTensor(gK[kvBlockBase], gmKLayout, Arch::PositionGM{});
            blockMmadQK.LoadKResident(gmKTensor, validSize, embed_);

            auto gmVLayout = tla::MakeLayout<ElementV, LayoutV>(blockSize_, strideKVRow);
            auto gmVTensor = tla::MakeTensor(gV[kvBlockBase], gmVLayout, Arch::PositionGM{});
            blockMmadPV.LoadVResident(gmVTensor, validSize, embed_);
#endif

            for (uint32_t qi = 0; qi < numQTokens + PRE_LAUNCH; qi++) {
                if (qi < numQTokens) {
                    uint32_t csrIdx = csrStart + qi;
                    uint32_t qToken = static_cast<uint32_t>(
                        gK2qQIndices.GetValue(csrDataBase + csrIdx));
                    uint32_t causalValidLen = CalcCausalValidLen(qToken, validSize, kvStartPos);
                    if (causalValidLen != 0U) {
                        uint32_t ubSBufId = qi % UB_S_OTMP_BUF_STAGES;
                        uint32_t l1PBufId = qi % P_L1_BUF_NUM;
                        uint32_t mm1ToSmFlagId = ubSBufId;
                        uint32_t smToMm2FlagId = l1PBufId + UB_S_OTMP_BUF_STAGES;
                        Arch::CrossCoreFlag mm1ToSmFlag(mm1ToSmFlagId);
                        Arch::CrossCoreFlag smToMm2Flag(smToMm2FlagId);

                        uint32_t qHeadStart = kvHeadIdx * groupSize_;
                        uint64_t qBase = static_cast<uint64_t>(qToken) * qHeads_ * embed_
                            + static_cast<uint64_t>(qHeadStart) * embed_;

#ifdef __DAV_CUBE__
                        auto ubSTensorTla = tla::MakeTensor(
                            ubSTensor[ubSBufId], ubSLayout, Arch::PositionUB{});
                        auto gmQLayout = tla::MakeLayout<ElementQ, LayoutQ>(groupSize_, embed_);
                        auto gmQTensor = tla::MakeTensor(gQ[qBase], gmQLayout, Arch::PositionGM{});
                        blockMmadQK(gmQTensor, ubSTensorTla, groupSize_, validSize, embed_, mm1ToSmFlag);
#endif
#ifdef __DAV_VEC__
                        epilogueSoftmax.Init(ubTmpBuf, ubRowMaxStage[l1PBufId], ubRowSumStage[l1PBufId]);
                        auto l1PLayout = tla::MakeLayout<ElementP, layout::zN>(groupSize_, causalValidLen);
                        auto l1PTensor = tla::MakeTensor(l1PBuf[l1PBufId], l1PLayout, Arch::PositionL1{});
                        //if (vecRows > 0)
                        //    AscendC::DumpTensor(ubSTensor[ubSBufId], 12, 8);
                        epilogueSoftmax(
                            ubSTensor[ubSBufId], ubPBuf[ubSBufId], l1PTensor, groupSize_, vecRows,
                            causalValidLen, sColsAligned,
                            mm1ToSmFlag, smToMm2Flag);
                        //if (vecRows > 0)
                        //    AscendC::DumpTensor(ubPBuf[ubSBufId], 34, 8);
                        uint32_t slotK = static_cast<uint32_t>(
                            gK2qSlotIndices.GetValue(csrDataBase + csrIdx));
                        uint64_t lseOffset = SlotStatOffset(qToken, kvHeadIdx, slotK, vecRowOffset);
                        CopyPartialStatsToGm(
                            gSoftmaxMax, gSoftmaxSum, ubRowMaxStage[l1PBufId], ubRowSumStage[l1PBufId],
                            lseOffset, 0U, vecRows);
                        //if (vecRows > 0)
                        //    AscendC::DumpTensor(ubRowMaxStage[l1PBufId], 56, 8);
                        //if (vecRows > 0)
                        //    AscendC::DumpTensor(ubRowSumStage[l1PBufId], 78, 8);
#endif
                    }
                }

                if (qi >= PRE_LAUNCH && qi - PRE_LAUNCH < numQTokens) {
                    uint32_t qiDe = qi - PRE_LAUNCH;
                    uint32_t csrIdx = csrStart + qiDe;
                    uint32_t qToken = static_cast<uint32_t>(
                        gK2qQIndices.GetValue(csrDataBase + csrIdx));
                    uint32_t slotK = static_cast<uint32_t>(
                        gK2qSlotIndices.GetValue(csrDataBase + csrIdx));
                    uint32_t causalValidLenDe = CalcCausalValidLen(qToken, validSize, kvStartPos);
                    if (causalValidLenDe != 0U) {
                        uint64_t wsOOff = SlotOOffset(qToken, kvHeadIdx, slotK);

                        uint32_t l1PBufIdDe = qiDe % P_L1_BUF_NUM;
                        uint32_t smToMm2FlagId = l1PBufIdDe + UB_S_OTMP_BUF_STAGES;
                        Arch::CrossCoreFlag smToMm2Flag(smToMm2FlagId);

#ifdef __DAV_CUBE__
                        auto gmOLayout = tla::MakeLayout<ElementWorkspaceO, LayoutO>(groupSize_, embed_);
                        auto gmOTensor = tla::MakeTensor(
                            gAccumOut[wsOOff], gmOLayout, Arch::PositionGM{});
                        blockMmadPV.SetL1PBuf(l1PBufIdDe);
                        blockMmadPV(gmOTensor, groupSize_, causalValidLenDe, embed_, smToMm2Flag);
                        //AscendC::DumpTensor(gAccumOut[wsOOff], 789 + qiDe, 8);
#endif
                    }
                }
            }
#ifdef __DAV_CUBE__
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
#endif
        }
    }

    // Phase 2: FlashDecode-style combine (IFA FlashDecodeCompute / CombineSplitKVRes).
    __aicore__ inline
    void Phase2CombineScale(
        uint32_t tmpBlockIdx,
        AscendC::GlobalTensor<ElementO> &gO,
        AscendC::GlobalTensor<ElementWorkspaceO> &gAccumOut,
        AscendC::GlobalTensor<float> &gSoftmaxMax,
        AscendC::GlobalTensor<float> &gSoftmaxSum)
    {
#ifdef __DAV_VEC__
        AscendC::LocalTensor<float> ubBase =
            resource.ubBuf.template GetBufferByByte<float>(0);

        EpilogueRescaleO epilogueRescaleO;
        epilogueRescaleO.InitFDBuffers(ubBase, embed_, groupSize_, topK_, kvHeads_);
        epilogueRescaleO.FlashDecodeCompute(
            tmpBlockIdx, totalTaskNumP2_,
            gO, gAccumOut, gSoftmaxMax, gSoftmaxSum,
            qHeads_, kvHeads_);
#endif
    }

private:
    Arch::Resource<ArchTag> resource;

    uint32_t batch_;
    uint32_t qHeads_;
    uint32_t kvHeads_;
    uint32_t groupSize_;
    uint32_t embed_;
    uint32_t blockSize_;
    uint32_t topK_;
    uint32_t totalKvRows_;
    uint32_t maxBlocksPerBatch_;
    uint32_t k2qNnzUpperBound_;
    uint32_t totalTaskNumP1_;
    uint32_t totalTaskNumP2_;
    float scaleValue_;
    uint64_t accumOutSize_;
    uint64_t lseStatSize_;
    uint64_t slotOElems_;
    uint64_t slotStatElems_;

    AscendC::GlobalTensor<int32_t> gActualQseqlen_;
    AscendC::GlobalTensor<int32_t> gActualKvseqlen_;
};

}  // namespace SasaPrefillKernelArch35

#endif  // SPARSE_ATTENTION_SCORE_PREFILL_KERNEL_ARCH35_H
