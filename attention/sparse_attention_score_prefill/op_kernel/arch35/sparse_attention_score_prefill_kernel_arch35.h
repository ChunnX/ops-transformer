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
    using ElementS = bfloat16_t;
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
    // Workspace softmaxMax/softmaxSum init: -inf for rowMax, 0 for rowSum so that
    // slots not written by Phase1 are treated as invalid by Phase2 combine.
    static constexpr float WS_ROWMAX_INIT = -3.4028235e38f;
    static constexpr float WS_ROWSUM_INIT = 0.0f;
    static constexpr uint32_t WS_INIT_EVT = 5U;          // free V_MTE3/MTE3_V id on VEC
    static constexpr uint32_t WS_INIT_CHUNK = 4096U;      // fp32 elems per chunk (multiple of 8)

    // gmUb/glUb are 2-deep by ubSBufId inside the prefill softmax epilogue (see
    // reg_low_prec_bf16.hpp ctor). CopyPartialStatsToGm (MTE3) reads gmUb[ubSBufId]/
    // glUb[ubSBufId] right after each softmax; the epilogue waits MTE3_V(ubSBufId)
    // before overwriting the same stage, and this kernel sets MTE3_V(ubSBufId) after
    // the copy. InitSyncFlags pre-sets MTE3_V(0,1) so the first block per stage is
    // unblocked. No single-buffer reuse -> no race.
    // Offsets alias the prefill softmax ctor's 2-deep gmUb/glUb regions (must stay
    // byte-identical to that ctor). gmUb = this AIV's rowMax (fp32), glUb = rowSum.
    static constexpr uint32_t SM_UB_BLOCK = 32768U;  // UB_UINT8_BLOCK_SIZE
    static constexpr uint32_t SM_UB_GM_OFFSET = 7U * SM_UB_BLOCK;                       // 229376 rowMax base
    static constexpr uint32_t SM_UB_GL_OFFSET = SM_UB_GM_OFFSET + 2U * 64U * sizeof(float);  // 229888 rowSum base
    static constexpr uint32_t SM_UB_STAGE_BYTES = 64U * sizeof(float);  // 256, per-stage stride

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
        // VEC-side workspace init: prefill softmaxMax<-(-inf), softmaxSum<-0 so
        // unwritten (invalid) slots are detected by Phase2. Placed before
        // SyncAll<false> so that barrier doubles as the init-completion sync
        // (no extra SyncAll needed). CUBE cores idle here.
        InitWorkspaceStats(gSoftmaxMax, gSoftmaxSum);
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
        //DumpPhase1OPartial(gAccumOut, gSoftmaxMax, gSoftmaxSum, gO);
#ifdef __DAV_VEC__
        Phase2CombineScale(gO, gAccumOut, gSoftmaxMax, gSoftmaxSum);
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

    // VEC-only: prefill contiguous softmaxMax<-(-inf) and softmaxSum<-0 in
    // WS_INIT_CHUNK blocks (tail chunk uses remainder length). UB filled once
    // per buffer; one loop writes both GM regions per chunk. Work split across
    // VEC sub-blocks by chunk index.
    __aicore__ inline
    void InitWorkspaceStats(AscendC::GlobalTensor<float> &gSoftmaxMax,
                            AscendC::GlobalTensor<float> &gSoftmaxSum)
    {
        if (lseStatSize_ == 0U) {
            return;
        }
#ifdef __DAV_VEC__
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t totalSubBlocks = AscendC::GetBlockNum() * subBlockNum;
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t numFullChunks = static_cast<uint32_t>(lseStatSize_ / WS_INIT_CHUNK);
        uint32_t tailElems = static_cast<uint32_t>(lseStatSize_ % WS_INIT_CHUNK);
        uint32_t totalChunks = numFullChunks + (tailElems > 0U ? 1U : 0U);

        AscendC::LocalTensor<float> ubMax = resource.ubBuf.template GetBufferByByte<float>(0);
        AscendC::LocalTensor<float> ubSum = resource.ubBuf.template GetBufferByByte<float>(
            WS_INIT_CHUNK * sizeof(float));
        AscendC::Duplicate(ubMax, WS_ROWMAX_INIT, WS_INIT_CHUNK);
        AscendC::Duplicate(ubSum, WS_ROWSUM_INIT, WS_INIT_CHUNK);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(WS_INIT_EVT);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(WS_INIT_EVT);

        for (uint32_t chunkIdx = blockIdx; chunkIdx < totalChunks; chunkIdx += totalSubBlocks) {
            uint64_t off = static_cast<uint64_t>(chunkIdx) * WS_INIT_CHUNK;
            uint32_t cnt = (chunkIdx < numFullChunks) ? WS_INIT_CHUNK : tailElems;
            uint16_t blockLen = static_cast<uint16_t>(cnt * sizeof(float));
            AscendC::DataCopyPad(gSoftmaxMax[off], ubMax, {1, blockLen, 0, 0});
            AscendC::DataCopyPad(gSoftmaxSum[off], ubSum, {1, blockLen, 0, 0});
        }
        AscendC::PipeBarrier<PIPE_ALL>();
#endif
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
        for (uint32_t qToken = 0U; qToken < 1; ++qToken) {
            for (uint32_t kvHeadIdx = 0U; kvHeadIdx < 1; ++kvHeadIdx) {
                for (uint32_t slotK = 0U; slotK < topK_; ++slotK) {
                    uint64_t oOff = SlotOOffset(qToken, kvHeadIdx, slotK);
                    uint64_t lseOff = SlotStatOffset(qToken, kvHeadIdx, slotK, 0U);
                    //AscendC::DumpTensor(gAccumOut[oOff], 123, 128);
                    break;
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

#ifdef __DAV_CUBE__
        uint32_t mAligned = RoundUp(mPerSubCore, 16u);

        BlockMmadQK blockMmadQK;
        BlockMmadPV blockMmadPV;

        blockMmadQK.Init(resource, blockSize_, embed_);
        uint32_t qkL1Used = blockSize_ * embed_ * sizeof(ElementK)
            + BlockMmadQK::L0_TILE_M * embed_ * sizeof(ElementQ);
        blockMmadPV.Init(resource, qkL1Used, blockSize_, embed_, groupSize_);

        // CUBE Fixpipe dst aliases decode softmax lsUbTensor (offset 0,
        // MAX_UB_S_ELEM_NUM bf16 = 32768B/stage), shared cross-core with VEC softmax.
        AscendC::LocalTensor<ElementS> ubSTensor[UB_S_OTMP_BUF_STAGES];
        for (uint32_t i = 0; i < UB_S_OTMP_BUF_STAGES; i++) {
            ubSTensor[i] = resource.ubBuf.template GetBufferByByte<ElementS>(
                EpilogueOnlineSoftmax::MAX_UB_S_ELEM_NUM * sizeof(ElementS) * i);
        }

        int64_t strideKVRow = static_cast<int64_t>(kvHeads_) * embed_;
#endif
#ifdef __DAV_VEC__
        uint32_t vecRows = VecNumRows(groupSize_);
        uint32_t vecRowOffset = VecGlobalRowOffset(groupSize_);

        // decode bf16 regbase softmax self-allocates its UB (lsUb/lpUb/lmUb/gmUb/glUb/
        // dmUb/llUb at fixed offsets, see reg_low_prec_bf16.hpp ctor). VEC no longer
        // pre-allocates ubSTensor/ubPBuf/ubRowMax/ubRowSum/ubTmpBuf.
        EpilogueOnlineSoftmax epilogueSoftmax(resource, scaleValue_);

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
                        // per-block S stride = nRound = RoundUp(validSize,16), matches the
                        // decode softmax LoadAlign(srcUb + i*S2BaseSize) read stride.
                        auto ubSLayout = tla::MakeLayout<ElementS, LayoutS>(mAligned, RoundUp(validSize, 16U));
                        auto ubSTensorTla = tla::MakeTensor(
                            ubSTensor[ubSBufId], ubSLayout, Arch::PositionUB{});
                        auto gmQLayout = tla::MakeLayout<ElementQ, LayoutQ>(groupSize_, embed_);
                        auto gmQTensor = tla::MakeTensor(gQ[qBase], gmQLayout, Arch::PositionGM{});
                        blockMmadQK(gmQTensor, ubSTensorTla, groupSize_, validSize, embed_, mm1ToSmFlag);
#endif
#ifdef __DAV_VEC__
                        // P L1 layout extent = causalValidLen (not validSize): softmax only
                        // writes P[0, causalValidLen) and PV reduces over causalValidLenDe
                        // (== causalValidLen for the same qToken). The L1 P zN N-fractal stride
                        // must match the layout PV reads, so it must be causalValidLen here.
                        // (V's L1 layout still uses the block valid size internally - see
                        // BlockMmadPV residentValidSize_.)
                        auto l1PLayout = tla::MakeLayout<ElementP, layout::zN>(groupSize_, causalValidLen);
                        auto l1PTensor = tla::MakeTensor(l1PBuf[l1PBufId], l1PLayout, Arch::PositionL1{});
                        epilogueSoftmax(
                            l1PTensor,
                            GemmCoord{groupSize_, validSize, embed_},
                            ubSBufId,
                            l1PBufId,
                            mm1ToSmFlag,
                            smToMm2Flag,
                            causalValidLen);
                        uint32_t slotK = static_cast<uint32_t>(
                            gK2qSlotIndices.GetValue(csrDataBase + csrIdx));
                        uint64_t lseOffset = SlotStatOffset(qToken, kvHeadIdx, slotK, vecRowOffset);
                        // softmax wrote this AIV's rowMax/rowSum into its 2-deep
                        // gmUb[ubSBufId]/glUb[ubSBufId]. Copy to GM per-slot for Phase2.
                        AscendC::LocalTensor<float> gmUbAlias =
                            resource.ubBuf.template GetBufferByByte<float>(
                                SM_UB_GM_OFFSET + ubSBufId * SM_UB_STAGE_BYTES);
                        AscendC::LocalTensor<float> glUbAlias =
                            resource.ubBuf.template GetBufferByByte<float>(
                                SM_UB_GL_OFFSET + ubSBufId * SM_UB_STAGE_BYTES);
                        CopyPartialStatsToGm(
                            gSoftmaxMax, gSoftmaxSum, gmUbAlias, glUbAlias,
                            lseOffset, 0U, vecRows);
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
                        // if (qToken == 0 && kvHeadIdx == 0 && slotK == 0) {
                        //     AscendC::printf("1111111111111111111111 causalValidLenDe:%d\n", causalValidLenDe);
                        // }
                        // PV reduces over causalValidLenDe: softmax wrote P[0, causalValidLen)
                        // (== causalValidLenDe for this qToken) and the L1 P layout is scoped to
                        // causalValidLen, so PV's K must be causalValidLenDe to match (reading
                        // validSize would read the stale [causalValidLen, validSize) P tail).
                        blockMmadPV(gmOTensor, groupSize_, causalValidLenDe, embed_, smToMm2Flag);
                        // if (qToken == 0 && kvHeadIdx == 0 && slotK == 0) {
                        //     AscendC::DumpTensor(gV[kvBlockBase], 456, 16);
                        //     AscendC::DumpTensor(gAccumOut[wsOOff], 789 + qiDe, 128);
                        // }
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
        // VEC: GetBlockIdx() is the linear AIV id in [0, GetBlockNum()*GetSubBlockNum());
        // GetBlockNum() is the AIC block count (half of AIV count). Stride must cover
        // all AIV sub-blocks, not blockNum alone.
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        if (coreNum == 0U) {
            coreNum = 1U;
        }
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNumP2_; taskIdx += coreNum) {
            epilogueRescaleO.FlashDecodeCompute(
                taskIdx, totalTaskNumP2_,
                gO, gAccumOut, gSoftmaxMax, gSoftmaxSum,
                qHeads_, kvHeads_);
        }
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
