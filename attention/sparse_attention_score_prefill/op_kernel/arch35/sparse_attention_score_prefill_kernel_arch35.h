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

    static constexpr uint32_t MM1_TO_SM_FLAG_V0 = 0;
    static constexpr uint32_t MM1_TO_SM_FLAG_V1 = 16;
    static constexpr uint32_t SM_TO_MM2_FLAG_V0 = 2;
    static constexpr uint32_t SM_TO_MM2_FLAG_V1 = 18;
    static constexpr uint32_t MM2_TO_WS_FLAG_V0 = 5;
    static constexpr uint32_t MM2_TO_WS_FLAG_V1 = 21;
    static constexpr uint32_t V0_V1_FLAG_ID_OFFSET = 16;

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
        AscendC::CrossCoreSetFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_MTE1>(SM_TO_MM2_FLAG_V0);
        AscendC::CrossCoreSetFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_MTE1>(SM_TO_MM2_FLAG_V1);
#endif
#ifdef __DAV_VEC__
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::CrossCoreSetFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_V>(MM1_TO_SM_FLAG_V0);
        AscendC::CrossCoreSetFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_V>(MM1_TO_SM_FLAG_V1);
        AscendC::CrossCoreSetFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_V>(MM2_TO_WS_FLAG_V0);
        AscendC::CrossCoreSetFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_V>(MM2_TO_WS_FLAG_V1);
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
        AscendC::CrossCoreWaitFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_FIX>(MM1_TO_SM_FLAG_V0);
        AscendC::CrossCoreWaitFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_FIX>(MM1_TO_SM_FLAG_V1);
        AscendC::CrossCoreWaitFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_MTE1>(SM_TO_MM2_FLAG_V0);
        AscendC::CrossCoreWaitFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_MTE1>(SM_TO_MM2_FLAG_V1);
#endif
#ifdef __DAV_VEC__
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::CrossCoreWaitFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_V>(MM1_TO_SM_FLAG_V0);
        AscendC::CrossCoreWaitFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_V>(MM1_TO_SM_FLAG_V1);
        AscendC::CrossCoreWaitFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_MTE3>(SM_TO_MM2_FLAG_V0);
        AscendC::CrossCoreWaitFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_MTE3>(SM_TO_MM2_FLAG_V1);
        AscendC::CrossCoreWaitFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_MTE3>(MM2_TO_WS_FLAG_V0);
        AscendC::CrossCoreWaitFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_MTE3>(MM2_TO_WS_FLAG_V1);
#endif
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline
    static uint32_t VecRowSplit(uint32_t groupSize)
    {
        return RoundUp(groupSize, 8U) / 2U;
    }

    __aicore__ inline
    static uint32_t VecNumRows(uint32_t groupSize)
    {
        uint32_t split = VecRowSplit(groupSize);
        return (AscendC::GetSubBlockIdx() == 0U) ? split : (groupSize - split);
    }

    __aicore__ inline
    static uint32_t VecGlobalRowOffset(uint32_t groupSize)
    {
        return (AscendC::GetSubBlockIdx() == 0U) ? 0U : VecRowSplit(groupSize);
    }

    __aicore__ inline
    static uint32_t VecCrossCoreFlagId(uint32_t flagV0)
    {
        return flagV0 + AscendC::GetSubBlockIdx() * V0_V1_FLAG_ID_OFFSET;
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
        AscendC::GlobalTensor<ElementWorkspaceO> gWorkspaceO;
        gWorkspaceO.SetGlobalBuffer((__gm__ ElementWorkspaceO *)params.workSpace);
        AscendC::GlobalTensor<float> gWorkspaceF;
        gWorkspaceF.SetGlobalBuffer((__gm__ float *)params.workSpace);

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
            gK2qSlotIndices, gBlockTable, gWorkspaceO, gWorkspaceF);
        AscendC::SyncAll<false>();
        Phase2CombineScale(coreIdx, coreNum, gO, gWorkspaceO, gWorkspaceF);
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
        partialSize_ = tilingData->partialSize;
        uint64_t oBytes = static_cast<uint64_t>(groupSize_) * embed_ * sizeof(ElementWorkspaceO);
        oBytesAligned_ = oBytes;
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
        AscendC::GlobalTensor<ElementWorkspaceO> &gWorkspaceO,
        AscendC::GlobalTensor<float> &gWorkspaceF)
    {
        uint32_t mPerSubCore = VecRowSplit(groupSize_);
        // CastTransdataDeconflict requires compile-time srcN=128 and ND stride 128.
        uint32_t sColMax = blockSize_;
        uint32_t oColAligned = RoundUp(embed_, 16u);

#ifdef __DAV_CUBE__
        uint32_t mAligned = RoundUp(mPerSubCore, 16u);
        uint32_t rowNumRound = RoundUp(groupSize_, 16u);
        uint32_t cubeUbSSize = mAligned * sColMax;
        uint32_t cubeUbOSize = rowNumRound * oColAligned;

        BlockMmadQK blockMmadQK;
        BlockMmadPV blockMmadPV;

        blockMmadQK.Init(resource, blockSize_, embed_);
        uint32_t qkL1Used = blockSize_ * embed_ * sizeof(ElementK)
            + BlockMmadQK::L0_TILE_M * embed_ * sizeof(ElementQ);
        blockMmadPV.Init(resource, qkL1Used, blockSize_, embed_, groupSize_);

        AscendC::LocalTensor<float> ubSBuf =
            resource.ubBuf.template GetBufferByByte<float>(0);
        AscendC::LocalTensor<ElementOTmp> ubOBuf =
            resource.ubBuf.template GetBufferByByte<ElementOTmp>(
                cubeUbSSize * sizeof(float));
        AscendC::LocalTensor<ElementP> ubPBuf =
            resource.ubBuf.template GetBufferByByte<ElementP>(
                cubeUbSSize * sizeof(float) + cubeUbOSize * sizeof(ElementOTmp));

        int64_t strideKVRow = static_cast<int64_t>(kvHeads_) * embed_;
        Arch::CrossCoreFlag mm1ToSmFlag(MM1_TO_SM_FLAG_V0);
        Arch::CrossCoreFlag smToMm2Flag(SM_TO_MM2_FLAG_V0);
        Arch::CrossCoreFlag mm2ToWsFlag(MM2_TO_WS_FLAG_V0);
#endif
#ifdef __DAV_VEC__
        uint32_t vecRows = VecNumRows(groupSize_);
        uint32_t vecRowOffset = VecGlobalRowOffset(groupSize_);
        uint32_t vecMAligned = RoundUp(vecRows, 16u);
        uint32_t vecUbSSize = vecMAligned * sColMax;
        uint32_t vecUbOSize = vecMAligned * oColAligned;
        uint32_t vecUbPSize = (vecMAligned + 1U) * sColMax;

        AscendC::LocalTensor<float> ubSBuf =
            resource.ubBuf.template GetBufferByByte<float>(0);
        AscendC::printf("hhhhhhhhhhhhhhhhhhhhhhhh:%d, %d\n", vecUbSSize * sizeof(float),
                        vecUbSSize * sizeof(float) + vecUbOSize * sizeof(ElementOTmp));
        AscendC::LocalTensor<ElementOTmp> ubOBuf =
            resource.ubBuf.template GetBufferByByte<ElementOTmp>(
                vecUbSSize * sizeof(float));
        AscendC::LocalTensor<ElementP> ubPBuf =
            resource.ubBuf.template GetBufferByByte<ElementP>(
                vecUbSSize * sizeof(float) + vecUbOSize * sizeof(ElementOTmp));
        AscendC::LocalTensor<float> ubTmpBuf =
            resource.ubBuf.template GetBufferByByte<float>(
                vecUbSSize * sizeof(float) + vecUbOSize * sizeof(ElementOTmp)
                + vecUbPSize * sizeof(ElementP));
        uint32_t ubWsMetaOffset = vecUbSSize * sizeof(float) + vecUbOSize * sizeof(ElementOTmp)
            + vecUbPSize * sizeof(ElementP) + 256 * sizeof(float);
        AscendC::LocalTensor<float> ubWsMeta =
            resource.ubBuf.template GetBufferByByte<float>(ubWsMetaOffset);
        AscendC::LocalTensor<float> ubLse = ubWsMeta;
        AscendC::LocalTensor<float> ubRowSum = ubWsMeta[vecMAligned];

        EpilogueOnlineSoftmax epilogueSoftmax;
        epilogueSoftmax.Init(ubTmpBuf, ubLse, ubRowSum);
        epilogueSoftmax.SetScale(scaleValue_);

        uint32_t qkL1Used = blockSize_ * embed_ * sizeof(ElementK)
            + BlockMmadQK::L0_TILE_M * embed_ * sizeof(ElementQ);
        uint32_t vBufBytes = blockSize_ * embed_ * sizeof(ElementV);
        AscendC::LocalTensor<ElementP> l1PBuf =
            resource.l1Buf.template GetBufferByByte<ElementP>(qkL1Used + vBufBytes);
        AscendC::printf("--------------vec init---------------:%d, %d\n", qkL1Used, vBufBytes);
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

            auto ubSLayout = tla::MakeLayout<ElementS, LayoutS>(mAligned, sColsAligned);
            auto ubSTensor = tla::MakeTensor(ubSBuf, ubSLayout, Arch::PositionUB{});
#endif

            for (uint32_t qi = 0; qi < numQTokens; qi++) {
                uint32_t csrIdx = csrStart + qi;
                uint32_t qToken = static_cast<uint32_t>(
                    gK2qQIndices.GetValue(csrDataBase + csrIdx));
                uint32_t slotK = static_cast<uint32_t>(
                    gK2qSlotIndices.GetValue(csrDataBase + csrIdx));

                uint32_t qBatch = 0;
                uint32_t localQIdx = 0;
                FindBatchAndLocalQ(qToken, qBatch, localQIdx);
                uint32_t kvSeqlen = static_cast<uint32_t>(gActualKvseqlen_.GetValue(qBatch));
                uint32_t qSeqlen = static_cast<uint32_t>(gActualQseqlen_.GetValue(qBatch));
                uint32_t qPosition = kvSeqlen - qSeqlen + localQIdx;

                uint32_t causalValidLen = 0;
                if (qPosition >= kvStartPos) {
                    uint32_t maxLen = qPosition - kvStartPos + 1;
                    causalValidLen = (maxLen < validSize) ? maxLen : validSize;
                }
                if (causalValidLen == 0) {
                    continue;
                }

                uint64_t wsBytes = (static_cast<uint64_t>(qToken) * kvHeads_ * topK_
                    + kvHeadIdx * topK_ + slotK) * partialSize_;
                uint64_t wsOOff = wsBytes / sizeof(ElementWorkspaceO);
                uint64_t wsFOff = (wsBytes + oBytesAligned_) / sizeof(float);

                uint32_t qHeadStart = kvHeadIdx * groupSize_;
                uint64_t qBase = static_cast<uint64_t>(qToken) * qHeads_ * embed_
                    + static_cast<uint64_t>(qHeadStart) * embed_;

#ifdef __DAV_CUBE__
                auto gmQLayout = tla::MakeLayout<ElementQ, LayoutQ>(groupSize_, embed_);
                auto gmQTensor = tla::MakeTensor(gQ[qBase], gmQLayout, Arch::PositionGM{});
                blockMmadQK(gmQTensor, ubSTensor, groupSize_, validSize, embed_, mm1ToSmFlag);
#endif
#ifdef __DAV_VEC__
                Arch::CrossCoreFlag mm1ToSmFlag(VecCrossCoreFlagId(MM1_TO_SM_FLAG_V0));
                Arch::CrossCoreFlag smToMm2Flag(VecCrossCoreFlagId(SM_TO_MM2_FLAG_V0));

                auto l1PLayout = tla::MakeLayout<ElementP, layout::zN>(groupSize_, validSize);
                auto l1PTensor = tla::MakeTensor(l1PBuf, l1PLayout, Arch::PositionL1{});
                AscendC::DumpTensor(ubSBuf, 123, 16);
                epilogueSoftmax(
                    ubSBuf, ubPBuf, l1PTensor, groupSize_, vecRows,
                    validSize, causalValidLen, sColsAligned,
                    mm1ToSmFlag, smToMm2Flag);
                AscendC::DumpTensor(ubPBuf, 456, 16);
#endif
#ifdef __DAV_CUBE__
                Arch::CrossCoreFlag smToMm2Flag(SM_TO_MM2_FLAG_V0);
                Arch::CrossCoreFlag mm2ToWsFlag(MM2_TO_WS_FLAG_V0);

                auto ubOLayout = tla::MakeLayout<ElementOTmp, LayoutO>(rowNumRound, oColAligned);
                auto ubOTensor = tla::MakeTensor(ubOBuf, ubOLayout, Arch::PositionUB{});
                blockMmadPV(ubOTensor, groupSize_, validSize, embed_,
                    smToMm2Flag, mm2ToWsFlag);
#endif
#ifdef __DAV_VEC__
                AscendC::printf("dddddddddddddddddddddd\n");
                Arch::CrossCoreFlag mm2ToWsFlag(VecCrossCoreFlagId(MM2_TO_WS_FLAG_V0));
                Arch::CrossCoreWaitFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_V>(mm2ToWsFlag);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::printf("eeeeeeeeeeeeeeeeeeeeeeee\n");
                if (vecRows > 0U) {
                    AscendC::printf("fffffffffffffffffffff:%f\n", ubPBuf.GetValue(0));
                    AscendC::printf("ggggggggggggggggggggg:%f\n", ubOBuf.GetValue(0));
                    AscendC::DumpTensor(ubOBuf, 789, 16);
                    int32_t dLen = static_cast<int32_t>(embed_);
                    for (uint32_t gh = 0; gh < vecRows; gh++) {
                        float rowSum = ubRowSum.GetValue(gh);
                        if (rowSum > 0.0f) {
                            float invSum = 1.0f / rowSum;
                            AscendC::Muls(
                                ubOBuf[gh * oColAligned], ubOBuf[gh * oColAligned],
                                invSum, dLen);
                        }
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    AscendC::DataCopy(gWorkspaceF[wsFOff + vecRowOffset], ubLse, vecRows);
                    //AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);

                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
                    AscendC::DataCopy(
                        gWorkspaceO[wsOOff + static_cast<uint64_t>(vecRowOffset) * embed_],
                        ubOBuf, vecRows * embed_);
                    //AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
                }
                AscendC::CrossCoreSetFlag<Arch::CROSS_CORE_SYNC_MODE_4, PIPE_V>(VecCrossCoreFlagId(MM2_TO_WS_FLAG_V0));
#endif
            }
        }
    }

    // Phase 2: Vector-only LSE logsumexp combine across topK partials (§7.2).
    __aicore__ inline
    void Phase2CombineScale(
        uint32_t coreIdx, uint32_t coreNum,
        AscendC::GlobalTensor<ElementO> &gO,
        AscendC::GlobalTensor<ElementWorkspaceO> &gWorkspaceO,
        AscendC::GlobalTensor<float> &gWorkspaceF)
    {
#ifdef __DAV_VEC__
        uint32_t accBytes = groupSize_ * embed_ * sizeof(float);
        uint32_t tmpBytes = embed_ * sizeof(float) + 64 * sizeof(float);
        AscendC::LocalTensor<float> ubAccBuf =
            resource.ubBuf.template GetBufferByByte<float>(0);
        AscendC::LocalTensor<float> ubTmpBuf =
            resource.ubBuf.template GetBufferByByte<float>(accBytes);
        AscendC::LocalTensor<float> ubMetaBuf =
            resource.ubBuf.template GetBufferByByte<float>(accBytes + tmpBytes);

        EpilogueRescaleO epilogueRescaleO;
        epilogueRescaleO.Init(ubAccBuf, ubTmpBuf, ubMetaBuf, embed_, groupSize_);

        for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNumP2_; taskIdx += coreNum) {
            uint32_t qToken = taskIdx / kvHeads_;
            uint32_t kvHeadIdx = taskIdx % kvHeads_;
            uint32_t qHeadStart = kvHeadIdx * groupSize_;

            uint64_t wsBase = (static_cast<uint64_t>(qToken) * kvHeads_ * topK_
                + kvHeadIdx * topK_) * partialSize_;

            epilogueRescaleO(
                gWorkspaceO, gWorkspaceF, gO,
                qToken, kvHeadIdx, qHeadStart,
                groupSize_, embed_, topK_, kvHeads_, qHeads_,
                partialSize_, oBytesAligned_, wsBase);
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
    uint64_t partialSize_;
    uint64_t oBytesAligned_;

    AscendC::GlobalTensor<int32_t> gActualQseqlen_;
    AscendC::GlobalTensor<int32_t> gActualKvseqlen_;
};

}  // namespace SasaPrefillKernelArch35

#endif  // SPARSE_ATTENTION_SCORE_PREFILL_KERNEL_ARCH35_H
