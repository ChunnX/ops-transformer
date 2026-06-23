/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#ifndef SPARSE_ATTENTION_SCORE_PREFILL_KERNEL_ARCH35_H
#define SPARSE_ATTENTION_SCORE_PREFILL_KERNEL_ARCH35_H

#include "kernel_utils.hpp"
#include "../kernel_common.hpp"
#include "block_mmad_prefill_qk.hpp"
#include "block_mmad_prefill_pv.hpp"
#include "block_softmax_prefill.hpp"
#include "block_rescale_o_prefill.hpp"

namespace SparseAttnPrefill {

struct SasaPrefillKernelParams {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR selectIdx;
    GM_ADDR blockTable;
    GM_ADDR kvToQSelectIdx;
    GM_ADDR kvToQSelectNum;
    GM_ADDR kvToQSlotIdx;
    GM_ADDR kvBlockValidSize;
    GM_ADDR actualQseqlen;
    GM_ADDR actualKvseqlen;
    GM_ADDR o;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

// Prefill dispatch policies
struct MmadAtlasA5PrefillQK : public NpuArch::Gemm::MmadAtlasA5 {
    static constexpr uint32_t L0_STAGES = 2;
};

struct MmadAtlasA5PrefillPV : public NpuArch::Gemm::MmadAtlasA5 {
    static constexpr uint32_t L0_STAGES = 2;
};

// ============================================================================
// Main kernel class. Template params allow block-level composition.
// ============================================================================
template <
    class BlockMmadQK_,
    class EpilogueSoftmax_,
    class BlockMmadPV_,
    class EpilogueRescaleO_>
class SasaPrefillKernelArch35 {
public:
    using ArchTag = NpuArch::Arch::AtlasA5;
    using BlockMmadQK = BlockMmadQK_;
    using EpilogueSoftmax = EpilogueSoftmax_;
    using BlockMmadPV = BlockMmadPV_;
    using EpilogueRescaleO = EpilogueRescaleO_;

    using InDtype = typename BlockMmadQK::ElementQ;
    using REDtype = typename BlockMmadPV::ElementO;

    __aicore__ inline void operator()(const SasaPrefillKernelParams& params);

private:
    __aicore__ inline void Init(const SasaPrefillKernelParams& params);
    __aicore__ inline void ComputeBatchOffsets();
    __aicore__ inline uint32_t FindBatchForQToken(uint32_t qToken);
    __aicore__ inline uint32_t FindBatchForKvBlock(uint32_t kvBlockIdx);
    __aicore__ inline void Phase1KvCentricCompute();
    __aicore__ inline void Phase2CombineScale();

private:
    NpuArch::Arch::Resource<ArchTag> resource_;

    SparseAttentionScorePrefillTilingData tilingData_;
    AscendC::GlobalTensor<InDtype> gmQ_;
    AscendC::GlobalTensor<InDtype> gmK_;
    AscendC::GlobalTensor<InDtype> gmV_;
    AscendC::GlobalTensor<int32_t> gmSelectIdx_;
    AscendC::GlobalTensor<int32_t> gmKvToQSelectIdx_;
    AscendC::GlobalTensor<int32_t> gmKvToQSelectNum_;
    AscendC::GlobalTensor<int32_t> gmKvToQSlotIdx_;
    AscendC::GlobalTensor<int32_t> gmKvBlockValidSize_;
    AscendC::GlobalTensor<int32_t> gmBlockTable_;
    AscendC::GlobalTensor<int64_t> gmQSeqlen_;
    AscendC::GlobalTensor<int64_t> gmKvSeqlen_;
    AscendC::GlobalTensor<InDtype> gmO_;
    AscendC::GlobalTensor<InDtype> gmWorkspaceO_;
    AscendC::GlobalTensor<float> gmWorkspaceF_;

    static constexpr uint32_t MAX_BATCH = 64;
    uint32_t batchQOffset_[MAX_BATCH + 1];
    uint32_t batchKvBlockOffset_[MAX_BATCH + 1];
    int64_t qSeqlenArr_[MAX_BATCH];
    int64_t kvSeqlenArr_[MAX_BATCH];

    uint32_t coreIdx_;
    uint32_t coreNum_;
    uint32_t D_;
    uint32_t kvHeads_;
    uint32_t groupSize_;
    uint32_t blockSz_;
    uint32_t maxQPerBlock_;
    uint32_t topK_;
    uint64_t partialSize_;
    float scale_;
    uint64_t oBytesAligned_;
};

// ============================================================================
// operator()
// ============================================================================
template <class BQK, class ES, class BPV, class ERO>
__aicore__ inline void SasaPrefillKernelArch35<BQK, ES, BPV, ERO>::operator()(
    const SasaPrefillKernelParams& params)
{
    Init(params);
    Phase1KvCentricCompute();
    AscendC::SyncAll<false>();
    Phase2CombineScale();
}

// ============================================================================
// Init
// ============================================================================
template <class BQK, class ES, class BPV, class ERO>
__aicore__ inline void SasaPrefillKernelArch35<BQK, ES, BPV, ERO>::Init(
    const SasaPrefillKernelParams& params)
{
    auto *tp = reinterpret_cast<__gm__ const SparseAttentionScorePrefillTilingData*>(params.tiling);
    tilingData_.batch = tp->batch;
    tilingData_.numHeads = tp->numHeads;
    tilingData_.kvHeads = tp->kvHeads;
    tilingData_.groupSize = tp->groupSize;
    tilingData_.embeddingSize = tp->embeddingSize;
    tilingData_.blockSize = tp->blockSize;
    tilingData_.topK = tp->topK;
    tilingData_.totalQTokens = tp->totalQTokens;
    tilingData_.numKvBlocks = tp->numKvBlocks;
    tilingData_.maxBlocksPerBatch = tp->maxBlocksPerBatch;
    tilingData_.maxQPerBlock = tp->maxQPerBlock;
    tilingData_.totalTaskNumP1 = tp->totalTaskNumP1;
    tilingData_.totalTaskNumP2 = tp->totalTaskNumP2;
    tilingData_.scaleValue = tp->scaleValue;
    tilingData_.innerPrecise = tp->innerPrecise;
    tilingData_.partialSize = tp->partialSize;
    tilingData_.workSpaceSize = tp->workSpaceSize;
    tilingData_.tilingKey = tp->tilingKey;

    coreIdx_ = AscendC::GetBlockIdx();
    coreNum_ = AscendC::GetBlockNum();
    D_ = tilingData_.embeddingSize;
    kvHeads_ = tilingData_.kvHeads;
    groupSize_ = tilingData_.groupSize;
    blockSz_ = tilingData_.blockSize;
    maxQPerBlock_ = tilingData_.maxQPerBlock;
    topK_ = tilingData_.topK;
    partialSize_ = tilingData_.partialSize;
    scale_ = tilingData_.scaleValue;
    uint64_t oBytes = static_cast<uint64_t>(groupSize_) * D_ * sizeof(InDtype);
    oBytesAligned_ = ((oBytes + sizeof(float) - 1) / sizeof(float)) * sizeof(float);

    gmQ_.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype*>(params.q));
    gmK_.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype*>(params.k));
    gmV_.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype*>(params.v));
    gmSelectIdx_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.selectIdx));
    gmKvToQSelectIdx_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.kvToQSelectIdx));
    gmKvToQSelectNum_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.kvToQSelectNum));
    gmKvToQSlotIdx_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.kvToQSlotIdx));
    gmKvBlockValidSize_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.kvBlockValidSize));
    gmBlockTable_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.blockTable));
    gmQSeqlen_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(params.actualQseqlen));
    gmKvSeqlen_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(params.actualKvseqlen));
    gmO_.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype*>(params.o));
    gmWorkspaceO_.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype*>(params.workspace));
    gmWorkspaceF_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(params.workspace));
    ComputeBatchOffsets();
}

// ============================================================================
// ComputeBatchOffsets
// ============================================================================
template <class BQK, class ES, class BPV, class ERO>
__aicore__ inline void SasaPrefillKernelArch35<BQK, ES, BPV, ERO>::ComputeBatchOffsets()
{
    batchQOffset_[0] = 0;
    batchKvBlockOffset_[0] = 0;
    for (uint32_t b = 0; b < tilingData_.batch; b++) {
        qSeqlenArr_[b] = gmQSeqlen_.GetValue(b);
        kvSeqlenArr_[b] = gmKvSeqlen_.GetValue(b);
        batchQOffset_[b + 1] = batchQOffset_[b] + static_cast<uint32_t>(qSeqlenArr_[b]);
        uint32_t nBlk = (static_cast<uint32_t>(kvSeqlenArr_[b]) + blockSz_ - 1) / blockSz_;
        batchKvBlockOffset_[b + 1] = batchKvBlockOffset_[b] + nBlk;
    }
}

// ============================================================================
// FindBatch helpers
// ============================================================================
template <class BQK, class ES, class BPV, class ERO>
__aicore__ inline uint32_t SasaPrefillKernelArch35<BQK, ES, BPV, ERO>::FindBatchForQToken(uint32_t qToken)
{
    for (uint32_t b = 0; b < tilingData_.batch; b++) {
        if (qToken < batchQOffset_[b + 1]) return b;
    }
    return tilingData_.batch - 1;
}

template <class BQK, class ES, class BPV, class ERO>
__aicore__ inline uint32_t SasaPrefillKernelArch35<BQK, ES, BPV, ERO>::FindBatchForKvBlock(uint32_t kvBlockIdx)
{
    for (uint32_t b = 0; b < tilingData_.batch; b++) {
        if (kvBlockIdx < batchKvBlockOffset_[b + 1]) return b;
    }
    return tilingData_.batch - 1;
}

// ============================================================================
// Phase 1: KV-centric partial compute (Cube core).
// ============================================================================
template <class BQK, class ES, class BPV, class ERO>
__aicore__ inline void SasaPrefillKernelArch35<BQK, ES, BPV, ERO>::Phase1KvCentricCompute()
{
#ifdef __DAV_CUBE__
    using namespace NpuArch;

    BlockMmadQK blockMmadQK;
    BlockMmadPV blockMmadPV;

    blockMmadQK.Init(resource_, blockSz_, D_);
    uint32_t qkL1Used = blockSz_ * D_ * sizeof(InDtype)
        + BlockMmadQK::L0_TILE_M * D_ * sizeof(InDtype);
    blockMmadPV.Init(resource_, qkL1Used, blockSz_, D_, groupSize_);

    const uint32_t numHeads = tilingData_.numHeads;
    uint32_t mAligned = RoundUp(groupSize_, 16u);
    uint32_t sColMax = RoundUp(blockSz_, 16u);
    uint32_t ubSSize = mAligned * sColMax;
    uint32_t oColAligned = RoundUp(D_, 16u);
    uint32_t ubOSize = mAligned * oColAligned;

    AscendC::LocalTensor<float> ubSBuf =
        resource_.ubBuf.template GetBufferByByte<float>(0);
    AscendC::LocalTensor<REDtype> ubOBuf =
        resource_.ubBuf.template GetBufferByByte<REDtype>(
            ubSSize * sizeof(float));
    AscendC::LocalTensor<InDtype> ubPBuf =
        resource_.ubBuf.template GetBufferByByte<InDtype>(
            ubSSize * sizeof(float) + ubOSize * sizeof(REDtype));
    AscendC::LocalTensor<float> ubTmpBuf =
        resource_.ubBuf.template GetBufferByByte<float>(
            ubSSize * sizeof(float) + ubOSize * sizeof(REDtype)
            + mAligned * sColMax * sizeof(InDtype));

    EpilogueSoftmax epilogueSoftmax;
    epilogueSoftmax.Init(ubTmpBuf);
    epilogueSoftmax.SetScale(scale_);

    for (uint32_t taskIdx = coreIdx_; taskIdx < tilingData_.totalTaskNumP1;
         taskIdx += coreNum_) {
        uint32_t kvBlockIdx = taskIdx / kvHeads_;
        uint32_t kvHeadIdx = taskIdx % kvHeads_;

        uint32_t batchIdx = FindBatchForKvBlock(kvBlockIdx);
        uint32_t localBlockIdx = kvBlockIdx - batchKvBlockOffset_[batchIdx];
        int32_t physicalBlockId = gmBlockTable_.GetValue(
            batchIdx * tilingData_.maxBlocksPerBatch + localBlockIdx);
        uint32_t validSize = static_cast<uint32_t>(
            gmKvBlockValidSize_.GetValue(kvBlockIdx));

        uint64_t kvBlockBase =
            static_cast<uint64_t>(physicalBlockId) * blockSz_ * kvHeads_ * D_
            + kvHeadIdx * D_;
        int64_t strideKVRow = static_cast<int64_t>(kvHeads_) * D_;

        auto gmKLayout = tla::MakeLayout<InDtype, layout::ColumnMajor>(
            strideKVRow, blockSz_);
        auto gmKTensor = tla::MakeTensor(
            gmK_[kvBlockBase], gmKLayout, Arch::PositionGM{});
        blockMmadQK.LoadKResident(gmKTensor, validSize, D_);

        auto gmVLayout = tla::MakeLayout<InDtype, layout::RowMajor>(
            blockSz_, strideKVRow);
        auto gmVTensor = tla::MakeTensor(
            gmV_[kvBlockBase], gmVLayout, Arch::PositionGM{});
        blockMmadPV.LoadVResident(gmVTensor, validSize, D_);

        uint32_t numQTokens = static_cast<uint32_t>(
            gmKvToQSelectNum_.GetValue(kvBlockIdx));

        for (uint32_t qi = 0; qi < numQTokens; qi++) {
            uint32_t qToken = static_cast<uint32_t>(
                gmKvToQSelectIdx_.GetValue(kvBlockIdx * maxQPerBlock_ + qi));
            uint32_t slotK = static_cast<uint32_t>(
                gmKvToQSlotIdx_.GetValue(kvBlockIdx * maxQPerBlock_ + qi));

            uint32_t qBatch = FindBatchForQToken(qToken);
            uint32_t localQIdx = qToken - batchQOffset_[qBatch];
            uint32_t kvSeqlen = static_cast<uint32_t>(kvSeqlenArr_[qBatch]);
            uint32_t qSeqlen = static_cast<uint32_t>(qSeqlenArr_[qBatch]);
            uint32_t qPosition = kvSeqlen - qSeqlen + localQIdx;
            uint32_t kvStartPos = localBlockIdx * blockSz_;
            uint32_t causalValidLen = 0;
            if (qPosition >= kvStartPos) {
                uint32_t maxLen = qPosition - kvStartPos + 1;
                causalValidLen = (maxLen < validSize) ? maxLen : validSize;
            }
            if (causalValidLen == 0) continue;

            uint64_t wsBytes = (static_cast<uint64_t>(qToken) * kvHeads_ * topK_
                + kvHeadIdx * topK_ + slotK) * partialSize_;
            uint64_t wsOOff = wsBytes / sizeof(InDtype);
            uint64_t wsFOff = (wsBytes + oBytesAligned_) / sizeof(float);

            uint32_t qHeadStart = kvHeadIdx * groupSize_;
            uint64_t qBase = static_cast<uint64_t>(qToken) * numHeads * D_
                + qHeadStart * D_;

            // --- QK matmul ---
            auto gmQLayout = tla::MakeLayout<InDtype, layout::RowMajor>(
                groupSize_, D_);
            auto gmQTensor = tla::MakeTensor(
                gmQ_[qBase], gmQLayout, Arch::PositionGM{});

            uint32_t sColsAligned = RoundUp(validSize, 16u);
            auto ubSLayout = tla::MakeLayout<float, layout::RowMajor>(
                mAligned, sColsAligned);
            auto ubSTensor = tla::MakeTensor(
                ubSBuf, ubSLayout, Arch::PositionUB{});

            blockMmadQK.ComputeQK(gmQTensor, ubSTensor, groupSize_, validSize, D_);

            // --- Epilogue Softmax ---
            float rowMaxArr[128];
            float rowSumArr[128];
            epilogueSoftmax(ubSBuf, ubPBuf,
                groupSize_, validSize, causalValidLen, sColsAligned,
                rowMaxArr, rowSumArr);

            // Write rowMax, rowSum to workspace
            for (uint32_t gh = 0; gh < groupSize_; gh++) {
                gmWorkspaceF_.SetValue(wsFOff + gh, rowMaxArr[gh]);
                gmWorkspaceF_.SetValue(wsFOff + groupSize_ + gh, rowSumArr[gh]);
            }

            // --- PV matmul ---
            auto ubPLayout = tla::MakeLayout<InDtype, layout::RowMajor>(
                groupSize_, validSize);
            auto ubPTensor = tla::MakeTensor(
                ubPBuf, ubPLayout, Arch::PositionUB{});

            auto ubOLayout = tla::MakeLayout<REDtype, layout::RowMajor>(
                mAligned, oColAligned);
            auto ubOTensor = tla::MakeTensor(
                ubOBuf, ubOLayout, Arch::PositionUB{});

            blockMmadPV.ComputePV(ubPTensor, ubOTensor, groupSize_, validSize, D_);

            // Write O[groupSize, D] to workspace
            for (uint32_t gh = 0; gh < groupSize_; gh++) {
                for (uint32_t d = 0; d < D_; d++) {
                    gmWorkspaceO_.SetValue(
                        wsOOff + static_cast<uint64_t>(gh) * D_ + d,
                        static_cast<InDtype>(
                            ubOBuf.GetValue(gh * oColAligned + d)));
                }
            }
        }
    }
#endif
}

// ============================================================================
// Phase 2: Vector-only CombineScale.
// ============================================================================
template <class BQK, class ES, class BPV, class ERO>
__aicore__ inline void SasaPrefillKernelArch35<BQK, ES, BPV, ERO>::Phase2CombineScale()
{
#ifdef __DAV_VECTOR__
    const uint32_t numHeads = tilingData_.numHeads;

    AscendC::LocalTensor<float> ubAccBuf =
        resource_.ubBuf.template GetBufferByByte<float>(0);
    AscendC::LocalTensor<float> ubTmpBuf =
        resource_.ubBuf.template GetBufferByByte<float>(
            groupSize_ * D_ * sizeof(float));

    EpilogueRescaleO epilogueRescaleO;
    epilogueRescaleO.Init(ubAccBuf, ubTmpBuf, D_);

    for (uint32_t taskIdx = coreIdx_; taskIdx < tilingData_.totalTaskNumP2; taskIdx += coreNum_) {
        uint32_t qToken = taskIdx / kvHeads_;
        uint32_t kvHeadIdx = taskIdx % kvHeads_;
        uint32_t qHeadStart = kvHeadIdx * groupSize_;

        uint64_t wsBase = (static_cast<uint64_t>(qToken) * kvHeads_ * topK_
                         + kvHeadIdx * topK_) * partialSize_;

        epilogueRescaleO(
            gmWorkspaceO_, gmWorkspaceF_, gmSelectIdx_, gmO_,
            qToken, kvHeadIdx, qHeadStart,
            groupSize_, D_, topK_, kvHeads_, numHeads,
            partialSize_, oBytesAligned_, wsBase);
    }
#endif
}

}  // namespace SparseAttnPrefill

#endif  // SPARSE_ATTENTION_SCORE_PREFILL_KERNEL_ARCH35_H