/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Phase2 combine epilogue for prefill (Arch35).
 *
 * Workspace layout matches IFA FlashDecode (separate buffers):
 *   accumOut  [totalQ, kvHead, topK, groupSize, D]
 *   softmaxMax / softmaxSum [totalQ, kvHead, topK, groupSize] (compact float)
 *
 * GM max/sum are compact; CopyLseIn expands to 32B-aligned UB for ComputeScaleValue_VF.
 * Unlike the IFA library, this operator's Phase1 leaves accumOut unnormalized (no /sum),
 * so Phase2 must divide each partial by its original per-row rowSum. That divide is
 * embedded as a regbase MicroAPI::Div (broadcast rowSum via DIST_BRC_B32) inside the
 * local copy of ReduceFinalRes_VF (arch35/vector_api/vf_flash_decode_msa.h). The per-row
 * rowSum is not re-read from GM: it reuses the compact rowSum already DataCopyPad'd into
 * broadCastTmp_ by CopyLseIn (lseSumTmpBuf = broadCastTmpBuf_[alignedLseCount],
 * layout [topK, dealRowCount]), sliced per split j. CopyAccumOutIn only copies accumOut.
 *
 * Pipeline: the per-split j loop is double-buffered so the MTE2 load of split j+1
 * overlaps the V reduce of split j (mirrors IFA fdMm2ResBuf1/2). accumOut is managed
 * like the kernel's ubSTensor[ubSBufId]: a 2-deep array accumOutBuf[bufId],
 * bufId = j % ACCUM_OUT_STAGES, with two strided stages carved from the free
 * [192,256)KiB region, and event ids ACCUM_OUT_FLAG_BASE + bufId for both MTE2_V and
 * V_MTE2. CopyLseIn(MTE2_V EVENT_ID0) and CopyFinalResOut(V_MTE3 EVENT_ID0) stay
 * per-chunk blocking (MTE3 overlap out of scope: production single chunk).
 */
#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_PREFILL_ARCH35_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_PREFILL_ARCH35_HPP

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/resource.hpp"
#include "../../../attn_infra/epilogue/dispatch_policy.hpp"
#include "kernel_operator.h"

using namespace AscendC;

#include "../../../../../common/op_kernel/arch35/vf/vf_flash_decode.h"
// Local copy of the library ReduceFinalRes_VF family with the per-row regbase Div
// embedded (this operator's Phase1 leaves accumOut unnormalized). Must follow the
// vf_flash_decode.h include above (relies on FLT_ZERO / MicroAPI).
#include "../../../arch35/vector_api/vf_flash_decode_msa.h"

namespace NpuArch::Epilogue::Block {

template <class InDtype>
class BlockEpilogue<
    EpilogueRescaleOPrefillArch35,
    InDtype>
{
public:
    using DispatchPolicy = EpilogueRescaleOPrefillArch35;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementO = InDtype;
    using T = float;

    static constexpr uint32_t FP32_ONE_BLOCK_SIZE = 8U;
    static constexpr uint32_t BYTE_BLOCK = 32U;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768U;
    static constexpr uint32_t BUF_32K_FLOATS = BUFFER_SIZE_BYTE_32K / sizeof(T);
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192U;
    static constexpr uint32_t BUF_8K_FLOATS = BUFFER_SIZE_BYTE_8K / sizeof(T);
    static constexpr float NEG_INF_LSE = -3.4028235e38f;

    // accumOut ping-pong (mirrors the kernel ubSTensor[ubSBufId] style): a 2-deep array of
    // UB tensors indexed by bufId = j % ACCUM_OUT_STAGES, each a strided stage, with event
    // ids derived as ACCUM_OUT_FLAG_BASE + bufId for both MTE2_V and V_MTE2 (distinct
    // HardEvent types share an id safely, per the IFA fdMm2ResBuf1/2 pattern).
    static constexpr uint32_t ACCUM_OUT_STAGES = 2U;
    static constexpr uint32_t ACCUM_OUT_FLAG_BASE = 3U;     // MTE2_V/V_MTE2 id = base + bufId

    __aicore__ inline BlockEpilogue() {}

    __aicore__ inline
    void InitFDBuffers(AscendC::LocalTensor<T> &ubBase,
                       uint32_t D,
                       uint32_t groupSize,
                       uint32_t topK,
                       uint32_t kvHeads)
    {
        D_ = D;
        groupSize_ = groupSize;
        topK_ = topK;
        kvHeads_ = kvHeads;
        headDimAlignFp32_ = RoundUp(D_, FP32_ONE_BLOCK_SIZE);

        // UB layout: contiguous, no holes. lse buffers capped at 8K (the [topK, dealRowCount, 8]
        // broadcast layout fits in 8KiB for topK, groupSize <= 16); the rest are 32K slots.
        lseMaxBuf_       = ubBase;                       // 8K
        lseSumBuf_       = ubBase[BUF_8K_FLOATS];        // 8K
        uint32_t off = BUF_8K_FLOATS * 2U;
        dstBuf_          = ubBase[off];
        off += BUF_32K_FLOATS;   // 32K
        broadCastTmpBuf_ = ubBase[off];
        off += BUF_32K_FLOATS;  // 32K (Broadcast scratch + lse staging)
        // Separate bf16 cast-output buffer so CopyFinalResOut's Cast is out-of-place.
        castOutBuf_      = ubBase[off];
        off += BUF_32K_FLOATS;  // 32K
        // accumOut ping-pong (kernel ubSTensor style): 2 stages, uniform 32K-slot stride.
        for (uint32_t i = 0U; i < ACCUM_OUT_STAGES; ++i) {
            accumOutBuf[i] = ubBase[off + BUF_32K_FLOATS * i];  // 32K each
        }
    }

    __aicore__ inline
    void FlashDecodeCompute(uint32_t tmpBlockIdx,
                            uint32_t totalTaskNumP2,
                            AscendC::GlobalTensor<ElementO> &gmO,
                            AscendC::GlobalTensor<T> &gmAccumOut,
                            AscendC::GlobalTensor<T> &gmSoftmaxMax,
                            AscendC::GlobalTensor<T> &gmSoftmaxSum,
                            uint32_t numHeads,
                            uint32_t kvHeads)
    {
        if (tmpBlockIdx >= totalTaskNumP2) {
            return;
        }
        (void)numHeads;

        uint32_t qToken = tmpBlockIdx / kvHeads;
        uint32_t kvHeadIdx = tmpBlockIdx % kvHeads;
        // IFA: bIdx * kvHeadNum * gSize * headDim + n2Idx * gSize * headDim
        uint64_t attenOutOffset = static_cast<uint64_t>(qToken) * kvHeads * groupSize_ * D_
            + static_cast<uint64_t>(kvHeadIdx) * groupSize_ * D_;

        actualCombineLoopSize_ = topK_;
        CombineSplitKVRes(attenOutOffset, qToken, kvHeadIdx, gmO, gmAccumOut, gmSoftmaxMax, gmSoftmaxSum);
    }

private:
    AscendC::LocalTensor<T> lseMaxBuf_;
    AscendC::LocalTensor<T> lseSumBuf_;
    AscendC::LocalTensor<T> dstBuf_;
    AscendC::LocalTensor<T> accumOutBuf[ACCUM_OUT_STAGES];
    AscendC::LocalTensor<T> broadCastTmpBuf_;
    AscendC::LocalTensor<T> castOutBuf_;

    uint32_t D_ = 0;
    uint32_t groupSize_ = 0;
    uint32_t topK_ = 0;
    uint32_t kvHeads_ = 0;
    uint32_t headDimAlignFp32_ = 0;
    uint32_t actualCombineLoopSize_ = 0;

    __aicore__ inline
    static uint32_t RoundUp(uint32_t a, uint32_t b)
    {
        return (a + b - 1U) / b * b;
    }

    __aicore__ inline
    uint64_t TaskStatBase(uint32_t qToken, uint32_t kvHeadIdx) const
    {
        return (static_cast<uint64_t>(qToken) * kvHeads_ + kvHeadIdx) * topK_ * groupSize_;
    }

    __aicore__ inline
    uint64_t TaskAccumBase(uint32_t qToken, uint32_t kvHeadIdx) const
    {
        return (static_cast<uint64_t>(qToken) * kvHeads_ + kvHeadIdx) * topK_ * groupSize_ * D_;
    }

    // GM max/sum compact [topK, groupSize] -> UB IFA layout [topK, dealRow, 8].
    __aicore__ inline
    void CopyLseIn(uint32_t qToken,
                   uint32_t kvHeadIdx,
                   uint32_t dealRowCount,
                   AscendC::GlobalTensor<T> &gmSoftmaxMax,
                   AscendC::GlobalTensor<T> &gmSoftmaxSum,
                   AscendC::LocalTensor<T> &lseMaxLocal,
                   AscendC::LocalTensor<T> &lseSumLocal)
    {
        uint32_t ubRowStride = dealRowCount * FP32_ONE_BLOCK_SIZE;
        auto lseCount = dealRowCount * topK_, alignedLseCount = RoundUp(lseCount, FP32_ONE_BLOCK_SIZE);

        AscendC::LocalTensor<T> lseMaxTmpBuf = broadCastTmpBuf_;
        AscendC::LocalTensor<T> lseSumTmpBuf = broadCastTmpBuf_[alignedLseCount];
        AscendC::LocalTensor<uint8_t> broadCastTmpBuf = broadCastTmpBuf_[alignedLseCount * 2U].template ReinterpretCast<uint8_t>();
        uint64_t taskBase = TaskStatBase(qToken, kvHeadIdx);
        DataCopyParams copyParams {
            1,                                      // blockCount
            uint16_t(lseCount * sizeof(float)),     // blockLen
            0,                                      // srcStride
            0,                                      // dstStride
        };
        DataCopyPadParams copyPadParams {
            false,                 // isPad
            0,                     // leftPadding
            0,                     // rightPadding
            0                      // paddingValue
        };
        AscendC::DataCopyPad(lseMaxTmpBuf, gmSoftmaxMax[taskBase], copyParams, copyPadParams);
        AscendC::DataCopyPad(lseSumTmpBuf, gmSoftmaxSum[taskBase], copyParams, copyPadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        const uint32_t srcShape[2] = {lseCount, 1};
        const uint32_t dstShape[2] = {lseCount, FP32_ONE_BLOCK_SIZE};
        AscendC::Broadcast<T, 2, 1>(lseMaxLocal, lseMaxTmpBuf, dstShape, srcShape, broadCastTmpBuf);
        AscendC::Broadcast<T, 2, 1>(lseSumLocal, lseSumTmpBuf, dstShape, srcShape, broadCastTmpBuf);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline
    void ComputeScaleValue(AscendC::LocalTensor<T> &lseMaxLocal,
                           AscendC::LocalTensor<T> &lseSumLocal,
                           uint32_t dealRowCount)
    {
        AscendC::LocalTensor<bfloat16_t> tmpSinkUb;
        bool softmaxLseFlag = false;
        bool learnableSinkFlag = false;
        //AscendC::DumpTensor(lseMaxLocal, 123, 64);
        //AscendC::DumpTensor(lseSumLocal, 456, 64);
        FaVectorApi::ComputeScaleValue_VF(
            tmpSinkUb, lseMaxLocal, lseSumLocal, accumOutBuf[0],
            dealRowCount, actualCombineLoopSize_, softmaxLseFlag, learnableSinkFlag);
        //AscendC::DumpTensor(lseSumLocal, 789, 64);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline
    void CopyAccumOutIn(uint32_t qToken,
                        uint32_t kvHeadIdx,
                        uint32_t splitKVIndex,
                        uint32_t dealRowCount,
                        AscendC::GlobalTensor<T> &gmAccumOut,
                        AscendC::LocalTensor<T> &accumOutLocal)
    {
        AscendC::DataCopyExtParams copyInParams;
        AscendC::DataCopyPadExtParams<T> copyInPadParams;
        copyInParams.blockCount = dealRowCount;
        copyInParams.blockLen = static_cast<uint32_t>(D_) * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (headDimAlignFp32_ - D_) / FP32_ONE_BLOCK_SIZE;

        copyInPadParams.isPad = true;
        copyInPadParams.leftPadding = 0;
        copyInPadParams.rightPadding = (headDimAlignFp32_ - D_) % FP32_ONE_BLOCK_SIZE;
        copyInPadParams.paddingValue = 0;

        uint64_t oOffset = TaskAccumBase(qToken, kvHeadIdx)
            + static_cast<uint64_t>(splitKVIndex) * groupSize_ * D_;
        AscendC::DataCopyPad(accumOutLocal, gmAccumOut[oOffset], copyInParams, copyInPadParams);
    }

    __aicore__ inline
    void ReduceFinalRes(uint32_t qToken,
                        uint32_t kvHeadIdx,
                        AscendC::LocalTensor<T> &dst,
                        AscendC::LocalTensor<T> &scaleLocal,
                        uint32_t dealRowCount,
                        AscendC::GlobalTensor<T> &gmAccumOut)
    {
        // Slot validity check from UB (no GM round-trip). After CopyLseIn +
        // ComputeScaleValue, lseMaxBuf_ holds the per-slot rowMax in broadcast
        // layout [topK, dealRowCount, FP32_ONE_BLOCK_SIZE] (topK outer, each
        // scalar replicated 8x). ComputeScaleValue_8_VF reads lseMax but does not
        // overwrite it, so the values are intact. InitWorkspaceStats prefills
        // -FLT_MAX (NEG_INF_LSE) for slots Phase1 never wrote; a valid slot's
        // rowMax is a finite score-derived value, so firstRowMax <= NEG_INF_LSE
        // discriminates the whole slot. Valid slots are left-contiguous in the
        // workspace (Phase1 writes slotK = 0..valid_k-1 per qToken/kvHead, since
        // build_k2q_csr assigns slot_flat = arange(topk) only to valid entries),
        // so the first invalid slot means every later slot is invalid too: break
        // once instead of continue. Skipping CopyAccumOutIn + ReduceFinalRes_VF
        // for invalid splits avoids Divs-by-zero (rowSum==0) and stale-UB
        // accumulation; scale[j]==0 from ComputeScaleValue makes them 0 anyway.
        // The per-row rowSum for the regbase Div is reused from the compact rowSum
        // already staged by CopyLseIn into broadCastTmp_ (lseSumTmpBuf =
        // broadCastTmpBuf_[alignedLseCount], layout [topK, dealRowCount]); no GM
        // re-read or SetValue here.
        uint32_t lseCount = dealRowCount * topK_;
        uint32_t alignedLseCount = RoundUp(lseCount, FP32_ONE_BLOCK_SIZE);

        // Double-buffered (ping-pong) accumOut, managed like the kernel's ubSTensor[ubSBufId]:
        // a 2-deep array accumOutBuf[curId], curId = j % ACCUM_OUT_STAGES, with event ids
        // ACCUM_OUT_FLAG_BASE + curId for both MTE2_V (forward) and V_MTE2 (reverse) — distinct
        // HardEvent types share an id safely (IFA fdMm2ResBuf1/2 does the same).
        //
        // Uniform loop (no prologue / no prefetch branch): each iteration loads its own split j
        // (MTE2) then reduces it (V). The MTE2 load of j overlaps the V reduce of j-1 (still in
        // flight): Wait<V_MTE2>(curId) drains curId's prior use (reduce j-2, or the prime) so the
        // MTE2 load may issue while V is still finishing reduce(j-1); Wait<MTE2_V>(curId) then
        // gates the V reduce on the just-loaded data. CopyAccumOutIn is in the loop.
        //
        // Validity is checked inline (same NEG_INF_LSE test). The break issues before this iter's
        // load, so it leaves no dangling MTE2_V; the prime + trailing drain absorb the dangling
        // V_MTE2, so flags stay balanced for any break point. (The scalar GetValue is a non-blocking
        // scalar-pipe read; if profiling ever shows it serializing the V reduce, hoist the scan into
        // a precomputed validK bound and loop `j < validK`.)

        // Prime each stage's V_MTE2: a stage's first reuse has no prior V reduce to drain.
        for (uint32_t s = 0U; s < ACCUM_OUT_STAGES; ++s) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ACCUM_OUT_FLAG_BASE + s);
        }

        for (uint32_t j = 0U; j < actualCombineLoopSize_; ++j) {
            uint32_t maxOff = j * dealRowCount * FP32_ONE_BLOCK_SIZE;
            if (lseMaxBuf_[maxOff].GetValue(0) <= NEG_INF_LSE) {
                break;
            }
            uint32_t curId = j % ACCUM_OUT_STAGES;
            // stage drained (reduce j-2, or primed) -> MTE2 may overwrite it.
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ACCUM_OUT_FLAG_BASE + curId);
            CopyAccumOutIn(qToken, kvHeadIdx, j, dealRowCount, gmAccumOut, accumOutBuf[curId]);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ACCUM_OUT_FLAG_BASE + curId);    // load j || V reduce j-1
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ACCUM_OUT_FLAG_BASE + curId);  // load j done
            // split j's compact per-row rowSum (one scalar per row): slice into the
            // lseSumTmpBuf region staged by CopyLseIn.
            uint64_t rowSumOff = static_cast<uint64_t>(alignedLseCount)
                + static_cast<uint64_t>(j) * dealRowCount;
            AscendC::LocalTensor<T> rowSumLocal = broadCastTmpBuf_[rowSumOff];
            FaVectorApiPrefill::ReduceFinalRes_VF<T>(
                dst, scaleLocal, accumOutBuf[curId], rowSumLocal, dealRowCount,
                static_cast<uint64_t>(headDimAlignFp32_), j);
            // V done reading accumOutBuf[curId] (and dst store committed; SetFlag<V_MTE2> is a V
            // completion barrier, ordering dst V-store->V-load across iterations) -> stage reusable.
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ACCUM_OUT_FLAG_BASE + curId);
        }
        // Trailing drain: consume the dangling V_MTE2 per stage (last reduce's set + the primed
        // twin that no later load reached) so it doesn't stale-signal the next chunk/task. MTE2_V
        // is set+waited 1:1 within each iter, so no MTE2_V trailing drain.
        for (uint32_t s = 0U; s < ACCUM_OUT_STAGES; ++s) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ACCUM_OUT_FLAG_BASE + s);
        }
    }

    __aicore__ inline
    void CopyFinalResOut(uint64_t attenOutOffset,
                         AscendC::GlobalTensor<ElementO> &gmO,
                         AscendC::LocalTensor<T> &accumOutLocal,
                         uint32_t dealRowCount)
    {
        // castBuf is a SEPARATE buffer from accumOutLocal (dstBuf_): the old code aliased them
        // (castBuf = dstBuf_.ReinterpretCast) making the float->bf16 Cast in-place, which corrupts.
        // accumOutLocal (= dstBuf_, the float combined result from ReduceFinalRes_VF) is READ by Cast;
        // castOutBuf_ is the Cast OUTPUT (bf16). No aliasing -> Cast is a normal out-of-place cast.
        AscendC::LocalTensor<ElementO> castBuf = castOutBuf_.template ReinterpretCast<ElementO>();
        uint32_t shapeArray[] = {dealRowCount, D_};
        castBuf.SetShapeInfo(AscendC::ShapeInfo(2, shapeArray, AscendC::DataFormat::ND));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(
            castBuf, accumOutLocal,
            AscendC::RoundMode::CAST_ROUND,
            dealRowCount * headDimAlignFp32_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        AscendC::DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = dealRowCount;
        dataCopyParams.blockLen = static_cast<uint32_t>(D_) * sizeof(ElementO);
        dataCopyParams.srcStride = (headDimAlignFp32_ - D_) / (BYTE_BLOCK / sizeof(ElementO));
        dataCopyParams.dstStride = 0;
        AscendC::DataCopyPad(gmO[attenOutOffset], castBuf, dataCopyParams);
    }

    __aicore__ inline
    void CombineSplitKVRes(uint64_t attenOutOffset,
                           uint32_t qToken,
                           uint32_t kvHeadIdx,
                           AscendC::GlobalTensor<ElementO> &gmO,
                           AscendC::GlobalTensor<T> &gmAccumOut,
                           AscendC::GlobalTensor<T> &gmSoftmaxMax,
                           AscendC::GlobalTensor<T> &gmSoftmaxSum)
    {
        // groupSize <= 16 fits every UB buffer (lse 8K cap, dst/accumOut/castOut 32K slots)
        // for topK <= 16, D <= 512, so a single chunk covers the whole group.
        CopyLseIn(qToken, kvHeadIdx, groupSize_,
                  gmSoftmaxMax, gmSoftmaxSum, lseMaxBuf_, lseSumBuf_);
        ComputeScaleValue(lseMaxBuf_, lseSumBuf_, groupSize_);
        AscendC::LocalTensor<T> dst = dstBuf_;
        ReduceFinalRes(qToken, kvHeadIdx, dst, lseSumBuf_, groupSize_, gmAccumOut);
        CopyFinalResOut(attenOutOffset, gmO, dst, groupSize_);
    }
};

}  // namespace NpuArch::Epilogue::Block

#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_PREFILL_ARCH35_HPP
