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
    static constexpr float NEG_INF_LSE = -3.4028235e38f;

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

        lseMaxBuf_ = ubBase;
        lseSumBuf_ = ubBase[BUF_32K_FLOATS];
        dstBuf_ = ubBase[BUF_32K_FLOATS * 2U];
        accumOutBuf_ = ubBase[BUF_32K_FLOATS * 3U];
        broadCastTmpBuf_ = ubBase[BUF_32K_FLOATS * 4U];
        // Dedicated bf16 cast-output buffer (separate from dstBuf_ so CopyFinalResOut's Cast is
        // out-of-place, not the corrupting in-place float->bf16 alias). Size need is tiny
        // (<=gSplitSize*headDimAlignFp32 bf16); carve a 32K-float slot to match the layout stride.
        castOutBuf_ = ubBase[BUF_32K_FLOATS * 5U];
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
    AscendC::LocalTensor<T> accumOutBuf_;
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
                   uint32_t startRow,
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
            tmpSinkUb, lseMaxLocal, lseSumLocal, accumOutBuf_,
            dealRowCount, actualCombineLoopSize_, softmaxLseFlag, learnableSinkFlag);
        //AscendC::DumpTensor(lseSumLocal, 789, 64);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline
    void CopyAccumOutIn(uint32_t qToken,
                        uint32_t kvHeadIdx,
                        uint32_t splitKVIndex,
                        uint32_t startRow,
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
            + static_cast<uint64_t>(splitKVIndex) * groupSize_ * D_
            + static_cast<uint64_t>(startRow) * D_;
        AscendC::DataCopyPad(accumOutLocal, gmAccumOut[oOffset], copyInParams, copyInPadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        // The per-row rowSum used by the regbase Div in ReduceFinalRes_VF is NOT staged here:
        // it is reused from the compact rowSum already DataCopyPad'd into broadCastTmp_ by
        // CopyLseIn (lseSumTmpBuf = broadCastTmpBuf_[alignedLseCount], layout [topK, dealRowCount]).
    }

    __aicore__ inline
    void ReduceFinalRes(uint32_t qToken,
                        uint32_t kvHeadIdx,
                        AscendC::LocalTensor<T> &dst,
                        AscendC::LocalTensor<T> &scaleLocal,
                        uint32_t startRow,
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
        for (uint32_t j = 0; j < actualCombineLoopSize_; ++j) {
            uint32_t maxOff = j * dealRowCount * FP32_ONE_BLOCK_SIZE;
            float firstRowMax = lseMaxBuf_[maxOff].GetValue(0);
            // AscendC::printf("qToken:%d, kvHeadIdx:%d, firstRowMax:%f, j:%d\n",
            //                 qToken, kvHeadIdx, firstRowMax, j);
            if (firstRowMax <= NEG_INF_LSE) {
                break;
            }
            CopyAccumOutIn(
                qToken, kvHeadIdx, j, startRow, dealRowCount,
                gmAccumOut, accumOutBuf_);
            // split j's compact per-row rowSum (one scalar per row): slice into the
            // lseSumTmpBuf region staged by CopyLseIn.
            uint64_t rowSumOff = static_cast<uint64_t>(alignedLseCount)
                + static_cast<uint64_t>(j) * dealRowCount;
            AscendC::LocalTensor<T> rowSumLocal = broadCastTmpBuf_[rowSumOff];
            FaVectorApiPrefill::ReduceFinalRes_VF<T>(
                dst, scaleLocal, accumOutBuf_, rowSumLocal, dealRowCount,
                static_cast<uint64_t>(headDimAlignFp32_), j);
            AscendC::PipeBarrier<PIPE_ALL>();  // TODO: remove this
            //AscendC::DumpTensor(accumOutBuf_, 123, 8);
            //AscendC::DumpTensor(dst, 456, 8);
        }
    }

    __aicore__ inline
    void CopyFinalResOut(uint64_t attenOutOffset,
                         AscendC::GlobalTensor<ElementO> &gmO,
                         AscendC::LocalTensor<T> &accumOutLocal,
                         uint32_t startRow,
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
        AscendC::DataCopyPad(
            gmO[attenOutOffset + static_cast<uint64_t>(startRow) * D_],
            castBuf, dataCopyParams);
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
        uint32_t gSplitSizeLse = BUFFER_SIZE_BYTE_32K / (BYTE_BLOCK * topK_);
        uint32_t gSplitSizeAccumOut = BUFFER_SIZE_BYTE_32K / sizeof(T) / headDimAlignFp32_;
        uint32_t gSplitSize = (gSplitSizeLse < gSplitSizeAccumOut) ? gSplitSizeLse : gSplitSizeAccumOut;
        gSplitSize = (gSplitSize > groupSize_) ? groupSize_ : gSplitSize;
        if (gSplitSize > 16U) {
            gSplitSize = 16U;
        }

        uint32_t loopCount = (groupSize_ + gSplitSize - 1U) / gSplitSize;
        uint32_t tailSplitSize = groupSize_ - (loopCount - 1U) * gSplitSize;

        uint32_t shapeArray[] = {gSplitSize, FP32_ONE_BLOCK_SIZE};
        dstBuf_.SetShapeInfo(AscendC::ShapeInfo(2, shapeArray, AscendC::DataFormat::ND));

        for (uint32_t i = 0; i < loopCount - 1U; ++i) {
            uint32_t startRow = i * gSplitSize;
            CopyLseIn(qToken, kvHeadIdx, startRow, gSplitSize,
                      gmSoftmaxMax, gmSoftmaxSum, lseMaxBuf_, lseSumBuf_);
            ComputeScaleValue(lseMaxBuf_, lseSumBuf_, gSplitSize);
            AscendC::LocalTensor<T> dst = dstBuf_;
            ReduceFinalRes(
                qToken, kvHeadIdx, dst, lseSumBuf_, startRow, gSplitSize,
                gmAccumOut);
            CopyFinalResOut(attenOutOffset, gmO, dst, startRow, gSplitSize);
        }

        if (tailSplitSize > 0U) {
            uint32_t startRow = (loopCount - 1U) * gSplitSize;
            CopyLseIn(qToken, kvHeadIdx, startRow, tailSplitSize,
                      gmSoftmaxMax, gmSoftmaxSum, lseMaxBuf_, lseSumBuf_);
            ComputeScaleValue(lseMaxBuf_, lseSumBuf_, tailSplitSize);
            AscendC::LocalTensor<T> dst = dstBuf_;
            ReduceFinalRes(
                qToken, kvHeadIdx, dst, lseSumBuf_, startRow, tailSplitSize,
                gmAccumOut);
            CopyFinalResOut(attenOutOffset, gmO, dst, startRow, tailSplitSize);
        }
    }
};

}  // namespace NpuArch::Epilogue::Block

#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_PREFILL_ARCH35_HPP
