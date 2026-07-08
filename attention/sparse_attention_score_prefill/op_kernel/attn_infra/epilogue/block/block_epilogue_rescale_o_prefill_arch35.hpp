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
 * CopyAccumOutIn reads rowSum from GM (lseSumBuf holds scale after ComputeScaleValue).
 */
#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_PREFILL_ARCH35_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_PREFILL_ARCH35_HPP

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/resource.hpp"
#include "../../../attn_infra/epilogue/dispatch_policy.hpp"
#include "kernel_operator.h"

using namespace AscendC;

#include "../../../../../common/op_kernel/arch35/vf/vf_flash_decode.h"

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
        //AscendC::Duplicate(lseMaxLocal, NEG_INF_LSE, ubRowStride * topK_);
        //AscendC::Duplicate(lseSumLocal, 0.0f, ubRowStride * topK_);
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
        AscendC::DumpTensor(lseMaxLocal, 123, 64);
        AscendC::DumpTensor(lseSumLocal, 456, 64);
        FaVectorApi::ComputeScaleValue_VF(
            tmpSinkUb, lseMaxLocal, lseSumLocal, accumOutBuf_,
            dealRowCount, actualCombineLoopSize_, softmaxLseFlag, learnableSinkFlag);
        //AscendC::DumpTensor(lseSumLocal, 123, 64);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline
    void CopyAccumOutIn(uint32_t qToken,
                        uint32_t kvHeadIdx,
                        uint32_t splitKVIndex,
                        uint32_t startRow,
                        uint32_t dealRowCount,
                        AscendC::GlobalTensor<T> &gmAccumOut,
                        AscendC::GlobalTensor<T> &gmSoftmaxSum,
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

        // lseSumBuf_ already holds scale after ComputeScaleValue; rowSum from GM.
        // Slot validity is checked once per split in ReduceFinalRes (first stat value),
        // so only valid slots reach here and rowSum > 0 is guaranteed.
        uint64_t statBase = TaskStatBase(qToken, kvHeadIdx);
        for (uint32_t r = 0U; r < dealRowCount; ++r) {
            uint32_t gh = startRow + r;
            uint64_t statIdx = statBase + static_cast<uint64_t>(splitKVIndex) * groupSize_ + gh;
            float rowSum = gmSoftmaxSum[statIdx].GetValue(0);
            uint32_t rowOff = r * headDimAlignFp32_;
            AscendC::Divs(
                    accumOutLocal[rowOff], accumOutLocal[rowOff], rowSum,
                    static_cast<int32_t>(headDimAlignFp32_));
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline
    void ReduceFinalRes(uint32_t qToken,
                        uint32_t kvHeadIdx,
                        AscendC::LocalTensor<T> &dst,
                        AscendC::LocalTensor<T> &scaleLocal,
                        uint32_t startRow,
                        uint32_t dealRowCount,
                        AscendC::GlobalTensor<T> &gmAccumOut,
                        AscendC::GlobalTensor<T> &gmSoftmaxSum)
    {
        // Slot validity check from UB (no GM round-trip). After CopyLseIn +
        // ComputeScaleValue, lseMaxBuf_ holds the per-slot rowMax in broadcast
        // layout [topK, dealRowCount, FP32_ONE_BLOCK_SIZE] (topK outer, each
        // scalar replicated 8x). ComputeScaleValue_8_VF reads lseMax but does not
        // overwrite it, so the values are intact. InitWorkspaceStats prefills
        // -FLT_MAX (NEG_INF_LSE) for slots Phase1 never wrote; a valid slot's
        // rowMax is a finite score-derived value, so firstRowMax <= NEG_INF_LSE
        // discriminates the whole slot. Skip CopyAccumOutIn + ReduceFinalRes_VF
        // for invalid splits; scale[j]==0 from ComputeScaleValue so they
        // contribute 0 anyway, and skipping avoids Divs-by-zero and stale-UB
        // accumulation.
        for (uint32_t j = 0; j < actualCombineLoopSize_; ++j) {
            uint32_t maxOff = j * dealRowCount * FP32_ONE_BLOCK_SIZE;
            float firstRowMax = lseMaxBuf_[maxOff].GetValue(0);
            if (firstRowMax <= NEG_INF_LSE) {
                continue;
            }
            CopyAccumOutIn(
                qToken, kvHeadIdx, j, startRow, dealRowCount,
                gmAccumOut, gmSoftmaxSum, accumOutBuf_);
            FaVectorApi::ReduceFinalRes_VF<T>(
                dst, scaleLocal, accumOutBuf_, dealRowCount,
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
        AscendC::LocalTensor<ElementO> castBuf = dstBuf_.template ReinterpretCast<ElementO>();
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
                gmAccumOut, gmSoftmaxSum);
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
                gmAccumOut, gmSoftmaxSum);
            CopyFinalResOut(attenOutOffset, gmO, dst, startRow, tailSplitSize);
        }
    }
};

}  // namespace NpuArch::Epilogue::Block

#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_PREFILL_ARCH35_HPP
