/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Block-level online softmax epilogue for prefill Phase1 (Arch35 / Atlas A5).
 *
 * Batch: CausalMask -> Scale -> RowMax -> Sub -> Exp -> RowSum -> Cast(P) -> LSE
 * Each AIV owns its local UB rows (QK FixPipe already split S across vectors).
 */
#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_PREFILL_ARCH35_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_PREFILL_ARCH35_HPP

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/detail/alignment.hpp"
#include "../../../attn_infra/epilogue/dispatch_policy.hpp"
#include "kernel_operator.h"

namespace NpuArch::Epilogue::Block {

template <class InDtype>
class BlockEpilogue<
    EpilogueOnlineSoftmaxPrefillArch35,
    InDtype>
{
public:
    using DispatchPolicy = EpilogueOnlineSoftmaxPrefillArch35;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementS = float;
    using ElementP = InDtype;

    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t REDUCE_SCRATCH_OFFSET = 64;
    static constexpr uint32_t TV_UB_OFFSET = 128;
    static constexpr float MIN_VALUE_FP32 = -3.389531390315715675e+38f;
    static constexpr float NEG_INF_LSE = -3.4028235e38f;

    __aicore__ inline
    BlockEpilogue() : scale_(0.0f) {}

    __aicore__ inline
    void Init(AscendC::LocalTensor<float> &ubTmpBuf,
              AscendC::LocalTensor<float> &ubLseOut,
              AscendC::LocalTensor<float> &ubRowSumOut)
    {
        tmpBuf_ = ubTmpBuf;
        lseBuf_ = ubLseOut;
        rowSumBuf_ = ubRowSumOut;
    }

    __aicore__ inline
    void SetScale(float s) { scale_ = s; }

    __aicore__ inline
    void operator()(AscendC::LocalTensor<float> &ubSBuf,
                    AscendC::LocalTensor<InDtype> &ubPBuf,
                    uint32_t numRows,
                    uint32_t validSize,
                    uint32_t causalValidLen,
                    uint32_t sColsAligned)
    {
        uint32_t maskLen = sColsAligned - causalValidLen;
            // AscendC::printf(
            //     "[SASAPrefillSM] subBlk=%u numRows=%u validSize=%u causalValidLen=%u "
            //     "sColsAligned=%u maskLen=%u scale=%f\n",
            //     AscendC::GetSubBlockIdx(), numRows, validSize, causalValidLen,
            //     sColsAligned, maskLen, scale_);
        if (numRows == 0U) {
            return;
        }
        uint32_t numRowsRound = RoundUp(numRows, FLOAT_BLOCK_SIZE);

        if (maskLen > 0U) {
            for (uint32_t row = 0; row < numRows; row++) {
                AscendC::Duplicate(
                    ubSBuf[row * sColsAligned + causalValidLen], MIN_VALUE_FP32,
                    static_cast<int32_t>(maskLen));
            }
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::Muls(
            ubSBuf, ubSBuf, scale_,
            static_cast<int32_t>(numRows * sColsAligned));
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<float> rowMaxUb = tmpBuf_;
        AscendC::LocalTensor<float> tvUb = tmpBuf_[TV_UB_OFFSET];
        AscendC::LocalTensor<float> tvScratch = tmpBuf_[TV_UB_OFFSET + REDUCE_SCRATCH_OFFSET];

        CalcRowMaxTail(ubSBuf, rowMaxUb, tvUb, tvScratch,
            numRows, numRowsRound, validSize, sColsAligned);
        //AscendC::DumpTensor(rowMaxUb, 456, 16);
        SubtractRowMaxAndExp(ubSBuf, rowMaxUb, tvUb,
            numRows, numRowsRound, validSize, sColsAligned);
        //AscendC::DumpTensor(ubSBuf, 456, 16);
        CalcRowSumTail(ubSBuf, rowSumBuf_, tvUb, tvScratch,
            numRows, numRowsRound, validSize, sColsAligned);
        AscendC::printf("0000000000000000000000000000\n");
        DownCastP(ubSBuf, ubPBuf, numRows, sColsAligned);
        AscendC::printf("1111111111111111111111111111\n");
        CalcLse(rowMaxUb, rowSumBuf_, lseBuf_, numRows);
        AscendC::printf("2222222222222222222222222222\n");
    }

private:
    __aicore__ inline
    void SetVecMask(int32_t len)
    {
        uint64_t mask = 0;
        uint64_t one = 1;
        uint64_t temp = len % FLOAT_VECTOR_SIZE;
        for (int64_t i = 0; i < static_cast<int64_t>(temp); i++) {
            mask |= one << i;
        }
        if (len >= FLOAT_VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>(mask, static_cast<uint64_t>(-1));
        } else if (len > 0) {
            AscendC::SetVectorMask<int8_t>(0x0, mask);
        } else {
            AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        }
    }

    __aicore__ inline
    void SetBlockReduceMask(int32_t len)
    {
        if (len > 8 || len < 1) {
            AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
            return;
        }
        uint64_t subMask = (static_cast<uint64_t>(1) << len) - 1;
        uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask +
            (subMask << 56) + (subMask << 40) + (subMask << 24) + (subMask << 8);
        AscendC::SetVectorMask<int8_t>(maskValue, maskValue);
    }

    __aicore__ inline
    void CalcRowMaxTail(const AscendC::LocalTensor<float> &srcUb,
                        const AscendC::LocalTensor<float> &rowmaxUb,
                        const AscendC::LocalTensor<float> &tvUb,
                        const AscendC::LocalTensor<float> &tvScratch,
                        uint32_t numRows, uint32_t numRowsRound,
                        uint32_t numElems, uint32_t numElemsAligned)
    {
        if (numElems >= FLOAT_VECTOR_SIZE) {
            AscendC::BlockReduceMax<float, false>(
                tvUb, srcUb, numRows, 0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BlockReduceMax<float, false>(
                rowmaxUb, tvUb,
                CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                0, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint64_t colIdx = 1; colIdx < static_cast<uint64_t>(numElems) / FLOAT_VECTOR_SIZE; ++colIdx) {
                AscendC::BlockReduceMax<float, false>(
                    tvUb, srcUb[colIdx * FLOAT_VECTOR_SIZE],
                    numRows, 0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::BlockReduceMax<float, false>(
                    tvScratch, tvUb,
                    CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(static_cast<int32_t>(numRows));
                AscendC::Max<float, false>(
                    rowmaxUb, rowmaxUb, tvScratch,
                    static_cast<uint64_t>(0), 1,
                    AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
            }
        }
        if (numElems % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(static_cast<int32_t>(numElems % FLOAT_VECTOR_SIZE));
            AscendC::BlockReduceMax<float, false>(
                tvUb, srcUb[numElems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                numRows, 0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            SetBlockReduceMask(CeilDiv(numElems % FLOAT_VECTOR_SIZE, FLOAT_BLOCK_SIZE));
            if (numElems < FLOAT_VECTOR_SIZE) {
                AscendC::BlockReduceMax<float, false>(
                    rowmaxUb, tvUb,
                    CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
            } else {
                AscendC::BlockReduceMax<float, false>(
                    tvScratch, tvUb,
                    CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(static_cast<int32_t>(numRows));
                AscendC::Max<float, false>(
                    rowmaxUb, rowmaxUb, tvScratch,
                    static_cast<uint64_t>(0), 1,
                    AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        }
    }

    __aicore__ inline
    void CalcRowSumTail(const AscendC::LocalTensor<float> &srcUb,
                        const AscendC::LocalTensor<float> &rowsumUb,
                        const AscendC::LocalTensor<float> &tvUb,
                        const AscendC::LocalTensor<float> &tvScratch,
                        uint32_t numRows, uint32_t numRowsRound,
                        uint32_t numElems, uint32_t numElemsAligned)
    {
        if (numElems >= FLOAT_VECTOR_SIZE) {
            AscendC::BlockReduceSum<float, false>(
                tvUb, srcUb, numRows, 0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BlockReduceSum<float, false>(
                rowsumUb, tvUb,
                CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                0, 1, 1, 8);
            
            AscendC::PipeBarrier<PIPE_V>();
            for (uint64_t colIdx = 1; colIdx < static_cast<uint64_t>(numElems) / FLOAT_VECTOR_SIZE; ++colIdx) {
                AscendC::BlockReduceSum<float, false>(
                    tvUb, srcUb[colIdx * FLOAT_VECTOR_SIZE],
                    numRows, 0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::BlockReduceSum<float, false>(
                    tvScratch, tvUb,
                    CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(static_cast<int32_t>(numRows));
                AscendC::Add<float, false>(
                    rowsumUb, rowsumUb, tvScratch,
                    static_cast<uint64_t>(0), 1,
                    AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
            }
        }
        if (numElems % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(static_cast<int32_t>(numElems % FLOAT_VECTOR_SIZE));
            AscendC::BlockReduceSum<float, false>(
                tvUb, srcUb[numElems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                numRows, 0, 1, 1, numElemsAligned / FLOAT_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
            SetBlockReduceMask(CeilDiv(numElems % FLOAT_VECTOR_SIZE, FLOAT_BLOCK_SIZE));
            if (numElems < FLOAT_VECTOR_SIZE) {
                AscendC::BlockReduceSum<float, false>(
                    rowsumUb, tvUb,
                    CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
            } else {
                AscendC::BlockReduceSum<float, false>(
                    tvScratch, tvUb,
                    CeilDiv(numRowsRound * FLOAT_BLOCK_SIZE, FLOAT_VECTOR_SIZE),
                    0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(static_cast<int32_t>(numRows));
                AscendC::Add<float, false>(
                    rowsumUb, rowsumUb, tvScratch,
                    static_cast<uint64_t>(0), 1,
                    AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        }
    }

    __aicore__ inline
    void SubtractRowMaxAndExp(const AscendC::LocalTensor<float> &srcUb,
                              const AscendC::LocalTensor<float> &rowmaxUb,
                              AscendC::LocalTensor<float> &tvUb,
                              uint32_t numRows, uint32_t numRowsRound,
                              uint32_t numElems, uint32_t numElemsAligned)
    {
        AscendC::Brcb(
            tvUb.ReinterpretCast<uint32_t>(),
            rowmaxUb.ReinterpretCast<uint32_t>(),
            numRowsRound / FLOAT_BLOCK_SIZE,
            AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();

        uint32_t colBlocks = numElemsAligned / FLOAT_BLOCK_SIZE;
        for (uint32_t colIdx = 0; colIdx < numElems / FLOAT_VECTOR_SIZE; ++colIdx) {
            AscendC::Sub<float, false>(
                srcUb[colIdx * FLOAT_VECTOR_SIZE],
                srcUb[colIdx * FLOAT_VECTOR_SIZE],
                tvUb,
                static_cast<uint64_t>(0),
                numRows,
                AscendC::BinaryRepeatParams(1, 1, 0, colBlocks, colBlocks, 1));
        }
        if (numElems % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(static_cast<int32_t>(numElems % FLOAT_VECTOR_SIZE));
            AscendC::Sub<float, false>(
                srcUb[numElems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                srcUb[numElems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                tvUb,
                static_cast<uint64_t>(0),
                numRows,
                AscendC::BinaryRepeatParams(1, 1, 0, colBlocks, colBlocks, 1));
            AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        }
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Exp<float, false>(
            srcUb, srcUb,
            static_cast<uint64_t>(0),
            CeilDiv(numRows * numElemsAligned, FLOAT_VECTOR_SIZE),
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline
    void DownCastP(const AscendC::LocalTensor<float> &srcUb,
                   AscendC::LocalTensor<InDtype> &dstUb,
                   uint32_t numRows, uint32_t numElemsAligned)
    {
        uint32_t repeatTimes = CeilDiv(numRows * numElemsAligned, FLOAT_VECTOR_SIZE);
        if constexpr (std::is_same<ElementP, bfloat16_t>::value) {
            AscendC::Cast<ElementP, float, false>(
                dstUb, srcUb,
                AscendC::RoundMode::CAST_RINT,
                static_cast<uint64_t>(0), repeatTimes,
                AscendC::UnaryRepeatParams(1, 1, 4, 8));
        } else {
            AscendC::Cast<ElementP, float, false>(
                dstUb, srcUb,
                AscendC::RoundMode::CAST_NONE,
                static_cast<uint64_t>(0), repeatTimes,
                AscendC::UnaryRepeatParams(1, 1, 4, 8));
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline
    void CalcLse(const AscendC::LocalTensor<float> &rowmaxUb,
                 const AscendC::LocalTensor<float> &rowsumUb,
                 AscendC::LocalTensor<float> &lseUb,
                 uint32_t numRows)
    {
        SetVecMask(static_cast<int32_t>(numRows));
        AscendC::Ln<float, false>(
            lseUb, rowsumUb,
            static_cast<uint64_t>(0), 1,
            AscendC::UnaryRepeatParams(1, 1, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float, false>(
            lseUb, lseUb, rowmaxUb,
            static_cast<uint64_t>(0), 1,
            AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));

        for (uint32_t row = 0; row < numRows; row++) {
            if (rowsumUb.GetValue(row) <= 0.0f) {
                lseUb.SetValue(row, NEG_INF_LSE);
            }
        }
    }

    float scale_;
    AscendC::LocalTensor<float> tmpBuf_;
    AscendC::LocalTensor<float> lseBuf_;
    AscendC::LocalTensor<float> rowSumBuf_;
};

}  // namespace NpuArch::Epilogue::Block

#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_PREFILL_ARCH35_HPP
