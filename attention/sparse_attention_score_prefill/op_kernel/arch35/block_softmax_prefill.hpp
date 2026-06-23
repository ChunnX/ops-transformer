/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Block-level softmax epilogue for prefill Phase1 (Arch35 / Atlas A5).
 *
 * Design follows attn_infra BlockEpilogue pattern:
 *   - Constructor receives Resource + scale, allocates UB from resource
 *   - operator() for composable integration with kernel main loop
 *   - Per-row processing: Scale -> CausalMask -> ReduceMax -> Sub -> Exp -> ReduceSum -> Cast
 *   - Outputs rowMax/rowSum to UB (accessible by caller for workspace write)
 *
 * Input:  S[groupSize, sColsAligned] in UB (float)
 * Output: P[groupSize, sColsAligned] in UB (InDtype), rowMax/rowSum in UB (float)
 */
#ifndef BLOCK_SOFTMAX_PREFILL_HPP
#define BLOCK_SOFTMAX_PREFILL_HPP

#include "kernel_utils.hpp"

namespace SparseAttnPrefill {

using namespace NpuArch;

template <class InDtype>
class BlockSoftmaxPrefill {
public:
    using ArchTag = Arch::AtlasA5;
    using ElementS = float;
    using ElementP = InDtype;

    static constexpr float NEG_INF = -3e38f;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t UB_SOFTMAX_TMP_OFFSET = 7 * 32768;
    static constexpr uint32_t UB_ROWMAX_OFFSET = UB_SOFTMAX_TMP_OFFSET + 256;
    static constexpr uint32_t UB_ROWSUM_OFFSET = UB_ROWMAX_OFFSET + 512;

    __aicore__ inline
    BlockSoftmaxPrefill() : scale_(0.0f) {}

    __aicore__ inline
    BlockSoftmaxPrefill(Arch::Resource<ArchTag> &resource, float scaleValue)
        : scale_(scaleValue)
    {
        tmpBuf_ = resource.ubBuf.template GetBufferByByte<float>(UB_SOFTMAX_TMP_OFFSET);
        rowMaxBuf_ = resource.ubBuf.template GetBufferByByte<float>(UB_ROWMAX_OFFSET);
        rowSumBuf_ = resource.ubBuf.template GetBufferByByte<float>(UB_ROWSUM_OFFSET);
    }

    /**
     * @brief Initialize with explicit tmp buffer (for callers that manage UB manually).
     */
    __aicore__ inline
    void Init(AscendC::LocalTensor<float> &ubTmpBuf)
    {
        tmpBuf_ = ubTmpBuf;
        rowMaxBuf_ = ubTmpBuf;
        rowSumBuf_ = ubTmpBuf;
    }

    __aicore__ inline
    void SetScale(float s) { scale_ = s; }

    /**
     * @brief operator() — composable softmax invocation.
     *
     * @param ubSBuf         S matrix [groupSize, sStride] (float) in UB
     * @param ubPBuf         P output [groupSize, sStride] (InDtype) in UB
     * @param groupSize      Number of rows (M)
     * @param validSize      Actual valid KV token count in this block
     * @param causalValidLen Per-token causal valid columns (<= validSize)
     * @param sColsAligned   Column stride = RoundUp(validSize, 16)
     * @param rowMaxOut      Output: per-row max values [groupSize]
     * @param rowSumOut      Output: per-row sum values [groupSize]
     */
    __aicore__ inline
    void operator()(AscendC::LocalTensor<float> &ubSBuf,
                    AscendC::LocalTensor<InDtype> &ubPBuf,
                    uint32_t groupSize,
                    uint32_t validSize,
                    uint32_t causalValidLen,
                    uint32_t sColsAligned,
                    float *rowMaxOut,
                    float *rowSumOut)
    {
        uint32_t sStride = sColsAligned;
        int32_t fullLen = static_cast<int32_t>(sColsAligned);

        // Step 1: Scale all rows
        for (uint32_t row = 0; row < groupSize; row++) {
            uint32_t off = row * sStride;
            AscendC::Muls(ubSBuf[off], ubSBuf[off], scale_, fullLen);
        }
        AscendC::PipeBarrier<PIPE_V>();

        // Step 2: Causal mask (set S[row, causalValidLen..end] = -inf)
        uint32_t maskLen = sColsAligned - causalValidLen;
        if (maskLen > 0) {
            for (uint32_t row = 0; row < groupSize; row++) {
                uint32_t maskOff = row * sStride + causalValidLen;
                AscendC::Duplicate(ubSBuf[maskOff], NEG_INF,
                                   static_cast<int32_t>(maskLen));
            }
            AscendC::PipeBarrier<PIPE_V>();
        }

        // Step 3: Per-row (ReduceMax -> Sub -> Exp -> ReduceSum -> Cast)
        for (uint32_t row = 0; row < groupSize; row++) {
            uint32_t off = row * sStride;

            AscendC::ReduceMax(tmpBuf_, ubSBuf[off], tmpBuf_[32], fullLen);
            AscendC::PipeBarrier<PIPE_V>();
            float rowMax = tmpBuf_.GetValue(0);

            AscendC::Adds(ubSBuf[off], ubSBuf[off], -rowMax, fullLen);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Exp(ubSBuf[off], ubSBuf[off], fullLen);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::ReduceSum(tmpBuf_, ubSBuf[off], tmpBuf_[32], fullLen);
            AscendC::PipeBarrier<PIPE_V>();
            float rowSum = tmpBuf_.GetValue(0);

            AscendC::Cast(ubPBuf[row * sStride], ubSBuf[off],
                AscendC::RoundMode::CAST_NONE, validSize);
            AscendC::PipeBarrier<PIPE_V>();

            rowMaxOut[row] = rowMax;
            rowSumOut[row] = rowSum;
        }
    }

    /**
     * @brief Legacy Compute interface (delegates to operator()).
     */
    __aicore__ inline
    void Compute(AscendC::LocalTensor<float> &ubSBuf,
                 AscendC::LocalTensor<InDtype> &ubPBuf,
                 uint32_t groupSize,
                 uint32_t validSize,
                 uint32_t causalValidLen,
                 uint32_t sColsAligned,
                 float scale,
                 float *rowMaxOut,
                 float *rowSumOut)
    {
        scale_ = scale;
        (*this)(ubSBuf, ubPBuf, groupSize, validSize, causalValidLen,
                sColsAligned, rowMaxOut, rowSumOut);
    }

private:
    float scale_;
    AscendC::LocalTensor<float> tmpBuf_;
    AscendC::LocalTensor<float> rowMaxBuf_;
    AscendC::LocalTensor<float> rowSumBuf_;
};

}  // namespace SparseAttnPrefill

#endif  // BLOCK_SOFTMAX_PREFILL_HPP
