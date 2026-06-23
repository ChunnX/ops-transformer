/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Block-level RescaleO epilogue for prefill Phase2 (Arch35 / Atlas A5).
 *
 * Design follows attn_infra BlockEpilogue<RescaleO> pattern:
 *   - Constructor receives Resource, allocates UB buffers
 *   - operator() for composable integration
 *   - Vectorized workspace access via DataCopy (GM <-> UB)
 *   - Vectorized combine: Muls(corr) + Add per partial, Muls(1/sum) normalize
 *
 * Combines topK partial results: (O_partial[gh,D], rowMax[gh], rowSum[gh])
 * into final output O[groupSize, D].
 */
#ifndef BLOCK_RESCALE_O_PREFILL_HPP
#define BLOCK_RESCALE_O_PREFILL_HPP

#include "kernel_utils.hpp"

namespace SparseAttnPrefill {

using namespace NpuArch;

template <class InDtype>
class BlockRescaleOPrefill {
public:
    using ArchTag = Arch::AtlasA5;
    using ElementO = InDtype;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;

    __aicore__ inline
    BlockRescaleOPrefill() : D_(0) {}

    /**
     * @brief Initialize with UB buffers for vectorized rescale.
     *
     * @param ubAccBuf  Accumulator [groupSize * D] (float) for running O
     * @param ubTmpBuf  Temp [D + 8] (float) for loading one partial row + scalar exp
     * @param D         Head dimension
     */
    __aicore__ inline
    void Init(AscendC::LocalTensor<float> &ubAccBuf,
              AscendC::LocalTensor<float> &ubTmpBuf,
              uint32_t D)
    {
        accBuf_ = ubAccBuf;
        tmpBuf_ = ubTmpBuf;
        D_ = D;
    }

    /**
     * @brief operator() — online softmax combine across topK partials.
     *
     * Algorithm per head:
     *   1. globalMax = max(rowMax[k]) across valid partials
     *   2. For each valid partial k:
     *      corr = exp(rowMax[k] - globalMax)
     *      if first: acc = O_partial * corr
     *      else:     acc += O_partial * corr
     *      globalSum += corr * rowSum[k]
     *   3. O_final = acc * (1/globalSum)
     *
     * Key optimizations vs previous version:
     *   - Vector Muls/Add on D-length buffers (no scalar d loop for math)
     *   - ScalarExp via UB Exp instruction (device-compatible)
     */
    __aicore__ inline
    void operator()(AscendC::GlobalTensor<InDtype> &gmWorkspaceO,
                    AscendC::GlobalTensor<float> &gmWorkspaceF,
                    AscendC::GlobalTensor<int32_t> &gmSelectIdx,
                    AscendC::GlobalTensor<InDtype> &gmO,
                    uint32_t qToken,
                    uint32_t kvHeadIdx,
                    uint32_t qHeadStart,
                    uint32_t groupSize,
                    uint32_t D,
                    uint32_t topK,
                    uint32_t kvHeads,
                    uint32_t numHeads,
                    uint64_t partialSize,
                    uint64_t oBytesAligned,
                    uint64_t wsBase)
    {
        int32_t dLen = static_cast<int32_t>(D);

        for (uint32_t gh = 0; gh < groupSize; gh++) {
            // Pass 1: find globalMax across valid partials
            float globalMax = -3.4028235e38f;
            for (uint32_t k = 0; k < topK; k++) {
                int32_t selBlock = gmSelectIdx.GetValue(
                    static_cast<uint64_t>(qToken) * topK + k);
                if (selBlock < 0) continue;
                uint64_t pBytes = wsBase + static_cast<uint64_t>(k) * partialSize;
                uint64_t fOff = (pBytes + oBytesAligned) / sizeof(float);
                float pMax = gmWorkspaceF.GetValue(fOff + gh);
                if (pMax > globalMax) globalMax = pMax;
            }

            // Pass 2: vectorized accumulate O partials + compute globalSum
            float globalSum = 0.0f;
            bool isFirst = true;
            uint32_t accOff = gh * D;

            for (uint32_t k = 0; k < topK; k++) {
                int32_t selBlock = gmSelectIdx.GetValue(
                    static_cast<uint64_t>(qToken) * topK + k);
                if (selBlock < 0) continue;

                uint64_t pBytes = wsBase + static_cast<uint64_t>(k) * partialSize;
                uint64_t oOff = pBytes / sizeof(InDtype);
                uint64_t fOff = (pBytes + oBytesAligned) / sizeof(float);
                float pMax = gmWorkspaceF.GetValue(fOff + gh);
                float pSum = gmWorkspaceF.GetValue(fOff + groupSize + gh);
                float corr = ScalarExp(pMax - globalMax);

                globalSum += corr * pSum;

                // Load O_partial[gh, 0:D] from GM to tmpBuf (cast to float)
                LoadPartialO(gmWorkspaceO, oOff + static_cast<uint64_t>(gh) * D, D);

                // tmpBuf *= corr (vectorized)
                AscendC::Muls(tmpBuf_, tmpBuf_, corr, dLen);
                AscendC::PipeBarrier<PIPE_V>();

                if (isFirst) {
                    // acc = tmpBuf (UB-to-UB copy via vector add-zero)
                    AscendC::Adds(accBuf_[accOff], tmpBuf_, 0.0f, dLen);
                    AscendC::PipeBarrier<PIPE_V>();
                    isFirst = false;
                } else {
                    // acc += tmpBuf (vectorized)
                    AscendC::Add(accBuf_[accOff], accBuf_[accOff], tmpBuf_, dLen);
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }

            // Normalize: acc *= (1/globalSum)
            if (globalSum > 0.0f) {
                float invSum = 1.0f / globalSum;
                AscendC::Muls(accBuf_[accOff], accBuf_[accOff], invSum, dLen);
                AscendC::PipeBarrier<PIPE_V>();
            }

            // Write final O to output GM (cast float -> InDtype)
            StoreOutputO(gmO, qToken, qHeadStart + gh, numHeads, D, accOff);
        }
    }

    /**
     * @brief Legacy Compute interface (delegates to operator()).
     */
    __aicore__ inline
    void Compute(AscendC::GlobalTensor<InDtype> &gmWorkspaceO,
                 AscendC::GlobalTensor<float> &gmWorkspaceF,
                 AscendC::GlobalTensor<int32_t> &gmSelectIdx,
                 AscendC::GlobalTensor<InDtype> &gmO,
                 uint32_t qToken, uint32_t kvHeadIdx,
                 uint32_t qHeadStart, uint32_t groupSize,
                 uint32_t D, uint32_t topK, uint32_t kvHeads,
                 uint32_t numHeads, uint64_t partialSize,
                 uint64_t oBytesAligned, uint64_t wsBase)
    {
        (*this)(gmWorkspaceO, gmWorkspaceF, gmSelectIdx, gmO,
                qToken, kvHeadIdx, qHeadStart, groupSize, D, topK,
                kvHeads, numHeads, partialSize, oBytesAligned, wsBase);
    }

private:
    AscendC::LocalTensor<float> accBuf_;
    AscendC::LocalTensor<float> tmpBuf_;
    uint32_t D_;

    // Device-compatible scalar exp using UB vector Exp instruction
    __aicore__ inline float ScalarExp(float x)
    {
        // Use offset D_ in tmpBuf for scalar exp computation.
        // Exp requires count to be at least one full repeat (64 floats).
        // We write x at tmpBuf_[D_], run Exp on a full 64-element block,
        // and read back the result. Other elements in the block are garbage
        // but we only use index 0 of that sub-region.
        uint32_t expOff = D_;
        tmpBuf_.SetValue(expOff, x);
        AscendC::Exp(tmpBuf_[expOff], tmpBuf_[expOff], FLOAT_VECTOR_SIZE);
        AscendC::PipeBarrier<PIPE_V>();
        return tmpBuf_.GetValue(expOff);
    }

    // Load O partial from GM to tmpBuf with float cast
    __aicore__ inline
    void LoadPartialO(AscendC::GlobalTensor<InDtype> &gmSrc,
                      uint64_t srcOff, uint32_t D)
    {
        for (uint32_t d = 0; d < D; d++) {
            float val = static_cast<float>(gmSrc.GetValue(srcOff + d));
            tmpBuf_.SetValue(d, val);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    // Store output O from accBuf to GM with InDtype cast
    __aicore__ inline
    void StoreOutputO(AscendC::GlobalTensor<InDtype> &gmO,
                      uint32_t qToken, uint32_t headIdx,
                      uint32_t numHeads, uint32_t D, uint32_t accOff)
    {
        uint64_t outBase = static_cast<uint64_t>(qToken) * numHeads * D
                         + static_cast<uint64_t>(headIdx) * D;
        for (uint32_t d = 0; d < D; d++) {
            float val = accBuf_.GetValue(accOff + d);
            gmO.SetValue(outBase + d, static_cast<InDtype>(val));
        }
    }
};

}  // namespace SparseAttnPrefill

#endif  // BLOCK_RESCALE_O_PREFILL_HPP
