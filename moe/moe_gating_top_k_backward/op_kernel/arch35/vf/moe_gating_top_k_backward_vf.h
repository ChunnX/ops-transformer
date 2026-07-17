/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_gating_top_k_backward_vf.h
 * \brief VF functions for MoeGatingTopKBackward on Arch35
 */

#ifndef MOE_GATING_TOP_K_BACKWARD_VF_H
#define MOE_GATING_TOP_K_BACKWARD_VF_H

#include "kernel_operator.h"

namespace MoeGatingTopKBackwardNs {
using namespace AscendC;

constexpr MicroAPI::CastTrait castTraitB322B16 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

constexpr MicroAPI::CastTrait castTraitB162B32 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

__simd_vf__ void SigmoidGradFP32VF(__ubuf__ float *xNormAddr, __ubuf__ float *gradNormXAddr, __ubuf__ float *gradXAddr,
                                   uint32_t totalElements, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    MicroAPI::RegTensor<float> xReg, gNormReg, tmpReg, dstReg;
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<float>();

    for (uint16_t i = 0; i < repeatTimes; i++) {
        uint32_t remaining = totalElements - i * oneRepeatSize;
        uint32_t chunkElems = (remaining > oneRepeatSize) ? oneRepeatSize : remaining;
        preg = MicroAPI::UpdateMask<float>(chunkElems);
        MicroAPI::LoadAlign(xReg, xNormAddr + i * oneRepeatSize);
        MicroAPI::LoadAlign(gNormReg, gradNormXAddr + i * oneRepeatSize);
        MicroAPI::Muls(tmpReg, xReg, static_cast<float>(-1), preg);
        MicroAPI::Adds(tmpReg, tmpReg, static_cast<float>(1), preg);
        MicroAPI::Mul(tmpReg, tmpReg, xReg, preg);
        MicroAPI::Mul(dstReg, tmpReg, gNormReg, preg);
        MicroAPI::StoreAlign(gradXAddr + i * oneRepeatSize, dstReg, preg);
    }
}

template <typename T>
__simd_vf__ void SigmoidGradHalfVF(__ubuf__ float *xNormAddr, __ubuf__ float *gradNormXAddr, __ubuf__ T *gradXAddr,
                                   uint32_t totalElements, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    MicroAPI::RegTensor<float> xReg, gNormReg, tmpReg, dstReg;
    MicroAPI::RegTensor<T> outReg;
    MicroAPI::MaskReg pregFp = MicroAPI::CreateMask<float>();
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    for (uint16_t i = 0; i < repeatTimes; i++) {
        uint32_t remaining = totalElements - i * oneRepeatSize;
        uint32_t chunkElems = (remaining > oneRepeatSize) ? oneRepeatSize : remaining;
        pregFp = MicroAPI::UpdateMask<float>(chunkElems);
        MicroAPI::LoadAlign(xReg, xNormAddr + i * oneRepeatSize);
        MicroAPI::LoadAlign(gNormReg, gradNormXAddr + i * oneRepeatSize);
        MicroAPI::Muls(tmpReg, xReg, static_cast<float>(-1), pregFp);
        MicroAPI::Adds(tmpReg, tmpReg, static_cast<float>(1), pregFp);
        MicroAPI::Mul(tmpReg, tmpReg, xReg, pregFp);
        MicroAPI::Mul(dstReg, tmpReg, gNormReg, pregFp);
        MicroAPI::Cast<T, float, castTraitB322B16>(outReg, dstReg, maskFull);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(gradXAddr + i * oneRepeatSize, outReg, maskFull);
    }
}

// ============================================================================
// CastGradYFlatVF: contiguous case (kAlign16_ == kAlign_) — single flat T->float
// cast over totalElements. Used by CopyInGradY when gradY rows need no re-striding.
// ============================================================================
template <typename T>
__simd_vf__ void CastGradYFlatVF(__ubuf__ T *srcAddr, __ubuf__ float *dstAddr, uint32_t totalElements,
                                 uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<float>();
    for (uint16_t i = 0; i < repeatTimes; i++) {
        uint32_t remaining = totalElements - i * oneRepeatSize;
        uint32_t chunkElems = (remaining > oneRepeatSize) ? oneRepeatSize : remaining;
        preg = MicroAPI::UpdateMask<float>(chunkElems);
        MicroAPI::RegTensor<T> srcB16;
        MicroAPI::RegTensor<float> dstF32;
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcB16, srcAddr + i * oneRepeatSize);
        MicroAPI::Cast<float, T, castTraitB162B32>(dstF32, srcB16, preg);
        MicroAPI::StoreAlign(dstAddr + i * oneRepeatSize, dstF32, preg);
    }
}

// ============================================================================
// CastGradYRowsVF: strided per-row case (kAlign16_ != kAlign_) — row width = k,
// src row stride = kAlign16_, dst row stride = kAlign_. Used when gradY's bf16/fp16
// row alignment differs from the fp32 destination row alignment.
//
// CHUNKING (k > chunkSize support): LoadAlign<T, DIST_UNPACK_B16> always reads
// a fixed chunkSize = VECTOR_REG_WIDTH / (2*sizeof(T)) source elements
// (e.g. 64 for half/bfloat16_t), regardless of mask or how wide the row
// actually is. A previous single-shot version issued exactly one LoadAlign
// per row assuming the whole row fit in one such read; for k (or
// srcRowStride) larger than chunkSize this silently over-reads past the row
// into adjacent memory. Each row is now processed in
// repeatTimes = ceil(k / chunkSize) chunks, matching the chunking approach
// used in SigmoidRenormBackwardVF.
// ============================================================================
template <typename T>
__simd_vf__ void CastGradYRowsVF(__ubuf__ T *srcAddr, __ubuf__ float *dstAddr, uint16_t curRows, uint16_t k,
                                 uint16_t srcRowStride, uint16_t dstRowStride)
{
    constexpr uint16_t chunkSize = VECTOR_REG_WIDTH / (2 * sizeof(T)); // 64 for half/bfloat16_t
    uint16_t repeatTimes = static_cast<uint16_t>((k + chunkSize - 1) / chunkSize);

    for (uint16_t row = 0; row < curRows; row++) {
        __ubuf__ T *srcRow = srcAddr + row * srcRowStride;
        __ubuf__ float *dstRow = dstAddr + row * dstRowStride;

        uint32_t remainingK = k;
        for (uint16_t c = 0; c < repeatTimes; c++) {
            uint16_t chunkOffset = c * chunkSize;
            MicroAPI::MaskReg preg = MicroAPI::UpdateMask<float>(remainingK);

            MicroAPI::RegTensor<T> srcB16;
            MicroAPI::RegTensor<float> dstF32;
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcB16, srcRow + chunkOffset);
            MicroAPI::Cast<float, T, castTraitB162B32>(dstF32, srcB16, preg);
            MicroAPI::StoreAlign(dstRow + chunkOffset, dstF32, preg);
        }
    }
}

// ============================================================================
// SigmoidRenormBackwardVF: per-row K-dimension renorm-backward, all rows in one
// VF call (row loop lives inside the VF, not in the __aicore__ caller).
//
// Per row: gather w' = xNorm[expertIdx] (K elements from an N-wide row);
//          D = eps + sum(w'); betaNum = sum(w' * gradYScaled);
//          beta = betaNum / D; gradWPrime = (gradYScaled - beta) / D;
//          scatter gradWPrime into gradNormX[expertIdx] (N-wide row, pre-zeroed
//          by the caller).
//
// CHUNKING (K > VL_T support): a single RegTensor<int32_t>/<float> can only
// hold VL_T = VECTOR_REG_WIDTH / sizeof(int32_t) = 64 lanes. For K > 64, the
// row's K elements are processed in repeatTimes = ceil(K / 64) chunks of up
// to 64 elements each. This is required for correctness, not just an
// optimization: a single LoadAlign of a K > 64 row would silently truncate
// to the first 64 elements otherwise.
//
//   Phase 1 (per chunk): gather this chunk's w' = xNorm[expertIdx], compute
//   this chunk's partial sum(w') and partial sum(w' * gradYScaled) via
//   Reduce<SUM> (lands at lane0), and accumulate those lane0 partials across
//   chunks into a single running D and betaNum (scalar-like accumulation
//   using RegTensor<float>'s lane0 via Adds/Add, mask=VL1).
//
//   Phase 2 (per chunk): recompute (or reuse) this chunk's w', broadcast the
//   final global invD/beta to every lane of this chunk, compute gradWPrime,
//   and Scatter.
//
// NOTE on expertIdx uniqueness: expertIdx is guaranteed by the caller/product
// contract to contain distinct values within each row (it originates from a
// TopK selection upstream), so no duplicate-index resolution is performed
// here. If this invariant is ever violated, the vectorized Scatter's
// write-conflict resolution for duplicate indices within a row is
// hardware-unspecified (see AscendC Scatter doc: "若有2个或以上index相同，
// 具体哪个数据生效未知") and will not reproduce any particular deterministic
// semantics.
//
// Reduce<SUM> lands the true horizontal sum at lane 0; it is broadcast back to
// all K lanes purely in registers via Duplicate<HighLowPart::LOWEST> (no UB
// round-trip, no scratch buffer, no LocalMemBar needed) -- same idiom as
// mla_prolog's RmsNormVFImpl.
//
// xNormBase   : [curRows, n]      row stride = n
// expertIdxBase: [curRows, kAlign] row stride = kAlign (int32 element indices, NOT byte offsets)
// gradYBase   : [curRows, kAlign] row stride = kAlign (already float32 + already *scalingFactor)
// gradNormXBase: [curRows, n]     row stride = n (scatter target, caller pre-zeroes)
// ============================================================================
__simd_vf__ void SigmoidRenormBackwardVF(__ubuf__ float *xNormBase, __ubuf__ int32_t *expertIdxBase,
                                         __ubuf__ float *gradYBase, __ubuf__ float *gradNormXBase, float eps,
                                         uint16_t curRows, uint16_t k, uint16_t kAlign, uint16_t n)
{
    constexpr uint16_t chunkSize = VECTOR_REG_WIDTH / sizeof(int32_t); // 64 for int32_t/float
    uint16_t repeatTimes = static_cast<uint16_t>((k + chunkSize - 1) / chunkSize);
    MicroAPI::MaskReg maskLane0 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();

    for (uint16_t row = 0; row < curRows; row++) {
        __ubuf__ int32_t *idxRow = expertIdxBase + row * kAlign;
        __ubuf__ float *xNormRow = xNormBase + row * n;
        __ubuf__ float *gradYRow = gradYBase + row * kAlign;
        __ubuf__ float *gradNormXRow = gradNormXBase + row * n;

        // ---- Phase 1: accumulate global D and betaNum across all chunks. ----
        MicroAPI::RegTensor<float> globalD, globalBetaNum;
        MicroAPI::Duplicate(globalD, eps, maskLane0);
        MicroAPI::Duplicate(globalBetaNum, 0.0f, maskLane0);

        // remainingK1 is decremented by UpdateMask each call (branch-free tail
        // handling: it naturally yields a full chunkSize mask for every chunk
        // except the last, which gets whatever is left). Kept separate from
        // Phase 2's counter since UpdateMask mutates its argument.
        uint32_t remainingK1 = k;
        for (uint16_t c = 0; c < repeatTimes; c++) {
            uint16_t chunkOffset = c * chunkSize;
            MicroAPI::MaskReg chunkMask = MicroAPI::UpdateMask<int32_t>(remainingK1);

            MicroAPI::RegTensor<int32_t> idxReg;
            MicroAPI::RegTensor<uint32_t> idxU32Reg;
            MicroAPI::RegTensor<float> wPrimeReg, gradYReg, tmpReg, chunkSum;

            MicroAPI::LoadAlign(idxReg, idxRow + chunkOffset);
            idxU32Reg = (MicroAPI::RegTensor<uint32_t> &)idxReg;
            MicroAPI::Gather(wPrimeReg, xNormRow, idxU32Reg, chunkMask);

            MicroAPI::Reduce<MicroAPI::ReduceType::SUM>(chunkSum, wPrimeReg, chunkMask);
            MicroAPI::Add(globalD, globalD, chunkSum, maskLane0);

            MicroAPI::LoadAlign(gradYReg, gradYRow + chunkOffset);
            MicroAPI::Mul(tmpReg, wPrimeReg, gradYReg, chunkMask);
            MicroAPI::Reduce<MicroAPI::ReduceType::SUM>(chunkSum, tmpReg, chunkMask);
            MicroAPI::Add(globalBetaNum, globalBetaNum, chunkSum, maskLane0);
        }

        MicroAPI::RegTensor<float> invDReg, betaReg;
        MicroAPI::Duplicate(invDReg, 1.0f, maskLane0);
        MicroAPI::Div(invDReg, invDReg, globalD, maskLane0);
        MicroAPI::Mul(betaReg, globalBetaNum, invDReg, maskLane0);

        // ---- Phase 2: recompute gradWPrime per chunk, scatter. ----
        uint32_t remainingK2 = k;
        for (uint16_t c = 0; c < repeatTimes; c++) {
            uint16_t chunkOffset = c * chunkSize;
            MicroAPI::MaskReg chunkMask = MicroAPI::UpdateMask<int32_t>(remainingK2);

            MicroAPI::RegTensor<int32_t> idxReg;
            MicroAPI::RegTensor<uint32_t> idxU32Reg;
            MicroAPI::RegTensor<float> wPrimeReg, gradYReg;

            MicroAPI::LoadAlign(idxReg, idxRow + chunkOffset);
            idxU32Reg = (MicroAPI::RegTensor<uint32_t> &)idxReg;
            MicroAPI::Gather(wPrimeReg, xNormRow, idxU32Reg, chunkMask);
            MicroAPI::LoadAlign(gradYReg, gradYRow + chunkOffset);

            // Broadcast the row-global invD/beta (lane0) to every lane of this chunk.
            MicroAPI::RegTensor<float> bcastReg;
            MicroAPI::Duplicate<float, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>(
                bcastReg, invDReg, chunkMask);
            MicroAPI::RegTensor<float> gradWPrimeReg;
            MicroAPI::Duplicate<float, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>(
                gradWPrimeReg, betaReg, chunkMask);
            MicroAPI::Sub(gradWPrimeReg, gradYReg, gradWPrimeReg, chunkMask);
            MicroAPI::Mul(gradWPrimeReg, gradWPrimeReg, bcastReg, chunkMask);

            MicroAPI::Scatter(gradNormXRow, gradWPrimeReg, idxU32Reg, chunkMask);
        }
    }
}

// ============================================================================
// CallSigmoidGradVF: __aicore__ wrapper.
//   Takes LocalTensor + element count; computes GetPhyAddr()/oneRepeatSize/
//   repeatTimes internally and dispatches to the correct __simd_vf__ variant.
// ============================================================================
template <typename T>
__aicore__ inline void CallSigmoidGradVF(LocalTensor<T> gradXOut, LocalTensor<float> xNorm,
                                         LocalTensor<float> gradNormX, uint32_t totalElements)
{
    __ubuf__ T *gradXAddr = (__ubuf__ T *)gradXOut.GetPhyAddr();
    __ubuf__ float *xNormAddr = (__ubuf__ float *)xNorm.GetPhyAddr();
    __ubuf__ float *gradNormXAddr = (__ubuf__ float *)gradNormX.GetPhyAddr();
    constexpr uint32_t oneRepeatSize = VECTOR_REG_WIDTH / sizeof(float);
    uint16_t repeatTimes = static_cast<uint16_t>((totalElements + oneRepeatSize - 1) / oneRepeatSize);
    if constexpr (IsSameType<T, float>::value) {
        SigmoidGradFP32VF(xNormAddr, gradNormXAddr, gradXAddr, totalElements, oneRepeatSize, repeatTimes);
    } else {
        SigmoidGradHalfVF<T>(xNormAddr, gradNormXAddr, gradXAddr, totalElements, oneRepeatSize, repeatTimes);
    }
}

// ============================================================================
// CallCastGradYFlatVF: __aicore__ wrapper for CastGradYFlatVF.
//   Takes LocalTensor + element count; computes GetPhyAddr()/oneRepeatSize/
//   repeatTimes internally.
// ============================================================================
template <typename T>
__aicore__ inline void CallCastGradYFlatVF(LocalTensor<T> gradYSrc, LocalTensor<float> dst, uint32_t totalElements)
{
    __ubuf__ T *srcAddr = (__ubuf__ T *)gradYSrc.GetPhyAddr();
    __ubuf__ float *dstAddr = (__ubuf__ float *)dst.GetPhyAddr();
    constexpr uint32_t oneRepeatSize = VECTOR_REG_WIDTH / sizeof(float);
    uint16_t repeatTimes = static_cast<uint16_t>((totalElements + oneRepeatSize - 1) / oneRepeatSize);
    CastGradYFlatVF<T>(srcAddr, dstAddr, totalElements, oneRepeatSize, repeatTimes);
}

// ============================================================================
// CallCastGradYRowsVF: __aicore__ wrapper for CastGradYRowsVF.
//   Takes LocalTensor + row geometry; computes GetPhyAddr() internally.
// ============================================================================
template <typename T>
__aicore__ inline void CallCastGradYRowsVF(LocalTensor<T> gradYSrc, LocalTensor<float> dst, uint16_t curRows,
                                           uint16_t k, uint16_t srcRowStride, uint16_t dstRowStride)
{
    __ubuf__ T *srcAddr = (__ubuf__ T *)gradYSrc.GetPhyAddr();
    __ubuf__ float *dstAddr = (__ubuf__ float *)dst.GetPhyAddr();
    CastGradYRowsVF<T>(srcAddr, dstAddr, curRows, k, srcRowStride, dstRowStride);
}

// ============================================================================
// CallSigmoidRenormBackwardVF: __aicore__ wrapper for SigmoidRenormBackwardVF.
//   Takes LocalTensor + row geometry; computes GetPhyAddr() internally.
//   Caller must pre-zero gradNormX before calling (VF only scatters K nonzero
//   entries per row; the rest of each row is left untouched). expertIdx is
//   assumed unique within each row (see SigmoidRenormBackwardVF comment).
// ============================================================================
__aicore__ inline void CallSigmoidRenormBackwardVF(LocalTensor<float> xNorm, LocalTensor<int32_t> expertIdx,
                                                    LocalTensor<float> gradY, LocalTensor<float> gradNormX, float eps,
                                                    uint16_t curRows, uint16_t k, uint16_t kAlign, uint16_t n)
{
    __ubuf__ float *xNormAddr = (__ubuf__ float *)xNorm.GetPhyAddr();
    __ubuf__ int32_t *expertIdxAddr = (__ubuf__ int32_t *)expertIdx.GetPhyAddr();
    __ubuf__ float *gradYAddr = (__ubuf__ float *)gradY.GetPhyAddr();
    __ubuf__ float *gradNormXAddr = (__ubuf__ float *)gradNormX.GetPhyAddr();
    SigmoidRenormBackwardVF(xNormAddr, expertIdxAddr, gradYAddr, gradNormXAddr, eps, curRows, k, kAlign, n);
}

} // namespace MoeGatingTopKBackwardNs
#endif // MOE_GATING_TOP_K_BACKWARD_VF_H
