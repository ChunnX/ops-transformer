/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_ATTN_RES_UPDATE_REGBASE_H
#define BLOCK_ATTN_RES_UPDATE_REGBASE_H

#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "block_attn_res_update_tiling_data.h"

namespace BlockAttnResUpdateOps {

namespace Reg = AscendC::Reg;
namespace Te = AscendC::Te;

constexpr uint32_t BARU_FP32_BYTES = sizeof(float);
constexpr uint32_t BARU_BF16_BYTES = sizeof(bfloat16_t);
constexpr uint32_t BARU_VREG_FP32_ELEMENTS = 64;
constexpr uint32_t BARU_VREG_FP32_SHIFT = 6;
constexpr uint32_t BARU_VREG_FP32_MASK = BARU_VREG_FP32_ELEMENTS - 1U;
constexpr uint32_t BARU_BUFFER_NUM = 2;
constexpr uint32_t BARU_ROW_BATCH_MIN_ROWS = 8;

constexpr AscendC::Reg::CastTrait BARU_CAST_BF16_TO_FP32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::Reg::CastTrait BARU_CAST_FP32_TO_BF16 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

/**
 * One-VREG Phase 1 path. The query is invariant across rows, so load and tail-clear it once.
 */
__simd_vf__ inline void BlockAttnResUpdatePhase1OneVLVF(
    __ubuf__ float *partial, __ubuf__ bfloat16_t *delta, __ubuf__ float *effectiveQuery,
    __ubuf__ float *stats, uint16_t rows, uint32_t cols, uint32_t colsAlignFp32,
    uint32_t colsAlignBf16, uint32_t scalarAlign, float epsilon, float invCols)
{
    Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg scalarMask = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
    const uint16_t hasTail = static_cast<uint16_t>(cols < BARU_VREG_FP32_ELEMENTS);
    uint32_t remainingCols = cols;
    Reg::MaskReg colMask = Reg::UpdateMask<float>(remainingCols);

    Reg::RegTensor<float> partialReg;
    Reg::RegTensor<float> deltaFp32Reg;
    Reg::RegTensor<float> queryReg;
    Reg::RegTensor<float> dotProductReg;
    Reg::RegTensor<float> squareSumReg;
    Reg::RegTensor<float> dotSumReg;
    Reg::RegTensor<float> rmsReg;
    Reg::RegTensor<float> scoreReg;
    Reg::RegTensor<bfloat16_t> deltaReg;

    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(queryReg, effectiveQuery);
    // Add below zeroes inactive partial lanes. Clear query padding once so the reductions can use allMask.
    for (uint16_t loop = 0; loop < hasTail; ++loop) {
        Reg::ShiftLefts((Reg::RegTensor<uint32_t> &)queryReg, (Reg::RegTensor<uint32_t> &)queryReg,
                        static_cast<int16_t>(0), colMask);
    }

    for (uint16_t row = 0; row < rows; ++row) {
        const uint32_t partialRowOffset = static_cast<uint32_t>(row) * colsAlignFp32;
        const uint32_t deltaRowOffset = static_cast<uint32_t>(row) * colsAlignBf16;

        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg, partial + partialRowOffset);
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(deltaReg, delta + deltaRowOffset);
        Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg, deltaReg, colMask);
        Reg::Add<float>(partialReg, partialReg, deltaFp32Reg, colMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + partialRowOffset, partialReg, colMask);

        // Form the dot product before reusing partialReg for the square product.
        Reg::Mul<float>(dotProductReg, partialReg, queryReg, allMask);
        Reg::Mul<float>(partialReg, partialReg, partialReg, allMask);
        Reg::Reduce<Reg::ReduceType::SUM>(squareSumReg, partialReg, allMask);
        Reg::Reduce<Reg::ReduceType::SUM>(dotSumReg, dotProductReg, allMask);
        Reg::Muls<float>(squareSumReg, squareSumReg, invCols, scalarMask);
        Reg::Adds<float>(squareSumReg, squareSumReg, epsilon, scalarMask);
        Reg::Sqrt<float>(rmsReg, squareSumReg, scalarMask);
        Reg::Div<float>(scoreReg, dotSumReg, rmsReg, scalarMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
            stats + 2 * scalarAlign + row, scoreReg, scalarMask);
    }
}

/**
 * Two-VREG Phase 1 path. Both query vectors stay resident across the row loop; the second is tail-cleared once.
 */
__simd_vf__ inline void BlockAttnResUpdatePhase1TwoVLVF(
    __ubuf__ float *partial, __ubuf__ bfloat16_t *delta, __ubuf__ float *effectiveQuery,
    __ubuf__ float *stats, uint16_t rows, uint32_t cols, uint32_t colsAlignFp32,
    uint32_t colsAlignBf16, uint32_t scalarAlign, float epsilon, float invCols)
{
    Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg scalarMask = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
    uint32_t tailCols = cols - BARU_VREG_FP32_ELEMENTS;
    const uint16_t hasTail = static_cast<uint16_t>(tailCols < BARU_VREG_FP32_ELEMENTS);
    Reg::MaskReg tailMask = Reg::UpdateMask<float>(tailCols);

    Reg::RegTensor<float> partialReg0;
    Reg::RegTensor<float> partialReg1;
    Reg::RegTensor<float> deltaFp32Reg0;
    Reg::RegTensor<float> deltaFp32Reg1;
    Reg::RegTensor<float> queryReg0;
    Reg::RegTensor<float> queryReg1;
    Reg::RegTensor<float> dotProductReg0;
    Reg::RegTensor<float> squareSumReg;
    Reg::RegTensor<float> dotSumReg;
    Reg::RegTensor<float> rmsReg;
    Reg::RegTensor<float> scoreReg;
    Reg::RegTensor<bfloat16_t> deltaReg0;
    Reg::RegTensor<bfloat16_t> deltaReg1;

    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(queryReg0, effectiveQuery);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
        queryReg1, effectiveQuery + BARU_VREG_FP32_ELEMENTS);
    for (uint16_t loop = 0; loop < hasTail; ++loop) {
        Reg::ShiftLefts((Reg::RegTensor<uint32_t> &)queryReg1, (Reg::RegTensor<uint32_t> &)queryReg1,
                        static_cast<int16_t>(0), tailMask);
    }

    for (uint16_t row = 0; row < rows; ++row) {
        const uint32_t partialRowOffset = static_cast<uint32_t>(row) * colsAlignFp32;
        const uint32_t deltaRowOffset = static_cast<uint32_t>(row) * colsAlignBf16;

        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0, partial + partialRowOffset);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
            partialReg1, partial + partialRowOffset + BARU_VREG_FP32_ELEMENTS);
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(deltaReg0, delta + deltaRowOffset);
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(
            deltaReg1, delta + deltaRowOffset + BARU_VREG_FP32_ELEMENTS);
        Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg0, deltaReg0, allMask);
        Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg1, deltaReg1, tailMask);
        Reg::Add<float>(partialReg0, partialReg0, deltaFp32Reg0, allMask);
        Reg::Add<float>(partialReg1, partialReg1, deltaFp32Reg1, tailMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + partialRowOffset, partialReg0, allMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(
            partial + partialRowOffset + BARU_VREG_FP32_ELEMENTS, partialReg1, tailMask);

        // Fold the second vector directly into the first product. queryReg1 padding was cleared above, and the
        // masked Add zeroes partialReg1 padding, so the ZEROING-only MulAddDst can safely use allMask.
        Reg::Mul<float>(dotProductReg0, partialReg0, queryReg0, allMask);
        Reg::Mul<float>(partialReg0, partialReg0, partialReg0, allMask);
        Reg::MulAddDst<float>(dotProductReg0, partialReg1, queryReg1, allMask);
        Reg::MulAddDst<float>(partialReg0, partialReg1, partialReg1, allMask);
        Reg::Reduce<Reg::ReduceType::SUM>(squareSumReg, partialReg0, allMask);
        Reg::Reduce<Reg::ReduceType::SUM>(dotSumReg, dotProductReg0, allMask);
        Reg::Muls<float>(squareSumReg, squareSumReg, invCols, scalarMask);
        Reg::Adds<float>(squareSumReg, squareSumReg, epsilon, scalarMask);
        Reg::Sqrt<float>(rmsReg, squareSumReg, scalarMask);
        Reg::Div<float>(scoreReg, dotSumReg, rmsReg, scalarMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
            stats + 2 * scalarAlign + row, scoreReg, scalarMask);
    }
}

/**
 * Phase 1 updates partial in FP32 and stores one score for every row.
 * cols is the effective reduction length; aligned strides are used only for UB addressing.
 */
__simd_vf__ inline void BlockAttnResUpdatePhase1VF(__ubuf__ float *partial, __ubuf__ bfloat16_t *delta,
                                                   __ubuf__ float *effectiveQuery, __ubuf__ float *stats,
                                                   uint16_t rows, uint32_t cols, uint32_t colsAlignFp32,
                                                   uint32_t colsAlignBf16, uint32_t scalarAlign,
                                                   float epsilon, float invCols)
{
    Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg scalarMask = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
    Reg::MaskReg tailMask;

    Reg::RegTensor<float> partialReg0;
    Reg::RegTensor<float> partialReg1;
    Reg::RegTensor<float> deltaFp32Reg0;
    Reg::RegTensor<float> deltaFp32Reg1;
    Reg::RegTensor<float> firstQueryReg0;
    Reg::RegTensor<float> firstQueryReg1;
    Reg::RegTensor<float> loopQueryReg0;
    Reg::RegTensor<float> loopQueryReg1;
    Reg::RegTensor<float> oddQueryReg;
    Reg::RegTensor<float> tailQueryReg;
    Reg::RegTensor<float> squareAccReg0;
    Reg::RegTensor<float> squareAccReg1;
    Reg::RegTensor<float> dotAccReg0;
    Reg::RegTensor<float> dotAccReg1;
    Reg::RegTensor<float> squareSumReg;
    Reg::RegTensor<float> dotSumReg;
    Reg::RegTensor<float> rmsReg;
    Reg::RegTensor<float> scoreReg;
    Reg::RegTensor<bfloat16_t> deltaReg0;
    Reg::RegTensor<bfloat16_t> deltaReg1;

    const uint16_t fullLoops = static_cast<uint16_t>(cols >> BARU_VREG_FP32_SHIFT);
    const uint16_t pairLoops = static_cast<uint16_t>(fullLoops >> 1U);
    // This VF is dispatched only for cols > 2 * VL, so at least one complete pair is always available to peel.
    const uint16_t remainingPairLoops = static_cast<uint16_t>(pairLoops - 1U);
    const uint16_t hasOddFull = static_cast<uint16_t>(fullLoops & 1U);
    const uint32_t tailCols = cols & BARU_VREG_FP32_MASK;
    const uint16_t hasTail = static_cast<uint16_t>(tailCols != 0U);
    const uint32_t oddVectorOffset =
        static_cast<uint32_t>(pairLoops) * 2U * BARU_VREG_FP32_ELEMENTS;
    const uint32_t tailVectorOffset = static_cast<uint32_t>(fullLoops) * BARU_VREG_FP32_ELEMENTS;
    // The tail length is row-invariant. UpdateMask mutates only tailRemaining; the returned MaskReg is reusable.
    uint32_t tailRemaining = tailCols;
    for (uint16_t loop = 0; loop < hasTail; ++loop) {
        tailMask = Reg::UpdateMask<float>(tailRemaining);
    }

    // Query is invariant across rows. Keep the first pair and the optional remainders resident; only middle pairs
    // are streamed inside the row loop. The tail query is cleared once for all-row allMask accumulation.
    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(firstQueryReg0, effectiveQuery);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
        firstQueryReg1, effectiveQuery + BARU_VREG_FP32_ELEMENTS);
    for (uint16_t loop = 0; loop < hasOddFull; ++loop) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(oddQueryReg, effectiveQuery + oddVectorOffset);
    }
    for (uint16_t loop = 0; loop < hasTail; ++loop) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(tailQueryReg, effectiveQuery + tailVectorOffset);
        Reg::ShiftLefts((Reg::RegTensor<uint32_t> &)tailQueryReg, (Reg::RegTensor<uint32_t> &)tailQueryReg,
                        static_cast<int16_t>(0), tailMask);
    }

    for (uint16_t row = 0; row < rows; ++row) {
        const uint32_t partialRowOffset = static_cast<uint32_t>(row) * colsAlignFp32;
        const uint32_t deltaRowOffset = static_cast<uint32_t>(row) * colsAlignBf16;

        // Peel the guaranteed first pair: Mul initializes all four accumulators and removes four per-row Duplicate
        // instructions. The two independent vectors still halve each accumulation dependency chain.
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0, partial + partialRowOffset);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
            partialReg1, partial + partialRowOffset + BARU_VREG_FP32_ELEMENTS);
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(deltaReg0, delta + deltaRowOffset);
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(
            deltaReg1, delta + deltaRowOffset + BARU_VREG_FP32_ELEMENTS);
        Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg0, deltaReg0, allMask);
        Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg1, deltaReg1, allMask);
        Reg::Add<float>(partialReg0, partialReg0, deltaFp32Reg0, allMask);
        Reg::Add<float>(partialReg1, partialReg1, deltaFp32Reg1, allMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + partialRowOffset, partialReg0, allMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(
            partial + partialRowOffset + BARU_VREG_FP32_ELEMENTS, partialReg1, allMask);
        Reg::Mul<float>(squareAccReg0, partialReg0, partialReg0, allMask);
        Reg::Mul<float>(squareAccReg1, partialReg1, partialReg1, allMask);
        Reg::Mul<float>(dotAccReg0, partialReg0, firstQueryReg0, allMask);
        Reg::Mul<float>(dotAccReg1, partialReg1, firstQueryReg1, allMask);

        for (uint16_t loop = 0; loop < remainingPairLoops; ++loop) {
            const uint32_t vectorOffset0 =
                (static_cast<uint32_t>(loop) * 2U + 2U) * BARU_VREG_FP32_ELEMENTS;
            const uint32_t vectorOffset1 = vectorOffset0 + BARU_VREG_FP32_ELEMENTS;

            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0,
                                                            partial + partialRowOffset + vectorOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg1,
                                                            partial + partialRowOffset + vectorOffset1);
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(
                deltaReg0, delta + deltaRowOffset + vectorOffset0);
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(
                deltaReg1, delta + deltaRowOffset + vectorOffset1);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(loopQueryReg0, effectiveQuery + vectorOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(loopQueryReg1, effectiveQuery + vectorOffset1);

            Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg0, deltaReg0, allMask);
            Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg1, deltaReg1, allMask);
            Reg::Add<float>(partialReg0, partialReg0, deltaFp32Reg0, allMask);
            Reg::Add<float>(partialReg1, partialReg1, deltaFp32Reg1, allMask);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + partialRowOffset + vectorOffset0,
                                                              partialReg0, allMask);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + partialRowOffset + vectorOffset1,
                                                              partialReg1, allMask);

            Reg::MulAddDst<float>(squareAccReg0, partialReg0, partialReg0, allMask);
            Reg::MulAddDst<float>(squareAccReg1, partialReg1, partialReg1, allMask);
            Reg::MulAddDst<float>(dotAccReg0, partialReg0, loopQueryReg0, allMask);
            Reg::MulAddDst<float>(dotAccReg1, partialReg1, loopQueryReg1, allMask);
        }

        for (uint16_t loop = 0; loop < hasOddFull; ++loop) {
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0,
                                                            partial + partialRowOffset + oddVectorOffset);
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(
                deltaReg0, delta + deltaRowOffset + oddVectorOffset);
            Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg0, deltaReg0, allMask);
            Reg::Add<float>(partialReg0, partialReg0, deltaFp32Reg0, allMask);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + partialRowOffset + oddVectorOffset,
                                                              partialReg0, allMask);
            Reg::MulAddDst<float>(squareAccReg0, partialReg0, partialReg0, allMask);
            Reg::MulAddDst<float>(dotAccReg0, partialReg0, oddQueryReg, allMask);
        }

        for (uint16_t loop = 0; loop < hasTail; ++loop) {
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0,
                                                            partial + partialRowOffset + tailVectorOffset);
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(
                deltaReg0, delta + deltaRowOffset + tailVectorOffset);
            Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg0, deltaReg0, tailMask);
            Reg::Add<float>(partialReg0, partialReg0, deltaFp32Reg0, tailMask);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + partialRowOffset + tailVectorOffset,
                                                              partialReg0, tailMask);

            // MulAddDst only supports ZEROING mode. partial padding was cleared by Add and query padding was
            // cleared before the row loop, so inactive lanes cannot erase the resident destination accumulator.
            Reg::MulAddDst<float>(squareAccReg1, partialReg0, partialReg0, allMask);
            Reg::MulAddDst<float>(dotAccReg1, partialReg0, tailQueryReg, allMask);
        }

        Reg::Add<float>(squareAccReg0, squareAccReg0, squareAccReg1, allMask);
        Reg::Add<float>(dotAccReg0, dotAccReg0, dotAccReg1, allMask);
        Reg::Reduce<Reg::ReduceType::SUM>(squareSumReg, squareAccReg0, allMask);
        Reg::Reduce<Reg::ReduceType::SUM>(dotSumReg, dotAccReg0, allMask);
        Reg::Muls<float>(squareSumReg, squareSumReg, invCols, scalarMask);
        Reg::Adds<float>(squareSumReg, squareSumReg, epsilon, scalarMask);
        Reg::Sqrt<float>(rmsReg, squareSumReg, scalarMask);
        Reg::Div<float>(scoreReg, dotSumReg, rmsReg, scalarMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(stats + 2 * scalarAlign + row, scoreReg,
                                                                       scalarMask);
    }
}

/**
 * Optional Phase 1 binary-reduction candidate. The current kernel does not dispatch to this VF.
 * binaryReduceTmp must point to 512 bytes of 256-byte-aligned UB scratch: one FP32 VREG for
 * square partials followed by one FP32 VREG for dot partials. With cols <= 8192, folding two
 * input VREGs into each partial produces at most 64 partials per statistic.
 */
__simd_vf__ inline void BlockAttnResUpdatePhase1BinaryReduceVF(
    __ubuf__ float *partial, __ubuf__ bfloat16_t *delta, __ubuf__ float *effectiveQuery,
    __ubuf__ float *stats, __ubuf__ float *binaryReduceTmp, uint16_t rows, uint32_t cols,
    uint32_t colsAlignFp32, uint32_t colsAlignBf16, uint32_t scalarAlign, float epsilon, float invCols)
{
    Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg scalarMask = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
    Reg::MaskReg tailMask;
    Reg::MaskReg partialMask;

    Reg::RegTensor<float> partialReg0;
    Reg::RegTensor<float> partialReg1;
    Reg::RegTensor<float> deltaFp32Reg0;
    Reg::RegTensor<float> deltaFp32Reg1;
    Reg::RegTensor<float> queryReg0;
    Reg::RegTensor<float> queryReg1;
    Reg::RegTensor<float> squarePairReg;
    Reg::RegTensor<float> dotPairReg;
    Reg::RegTensor<float> squarePartialReg;
    Reg::RegTensor<float> dotPartialReg;
    Reg::RegTensor<float> squareSumReg;
    Reg::RegTensor<float> dotSumReg;
    Reg::RegTensor<float> rmsReg;
    Reg::RegTensor<float> scoreReg;
    Reg::RegTensor<bfloat16_t> deltaReg0;
    Reg::RegTensor<bfloat16_t> deltaReg1;

    __ubuf__ float *squareReduceTmp = binaryReduceTmp;
    __ubuf__ float *dotReduceTmp = binaryReduceTmp + BARU_VREG_FP32_ELEMENTS;

    const uint16_t fullLoops = static_cast<uint16_t>(cols >> BARU_VREG_FP32_SHIFT);
    const uint16_t pairLoops = static_cast<uint16_t>(fullLoops >> 1U);
    const uint16_t hasOddFull = static_cast<uint16_t>(fullLoops & 1U);
    const uint32_t tailCols = cols & BARU_VREG_FP32_MASK;
    const uint16_t hasTail = static_cast<uint16_t>(tailCols != 0U);
    const uint16_t hasRemainder = static_cast<uint16_t>(hasOddFull | hasTail);
    const uint32_t partialCount = static_cast<uint32_t>(pairLoops + hasRemainder);
    // Both mask lengths are invariant across rows, so materialize them before the row loop.
    uint32_t tailRemaining = tailCols;
    for (uint16_t loop = 0; loop < hasTail; ++loop) {
        tailMask = Reg::UpdateMask<float>(tailRemaining);
    }
    uint32_t partialRemaining = partialCount;
    partialMask = Reg::UpdateMask<float>(partialRemaining);

    for (uint16_t row = 0; row < rows; ++row) {
        const uint32_t partialRowOffset = static_cast<uint32_t>(row) * colsAlignFp32;
        const uint32_t deltaRowOffset = static_cast<uint32_t>(row) * colsAlignBf16;

        // Each pair is independent: reduce it immediately to remove the cross-pair accumulation chain.
        for (uint16_t loop = 0; loop < pairLoops; ++loop) {
            const uint32_t vectorOffset0 =
                static_cast<uint32_t>(loop) * 2U * BARU_VREG_FP32_ELEMENTS;
            const uint32_t vectorOffset1 = vectorOffset0 + BARU_VREG_FP32_ELEMENTS;

            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0,
                                                            partial + partialRowOffset + vectorOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg1,
                                                            partial + partialRowOffset + vectorOffset1);
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(
                deltaReg0, delta + deltaRowOffset + vectorOffset0);
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(
                deltaReg1, delta + deltaRowOffset + vectorOffset1);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(queryReg0, effectiveQuery + vectorOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(queryReg1, effectiveQuery + vectorOffset1);

            Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg0, deltaReg0, allMask);
            Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg1, deltaReg1, allMask);
            Reg::Add<float>(partialReg0, partialReg0, deltaFp32Reg0, allMask);
            Reg::Add<float>(partialReg1, partialReg1, deltaFp32Reg1, allMask);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + partialRowOffset + vectorOffset0,
                                                              partialReg0, allMask);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + partialRowOffset + vectorOffset1,
                                                              partialReg1, allMask);

            Reg::Mul<float>(squarePairReg, partialReg0, partialReg0, allMask);
            Reg::Mul<float>(dotPairReg, partialReg0, queryReg0, allMask);
            Reg::MulAddDst<float>(squarePairReg, partialReg1, partialReg1, allMask);
            Reg::MulAddDst<float>(dotPairReg, partialReg1, queryReg1, allMask);
            Reg::Reduce<Reg::ReduceType::SUM>(squarePartialReg, squarePairReg, allMask);
            Reg::Reduce<Reg::ReduceType::SUM>(dotPartialReg, dotPairReg, allMask);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(squareReduceTmp + loop,
                                                                           squarePartialReg, scalarMask);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dotReduceTmp + loop,
                                                                           dotPartialReg, scalarMask);
        }

        // Fold an odd full VREG and the tail, when present, into one final partial.
        for (uint16_t remainder = 0; remainder < hasRemainder; ++remainder) {
            Reg::Duplicate(squarePairReg, 0.0F);
            Reg::Duplicate(dotPairReg, 0.0F);

            for (uint16_t loop = 0; loop < hasOddFull; ++loop) {
                const uint32_t vectorOffset =
                    (static_cast<uint32_t>(pairLoops) * 2U + loop) * BARU_VREG_FP32_ELEMENTS;
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0,
                                                                partial + partialRowOffset + vectorOffset);
                Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(
                    deltaReg0, delta + deltaRowOffset + vectorOffset);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(queryReg0, effectiveQuery + vectorOffset);
                Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg0, deltaReg0, allMask);
                Reg::Add<float>(partialReg0, partialReg0, deltaFp32Reg0, allMask);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + partialRowOffset + vectorOffset,
                                                                  partialReg0, allMask);
                Reg::MulAddDst<float>(squarePairReg, partialReg0, partialReg0, allMask);
                Reg::MulAddDst<float>(dotPairReg, partialReg0, queryReg0, allMask);
            }

            for (uint16_t loop = 0; loop < hasTail; ++loop) {
                const uint32_t vectorOffset =
                    (static_cast<uint32_t>(fullLoops) + loop) * BARU_VREG_FP32_ELEMENTS;
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0,
                                                                partial + partialRowOffset + vectorOffset);
                Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(
                    deltaReg0, delta + deltaRowOffset + vectorOffset);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(queryReg0, effectiveQuery + vectorOffset);
                Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg0, deltaReg0, tailMask);
                Reg::Add<float>(partialReg0, partialReg0, deltaFp32Reg0, tailMask);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + partialRowOffset + vectorOffset,
                                                                  partialReg0, tailMask);
                Reg::ShiftLefts((Reg::RegTensor<uint32_t> &)queryReg0, (Reg::RegTensor<uint32_t> &)queryReg0,
                                static_cast<int16_t>(0), tailMask);
                Reg::MulAddDst<float>(squarePairReg, partialReg0, partialReg0, allMask);
                Reg::MulAddDst<float>(dotPairReg, partialReg0, queryReg0, allMask);
            }

            Reg::Reduce<Reg::ReduceType::SUM>(squarePartialReg, squarePairReg, allMask);
            Reg::Reduce<Reg::ReduceType::SUM>(dotPartialReg, dotPairReg, allMask);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(squareReduceTmp + pairLoops,
                                                                           squarePartialReg, scalarMask);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dotReduceTmp + pairLoops,
                                                                           dotPartialReg, scalarMask);
        }

        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(squarePairReg, squareReduceTmp);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(dotPairReg, dotReduceTmp);
        Reg::Reduce<Reg::ReduceType::SUM>(squareSumReg, squarePairReg, partialMask);
        Reg::Reduce<Reg::ReduceType::SUM>(dotSumReg, dotPairReg, partialMask);
        Reg::Muls<float>(squareSumReg, squareSumReg, invCols, scalarMask);
        Reg::Adds<float>(squareSumReg, squareSumReg, epsilon, scalarMask);
        Reg::Sqrt<float>(rmsReg, squareSumReg, scalarMask);
        Reg::Div<float>(scoreReg, dotSumReg, rmsReg, scalarMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(stats + 2 * scalarAlign + row, scoreReg,
                                                                       scalarMask);
    }
}

/**
 * Phase 2 updates the online-softmax coefficients and forms BF16 h.
 * Alpha and beta are normalized once per row so that each column loop needs only Mul + MulDstAdd.
 * The two VF calls are ordered on V; delta has been fully consumed before this phase reuses it as h.
 */
__simd_vf__ inline void BlockAttnResUpdatePhase2VF(__ubuf__ float *partial, __ubuf__ bfloat16_t *deltaH,
                                                   __ubuf__ float *interNumerator, __ubuf__ float *stats,
                                                   uint16_t rows, uint32_t cols, uint32_t colsAlignFp32,
                                                   uint32_t colsAlignBf16, uint32_t scalarAlign,
                                                   uint16_t colsLoops)
{
    Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg tailMask;
    Reg::RegTensor<float> partialReg0;
    Reg::RegTensor<float> partialReg1;
    Reg::RegTensor<float> tmpReg0;
    Reg::RegTensor<float> tmpReg1;
    Reg::RegTensor<float> scoreReg;
    Reg::RegTensor<float> historyMaxReg;
    Reg::RegTensor<float> historyEllReg;
    Reg::RegTensor<float> currentMaxReg;
    Reg::RegTensor<float> alphaReg;
    Reg::RegTensor<float> betaReg;
    Reg::RegTensor<float> numeratorReg0;
    Reg::RegTensor<float> numeratorReg1;
    Reg::RegTensor<bfloat16_t> outputBf16Reg0;
    Reg::RegTensor<bfloat16_t> outputBf16Reg1;

    const uint32_t tailCols = cols & BARU_VREG_FP32_MASK;
    const uint16_t hasTail = static_cast<uint16_t>(tailCols != 0U);
    const uint16_t fullLoops = static_cast<uint16_t>(colsLoops - hasTail);
    const uint16_t pairFullLoops = static_cast<uint16_t>(fullLoops >> 1U);
    const uint16_t hasOddFull = static_cast<uint16_t>(fullLoops & 1U);
    const uint16_t hasMixedPair = static_cast<uint16_t>(hasOddFull & hasTail);
    const uint16_t hasOddFullOnly = static_cast<uint16_t>(hasOddFull - hasMixedPair);
    const uint16_t hasTailOnly = static_cast<uint16_t>(hasTail - hasMixedPair);
    // Full vectors reuse allMask; only the true tail needs one dynamic mask for the whole VF call.
    uint32_t tailRemaining = tailCols;
    for (uint16_t loop = 0; loop < hasTail; ++loop) {
        tailMask = Reg::UpdateMask<float>(tailRemaining);
    }

    for (uint16_t row = 0; row < rows; ++row) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(historyMaxReg, stats + row);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(historyEllReg, stats + scalarAlign + row);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(scoreReg, stats + 2 * scalarAlign + row);
        Reg::Max<float>(currentMaxReg, historyMaxReg, scoreReg, allMask);
        Reg::ExpSub<float>(alphaReg, historyMaxReg, currentMaxReg, allMask);
        Reg::ExpSub<float>(betaReg, scoreReg, currentMaxReg, allMask);
        Reg::MulDstAdd<float>(historyEllReg, alphaReg, betaReg, allMask);
        // The two direct divisions are independent and remove the reciprocal-and-multiply dependency chain.
        Reg::Div<float>(alphaReg, alphaReg, historyEllReg, allMask);
        Reg::Div<float>(betaReg, betaReg, historyEllReg, allMask);
        const uint32_t fp32RowOffset = static_cast<uint32_t>(row) * colsAlignFp32;
        const uint32_t bf16RowOffset = static_cast<uint32_t>(row) * colsAlignBf16;
        // Separate register sets expose two adjacent column vectors to RVEC dual issue.
        for (uint16_t loop = 0; loop < pairFullLoops; ++loop) {
            const uint32_t vectorOffset0 =
                static_cast<uint32_t>(loop) * 2U * BARU_VREG_FP32_ELEMENTS;
            const uint32_t vectorOffset1 = vectorOffset0 + BARU_VREG_FP32_ELEMENTS;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0,
                                                            partial + fp32RowOffset + vectorOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg1,
                                                            partial + fp32RowOffset + vectorOffset1);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                numeratorReg0, interNumerator + fp32RowOffset + vectorOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                numeratorReg1, interNumerator + fp32RowOffset + vectorOffset1);
            Reg::Mul<float>(tmpReg0, partialReg0, betaReg, allMask);
            Reg::Mul<float>(tmpReg1, partialReg1, betaReg, allMask);
            Reg::MulDstAdd<float>(numeratorReg0, alphaReg, tmpReg0, allMask);
            Reg::MulDstAdd<float>(numeratorReg1, alphaReg, tmpReg1, allMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg0, numeratorReg0, allMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg1, numeratorReg1, allMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
                deltaH + bf16RowOffset + vectorOffset0, outputBf16Reg0, allMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
                deltaH + bf16RowOffset + vectorOffset1, outputBf16Reg1, allMask);
        }

        // Keep an odd full vector paired with the tail so the original two-column ILP is preserved.
        for (uint16_t loop = 0; loop < hasMixedPair; ++loop) {
            const uint32_t vectorOffset0 =
                (static_cast<uint32_t>(pairFullLoops) * 2U + static_cast<uint32_t>(loop) * 2U) *
                BARU_VREG_FP32_ELEMENTS;
            const uint32_t vectorOffset1 = vectorOffset0 + BARU_VREG_FP32_ELEMENTS;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0,
                                                            partial + fp32RowOffset + vectorOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg1,
                                                            partial + fp32RowOffset + vectorOffset1);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                numeratorReg0, interNumerator + fp32RowOffset + vectorOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                numeratorReg1, interNumerator + fp32RowOffset + vectorOffset1);
            Reg::Mul<float>(tmpReg0, partialReg0, betaReg, allMask);
            Reg::Mul<float>(tmpReg1, partialReg1, betaReg, tailMask);
            Reg::MulDstAdd<float>(numeratorReg0, alphaReg, tmpReg0, allMask);
            Reg::MulDstAdd<float>(numeratorReg1, alphaReg, tmpReg1, tailMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg0, numeratorReg0, allMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg1, numeratorReg1, tailMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
                deltaH + bf16RowOffset + vectorOffset0, outputBf16Reg0, allMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
                deltaH + bf16RowOffset + vectorOffset1, outputBf16Reg1, tailMask);
        }

        for (uint16_t loop = 0; loop < hasOddFullOnly; ++loop) {
            const uint32_t vectorOffset =
                (static_cast<uint32_t>(pairFullLoops) * 2U + loop) * BARU_VREG_FP32_ELEMENTS;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0,
                                                            partial + fp32RowOffset + vectorOffset);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                numeratorReg0, interNumerator + fp32RowOffset + vectorOffset);
            Reg::Mul<float>(tmpReg0, partialReg0, betaReg, allMask);
            Reg::MulDstAdd<float>(numeratorReg0, alphaReg, tmpReg0, allMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg0, numeratorReg0, allMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
                deltaH + bf16RowOffset + vectorOffset, outputBf16Reg0, allMask);
        }

        for (uint16_t loop = 0; loop < hasTailOnly; ++loop) {
            const uint32_t vectorOffset =
                (static_cast<uint32_t>(fullLoops) + loop) * BARU_VREG_FP32_ELEMENTS;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0,
                                                            partial + fp32RowOffset + vectorOffset);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                numeratorReg0, interNumerator + fp32RowOffset + vectorOffset);
            Reg::Mul<float>(tmpReg0, partialReg0, betaReg, tailMask);
            Reg::MulDstAdd<float>(numeratorReg0, alphaReg, tmpReg0, tailMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg0, numeratorReg0, tailMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
                deltaH + bf16RowOffset + vectorOffset, outputBf16Reg0, tailMask);
        }
    }
}

/**
 * Small-D Phase 2 vectorizes the coefficient calculation across tokens. Normalized alpha and beta overwrite
 * the consumed history-max and history-ell regions in stats, then are broadcast per row to form BF16 h.
 * scalarAlign is VREG-aligned for this path, so the masked final coefficient load/store stays in its region.
 */
__simd_vf__ inline void BlockAttnResUpdatePhase2RowBatchVF(
    __ubuf__ float *partial, __ubuf__ bfloat16_t *deltaH, __ubuf__ float *interNumerator,
    __ubuf__ float *stats, uint16_t rows, uint32_t cols, uint32_t colsAlignFp32,
    uint32_t colsAlignBf16, uint32_t scalarAlign, uint16_t colsLoops)
{
    Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg rowTailMask;
    Reg::MaskReg colTailMask;
    Reg::RegTensor<float> partialReg0;
    Reg::RegTensor<float> partialReg1;
    Reg::RegTensor<float> tmpReg0;
    Reg::RegTensor<float> tmpReg1;
    Reg::RegTensor<float> scoreReg;
    Reg::RegTensor<float> historyMaxReg;
    Reg::RegTensor<float> historyEllReg;
    Reg::RegTensor<float> currentMaxReg;
    Reg::RegTensor<float> alphaReg;
    Reg::RegTensor<float> betaReg;
    Reg::RegTensor<float> numeratorReg0;
    Reg::RegTensor<float> numeratorReg1;
    Reg::RegTensor<bfloat16_t> outputBf16Reg0;
    Reg::RegTensor<bfloat16_t> outputBf16Reg1;

    const uint16_t fullRowLoops = static_cast<uint16_t>(rows >> BARU_VREG_FP32_SHIFT);
    const uint32_t tailRows = static_cast<uint32_t>(rows) & BARU_VREG_FP32_MASK;
    const uint16_t hasRowTail = static_cast<uint16_t>(tailRows != 0U);
    const uint32_t tailCols = cols & BARU_VREG_FP32_MASK;
    const uint16_t hasColTail = static_cast<uint16_t>(tailCols != 0U);
    const uint16_t fullColLoops = static_cast<uint16_t>(colsLoops - hasColTail);
    const uint16_t pairFullColLoops = static_cast<uint16_t>(fullColLoops >> 1U);
    const uint16_t hasOddFullCol = static_cast<uint16_t>(fullColLoops & 1U);
    const uint16_t hasMixedColPair = static_cast<uint16_t>(hasOddFullCol & hasColTail);
    const uint16_t hasOddFullColOnly = static_cast<uint16_t>(hasOddFullCol - hasMixedColPair);
    const uint16_t hasColTailOnly = static_cast<uint16_t>(hasColTail - hasMixedColPair);
    // Row and column tails are tile-invariant, so neither mask needs regeneration in the compute loops.
    uint32_t remainingRows = tailRows;
    for (uint16_t loop = 0; loop < hasRowTail; ++loop) {
        rowTailMask = Reg::UpdateMask<float>(remainingRows);
    }
    uint32_t remainingCols = tailCols;
    for (uint16_t loop = 0; loop < hasColTail; ++loop) {
        colTailMask = Reg::UpdateMask<float>(remainingCols);
    }

    for (uint16_t loop = 0; loop < fullRowLoops; ++loop) {
        const uint32_t rowOffset = static_cast<uint32_t>(loop) * BARU_VREG_FP32_ELEMENTS;
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(historyMaxReg, stats + rowOffset);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(historyEllReg, stats + scalarAlign + rowOffset);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(scoreReg, stats + 2 * scalarAlign + rowOffset);
        Reg::Max<float>(currentMaxReg, historyMaxReg, scoreReg, allMask);
        Reg::ExpSub<float>(alphaReg, historyMaxReg, currentMaxReg, allMask);
        Reg::ExpSub<float>(betaReg, scoreReg, currentMaxReg, allMask);
        Reg::MulDstAdd<float>(historyEllReg, alphaReg, betaReg, allMask);
        // The two direct divisions are independent and remove the reciprocal-and-multiply dependency chain.
        Reg::Div<float>(alphaReg, alphaReg, historyEllReg, allMask);
        Reg::Div<float>(betaReg, betaReg, historyEllReg, allMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(stats + rowOffset, alphaReg, allMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(stats + scalarAlign + rowOffset, betaReg, allMask);
    }

    for (uint16_t loop = 0; loop < hasRowTail; ++loop) {
        const uint32_t rowOffset =
            (static_cast<uint32_t>(fullRowLoops) + loop) * BARU_VREG_FP32_ELEMENTS;
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(historyMaxReg, stats + rowOffset);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(historyEllReg, stats + scalarAlign + rowOffset);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(scoreReg, stats + 2 * scalarAlign + rowOffset);
        Reg::Max<float>(currentMaxReg, historyMaxReg, scoreReg, rowTailMask);
        Reg::ExpSub<float>(alphaReg, historyMaxReg, currentMaxReg, rowTailMask);
        Reg::ExpSub<float>(betaReg, scoreReg, currentMaxReg, rowTailMask);
        Reg::MulDstAdd<float>(historyEllReg, alphaReg, betaReg, rowTailMask);
        Reg::Div<float>(alphaReg, alphaReg, historyEllReg, rowTailMask);
        Reg::Div<float>(betaReg, betaReg, historyEllReg, rowTailMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(stats + rowOffset, alphaReg, rowTailMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(stats + scalarAlign + rowOffset, betaReg, rowTailMask);
    }

    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    for (uint16_t row = 0; row < rows; ++row) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(alphaReg, stats + row);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(betaReg, stats + scalarAlign + row);
        const uint32_t fp32RowOffset = static_cast<uint32_t>(row) * colsAlignFp32;
        const uint32_t bf16RowOffset = static_cast<uint32_t>(row) * colsAlignBf16;
        // Separate register sets expose two adjacent column vectors to RVEC dual issue.
        for (uint16_t loop = 0; loop < pairFullColLoops; ++loop) {
            const uint32_t vectorOffset0 =
                static_cast<uint32_t>(loop) * 2U * BARU_VREG_FP32_ELEMENTS;
            const uint32_t vectorOffset1 = vectorOffset0 + BARU_VREG_FP32_ELEMENTS;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0,
                                                            partial + fp32RowOffset + vectorOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg1,
                                                            partial + fp32RowOffset + vectorOffset1);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                numeratorReg0, interNumerator + fp32RowOffset + vectorOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                numeratorReg1, interNumerator + fp32RowOffset + vectorOffset1);
            Reg::Mul<float>(tmpReg0, partialReg0, betaReg, allMask);
            Reg::Mul<float>(tmpReg1, partialReg1, betaReg, allMask);
            Reg::MulDstAdd<float>(numeratorReg0, alphaReg, tmpReg0, allMask);
            Reg::MulDstAdd<float>(numeratorReg1, alphaReg, tmpReg1, allMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg0, numeratorReg0, allMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg1, numeratorReg1, allMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
                deltaH + bf16RowOffset + vectorOffset0, outputBf16Reg0, allMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
                deltaH + bf16RowOffset + vectorOffset1, outputBf16Reg1, allMask);
        }

        // Keep an odd full vector paired with the tail so the original two-column ILP is preserved.
        for (uint16_t loop = 0; loop < hasMixedColPair; ++loop) {
            const uint32_t vectorOffset0 =
                (static_cast<uint32_t>(pairFullColLoops) * 2U + static_cast<uint32_t>(loop) * 2U) *
                BARU_VREG_FP32_ELEMENTS;
            const uint32_t vectorOffset1 = vectorOffset0 + BARU_VREG_FP32_ELEMENTS;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0,
                                                            partial + fp32RowOffset + vectorOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg1,
                                                            partial + fp32RowOffset + vectorOffset1);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                numeratorReg0, interNumerator + fp32RowOffset + vectorOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                numeratorReg1, interNumerator + fp32RowOffset + vectorOffset1);
            Reg::Mul<float>(tmpReg0, partialReg0, betaReg, allMask);
            Reg::Mul<float>(tmpReg1, partialReg1, betaReg, colTailMask);
            Reg::MulDstAdd<float>(numeratorReg0, alphaReg, tmpReg0, allMask);
            Reg::MulDstAdd<float>(numeratorReg1, alphaReg, tmpReg1, colTailMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg0, numeratorReg0, allMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg1, numeratorReg1, colTailMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
                deltaH + bf16RowOffset + vectorOffset0, outputBf16Reg0, allMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
                deltaH + bf16RowOffset + vectorOffset1, outputBf16Reg1, colTailMask);
        }

        for (uint16_t loop = 0; loop < hasOddFullColOnly; ++loop) {
            const uint32_t vectorOffset =
                (static_cast<uint32_t>(pairFullColLoops) * 2U + loop) * BARU_VREG_FP32_ELEMENTS;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0,
                                                            partial + fp32RowOffset + vectorOffset);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                numeratorReg0, interNumerator + fp32RowOffset + vectorOffset);
            Reg::Mul<float>(tmpReg0, partialReg0, betaReg, allMask);
            Reg::MulDstAdd<float>(numeratorReg0, alphaReg, tmpReg0, allMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg0, numeratorReg0, allMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
                deltaH + bf16RowOffset + vectorOffset, outputBf16Reg0, allMask);
        }

        for (uint16_t loop = 0; loop < hasColTailOnly; ++loop) {
            const uint32_t vectorOffset =
                (static_cast<uint32_t>(fullColLoops) + loop) * BARU_VREG_FP32_ELEMENTS;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0,
                                                            partial + fp32RowOffset + vectorOffset);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                numeratorReg0, interNumerator + fp32RowOffset + vectorOffset);
            Reg::Mul<float>(tmpReg0, partialReg0, betaReg, colTailMask);
            Reg::MulDstAdd<float>(numeratorReg0, alphaReg, tmpReg0, colTailMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg0, numeratorReg0, colTailMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
                deltaH + bf16RowOffset + vectorOffset, outputBf16Reg0, colTailMask);
        }
    }
}

class BlockAttnResUpdateKernel {
public:
    __aicore__ inline __attribute__((always_inline)) BlockAttnResUpdateKernel()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    __aicore__ inline __attribute__((always_inline)) ~BlockAttnResUpdateKernel()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    __aicore__ inline __attribute__((always_inline)) void Init(
        GM_ADDR partialBlock, GM_ADDR delta, GM_ADDR interNumerator, GM_ADDR interMax,
        GM_ADDR interExpSum, GM_ADDR effectiveQuery, GM_ADDR h,
        const BlockAttnResUpdateTilingData *tilingData)
    {
        tilingData_ = tilingData;
        blockIdx_ = AscendC::GetBlockIdx();
        InitCoreRange();

        partialBlockGm_ = reinterpret_cast<__gm__ float *>(partialBlock);
        deltaGm_ = reinterpret_cast<__gm__ bfloat16_t *>(delta);
        interNumeratorGm_ = reinterpret_cast<__gm__ float *>(interNumerator);
        interMaxGm_ = reinterpret_cast<__gm__ float *>(interMax);
        interExpSumGm_ = reinterpret_cast<__gm__ float *>(interExpSum);
        effectiveQueryGm_ = reinterpret_cast<__gm__ float *>(effectiveQuery);
        hGm_ = reinterpret_cast<__gm__ bfloat16_t *>(h);
        InitUbLayout();
    }

    __aicore__ inline __attribute__((always_inline)) void Process()
    {
        if (blockIdx_ >= tilingData_->usedCoreNum || coreRows_ == 0) {
            return;
        }

        const uint32_t coreRows = coreRows_;
        const uint32_t cols = tilingData_->cols;
        const uint32_t tileRows = tilingData_->tileRows;
        const uint32_t colsAlignFp32 = tilingData_->colsAlignFp32;
        const uint32_t colsAlignBf16 = tilingData_->colsAlignBf16;
        const uint32_t scalarAlign = tilingData_->scalarAlign;
        const uint16_t colsLoops = static_cast<uint16_t>(tilingData_->colsLoops);
        const float epsilon = tilingData_->epsilon;
        const float invCols = tilingData_->invCols;
        const bool usePhase2RowBatch =
            tilingData_->phase2Mode == static_cast<uint8_t>(BlockAttnResUpdatePhase2Mode::ROW_BATCH);
        const int64_t coreRowsDim = static_cast<int64_t>(coreRows);
        const int64_t colsDim = static_cast<int64_t>(cols);
        const int64_t tileRowsDim = static_cast<int64_t>(tileRows);
        const int64_t colsAlignFp32Dim = static_cast<int64_t>(colsAlignFp32);
        const int64_t colsAlignBf16Dim = static_cast<int64_t>(colsAlignBf16);
        const int64_t scalarAlignDim = static_cast<int64_t>(scalarAlign);
        const uint64_t coreGmOffset = coreRowStart_ * static_cast<uint64_t>(cols);

        auto matrixGmLayout = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(coreRowsDim, colsDim);
        auto scalarGmLayout = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(1L, coreRowsDim);
        auto queryGmLayout = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(1L, colsDim);
        auto partialBlockGm = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(partialBlockGm_ + coreGmOffset), matrixGmLayout);
        auto deltaGm =
            Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(deltaGm_ + coreGmOffset), matrixGmLayout);
        auto interNumeratorGm = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(interNumeratorGm_ + coreGmOffset), matrixGmLayout);
        auto interMaxGm = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(interMaxGm_ + coreRowStart_), scalarGmLayout);
        auto interExpSumGm = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(interExpSumGm_ + coreRowStart_), scalarGmLayout);
        auto effectiveQueryGm =
            Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(effectiveQueryGm_), queryGmLayout);
        auto hGm = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(hGm_ + coreGmOffset), matrixGmLayout);

        auto fp32UbLayout = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(tileRowsDim, colsAlignFp32Dim);
        auto bf16UbLayout = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(tileRowsDim, colsAlignBf16Dim);
        auto statsUbLayout = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(3L, scalarAlignDim);
        auto queryUbLayout = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(1L, colsAlignFp32Dim);
        auto effectiveQueryUbMem = Te::MakeMemPtr<Te::Location::UB, float>(0);
        auto effectiveQueryUb = Te::MakeTensor(effectiveQueryUbMem, queryUbLayout);

        auto copyGmToUb = Te::MakeCopy(Te::CopyGM2UB{});
        auto copyUbToGm = Te::MakeCopy(Te::CopyUB2GM{});
        // effectiveQuery is core-invariant and occupies a disjoint UB region. Issue its single GM2UB copy once;
        // the first buffer's MTE2_V event still covers this copy and the following partial/delta copies.
        Te::Copy(copyGmToUb, effectiveQueryUb, effectiveQueryGm);

        uint32_t tileRow = 0;
        uint32_t bufferId = 0;
        while (tileRow < coreRows) {
            const uint32_t remainingRows = coreRows - tileRow;
            const uint32_t currentRows = remainingRows < tileRows ? remainingRows : tileRows;
            const int64_t tileRowDim = static_cast<int64_t>(tileRow);
            const auto matrixTileShape = Te::MakeShape(static_cast<int64_t>(currentRows), colsDim);
            const auto scalarTileShape = Te::MakeShape(1L, static_cast<int64_t>(currentRows));

            auto partialBlockGmTile =
                partialBlockGm.Slice(Te::MakeCoord(tileRowDim, 0L), matrixTileShape);
            auto deltaGmTile = deltaGm.Slice(Te::MakeCoord(tileRowDim, 0L), matrixTileShape);
            auto interNumeratorGmTile =
                interNumeratorGm.Slice(Te::MakeCoord(tileRowDim, 0L), matrixTileShape);
            auto interMaxGmTile =
                interMaxGm.Slice(Te::MakeCoord(0L, tileRowDim), scalarTileShape);
            auto interExpSumGmTile =
                interExpSumGm.Slice(Te::MakeCoord(0L, tileRowDim), scalarTileShape);
            auto hGmTile = hGm.Slice(Te::MakeCoord(tileRowDim, 0L), matrixTileShape);

            const uint64_t partialUbOffset =
                queryUbBytes_ + static_cast<uint64_t>(bufferId) * bufferUbBytes_;
            const uint64_t deltaHUbOffset = partialUbOffset + partialUbBytes_;
            const uint64_t interNumeratorUbOffset = deltaHUbOffset + deltaHUbBytes_;
            const uint64_t statsUbOffset = interNumeratorUbOffset + partialUbBytes_;
            auto partialUbMem = Te::MakeMemPtr<Te::Location::UB, float>(partialUbOffset);
            auto deltaHUbMem = Te::MakeMemPtr<Te::Location::UB, bfloat16_t>(deltaHUbOffset);
            auto interNumeratorUbMem = Te::MakeMemPtr<Te::Location::UB, float>(interNumeratorUbOffset);
            auto statsUbMem = Te::MakeMemPtr<Te::Location::UB, float>(statsUbOffset);
            auto partialUbStorage = Te::MakeTensor(partialUbMem, fp32UbLayout);
            auto deltaHUbStorage = Te::MakeTensor(deltaHUbMem, bf16UbLayout);
            auto interNumeratorUbStorage = Te::MakeTensor(interNumeratorUbMem, fp32UbLayout);
            auto statsUbStorage = Te::MakeTensor(statsUbMem, statsUbLayout);
            auto partialUb = partialUbStorage.Slice(Te::MakeCoord(0L, 0L), matrixTileShape);
            auto deltaHUb = deltaHUbStorage.Slice(Te::MakeCoord(0L, 0L), matrixTileShape);
            auto interNumeratorUb =
                interNumeratorUbStorage.Slice(Te::MakeCoord(0L, 0L), matrixTileShape);
            auto interMaxUb = statsUbStorage.Slice(Te::MakeCoord(0L, 0L), scalarTileShape);
            auto interExpSumUb = statsUbStorage.Slice(Te::MakeCoord(1L, 0L), scalarTileShape);

            // Phase 1 copy-in.
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(bufferId);
            Te::Copy(copyGmToUb, partialUb, partialBlockGmTile);
            Te::Copy(copyGmToUb, deltaHUb, deltaGmTile);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(bufferId);

            // Phase 1 compute.
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(bufferId);
            if (cols <= BARU_VREG_FP32_ELEMENTS) {
                asc_vf_call<BlockAttnResUpdatePhase1OneVLVF>(
                    partialUbMem.Get(), deltaHUbMem.Get(), effectiveQueryUbMem.Get(), statsUbMem.Get(),
                    static_cast<uint16_t>(currentRows), cols, colsAlignFp32, colsAlignBf16, scalarAlign, epsilon,
                    invCols);
            } else if (cols <= 2U * BARU_VREG_FP32_ELEMENTS) {
                asc_vf_call<BlockAttnResUpdatePhase1TwoVLVF>(
                    partialUbMem.Get(), deltaHUbMem.Get(), effectiveQueryUbMem.Get(), statsUbMem.Get(),
                    static_cast<uint16_t>(currentRows), cols, colsAlignFp32, colsAlignBf16, scalarAlign, epsilon,
                    invCols);
            } else {
                asc_vf_call<BlockAttnResUpdatePhase1VF>(
                    partialUbMem.Get(), deltaHUbMem.Get(), effectiveQueryUbMem.Get(), statsUbMem.Get(),
                    static_cast<uint16_t>(currentRows), cols, colsAlignFp32, colsAlignBf16, scalarAlign, epsilon,
                    invCols);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(bufferId);

            // Queue the partial copy-out as soon as its Phase 1 dependency is established.
            // MTE3 waits for Phase 1 while Scalar continues issuing the independent Phase 2 MTE2 copy-in.
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(bufferId);
            Te::Copy(copyUbToGm, partialBlockGmTile, partialUb);

            // Phase 2 copy-in can overlap Phase 1 compute because it writes disjoint UB regions.
            const uint32_t phase2EventId = bufferId + BARU_BUFFER_NUM;
            Te::Copy(copyGmToUb, interNumeratorUb, interNumeratorGmTile);
            Te::Copy(copyGmToUb, interMaxUb, interMaxGmTile);
            Te::Copy(copyGmToUb, interExpSumUb, interExpSumGmTile);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(phase2EventId);

            // The partial copy-out and Phase 2 only read partial, so MTE3 and V may overlap.
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(phase2EventId);
            if (usePhase2RowBatch && currentRows >= BARU_ROW_BATCH_MIN_ROWS) {
                asc_vf_call<BlockAttnResUpdatePhase2RowBatchVF>(
                    partialUbMem.Get(), deltaHUbMem.Get(), interNumeratorUbMem.Get(), statsUbMem.Get(),
                    static_cast<uint16_t>(currentRows), cols, colsAlignFp32, colsAlignBf16, scalarAlign, colsLoops);
            } else {
                asc_vf_call<BlockAttnResUpdatePhase2VF>(
                    partialUbMem.Get(), deltaHUbMem.Get(), interNumeratorUbMem.Get(), statsUbMem.Get(),
                    static_cast<uint16_t>(currentRows), cols, colsAlignFp32, colsAlignBf16, scalarAlign, colsLoops);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(phase2EventId);

            // Phase 2 copy-out closes this buffer's reuse dependency chain.
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(phase2EventId);
            Te::Copy(copyUbToGm, hGmTile, deltaHUb);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(bufferId);
            tileRow += currentRows;
            bufferId ^= 1U;
        }
    }

private:
    __aicore__ inline __attribute__((always_inline)) void InitCoreRange()
    {
        if (blockIdx_ >= tilingData_->usedCoreNum) {
            coreRowStart_ = 0;
            coreRows_ = 0;
            return;
        }
        const uint32_t rowsPerCoreBase = tilingData_->rowsPerCoreBase;
        if (blockIdx_ < tilingData_->rowsRemainder) {
            coreRows_ = rowsPerCoreBase + 1U;
            coreRowStart_ = static_cast<uint64_t>(blockIdx_) * static_cast<uint64_t>(coreRows_);
        } else {
            coreRows_ = rowsPerCoreBase;
            coreRowStart_ = static_cast<uint64_t>(tilingData_->rowsRemainder) *
                                (rowsPerCoreBase + 1U) +
                            static_cast<uint64_t>(blockIdx_ - tilingData_->rowsRemainder) *
                                rowsPerCoreBase;
        }
    }

    __aicore__ inline __attribute__((always_inline)) void InitUbLayout()
    {
        queryUbBytes_ = static_cast<uint64_t>(tilingData_->colsAlignFp32) * BARU_FP32_BYTES;
        partialUbBytes_ = static_cast<uint64_t>(tilingData_->tileRows) * queryUbBytes_;
        deltaHUbBytes_ =
            static_cast<uint64_t>(tilingData_->tileRows) * tilingData_->colsAlignBf16 * BARU_BF16_BYTES;
        const uint64_t statsBytes = static_cast<uint64_t>(3) * tilingData_->scalarAlign * BARU_FP32_BYTES;
        bufferUbBytes_ = 2 * partialUbBytes_ + deltaHUbBytes_ + statsBytes;
    }

    const BlockAttnResUpdateTilingData *tilingData_ = nullptr;
    uint32_t blockIdx_ = 0;
    uint64_t coreRowStart_ = 0;
    uint32_t coreRows_ = 0;

    __gm__ float *partialBlockGm_ = nullptr;
    __gm__ bfloat16_t *deltaGm_ = nullptr;
    __gm__ float *interNumeratorGm_ = nullptr;
    __gm__ float *interMaxGm_ = nullptr;
    __gm__ float *interExpSumGm_ = nullptr;
    __gm__ float *effectiveQueryGm_ = nullptr;
    __gm__ bfloat16_t *hGm_ = nullptr;

    uint64_t queryUbBytes_ = 0;
    uint64_t partialUbBytes_ = 0;
    uint64_t deltaHUbBytes_ = 0;
    uint64_t bufferUbBytes_ = 0;
};

} // namespace BlockAttnResUpdateOps

#endif // BLOCK_ATTN_RES_UPDATE_REGBASE_H
