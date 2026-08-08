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
constexpr uint32_t BARU_MIN_T_FOR_BATCH = 8;

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
 * One-VREG Phase 1 path. The query is invariant across T, so load and tail-clear it once.
 */
__simd_vf__ inline void BlockAttnResUpdatePhase1OneVLVF(
    __ubuf__ float *partial, __ubuf__ bfloat16_t *delta, __ubuf__ float *effectiveQuery,
    __ubuf__ float *stats, uint16_t tSize, uint32_t dSize, uint32_t dAlignFp32,
    uint32_t dAlignBf16, uint32_t statsTStride, float eps, float invD)
{
    Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg scalarMask = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
    const uint16_t hasTail = static_cast<uint16_t>(dSize < BARU_VREG_FP32_ELEMENTS);
    uint32_t remainingD = dSize;
    Reg::MaskReg dMask = Reg::UpdateMask<float>(remainingD);

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
                        static_cast<int16_t>(0), dMask);
    }

    for (uint16_t tIdx = 0; tIdx < tSize; ++tIdx) {
        const uint32_t partialTOffset = static_cast<uint32_t>(tIdx) * dAlignFp32;
        const uint32_t deltaTOffset = static_cast<uint32_t>(tIdx) * dAlignBf16;

        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg, partial + partialTOffset);
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(deltaReg, delta + deltaTOffset);
        Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg, deltaReg, dMask);
        Reg::Add<float>(partialReg, partialReg, deltaFp32Reg, dMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + partialTOffset, partialReg, dMask);

        // Form the dot product before reusing partialReg for the square product.
        Reg::Mul<float>(dotProductReg, partialReg, queryReg, allMask);
        Reg::Mul<float>(partialReg, partialReg, partialReg, allMask);
        Reg::Reduce<Reg::ReduceType::SUM>(squareSumReg, partialReg, allMask);
        Reg::Reduce<Reg::ReduceType::SUM>(dotSumReg, dotProductReg, allMask);
        Reg::Muls<float>(squareSumReg, squareSumReg, invD, scalarMask);
        Reg::Adds<float>(squareSumReg, squareSumReg, eps, scalarMask);
        Reg::Sqrt<float>(rmsReg, squareSumReg, scalarMask);
        Reg::Div<float>(scoreReg, dotSumReg, rmsReg, scalarMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
            stats + 2 * statsTStride + tIdx, scoreReg, scalarMask);
    }
}

/**
 * Two-VREG Phase 1 path. Both query vectors stay resident across the T loop; the second is tail-cleared once.
 */
__simd_vf__ inline void BlockAttnResUpdatePhase1TwoVLVF(
    __ubuf__ float *partial, __ubuf__ bfloat16_t *delta, __ubuf__ float *effectiveQuery,
    __ubuf__ float *stats, uint16_t tSize, uint32_t dSize, uint32_t dAlignFp32,
    uint32_t dAlignBf16, uint32_t statsTStride, float eps, float invD)
{
    Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg scalarMask = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
    uint32_t tailD = dSize - BARU_VREG_FP32_ELEMENTS;
    const uint16_t hasTail = static_cast<uint16_t>(tailD < BARU_VREG_FP32_ELEMENTS);
    Reg::MaskReg tailMask = Reg::UpdateMask<float>(tailD);

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

    for (uint16_t tIdx = 0; tIdx < tSize; ++tIdx) {
        const uint32_t partialTOffset = static_cast<uint32_t>(tIdx) * dAlignFp32;
        const uint32_t deltaTOffset = static_cast<uint32_t>(tIdx) * dAlignBf16;

        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0, partial + partialTOffset);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
            partialReg1, partial + partialTOffset + BARU_VREG_FP32_ELEMENTS);
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(deltaReg0, delta + deltaTOffset);
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(
            deltaReg1, delta + deltaTOffset + BARU_VREG_FP32_ELEMENTS);
        Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg0, deltaReg0, allMask);
        Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg1, deltaReg1, tailMask);
        Reg::Add<float>(partialReg0, partialReg0, deltaFp32Reg0, allMask);
        Reg::Add<float>(partialReg1, partialReg1, deltaFp32Reg1, tailMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + partialTOffset, partialReg0, allMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(
            partial + partialTOffset + BARU_VREG_FP32_ELEMENTS, partialReg1, tailMask);

        // Fold the second vector directly into the first product. queryReg1 padding was cleared above, and the
        // masked Add zeroes partialReg1 padding, so the ZEROING-only MulAddDst can safely use allMask.
        Reg::Mul<float>(dotProductReg0, partialReg0, queryReg0, allMask);
        Reg::Mul<float>(partialReg0, partialReg0, partialReg0, allMask);
        Reg::MulAddDst<float>(dotProductReg0, partialReg1, queryReg1, allMask);
        Reg::MulAddDst<float>(partialReg0, partialReg1, partialReg1, allMask);
        Reg::Reduce<Reg::ReduceType::SUM>(squareSumReg, partialReg0, allMask);
        Reg::Reduce<Reg::ReduceType::SUM>(dotSumReg, dotProductReg0, allMask);
        Reg::Muls<float>(squareSumReg, squareSumReg, invD, scalarMask);
        Reg::Adds<float>(squareSumReg, squareSumReg, eps, scalarMask);
        Reg::Sqrt<float>(rmsReg, squareSumReg, scalarMask);
        Reg::Div<float>(scoreReg, dotSumReg, rmsReg, scalarMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
            stats + 2 * statsTStride + tIdx, scoreReg, scalarMask);
    }
}

/**
 * Phase 1 updates partial in FP32 and stores one score for every T index.
 * D is the effective reduction length; aligned strides are used only for UB addressing.
 */
__simd_vf__ inline void BlockAttnResUpdatePhase1VF(__ubuf__ float *partial, __ubuf__ bfloat16_t *delta,
                                                   __ubuf__ float *effectiveQuery, __ubuf__ float *stats,
                                                   uint16_t tSize, uint32_t dSize, uint32_t dAlignFp32,
                                                   uint32_t dAlignBf16, uint32_t statsTStride,
                                                   float eps, float invD)
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

    const uint16_t fullLoops = static_cast<uint16_t>(dSize >> BARU_VREG_FP32_SHIFT);
    const uint16_t pairLoops = static_cast<uint16_t>(fullLoops >> 1U);
    // This VF is dispatched only for D > 2 * VL, so at least one complete pair is always available to peel.
    const uint16_t remainingPairLoops = static_cast<uint16_t>(pairLoops - 1U);
    const uint16_t hasOddFull = static_cast<uint16_t>(fullLoops & 1U);
    const uint32_t tailD = dSize & BARU_VREG_FP32_MASK;
    const uint16_t hasTail = static_cast<uint16_t>(tailD != 0U);
    const uint32_t oddVectorOffset =
        static_cast<uint32_t>(pairLoops) * 2U * BARU_VREG_FP32_ELEMENTS;
    const uint32_t tailVectorOffset = static_cast<uint32_t>(fullLoops) * BARU_VREG_FP32_ELEMENTS;
    // The D tail is invariant across T. UpdateMask mutates only tailRemaining; the returned MaskReg is reusable.
    uint32_t tailRemaining = tailD;
    for (uint16_t loop = 0; loop < hasTail; ++loop) {
        tailMask = Reg::UpdateMask<float>(tailRemaining);
    }

    // Query is invariant across T. Keep the first pair and the optional remainders resident; only middle pairs
    // are streamed inside the T loop. The tail query is cleared once for full-mask accumulation at every T index.
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

    for (uint16_t tIdx = 0; tIdx < tSize; ++tIdx) {
        const uint32_t partialTOffset = static_cast<uint32_t>(tIdx) * dAlignFp32;
        const uint32_t deltaTOffset = static_cast<uint32_t>(tIdx) * dAlignBf16;

        // Peel the guaranteed first pair: Mul initializes all four accumulators and removes four per-T Duplicate
        // instructions. The two independent vectors still halve each accumulation dependency chain.
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0, partial + partialTOffset);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
            partialReg1, partial + partialTOffset + BARU_VREG_FP32_ELEMENTS);
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(deltaReg0, delta + deltaTOffset);
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(
            deltaReg1, delta + deltaTOffset + BARU_VREG_FP32_ELEMENTS);
        Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg0, deltaReg0, allMask);
        Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg1, deltaReg1, allMask);
        Reg::Add<float>(partialReg0, partialReg0, deltaFp32Reg0, allMask);
        Reg::Add<float>(partialReg1, partialReg1, deltaFp32Reg1, allMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + partialTOffset, partialReg0, allMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(
            partial + partialTOffset + BARU_VREG_FP32_ELEMENTS, partialReg1, allMask);
        Reg::Mul<float>(squareAccReg0, partialReg0, partialReg0, allMask);
        Reg::Mul<float>(squareAccReg1, partialReg1, partialReg1, allMask);
        Reg::Mul<float>(dotAccReg0, partialReg0, firstQueryReg0, allMask);
        Reg::Mul<float>(dotAccReg1, partialReg1, firstQueryReg1, allMask);

        for (uint16_t loop = 0; loop < remainingPairLoops; ++loop) {
            const uint32_t vectorOffset0 =
                (static_cast<uint32_t>(loop) * 2U + 2U) * BARU_VREG_FP32_ELEMENTS;
            const uint32_t vectorOffset1 = vectorOffset0 + BARU_VREG_FP32_ELEMENTS;

            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0,
                                                            partial + partialTOffset + vectorOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg1,
                                                            partial + partialTOffset + vectorOffset1);
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(
                deltaReg0, delta + deltaTOffset + vectorOffset0);
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(
                deltaReg1, delta + deltaTOffset + vectorOffset1);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(loopQueryReg0, effectiveQuery + vectorOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(loopQueryReg1, effectiveQuery + vectorOffset1);

            Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg0, deltaReg0, allMask);
            Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg1, deltaReg1, allMask);
            Reg::Add<float>(partialReg0, partialReg0, deltaFp32Reg0, allMask);
            Reg::Add<float>(partialReg1, partialReg1, deltaFp32Reg1, allMask);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + partialTOffset + vectorOffset0,
                                                              partialReg0, allMask);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + partialTOffset + vectorOffset1,
                                                              partialReg1, allMask);

            Reg::MulAddDst<float>(squareAccReg0, partialReg0, partialReg0, allMask);
            Reg::MulAddDst<float>(squareAccReg1, partialReg1, partialReg1, allMask);
            Reg::MulAddDst<float>(dotAccReg0, partialReg0, loopQueryReg0, allMask);
            Reg::MulAddDst<float>(dotAccReg1, partialReg1, loopQueryReg1, allMask);
        }

        for (uint16_t loop = 0; loop < hasOddFull; ++loop) {
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0,
                                                            partial + partialTOffset + oddVectorOffset);
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(
                deltaReg0, delta + deltaTOffset + oddVectorOffset);
            Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg0, deltaReg0, allMask);
            Reg::Add<float>(partialReg0, partialReg0, deltaFp32Reg0, allMask);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + partialTOffset + oddVectorOffset,
                                                              partialReg0, allMask);
            Reg::MulAddDst<float>(squareAccReg0, partialReg0, partialReg0, allMask);
            Reg::MulAddDst<float>(dotAccReg0, partialReg0, oddQueryReg, allMask);
        }

        for (uint16_t loop = 0; loop < hasTail; ++loop) {
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0,
                                                            partial + partialTOffset + tailVectorOffset);
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(
                deltaReg0, delta + deltaTOffset + tailVectorOffset);
            Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg0, deltaReg0, tailMask);
            Reg::Add<float>(partialReg0, partialReg0, deltaFp32Reg0, tailMask);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + partialTOffset + tailVectorOffset,
                                                              partialReg0, tailMask);

            // MulAddDst only supports ZEROING mode. partial padding was cleared by Add and query padding was
            // cleared before the T loop, so inactive lanes cannot erase the resident destination accumulator.
            Reg::MulAddDst<float>(squareAccReg1, partialReg0, partialReg0, allMask);
            Reg::MulAddDst<float>(dotAccReg1, partialReg0, tailQueryReg, allMask);
        }

        Reg::Add<float>(squareAccReg0, squareAccReg0, squareAccReg1, allMask);
        Reg::Add<float>(dotAccReg0, dotAccReg0, dotAccReg1, allMask);
        Reg::Reduce<Reg::ReduceType::SUM>(squareSumReg, squareAccReg0, allMask);
        Reg::Reduce<Reg::ReduceType::SUM>(dotSumReg, dotAccReg0, allMask);
        Reg::Muls<float>(squareSumReg, squareSumReg, invD, scalarMask);
        Reg::Adds<float>(squareSumReg, squareSumReg, eps, scalarMask);
        Reg::Sqrt<float>(rmsReg, squareSumReg, scalarMask);
        Reg::Div<float>(scoreReg, dotSumReg, rmsReg, scalarMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(stats + 2 * statsTStride + tIdx, scoreReg,
                                                                       scalarMask);
    }
}

/**
 * Optional Phase 1 binary-reduction candidate. The current kernel does not dispatch to this VF.
 * binaryReduceTmp must point to 512 bytes of 256-byte-aligned UB scratch: one FP32 VREG for
 * square partials followed by one FP32 VREG for dot partials. With D <= 8192, folding two
 * input VREGs into each partial produces at most 64 partials per statistic.
 */
__simd_vf__ inline void BlockAttnResUpdatePhase1BinaryReduceVF(
    __ubuf__ float *partial, __ubuf__ bfloat16_t *delta, __ubuf__ float *effectiveQuery,
    __ubuf__ float *stats, __ubuf__ float *binaryReduceTmp, uint16_t tSize, uint32_t dSize,
    uint32_t dAlignFp32, uint32_t dAlignBf16, uint32_t statsTStride, float eps, float invD)
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

    const uint16_t fullLoops = static_cast<uint16_t>(dSize >> BARU_VREG_FP32_SHIFT);
    const uint16_t pairLoops = static_cast<uint16_t>(fullLoops >> 1U);
    const uint16_t hasOddFull = static_cast<uint16_t>(fullLoops & 1U);
    const uint32_t tailD = dSize & BARU_VREG_FP32_MASK;
    const uint16_t hasTail = static_cast<uint16_t>(tailD != 0U);
    const uint16_t hasRemainder = static_cast<uint16_t>(hasOddFull | hasTail);
    const uint32_t partialCount = static_cast<uint32_t>(pairLoops + hasRemainder);
    // Both mask lengths are invariant across T, so materialize them before the T loop.
    uint32_t tailRemaining = tailD;
    for (uint16_t loop = 0; loop < hasTail; ++loop) {
        tailMask = Reg::UpdateMask<float>(tailRemaining);
    }
    uint32_t partialRemaining = partialCount;
    partialMask = Reg::UpdateMask<float>(partialRemaining);

    for (uint16_t tIdx = 0; tIdx < tSize; ++tIdx) {
        const uint32_t partialTOffset = static_cast<uint32_t>(tIdx) * dAlignFp32;
        const uint32_t deltaTOffset = static_cast<uint32_t>(tIdx) * dAlignBf16;

        // Each pair is independent: reduce it immediately to remove the cross-pair accumulation chain.
        for (uint16_t loop = 0; loop < pairLoops; ++loop) {
            const uint32_t vectorOffset0 =
                static_cast<uint32_t>(loop) * 2U * BARU_VREG_FP32_ELEMENTS;
            const uint32_t vectorOffset1 = vectorOffset0 + BARU_VREG_FP32_ELEMENTS;

            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0,
                                                            partial + partialTOffset + vectorOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg1,
                                                            partial + partialTOffset + vectorOffset1);
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(
                deltaReg0, delta + deltaTOffset + vectorOffset0);
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(
                deltaReg1, delta + deltaTOffset + vectorOffset1);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(queryReg0, effectiveQuery + vectorOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(queryReg1, effectiveQuery + vectorOffset1);

            Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg0, deltaReg0, allMask);
            Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg1, deltaReg1, allMask);
            Reg::Add<float>(partialReg0, partialReg0, deltaFp32Reg0, allMask);
            Reg::Add<float>(partialReg1, partialReg1, deltaFp32Reg1, allMask);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + partialTOffset + vectorOffset0,
                                                              partialReg0, allMask);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + partialTOffset + vectorOffset1,
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
                                                                partial + partialTOffset + vectorOffset);
                Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(
                    deltaReg0, delta + deltaTOffset + vectorOffset);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(queryReg0, effectiveQuery + vectorOffset);
                Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg0, deltaReg0, allMask);
                Reg::Add<float>(partialReg0, partialReg0, deltaFp32Reg0, allMask);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + partialTOffset + vectorOffset,
                                                                  partialReg0, allMask);
                Reg::MulAddDst<float>(squarePairReg, partialReg0, partialReg0, allMask);
                Reg::MulAddDst<float>(dotPairReg, partialReg0, queryReg0, allMask);
            }

            for (uint16_t loop = 0; loop < hasTail; ++loop) {
                const uint32_t vectorOffset =
                    (static_cast<uint32_t>(fullLoops) + loop) * BARU_VREG_FP32_ELEMENTS;
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0,
                                                                partial + partialTOffset + vectorOffset);
                Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(
                    deltaReg0, delta + deltaTOffset + vectorOffset);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(queryReg0, effectiveQuery + vectorOffset);
                Reg::Cast<float, bfloat16_t, BARU_CAST_BF16_TO_FP32>(deltaFp32Reg0, deltaReg0, tailMask);
                Reg::Add<float>(partialReg0, partialReg0, deltaFp32Reg0, tailMask);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(partial + partialTOffset + vectorOffset,
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
        Reg::Muls<float>(squareSumReg, squareSumReg, invD, scalarMask);
        Reg::Adds<float>(squareSumReg, squareSumReg, eps, scalarMask);
        Reg::Sqrt<float>(rmsReg, squareSumReg, scalarMask);
        Reg::Div<float>(scoreReg, dotSumReg, rmsReg, scalarMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(stats + 2 * statsTStride + tIdx, scoreReg,
                                                                       scalarMask);
    }
}

/**
 * Phase 2 updates the online-softmax coefficients and forms BF16 h.
 * Alpha and beta are normalized once per T index so that each D loop needs only Mul + MulDstAdd.
 * The two VF calls are ordered on V; delta has been fully consumed before this phase reuses it as h.
 */
__simd_vf__ inline void BlockAttnResUpdatePhase2TokenVF(
    __ubuf__ float *partial, __ubuf__ bfloat16_t *deltaH, __ubuf__ float *interNumerator,
    __ubuf__ float *stats, uint16_t tSize, uint32_t dSize, uint32_t dAlignFp32,
    uint32_t dAlignBf16, uint32_t statsTStride, uint16_t dVregNum)
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

    const uint32_t tailD = dSize & BARU_VREG_FP32_MASK;
    const uint16_t hasTail = static_cast<uint16_t>(tailD != 0U);
    const uint16_t fullLoops = static_cast<uint16_t>(dVregNum - hasTail);
    const uint16_t pairFullLoops = static_cast<uint16_t>(fullLoops >> 1U);
    const uint16_t hasOddFull = static_cast<uint16_t>(fullLoops & 1U);
    const uint16_t hasMixedPair = static_cast<uint16_t>(hasOddFull & hasTail);
    const uint16_t hasOddFullOnly = static_cast<uint16_t>(hasOddFull - hasMixedPair);
    const uint16_t hasTailOnly = static_cast<uint16_t>(hasTail - hasMixedPair);
    // Full vectors reuse allMask; only the true tail needs one dynamic mask for the whole VF call.
    uint32_t tailRemaining = tailD;
    for (uint16_t loop = 0; loop < hasTail; ++loop) {
        tailMask = Reg::UpdateMask<float>(tailRemaining);
    }

    for (uint16_t tIdx = 0; tIdx < tSize; ++tIdx) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(historyMaxReg, stats + tIdx);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(historyEllReg, stats + statsTStride + tIdx);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(scoreReg, stats + 2 * statsTStride + tIdx);
        Reg::Max<float>(currentMaxReg, historyMaxReg, scoreReg, allMask);
        Reg::ExpSub<float>(alphaReg, historyMaxReg, currentMaxReg, allMask);
        Reg::ExpSub<float>(betaReg, scoreReg, currentMaxReg, allMask);
        Reg::MulDstAdd<float>(historyEllReg, alphaReg, betaReg, allMask);
        // The two direct divisions are independent and remove the reciprocal-and-multiply dependency chain.
        Reg::Div<float>(alphaReg, alphaReg, historyEllReg, allMask);
        Reg::Div<float>(betaReg, betaReg, historyEllReg, allMask);
        const uint32_t fp32TOffset = static_cast<uint32_t>(tIdx) * dAlignFp32;
        const uint32_t bf16TOffset = static_cast<uint32_t>(tIdx) * dAlignBf16;
        // Separate register sets expose two adjacent D vectors to RVEC dual issue.
        for (uint16_t loop = 0; loop < pairFullLoops; ++loop) {
            const uint32_t vectorOffset0 =
                static_cast<uint32_t>(loop) * 2U * BARU_VREG_FP32_ELEMENTS;
            const uint32_t vectorOffset1 = vectorOffset0 + BARU_VREG_FP32_ELEMENTS;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0,
                                                            partial + fp32TOffset + vectorOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg1,
                                                            partial + fp32TOffset + vectorOffset1);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                numeratorReg0, interNumerator + fp32TOffset + vectorOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                numeratorReg1, interNumerator + fp32TOffset + vectorOffset1);
            Reg::Mul<float>(tmpReg0, partialReg0, betaReg, allMask);
            Reg::Mul<float>(tmpReg1, partialReg1, betaReg, allMask);
            Reg::MulDstAdd<float>(numeratorReg0, alphaReg, tmpReg0, allMask);
            Reg::MulDstAdd<float>(numeratorReg1, alphaReg, tmpReg1, allMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg0, numeratorReg0, allMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg1, numeratorReg1, allMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
                deltaH + bf16TOffset + vectorOffset0, outputBf16Reg0, allMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
                deltaH + bf16TOffset + vectorOffset1, outputBf16Reg1, allMask);
        }

        // Keep an odd full vector paired with the tail so the original two-vector ILP is preserved.
        for (uint16_t loop = 0; loop < hasMixedPair; ++loop) {
            const uint32_t vectorOffset0 =
                (static_cast<uint32_t>(pairFullLoops) * 2U + static_cast<uint32_t>(loop) * 2U) *
                BARU_VREG_FP32_ELEMENTS;
            const uint32_t vectorOffset1 = vectorOffset0 + BARU_VREG_FP32_ELEMENTS;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0,
                                                            partial + fp32TOffset + vectorOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg1,
                                                            partial + fp32TOffset + vectorOffset1);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                numeratorReg0, interNumerator + fp32TOffset + vectorOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                numeratorReg1, interNumerator + fp32TOffset + vectorOffset1);
            Reg::Mul<float>(tmpReg0, partialReg0, betaReg, allMask);
            Reg::Mul<float>(tmpReg1, partialReg1, betaReg, tailMask);
            Reg::MulDstAdd<float>(numeratorReg0, alphaReg, tmpReg0, allMask);
            Reg::MulDstAdd<float>(numeratorReg1, alphaReg, tmpReg1, tailMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg0, numeratorReg0, allMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg1, numeratorReg1, tailMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
                deltaH + bf16TOffset + vectorOffset0, outputBf16Reg0, allMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
                deltaH + bf16TOffset + vectorOffset1, outputBf16Reg1, tailMask);
        }

        for (uint16_t loop = 0; loop < hasOddFullOnly; ++loop) {
            const uint32_t vectorOffset =
                (static_cast<uint32_t>(pairFullLoops) * 2U + loop) * BARU_VREG_FP32_ELEMENTS;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0,
                                                            partial + fp32TOffset + vectorOffset);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                numeratorReg0, interNumerator + fp32TOffset + vectorOffset);
            Reg::Mul<float>(tmpReg0, partialReg0, betaReg, allMask);
            Reg::MulDstAdd<float>(numeratorReg0, alphaReg, tmpReg0, allMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg0, numeratorReg0, allMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
                deltaH + bf16TOffset + vectorOffset, outputBf16Reg0, allMask);
        }

        for (uint16_t loop = 0; loop < hasTailOnly; ++loop) {
            const uint32_t vectorOffset =
                (static_cast<uint32_t>(fullLoops) + loop) * BARU_VREG_FP32_ELEMENTS;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0,
                                                            partial + fp32TOffset + vectorOffset);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                numeratorReg0, interNumerator + fp32TOffset + vectorOffset);
            Reg::Mul<float>(tmpReg0, partialReg0, betaReg, tailMask);
            Reg::MulDstAdd<float>(numeratorReg0, alphaReg, tmpReg0, tailMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg0, numeratorReg0, tailMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
                deltaH + bf16TOffset + vectorOffset, outputBf16Reg0, tailMask);
        }
    }
}

/**
 * Small-D Phase 2 vectorizes the coefficient calculation across T. Normalized alpha and beta overwrite
 * the consumed history-max and history-ell regions in stats, then are broadcast per T index to form BF16 h.
 * statsTStride is VREG-aligned for this path, so the masked final coefficient load/store stays in its region.
 */
__simd_vf__ inline void BlockAttnResUpdatePhase2BatchVF(
    __ubuf__ float *partial, __ubuf__ bfloat16_t *deltaH, __ubuf__ float *interNumerator,
    __ubuf__ float *stats, uint16_t tSize, uint32_t dSize, uint32_t dAlignFp32,
    uint32_t dAlignBf16, uint32_t statsTStride, uint16_t dVregNum)
{
    Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg tTailMask;
    Reg::MaskReg dTailMask;
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

    const uint16_t fullTLoops = static_cast<uint16_t>(tSize >> BARU_VREG_FP32_SHIFT);
    const uint32_t tailT = static_cast<uint32_t>(tSize) & BARU_VREG_FP32_MASK;
    const uint16_t hasTTail = static_cast<uint16_t>(tailT != 0U);
    const uint32_t tailD = dSize & BARU_VREG_FP32_MASK;
    const uint16_t hasDTail = static_cast<uint16_t>(tailD != 0U);
    const uint16_t fullDLoops = static_cast<uint16_t>(dVregNum - hasDTail);
    const uint16_t pairFullDLoops = static_cast<uint16_t>(fullDLoops >> 1U);
    const uint16_t hasOddFullD = static_cast<uint16_t>(fullDLoops & 1U);
    const uint16_t hasMixedDPair = static_cast<uint16_t>(hasOddFullD & hasDTail);
    const uint16_t hasOddFullDOnly = static_cast<uint16_t>(hasOddFullD - hasMixedDPair);
    const uint16_t hasDTailOnly = static_cast<uint16_t>(hasDTail - hasMixedDPair);
    // T and D tails are tile-invariant, so neither mask needs regeneration in the compute loops.
    uint32_t remainingT = tailT;
    for (uint16_t loop = 0; loop < hasTTail; ++loop) {
        tTailMask = Reg::UpdateMask<float>(remainingT);
    }
    uint32_t remainingD = tailD;
    for (uint16_t loop = 0; loop < hasDTail; ++loop) {
        dTailMask = Reg::UpdateMask<float>(remainingD);
    }

    for (uint16_t loop = 0; loop < fullTLoops; ++loop) {
        const uint32_t tOffset = static_cast<uint32_t>(loop) * BARU_VREG_FP32_ELEMENTS;
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(historyMaxReg, stats + tOffset);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(historyEllReg, stats + statsTStride + tOffset);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(scoreReg, stats + 2 * statsTStride + tOffset);
        Reg::Max<float>(currentMaxReg, historyMaxReg, scoreReg, allMask);
        Reg::ExpSub<float>(alphaReg, historyMaxReg, currentMaxReg, allMask);
        Reg::ExpSub<float>(betaReg, scoreReg, currentMaxReg, allMask);
        Reg::MulDstAdd<float>(historyEllReg, alphaReg, betaReg, allMask);
        // The two direct divisions are independent and remove the reciprocal-and-multiply dependency chain.
        Reg::Div<float>(alphaReg, alphaReg, historyEllReg, allMask);
        Reg::Div<float>(betaReg, betaReg, historyEllReg, allMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(stats + tOffset, alphaReg, allMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(stats + statsTStride + tOffset, betaReg, allMask);
    }

    for (uint16_t loop = 0; loop < hasTTail; ++loop) {
        const uint32_t tOffset =
            (static_cast<uint32_t>(fullTLoops) + loop) * BARU_VREG_FP32_ELEMENTS;
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(historyMaxReg, stats + tOffset);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(historyEllReg, stats + statsTStride + tOffset);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(scoreReg, stats + 2 * statsTStride + tOffset);
        Reg::Max<float>(currentMaxReg, historyMaxReg, scoreReg, tTailMask);
        Reg::ExpSub<float>(alphaReg, historyMaxReg, currentMaxReg, tTailMask);
        Reg::ExpSub<float>(betaReg, scoreReg, currentMaxReg, tTailMask);
        Reg::MulDstAdd<float>(historyEllReg, alphaReg, betaReg, tTailMask);
        Reg::Div<float>(alphaReg, alphaReg, historyEllReg, tTailMask);
        Reg::Div<float>(betaReg, betaReg, historyEllReg, tTailMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(stats + tOffset, alphaReg, tTailMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(stats + statsTStride + tOffset, betaReg, tTailMask);
    }

    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    for (uint16_t tIdx = 0; tIdx < tSize; ++tIdx) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(alphaReg, stats + tIdx);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(betaReg, stats + statsTStride + tIdx);
        const uint32_t fp32TOffset = static_cast<uint32_t>(tIdx) * dAlignFp32;
        const uint32_t bf16TOffset = static_cast<uint32_t>(tIdx) * dAlignBf16;
        // Separate register sets expose two adjacent D vectors to RVEC dual issue.
        for (uint16_t loop = 0; loop < pairFullDLoops; ++loop) {
            const uint32_t vectorOffset0 =
                static_cast<uint32_t>(loop) * 2U * BARU_VREG_FP32_ELEMENTS;
            const uint32_t vectorOffset1 = vectorOffset0 + BARU_VREG_FP32_ELEMENTS;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0,
                                                            partial + fp32TOffset + vectorOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg1,
                                                            partial + fp32TOffset + vectorOffset1);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                numeratorReg0, interNumerator + fp32TOffset + vectorOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                numeratorReg1, interNumerator + fp32TOffset + vectorOffset1);
            Reg::Mul<float>(tmpReg0, partialReg0, betaReg, allMask);
            Reg::Mul<float>(tmpReg1, partialReg1, betaReg, allMask);
            Reg::MulDstAdd<float>(numeratorReg0, alphaReg, tmpReg0, allMask);
            Reg::MulDstAdd<float>(numeratorReg1, alphaReg, tmpReg1, allMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg0, numeratorReg0, allMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg1, numeratorReg1, allMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
                deltaH + bf16TOffset + vectorOffset0, outputBf16Reg0, allMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
                deltaH + bf16TOffset + vectorOffset1, outputBf16Reg1, allMask);
        }

        // Keep an odd full vector paired with the tail so the original two-vector ILP is preserved.
        for (uint16_t loop = 0; loop < hasMixedDPair; ++loop) {
            const uint32_t vectorOffset0 =
                (static_cast<uint32_t>(pairFullDLoops) * 2U + static_cast<uint32_t>(loop) * 2U) *
                BARU_VREG_FP32_ELEMENTS;
            const uint32_t vectorOffset1 = vectorOffset0 + BARU_VREG_FP32_ELEMENTS;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0,
                                                            partial + fp32TOffset + vectorOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg1,
                                                            partial + fp32TOffset + vectorOffset1);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                numeratorReg0, interNumerator + fp32TOffset + vectorOffset0);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                numeratorReg1, interNumerator + fp32TOffset + vectorOffset1);
            Reg::Mul<float>(tmpReg0, partialReg0, betaReg, allMask);
            Reg::Mul<float>(tmpReg1, partialReg1, betaReg, dTailMask);
            Reg::MulDstAdd<float>(numeratorReg0, alphaReg, tmpReg0, allMask);
            Reg::MulDstAdd<float>(numeratorReg1, alphaReg, tmpReg1, dTailMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg0, numeratorReg0, allMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg1, numeratorReg1, dTailMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
                deltaH + bf16TOffset + vectorOffset0, outputBf16Reg0, allMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
                deltaH + bf16TOffset + vectorOffset1, outputBf16Reg1, dTailMask);
        }

        for (uint16_t loop = 0; loop < hasOddFullDOnly; ++loop) {
            const uint32_t vectorOffset =
                (static_cast<uint32_t>(pairFullDLoops) * 2U + loop) * BARU_VREG_FP32_ELEMENTS;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0,
                                                            partial + fp32TOffset + vectorOffset);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                numeratorReg0, interNumerator + fp32TOffset + vectorOffset);
            Reg::Mul<float>(tmpReg0, partialReg0, betaReg, allMask);
            Reg::MulDstAdd<float>(numeratorReg0, alphaReg, tmpReg0, allMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg0, numeratorReg0, allMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
                deltaH + bf16TOffset + vectorOffset, outputBf16Reg0, allMask);
        }

        for (uint16_t loop = 0; loop < hasDTailOnly; ++loop) {
            const uint32_t vectorOffset =
                (static_cast<uint32_t>(fullDLoops) + loop) * BARU_VREG_FP32_ELEMENTS;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partialReg0,
                                                            partial + fp32TOffset + vectorOffset);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                numeratorReg0, interNumerator + fp32TOffset + vectorOffset);
            Reg::Mul<float>(tmpReg0, partialReg0, betaReg, dTailMask);
            Reg::MulDstAdd<float>(numeratorReg0, alphaReg, tmpReg0, dTailMask);
            Reg::Cast<bfloat16_t, float, BARU_CAST_FP32_TO_BF16>(outputBf16Reg0, numeratorReg0, dTailMask);
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(
                deltaH + bf16TOffset + vectorOffset, outputBf16Reg0, dTailMask);
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
        if (blockIdx_ >= tilingData_->usedCoreNum || coreTSize_ == 0) {
            return;
        }

        const uint32_t coreTSize = coreTSize_;
        const uint32_t dSize = tilingData_->dSize;
        const uint32_t tileT = tilingData_->tileT;
        const uint32_t dAlignFp32 = tilingData_->dAlignFp32;
        const uint32_t dAlignBf16 = tilingData_->dAlignBf16;
        const uint32_t statsTStride = tilingData_->statsTStride;
        const uint16_t dVregNum = static_cast<uint16_t>(tilingData_->dVregNum);
        const float eps = tilingData_->eps;
        const float invD = 1.0F / static_cast<float>(dSize);
        const bool usePhase2Batch =
            tilingData_->phase2Mode == static_cast<uint8_t>(BlockAttnResUpdatePhase2Mode::BATCH);
        const int64_t coreTDim = static_cast<int64_t>(coreTSize);
        const int64_t dDim = static_cast<int64_t>(dSize);
        const int64_t tileTDim = static_cast<int64_t>(tileT);
        const int64_t dAlignFp32Dim = static_cast<int64_t>(dAlignFp32);
        const int64_t dAlignBf16Dim = static_cast<int64_t>(dAlignBf16);
        const int64_t statsTStrideDim = static_cast<int64_t>(statsTStride);
        const int64_t coreGmOffset = coreTStart_ * static_cast<int64_t>(dSize);

        auto matrixGmLayout = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(coreTDim, dDim);
        auto statsGmLayout = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(1L, coreTDim);
        auto queryGmLayout = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(1L, dDim);
        auto partialBlockGm = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(partialBlockGm_ + coreGmOffset), matrixGmLayout);
        auto deltaGm =
            Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(deltaGm_ + coreGmOffset), matrixGmLayout);
        auto interNumeratorGm = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(interNumeratorGm_ + coreGmOffset), matrixGmLayout);
        auto interMaxGm = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(interMaxGm_ + coreTStart_), statsGmLayout);
        auto interExpSumGm = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::GM>(interExpSumGm_ + coreTStart_), statsGmLayout);
        auto effectiveQueryGm =
            Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(effectiveQueryGm_), queryGmLayout);
        auto hGm = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(hGm_ + coreGmOffset), matrixGmLayout);

        auto fp32UbLayout = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(tileTDim, dAlignFp32Dim);
        auto bf16UbLayout = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(tileTDim, dAlignBf16Dim);
        auto statsUbLayout = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(3L, statsTStrideDim);
        auto queryUbLayout = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(1L, dAlignFp32Dim);
        auto effectiveQueryUbMem = Te::MakeMemPtr<Te::Location::UB, float>(0);
        auto effectiveQueryUb = Te::MakeTensor(effectiveQueryUbMem, queryUbLayout);

        auto copyGmToUb = Te::MakeCopy(Te::CopyGM2UB{});
        auto copyUbToGm = Te::MakeCopy(Te::CopyUB2GM{});
        // effectiveQuery is core-invariant and occupies a disjoint UB region. Issue its single GM2UB copy once;
        // the first buffer's MTE2_V event still covers this copy and the following partial/delta copies.
        Te::Copy(copyGmToUb, effectiveQueryUb, effectiveQueryGm);

        uint32_t tileTStart = 0;
        uint32_t bufferId = 0;
        while (tileTStart < coreTSize) {
            const uint32_t remainingT = coreTSize - tileTStart;
            const uint32_t currentTSize = remainingT < tileT ? remainingT : tileT;
            const int64_t tileTStartDim = static_cast<int64_t>(tileTStart);
            const auto matrixTileShape = Te::MakeShape(static_cast<int64_t>(currentTSize), dDim);
            const auto statsTileShape = Te::MakeShape(1L, static_cast<int64_t>(currentTSize));

            auto partialBlockGmTile =
                partialBlockGm.Slice(Te::MakeCoord(tileTStartDim, 0L), matrixTileShape);
            auto deltaGmTile = deltaGm.Slice(Te::MakeCoord(tileTStartDim, 0L), matrixTileShape);
            auto interNumeratorGmTile =
                interNumeratorGm.Slice(Te::MakeCoord(tileTStartDim, 0L), matrixTileShape);
            auto interMaxGmTile =
                interMaxGm.Slice(Te::MakeCoord(0L, tileTStartDim), statsTileShape);
            auto interExpSumGmTile =
                interExpSumGm.Slice(Te::MakeCoord(0L, tileTStartDim), statsTileShape);
            auto hGmTile = hGm.Slice(Te::MakeCoord(tileTStartDim, 0L), matrixTileShape);

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
            auto interMaxUb = statsUbStorage.Slice(Te::MakeCoord(0L, 0L), statsTileShape);
            auto interExpSumUb = statsUbStorage.Slice(Te::MakeCoord(1L, 0L), statsTileShape);

            // Phase 1 copy-in.
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(bufferId);
            Te::Copy(copyGmToUb, partialUb, partialBlockGmTile);
            Te::Copy(copyGmToUb, deltaHUb, deltaGmTile);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(bufferId);

            // Phase 1 compute.
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(bufferId);
            if (dSize <= BARU_VREG_FP32_ELEMENTS) {
                asc_vf_call<BlockAttnResUpdatePhase1OneVLVF>(
                    partialUbMem.Get(), deltaHUbMem.Get(), effectiveQueryUbMem.Get(), statsUbMem.Get(),
                    static_cast<uint16_t>(currentTSize), dSize, dAlignFp32,
                    dAlignBf16, statsTStride, eps, invD);
            } else if (dSize <= 2U * BARU_VREG_FP32_ELEMENTS) {
                asc_vf_call<BlockAttnResUpdatePhase1TwoVLVF>(
                    partialUbMem.Get(), deltaHUbMem.Get(), effectiveQueryUbMem.Get(), statsUbMem.Get(),
                    static_cast<uint16_t>(currentTSize), dSize, dAlignFp32,
                    dAlignBf16, statsTStride, eps, invD);
            } else {
                asc_vf_call<BlockAttnResUpdatePhase1VF>(
                    partialUbMem.Get(), deltaHUbMem.Get(), effectiveQueryUbMem.Get(), statsUbMem.Get(),
                    static_cast<uint16_t>(currentTSize), dSize, dAlignFp32,
                    dAlignBf16, statsTStride, eps, invD);
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
            if (usePhase2Batch && currentTSize >= BARU_MIN_T_FOR_BATCH) {
                asc_vf_call<BlockAttnResUpdatePhase2BatchVF>(
                    partialUbMem.Get(), deltaHUbMem.Get(), interNumeratorUbMem.Get(), statsUbMem.Get(),
                    static_cast<uint16_t>(currentTSize), dSize, dAlignFp32,
                    dAlignBf16, statsTStride, dVregNum);
            } else {
                asc_vf_call<BlockAttnResUpdatePhase2TokenVF>(
                    partialUbMem.Get(), deltaHUbMem.Get(), interNumeratorUbMem.Get(), statsUbMem.Get(),
                    static_cast<uint16_t>(currentTSize), dSize, dAlignFp32,
                    dAlignBf16, statsTStride, dVregNum);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(phase2EventId);

            // Phase 2 copy-out closes this buffer's reuse dependency chain.
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(phase2EventId);
            Te::Copy(copyUbToGm, hGmTile, deltaHUb);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(bufferId);
            tileTStart += currentTSize;
            bufferId ^= 1U;
        }
    }

private:
    __aicore__ inline __attribute__((always_inline)) void InitCoreRange()
    {
        if (blockIdx_ >= tilingData_->usedCoreNum) {
            coreTStart_ = 0;
            coreTSize_ = 0;
            return;
        }
        const uint32_t baseTPerCore = tilingData_->baseTPerCore;
        if (blockIdx_ < tilingData_->extraTCoreNum) {
            coreTSize_ = baseTPerCore + 1U;
            coreTStart_ = static_cast<int64_t>(blockIdx_) * static_cast<int64_t>(coreTSize_);
        } else {
            coreTSize_ = baseTPerCore;
            coreTStart_ = static_cast<int64_t>(tilingData_->extraTCoreNum) *
                              (static_cast<int64_t>(baseTPerCore) + 1) +
                          static_cast<int64_t>(blockIdx_ - tilingData_->extraTCoreNum) *
                              static_cast<int64_t>(baseTPerCore);
        }
    }

    __aicore__ inline __attribute__((always_inline)) void InitUbLayout()
    {
        queryUbBytes_ = static_cast<uint64_t>(tilingData_->dAlignFp32) * BARU_FP32_BYTES;
        partialUbBytes_ = static_cast<uint64_t>(tilingData_->tileT) * queryUbBytes_;
        deltaHUbBytes_ =
            static_cast<uint64_t>(tilingData_->tileT) * tilingData_->dAlignBf16 * BARU_BF16_BYTES;
        const uint64_t statsBytes = static_cast<uint64_t>(3) * tilingData_->statsTStride * BARU_FP32_BYTES;
        bufferUbBytes_ = 2 * partialUbBytes_ + deltaHUbBytes_ + statsBytes;
    }

    const BlockAttnResUpdateTilingData *tilingData_ = nullptr;
    uint32_t blockIdx_ = 0;
    int64_t coreTStart_ = 0;
    uint32_t coreTSize_ = 0;

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
