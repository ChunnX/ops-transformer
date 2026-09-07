/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHUNK_GATED_DELTA_RULE_STAGE2_VF_H
#define CHUNK_GATED_DELTA_RULE_STAGE2_VF_H

#include "kernel_tensor.h"

namespace ChunkGatedDeltaRule {
using namespace AscendC;
using namespace Reg;

/*
 * FP32 状态缩放：scale = exp(gCumLast)，state = state * scale。
 * exp 结果保留在寄存器中，整条计算链只读写一次 state UB。
 */
__simd_vf__ inline void ScaleFp32StateByExpVF(__ubuf__ float *stateAddr, float gCumLast, uint32_t count)
{
    constexpr uint32_t elementsPerRepeat = VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<float> scaleReg;
    RegTensor<float> stateReg;
    MaskReg allMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg mask;

    Duplicate(scaleReg, gCumLast);
    Exp<float, MaskMergeMode::ZEROING>(scaleReg, scaleReg, allMask);

    uint32_t remaining = count;
    uint16_t repeatTimes = static_cast<uint16_t>((count + elementsPerRepeat - 1) / elementsPerRepeat);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = UpdateMask<float>(remaining);
        uint32_t offset = static_cast<uint32_t>(i) * elementsPerRepeat;
        LoadAlign(stateReg, stateAddr + offset);
        Mul(stateReg, stateReg, scaleReg, mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(stateAddr + offset, stateReg, mask);
    }
}

/*
 * BF16 gated 状态缩放：scale = exp(gCumLast)，dst = BF16(FP32(src) * scale)。
 * Cast、Exp、Mul、Cast 在同一个 VF 中完成，FP32 中间结果不再写回临时 UB。
 */
__simd_vf__ inline void ScaleBf16StateByExpVF(__ubuf__ bfloat16_t *dstAddr, __ubuf__ bfloat16_t *srcAddr,
                                              float gCumLast, uint32_t count)
{
    constexpr uint32_t elementsPerRepeat = VECTOR_REG_WIDTH / sizeof(float);
    static constexpr CastTrait castBf16ToFp32 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                 RoundMode::UNKNOWN};
    static constexpr CastTrait castFp32ToBf16 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                                 RoundMode::CAST_RINT};
    RegTensor<bfloat16_t> srcBf16Reg;
    RegTensor<bfloat16_t> dstBf16Reg;
    RegTensor<float> stateReg;
    RegTensor<float> scaleReg;
    MaskReg allMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg mask;

    Duplicate(scaleReg, gCumLast);
    Exp<float, MaskMergeMode::ZEROING>(scaleReg, scaleReg, allMask);

    uint32_t remaining = count;
    uint16_t repeatTimes = static_cast<uint16_t>((count + elementsPerRepeat - 1) / elementsPerRepeat);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = UpdateMask<float>(remaining);
        uint32_t offset = static_cast<uint32_t>(i) * elementsPerRepeat;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(srcBf16Reg, srcAddr + offset);
        Cast<float, bfloat16_t, castBf16ToFp32>(stateReg, srcBf16Reg, mask);
        Mul(stateReg, stateReg, scaleReg, mask);
        Cast<bfloat16_t, float, castFp32ToBf16>(dstBf16Reg, stateReg, mask);
        StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(dstAddr + offset, dstBf16Reg, mask);
    }
}

/* 无门控路径保持原 Cast -> Muls(1.0) -> Cast 顺序，并在寄存器中完成。 */
__simd_vf__ inline void ScaleBf16StateVF(__ubuf__ bfloat16_t *dstAddr, __ubuf__ bfloat16_t *srcAddr, uint32_t count)
{
    constexpr uint32_t elementsPerRepeat = VECTOR_REG_WIDTH / sizeof(float);
    static constexpr CastTrait castBf16ToFp32 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                 RoundMode::UNKNOWN};
    static constexpr CastTrait castFp32ToBf16 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                                 RoundMode::CAST_RINT};
    RegTensor<bfloat16_t> srcBf16Reg;
    RegTensor<bfloat16_t> dstBf16Reg;
    RegTensor<float> stateReg;
    RegTensor<float> oneReg;
    MaskReg mask;

    Duplicate(oneReg, static_cast<float>(1.0f));
    uint32_t remaining = count;
    uint16_t repeatTimes = static_cast<uint16_t>((count + elementsPerRepeat - 1) / elementsPerRepeat);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = UpdateMask<float>(remaining);
        uint32_t offset = static_cast<uint32_t>(i) * elementsPerRepeat;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(srcBf16Reg, srcAddr + offset);
        Cast<float, bfloat16_t, castBf16ToFp32>(stateReg, srcBf16Reg, mask);
        Mul(stateReg, stateReg, oneReg, mask);
        Cast<bfloat16_t, float, castFp32ToBf16>(dstBf16Reg, stateReg, mask);
        StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(dstAddr + offset, dstBf16Reg, mask);
    }
}

/* Keep the chunk state update in FP32 and round once when the recurrent state is stored. */
__simd_vf__ inline void UpdateBf16StateByExpVF(__ubuf__ bfloat16_t *stateAddr, __ubuf__ bfloat16_t *updateAddr,
                                               __ubuf__ float *gCumLastAddr, uint32_t count)
{
    constexpr uint32_t elementsPerRepeat = VECTOR_REG_WIDTH / sizeof(float);
    static constexpr CastTrait castBf16ToFp32 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                 RoundMode::UNKNOWN};
    static constexpr CastTrait castFp32ToBf16 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                                 RoundMode::CAST_RINT};
    RegTensor<bfloat16_t> stateBf16Reg;
    RegTensor<bfloat16_t> updateBf16Reg;
    RegTensor<bfloat16_t> roundedBf16Reg;
    RegTensor<float> stateReg;
    RegTensor<float> updateReg;
    RegTensor<float> scaleReg;
    MaskReg allMask = CreateMask<float, MaskPattern::ALL>();
    DataCopy<float, LoadDist::DIST_BRC_B32>(scaleReg, gCumLastAddr);
    Exp<float, MaskMergeMode::ZEROING>(scaleReg, scaleReg, allMask);

    uint32_t remaining = count;
    const uint16_t repeats = static_cast<uint16_t>((count + elementsPerRepeat - 1) / elementsPerRepeat);
    for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
        MaskReg mask = UpdateMask<float>(remaining);
        const uint32_t offset = static_cast<uint32_t>(repeat) * elementsPerRepeat;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(stateBf16Reg, stateAddr + offset);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(updateBf16Reg, updateAddr + offset);
        Cast<float, bfloat16_t, castBf16ToFp32>(stateReg, stateBf16Reg, mask);
        Mul(stateReg, stateReg, scaleReg, mask);
        Cast<float, bfloat16_t, castBf16ToFp32>(updateReg, updateBf16Reg, mask);
        Add(stateReg, stateReg, updateReg, mask);
        Cast<bfloat16_t, float, castFp32ToBf16>(roundedBf16Reg, stateReg, mask);
        StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(stateAddr + offset, roundedBf16Reg, mask);
    }
}

/* Keep the canonical FP32 state in row-major UB and preserve the Golden
 * multiply-then-add recurrence without an intermediate BF16 rounding. */
__simd_vf__ inline void UpdateFp32StateByExpVF(__ubuf__ float *stateAddr, __ubuf__ float *updateAddr,
                                               __ubuf__ float *gCumLastAddr, uint32_t count)
{
    constexpr uint32_t elementsPerRepeat = VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<float> stateReg;
    RegTensor<float> updateReg;
    RegTensor<float> scaleReg;
    MaskReg allMask = CreateMask<float, MaskPattern::ALL>();
    DataCopy<float, LoadDist::DIST_BRC_B32>(scaleReg, gCumLastAddr);
    Exp<float, MaskMergeMode::ZEROING>(scaleReg, scaleReg, allMask);

    uint32_t remaining = count;
    const uint16_t repeats = static_cast<uint16_t>((count + elementsPerRepeat - 1) / elementsPerRepeat);
    for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
        MaskReg mask = UpdateMask<float>(remaining);
        const uint32_t offset = static_cast<uint32_t>(repeat) * elementsPerRepeat;
        LoadAlign(stateReg, stateAddr + offset);
        LoadAlign(updateReg, updateAddr + offset);
        Mul(stateReg, stateReg, scaleReg, mask);
        Add(stateReg, stateReg, updateReg, mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(stateAddr + offset, stateReg, mask);
    }
}

/* Publish a row-major FP32 state as the compact BF16 zN shadow consumed by
 * AIC.  FP32 remains the sole canonical state; this is a derived RNE view. */
__simd_vf__ inline void RowMajorFp32ToCompactZnBf16VF(__ubuf__ float *srcAddr, __ubuf__ bfloat16_t *dstZnAddr,
                                                      uint16_t rowCount, uint16_t colCount, uint16_t srcRowStride,
                                                      uint16_t dstFullRows)
{
    constexpr uint32_t elementsPerRepeat = VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint32_t elementsPerBlock = 32 / sizeof(bfloat16_t);
    static constexpr CastTrait castFp32ToBf16 = {RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING,
                                                 RoundMode::CAST_RINT};
    RegTensor<float> srcReg;
    RegTensor<bfloat16_t> bf16Reg;
    RegTensor<uint16_t> packedBf16Reg;

    const uint16_t repeats = static_cast<uint16_t>((colCount + elementsPerRepeat - 1) / elementsPerRepeat);
    for (uint16_t row = 0; row < rowCount; ++row) {
        for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
            const uint32_t featureBegin = static_cast<uint32_t>(repeat) * elementsPerRepeat;
            const uint32_t remaining = colCount - featureBegin;
            uint32_t validCount = remaining < elementsPerRepeat ? remaining : elementsPerRepeat;
            uint32_t packedCount = validCount;
            MaskReg validMask = UpdateMask<float>(validCount);
            MaskReg packedMask = UpdateMask<uint16_t>(packedCount);
            const uint32_t srcOffset = static_cast<uint32_t>(row) * srcRowStride + featureBegin;
            const uint32_t dstOffset = static_cast<uint32_t>(repeat) * (elementsPerRepeat / elementsPerBlock) *
                                           dstFullRows * elementsPerBlock +
                                       static_cast<uint32_t>(row) * elementsPerBlock;
            LoadAlign(srcReg, srcAddr + srcOffset);
            Cast<bfloat16_t, float, castFp32ToBf16>(bf16Reg, srcReg, validMask);
            Pack<uint16_t, uint32_t, HighLowPart::LOWEST>(packedBf16Reg,
                                                          reinterpret_cast<RegTensor<uint32_t> &>(bf16Reg));
            StoreAlign<uint16_t, DataCopyMode::DATA_BLOCK_COPY>(
                reinterpret_cast<__ubuf__ uint16_t *>(dstZnAddr) + dstOffset, packedBf16Reg, dstFullRows, packedMask);
        }
    }
}

/* Convert a row-major BF16 matrix to the compact zN layout consumed by both
 * the recurrence update and the explicit UB/L1 publication path. */
__simd_vf__ inline void RowMajorToCompactZnBf16VF(__ubuf__ bfloat16_t *srcAddr, __ubuf__ bfloat16_t *dstZnAddr,
                                                  uint16_t rowCount, uint16_t colCount, uint16_t srcRowStride,
                                                  uint16_t dstFullRows)
{
    constexpr uint32_t elementsPerRepeat = VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint32_t elementsPerBlock = 32 / sizeof(bfloat16_t);
    RegTensor<bfloat16_t> srcBf16Reg;
    RegTensor<uint16_t> packedBf16Reg;

    const uint16_t repeats = static_cast<uint16_t>((colCount + elementsPerRepeat - 1) / elementsPerRepeat);
    for (uint16_t row = 0; row < rowCount; ++row) {
        for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
            const uint32_t srcOffset =
                static_cast<uint32_t>(row) * srcRowStride + static_cast<uint32_t>(repeat) * elementsPerRepeat;
            const uint32_t dstOffset = static_cast<uint32_t>(repeat) * (elementsPerRepeat / elementsPerBlock) *
                                           dstFullRows * elementsPerBlock +
                                       static_cast<uint32_t>(row) * elementsPerBlock;
            const uint32_t remaining = colCount - static_cast<uint32_t>(repeat) * elementsPerRepeat;
            uint32_t packedCount = remaining < elementsPerRepeat ? remaining : elementsPerRepeat;
            MaskReg packedMask = UpdateMask<uint16_t>(packedCount);
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(srcBf16Reg, srcAddr + srcOffset);
            Pack<uint16_t, uint32_t, HighLowPart::LOWEST>(packedBf16Reg,
                                                          reinterpret_cast<RegTensor<uint32_t> &>(srcBf16Reg));
            StoreAlign<uint16_t, DataCopyMode::DATA_BLOCK_COPY>(
                reinterpret_cast<__ubuf__ uint16_t *>(dstZnAddr) + dstOffset, packedBf16Reg, dstFullRows, packedMask);
        }
    }
}

/*
 * Direct public recurrence for the low-work exact path.  Each invocation owns
 * one token and one AIV's independent state-row slice.  The canonical state is
 * FP32 for both public state dtypes and is rounded only when a BF16 final state
 * is published by the caller.
 */
template <bool hasGamma>
__simd_vf__ inline void StreamingExactTokenVF(__ubuf__ float *stateAddr, __ubuf__ bfloat16_t *queryAddr,
                                              __ubuf__ bfloat16_t *keyAddr, __ubuf__ float *valueAddr,
                                              __ubuf__ float *betaAddr, __ubuf__ float *gAddr, __ubuf__ float *outAddr,
                                              float scale, uint16_t rowCount, uint16_t dk, uint16_t stateRowStride)
{
    constexpr uint32_t elementsPerRepeat = VECTOR_REG_WIDTH / sizeof(float);
    static constexpr CastTrait castBf16ToFp32 = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                 RoundMode::UNKNOWN};

    RegTensor<bfloat16_t> inputBf16Reg;
    RegTensor<float> stateReg;
    RegTensor<float> keyReg;
    RegTensor<float> queryReg;
    RegTensor<float> productReg;
    RegTensor<float> reducedReg;
    RegTensor<float> totalReg;
    RegTensor<float> alphaReg;
    RegTensor<float> betaReg;
    RegTensor<float> valueReg;
    RegTensor<float> deltaReg;
    RegTensor<float> deltaBroadcastReg;
    MaskReg allMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg oneMask = CreateMask<float, MaskPattern::VL1>();

    if constexpr (hasGamma) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(alphaReg, gAddr);
        Exp<float, MaskMergeMode::ZEROING>(alphaReg, alphaReg, allMask);
    } else {
        Duplicate(alphaReg, static_cast<float>(1.0f));
    }
    LoadAlign<float, LoadDist::DIST_BRC_B32>(betaReg, betaAddr);

    const uint16_t repeats = static_cast<uint16_t>((dk + elementsPerRepeat - 1) / elementsPerRepeat);
    for (uint16_t row = 0; row < rowCount; ++row) {
        Duplicate(totalReg, static_cast<float>(0.0f));
        for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
            const uint32_t featureBegin = static_cast<uint32_t>(repeat) * elementsPerRepeat;
            const uint32_t remaining = static_cast<uint32_t>(dk) - featureBegin;
            uint32_t validCount = remaining < elementsPerRepeat ? remaining : elementsPerRepeat;
            MaskReg validMask = UpdateMask<float>(validCount);
            const uint32_t stateOffset = static_cast<uint32_t>(row) * stateRowStride + featureBegin;

            LoadAlign(stateReg, stateAddr + stateOffset);
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(inputBf16Reg, keyAddr + featureBegin);
            Cast<float, bfloat16_t, castBf16ToFp32>(keyReg, inputBf16Reg, validMask);
            Mul(productReg, stateReg, keyReg, validMask);
            ReduceSum(reducedReg, productReg, validMask);
            Add(totalReg, totalReg, reducedReg, oneMask);
        }

        LoadAlign<float, LoadDist::DIST_BRC_B32>(valueReg, valueAddr + row);
        Mul(productReg, alphaReg, totalReg, oneMask);
        Sub(deltaReg, valueReg, productReg, oneMask);
        Mul(deltaReg, deltaReg, betaReg, oneMask);
        // ReduceSum and the scalar recurrence above define only lane 0.  The
        // outer-product update consumes delta across every valid dk lane.
        Duplicate(deltaBroadcastReg, deltaReg, allMask);

        Duplicate(totalReg, static_cast<float>(0.0f));
        for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
            const uint32_t featureBegin = static_cast<uint32_t>(repeat) * elementsPerRepeat;
            const uint32_t remaining = static_cast<uint32_t>(dk) - featureBegin;
            uint32_t validCount = remaining < elementsPerRepeat ? remaining : elementsPerRepeat;
            MaskReg validMask = UpdateMask<float>(validCount);
            const uint32_t stateOffset = static_cast<uint32_t>(row) * stateRowStride + featureBegin;

            LoadAlign(stateReg, stateAddr + stateOffset);
            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(inputBf16Reg, keyAddr + featureBegin);
            Cast<float, bfloat16_t, castBf16ToFp32>(keyReg, inputBf16Reg, validMask);
            Mul(stateReg, stateReg, alphaReg, validMask);
            Mul(productReg, keyReg, deltaBroadcastReg, validMask);
            Add(stateReg, stateReg, productReg, validMask);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(stateAddr + stateOffset, stateReg, validMask);

            LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(inputBf16Reg, queryAddr + featureBegin);
            Cast<float, bfloat16_t, castBf16ToFp32>(queryReg, inputBf16Reg, validMask);
            Muls(queryReg, queryReg, scale, validMask);
            Mul(productReg, stateReg, queryReg, validMask);
            ReduceSum(reducedReg, productReg, validMask);
            Add(totalReg, totalReg, reducedReg, oneMask);
        }
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(outAddr + row, totalReg, oneMask);
    }
}

} // namespace ChunkGatedDeltaRule

#endif // CHUNK_GATED_DELTA_RULE_STAGE2_VF_H
