/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef VF_SOFTMAX_DN_SEGMENTED_H_
#define VF_SOFTMAX_DN_SEGMENTED_H_

#include "../../../common/op_kernel/arch35/vf/vf_basic_block_utils.h"

namespace FaVectorApi {
using namespace AscendC;
using namespace Reg;

// Only scalar code accesses this descriptor. No UB predicate or score cache.
struct DnSegmentPlan {
    uint64_t boundaries = 0;
    uint64_t maxCounts[2] = {};
    uint64_t pairCounts[2][3] = {};
    int32_t lower = 0;
    int32_t upper = 0;
    bool narrow = false;
};

__aicore__ inline int32_t DnClip(int64_t value, int32_t low, int32_t high)
{
    return static_cast<int32_t>(value < low ? low : (value > high ? high : value));
}

// Fixed state intersections remove the data-dependent state walk. Only
// geometrically possible pairs for the selected boundary mode are emitted.
template <uint16_t N, uint16_t Pair, uint16_t First, uint16_t Second, bool Narrow, uint16_t BoundaryMode,
          bool FullColumns>
__aicore__ inline void DnSetPairCount(DnSegmentPlan &plan, const int32_t *ends)
{
    if constexpr ((BoundaryMode != 1 || (First >= 2 && Second >= 2)) &&
                  (BoundaryMode != 2 || ((First <= 2 || First == 5) && (Second <= 2 || Second == 5))) &&
                  (!FullColumns || Second != 5) && !(First == 1 && Second == 1) && !(First == 3 && Second == 3) &&
                  (!Narrow || !((First == 1 && Second == 2) || (First == 2 && (Second == 2 || Second == 3))))) {
        constexpr int32_t firstOffset = Pair * (N / 4U);
        constexpr int32_t secondOffset = firstOffset + N / 2U;
        const int32_t firstBegin = First == 0 ? 0 : ends[First - 1];
        const int32_t secondBegin = Second == 0 ? 0 : ends[Second - 1];
        const int32_t beginA = firstBegin - firstOffset;
        const int32_t beginB = secondBegin - secondOffset;
        const int32_t endA = ends[First] - firstOffset;
        const int32_t endB = ends[Second] - secondOffset;
        const int32_t intervalBegin = beginA > beginB ? beginA : beginB;
        const int32_t intervalEnd = endA < endB ? endA : endB;
        const int32_t begin = intervalBegin > 0 ? intervalBegin : 0;
        const int32_t end = intervalEnd < N / 4 ? intervalEnd : N / 4;
        const uint32_t count = end > begin ? static_cast<uint32_t>(end - begin) : 0U;
        constexpr uint32_t index = First * (11U - First) / 2U + Second;
        plan.pairCounts[Pair][index / 8U] |= static_cast<uint64_t>(count) << (8U * (index % 8U));
        plan.pairCounts[Pair][2] |= static_cast<uint64_t>(count != 0) << (40U + index);
    }
}

template <uint16_t N, bool Narrow = false, uint16_t BoundaryMode = 0, bool FullColumns = false>
__aicore__ inline DnSegmentPlan MakeDnSegmentPlan(int64_t lower, int64_t upper, uint16_t validN)
{
    static_assert(N >= 128 && N < 256 && N % 16 == 0,
                  "Segment descriptors use byte counts for current masked DN tiles");
    DnSegmentPlan plan;
    plan.lower = DnClip(lower, -64, N);
    plan.upper = DnClip(upper, -64, N);
    plan.narrow = upper - lower + 1 < 64;
    // States: masked-left, lower, visible/both, upper, masked-right, padding.
    int32_t ends[6];
    if constexpr (BoundaryMode == 1) {
        ends[0] = ends[1] = 0;
        ends[2] = DnClip(upper + 1, 0, validN);
        ends[3] = DnClip(upper + 64, 0, validN);
    } else if constexpr (BoundaryMode == 2) {
        ends[0] = DnClip(lower, 0, validN);
        ends[1] = DnClip(lower + 63, 0, validN);
        ends[2] = ends[3] = validN;
    } else if (lower > upper) {
        ends[0] = ends[1] = ends[2] = ends[3] = validN;
    } else {
        ends[0] = DnClip(lower, 0, validN);
        int32_t lowerEnd = DnClip(lower + 63, 0, validN);
        int32_t upperBegin = DnClip(upper + 1, 0, validN);
        ends[1] = lowerEnd < upperBegin ? lowerEnd : upperBegin;
        ends[2] = lowerEnd > upperBegin ? lowerEnd : upperBegin;
        ends[3] = DnClip(upper + 64, 0, validN);
    }
    ends[4] = validN;
    ends[5] = N;
    for (uint32_t i = 0; i < 4; ++i) {
        plan.boundaries |= static_cast<uint64_t>(ends[i]) << (8U * i);
    }
    for (uint32_t lane = 0; lane < 4; ++lane) {
        for (uint32_t kind = 1; kind <= 3; ++kind) {
            uint32_t count = ((ends[kind] + 3U - lane) / 4U) - ((ends[kind - 1] + 3U - lane) / 4U);
            uint32_t index = lane * 3U + kind - 1U;
            plan.maxCounts[index / 8U] |= static_cast<uint64_t>(count) << (8U * (index % 8U));
        }
    }
    for (uint32_t kind = 1; kind <= 3; ++kind) {
        plan.maxCounts[1] |= static_cast<uint64_t>((ends[kind] - ends[kind - 1]) / 4U) << (32U + (kind - 1U) * 8U);
    }
    DnSetPairCount<N, 0, 0, 0, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 0, 0, 1, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 0, 0, 2, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 0, 0, 3, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 0, 0, 4, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 0, 0, 5, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 0, 1, 1, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 0, 1, 2, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 0, 1, 3, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 0, 1, 4, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 0, 1, 5, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 0, 2, 2, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 0, 2, 3, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 0, 2, 4, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 0, 2, 5, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 0, 3, 3, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 0, 3, 4, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 0, 3, 5, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 0, 4, 4, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 0, 4, 5, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 0, 5, 5, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 1, 0, 0, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 1, 0, 1, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 1, 0, 2, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 1, 0, 3, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 1, 0, 4, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 1, 0, 5, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 1, 1, 1, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 1, 1, 2, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 1, 1, 3, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 1, 1, 4, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 1, 1, 5, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 1, 2, 2, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 1, 2, 3, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 1, 2, 4, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 1, 2, 5, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 1, 3, 3, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 1, 3, 4, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 1, 3, 5, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 1, 4, 4, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 1, 4, 5, Narrow, BoundaryMode, FullColumns>(plan, ends);
    DnSetPairCount<N, 1, 5, 5, Narrow, BoundaryMode, FullColumns>(plan, ends);
    return plan;
}

// Keep the threshold in a vector register: the scalar loop counter is used
// for affine addresses only, never as a changing Compare immediate.
__simd_vf__ inline void InitDnTriangleMaskVF(__ubuf__ int8_t *table)
{
    RegTensor<int32_t> rows, zero, one, a, b;
    Arange(rows, 0);
    Duplicate(zero, 0);
    Duplicate(one, 1);
    MaskReg active = CreateMask<int32_t, MaskPattern::ALL>();
    for (uint16_t c = 0; c < 64; ++c) {
        MaskReg less;
        Compares<int32_t, CMPMODE::LT>(less, rows, 0, active);
        Select(a, one, zero, less);
        Select(b, zero, one, less);
        StoreAlign<int32_t, StoreDist::DIST_PACK4_B32>((__ubuf__ int32_t *)(table + c * 64), a, active);
        StoreAlign<int32_t, StoreDist::DIST_PACK4_B32>((__ubuf__ int32_t *)(table + 4096 + c * 64), b, active);
        Adds(rows, rows, -1, active);
    }
}

template <uint16_t Index>
__simd_callee__ inline uint16_t DnSegmentCount(uint64_t a, uint64_t b, uint64_t c = 0)
{
    if constexpr (Index < 8) {
        return static_cast<uint16_t>((a >> (Index * 8U)) & 255U);
    } else if constexpr (Index < 16) {
        return static_cast<uint16_t>((b >> ((Index - 8U) * 8U)) & 255U);
    } else {
        return static_cast<uint16_t>((c >> ((Index - 16U) * 8U)) & 255U);
    }
}

template <uint16_t Kind, bool Narrow>
__simd_callee__ inline void DnSegmentVisibility(MaskReg &visible, __ubuf__ int8_t *upper, __ubuf__ int8_t *lower,
                                                uint32_t rowOffset, MaskReg &active)
{
    RegTensor<uint32_t> bytes, other;
    if constexpr (Kind == 1) {
        LoadAlign<int8_t, LoadDist::DIST_UNPACK4_B8>(reinterpret_cast<RegTensor<int8_t> &>(bytes), lower + rowOffset);
    } else if constexpr (Kind == 3) {
        LoadAlign<int8_t, LoadDist::DIST_UNPACK4_B8>(reinterpret_cast<RegTensor<int8_t> &>(bytes), upper + rowOffset);
    } else if constexpr (Kind == 2 && Narrow) {
        LoadAlign<int8_t, LoadDist::DIST_UNPACK4_B8>(reinterpret_cast<RegTensor<int8_t> &>(bytes), lower + rowOffset);
        LoadAlign<int8_t, LoadDist::DIST_UNPACK4_B8>(reinterpret_cast<RegTensor<int8_t> &>(other), upper + rowOffset);
        Or(bytes, bytes, other, active);
    }
    Compares<uint32_t, CMPMODE::EQ>(visible, bytes, 0U, active);
}

// Each max/min accumulator retains its original column-modulo-four stream.
// Process their common length together for ILP, then their zero/one tails.
template <uint16_t Kind, bool Narrow, int ScaleSign>
__simd_callee__ inline void DnReduceValue(RegTensor<float> &maximum, __ubuf__ float *scores, __ubuf__ int8_t *up,
                                          __ubuf__ int8_t *lo, uint32_t column, uint32_t pitch,
                                          RegTensor<float> &excluded, MaskReg &active)
{
    RegTensor<float> value;
    if constexpr (ScaleSign == 0) {
        Duplicate(value, 0.0F);
    } else {
        LoadAlign(value, scores + column * pitch);
    }
    if constexpr (Kind != 2 || Narrow) {
        MaskReg visible;
        DnSegmentVisibility<Kind, Narrow>(visible, up, lo, column * 64U, active);
        Select(value, value, excluded, visible);
    }
    if constexpr (ScaleSign < 0) {
        Min(maximum, maximum, value, active);
    } else {
        Max(maximum, maximum, value, active);
    }
}

template <uint16_t Kind, bool Narrow, int ScaleSign>
__simd_callee__ inline void DnReduceSegment(RegTensor<float> &max0, RegTensor<float> &max1, RegTensor<float> &max2,
                                            RegTensor<float> &max3, __ubuf__ float *scores, __ubuf__ int8_t *table,
                                            uint16_t common, uint16_t count0, uint16_t count1, uint16_t count2,
                                            uint16_t count3, uint16_t begin, uint32_t pitch, int32_t lower,
                                            int32_t upper, RegTensor<float> &excluded, MaskReg &active)
{
    const uint32_t offset0 = (4U - (begin & 3U)) & 3U;
    const uint32_t offset1 = (5U - (begin & 3U)) & 3U;
    const uint32_t offset2 = (6U - (begin & 3U)) & 3U;
    const uint32_t offset3 = (7U - (begin & 3U)) & 3U;
    const uint16_t tail0 = count0 - common;
    const uint16_t tail1 = count1 - common;
    const uint16_t tail2 = count2 - common;
    const uint16_t tail3 = count3 - common;
    __ubuf__ float *source = scores + begin * pitch;
    // Wrapped setup is only relevant to empty segments. Executed table rows
    // stay in [1,63], and all addresses in the loops are affine.
    __ubuf__ int8_t *up = table + ((begin - upper) & 63U) * 64U;
    __ubuf__ int8_t *lo = table + 4096U + ((begin - lower + 1U) & 63U) * 64U;
    for (uint16_t i = 0; i < common; ++i) {
        DnReduceValue<Kind, Narrow, ScaleSign>(max0, source, up, lo, i * 4U + offset0, pitch, excluded, active);
        DnReduceValue<Kind, Narrow, ScaleSign>(max1, source, up, lo, i * 4U + offset1, pitch, excluded, active);
        DnReduceValue<Kind, Narrow, ScaleSign>(max2, source, up, lo, i * 4U + offset2, pitch, excluded, active);
        DnReduceValue<Kind, Narrow, ScaleSign>(max3, source, up, lo, i * 4U + offset3, pitch, excluded, active);
    }
    const uint32_t tailBase = common * 4U;
    for (uint16_t i = 0; i < tail0; ++i) {
        DnReduceValue<Kind, Narrow, ScaleSign>(max0, source, up, lo, tailBase + offset0 + i * 4U, pitch, excluded,
                                               active);
    }
    for (uint16_t i = 0; i < tail1; ++i) {
        DnReduceValue<Kind, Narrow, ScaleSign>(max1, source, up, lo, tailBase + offset1 + i * 4U, pitch, excluded,
                                               active);
    }
    for (uint16_t i = 0; i < tail2; ++i) {
        DnReduceValue<Kind, Narrow, ScaleSign>(max2, source, up, lo, tailBase + offset2 + i * 4U, pitch, excluded,
                                               active);
    }
    for (uint16_t i = 0; i < tail3; ++i) {
        DnReduceValue<Kind, Narrow, ScaleSign>(max3, source, up, lo, tailBase + offset3 + i * 4U, pitch, excluded,
                                               active);
    }
}

template <uint16_t Kind, bool Narrow>
__simd_callee__ inline void DnSegmentExp(RegTensor<float> &value, __ubuf__ float *scores, __ubuf__ int8_t *up,
                                         __ubuf__ int8_t *lo, uint32_t offset, uint32_t pitch, float scale,
                                         RegTensor<float> &maximum, RegTensor<float> &masked, MaskReg &active)
{
    LoadAlign(value, scores + offset * pitch);
    Muls(value, value, scale, active);
    if constexpr (Kind != 2 || Narrow) {
        MaskReg visible;
        DnSegmentVisibility<Kind, Narrow>(visible, up, lo, offset * 64U, active);
        Select(value, value, masked, visible);
    }
    FusedExpSub(value, value, maximum, active);
}

template <uint16_t Kind>
__simd_callee__ inline void DnAddSegmentSum(RegTensor<float> &sum, RegTensor<float> &value, RegTensor<float> &maskedExp,
                                            RegTensor<float> &paddingExp, MaskReg &active)
{
    if constexpr (Kind == 0 || Kind == 4) {
        Add(sum, maskedExp, sum, active);
    } else if constexpr (Kind == 5) {
        Add(sum, paddingExp, sum, active);
    } else {
        Add(sum, value, sum, active);
    }
}

template <typename T, uint16_t N, bool Narrow, uint16_t First, uint16_t Second>
__simd_callee__ inline void DnExpSegmentPair(__ubuf__ T *&output, __ubuf__ float *scores, __ubuf__ int8_t *table,
                                             uint32_t column, uint16_t count, uint32_t pitch, int32_t lower,
                                             int32_t upper, float scale, RegTensor<float> &maximum,
                                             RegTensor<float> &masked, RegTensor<float> &maskedExp,
                                             RegTensor<float> &paddingExp, RegTensor<float> &sumFirst,
                                             RegTensor<float> &sumSecond, MaskReg &active, MaskReg &storeMask)
{
    constexpr bool CONST_FIRST = First == 0 || First >= 4;
    constexpr bool CONST_SECOND = Second == 0 || Second >= 4;
    __ubuf__ float *firstSrc = scores + column * pitch;
    __ubuf__ float *secondSrc = firstSrc + (N / 2U) * pitch;
    __ubuf__ int8_t *firstUp = table + ((column - upper) & 63U) * 64U;
    __ubuf__ int8_t *firstLo = table + 4096U + ((column - lower + 1U) & 63U) * 64U;
    __ubuf__ int8_t *secondUp = table + ((column + N / 2U - upper) & 63U) * 64U;
    __ubuf__ int8_t *secondLo = table + 4096U + ((column + N / 2U - lower + 1U) & 63U) * 64U;
    RegTensor<T> firstCast, secondCast, packed, unused;
    if constexpr (CONST_FIRST) {
        if constexpr (First == 5) {
            Cast<T, float, castTraitZero>(firstCast, paddingExp, active);
        } else {
            Cast<T, float, castTraitZero>(firstCast, maskedExp, active);
        }
    }
    if constexpr (CONST_SECOND) {
        if constexpr (Second == 5) {
            Cast<T, float, castTraitZero>(secondCast, paddingExp, active);
        } else {
            Cast<T, float, castTraitZero>(secondCast, maskedExp, active);
        }
    }
    if constexpr (CONST_FIRST && CONST_SECOND) {
        DeInterleave(packed, unused, firstCast, secondCast);
    }
    for (uint16_t i = 0; i < count; ++i) {
        RegTensor<float> firstValue, secondValue;
        if constexpr (!CONST_FIRST) {
            DnSegmentExp<First, Narrow>(firstValue, firstSrc, firstUp, firstLo, i, pitch, scale, maximum, masked,
                                        active);
            Cast<T, float, castTraitZero>(firstCast, firstValue, active);
        }
        if constexpr (!CONST_SECOND) {
            DnSegmentExp<Second, Narrow>(secondValue, secondSrc, secondUp, secondLo, i, pitch, scale, maximum, masked,
                                         active);
            Cast<T, float, castTraitZero>(secondCast, secondValue, active);
        }
        if constexpr (!CONST_FIRST || !CONST_SECOND) {
            DeInterleave(packed, unused, firstCast, secondCast);
        }
        StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(output, packed, (N / 2U) | 1U, 1U,
                                                                                    storeMask);
        DnAddSegmentSum<First>(sumFirst, firstValue, maskedExp, paddingExp, active);
        DnAddSegmentSum<Second>(sumSecond, secondValue, maskedExp, paddingExp, active);
    }
}

template <typename T, uint16_t N, bool Narrow, uint16_t Pair, uint16_t BoundaryMode, bool FullColumns>
__simd_callee__ inline void DnExpPair(__ubuf__ T *output, __ubuf__ float *scores, __ubuf__ int8_t *table,
                                      uint64_t counts0, uint64_t counts1, uint64_t counts2, uint32_t pitch,
                                      int32_t lower, int32_t upper, float scale, RegTensor<float> &maximum,
                                      RegTensor<float> &masked, RegTensor<float> &maskedExp,
                                      RegTensor<float> &paddingExp, RegTensor<float> &sumFirst,
                                      RegTensor<float> &sumSecond, MaskReg &active, MaskReg &storeMask)
{
    uint32_t column = Pair * (N / 4U);
#define DN_EXP_SEGMENT(FIRST, SECOND, INDEX) \
    if constexpr ((BoundaryMode != 1 || (FIRST >= 2 && SECOND >= 2)) && \
                  (BoundaryMode != 2 || ((FIRST <= 2 || FIRST == 5) && (SECOND <= 2 || SECOND == 5))) && \
                  (!FullColumns || SECOND != 5) && !(FIRST == 1 && SECOND == 1) && !(FIRST == 3 && SECOND == 3) && \
                  (!Narrow || !((FIRST == 1 && SECOND == 2) || (FIRST == 2 && (SECOND == 2 || SECOND == 3))))) { \
        const uint16_t count = DnSegmentCount<INDEX>(counts0, counts1, counts2); \
        const uint16_t enabled = static_cast<uint16_t>((counts2 >> (40U + INDEX)) & 1U); \
        for (uint16_t segment = 0; segment < enabled; ++segment) { \
            DnExpSegmentPair<T, N, Narrow, FIRST, SECOND>(output, scores, table, column, count, pitch, lower, upper, \
                                                          scale, maximum, masked, maskedExp, paddingExp, sumFirst, \
                                                          sumSecond, active, storeMask); \
        } \
        column += count; \
    }
    DN_EXP_SEGMENT(0, 0, 0);
    DN_EXP_SEGMENT(0, 1, 1);
    DN_EXP_SEGMENT(0, 2, 2);
    DN_EXP_SEGMENT(0, 3, 3);
    DN_EXP_SEGMENT(0, 4, 4);
    DN_EXP_SEGMENT(0, 5, 5);
    DN_EXP_SEGMENT(1, 1, 6);
    DN_EXP_SEGMENT(1, 2, 7);
    DN_EXP_SEGMENT(1, 3, 8);
    DN_EXP_SEGMENT(1, 4, 9);
    DN_EXP_SEGMENT(1, 5, 10);
    DN_EXP_SEGMENT(2, 2, 11);
    DN_EXP_SEGMENT(2, 3, 12);
    DN_EXP_SEGMENT(2, 4, 13);
    DN_EXP_SEGMENT(2, 5, 14);
    DN_EXP_SEGMENT(3, 3, 15);
    DN_EXP_SEGMENT(3, 4, 16);
    DN_EXP_SEGMENT(3, 5, 17);
    DN_EXP_SEGMENT(4, 4, 18);
    DN_EXP_SEGMENT(4, 5, 19);
    DN_EXP_SEGMENT(5, 5, 20);

#undef DN_EXP_SEGMENT
}

template <typename T, uint16_t N, int ScaleSign, bool Rows64, bool Narrow, uint16_t BoundaryMode, bool FullColumns>
__simd_vf__ inline void DnSegmentedSoftmaxVF(__ubuf__ T *output, __ubuf__ float *scores, __ubuf__ float *sumUb,
                                             __ubuf__ float *maxUb, __ubuf__ float *expUb, __ubuf__ int8_t *table,
                                             uint32_t runtimePitch, uint32_t originN, float scale, float minValue,
                                             int32_t lower, int32_t upper, uint64_t boundaries, uint64_t maxCounts0,
                                             uint64_t maxCounts1, uint64_t pair00, uint64_t pair01, uint64_t pair02,
                                             uint64_t pair10, uint64_t pair11, uint64_t pair12)
{
    const uint32_t pitch = Rows64 ? 64U : runtimePitch;
    uint32_t laneCount = pitch;
    MaskReg active = UpdateMask<float>(laneCount);
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    uint32_t storeCount = 128;
    MaskReg storeMask = UpdateMask<T>(storeCount);
    constexpr float INITIAL = ScaleSign < 0 ? __builtin_inff() : -__builtin_inff();
    RegTensor<float> max0, max1, max2, max3, excluded;
    Duplicate(max0, INITIAL);
    Duplicate(max1, INITIAL);
    Duplicate(max2, INITIAL);
    Duplicate(max3, INITIAL);
    Duplicate(excluded, INITIAL);
#define DN_MAX_SEGMENT(KIND) \
    if constexpr ((BoundaryMode != 1 || KIND != 1) && (BoundaryMode != 2 || KIND != 3)) { \
        const uint16_t common = static_cast<uint16_t>((maxCounts1 >> (32U + (KIND - 1U) * 8U)) & 255U); \
        DnReduceSegment<KIND, Narrow, ScaleSign>( \
            max0, max1, max2, max3, scores, table, common, DnSegmentCount<KIND - 1>(maxCounts0, maxCounts1), \
            DnSegmentCount<KIND + 2>(maxCounts0, maxCounts1), DnSegmentCount<KIND + 5>(maxCounts0, maxCounts1), \
            DnSegmentCount<KIND + 8>(maxCounts0, maxCounts1), \
            static_cast<uint16_t>((boundaries >> ((KIND - 1U) * 8U)) & 255U), pitch, lower, upper, excluded, active); \
    }
    DN_MAX_SEGMENT(1);
    DN_MAX_SEGMENT(2);
    DN_MAX_SEGMENT(3);
#undef DN_MAX_SEGMENT
    if constexpr (ScaleSign < 0) {
        Min(max0, max0, max2, all);
        Min(max1, max1, max3, all);
        Min(max0, max0, max1, all);
    } else {
        Max(max0, max0, max2, all);
        Max(max1, max1, max3, all);
        Max(max0, max0, max1, all);
    }
    if constexpr (ScaleSign != 0) {
        Muls(max0, max0, scale, active);
    }
    RegTensor<float> padding, masked, maskBound, oldMax;
    Duplicate(padding, minValue);
    Duplicate(masked, -1000000000000.0F);
    Max(max0, max0, padding, active);
    RegTensor<int32_t> rows, lowerRows, upperRows;
    Arange(rows, 0);
    Adds(lowerRows, rows, lower, active);
    Adds(upperRows, rows, upper, active);
    MaskReg firstVisible, lastVisible, allVisible;
    Compares<int32_t, CMPMODE::LE>(firstVisible, lowerRows, 0, active);
    Compares<int32_t, CMPMODE::GE>(lastVisible, upperRows, static_cast<int32_t>(originN) - 1, active);
    And(allVisible, firstVisible, lastVisible, active);
    Select(maskBound, padding, masked, allVisible);
    Max(max0, max0, maskBound, active);
    // Keep the existing unit-PScale subtraction and online-softmax update.
    RegTensor<float> zero;
    Duplicate(zero, 0.0F);
    Sub(max0, max0, zero, all);
    LoadAlign(oldMax, maxUb);
    Max(max0, max0, oldMax, all);
    FusedExpSub(oldMax, oldMax, max0, all);
    StoreAlign(maxUb, max0, all);
    StoreAlign(expUb, oldMax, all);
    RegTensor<float> maskedExp, paddingExp;
    FusedExpSub(maskedExp, masked, max0, all);
    FusedExpSub(paddingExp, padding, max0, all);
    RegTensor<float> sum0, sum1, sum2, sum3;
    Duplicate(sum0, 0.0F);
    Duplicate(sum1, 0.0F);
    Duplicate(sum2, 0.0F);
    Duplicate(sum3, 0.0F);
    DnExpPair<T, N, Narrow, 0, BoundaryMode, FullColumns>(output, scores, table, pair00, pair01, pair02, pitch, lower,
                                                          upper, scale, max0, masked, maskedExp, paddingExp, sum0, sum2,
                                                          all, storeMask);
    DnExpPair<T, N, Narrow, 1, BoundaryMode, FullColumns>(output + N * 4U, scores, table, pair10, pair11, pair12, pitch,
                                                          lower, upper, scale, max0, masked, maskedExp, paddingExp,
                                                          sum1, sum3, all, storeMask);
    Add(sum0, sum2, sum0, all);
    Add(sum1, sum3, sum1, all);
    Add(sum0, sum0, sum1, all);
    RegTensor<float> oldSum;
    LoadAlign(oldSum, sumUb);
    Mul(oldSum, oldMax, oldSum, all);
    Add(oldSum, oldSum, sum0, all);
    StoreAlign(sumUb, oldSum, all);
}

template <typename T, uint16_t N, int ScaleSign, bool Rows64, bool FullColumns>
__aicore__ inline void DnSegmentedSoftmax(const LocalTensor<T> &output, const LocalTensor<float> &scores,
                                          const LocalTensor<float> &sum, const LocalTensor<float> &maximum,
                                          const LocalTensor<float> &exp, const LocalTensor<uint8_t> &table,
                                          uint32_t pitch, uint32_t originN, float scale, float minValue, int64_t lower,
                                          int64_t upper)
{
    const uint32_t validN = FullColumns ? N : originN;
    const int32_t clippedLower = DnClip(lower, -64, N);
    const int32_t clippedUpper = DnClip(upper, -64, N);
#define DN_SEGMENTED_SOFTMAX(NARROW, BOUNDARIES) \
    const DnSegmentPlan plan = MakeDnSegmentPlan<N, NARROW, BOUNDARIES, FullColumns>(lower, upper, validN); \
    DnSegmentedSoftmaxVF<T, N, ScaleSign, Rows64, NARROW, BOUNDARIES, FullColumns>( \
        (__ubuf__ T *)output.GetPhyAddr(), (__ubuf__ float *)scores.GetPhyAddr(), (__ubuf__ float *)sum.GetPhyAddr(), \
        (__ubuf__ float *)maximum.GetPhyAddr(), (__ubuf__ float *)exp.GetPhyAddr(), \
        (__ubuf__ int8_t *)table.GetPhyAddr(), pitch, validN, scale, minValue, plan.lower, plan.upper, \
        plan.boundaries, plan.maxCounts[0], plan.maxCounts[1], plan.pairCounts[0][0], plan.pairCounts[0][1], \
        plan.pairCounts[0][2], plan.pairCounts[1][0], plan.pairCounts[1][1], plan.pairCounts[1][2])
    if (lower > upper) {
        DN_SEGMENTED_SOFTMAX(false, 3);
    } else if (clippedLower <= -64) {
        DN_SEGMENTED_SOFTMAX(false, 1);
    } else if (clippedUpper >= static_cast<int32_t>(validN) - 1) {
        DN_SEGMENTED_SOFTMAX(false, 2);
    } else if (upper - lower + 1 < 64) {
        DN_SEGMENTED_SOFTMAX(true, 3);
    } else {
        DN_SEGMENTED_SOFTMAX(false, 3);
    }
#undef DN_SEGMENTED_SOFTMAX
}
} // namespace FaVectorApi
#endif
