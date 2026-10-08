/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_ATTN_RES_PREPARE_VF_H
#define BLOCK_ATTN_RES_PREPARE_VF_H

#include <type_traits>

#include "kernel_operator.h"

namespace BlockAttnResPrepareVF {

using HighLowPart = AscendC::Reg::HighLowPart;
using LoadDist = AscendC::Reg::LoadDist;
using MaskMergeMode = AscendC::Reg::MaskMergeMode;
using MaskPattern = AscendC::Reg::MaskPattern;
using MaskReg = AscendC::Reg::MaskReg;
using MemType = AscendC::Reg::MemType;
using ReduceType = AscendC::Reg::ReduceType;
using StoreDist = AscendC::Reg::StoreDist;

template <typename T>
using RegTensor = AscendC::Reg::RegTensor<T>;

using AscendC::Reg::Add;
using AscendC::Reg::Adds;
using AscendC::Reg::CreateMask;
using AscendC::Reg::Div;
using AscendC::Reg::Duplicate;
using AscendC::Reg::Exp;
using AscendC::Reg::LoadAlign;
using AscendC::Reg::LocalMemBar;
using AscendC::Reg::Mul;
using AscendC::Reg::MulAddDst;
using AscendC::Reg::Muls;
using AscendC::Reg::Reduce;
using AscendC::Reg::RegTraitNumOne;
using AscendC::Reg::Sqrt;
using AscendC::Reg::StoreAlign;
using AscendC::Reg::Sub;
using AscendC::Reg::UpdateMask;

constexpr uint32_t FP32_REG_ELEMS = 64U;
constexpr uint32_t STAT_VECTOR_COUNT = 2U;
constexpr uint32_t SCALAR_BLOCK_ELEMS = 8U;
constexpr uint32_t SUM_SQUARE_OFFSET = 0U;
constexpr uint32_t DOT_OFFSET = FP32_REG_ELEMS;
constexpr uint32_t MAX_OFFSET = STAT_VECTOR_COUNT * FP32_REG_ELEMS;
constexpr uint32_t SUM_OFFSET = MAX_OFFSET + SCALAR_BLOCK_ELEMS;
constexpr float FP32_LOWEST_FINITE = -3.4028234663852886e+38F;
constexpr AscendC::Reg::DivSpecificMode BARP_DIV_0ULP_FTZ_TRUE_MODE = {
    MaskMergeMode::ZEROING,
    true,
    AscendC::DivAlgo::PRECISION_0ULP_FTZ_TRUE,
};

constexpr AscendC::Reg::CastTrait BARP_CAST_BF16_TO_FP32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

__simd_vf__ inline void InitializeEmptyOnlineSoftmax(__ubuf__ float *statAddr)
{
    RegTensor<float> maxReg;
    RegTensor<float> sumReg;
    MaskReg oneMask = CreateMask<float, MaskPattern::VL1>();

    Duplicate(maxReg, FP32_LOWEST_FINITE, oneMask);
    Duplicate(sumReg, 0.0F, oneMask);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + MAX_OFFSET, maxReg, oneMask);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + SUM_OFFSET, sumReg, oneMask);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

__simd_vf__ inline void FillZero(__ubuf__ float *outputAddr, uint32_t validD)
{
    RegTensor<float> zeroReg;
    MaskReg allMask = CreateMask<float, MaskPattern::ALL>();
    Duplicate(zeroReg, 0.0F, allMask);

    uint32_t remaining = validD;
    const uint16_t loopCount = static_cast<uint16_t>((validD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
    for (uint16_t loop = 0; loop < loopCount; ++loop) {
        MaskReg validMask = UpdateMask<float, RegTraitNumOne>(remaining);
        const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
        StoreAlign<float, StoreDist::DIST_NORM>(outputAddr + offset, zeroReg, validMask);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

// These raw UB arguments are internal static byte offsets. UB offset zero is valid and must not be treated as nullptr.
template <typename INPUT_T>
__simd_callee__ inline void LoadInputAsFloat(RegTensor<float> &dst, __ubuf__ INPUT_T *src, MaskReg mask)
{
    if constexpr (std::is_same<INPUT_T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> srcReg;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(srcReg, src);
        Cast<float, bfloat16_t, BARP_CAST_BF16_TO_FP32>(dst, srcReg, mask);
    } else {
        LoadAlign<float, LoadDist::DIST_NORM>(dst, reinterpret_cast<__ubuf__ float *>(src));
    }
}

template <typename INPUT_T, bool FIRST_D_TILE>
__simd_vf__ inline void AccumulateSquareDot(__ubuf__ INPUT_T *qAddr, __ubuf__ INPUT_T *vAddr,
                                            __ubuf__ float *statAddr, uint32_t validD, uint32_t nIndex)
{
    RegTensor<float> qReg;
    RegTensor<float> vReg;
    RegTensor<float> squareReg;
    RegTensor<float> dotReg;
    RegTensor<float> squareAccReg;
    RegTensor<float> dotAccReg;
    RegTensor<float> squareReduceReg;
    RegTensor<float> dotReduceReg;

    MaskReg allMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg oneMask = CreateMask<float, MaskPattern::VL1>();
    Duplicate(squareAccReg, 0.0F, allMask);
    Duplicate(dotAccReg, 0.0F, allMask);

    uint32_t remaining = validD;
    const uint16_t loopCount = static_cast<uint16_t>((validD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
    for (uint16_t loop = 0; loop < loopCount; ++loop) {
        MaskReg validMask = UpdateMask<float, RegTraitNumOne>(remaining);
        const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
        LoadInputAsFloat<INPUT_T>(qReg, qAddr + offset, validMask);
        LoadInputAsFloat<INPUT_T>(vReg, vAddr + offset, validMask);
        Mul<float, MaskMergeMode::ZEROING>(squareReg, vReg, vReg, validMask);
        Mul<float, MaskMergeMode::ZEROING>(dotReg, qReg, vReg, validMask);
        Add<float, MaskMergeMode::ZEROING>(squareAccReg, squareAccReg, squareReg, allMask);
        Add<float, MaskMergeMode::ZEROING>(dotAccReg, dotAccReg, dotReg, allMask);
    }

    Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(squareReduceReg, squareAccReg, allMask);
    Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(dotReduceReg, dotAccReg, allMask);

    if constexpr (!FIRST_D_TILE) {
        RegTensor<float> oldValueReg;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(oldValueReg, statAddr + SUM_SQUARE_OFFSET + nIndex);
        Add<float, MaskMergeMode::ZEROING>(squareReduceReg, squareReduceReg, oldValueReg, oneMask);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(oldValueReg, statAddr + DOT_OFFSET + nIndex);
        Add<float, MaskMergeMode::ZEROING>(dotReduceReg, dotReduceReg, oldValueReg, oneMask);
    }

    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + SUM_SQUARE_OFFSET + nIndex, squareReduceReg,
                                                         oneMask);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + DOT_OFFSET + nIndex, dotReduceReg, oneMask);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

__simd_vf__ inline void CacheSquareNorms(__ubuf__ float *statAddr, __ubuf__ float *normAddr, uint32_t validN)
{
    RegTensor<float> norms;
    MaskReg validMask = UpdateMask<float, RegTraitNumOne>(validN);
    LoadAlign<float, LoadDist::DIST_NORM>(norms, statAddr + SUM_SQUARE_OFFSET);
    StoreAlign<float, StoreDist::DIST_NORM>(normAddr, norms, validMask);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

template <typename INPUT_T, bool FIRST_GROUP>
__simd_vf__ inline void AccumulateFourStats(__ubuf__ INPUT_T *qAddr, __ubuf__ INPUT_T *vAddr,
                                             __ubuf__ float *statAddr, __ubuf__ float *normAddr,
                                             uint32_t validD, uint32_t nIndex,
                                             uint32_t qPitch, uint32_t statPitch)
{
    RegTensor<float> vReg;
    RegTensor<float> q0, q1, q2, q3;
    RegTensor<float> dot0, dot1, dot2, dot3;
    RegTensor<float> acc0, acc1, acc2, acc3;
    RegTensor<float> red0, red1, red2, red3;
    RegTensor<float> squareReg, squareAcc, normReg;
    MaskReg allMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg oneMask = CreateMask<float, MaskPattern::VL1>();
    Duplicate(acc0, 0.0F, allMask);
    Duplicate(acc1, 0.0F, allMask);
    Duplicate(acc2, 0.0F, allMask);
    Duplicate(acc3, 0.0F, allMask);
    if constexpr (FIRST_GROUP) {
        Duplicate(squareAcc, 0.0F, allMask);
    }
    uint32_t remaining = validD;
    const uint16_t loopCount = static_cast<uint16_t>((validD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
    for (uint16_t loop = 0; loop < loopCount; ++loop) {
        MaskReg validMask = UpdateMask<float, RegTraitNumOne>(remaining);
        const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
        LoadInputAsFloat<INPUT_T>(vReg, vAddr + offset, validMask);
#define BARP_DOT_ROW(Q, DOT, ACC, ROW) \
        LoadInputAsFloat<INPUT_T>(Q, qAddr + (ROW) * qPitch + offset, validMask); \
        Mul<float, MaskMergeMode::ZEROING>(DOT, Q, vReg, validMask); \
        Add<float, MaskMergeMode::ZEROING>(ACC, ACC, DOT, allMask)
        BARP_DOT_ROW(q0, dot0, acc0, 0U);
        BARP_DOT_ROW(q1, dot1, acc1, 1U);
        BARP_DOT_ROW(q2, dot2, acc2, 2U);
        BARP_DOT_ROW(q3, dot3, acc3, 3U);
#undef BARP_DOT_ROW
        if constexpr (FIRST_GROUP) {
            Mul<float, MaskMergeMode::ZEROING>(squareReg, vReg, vReg, validMask);
            Add<float, MaskMergeMode::ZEROING>(squareAcc, squareAcc, squareReg, allMask);
        }
    }
    Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(red0, acc0, allMask);
    Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(red1, acc1, allMask);
    Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(red2, acc2, allMask);
    Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(red3, acc3, allMask);
    if constexpr (FIRST_GROUP) {
        Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(normReg, squareAcc, allMask);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(normAddr + nIndex, normReg, oneMask);
    } else {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(normReg, normAddr + nIndex);
    }
#define BARP_STORE_STAT(ROW, DOT) \
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + (ROW) * statPitch + \
        SUM_SQUARE_OFFSET + nIndex, normReg, oneMask); \
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + (ROW) * statPitch + \
        DOT_OFFSET + nIndex, DOT, oneMask)
    BARP_STORE_STAT(0U, red0);
    BARP_STORE_STAT(1U, red1);
    BARP_STORE_STAT(2U, red2);
    BARP_STORE_STAT(3U, red3);
#undef BARP_STORE_STAT
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

template <typename INPUT_T, bool FIRST_GROUP>
__simd_vf__ inline void AccumulateFourStatsTwoN(__ubuf__ INPUT_T *qAddr,
                                                 __ubuf__ INPUT_T *v0Addr, __ubuf__ INPUT_T *v1Addr,
                                                 __ubuf__ float *statAddr, __ubuf__ float *normAddr,
                                                 uint32_t validD, uint32_t nIndex,
                                                 uint32_t qPitch, uint32_t statPitch)
{
    RegTensor<float> v0, v1, qReg, dotReg, redReg;
    RegTensor<float> acc00, acc01, acc10, acc11, acc20, acc21, acc30, acc31;
    RegTensor<float> sq0, sq1, norm0, norm1;
    MaskReg allMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg oneMask = CreateMask<float, MaskPattern::VL1>();
    Duplicate(acc00, 0.0F, allMask); Duplicate(acc01, 0.0F, allMask);
    Duplicate(acc10, 0.0F, allMask); Duplicate(acc11, 0.0F, allMask);
    Duplicate(acc20, 0.0F, allMask); Duplicate(acc21, 0.0F, allMask);
    Duplicate(acc30, 0.0F, allMask); Duplicate(acc31, 0.0F, allMask);
    if constexpr (FIRST_GROUP) {
        Duplicate(sq0, 0.0F, allMask);
        Duplicate(sq1, 0.0F, allMask);
    }
    uint32_t remaining = validD;
    const uint16_t loopCount = static_cast<uint16_t>((validD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
    for (uint16_t loop = 0; loop < loopCount; ++loop) {
        MaskReg validMask = UpdateMask<float, RegTraitNumOne>(remaining);
        const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
        LoadInputAsFloat<INPUT_T>(v0, v0Addr + offset, validMask);
        LoadInputAsFloat<INPUT_T>(v1, v1Addr + offset, validMask);
#define BARP_PAIR_ROW(ROW, ACC0, ACC1) \
        LoadInputAsFloat<INPUT_T>(qReg, qAddr + (ROW) * qPitch + offset, validMask); \
        Mul<float, MaskMergeMode::ZEROING>(dotReg, qReg, v0, validMask); \
        Add<float, MaskMergeMode::ZEROING>(ACC0, ACC0, dotReg, allMask); \
        Mul<float, MaskMergeMode::ZEROING>(dotReg, qReg, v1, validMask); \
        Add<float, MaskMergeMode::ZEROING>(ACC1, ACC1, dotReg, allMask)
        BARP_PAIR_ROW(0U, acc00, acc01);
        BARP_PAIR_ROW(1U, acc10, acc11);
        BARP_PAIR_ROW(2U, acc20, acc21);
        BARP_PAIR_ROW(3U, acc30, acc31);
#undef BARP_PAIR_ROW
        if constexpr (FIRST_GROUP) {
            Mul<float, MaskMergeMode::ZEROING>(dotReg, v0, v0, validMask);
            Add<float, MaskMergeMode::ZEROING>(sq0, sq0, dotReg, allMask);
            Mul<float, MaskMergeMode::ZEROING>(dotReg, v1, v1, validMask);
            Add<float, MaskMergeMode::ZEROING>(sq1, sq1, dotReg, allMask);
        }
    }
    if constexpr (FIRST_GROUP) {
        Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(norm0, sq0, allMask);
        Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(norm1, sq1, allMask);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(normAddr + nIndex, norm0, oneMask);
        StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(normAddr + nIndex + 1U, norm1, oneMask);
    } else {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(norm0, normAddr + nIndex);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(norm1, normAddr + nIndex + 1U);
    }
#define BARP_STORE_PAIR(ROW, ACC0, ACC1) \
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + (ROW) * statPitch + \
        SUM_SQUARE_OFFSET + nIndex, norm0, oneMask); \
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + (ROW) * statPitch + \
        SUM_SQUARE_OFFSET + nIndex + 1U, norm1, oneMask); \
    Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(redReg, ACC0, allMask); \
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + (ROW) * statPitch + \
        DOT_OFFSET + nIndex, redReg, oneMask); \
    Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(redReg, ACC1, allMask); \
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + (ROW) * statPitch + \
        DOT_OFFSET + nIndex + 1U, redReg, oneMask)
    BARP_STORE_PAIR(0U, acc00, acc01);
    BARP_STORE_PAIR(1U, acc10, acc11);
    BARP_STORE_PAIR(2U, acc20, acc21);
    BARP_STORE_PAIR(3U, acc30, acc31);
#undef BARP_STORE_PAIR
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

template <typename INPUT_T>
__simd_vf__ inline void AccumulateDotWithCachedNorm(__ubuf__ INPUT_T *qAddr, __ubuf__ INPUT_T *vAddr,
                                                     __ubuf__ float *statAddr, __ubuf__ float *normAddr,
                                                     uint32_t validD, uint32_t nIndex)
{
    RegTensor<float> qReg;
    RegTensor<float> vReg;
    RegTensor<float> dotReg;
    RegTensor<float> dotAccReg;
    RegTensor<float> dotReduceReg;
    RegTensor<float> normReg;
    MaskReg allMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg oneMask = CreateMask<float, MaskPattern::VL1>();
    Duplicate(dotAccReg, 0.0F, allMask);

    uint32_t remaining = validD;
    const uint16_t loopCount = static_cast<uint16_t>((validD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
    for (uint16_t loop = 0; loop < loopCount; ++loop) {
        MaskReg validMask = UpdateMask<float, RegTraitNumOne>(remaining);
        const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
        LoadInputAsFloat<INPUT_T>(qReg, qAddr + offset, validMask);
        LoadInputAsFloat<INPUT_T>(vReg, vAddr + offset, validMask);
        Mul<float, MaskMergeMode::ZEROING>(dotReg, qReg, vReg, validMask);
        Add<float, MaskMergeMode::ZEROING>(dotAccReg, dotAccReg, dotReg, allMask);
    }
    Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(dotReduceReg, dotAccReg, allMask);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(normReg, normAddr + nIndex);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + SUM_SQUARE_OFFSET + nIndex, normReg, oneMask);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + DOT_OFFSET + nIndex, dotReduceReg, oneMask);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

__simd_vf__ inline void FinalizeSingleBlock(__ubuf__ float *statAddr, float reciprocalD, float eps)
{
    RegTensor<float> sumSquareReg;
    RegTensor<float> dotReg;
    RegTensor<float> rmsReg;
    RegTensor<float> zReg;
    RegTensor<float> oneReg;
    MaskReg oneMask = CreateMask<float, MaskPattern::VL1>();

    LoadAlign<float, LoadDist::DIST_NORM>(sumSquareReg, statAddr + SUM_SQUARE_OFFSET);
    LoadAlign<float, LoadDist::DIST_NORM>(dotReg, statAddr + DOT_OFFSET);
    Muls<float, float, MaskMergeMode::ZEROING>(sumSquareReg, sumSquareReg, reciprocalD, oneMask);
    Adds<float, float, MaskMergeMode::ZEROING>(sumSquareReg, sumSquareReg, eps, oneMask);
    Sqrt<float, MaskMergeMode::ZEROING>(rmsReg, sumSquareReg, oneMask);
    Div<float, &BARP_DIV_0ULP_FTZ_TRUE_MODE>(zReg, dotReg, rmsReg, oneMask);
    Duplicate(oneReg, 1.0F, oneMask);

    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + MAX_OFFSET, zReg, oneMask);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + SUM_OFFSET, oneReg, oneMask);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

__simd_vf__ inline void FinalizeSoftmax(__ubuf__ float *statAddr, uint32_t validN, float reciprocalD, float eps)
{
    RegTensor<float> sumSquareReg;
    RegTensor<float> dotReg;
    RegTensor<float> rmsReg;
    RegTensor<float> zReg;
    RegTensor<float> maxReg;
    RegTensor<float> maxBroadcastReg;
    RegTensor<float> expReg;
    RegTensor<float> expSumReg;

    MaskReg allMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg oneMask = CreateMask<float, MaskPattern::VL1>();
    uint32_t remaining = validN;
    MaskReg validMask = UpdateMask<float, RegTraitNumOne>(remaining);

    LoadAlign<float, LoadDist::DIST_NORM>(sumSquareReg, statAddr + SUM_SQUARE_OFFSET);
    LoadAlign<float, LoadDist::DIST_NORM>(dotReg, statAddr + DOT_OFFSET);
    Muls<float, float, MaskMergeMode::ZEROING>(sumSquareReg, sumSquareReg, reciprocalD, validMask);
    Adds<float, float, MaskMergeMode::ZEROING>(sumSquareReg, sumSquareReg, eps, validMask);
    Sqrt<float, MaskMergeMode::ZEROING>(rmsReg, sumSquareReg, validMask);
    Div<float, &BARP_DIV_0ULP_FTZ_TRUE_MODE>(zReg, dotReg, rmsReg, validMask);

    Reduce<ReduceType::MAX, float, float, MaskMergeMode::ZEROING>(maxReg, zReg, validMask);
    Duplicate<float, HighLowPart::LOWEST, MaskMergeMode::ZEROING>(maxBroadcastReg, maxReg, allMask);
    Sub<float, MaskMergeMode::ZEROING>(expReg, zReg, maxBroadcastReg, validMask);
    Exp<float, MaskMergeMode::ZEROING>(expReg, expReg, validMask);
    Reduce<ReduceType::SUM, float, float, MaskMergeMode::ZEROING>(expSumReg, expReg, validMask);

    // E reuses the physical dot buffer after Z no longer needs to be retained.
    StoreAlign<float, StoreDist::DIST_NORM>(statAddr + DOT_OFFSET, expReg, validMask);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + MAX_OFFSET, maxReg, oneMask);
    StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + SUM_OFFSET, expSumReg, oneMask);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

template <typename INPUT_T, bool FIRST_BLOCK>
__simd_vf__ inline void WeightedAccumulate(__ubuf__ INPUT_T *vAddr, __ubuf__ float *weightAddr,
                                           __ubuf__ float *outputAddr, uint32_t validD)
{
    RegTensor<float> vReg;
    RegTensor<float> weightReg;
    RegTensor<float> outputReg;
    LoadAlign<float, LoadDist::DIST_BRC_B32>(weightReg, weightAddr);

    uint32_t remaining = validD;
    const uint16_t loopCount = static_cast<uint16_t>((validD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
    for (uint16_t loop = 0; loop < loopCount; ++loop) {
        MaskReg validMask = UpdateMask<float, RegTraitNumOne>(remaining);
        const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
        LoadInputAsFloat<INPUT_T>(vReg, vAddr + offset, validMask);
        if constexpr (FIRST_BLOCK) {
            Mul<float, MaskMergeMode::ZEROING>(outputReg, vReg, weightReg, validMask);
        } else {
            LoadAlign<float, LoadDist::DIST_NORM>(outputReg, outputAddr + offset);
            MulAddDst<float, MaskMergeMode::ZEROING>(outputReg, vReg, weightReg, validMask);
        }
        StoreAlign<float, StoreDist::DIST_NORM>(outputAddr + offset, outputReg, validMask);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

template <typename INPUT_T>
__simd_vf__ inline void CopySingleBlock(__ubuf__ INPUT_T *vAddr, __ubuf__ float *outputAddr, uint32_t validD)
{
    RegTensor<float> vReg;
    uint32_t remaining = validD;
    const uint16_t loopCount = static_cast<uint16_t>((validD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
    for (uint16_t loop = 0; loop < loopCount; ++loop) {
        MaskReg validMask = UpdateMask<float, RegTraitNumOne>(remaining);
        const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
        LoadInputAsFloat<INPUT_T>(vReg, vAddr + offset, validMask);
        StoreAlign<float, StoreDist::DIST_NORM>(outputAddr + offset, vReg, validMask);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

template <typename INPUT_T, bool FIRST_BLOCK, bool SINGLE_BLOCK>
__simd_vf__ inline void WeightedAccumulateTwoRows(__ubuf__ INPUT_T *vAddr,
                                                    __ubuf__ float *weight0Addr,
                                                    __ubuf__ float *weight1Addr,
                                                    __ubuf__ float *output0Addr,
                                                    __ubuf__ float *output1Addr,
                                                    uint32_t validD)
{
    RegTensor<float> vReg;
    RegTensor<float> weight0Reg;
    RegTensor<float> weight1Reg;
    RegTensor<float> output0Reg;
    RegTensor<float> output1Reg;
    if constexpr (!SINGLE_BLOCK) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(weight0Reg, weight0Addr);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(weight1Reg, weight1Addr);
    }
    uint32_t remaining = validD;
    const uint16_t loopCount = static_cast<uint16_t>((validD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
    for (uint16_t loop = 0; loop < loopCount; ++loop) {
        MaskReg validMask = UpdateMask<float, RegTraitNumOne>(remaining);
        const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
        LoadInputAsFloat<INPUT_T>(vReg, vAddr + offset, validMask);
        if constexpr (SINGLE_BLOCK) {
            output0Reg = vReg;
            output1Reg = vReg;
        } else if constexpr (FIRST_BLOCK) {
            Mul<float, MaskMergeMode::ZEROING>(output0Reg, vReg, weight0Reg, validMask);
            Mul<float, MaskMergeMode::ZEROING>(output1Reg, vReg, weight1Reg, validMask);
        } else {
            LoadAlign<float, LoadDist::DIST_NORM>(output0Reg, output0Addr + offset);
            LoadAlign<float, LoadDist::DIST_NORM>(output1Reg, output1Addr + offset);
            MulAddDst<float, MaskMergeMode::ZEROING>(output0Reg, vReg, weight0Reg, validMask);
            MulAddDst<float, MaskMergeMode::ZEROING>(output1Reg, vReg, weight1Reg, validMask);
        }
        StoreAlign<float, StoreDist::DIST_NORM>(output0Addr + offset, output0Reg, validMask);
        StoreAlign<float, StoreDist::DIST_NORM>(output1Addr + offset, output1Reg, validMask);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

template <typename INPUT_T, bool FIRST_PAIR>
__simd_vf__ inline void WeightedAccumulateTwoRowsTwoN(__ubuf__ INPUT_T *v0Addr,
                                                       __ubuf__ INPUT_T *v1Addr,
                                                       __ubuf__ float *weight0Addr,
                                                       __ubuf__ float *weight1Addr,
                                                       __ubuf__ float *output0Addr,
                                                       __ubuf__ float *output1Addr,
                                                       uint32_t validD)
{
    RegTensor<float> v0Reg;
    RegTensor<float> v1Reg;
    RegTensor<float> weight00Reg;
    RegTensor<float> weight01Reg;
    RegTensor<float> weight10Reg;
    RegTensor<float> weight11Reg;
    RegTensor<float> output0Reg;
    RegTensor<float> output1Reg;
    LoadAlign<float, LoadDist::DIST_BRC_B32>(weight00Reg, weight0Addr);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(weight01Reg, weight0Addr + 1);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(weight10Reg, weight1Addr);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(weight11Reg, weight1Addr + 1);

    uint32_t remaining = validD;
    const uint16_t loopCount = static_cast<uint16_t>((validD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
    for (uint16_t loop = 0; loop < loopCount; ++loop) {
        MaskReg validMask = UpdateMask<float, RegTraitNumOne>(remaining);
        const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
        LoadInputAsFloat<INPUT_T>(v0Reg, v0Addr + offset, validMask);
        LoadInputAsFloat<INPUT_T>(v1Reg, v1Addr + offset, validMask);
        if constexpr (FIRST_PAIR) {
            Mul<float, MaskMergeMode::ZEROING>(output0Reg, v0Reg, weight00Reg, validMask);
            Mul<float, MaskMergeMode::ZEROING>(output1Reg, v0Reg, weight10Reg, validMask);
        } else {
            LoadAlign<float, LoadDist::DIST_NORM>(output0Reg, output0Addr + offset);
            LoadAlign<float, LoadDist::DIST_NORM>(output1Reg, output1Addr + offset);
            MulAddDst<float, MaskMergeMode::ZEROING>(output0Reg, v0Reg, weight00Reg, validMask);
            MulAddDst<float, MaskMergeMode::ZEROING>(output1Reg, v0Reg, weight10Reg, validMask);
        }
        MulAddDst<float, MaskMergeMode::ZEROING>(output0Reg, v1Reg, weight01Reg, validMask);
        MulAddDst<float, MaskMergeMode::ZEROING>(output1Reg, v1Reg, weight11Reg, validMask);
        StoreAlign<float, StoreDist::DIST_NORM>(output0Addr + offset, output0Reg, validMask);
        StoreAlign<float, StoreDist::DIST_NORM>(output1Addr + offset, output1Reg, validMask);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

template <typename INPUT_T, bool FIRST_GROUP>
__simd_vf__ inline void WeightedAccumulateTwoRowsFourN(__ubuf__ INPUT_T *vAddr,
                                                        uint32_t vPitch,
                                                        __ubuf__ float *weight0Addr,
                                                        __ubuf__ float *weight1Addr,
                                                        __ubuf__ float *output0Addr,
                                                        __ubuf__ float *output1Addr,
                                                        uint32_t validD)
{
    RegTensor<float> v0Reg;
    RegTensor<float> v1Reg;
    RegTensor<float> v2Reg;
    RegTensor<float> v3Reg;
    RegTensor<float> w00Reg;
    RegTensor<float> w01Reg;
    RegTensor<float> w02Reg;
    RegTensor<float> w03Reg;
    RegTensor<float> w10Reg;
    RegTensor<float> w11Reg;
    RegTensor<float> w12Reg;
    RegTensor<float> w13Reg;
    RegTensor<float> output0Reg;
    RegTensor<float> output1Reg;
    LoadAlign<float, LoadDist::DIST_BRC_B32>(w00Reg, weight0Addr);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(w01Reg, weight0Addr + 1);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(w02Reg, weight0Addr + 2);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(w03Reg, weight0Addr + 3);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(w10Reg, weight1Addr);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(w11Reg, weight1Addr + 1);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(w12Reg, weight1Addr + 2);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(w13Reg, weight1Addr + 3);

    uint32_t remaining = validD;
    const uint16_t loopCount = static_cast<uint16_t>((validD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
    for (uint16_t loop = 0; loop < loopCount; ++loop) {
        MaskReg validMask = UpdateMask<float, RegTraitNumOne>(remaining);
        const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
        LoadInputAsFloat<INPUT_T>(v0Reg, vAddr + offset, validMask);
        LoadInputAsFloat<INPUT_T>(v1Reg, vAddr + vPitch + offset, validMask);
        LoadInputAsFloat<INPUT_T>(v2Reg, vAddr + 2U * vPitch + offset, validMask);
        LoadInputAsFloat<INPUT_T>(v3Reg, vAddr + 3U * vPitch + offset, validMask);
        if constexpr (FIRST_GROUP) {
            Mul<float, MaskMergeMode::ZEROING>(output0Reg, v0Reg, w00Reg, validMask);
            Mul<float, MaskMergeMode::ZEROING>(output1Reg, v0Reg, w10Reg, validMask);
        } else {
            LoadAlign<float, LoadDist::DIST_NORM>(output0Reg, output0Addr + offset);
            LoadAlign<float, LoadDist::DIST_NORM>(output1Reg, output1Addr + offset);
            MulAddDst<float, MaskMergeMode::ZEROING>(output0Reg, v0Reg, w00Reg, validMask);
            MulAddDst<float, MaskMergeMode::ZEROING>(output1Reg, v0Reg, w10Reg, validMask);
        }
        MulAddDst<float, MaskMergeMode::ZEROING>(output0Reg, v1Reg, w01Reg, validMask);
        MulAddDst<float, MaskMergeMode::ZEROING>(output0Reg, v2Reg, w02Reg, validMask);
        MulAddDst<float, MaskMergeMode::ZEROING>(output0Reg, v3Reg, w03Reg, validMask);
        MulAddDst<float, MaskMergeMode::ZEROING>(output1Reg, v1Reg, w11Reg, validMask);
        MulAddDst<float, MaskMergeMode::ZEROING>(output1Reg, v2Reg, w12Reg, validMask);
        MulAddDst<float, MaskMergeMode::ZEROING>(output1Reg, v3Reg, w13Reg, validMask);
        StoreAlign<float, StoreDist::DIST_NORM>(output0Addr + offset, output0Reg, validMask);
        StoreAlign<float, StoreDist::DIST_NORM>(output1Addr + offset, output1Reg, validMask);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

} // namespace BlockAttnResPrepareVF

#endif // BLOCK_ATTN_RES_PREPARE_VF_H
