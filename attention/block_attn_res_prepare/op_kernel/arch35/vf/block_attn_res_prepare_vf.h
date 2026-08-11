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

#include "kernel_operator.h"

namespace BlockAttnResPrepareVF {

using namespace AscendC;
using MemType = AscendC::MicroAPI::MemType;

constexpr uint32_t FP32_REG_ELEMS = 64U;
constexpr uint32_t SUM_SQUARE_OFFSET = 0U;
constexpr uint32_t DOT_OFFSET = FP32_REG_ELEMS;
constexpr uint32_t MAX_OFFSET = 2U * FP32_REG_ELEMS;
constexpr uint32_t SUM_OFFSET = MAX_OFFSET + 8U;
constexpr float FP32_LOWEST_FINITE = -3.4028234663852886e+38F;

__simd_vf__ inline void InitializeEmptyOnlineSoftmax(__ubuf__ float *statAddr)
{
    MicroAPI::RegTensor<float> maxReg;
    MicroAPI::RegTensor<float> sumReg;
    MicroAPI::MaskReg oneMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();

    MicroAPI::Duplicate(maxReg, FP32_LOWEST_FINITE, oneMask);
    MicroAPI::Duplicate(sumReg, 1.0F, oneMask);
    MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + MAX_OFFSET, maxReg, oneMask);
    MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + SUM_OFFSET, sumReg, oneMask);
    MicroAPI::LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

__simd_vf__ inline void FillZero(__ubuf__ float *outputAddr, uint32_t validD)
{
    MicroAPI::RegTensor<float> zeroReg;
    MicroAPI::MaskReg allMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Duplicate(zeroReg, 0.0F, allMask);

    uint32_t remaining = validD;
    const uint16_t loopCount = static_cast<uint16_t>((validD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
    for (uint16_t loop = 0; loop < loopCount; ++loop) {
        MicroAPI::MaskReg validMask = MicroAPI::UpdateMask<float, MicroAPI::RegTraitNumOne>(remaining);
        const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset, zeroReg, validMask);
    }
    MicroAPI::LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

// These raw UB arguments are internal static byte offsets. UB offset zero is valid and must not be treated as nullptr.
template <bool FIRST_D_TILE>
__simd_vf__ inline void AccumulateSquareDot(__ubuf__ float *qAddr, __ubuf__ float *vAddr,
                                            __ubuf__ float *statAddr, uint32_t validD, uint32_t nIndex)
{
    MicroAPI::RegTensor<float> qReg;
    MicroAPI::RegTensor<float> vReg;
    MicroAPI::RegTensor<float> squareReg;
    MicroAPI::RegTensor<float> dotReg;
    MicroAPI::RegTensor<float> squareAccReg;
    MicroAPI::RegTensor<float> dotAccReg;
    MicroAPI::RegTensor<float> squareReduceReg;
    MicroAPI::RegTensor<float> dotReduceReg;

    MicroAPI::MaskReg allMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg oneMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
    MicroAPI::Duplicate(squareAccReg, 0.0F, allMask);
    MicroAPI::Duplicate(dotAccReg, 0.0F, allMask);

    uint32_t remaining = validD;
    const uint16_t loopCount = static_cast<uint16_t>((validD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
    for (uint16_t loop = 0; loop < loopCount; ++loop) {
        MicroAPI::MaskReg validMask = MicroAPI::UpdateMask<float, MicroAPI::RegTraitNumOne>(remaining);
        const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(qReg, qAddr + offset);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vReg, vAddr + offset);
        MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(squareReg, vReg, vReg, validMask);
        MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(dotReg, qReg, vReg, validMask);
        MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(squareAccReg, squareAccReg, squareReg, allMask);
        MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(dotAccReg, dotAccReg, dotReg, allMask);
    }

    MicroAPI::Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(
        squareReduceReg, squareAccReg, allMask);
    MicroAPI::Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(
        dotReduceReg, dotAccReg, allMask);

    if constexpr (!FIRST_D_TILE) {
        MicroAPI::RegTensor<float> oldValueReg;
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(oldValueReg,
                                                                     statAddr + SUM_SQUARE_OFFSET + nIndex);
        MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(squareReduceReg, squareReduceReg, oldValueReg,
                                                               oneMask);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(oldValueReg, statAddr + DOT_OFFSET + nIndex);
        MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(dotReduceReg, dotReduceReg, oldValueReg, oneMask);
    }

    MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
        statAddr + SUM_SQUARE_OFFSET + nIndex, squareReduceReg, oneMask);
    MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
        statAddr + DOT_OFFSET + nIndex, dotReduceReg, oneMask);
    MicroAPI::LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

template <bool FIRST_D_TILE>
__simd_vf__ inline void AccumulateSquare(__ubuf__ float *vAddr, __ubuf__ float *sumSquareAddr,
                                         uint32_t validD, uint32_t nIndex)
{
    MicroAPI::RegTensor<float> vReg;
    MicroAPI::RegTensor<float> squareReg;
    MicroAPI::RegTensor<float> squareAccReg;
    MicroAPI::RegTensor<float> squareReduceReg;
    MicroAPI::MaskReg allMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg oneMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
    MicroAPI::Duplicate(squareAccReg, 0.0F, allMask);

    uint32_t remaining = validD;
    const uint16_t loopCount = static_cast<uint16_t>((validD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
    for (uint16_t loop = 0; loop < loopCount; ++loop) {
        MicroAPI::MaskReg validMask = MicroAPI::UpdateMask<float, MicroAPI::RegTraitNumOne>(remaining);
        const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vReg, vAddr + offset);
        MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(squareReg, vReg, vReg, validMask);
        MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(squareAccReg, squareAccReg, squareReg, allMask);
    }
    MicroAPI::Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(
        squareReduceReg, squareAccReg, allMask);
    if constexpr (!FIRST_D_TILE) {
        MicroAPI::RegTensor<float> oldValueReg;
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(oldValueReg, sumSquareAddr + nIndex);
        MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(squareReduceReg, squareReduceReg, oldValueReg,
                                                               oneMask);
    }
    MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(sumSquareAddr + nIndex,
                                                                             squareReduceReg, oneMask);
    MicroAPI::LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

__simd_vf__ inline void FinalizeSingleBlock(__ubuf__ float *statAddr, float reciprocalD, float epsilon)
{
    MicroAPI::RegTensor<float> sumSquareReg;
    MicroAPI::RegTensor<float> dotReg;
    MicroAPI::RegTensor<float> rmsReg;
    MicroAPI::RegTensor<float> zReg;
    MicroAPI::RegTensor<float> oneReg;
    MicroAPI::MaskReg oneMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();

    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(sumSquareReg, statAddr + SUM_SQUARE_OFFSET);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(dotReg, statAddr + DOT_OFFSET);
    MicroAPI::Muls<float, float, MicroAPI::MaskMergeMode::ZEROING>(sumSquareReg, sumSquareReg, reciprocalD, oneMask);
    MicroAPI::Adds<float, float, MicroAPI::MaskMergeMode::ZEROING>(sumSquareReg, sumSquareReg, epsilon, oneMask);
    MicroAPI::Sqrt<float, MicroAPI::MaskMergeMode::ZEROING>(rmsReg, sumSquareReg, oneMask);
    MicroAPI::Div<float, MicroAPI::MaskMergeMode::ZEROING>(zReg, dotReg, rmsReg, oneMask);
    MicroAPI::Duplicate(oneReg, 1.0F, oneMask);

    MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + MAX_OFFSET, zReg, oneMask);
    MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + SUM_OFFSET, oneReg, oneMask);
    MicroAPI::LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

__simd_vf__ inline void FinalizeSoftmax(__ubuf__ float *statAddr, uint32_t validN, float reciprocalD, float epsilon)
{
    MicroAPI::RegTensor<float> sumSquareReg;
    MicroAPI::RegTensor<float> dotReg;
    MicroAPI::RegTensor<float> rmsReg;
    MicroAPI::RegTensor<float> zReg;
    MicroAPI::RegTensor<float> maxReg;
    MicroAPI::RegTensor<float> maxBroadcastReg;
    MicroAPI::RegTensor<float> expReg;
    MicroAPI::RegTensor<float> expSumReg;

    MicroAPI::MaskReg allMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg oneMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
    uint32_t remaining = validN;
    MicroAPI::MaskReg validMask = MicroAPI::UpdateMask<float, MicroAPI::RegTraitNumOne>(remaining);

    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(sumSquareReg, statAddr + SUM_SQUARE_OFFSET);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(dotReg, statAddr + DOT_OFFSET);
    MicroAPI::Muls<float, float, MicroAPI::MaskMergeMode::ZEROING>(sumSquareReg, sumSquareReg, reciprocalD,
                                                                   validMask);
    MicroAPI::Adds<float, float, MicroAPI::MaskMergeMode::ZEROING>(sumSquareReg, sumSquareReg, epsilon, validMask);
    MicroAPI::Sqrt<float, MicroAPI::MaskMergeMode::ZEROING>(rmsReg, sumSquareReg, validMask);
    MicroAPI::Div<float, MicroAPI::MaskMergeMode::ZEROING>(zReg, dotReg, rmsReg, validMask);

    MicroAPI::Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(maxReg, zReg,
                                                                                                validMask);
    MicroAPI::Duplicate<float, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>(
        maxBroadcastReg, maxReg, allMask);
    MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(expReg, zReg, maxBroadcastReg, validMask);
    MicroAPI::Exp<float, MicroAPI::MaskMergeMode::ZEROING>(expReg, expReg, validMask);
    MicroAPI::Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(expSumReg, expReg,
                                                                                                validMask);

    // E reuses the physical dot buffer after Z no longer needs to be retained.
    MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(statAddr + DOT_OFFSET, expReg, validMask);
    MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + MAX_OFFSET, maxReg, oneMask);
    MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(statAddr + SUM_OFFSET, expSumReg,
                                                                             oneMask);
    MicroAPI::LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

__simd_vf__ inline void FinalizeCubeSoftmax(__ubuf__ float *sumSquareAddr, __ubuf__ float *dotAddr,
                                            __ubuf__ float *maxAddr, __ubuf__ float *sumAddr, uint32_t validN,
                                            uint32_t nAlign, float reciprocalD, float epsilon)
{
    MicroAPI::RegTensor<float> sumSquareReg;
    MicroAPI::RegTensor<float> dotReg;
    MicroAPI::RegTensor<float> rmsReg;
    MicroAPI::RegTensor<float> zReg;
    MicroAPI::RegTensor<float> maxReg;
    MicroAPI::RegTensor<float> maxBroadcastReg;
    MicroAPI::RegTensor<float> expReg;
    MicroAPI::RegTensor<float> expSumReg;
    MicroAPI::RegTensor<float> zeroReg;

    MicroAPI::MaskReg allMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg oneMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
    uint32_t validRemaining = validN;
    uint32_t alignRemaining = nAlign;
    MicroAPI::MaskReg validMask = MicroAPI::UpdateMask<float, MicroAPI::RegTraitNumOne>(validRemaining);
    MicroAPI::MaskReg alignMask = MicroAPI::UpdateMask<float, MicroAPI::RegTraitNumOne>(alignRemaining);

    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(sumSquareReg, sumSquareAddr);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(dotReg, dotAddr);
    MicroAPI::Muls<float, float, MicroAPI::MaskMergeMode::ZEROING>(sumSquareReg, sumSquareReg, reciprocalD,
                                                                   validMask);
    MicroAPI::Adds<float, float, MicroAPI::MaskMergeMode::ZEROING>(sumSquareReg, sumSquareReg, epsilon, validMask);
    MicroAPI::Sqrt<float, MicroAPI::MaskMergeMode::ZEROING>(rmsReg, sumSquareReg, validMask);
    MicroAPI::Div<float, MicroAPI::MaskMergeMode::ZEROING>(zReg, dotReg, rmsReg, validMask);
    MicroAPI::Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(maxReg, zReg,
                                                                                                validMask);
    MicroAPI::Duplicate<float, MicroAPI::HighLowPart::LOWEST, MicroAPI::MaskMergeMode::ZEROING>(
        maxBroadcastReg, maxReg, allMask);
    MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(expReg, zReg, maxBroadcastReg, validMask);
    MicroAPI::Exp<float, MicroAPI::MaskMergeMode::ZEROING>(expReg, expReg, validMask);
    MicroAPI::Reduce<MicroAPI::ReduceType::SUM, float, float, MicroAPI::MaskMergeMode::ZEROING>(expSumReg, expReg,
                                                                                                validMask);

    MicroAPI::Duplicate(zeroReg, 0.0F, allMask);
    MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(dotAddr, zeroReg, alignMask);
    MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(dotAddr, expReg, validMask);
    MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(maxAddr, maxReg, oneMask);
    MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(sumAddr, expSumReg, oneMask);
    MicroAPI::LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

template <bool FIRST_BLOCK>
__simd_vf__ inline void WeightedAccumulate(__ubuf__ float *vAddr, __ubuf__ float *weightAddr,
                                           __ubuf__ float *outputAddr, uint32_t validD)
{
    MicroAPI::RegTensor<float> vReg;
    MicroAPI::RegTensor<float> weightReg;
    MicroAPI::RegTensor<float> outputReg;
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(weightReg, weightAddr);

    uint32_t remaining = validD;
    const uint16_t loopCount = static_cast<uint16_t>((validD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
    for (uint16_t loop = 0; loop < loopCount; ++loop) {
        MicroAPI::MaskReg validMask = MicroAPI::UpdateMask<float, MicroAPI::RegTraitNumOne>(remaining);
        const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vReg, vAddr + offset);
        if constexpr (FIRST_BLOCK) {
            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(outputReg, vReg, weightReg, validMask);
        } else {
            MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(outputReg, outputAddr + offset);
            MicroAPI::MulAddDst<float, MicroAPI::MaskMergeMode::ZEROING>(outputReg, vReg, weightReg, validMask);
        }
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset, outputReg, validMask);
    }
    MicroAPI::LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

__simd_vf__ inline void CopySingleBlock(__ubuf__ float *vAddr, __ubuf__ float *outputAddr, uint32_t validD)
{
    MicroAPI::RegTensor<float> vReg;
    uint32_t remaining = validD;
    const uint16_t loopCount = static_cast<uint16_t>((validD + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS);
    for (uint16_t loop = 0; loop < loopCount; ++loop) {
        MicroAPI::MaskReg validMask = MicroAPI::UpdateMask<float, MicroAPI::RegTraitNumOne>(remaining);
        const uint32_t offset = static_cast<uint32_t>(loop) * FP32_REG_ELEMS;
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vReg, vAddr + offset);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset, vReg, validMask);
    }
    MicroAPI::LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

} // namespace BlockAttnResPrepareVF

#endif // BLOCK_ATTN_RES_PREPARE_VF_H
