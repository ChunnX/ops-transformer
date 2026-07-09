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
 * \file stem_indexer_vector1.h
 * \brief
 */
#ifndef STEM_INDEXER_VECTOR1_H
#define STEM_INDEXER_VECTOR1_H

#include "kernel_operator.h"
#include "../stem_indexer_common.h"

namespace vector1 {

template <typename T>
struct FloatSortTraits;

// fp32
template <>
struct FloatSortTraits<float> {
    using UInt = uint32_t;
    static constexpr UInt ZERO      = 0x00000000;
    static constexpr UInt SIGN_MASK = 0x80000000;
    static constexpr UInt NAN_MASK  = 0x7FC00000;
    static constexpr UInt ALL_ONE   = 0xFFFFFFFF;
};

// bf16
template <>
struct FloatSortTraits<bfloat16_t> {
    using UInt = uint16_t;
    static constexpr UInt ZERO      = 0x0000;
    static constexpr UInt SIGN_MASK = 0x8000;
    static constexpr UInt NAN_MASK  = 0x7FC0;
    static constexpr UInt ALL_ONE   = 0xFFFF;
};

template <typename FloatT>
struct FloatSortConstCtx {
    using Traits = FloatSortTraits<FloatT>;
    using UInt   = typename Traits::UInt;
    AscendC::MicroAPI::RegTensor<UInt> zeros;
    AscendC::MicroAPI::RegTensor<UInt> allOne;
    AscendC::MicroAPI::RegTensor<UInt> signMask;
    AscendC::MicroAPI::RegTensor<UInt> nan;
};


template <typename FloatT>
__simd_callee__ inline void InitFloatSortConstCtx(FloatSortConstCtx<FloatT>& ctx, AscendC::MicroAPI::MaskReg& maskAll)
{
    using Traits = FloatSortTraits<FloatT>;
    AscendC::MicroAPI::Duplicate(ctx.zeros,    Traits::ZERO,      maskAll);
    AscendC::MicroAPI::Duplicate(ctx.allOne,   Traits::ALL_ONE,   maskAll);
    AscendC::MicroAPI::Duplicate(ctx.signMask, Traits::SIGN_MASK, maskAll);
    AscendC::MicroAPI::Duplicate(ctx.nan,      Traits::NAN_MASK,  maskAll);
}

template <typename FloatT>
__simd_callee__ inline void FloatToSortableKey(
    AscendC::MicroAPI::RegTensor<typename FloatSortTraits<FloatT>::UInt>& outKey,
    AscendC::MicroAPI::RegTensor<FloatT>& inVal,
    FloatSortConstCtx<FloatT>& ctx,
    AscendC::MicroAPI::MaskReg& maskAll)
{
    using Traits = FloatSortTraits<FloatT>;
    using UInt   = typename Traits::UInt;

    AscendC::MicroAPI::RegTensor<UInt> regTemp;
    AscendC::MicroAPI::RegTensor<UInt> regMask;
    AscendC::MicroAPI::MaskReg regSelectNan;
    AscendC::MicroAPI::MaskReg regSelectSign;

    auto& inBits = (AscendC::MicroAPI::RegTensor<UInt>&)inVal;

    // 1. NaN check
    AscendC::MicroAPI::Compare<UInt, CMPMODE::EQ>(regSelectNan, inBits, ctx.nan, maskAll);

    // 2. NaN -> ALL_ONE
    AscendC::MicroAPI::Select(outKey, ctx.allOne, inBits, regSelectNan);

    // 3. sign bit
    AscendC::MicroAPI::And(regTemp, outKey, ctx.signMask, maskAll);

    AscendC::MicroAPI::Compare<UInt, CMPMODE::GT>(regSelectSign, regTemp, ctx.zeros, maskAll);

    // 4. xor mask
    AscendC::MicroAPI::Select(regMask, ctx.allOne, ctx.signMask, regSelectSign);
    AscendC::MicroAPI::Xor(outKey, outKey, regMask, maskAll);
}

template <typename FloatT>
__simd_callee__ inline void FloatX2ToSortableKey(
    AscendC::MicroAPI::RegTensor<typename FloatSortTraits<FloatT>::UInt>& outKey0,
    AscendC::MicroAPI::RegTensor<typename FloatSortTraits<FloatT>::UInt>& outKey1,
    AscendC::MicroAPI::RegTensor<FloatT>& inVal0,
    AscendC::MicroAPI::RegTensor<FloatT>& inVal1,
    FloatSortConstCtx<FloatT>& ctx,
    AscendC::MicroAPI::MaskReg& maskAll)
{
    using Traits = FloatSortTraits<FloatT>;
    using UInt   = typename Traits::UInt;

    AscendC::MicroAPI::RegTensor<UInt> regTemp[2];
    AscendC::MicroAPI::RegTensor<UInt> regMask[2];
    AscendC::MicroAPI::MaskReg regSelectNan[2];
    AscendC::MicroAPI::MaskReg regSelectSign[2];

    auto& inBits0 = (AscendC::MicroAPI::RegTensor<UInt>&)inVal0;
    auto& inBits1 = (AscendC::MicroAPI::RegTensor<UInt>&)inVal1;

    // 1. NaN check
    AscendC::MicroAPI::Compare<UInt, CMPMODE::EQ>(regSelectNan[0], inBits0, ctx.nan, maskAll);
    AscendC::MicroAPI::Compare<UInt, CMPMODE::EQ>(regSelectNan[1], inBits1, ctx.nan, maskAll);

    // 2. NaN -> ALL_ONE
    AscendC::MicroAPI::Select(outKey0, ctx.allOne, inBits0, regSelectNan[0]);
    AscendC::MicroAPI::Select(outKey1, ctx.allOne, inBits1, regSelectNan[1]);

    // 3. sign bit
    AscendC::MicroAPI::And(regTemp[0], outKey0, ctx.signMask, maskAll);
    AscendC::MicroAPI::And(regTemp[1], outKey1, ctx.signMask, maskAll);

    AscendC::MicroAPI::Compare<UInt, CMPMODE::GT>(regSelectSign[0], regTemp[0], ctx.zeros, maskAll);
    AscendC::MicroAPI::Compare<UInt, CMPMODE::GT>(regSelectSign[1], regTemp[1], ctx.zeros, maskAll);

    // 4. xor mask
    AscendC::MicroAPI::Select(regMask[0], ctx.allOne, ctx.signMask, regSelectSign[0]);
    AscendC::MicroAPI::Select(regMask[1], ctx.allOne, ctx.signMask, regSelectSign[1]);
    AscendC::MicroAPI::Xor(outKey0, outKey0, regMask[0], maskAll);
    AscendC::MicroAPI::Xor(outKey1, outKey1, regMask[1], maskAll);
}

template <typename SCORE_T>
__simd_callee__ inline void PadMrgRowRangeVFCore(__ubuf__ SCORE_T* rowBuf,
                                                const uint32_t padStart,
                                                const uint32_t padEnd)
{
    constexpr uint32_t VF_ELEMS = 256U / sizeof(SCORE_T);
    // Arange only supports signed index tensors; later casts reuse the same lane bits for SCORE_T comparisons.
    using IDX_T = std::conditional_t<std::is_same_v<SCORE_T, uint16_t>, int16_t, int32_t>;
    const uint32_t blockStart = (padStart / VF_ELEMS) * VF_ELEMS;
    const uint32_t blockEnd = ((padEnd + VF_ELEMS - 1U) / VF_ELEMS) * VF_ELEMS;

    MicroAPI::MaskReg maskAll = AscendC::MicroAPI::CreateMask<SCORE_T, AscendC::MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskGeStart;
    MicroAPI::MaskReg maskLtEnd;
    MicroAPI::MaskReg maskPad;
    MicroAPI::RegTensor<IDX_T> regIdx;
    MicroAPI::RegTensor<SCORE_T> regOldValue;
    MicroAPI::RegTensor<SCORE_T> regZero;
    MicroAPI::RegTensor<SCORE_T> regOutValue;
    MicroAPI::Duplicate(regZero, (SCORE_T)0, maskAll);

    for (uint32_t blockBase = blockStart; blockBase < blockEnd; blockBase += VF_ELEMS) {
        MicroAPI::Arange(regIdx, static_cast<IDX_T>(blockBase));
        MicroAPI::LoadAlign<SCORE_T, MicroAPI::LoadDist::DIST_NORM>(regOldValue, rowBuf + blockBase);
        MicroAPI::Compares<SCORE_T, CMPMODE::GE>(maskGeStart, (MicroAPI::RegTensor<SCORE_T>&)regIdx,
                                                 (SCORE_T)padStart, maskAll);
        MicroAPI::Compares<SCORE_T, CMPMODE::LT>(maskLtEnd, (MicroAPI::RegTensor<SCORE_T>&)regIdx,
                                                 (SCORE_T)padEnd, maskAll);
        MicroAPI::And(maskPad, maskGeStart, maskLtEnd, maskAll);
        MicroAPI::Select(regOutValue, regZero, regOldValue, maskPad);
        MicroAPI::StoreAlign<SCORE_T, MicroAPI::StoreDist::DIST_NORM>(rowBuf + blockBase, regOutValue, maskAll);
    }
}

__simd_callee__ inline uint32_t PadMrgAlignUp(const uint32_t value, const uint32_t align)
{
    return ((value + align - 1U) / align) * align;
}

__simd_vf__ inline void PadMrgSingleRowRangeVFImpl(__ubuf__ uint32_t* rowBuf,
                                                  const uint32_t padStart,
                                                  const uint32_t padEnd)
{
    PadMrgRowRangeVFCore(rowBuf, padStart, padEnd);
}

template <typename SCORE_T>
__simd_vf__ inline void PadMrgOneRowFirstS2RangeVFImpl(__ubuf__ SCORE_T* rowBuf,
                                                      const uint32_t topkSelectNum,
                                                      const uint32_t s2BlockValidLen,
                                                      const uint32_t s2BlockValidLenAlign)
{
    const uint32_t topkSelectNumAlign = PadMrgAlignUp(topkSelectNum, 256U);
    PadMrgRowRangeVFCore(rowBuf,
                         topkSelectNumAlign + s2BlockValidLen,
                         topkSelectNumAlign + s2BlockValidLenAlign);
}

template <typename SCORE_T>
__simd_vf__ inline void PadMrgOneRowNotFirstS2RangeVFImpl(__ubuf__ SCORE_T* rowBuf,
                                                         const uint32_t topkSelectNum,
                                                         const uint32_t s2BlockValidLen,
                                                         const uint32_t s2BlockValidLenAlign)
{
    const uint32_t topkSelectNumAlign = PadMrgAlignUp(topkSelectNum, 256U);
    PadMrgRowRangeVFCore(rowBuf, topkSelectNum, topkSelectNumAlign);
    PadMrgRowRangeVFCore(rowBuf,
                         topkSelectNumAlign + s2BlockValidLen,
                         topkSelectNumAlign + s2BlockValidLenAlign);
}

__simd_vf__ inline void PadMrgRowsRangeVFImpl(__ubuf__ uint32_t* mrgValueBuf,
                                             const uint32_t rowStride,
                                             const uint32_t actS1Size,
                                             const uint32_t s1Idx,
                                             const uint32_t validGSize,
                                             const uint32_t padStart,
                                             const uint32_t padEnd)
{
    for (uint32_t gIdx = 0; gIdx < validGSize; gIdx++) {
        const uint32_t mInnerIdx = gIdx * actS1Size + s1Idx;
        PadMrgRowRangeVFCore(mrgValueBuf + mInnerIdx * rowStride, padStart, padEnd);
    }
}

template <typename SCORE_T>
__simd_vf__ inline void PadMrgRowsInitBlockVFImpl(__ubuf__ SCORE_T* mrgValueBuf,
                                                 const uint32_t rowStride,
                                                 const uint32_t rowNum,
                                                 const uint32_t topkSelectNumAlign,
                                                 const uint32_t initialBlocks)
{
    using IDX_T = std::conditional_t<std::is_same_v<SCORE_T, uint16_t>, int16_t, int32_t>;
    MicroAPI::MaskReg maskAll = AscendC::MicroAPI::CreateMask<SCORE_T, AscendC::MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskInit;
    MicroAPI::RegTensor<IDX_T> regIdx;
    MicroAPI::RegTensor<SCORE_T> regZero;
    MicroAPI::Duplicate(regZero, (SCORE_T)0, maskAll);
    // Arange only supports signed index tensors; later casts reuse the same lane bits for SCORE_T comparisons.
    MicroAPI::Arange(regIdx, static_cast<IDX_T>(0));
    MicroAPI::Compares<SCORE_T, CMPMODE::LT>(maskInit, (MicroAPI::RegTensor<SCORE_T>&)regIdx,
                                             (SCORE_T)initialBlocks, maskAll);

    for (uint32_t rowIdx = 0; rowIdx < rowNum; rowIdx++) {
        __ubuf__ SCORE_T* rowBuf = mrgValueBuf + rowIdx * rowStride + topkSelectNumAlign;
        MicroAPI::StoreAlign<SCORE_T, MicroAPI::StoreDist::DIST_NORM>(rowBuf, regZero, maskInit);
    }
}

__aicore__ inline void PadMrgSingleRowRangeVF(const LocalTensor<uint32_t> &mrgValueLocal,
                                             const uint32_t rowOffset,
                                             const uint32_t padStart,
                                             const uint32_t padEnd)
{
    if (padEnd <= padStart) {
        return;
    }

    __ubuf__ uint32_t *mrgValueBuf = (__ubuf__ uint32_t *)mrgValueLocal.GetPhyAddr();
    PadMrgSingleRowRangeVFImpl(mrgValueBuf + rowOffset, padStart, padEnd);
}

template <typename SCORE_T>
__aicore__ inline void PadMrgRowsInitBlockVF(LocalTensor<SCORE_T> &mrgValueLocal,
                                             const uint32_t rowStride,
                                             const uint32_t rowNum,
                                             const uint32_t topkSelectNumAlign,
                                             const uint32_t initialBlocks)
{
    if (rowNum == 0U || initialBlocks == 0U) {
        return;
    }

    __ubuf__ SCORE_T *mrgValueBuf = (__ubuf__ SCORE_T *)mrgValueLocal.GetPhyAddr();
    PadMrgRowsInitBlockVFImpl<SCORE_T>(mrgValueBuf, rowStride, rowNum, topkSelectNumAlign, initialBlocks);
}

template <typename SCORE_T>
__aicore__ inline void PadMrgFourRowsFirstS2RangeVF(const LocalTensor<SCORE_T> &mrgValueLocal,
                                                    const SICommon::RowIdx4 &rowIdx4,
                                                    const SICommon::TopkNum4 &topkNum4,
                                                    const SICommon::S2ValidLen4 &s2ValidLen4,
                                                    const uint32_t batchRowNum,
                                                    const uint32_t rowStride,
                                                    const uint32_t trunkLen)
{
    if (batchRowNum == 0U) {
        return;
    }

    __ubuf__ SCORE_T *mrgValueBuf = (__ubuf__ SCORE_T *)mrgValueLocal.GetPhyAddr();
    for (uint32_t rowIdx = 0; rowIdx < batchRowNum; rowIdx++) {
        const uint32_t mInnerIdx = SICommon::GetLane(rowIdx4, rowIdx);
        const uint32_t topkSelectNum = SICommon::GetLane(topkNum4, rowIdx);
        const uint32_t s2BlockValidLen = SICommon::GetLane(s2ValidLen4, rowIdx);
        const uint32_t s2BlockValidLenAlign = SICommon::Align(s2BlockValidLen, trunkLen);
        const uint32_t rowOffset = mInnerIdx * rowStride;
        PadMrgOneRowFirstS2RangeVFImpl(mrgValueBuf + rowOffset,
                                       topkSelectNum, s2BlockValidLen,
                                       s2BlockValidLenAlign);
    }
}

template <typename SCORE_T>
__aicore__ inline void PadMrgFourRowsNotFirstS2RangeVF(const LocalTensor<SCORE_T> &mrgValueLocal,
                                                       const SICommon::RowIdx4 &rowIdx4,
                                                       const SICommon::TopkNum4 &topkNum4,
                                                       const SICommon::S2ValidLen4 &s2ValidLen4,
                                                       const uint32_t batchRowNum,
                                                       const uint32_t rowStride,
                                                       const uint32_t trunkLen)
{
    if (batchRowNum == 0U) {
        return;
    }

    __ubuf__ SCORE_T *mrgValueBuf = (__ubuf__ SCORE_T *)mrgValueLocal.GetPhyAddr();
    for (uint32_t rowIdx = 0; rowIdx < batchRowNum; rowIdx++) {
        const uint32_t mInnerIdx = SICommon::GetLane(rowIdx4, rowIdx);
        const uint32_t topkSelectNum = SICommon::GetLane(topkNum4, rowIdx);
        const uint32_t s2BlockValidLen = SICommon::GetLane(s2ValidLen4, rowIdx);
        const uint32_t s2BlockValidLenAlign = 256U;
        const uint32_t rowOffset = mInnerIdx * rowStride;
        PadMrgOneRowNotFirstS2RangeVFImpl(mrgValueBuf + rowOffset,
                                          topkSelectNum, s2BlockValidLen,
                                          s2BlockValidLenAlign);
    }
}

__aicore__ inline void PadMrgRowsRangeVF(const LocalTensor<uint32_t> &mrgValueLocal,
                                        const uint32_t rowStride,
                                        const uint32_t actS1Size,
                                        const uint32_t s1Idx,
                                        const uint32_t curAivGS1ProcNum,
                                        const uint32_t curAivGSize,
                                        const uint32_t padStart,
                                        const uint32_t padEnd)
{
    if (padEnd <= padStart || s1Idx >= curAivGS1ProcNum) {
        return;
    }

    const uint32_t s1ValidGSize = CeilDiv(curAivGS1ProcNum - s1Idx, actS1Size);
    const uint32_t validGSize = (curAivGSize < s1ValidGSize) ? curAivGSize : s1ValidGSize;
    __ubuf__ uint32_t *mrgValueBuf = (__ubuf__ uint32_t *)mrgValueLocal.GetPhyAddr();
    PadMrgRowsRangeVFImpl(mrgValueBuf, rowStride, actS1Size, s1Idx, validGSize, padStart, padEnd);
}

// float in uint32 out
template <typename QK_T>
__simd_vf__ inline void MulRSquareAndAddVBiasVFImpl(__ubuf__ uint32_t* outBuf,
                                                    __ubuf__ QK_T* qkBuf,
                                                    __ubuf__ float* vBiasBuf,
                                                    const float rSquare,
                                                    const int gS1BasePerVecSize_,
                                                    const int s2BaseSize,
                                                    const int mrgValueLen)
{
    MicroAPI::RegTensor<float> regvBias[4];
    MicroAPI::RegTensor<float> regQK[4];
    MicroAPI::RegTensor<float> regQKMul[4];
    MicroAPI::RegTensor<float> regQKBias[4];
    MicroAPI::RegTensor<uint32_t> regOut[4];
    MicroAPI::MaskReg maskAllB32 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

    // float -> uint32相关参数结构体
    FloatSortConstCtx<float> fp32Ctx;
    InitFloatSortConstCtx(fp32Ctx, maskAllB32);

    // s2BaseSize固定为256时，搬运vbias写在外面性能最佳
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(regvBias[0], vBiasBuf);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(regvBias[1], vBiasBuf + 64);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(regvBias[2], vBiasBuf + 128);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(regvBias[3], vBiasBuf + 192);

    // 一次处理s2BaseSize 256个元素
    for (uint16_t i = 0; i < (uint16_t)(gS1BasePerVecSize_); ++i) {
        MicroAPI::LoadAlign<float>(regQK[0], qkBuf + s2BaseSize * i); // RowStride是128, 行都落在一个bank上
        MicroAPI::LoadAlign<float>(regQK[1], qkBuf + s2BaseSize * i + 64);
        MicroAPI::LoadAlign<float>(regQK[2], qkBuf + s2BaseSize * i + 128);
        MicroAPI::LoadAlign<float>(regQK[3], qkBuf + s2BaseSize * i + 192);
        // qk * (stem_block_size / stem_stride) ^ 2
        MicroAPI::Muls(regQKMul[0], regQK[0], rSquare, maskAllB32);
        MicroAPI::Muls(regQKMul[1], regQK[1], rSquare, maskAllB32);
        MicroAPI::Muls(regQKMul[2], regQK[2], rSquare, maskAllB32);
        MicroAPI::Muls(regQKMul[3], regQK[3], rSquare, maskAllB32);
        // + vbias [1，s2BaseSize] 重复gS1BasePerVecSize_行
        MicroAPI::Add(regQKBias[0], regQKMul[0], regvBias[0], maskAllB32);
        MicroAPI::Add(regQKBias[1], regQKMul[1], regvBias[1], maskAllB32);
        MicroAPI::Add(regQKBias[2], regQKMul[2], regvBias[2], maskAllB32);
        MicroAPI::Add(regQKBias[3], regQKMul[3], regvBias[3], maskAllB32);
        // float -> uint32
        FloatX2ToSortableKey<float>(regOut[0], regOut[1], regQKBias[0], regQKBias[1], fp32Ctx, maskAllB32);
        FloatX2ToSortableKey<float>(regOut[2], regOut[3], regQKBias[2], regQKBias[3], fp32Ctx, maskAllB32);
        // 搬入outBuf的value部分
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(outBuf + (mrgValueLen * i), regOut[0], maskAllB32);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(outBuf + (mrgValueLen * i) + 64, regOut[1], maskAllB32);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(outBuf + (mrgValueLen * i) + 128, regOut[2], maskAllB32);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(outBuf + (mrgValueLen * i) + 192, regOut[3], maskAllB32);
    }
}

// float in uint16 out, use bf16 sortable key to match uint16 topk.
template <typename QK_T>
__simd_vf__ inline void MulRSquareAndAddVBiasVFImpl(__ubuf__ uint16_t* outBuf,
                                                    __ubuf__ QK_T* qkBuf,
                                                    __ubuf__ float* vBiasBuf,
                                                    const float rSquare,
                                                    const int gS1BasePerVecSize_,
                                                    const int s2BaseSize,
                                                    const int mrgValueLen)
{
    MicroAPI::RegTensor<float> regvBias[4];
    MicroAPI::RegTensor<float> regQK[4];
    MicroAPI::RegTensor<float> regQKMul[4];
    MicroAPI::RegTensor<float> regQKBias[4];
    MicroAPI::RegTensor<bfloat16_t> regQKBiasBF16[2];
    MicroAPI::RegTensor<uint16_t> regOut[2];
    MicroAPI::MaskReg maskAllB32 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskAllB16 = AscendC::MicroAPI::CreateMask<bfloat16_t, AscendC::MicroAPI::MaskPattern::ALL>();

    FloatSortConstCtx<bfloat16_t> bf16Ctx;
    InitFloatSortConstCtx(bf16Ctx, maskAllB16);

    constexpr static MicroAPI::CastTrait castTraitF32ToF16_EVEN = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
        MicroAPI::MaskMergeMode::MERGING, RoundMode::CAST_ROUND};
    constexpr static MicroAPI::CastTrait castTraitF32ToF16_ODD = {
        MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(regvBias[0], vBiasBuf);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(regvBias[1], vBiasBuf + 64);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(regvBias[2], vBiasBuf + 128);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(regvBias[3], vBiasBuf + 192);

    for (uint16_t i = 0; i < (uint16_t)(gS1BasePerVecSize_); ++i) {
        MicroAPI::LoadAlign<float>(regQK[0], qkBuf + s2BaseSize * i);
        MicroAPI::LoadAlign<float>(regQK[1], qkBuf + s2BaseSize * i + 64);
        MicroAPI::LoadAlign<float>(regQK[2], qkBuf + s2BaseSize * i + 128);
        MicroAPI::LoadAlign<float>(regQK[3], qkBuf + s2BaseSize * i + 192);
        MicroAPI::Muls(regQKMul[0], regQK[0], rSquare, maskAllB32);
        MicroAPI::Muls(regQKMul[1], regQK[1], rSquare, maskAllB32);
        MicroAPI::Muls(regQKMul[2], regQK[2], rSquare, maskAllB32);
        MicroAPI::Muls(regQKMul[3], regQK[3], rSquare, maskAllB32);
        MicroAPI::Add(regQKBias[0], regQKMul[0], regvBias[0], maskAllB32);
        MicroAPI::Add(regQKBias[1], regQKMul[1], regvBias[1], maskAllB32);
        MicroAPI::Add(regQKBias[2], regQKMul[2], regvBias[2], maskAllB32);
        MicroAPI::Add(regQKBias[3], regQKMul[3], regvBias[3], maskAllB32);

        MicroAPI::DeInterleave(regQKBias[0], regQKBias[1], regQKBias[0], regQKBias[1]);
        MicroAPI::DeInterleave(regQKBias[2], regQKBias[3], regQKBias[2], regQKBias[3]);
        MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regQKBiasBF16[0], regQKBias[1], maskAllB32);
        MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_ODD>(regQKBiasBF16[1], regQKBias[3], maskAllB32);
        MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regQKBiasBF16[0], regQKBias[0], maskAllB32);
        MicroAPI::Cast<bfloat16_t, float, castTraitF32ToF16_EVEN>(regQKBiasBF16[1], regQKBias[2], maskAllB32);
        FloatX2ToSortableKey<bfloat16_t>(regOut[0], regOut[1], regQKBiasBF16[0], regQKBiasBF16[1],
                                         bf16Ctx, maskAllB16);

        MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_NORM>(outBuf + (mrgValueLen * i), regOut[0], maskAllB16);
        MicroAPI::StoreAlign<uint16_t, MicroAPI::StoreDist::DIST_NORM>(outBuf + (mrgValueLen * i) + 128, regOut[1], maskAllB16);
    }
}

/**
 * @brief MUL RSquare后加上偏移
 * @param outLocal VEC1结果 转为SCORE_T sortable key
 * @param qkLocal mm1 res的结果 注意力分数
 * @param vBiasLocal kb的偏移向量
 * @param rSquare (stemStride / stemBlockSize) ^ 2
 * @param gS1BasePerVecSize_ g1S方向每个vec核分到基本块大小
 * @param s2BaseSize s2方向每个vec核分到基本块大小
 * @param mrgValueLen 每次流式topk分到的长度
 */
template<typename SCORE_T, typename QK_T>
__aicore__ inline void MulRSquareAndAddVBiasVF(const LocalTensor<SCORE_T> &outLocal,   // out    [S2Base]       [256   ]
                                               const LocalTensor<QK_T> &qkLocal,        // q*k^t  [qbG, S2Base]  [48 256]
                                               const LocalTensor<float> &vBiasLocal,    // vBias  [1, S2Base]    [256   ]
                                               const float rSquare,
                                               const int gS1BasePerVecSize_,
                                               const int s2BaseSize,
                                               const int mrgValueLen)
{
    __ubuf__ QK_T * qkBuf = (__ubuf__ QK_T *)qkLocal.GetPhyAddr();
    __ubuf__ float * vBiasBuf = (__ubuf__ float *)vBiasLocal.GetPhyAddr();
    __ubuf__ SCORE_T * outBuf = (__ubuf__ SCORE_T *)outLocal.GetPhyAddr();

    MulRSquareAndAddVBiasVFImpl(outBuf, qkBuf, vBiasBuf, rSquare, gS1BasePerVecSize_, s2BaseSize, mrgValueLen);
}

}

#endif
