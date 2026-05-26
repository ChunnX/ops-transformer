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
 * \file indexer_quant_cache_base.h
 * \brief
 */

#ifndef INDEXER_QUANT_CACHE_BASE_H
#define INDEXER_QUANT_CACHE_BASE_H

#include "kernel_operator.h"

namespace IndexerQuantCache {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::UnalignReg;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t VL_FP32 = 64;
constexpr int32_t PER_BLOCK_FP16 = 128;
constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
constexpr float FP8_E4M3FN_MAX_VALUE = 448.0f;
constexpr float FP8_E5M2_MIN_VALUE = -57344.0f;
constexpr float FP8_E4M3FN_MIN_VALUE = -448.0f;
constexpr float HIFLOAT8_MAX_VALUE = 32768.0f;
constexpr uint32_t FAST_LOG_SHIFT_BITS = 23U;
constexpr uint32_t FAST_LOG_AND_VALUE1 = 0xFF;
constexpr uint32_t FAST_LOG_AND_VALUE2 = (((uint32_t)1 << (uint32_t)23) - (uint32_t)1);
constexpr uint32_t INV_FP8_E5M2_MAX_VALUE = 0x37924925;
constexpr uint32_t INV_FP8_E4M3_MAX_VALUE = 0x3b124925;
constexpr uint32_t INV_HIFP8_MAX_VALUE = 0x38000000;
constexpr int64_t HIFLOAT_QUANT_MODE = 2;

#define FLOAT_OVERFLOW_MODE_CTRL 60
#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif
constexpr float POS_INFINITY = INFINITY;
constexpr float NEG_INFINITY = -INFINITY;

__aicore__ inline int32_t CeilDiv(int32_t a, int b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

// Decode a flat slot into its physical offset in a 4D paged buffer
// [blockNum, blockSize, 1, headDim] whose blockNum dimension is non-contiguous:
//   offset = (slot / blockSize) * blockStride + (slot % blockSize) * rowStride
// blockSize==1 (contiguous default) degrades to slot * rowStride.
__aicore__ inline int64_t PagedSlotOffset(int64_t slot, int64_t blockSize, int64_t blockStride, int64_t rowStride)
{
    int64_t block = slot / blockSize;
    int64_t pos = slot % blockSize;
    return block * blockStride + pos * rowStride;
}

__aicore__ inline int32_t CeilAlign(int32_t a, int b)
{
    return CeilDiv(a, b) * b;
}

template <typename T>
__aicore__ inline int32_t RoundUp(int32_t num)
{
    int32_t elemNum = BLOCK_SIZE / sizeof(T);
    return CeilAlign(num, elemNum);
}

template <typename T>
__aicore__ inline int32_t RoundUp(int32_t num, int32_t elemNum)
{
    return CeilAlign(num, elemNum);
}

constexpr AscendC::MicroAPI::CastTrait castTraitB162B32Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::MicroAPI::CastTrait castTraitB322B16Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitF32toFp8Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitU32toU8Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_NONE,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitF32toh8 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND
};


template <typename T>
__aicore__ inline void LoadInputData(RegTensor<float>& dst, __local_mem__ T* src, MaskReg pregLoop, uint32_t srcOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy(dst, src + srcOffset);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(tmp, src + srcOffset);
        Cast<float, T, castTraitB162B32Even>(dst, tmp, pregLoop);
    }
}

template <typename T>
__aicore__ inline void StoreOutputData(
    __local_mem__ T* dst, RegTensor<float>& src, MaskReg pregLoop, uint32_t dstOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy(dst + dstOffset, src, pregLoop);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        Cast<T, float, castTraitB322B16Even>(tmp, src, pregLoop);
        DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(dst + dstOffset, tmp, pregLoop);
    } else if constexpr (IsSameType<T, fp8_e4m3fn_t>::value || IsSameType<T, fp8_e5m2_t>::value) {
        RegTensor<T> tmp;
        Cast<T, float, castTraitF32toFp8Even>(tmp, src, pregLoop);
        DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(dst + dstOffset, tmp, pregLoop);
    } else if constexpr (IsSameType<T, hifloat8_t>::value) {
        RegTensor<T> tmp;
        Cast<T, float, castTraitF32toh8>(tmp, src, pregLoop);
        DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(dst + dstOffset, tmp, pregLoop);
    }
}

template <typename T>
__aicore__ inline void StoreMxFp8Scale(
    __local_mem__ T* dst, RegTensor<float>& src, MaskReg pregLoop, uint32_t dstOffset)
{
    RegTensor<uint32_t> tmp;
    RegTensor<uint8_t> tmp1;
    ShiftRights(tmp, (RegTensor<uint32_t> &)src, static_cast<int16_t>(FAST_LOG_SHIFT_BITS), pregLoop);
    Cast<uint8_t, uint32_t, castTraitU32toU8Even>(tmp1, tmp, pregLoop);
    DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B8>(dst + dstOffset, (RegTensor<T> &)tmp1, pregLoop);
}


// MX-FP8 量化, 每 32 个元素一个 e8m0 scale (标准 MX 块, amax-first + 正确尾数舍入,
// 与 swiglu_group_quant 的 mxfp8 实现一致)。要求 curColNum % 32 == 0。
template <typename T0, typename T1, typename T2, bool roundScale = true>
__aicore__ inline void VFProcessDynamicMxFp8Quant(
    const LocalTensor<T0>& yLocal, const LocalTensor<T1>& scaleLocal, const LocalTensor<T2>& xLocal,
    float coeff, float fp8Min, float fp8Max, const uint16_t curRowNum, const uint32_t curColNum)
{
    __local_mem__ T0* yLocalAddr = (__local_mem__ T0*)yLocal.GetPhyAddr();
    __local_mem__ T1* scaleLocalAddr = (__local_mem__ T1*)scaleLocal.GetPhyAddr();
    __local_mem__ T2* xLocalAddr = (__local_mem__ T2*)xLocal.GetPhyAddr();
    constexpr uint32_t MXB = 32;
    uint32_t scaleCol = CeilDiv(curColNum, MXB);
    uint32_t curColNumAlign = RoundUp<T2>(curColNum);
    uint32_t dstCurColNumAlign = RoundUp<T0>(curColNum);
    uint32_t scaleColNumAlign = RoundUp<T1>(scaleCol);
    __VEC_SCOPE__
    {
        RegTensor<T2> raw;
        RegTensor<float> x;
        RegTensor<float> xAbs;
        RegTensor<float> max2;
        RegTensor<float> dupScale;
        RegTensor<uint32_t> tmp0;
        RegTensor<uint32_t> tmp1;
        RegTensor<uint32_t> vreg0;
        RegTensor<uint32_t> vreg1;
        RegTensor<uint32_t> vreg2;
        RegTensor<uint32_t> vreg4;
        RegTensor<int32_t> vreg5;
        RegTensor<uint32_t> zero;
        RegTensor<uint32_t> one;
        RegTensor<uint32_t> tmp3;
        RegTensor<T0> outReg;
        MaskReg pm;
        MaskReg preg1 = CreateMask<T1, AscendC::MicroAPI::MaskPattern::VL1>();
        MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        MaskReg cmpMask;
        Duplicate(tmp0, FAST_LOG_AND_VALUE1, pregMerge);
        Duplicate(tmp1, FAST_LOG_AND_VALUE2, pregMerge);
        Duplicate(zero, static_cast<uint32_t>(0), pregMerge);
        Duplicate(one, static_cast<uint32_t>(1), pregMerge);
        Duplicate(tmp3, static_cast<uint32_t>(127), pregMerge);
        for (uint16_t i = 0; i < curRowNum; i++) {
            __local_mem__ T2* xp = xLocalAddr + i * curColNumAlign;
            __local_mem__ T0* yp = yLocalAddr + i * dstCurColNumAlign;
            for (uint16_t g = 0; g < scaleCol; g++) {
                uint32_t cnt = MXB;
                pm = UpdateMask<float>(cnt);  // 32-lane mask
                AscendC::MicroAPI::DataCopy<
                    T2, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    raw, xp, MXB);
                Cast<float, T2, castTraitB162B32Even>(x, raw, pm);
                Abs(xAbs, x, pm);
                ReduceMax(max2, xAbs, pm);
                Maxs(max2, max2, static_cast<float>(1e-4), pregMerge);
                Muls(max2, max2, coeff, pregMerge);  // sf = amax / fp8max
                if constexpr (roundScale) {
                    // round to power-of-two with correct mantissa check (== swiglu mxfp8):
                    // e8m0 = exp + (mantissa != 0); scale = 2^(e8m0 - 127)
                    ShiftRights(vreg0, (RegTensor<uint32_t> &)max2, static_cast<int16_t>(FAST_LOG_SHIFT_BITS), pregMerge);
                    And(vreg1, vreg0, tmp0, pregMerge);                              // exp field
                    And(vreg2, (RegTensor<uint32_t> &)max2, tmp1, pregMerge);        // mantissa field
                    Compare<uint32_t, AscendC::CMPMODE::NE>(cmpMask, vreg2, zero, pregMerge);
                    Select(vreg4, one, zero, cmpMask);
                    Sub(vreg1, vreg1, tmp3, pregMerge);
                    Add(vreg1, vreg1, vreg4, pregMerge);
                    Adds(vreg5, (RegTensor<int32_t> &)vreg1, static_cast<int32_t>(127), pregMerge);
                    ShiftLefts((RegTensor<int32_t> &)max2, vreg5, static_cast<int16_t>(23), pregMerge);
                }
                Duplicate(dupScale, max2, pm);
                Div(x, x, dupScale, pm);
                Maxs(x, x, fp8Min, pm);
                Mins(x, x, fp8Max, pm);
                Cast<T0, float, castTraitF32toFp8Even>(outReg, x, pm);
                AscendC::MicroAPI::DataCopy<
                    T0, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                    yp, outReg, MXB, pm);
                StoreMxFp8Scale<T1>(scaleLocalAddr, max2, preg1, g + i * scaleColNumAlign);
            }
        }
    }
}

// Normal 动态量化, 整行一个 float32 scale (scale = rowmax / fp8max)。两遍处理:
// 第一遍跨整行求 amax, 第二遍按该 scale 量化。NaN 安全 (排除 NaN 求 max, 结果 NaN 时保留原值)。
template <typename T0, typename T1>
__aicore__ inline void VFProcessDynamicBlockQuant(
    const LocalTensor<T0>& yLocal, const LocalTensor<float>& scaleLocal, const LocalTensor<T1>& xLocal,
    float coeff, const uint16_t curRowNum, const uint32_t curColNum)
{
    (void)coeff;
    __local_mem__ T0* yLocalAddr = (__local_mem__ T0*)yLocal.GetPhyAddr();
    __local_mem__ float* scaleLocalAddr = (__local_mem__ float*)scaleLocal.GetPhyAddr();
    __local_mem__ T1* xLocalAddr = (__local_mem__ T1*)xLocal.GetPhyAddr();
    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint32_t curColNumAlign = RoundUp<T1>(curColNum);
    uint32_t dstCurColNumAlign = RoundUp<T0>(curColNum);
    uint32_t scaleRowAlign = RoundUp<float>(1);  // 整行一个 scale
    static constexpr AscendC::MicroAPI::DivSpecificMode mode = {AscendC::MicroAPI::MaskMergeMode::ZEROING, false};
    uint32_t maxValueInt = 0;
    if constexpr (IsSameType<T0, fp8_e5m2_t>::value) {
        maxValueInt = INV_FP8_E5M2_MAX_VALUE;
    } else if constexpr (IsSameType<T0, fp8_e4m3fn_t>::value) {
        maxValueInt = INV_FP8_E4M3_MAX_VALUE;
    } else if constexpr (IsSameType<T0, hifloat8_t>::value) {
        maxValueInt = INV_HIFP8_MAX_VALUE;
    }

    __VEC_SCOPE__
    {
        RegTensor<float> x;
        RegTensor<float> xAbs;
        RegTensor<float> partMax;
        RegTensor<float> rowMax;
        RegTensor<float> x0;
        RegTensor<float> x1;
        RegTensor<float> scale;
        RegTensor<float> dupScale;
        RegTensor<float> inf;
        RegTensor<float> zero;
        RegTensor<uint32_t> coeffReg;
        MaskReg pregMain = CreateMask<float>();
        MaskReg preg1 = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        MaskReg pm;
        MaskReg cmpAbs;
        MaskReg cmpScalar;
        MaskReg cmpQ;
        Duplicate(coeffReg, maxValueInt, pregMain);
        Duplicate(zero, 0.0f, pregMain);
        Duplicate(inf, 1.0f, pregMain);
        Div<float, &mode>(inf, inf, zero, pregMain);
        for (uint16_t i = 0; i < curRowNum; i++) {
            // pass 1: 整行 amax (排除 NaN)
            Duplicate(rowMax, 0.0f, preg1);
            uint32_t sreg = curColNum;
            for (uint16_t j = 0; j < loopCount; j++) {
                pm = UpdateMask<float>(sreg);
                LoadInputData<T1>(x, xLocalAddr, pm, j * VL_FP32 + i * curColNumAlign);
                Muls(xAbs, x, 0.0f, pm);
                Compare<float, CMPMODE::NE>(cmpAbs, xAbs, xAbs, pm);
                MaskNot(cmpAbs, cmpAbs, pm);
                Abs(xAbs, x, cmpAbs);
                ReduceMax(partMax, xAbs, cmpAbs);
                Max(rowMax, rowMax, partMax, preg1);
            }
            CompareScalar<float, CMPMODE::NE>(cmpScalar, rowMax, (float)0.0, preg1);
            Mul(scale, rowMax, (RegTensor<float>&)coeffReg, cmpScalar);  // rowmax / fp8max
            Min(scale, scale, inf, preg1);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                scaleLocalAddr + i * scaleRowAlign, scale, preg1);
            // pass 2: 量化
            Duplicate(dupScale, scale, pregMain);
            sreg = curColNum;
            for (uint16_t j = 0; j < loopCount; j++) {
                pm = UpdateMask<float>(sreg);
                LoadInputData<T1>(x, xLocalAddr, pm, j * VL_FP32 + i * curColNumAlign);
                Div<float, &mode>(x0, x, dupScale, pm);
                Muls(x1, x0, 0.0f, pm);
                Compare<float, CMPMODE::NE>(cmpQ, x1, x1, pm);
                Select(x, x, x0, cmpQ);
                StoreOutputData<T0>(yLocalAddr, x, pm, j * VL_FP32 + i * dstCurColNumAlign);
            }
        }
    }
}

template <typename T0, typename T1>
__aicore__ inline void VFProcessHifp8Quant(
    const LocalTensor<T0>& yLocal, const LocalTensor<float>& scaleLocal, const LocalTensor<T1>& xLocal,
    float coeff, const uint16_t curRowNum, const uint32_t curColNum, const float scales)
{
    LocalTensor<hifloat8_t> outLocal = yLocal.template ReinterpretCast<hifloat8_t>();
    __local_mem__ hifloat8_t* yLocalAddr = (__local_mem__ hifloat8_t*)outLocal.GetPhyAddr();
    __local_mem__ float* scaleLocalAddr = (__local_mem__ float*)scaleLocal.GetPhyAddr();
    __local_mem__ T1* xLocalAddr = (__local_mem__ T1*)xLocal.GetPhyAddr();
    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint32_t curColNumAlign = RoundUp<T1>(curColNum);
    uint32_t dstCurColNumAlign = RoundUp<T0>(curColNum);
    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<hifloat8_t> tmp;
        MaskReg pregLoop = CreateMask<float>();
        for (uint16_t i = 0; i < curRowNum; i++) {
            uint32_t sreg = curColNum;
            for (uint16_t j = 0; j < loopCount; j++) {
                pregLoop = UpdateMask<float>(sreg);
                LoadInputData<T1>(xReg, xLocalAddr, pregLoop, j * VL_FP32 + i * curColNumAlign);
                Muls(xReg, xReg, scales, pregLoop);
                Cast<hifloat8_t, float, castTraitF32toh8>(tmp, xReg, pregLoop);
                DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                    yLocalAddr + j * VL_FP32 + i * dstCurColNumAlign, tmp, pregLoop);
            }
        }
    }
}

template <typename T>
__aicore__ inline void CopyIn(
    const GlobalTensor<T>& inputGm, const LocalTensor<T>& inputTensor, const uint16_t nBurst, const uint32_t copyLen,
    uint32_t srcStride = 0)
{
    DataCopyPadExtParams<T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen * sizeof(T);
    dataCoptExtParams.srcStride = srcStride * sizeof(T);
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(inputTensor, inputGm, dataCoptExtParams, dataCopyPadExtParams);
}


template <typename T, AscendC::PaddingMode mode = AscendC::PaddingMode::Normal>
__aicore__ inline void CopyOut(
    const LocalTensor<T>& outputTensor, const GlobalTensor<T>& outputGm, const uint16_t nBurst, const uint32_t copyLen,
    uint32_t dstStride = 0)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = nBurst;
    dataCopyParams.blockLen = copyLen * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = dstStride * sizeof(T);
    DataCopyPad<T, mode>(outputGm, outputTensor, dataCopyParams);
}

} // namespace IndexerQuantCache

#endif