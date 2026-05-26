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
 * \file swiglu_block_quant_base.h
 * \brief
 */

#ifndef KV_COMPRESS_EPILOG_COMMON_H
#define KV_COMPRESS_EPILOG_COMMON_H

#include "kernel_operator.h"

namespace KvCompressEpilogOps {
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
constexpr int64_t QUANT_MDOE_GROUP_QUANT_BF16 = 1;
constexpr int64_t QUANT_MDOE_GROUP_QUANT_E8M0 = 2;
constexpr int64_t QUANT_MDOE_HIFLOAT = 3;
constexpr uint32_t FAST_LOG_SHIFT_BITS = 23U;
constexpr uint32_t FAST_LOG_AND_VALUE1 = 0xFF;
constexpr uint32_t FAST_LOG_AND_VALUE2 = (((uint32_t)1 << (uint32_t)23) - (uint32_t)1);
constexpr uint32_t INV_FP8_E5M2_MAX_VALUE = 0x37924925;
constexpr uint32_t INV_FP8_E4M3_MAX_VALUE = 0x3b124925;

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

constexpr static AscendC::MicroAPI::CastTrait castTraitU32toU16Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_NONE,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitB322B8Even = {
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

template <typename T0, typename T1, bool roundScale = true, bool castBf16 = false>
__aicore__ inline void VFProcessFP8PerGroupQuant(
    const LocalTensor<T1>& yLocal, const LocalTensor<T0>& xLocal, 
    float coeff, float fp8Min, float fp8Max, const uint16_t curRowNum, const uint32_t curColNum, const uint32_t concatColNum, const uint32_t padColNum) 
{
    __local_mem__ T1* yLocalAddr = (__local_mem__ T1*)yLocal.GetPhyAddr();
    __local_mem__ T0* xLocalAddr = (__local_mem__ T0*)xLocal.GetPhyAddr();
    __local_mem__ T1* scaleLocalAddr;
    __local_mem__ T0* scaleLocalBf16Addr;
    if constexpr (castBf16) {
        LocalTensor<T0> scaleBf16Tensor = yLocal.template ReinterpretCast<T0>();
        scaleLocalBf16Addr = (__local_mem__ T0*)scaleBf16Tensor.GetPhyAddr();
    } else {
        scaleLocalAddr = (__local_mem__ T1*)yLocal.GetPhyAddr();
    }
    __local_mem__ T0* ropeYLocalAddr = (__local_mem__ T0*)yLocal.GetPhyAddr();

    uint32_t quantColNum = curColNum - 64;
    uint16_t scaleColNum = CeilDiv(quantColNum, 128);
    uint16_t ropeNum = 128;
    uint16_t loopCount = CeilDiv(quantColNum, VL_FP32);
    uint32_t curColNumAlign = RoundUp<T0>(curColNum);
    uint32_t dstCurColNumAlign = RoundUp<T1>(concatColNum+padColNum);
    uint16_t loopCountFoldTwo = loopCount / 2;
    uint16_t loopCountReminder = loopCount % 2;
    uint32_t tailReminder = quantColNum - (loopCount - 1) * VL_FP32;
    uint32_t scaleColNumAlign = RoundUp<T0>(scaleColNum);
    uint32_t sregNum = quantColNum;
    __VEC_SCOPE__
    {
        RegTensor<float> x0;
        RegTensor<float> x0Abs;
        RegTensor<float> x1;
        RegTensor<float> x1Abs;
        RegTensor<float> max0;
        RegTensor<float> max1;
        RegTensor<float> max2;
        RegTensor<uint32_t> tmp1;
        RegTensor<uint32_t> vreg0;
        RegTensor<uint32_t> vreg1;
        RegTensor<uint32_t> vreg2;
        RegTensor<uint32_t> vreg3;
        RegTensor<uint32_t> vreg4;
        RegTensor<int32_t> vreg5;
        RegTensor<uint32_t> zero;
        RegTensor<uint32_t> one;
        RegTensor<float> dupScale;
        RegTensor<T0> ropeReg;
        RegTensor<uint32_t> scaleTmp0;
        RegTensor<uint8_t> scaleTmp1;
        RegTensor<uint16_t> scaleTmp1Bf16;
        UnalignRegForStore ureg0;
        UnalignRegForLoad ureg1;
        UnalignRegForStore ureg2;
        MaskReg pregLoop;
        MaskReg preg1 = CreateMask<T0, AscendC::MicroAPI::MaskPattern::VL1>();
        MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        MaskReg pregMain = CreateMask<float>();
        MaskReg pregRope = CreateMask<T0, AscendC::MicroAPI::MaskPattern::VL64>();
        MaskReg cmpMask;
        Duplicate(tmp1, FAST_LOG_AND_VALUE2, pregMerge);
        Duplicate(zero, static_cast<uint32_t>(0), pregMerge);
        Duplicate(one, static_cast<uint32_t>(1), pregMerge);
        for (uint16_t i = 0; i < curRowNum; i++) {
            // cat rope
            __local_mem__ T0* rowRopeXLocalAddr = xLocalAddr + i * curColNumAlign + quantColNum;
            
            LoadUnAlignPre(ureg0, rowRopeXLocalAddr);
            LoadUnAlign(ropeReg, ureg0, rowRopeXLocalAddr, 64);
            DataCopy(ropeYLocalAddr + i * dstCurColNumAlign / 2, ropeReg, pregRope);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_STORE>();
            uint32_t sreg = sregNum;
            for (uint16_t j = 0; j < loopCount; j++) {
                pregLoop = UpdateMask<float>(sreg);
                LoadInputData<T0>(x0, xLocalAddr, pregLoop, j * VL_FP32 + i * curColNumAlign);
                Abs(x0Abs, x0, pregLoop);
                ReduceMax(max0, x0Abs, pregLoop);
                Maxs(max2, max0, static_cast<float>(1e-4), pregMerge);
                Muls(max2, max2, coeff, pregMerge);
                if constexpr (roundScale) {
                    ShiftRights(vreg1, (RegTensor<uint32_t> &)max2, static_cast<int16_t>(FAST_LOG_SHIFT_BITS), pregMerge);
                    And(vreg2, (RegTensor<uint32_t> &)max2, tmp1, pregMerge);
                    Compare<uint32_t, AscendC::CMPMODE::NE>(cmpMask, vreg2, zero, pregMerge);
                    Select(vreg4, one, zero, cmpMask);
                    Add(vreg1, vreg1, vreg4, pregMerge);
                    ShiftLefts((RegTensor<int32_t> &)max2, (RegTensor<int32_t> &)vreg1, static_cast<int16_t>(23), pregMerge);
                }
                Duplicate(dupScale, max2, pregMain);
                Div(x0, x0, dupScale, pregLoop);
                Maxs(x0, x0, fp8Min, pregLoop);
                Mins(x0, x0, fp8Max, pregLoop);
                {
                    RegTensor<fp8_e4m3fn_t> fp8Out;
                    Cast<fp8_e4m3fn_t, float, castTraitF32toFp8Even>(fp8Out, x0, pregLoop);
                    DataCopy<T1, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                        yLocalAddr + j * VL_FP32 + i * dstCurColNumAlign + ropeNum, (RegTensor<T1> &)fp8Out, pregLoop);
                }

                // cat scale
                ShiftRights(scaleTmp0, (RegTensor<uint32_t> &)max2, static_cast<int16_t>(FAST_LOG_SHIFT_BITS), preg1);
                if constexpr (castBf16) {
                    Cast<uint16_t, uint32_t, castTraitU32toU16Even>(scaleTmp1Bf16, scaleTmp0, preg1);
                    DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B16>(scaleLocalBf16Addr + (((quantColNum + ropeNum + i * dstCurColNumAlign) >> 1) + j), (RegTensor<bfloat16_t> &)scaleTmp1Bf16, preg1);
                } else {
                    Cast<uint8_t, uint32_t, castTraitB322B8Even>(scaleTmp1, scaleTmp0, preg1);
                    DataCopy<T1, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B8>(scaleLocalAddr + quantColNum + ropeNum + j + i * dstCurColNumAlign, (RegTensor<T1>&)scaleTmp1, preg1);
                }
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_STORE>();

            // pad zero
            __local_mem__ T1* rowPadLocalAddr = yLocalAddr + i * dstCurColNumAlign + concatColNum;
            StoreUnAlign(rowPadLocalAddr, (RegTensor<T1>&)zero, ureg1, padColNum);
            StoreUnAlignPost(rowPadLocalAddr, ureg1, 0);
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

} // namespace KvCompressEpilogOps

#endif