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
 * \file gmmsq_quant_mx_vf.h
 * \brief SIMD-VF functions for GMM SwiGLU + MX Quantization.
 */

#ifndef GMMSQ_QUANT_MX_VF_H
#define GMMSQ_QUANT_MX_VF_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#ifndef GMMSQ_VF_USE_MICRO_API
#include "basic_block_vf_mx.h"
#endif

namespace GMMSQWeightQuant {

#ifdef GMMSQ_VF_USE_MICRO_API
namespace Reg = AscendC::MicroAPI;
#else
namespace Reg = AscendC::Reg;
#endif
using AscendC::VECTOR_REG_WIDTH;
using Reg::AddrReg;
using Reg::MaskReg;
using Reg::RegTensor;

static constexpr Reg::DivSpecificMode DIV_MODE = {
    Reg::MaskMergeMode::ZEROING,
    true,
};

static constexpr Reg::CastTrait CAST_FP32_TO_BF16 = {
    Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

/*!
 * \brief Computes SwiGLU activation using vector instructions.
 *
 * \param[in] firstSrc          Swish branch input in UB.
 * \param[in] secondSrc         Gate branch input in UB.
 * \param[in] gluResAddr        Output UB address for SwiGLU result.
 * \param[in] mSize             M dimension size.
 * \param[in] nSize             N dimension size.
 * \param[in] nSrcUbAligned     Aligned N stride for source UB.
 * \param[in] nDstUbAligned     Aligned N stride for destination UB.
 * \param[in] oneRowRepeatTimes Number of vector repeats per row.
 * \param[in] sizePerRepeat     Elements processed per vector repeat.
 */
template <typename DataTypeIn, bool HasMulFactor = true>
__simd_vf__ inline void GmmSwigluVf(__ubuf__ DataTypeIn *firstSrc, __ubuf__ DataTypeIn *secondSrc,
                                    __ubuf__ bfloat16_t *gluResAddr, uint16_t mSize, uint16_t nSize,
                                    uint32_t nSrcUbAligned, uint32_t nDstUbAligned, uint16_t oneRowRepeatTimes,
                                    uint16_t sizePerRepeat)
{
    const float scalarOne = 1.0;
    const float mulFactor = 64.0; // 64：GMM缩小64倍在此处补齐
    for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
        uint32_t elementNum = nSize;
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < oneRowRepeatTimes; vfBlockIdx++) {
            Reg::MaskReg mask = Reg::UpdateMask<DataTypeIn>(elementNum);
            Reg::RegTensor<bfloat16_t> verg4;
            Reg::RegTensor<float> swishInput, gateInput;
            Reg::RegTensor<float> verg0, verg1, verg2, verg3, swishOutput;

            uint32_t l0cOutOffset = mIdx * nSrcUbAligned + vfBlockIdx * sizePerRepeat;

            // 1. Load Swish branch
            Reg::LoadAlign(swishInput, firstSrc + l0cOutOffset);
            if constexpr (HasMulFactor) {
                Reg::Muls(swishInput, swishInput, mulFactor, mask);
            }

            // 2. Swish: x / (1 + exp(-x))
            Reg::Muls(verg0, swishInput, -(scalarOne), mask);
            Reg::Exp(verg1, verg0, mask);
            Reg::Adds(verg2, verg1, scalarOne, mask);
            Reg::Div<float, &DIV_MODE>(swishOutput, swishInput, verg2, mask);

            // 3. Load Gate branch
            Reg::LoadAlign(gateInput, secondSrc + l0cOutOffset);
            if constexpr (HasMulFactor) {
                Reg::Muls(gateInput, gateInput, mulFactor, mask);
            }

            // 4. SwiGLU: Swish(x) * gate
            Reg::Mul(verg3, swishOutput, gateInput, mask);

            // 5. Cast to BF16 and store
            Reg::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(verg4, verg3, mask);
            uint32_t dstUbOffset = mIdx * nDstUbAligned + vfBlockIdx * sizePerRepeat;
            Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(gluResAddr + dstUbOffset, verg4, mask);
        }
    }
}

/*!
 * \brief Extracts maximum exponent per 32-byte block from source UB.
 *
 * \param[in] srcAddr             Source UB address (bfloat16_t).
 * \param[in] maxExpAddr          Destination UB address for max exponents.
 * \param[in] totalCountInUB      Total element count in UB to process.
 * \param[in] loopNum             Number of vector loops required.
 * \param[in] vlForHalfNumber     Vector length for half precision elements.
 * \param[in] elementAfterReduce  Element count after reduction.
 */
__simd_vf__ inline void ComputeMaxExpVf(__ubuf__ bfloat16_t *srcAddr, __ubuf__ uint16_t *maxExpAddr,
                                        uint32_t totalCountInUB, uint16_t loopNum, uint32_t vlForHalfNumber,
                                        uint16_t elementAfterReduce)
{
    Reg::RegTensor<bfloat16_t> vdExp0, vdExp1;
    Reg::RegTensor<uint16_t> vdExpExtract0, vdExpExtract1;
    Reg::RegTensor<uint16_t> expMaskBF16;
    Reg::Duplicate(expMaskBF16, static_cast<uint16_t>(0x7f80));

    Reg::RegTensor<uint16_t> vdMaxExp;
    Reg::MaskReg scaleMask1;
    Reg::UnalignReg u1;

    for (uint16_t i = 0; i < loopNum; i++) {
        scaleMask1 = Reg::UpdateMask<bfloat16_t>(totalCountInUB);

        Reg::LoadAlign<bfloat16_t, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, vlForHalfNumber * 2);

        Reg::And(vdExpExtract0, (Reg::RegTensor<uint16_t> &)vdExp0, expMaskBF16, scaleMask1);
        Reg::And(vdExpExtract1, (Reg::RegTensor<uint16_t> &)vdExp1, expMaskBF16, scaleMask1);

        Reg::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
        Reg::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);

        Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(maxExpAddr, vdMaxExp, u1,
                                                                                          elementAfterReduce);
    }
    Reg::StoreUnAlignPost(maxExpAddr, u1, 0);
}

/*!
 * \brief Computes MX scale (E8M0) and half-scale from max exponents.
 *
 * \param[in] maxExpAddr          Source UB address for max exponents.
 * \param[in] mxScaleLocalAddr    Destination UB address for MX scale.
 * \param[in] halfScaleLocalAddr  Destination UB address for half-scale.
 * \param[in] totalScaleInUB      Total scale element count in UB.
 * \param[in] loopNumScale        Number of vector loops for scale computation.
 * \param[in] vlForHalfNumber     Vector length for half precision elements.
 * \param[in] fpEmax              FP format max exponent value.
 */
__simd_vf__ inline void ComputeScaleVf(__ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr,
                                       __ubuf__ uint16_t *halfScaleLocalAddr, uint32_t totalScaleInUB,
                                       uint16_t loopNumScale, uint32_t vlForHalfNumber, uint16_t fpEmax)
{
    Reg::RegTensor<uint16_t> expMask, sharedExp, scaleValue, scaleOffset, halfScale, fp8NanRegTensor;
    Reg::Duplicate(expMask, static_cast<uint16_t>(0x7f80));
    Reg::MaskReg cmpResult, zeroMask, invalidDataMask, specialDataMask, preMaskScale;
    Reg::RegTensor<uint16_t> vdMaxExp;
    Reg::RegTensor<uint16_t> maxExpValue, zeroRegTensor, nanRegTensor, specialExpRegTensor;
    Reg::Duplicate(maxExpValue, fpEmax);
    Reg::Duplicate(scaleOffset, static_cast<uint16_t>(0x7f00));
    Reg::Duplicate(fp8NanRegTensor, static_cast<uint16_t>(0x00ff));
    Reg::Duplicate(zeroRegTensor, static_cast<uint16_t>(0));
    Reg::Duplicate(nanRegTensor, static_cast<uint16_t>(0x7f81));
    // 0x0040：sharedExp == 0x7f00的边界情况
    Reg::Duplicate(specialExpRegTensor, static_cast<uint16_t>(0x0040));

    for (uint16_t i = 0; i < loopNumScale; i++) {
        preMaskScale = Reg::UpdateMask<uint16_t>(totalScaleInUB);

        Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(vdMaxExp, maxExpAddr,
                                                                                       vlForHalfNumber);
        // 检测 INF/NAN 和 zero
        Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale);
        Reg::Compare<uint16_t, AscendC::CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
        Reg::Compare<uint16_t, AscendC::CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);
        Reg::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);

        Reg::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
        // 7：右移7位
        Reg::ShiftRights(scaleValue, sharedExp, static_cast<int16_t>(7), preMaskScale);
        Reg::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
        Reg::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

        Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE,
                                 Reg::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr, scaleValue,
                                                                         vlForHalfNumber >> 1, preMaskScale);

        Reg::Compare<uint16_t, AscendC::CMPMODE::EQ>(specialDataMask, sharedExp, scaleOffset, preMaskScale);

        Reg::Sub(halfScale, scaleOffset, sharedExp, preMaskScale);
        Reg::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
        Reg::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
        Reg::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

        Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(halfScaleLocalAddr, halfScale,
                                                                                        vlForHalfNumber, preMaskScale);
    }
}

/*!
 * \brief Transposes MX scale layout from interleaved to block format per row.
 *
 * \param[in] srcAddr     Source UB address for scale data.
 * \param[in] dstAddr     Destination UB address for transposed scale block.
 * \param[in] mSize       M dimension size.
 * \param[in] scaleBlockN Number of scale elements per block.
 */
__simd_vf__ inline void TransMxScaleLayoutVf(__ubuf__ int8_t *srcAddr, __ubuf__ int8_t *dstAddr, uint16_t mSize,
                                             uint16_t scaleBlockN)
{
    for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
        uint32_t elemNum = scaleBlockN;
        Reg::MaskReg maskScaleN = Reg::UpdateMask<int8_t>(elemNum);
        Reg::RegTensor<int8_t> vreg0;
        Reg::UnalignReg u0;

        auto srcUb = srcAddr + mIdx * scaleBlockN;
        Reg::LoadUnAlignPre(u0, srcUb);
        Reg::LoadUnAlign(vreg0, u0, srcUb);

        auto dstUb = dstAddr + mIdx * AscendC::ONE_BLK_SIZE;
        Reg::StoreAlign<int8_t, Reg::StoreDist::DIST_NORM_B8>(dstUb, vreg0, maskScaleN);
    }
}

/*!
 * \brief Quantizes FP8 data using computed half-scale.
 *
 * \param[in] srcAddr             Source UB address (bfloat16_t).
 * \param[in] halfScaleLocalAddr  Half-scale UB address.
 * \param[in] outLocalAddr        Output UB address (int8_t).
 * \param[in] totalCountInUB      Total element count in UB to quantize.
 * \param[in] loopNum             Number of vector loops required.
 * \param[in] vlForHalfNumber     Vector length for half precision elements.
 * \param[in] elementAfterReduce  Element count after reduction.
 */
template <typename DataTypeOut>
__simd_vf__ inline void ComputeDataForQuantTargetFp8Vf(__ubuf__ bfloat16_t *srcAddr,
                                                       __ubuf__ uint16_t *halfScaleLocalAddr,
                                                       __ubuf__ int8_t *outLocalAddr, uint32_t totalCountInUB,
                                                       uint16_t loopNum, uint32_t vlForHalfNumber,
                                                       uint16_t elementAfterReduce)
{
    uint32_t totalCountInUB2 = totalCountInUB * 2;

    Reg::MaskReg dataMask1, dataMask2, dataMask3, dataMask4;
    Reg::RegTensor<uint16_t> halfScaleForMul;
    Reg::RegTensor<bfloat16_t> vdExp0, vdExp1;
    Reg::RegTensor<float> vdExp0FP32Zero, vdExp0FP32One, vdExp1FP32Zero, vdExp1FP32One;
    Reg::RegTensor<DataTypeOut> vdExp0FP8Zero, vdExp0FP8One, vdExp1FP8Zero, vdExp1FP8One;

    static constexpr Reg::CastTrait castTraitZero = {
        Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    static constexpr Reg::CastTrait castTraitOne = {
        Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    static constexpr Reg::CastTrait castTrait32to8 = {
        Reg::RegLayout::ZERO, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};

    for (uint16_t i = 0; i < loopNum; i++) {
        dataMask1 = Reg::UpdateMask<bfloat16_t>(totalCountInUB);
        dataMask2 = Reg::UpdateMask<bfloat16_t>(totalCountInUB);
        dataMask3 = Reg::UpdateMask<bfloat16_t>(totalCountInUB2);
        dataMask4 = Reg::UpdateMask<bfloat16_t>(totalCountInUB2);

        Reg::LoadAlign<bfloat16_t, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, vlForHalfNumber * 2);

        Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                      elementAfterReduce);

        Reg::Mul(vdExp0, vdExp0, (Reg::RegTensor<bfloat16_t> &)halfScaleForMul, dataMask1);
        Reg::Mul(vdExp1, vdExp1, (Reg::RegTensor<bfloat16_t> &)halfScaleForMul, dataMask1);

        Reg::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);

        Reg::Cast<float, bfloat16_t, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
        Reg::Cast<float, bfloat16_t, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
        Reg::Interleave(vdExp0FP32Zero, vdExp0FP32One, vdExp0FP32Zero, vdExp0FP32One);
        Reg::Cast<DataTypeOut, float, castTrait32to8>(vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
        Reg::Cast<DataTypeOut, float, castTrait32to8>(vdExp0FP8One, vdExp0FP32One, dataMask3);

        Reg::Cast<float, bfloat16_t, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask2);
        Reg::Cast<float, bfloat16_t, castTraitOne>(vdExp1FP32One, vdExp1, dataMask2);
        Reg::Interleave(vdExp1FP32Zero, vdExp1FP32One, vdExp1FP32Zero, vdExp1FP32One);
        Reg::Cast<DataTypeOut, float, castTrait32to8>(vdExp1FP8Zero, vdExp1FP32Zero, dataMask4);
        Reg::Cast<DataTypeOut, float, castTrait32to8>(vdExp1FP8One, vdExp1FP32One, dataMask4);

        // 64：使用DIST_PACK4_B32进行搬运4个float8拼成一个float32，256 / 4 = 64
        Reg::StoreAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE,
                                 Reg::StoreDist::DIST_PACK4_B32>(
            outLocalAddr, (Reg::RegTensor<int8_t> &)vdExp0FP8Zero, static_cast<int64_t>(64), dataMask3);
        Reg::StoreAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE,
                                 Reg::StoreDist::DIST_PACK4_B32>(
            outLocalAddr, (Reg::RegTensor<int8_t> &)vdExp0FP8One, static_cast<int64_t>(64), dataMask3);
        Reg::StoreAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE,
                                 Reg::StoreDist::DIST_PACK4_B32>(
            outLocalAddr, (Reg::RegTensor<int8_t> &)vdExp1FP8Zero, static_cast<int64_t>(64), dataMask4);
        Reg::StoreAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE,
                                 Reg::StoreDist::DIST_PACK4_B32>(
            outLocalAddr, (Reg::RegTensor<int8_t> &)vdExp1FP8One, static_cast<int64_t>(64), dataMask4);
    }
}

#ifndef GMMSQ_VF_USE_MICRO_API
template <typename xType, typename wType>
__simd_vf__ inline void GmmsqAntiQuantMxA8W4NzNkVf(MxA8W4NzParams<xType, wType> mxA8W4NzParams)
{
    Reg::RegTensor<int8_t> wShrReg, wShlReg, wAndReg, wLoad, wShl, wShr0, wShr1, wSel, wAnd;
    Reg::MaskReg preg = Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
    Reg::MaskReg pregVsel = Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();

    Reg::Duplicate<int8_t, AscendC::Reg::MaskMergeMode::ZEROING>(wShrReg, E2M1_SHIFT_RIGHT_SIZE, preg);
    Reg::Duplicate<int8_t, AscendC::Reg::MaskMergeMode::ZEROING>(wShlReg, SHIFT_LEFT_SIZE, preg);
    Reg::Duplicate<int8_t, AscendC::Reg::MaskMergeMode::ZEROING>(wAndReg, E2M1_AND_MASK, preg);

    for (uint16_t loopKIdx = 0; loopKIdx < mxA8W4NzParams.loopKNum; ++loopKIdx) {
        for (uint16_t innerLoopIdx = 0; innerLoopIdx < mxA8W4NzParams.innerLoopNum; ++innerLoopIdx) {
            Reg::AddrReg aregWeightB8In = Reg::CreateAddrReg<uint8_t>(
                loopKIdx, (C0_SIZE_B8 * mxA8W4NzParams.nRealSizeAlign) >> 1, innerLoopIdx, VECTOR_REG_WIDTH >> 1);
            Reg::LoadAlign<uint8_t, Reg::LoadDist::DIST_US_B8>((Reg::RegTensor<uint8_t> &)wLoad,
                                                               (__ubuf__ uint8_t *&)mxA8W4NzParams.weightLowBitPhyAddr,
                                                               aregWeightB8In);

            Reg::ShiftRight(wShr0, wLoad, wShrReg, preg);
            Reg::ShiftLeft(wShl, wLoad, wShlReg, preg);
            Reg::ShiftRight(wShr1, wShl, wShrReg, preg);
            Reg::Select(wSel, wShr1, wShr0, pregVsel);
            Reg::And(wAnd, wSel, wAndReg, preg);

            Reg::AddrReg aregWeightB8Out = Reg::CreateAddrReg<uint8_t>(loopKIdx, mxA8W4NzParams.loopKDstStride,
                                                                       innerLoopIdx, mxA8W4NzParams.innerDstStride);
            Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_NORM_B8>(
                (__ubuf__ uint8_t *&)mxA8W4NzParams.weightHighBitPhyAddr, (Reg::RegTensor<uint8_t> &)wAnd,
                aregWeightB8Out, preg);
        }
    }
}

#endif // GMMSQ_VF_USE_MICRO_API

} // namespace GMMSQWeightQuant

#endif // GMMSQ_QUANT_MX_VF_H
