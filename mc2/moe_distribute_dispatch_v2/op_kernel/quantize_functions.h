/**
聽* Copyright (c) 2025 Huawei Technologies Co., Ltd.
聽* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
聽* CANN Open Software License Agreement Version 2.0 (the "License").
聽* Please refer to the License for details. You may not use this file except in compliance with the License.
聽* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
聽* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
聽* See LICENSE in the root of the software repository for the full text of the License.
聽*/

/*!
 * \file quantize_functions.h
 * \brief
 */

#ifndef QUANTIZE_FUNCTIONS_H
#define QUANTIZE_FUNCTIONS_H

#include "common.h"

namespace Quant {

constexpr int DIGIT_TWO = 2;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;
constexpr uint32_t MAX_EXP_FOR_FP8_IN_FP32 = 0x000000ff;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr int16_t SHR_NUM_FOR_FP32 = 23;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400; // elem_emax鍙崇Щ7浣?BF16E8M7)
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
constexpr uint16_t FP4_E2M1_BF16_MAX_EXP = 0x0100;
constexpr uint16_t FP4_E1M2_BF16_MAX_EXP = 0x0000;
constexpr uint16_t SPECIAL_VALUE_E2M1 = 0x00ff;
constexpr uint16_t SPECIAL_VALUE_E1M2 = 0x007f;
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
constexpr float FP8_E4M3_MAX_VALUE = 448.0f;
constexpr uint32_t FP8_E5M2_MAX = 0x37924925; // 1/57344鐨刦loat32琛ㄧず 57344鏄疎5M2鎵€鑳借〃绀虹殑鏈€澶у€?constexpr uint32_t FP8_E4M3_MAX = 0x3b124925; // 1/448鐨刦loat32琛ㄧず 448鏄疎4M3鎵€鑳借〃绀虹殑鏈€澶у€?constexpr uint32_t NUMBER_ZERO = 0x00000000;
constexpr uint32_t FP32_BIASED_EXP_MAX_NORMAL = 0x000000fe;
constexpr uint32_t FP32_MANTISSA_HALF = 0x00400000;
constexpr float HIFP8_MAX_VALUE = 32768.0f;
constexpr float INT8_MAX_VALUE = 127.0f;
constexpr uint16_t INVALID_FLOAT16 = 0x7c00;
constexpr uint16_t NEW_MANTISSA = 0x0008;
constexpr uint16_t NAN_CUSTOMIZATION_PACK = 0x00007f81;
constexpr uint32_t MAN_MASK_FLOAT = 0x007fffff;
constexpr uint32_t FP32_EXP_BIAS_CUBLAS = 0x00007f00;
constexpr uint16_t ABS_MASK_FOR_16BIT = 0x7fff;
constexpr uint32_t EPS_1E_4_FP32 = 0x38D1B717; // 1e-4 鐨?FP32 bit pattern锛岀敤浜?cuBLAS scale 璺緞涓嬮檺 clamp

using namespace AscendC;

__aicore__ inline constexpr uint32_t GetUbBlockSizeDispatch()
{
    return 32U;
}

__aicore__ inline constexpr uint32_t GetVRegSizeDispatch()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

template <typename T>
__aicore__ inline void ComputeMaxExp(__ubuf__ T *srcAddr, __ubuf__ uint16_t *maxExpAddr, uint32_t totalCountInUB)
{
    uint32_t vlForHalfNumber = GetVRegSizeDispatch() / sizeof(T); // 姣忎釜鍚戦噺瀵勫瓨鍣ㄥ彲浠ュ瓨鍌ㄧ殑鍏冪礌涓暟
    uint16_t elementAfterReduce = GetVRegSizeDispatch() / GetUbBlockSizeDispatch(); // Reduce鎿嶄綔鍚庢惉鍑虹殑鍏冪礌涓暟
    uint16_t loopNum = Ceil(totalCountInUB, 2 * vlForHalfNumber);

    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vdExp0;
        Reg::RegTensor<T> vdExp1;
        Reg::RegTensor<bfloat16_t> vdExp0BF16;
        Reg::RegTensor<bfloat16_t> vdExp1BF16;
        Reg::RegTensor<uint16_t> vdExpSelect0;
        Reg::RegTensor<uint16_t> vdExpSelect1;
        Reg::RegTensor<uint16_t> vdExpExtract0;
        Reg::RegTensor<uint16_t> vdExpExtract1;

        Reg::RegTensor<uint16_t> expMaskBF16;
        Reg::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);

        Reg::RegTensor<uint16_t> invalidMaskFP16;
        Reg::Duplicate(invalidMaskFP16, INVALID_FLOAT16);
        Reg::RegTensor<uint16_t> vdMaxExp;
        Reg::MaskReg scaleMask1;
        Reg::MaskReg scaleMask2;
        Reg::MaskReg invalidDataMask0;
        Reg::MaskReg invalidDataMask1;
        Reg::UnalignReg u1;
        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {
            Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
            RoundMode::CAST_TRUNC};
        for (uint16_t i = 0; i < loopNum; i++) {
            scaleMask1 = Reg::UpdateMask<T>(totalCountInUB);
            scaleMask2 = Reg::UpdateMask<T>(totalCountInUB);
            // 鍙屾惉锛屽皢鏁版嵁浜ょ粐鎼繍鍒颁袱涓悜閲忓瘎瀛樺櫒涓?            Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, vlForHalfNumber * DIGIT_TWO);
            if constexpr (Std::IsSame<T, half>::value) {
                Reg::And(vdExpSelect0, (Reg::RegTensor<uint16_t> &)vdExp0, invalidMaskFP16,
                              scaleMask1); // 灏咶P16闈炴甯告寚鏁颁綅涓庢瘡涓厓绱犵殑鎸囨暟浣嶈繘琛屼笌鎿嶄綔锛屾秷闄ゅ熬鏁颁綅
                Reg::And(vdExpSelect1, (Reg::RegTensor<uint16_t> &)vdExp1, invalidMaskFP16, scaleMask1);
                // 灏咶P16闈炴甯告寚鏁颁綅涓庡疄闄呮寚鏁颁綅浣滃姣旓紝鐢熸垚闈炴甯告寚鏁颁綅鎺╃爜
                Reg::Compare<uint16_t, CMPMODE::NE>(invalidDataMask0, vdExpSelect0, invalidMaskFP16, scaleMask1);
                Reg::Compare<uint16_t, CMPMODE::NE>(invalidDataMask1, vdExpSelect1, invalidMaskFP16, scaleMask1);
                Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, scaleMask1);
                Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, scaleMask1);
                Reg::And(vdExpExtract0, (Reg::RegTensor<uint16_t> &)vdExp0BF16, expMaskBF16,
                              scaleMask1); // 涓庢搷浣滀繚鐣欐寚鏁颁綅
                Reg::And(vdExpExtract1, (Reg::RegTensor<uint16_t> &)vdExp1BF16, expMaskBF16, scaleMask1);
                // 绛涢€夋甯告寚鏁颁綅锛岄潪姝ｅ父鍊煎垯浣跨敤expMaskBF16鏇夸唬
                Reg::Select<uint16_t>(vdExpExtract0, vdExpExtract0, expMaskBF16, invalidDataMask0);
                Reg::Select<uint16_t>(vdExpExtract1, vdExpExtract1, expMaskBF16, invalidDataMask1);
            } else {
                Reg::And(vdExpExtract0, (Reg::RegTensor<uint16_t> &)vdExp0, expMaskBF16, scaleMask1);
                Reg::And(vdExpExtract1, (Reg::RegTensor<uint16_t> &)vdExp1, expMaskBF16, scaleMask1);
            }
            // 涓や釜鍚戦噺瀵勫瓨鍣ㄤ笂鐨勫厓绱犱竴涓€瀵规瘮杈撳嚭鏈€澶ф寚鏁颁綅
            Reg::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
            // 鎸夋瘡涓猙lock杈撳嚭涓€涓渶澶ф寚鏁颁綅
            Reg::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);

            Reg::DataCopyUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(maxExpAddr, vdMaxExp, u1,
                                                                                         elementAfterReduce);
        }
        Reg::DataCopyUnAlignPost(maxExpAddr, u1, 0);
    }
}

template <typename T>
__aicore__ inline void ComputeMaxExpClip(__ubuf__ T *srcAddr, __ubuf__ uint16_t *maxExpAddr, uint32_t totalCountInUB)
{
    uint32_t vlForHalfNumber = GetVRegSizeDispatch() / sizeof(T); // 姣忎釜鍚戦噺瀵勫瓨鍣ㄥ彲浠ュ瓨鍌ㄧ殑鍏冪礌涓暟
    uint16_t elementAfterReduce = GetVRegSizeDispatch() / GetUbBlockSizeDispatch(); // Reduce鎿嶄綔鍚庢惉鍑虹殑鍏冪礌涓暟
    uint16_t loopNum = Ceil(totalCountInUB, 2 * vlForHalfNumber);
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vdExp0;
        Reg::RegTensor<T> vdExp1;
        Reg::RegTensor<uint16_t> vdMaxExp;

        Reg::RegTensor<uint16_t> absMask16Bit;
        Reg::Duplicate(absMask16Bit, ABS_MASK_FOR_16BIT);

        Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::UnalignReg ureg;

        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, vlForHalfNumber * DIGIT_TWO);
            Reg::And((Reg::RegTensor<uint16_t> &)vdExp0, (Reg::RegTensor<uint16_t> &)vdExp0,
                          absMask16Bit, Mask);
            Reg::And((Reg::RegTensor<uint16_t> &)vdExp1, (Reg::RegTensor<uint16_t> &)vdExp1,
                          absMask16Bit, Mask);
            Reg::Max(vdMaxExp, (Reg::RegTensor<uint16_t> &)vdExp0, (Reg::RegTensor<uint16_t> &)vdExp1,
                          Mask);
            Reg::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, Mask);
            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(maxExpAddr, vdMaxExp, ureg,
                                                                                      elementAfterReduce);
        }
        Reg::StoreUnAlignPost(maxExpAddr, ureg, 0);
    }
}

template <typename T>
__aicore__ inline void ComputeScale(__ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr,
                                    __ubuf__ uint16_t *halfScaleLocalAddr, uint32_t totalScaleInUB)
{
    uint32_t vlForHalfNumber = GetVRegSizeDispatch() / sizeof(uint16_t);
    uint16_t loopNumScale = Ceil(totalScaleInUB, vlForHalfNumber);
    uint16_t maxExponent;
    if constexpr (Std::IsSame<T, fp8_e4m3fn_t>::value) {
        maxExponent = FP8_E4M3_MAX_EXP;
    } else if constexpr (Std::IsSame<T, fp8_e5m2_t>::value) {
        maxExponent = FP8_E5M2_MAX_EXP;
    } else if constexpr (Std::IsSame<T, fp4x2_e2m1_t>::value) {
        maxExponent = FP4_E2M1_BF16_MAX_EXP;
    } else {
        maxExponent = FP4_E1M2_BF16_MAX_EXP;
    }

    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> expMask, vdMaxExp;
        Reg::Duplicate(expMask, MAX_EXP_FOR_BF16);
        Reg::MaskReg cmpResult, zeroMask, preMaskScale;
        Reg::RegTensor<uint16_t> maxExpValue;
        Reg::Duplicate(maxExpValue, maxExponent);
        Reg::RegTensor<uint16_t> sharedExp, scaleValue, scaleBias;
        Reg::Duplicate(scaleBias, BF16_EXP_BIAS);
        Reg::RegTensor<uint16_t> halfScale, fp8NanRegTensor;
        Reg::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        Reg::RegTensor<uint16_t> zeroRegTensor;
        Reg::Duplicate(zeroRegTensor, 0);
        Reg::RegTensor<uint16_t> nanRegTensor;
        Reg::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        Reg::MaskReg invalidDataMask, specialDataMask;
        Reg::RegTensor<uint16_t> specialExpRegTensor;
        Reg::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
        for (uint16_t i = 0; i < loopNumScale; i++) {
            preMaskScale = Reg::UpdateMask<uint16_t>(totalScaleInUB);
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(vdMaxExp, maxExpAddr,
                                                                                  vlForHalfNumber);
            // 妫€娴嬮潪姝ｅ父鍊?            Reg::Compare<uint16_t, CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale);
            // 妫€娴嬮浂鍊?            Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
            // 妫€娴嬭秴鍑虹洰鏍囨牸寮忕殑鏈€澶у€?            Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);
            // 闄愬埗鏈€澶ф寚鏁颁笉瓒呰繃鐩爣鏍煎紡鏈€澶ф寚鏁?            Reg::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);
            // 璁＄畻鐩稿鎸囨暟宸€?            Reg::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
            // 鍙崇Щ寰楀埌缂╂斁鍊?            Reg::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
            // 鐗规畩鍊煎鐞?            Reg::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            Reg::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr, scaleValue, vlForHalfNumber / DIGIT_TWO, preMaskScale);

            Reg::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
            Reg::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            Reg::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            Reg::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            Reg::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(halfScaleLocalAddr, halfScale,
                                                                                  vlForHalfNumber, preMaskScale);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void ComputeScaleClip(__ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr,
                                        __ubuf__ uint16_t *halfScaleLocalAddr, uint32_t totalScaleInUB)
{
    uint32_t vlForHalfNumber = GetVRegSizeDispatch() / sizeof(uint32_t);
    uint16_t loopNumScale = Ceil(totalScaleInUB, vlForHalfNumber);
    uint32_t dtypeMax;
    if constexpr (Std::IsSame<T, fp8_e4m3fn_t>::value) {
        dtypeMax = FP8_E4M3_MAX;
    } else if constexpr (Std::IsSame<T, fp8_e5m2_t>::value) {
        dtypeMax = FP8_E5M2_MAX;
    }
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> max16;
        Reg::RegTensor<uint32_t> max32;
        Reg::RegTensor<uint32_t> exp32;
        Reg::RegTensor<uint32_t> man32;
        Reg::RegTensor<uint32_t> normalExp32;
        Reg::RegTensor<uint32_t> expAddOne32;
        Reg::RegTensor<uint32_t> extractExp;
        Reg::RegTensor<uint16_t> expOut;
        Reg::RegTensor<uint32_t> halfScale;
        Reg::RegTensor<uint16_t> recExpOut;

        Reg::RegTensor<uint32_t> invMax;
        Reg::Duplicate(invMax, dtypeMax);
        Reg::RegTensor<uint32_t> manMaskFP32;
        Reg::Duplicate(manMaskFP32, MAN_MASK_FLOAT);
        Reg::RegTensor<uint32_t> expMask;
        Reg::Duplicate(expMask, MAX_EXP_FOR_FP32);
        Reg::RegTensor<uint32_t> zeroRegTensor32;
        Reg::Duplicate(zeroRegTensor32, 0);
        Reg::RegTensor<uint32_t> scaleBias;
        Reg::Duplicate(scaleBias, FP32_EXP_BIAS_CUBLAS);
        Reg::RegTensor<uint32_t> nanRegTensor;
        Reg::Duplicate(nanRegTensor, NAN_CUSTOMIZATION_PACK);
        Reg::RegTensor<uint32_t> fp8NanRegTensor;
        Reg::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8_IN_FP32);
        Reg::RegTensor<uint32_t> epsTensor;
        Reg::Duplicate(epsTensor, EPS_1E_4_FP32);

        Reg::MaskReg cmpResult;
        Reg::MaskReg zeroMask;
        Reg::MaskReg p0;
        Reg::MaskReg p1;
        Reg::MaskReg p2;
        // Reg::MaskReg maskHalf = Reg::CreateMask<uint16_t>();
        uint32_t B16_HALF_MASK_ELEMENT_NUM = 64;
        Reg::MaskReg dataMaskB16Half = Reg::UpdateMask<uint16_t>(B16_HALF_MASK_ELEMENT_NUM);
        Reg::MaskReg maskFloat = Reg::CreateMask<uint32_t>();

        static constexpr Reg::CastTrait castTraitHalf2Float = {
            Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
            RoundMode::UNKNOWN};
        for (uint16_t i = 0; i < loopNumScale; i++) {
            // 鍗曟惉 64 涓暟
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(
                max16, maxExpAddr, vlForHalfNumber);

            Reg::Cast<float, U, castTraitHalf2Float>((Reg::RegTensor<float> &)max32,
                                                          (Reg::RegTensor<U> &)max16, maskFloat);
            Reg::Compare<uint32_t, CMPMODE::LT>(cmpResult, max32, expMask, maskFloat);
            Reg::Compare<uint32_t, CMPMODE::NE>(zeroMask, max32, zeroRegTensor32, maskFloat);

            // 1e-4 涓嬮檺 clamp锛氭姂鍒?block 鍐?max_abs 杩囧皬瀵艰嚧 scale 鍣０鏀惧ぇ
            // 娉ㄦ剰锛歝mpResult/zeroMask 宸插湪 clamp 鍓嶇敤鍘熷 max32 绠楀ソ锛岀壒娈婅矾寰勪笉鍙楀奖鍝?            Reg::Max((Reg::RegTensor<float> &)max32, (Reg::RegTensor<float> &)max32,
                          (Reg::RegTensor<float> &)epsTensor, maskFloat);
            // 绛変环鐨勬暣鏁?Max 鐗堟湰锛坢ax32 鏉ヨ嚜 |x|锛岄潪璐?IEEE 娴偣 bit pattern 涓庡€煎簭涓€鑷达級锛?            // Reg::Max(max32, max32, epsTensor, maskFloat);

            Reg::Mul((Reg::RegTensor<float> &)max32, (Reg::RegTensor<float> &)max32,
                          (Reg::RegTensor<float> &)invMax, maskFloat);
            Reg::ShiftRights(exp32, max32, SHR_NUM_FOR_FP32, maskFloat);
            Reg::And(man32, max32, manMaskFP32, maskFloat);

            Reg::CompareScalar<uint32_t, CMPMODE::GT>(p0, exp32, NUMBER_ZERO, maskFloat);
            Reg::CompareScalar<uint32_t, CMPMODE::LT>(p1, exp32, FP32_BIASED_EXP_MAX_NORMAL, maskFloat);
            Reg::CompareScalar<uint32_t, CMPMODE::GT>(p2, man32, NUMBER_ZERO, maskFloat);
            Reg::MaskAnd(p0, p0, p1, maskFloat);
            Reg::MaskAnd(p0, p0, p2, maskFloat);

            Reg::CompareScalar<uint32_t, CMPMODE::EQ>(p1, exp32, NUMBER_ZERO, maskFloat);
            Reg::CompareScalar<uint32_t, CMPMODE::GT>(p2, man32, FP32_MANTISSA_HALF, maskFloat);
            Reg::MaskAnd(p1, p1, p2, maskFloat);
            Reg::MaskOr(p0, p0, p1, maskFloat);

            Reg::Adds(expAddOne32, exp32, 1, maskFloat);
            Reg::Select(extractExp, expAddOne32, exp32, p0);
            Reg::Select<uint32_t>(extractExp, extractExp, fp8NanRegTensor, cmpResult);
            Reg::Select<uint32_t>(extractExp, extractExp, zeroRegTensor32, zeroMask);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(expOut, extractExp);
            // 64 涓?max 璁＄畻寰楀埌 64 * 1 Bytes = 32 * sizeof(uint16_t) Bytes
            Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr + i * 32, expOut,
                                                                               dataMaskB16Half);

            Reg::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, maskFloat);
            Reg::Sub(halfScale, scaleBias, extractExp, maskFloat);
            Reg::Select<uint32_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            Reg::Select<uint32_t>(halfScale, halfScale, zeroRegTensor32, zeroMask);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(recExpOut, halfScale);

            Reg::StoreAlign<uint16_t>(halfScaleLocalAddr + i * vlForHalfNumber, recExpOut, dataMaskB16Half);
        }
    }
}

template <typename T, typename U, RoundMode toBf16RoundMode, RoundMode roundMode>
__aicore__ inline void ComputeFp8Data(__ubuf__ T *srcAddr, __ubuf__ uint16_t *halfScaleLocalAddr,
                                      __ubuf__ int8_t *outLocalAddr, uint32_t totalCountInUB)
{
    uint32_t vlForHalfNumber = GetVRegSizeDispatch() / sizeof(T);
    uint16_t elementAfterReduce = GetVRegSizeDispatch() / GetUbBlockSizeDispatch();
    uint32_t totalCountInUB2 = totalCountInUB * DIGIT_TWO;
    uint16_t loopNum = Ceil(totalCountInUB, 2 * vlForHalfNumber);
    __VEC_SCOPE__
    {
        Reg::MaskReg dataMask1;
        Reg::MaskReg dataMask2;
        Reg::MaskReg dataMask3;
        Reg::MaskReg dataMask4;
        Reg::MaskReg maskAll = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::RegTensor<uint16_t> halfScaleForMul;
        Reg::RegTensor<float> floatScaleForMul;
        Reg::RegTensor<T> vdExp0;
        Reg::RegTensor<T> vdExp1;

        Reg::RegTensor<float> vdExp0FP32Zero;
        Reg::RegTensor<float> vdExp0FP32One;
        Reg::RegTensor<float> vdExp1FP32Zero;
        Reg::RegTensor<float> vdExp1FP32One;
        Reg::RegTensor<U> vdExp0FP8Zero;
        Reg::RegTensor<U> vdExp0FP8One;
        Reg::RegTensor<U> vdExp1FP8Zero;
        Reg::RegTensor<U> vdExp1FP8One;
        // 鏀惧埌绱㈠紩浣嶇疆0
        static constexpr Reg::CastTrait castTraitZero = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                              Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        // 鏀惧埌绱㈠紩浣嶇疆1
        static constexpr Reg::CastTrait castTraitOne = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                             Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTrait32to8 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                               Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = Reg::UpdateMask<T>(totalCountInUB);
            dataMask2 = Reg::UpdateMask<T>(totalCountInUB);
            dataMask3 = Reg::UpdateMask<T>(totalCountInUB2);
            dataMask4 = Reg::UpdateMask<T>(totalCountInUB2);
            Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, vlForHalfNumber * DIGIT_TWO);
            // 杩欓噷DIST_E2B_B16鏄皢姣忎釜16bit鍏冪礌骞挎挱鍒颁竴涓狣ataBlock(32B)涓?            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                halfScaleForMul, halfScaleLocalAddr, elementAfterReduce);
            // 鍥犱负鍓嶉潰scale璁＄畻鐢ㄧ殑BF16绫诲瀷锛屾墍浠ュ浜巉p16闇€瑕佸厛cast鍒癴loat鍐嶈绠?            if constexpr (Std::IsSame<T, half>::value) {
                // 鍙栧伓鏁扮储寮曠殑鍏冪礌Cast鍒皏dExp0FP32Zero
                Reg::Cast<float, T, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
                // 鍙栧鏁扮储寮曠殑鍏冪礌Cast鍒皏dExp0FP32One
                Reg::Cast<float, T, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
                // 鐢变簬鍓嶉潰鎼叆鏄竴涓箍鎾搷浣滐紝鎵€浠ョ洿鎺ュ彇鍋舵暟绱㈠紩鍏冪礌鍗冲彲
                Reg::Cast<float, bfloat16_t, castTraitZero>(
                    floatScaleForMul, (Reg::RegTensor<bfloat16_t> &)halfScaleForMul, maskAll);
                Reg::Mul(vdExp0FP32Zero, vdExp0FP32Zero, floatScaleForMul, dataMask3);
                Reg::Mul(vdExp0FP32One, vdExp0FP32One, floatScaleForMul, dataMask4);
                Reg::Interleave(vdExp0FP32Zero, vdExp0FP32One, vdExp0FP32Zero, vdExp0FP32One);
                Reg::Cast<float, T, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask1);
                Reg::Cast<float, T, castTraitOne>(vdExp1FP32One, vdExp1, dataMask1);
                Reg::Mul(vdExp1FP32Zero, vdExp1FP32Zero, floatScaleForMul, dataMask3);
                Reg::Mul(vdExp1FP32One, vdExp1FP32One, floatScaleForMul, dataMask4);
                Reg::Interleave(vdExp1FP32Zero, vdExp1FP32One, vdExp1FP32Zero, vdExp1FP32One);
                Reg::Interleave(vdExp0FP32Zero, vdExp1FP32Zero, vdExp0FP32Zero, vdExp1FP32Zero);
                Reg::Interleave(vdExp0FP32One, vdExp1FP32One, vdExp0FP32One, vdExp1FP32One);
                Reg::Cast<U, float, castTrait32to8>(vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
                Reg::Cast<U, float, castTrait32to8>(vdExp0FP8One, vdExp1FP32Zero, dataMask3);
                Reg::Cast<U, float, castTrait32to8>(vdExp1FP8Zero, vdExp0FP32One, dataMask4);
                Reg::Cast<U, float, castTrait32to8>(vdExp1FP8One, vdExp1FP32One, dataMask4);
            } else {
                Reg::Mul(vdExp0, vdExp0, (Reg::RegTensor<T> &)halfScaleForMul, dataMask1);
                Reg::Mul(vdExp1, vdExp1, (Reg::RegTensor<T> &)halfScaleForMul, dataMask1);
                Reg::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
                Reg::Cast<float, T, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
                Reg::Cast<float, T, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
                Reg::Interleave(vdExp0FP32Zero, vdExp0FP32One, vdExp0FP32Zero, vdExp0FP32One);
                Reg::Cast<U, float, castTrait32to8>(vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
                Reg::Cast<U, float, castTrait32to8>(vdExp0FP8One, vdExp0FP32One, dataMask3);
                Reg::Cast<float, T, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask2);
                Reg::Cast<float, T, castTraitOne>(vdExp1FP32One, vdExp1, dataMask2);
                Reg::Interleave(vdExp1FP32Zero, vdExp1FP32One, vdExp1FP32Zero, vdExp1FP32One);
                Reg::Cast<U, float, castTrait32to8>(vdExp1FP8Zero, vdExp1FP32Zero, dataMask4);
                Reg::Cast<U, float, castTrait32to8>(vdExp1FP8One, vdExp1FP32One, dataMask4);
            }
            Reg::DataCopy<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (Reg::RegTensor<int8_t> &)vdExp0FP8Zero, OUT_ELE_NUM_ONE_BLK, dataMask3);
            Reg::DataCopy<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (Reg::RegTensor<int8_t> &)vdExp0FP8One, OUT_ELE_NUM_ONE_BLK, dataMask3);
            Reg::DataCopy<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (Reg::RegTensor<int8_t> &)vdExp1FP8Zero, OUT_ELE_NUM_ONE_BLK, dataMask4);
            Reg::DataCopy<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (Reg::RegTensor<int8_t> &)vdExp1FP8One, OUT_ELE_NUM_ONE_BLK, dataMask4);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void FP16Convert(Reg::RegTensor<half> &output, Reg::RegTensor<half> &input,
                                   Reg::MaskReg &mask)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> specialValueTensor;
        Reg::RegTensor<uint16_t> newMantissa;
        Reg::RegTensor<uint16_t> andResult;
        Reg::RegTensor<uint16_t> newValue;
        Reg::MaskReg specialMask;
        Reg::MaskReg nonzeroMask;
        uint16_t specialValue = SPECIAL_VALUE_E1M2;
        if constexpr (Std::IsSame<U, fp4x2_e2m1_t>::value) {
            specialValue = SPECIAL_VALUE_E2M1;
        }
        Reg::Duplicate(specialValueTensor, specialValue);
        Reg::Duplicate(newMantissa, NEW_MANTISSA);
        Reg::And(andResult, (Reg::RegTensor<uint16_t> &)input, specialValueTensor, mask);
        Reg::CompareScalar<uint16_t, CMPMODE::GT>(nonzeroMask, andResult, 0, mask);
        Reg::CompareScalar<uint16_t, CMPMODE::LT>(specialMask, andResult, NEW_MANTISSA, mask);
        Reg::MaskAnd(specialMask, specialMask, nonzeroMask, mask);
        Reg::Or(newValue, (Reg::RegTensor<uint16_t> &)input, newMantissa, mask);
        Reg::Select<uint16_t>((Reg::RegTensor<uint16_t> &)output, newValue,
                                   (Reg::RegTensor<uint16_t> &)input, specialMask);
    }
}

template <typename T, typename U, RoundMode toBf16RoundMode, RoundMode roundMode>
__aicore__ inline void ComputeFp4Data(__ubuf__ T *srcAddr, __ubuf__ uint16_t *halfScaleLocalAddr,
                                      __ubuf__ int8_t *outLocalAddr, uint32_t totalCountInUB)
{
    uint32_t vlForHalfNumber = GetVRegSizeDispatch() / sizeof(T);
    uint16_t elementAfterReduce = GetVRegSizeDispatch() / GetUbBlockSizeDispatch();
    uint16_t loopNum = Ceil(totalCountInUB, 2 * vlForHalfNumber);
    __VEC_SCOPE__
    {
        Reg::MaskReg dataMask1;
        Reg::RegTensor<uint16_t> halfScaleForMul;
        Reg::RegTensor<T> vdExp0;
        Reg::RegTensor<T> vdExp1;

        Reg::RegTensor<bfloat16_t> vdExp0BF16;
        Reg::RegTensor<bfloat16_t> vdExp1BF16;

        Reg::RegTensor<U> vdExp0FP4;
        Reg::RegTensor<U> vdExp1FP4;

        static constexpr Reg::CastTrait castTrait = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                          Reg::MaskMergeMode::ZEROING, roundMode};
        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {Reg::RegLayout::UNKNOWN,
                                                                   Reg::SatMode::UNKNOWN,
                                                                   Reg::MaskMergeMode::ZEROING, toBf16RoundMode};
        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = Reg::UpdateMask<T>(totalCountInUB);
            Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, srcAddr, vlForHalfNumber * DIGIT_TWO);
            Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                halfScaleForMul, halfScaleLocalAddr, elementAfterReduce);
            if constexpr (Std::IsSame<T, half>::value) {
                if constexpr (roundMode == RoundMode::CAST_RINT) {
                    FP16Convert<T, U>(vdExp0, vdExp0, dataMask1);
                    FP16Convert<T, U>(vdExp1, vdExp1, dataMask1);
                }
                Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, dataMask1);
                Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, dataMask1);
                Reg::Mul(vdExp0BF16, vdExp0BF16, (Reg::RegTensor<bfloat16_t> &)halfScaleForMul, dataMask1);
                Reg::Mul(vdExp1BF16, vdExp1BF16, (Reg::RegTensor<bfloat16_t> &)halfScaleForMul, dataMask1);
                Reg::Interleave(vdExp0BF16, vdExp1BF16, vdExp0BF16, vdExp1BF16);
                Reg::Cast<U, bfloat16_t, castTrait>(vdExp0FP4, vdExp0BF16, dataMask1);
                Reg::Cast<U, bfloat16_t, castTrait>(vdExp1FP4, vdExp1BF16, dataMask1);
            } else {
                Reg::Mul(vdExp0, vdExp0, (Reg::RegTensor<T> &)halfScaleForMul, dataMask1);
                Reg::Mul(vdExp1, vdExp1, (Reg::RegTensor<T> &)halfScaleForMul, dataMask1);
                Reg::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
                Reg::Cast<U, T, castTrait>(vdExp0FP4, vdExp0, dataMask1);
                Reg::Cast<U, T, castTrait>(vdExp1FP4, vdExp1, dataMask1);
            }
            Reg::DataCopy<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (Reg::RegTensor<int8_t> &)vdExp0FP4, OUT_ELE_NUM_ONE_BLK, dataMask1);
            Reg::DataCopy<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (Reg::RegTensor<int8_t> &)vdExp1FP4, OUT_ELE_NUM_ONE_BLK, dataMask1);
        }
    }
}

template <typename T, typename U, RoundMode RMode, bool HasSmooth>
__aicore__ inline void ComputePerTileDynamic(__ubuf__ T *srcAddr, __ubuf__ float *smoothLocalAddr,
                                             __ubuf__ float *scaleOutLocalAddr, __ubuf__ int8_t *outLocalAddr,
                                             uint32_t totalCountInUB)
{
    uint32_t vlB16 = GetVRegSizeDispatch() / sizeof(T);
    uint32_t vlB32 = GetVRegSizeDispatch() / sizeof(float);
    uint16_t loopNum = Ceil(totalCountInUB, vlB16);
    uint32_t totalCntForB32 = totalCountInUB;
    float maxVal = 0.0f;
    float invMaxVal = 0.0f;
    if constexpr (Std::IsSame<U, fp8_e5m2_t>::value) {
        maxVal = FP8_E5M2_MAX_VALUE;
        invMaxVal = 1.0f / FP8_E5M2_MAX_VALUE;
    } else if constexpr (Std::IsSame<U, fp8_e4m3fn_t>::value) {
        maxVal = FP8_E4M3_MAX_VALUE;
        invMaxVal = 1.0f / FP8_E4M3_MAX_VALUE;
    } else if constexpr (Std::IsSame<U, hifloat8_t>::value) {
        maxVal = HIFP8_MAX_VALUE;
        invMaxVal = 1.0f / HIFP8_MAX_VALUE;
    } else if constexpr (Std::IsSame<U, int8_t>::value) {
        maxVal = INT8_MAX_VALUE;
        invMaxVal = 1.0f / INT8_MAX_VALUE;
    }

    __VEC_SCOPE__
    {
        Reg::MaskReg dataMask1;
        Reg::MaskReg dataMask2;
        Reg::MaskReg dataMask3;
        Reg::MaskReg maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskOne = Reg::CreateMask<float, Reg::MaskPattern::VL1>();

        Reg::RegTensor<T> vInB16;

        Reg::RegTensor<float> vInFP32Zero;
        Reg::RegTensor<float> vInFP32One;
        Reg::RegTensor<float> vSmooth0;
        Reg::RegTensor<float> vSmooth1;
        Reg::RegTensor<float> vTileMax;
        Reg::RegTensor<float> vDynScale;
        Reg::RegTensor<float> vOutputScale;
        Reg::RegTensor<float> vMaxVal;
        Reg::RegTensor<float> vInvMaxVal;

        Reg::RegTensor<U> vOut0;
        Reg::RegTensor<U> vOut1;

        Reg::Duplicate(vMaxVal, maxVal);
        Reg::Duplicate(vInvMaxVal, invMaxVal);

        static constexpr Reg::DivSpecificMode divMode = {Reg::MaskMergeMode::ZEROING, true};

        static constexpr Reg::CastTrait castTraitZero = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                              Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitOne = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                             Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        constexpr static Reg::CastTrait castTrait32tof8 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                                Reg::MaskMergeMode::ZEROING, RMode};

        constexpr static Reg::CastTrait castTrait32tof16 = {
            Reg::RegLayout::ZERO, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

        constexpr static Reg::CastTrait castTrait16toi8 = {
            Reg::RegLayout::ZERO, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};

        for (uint16_t i = 0; i < loopNum; i++) {
            dataMask1 = Reg::UpdateMask<T>(totalCountInUB);
            dataMask2 = Reg::UpdateMask<float>(totalCntForB32);
            dataMask3 = Reg::UpdateMask<float>(totalCntForB32);

            Reg::DataCopy(vInB16, srcAddr + i * vlB16);
            Reg::Cast<float, T, castTraitZero>(vInFP32Zero, vInB16, dataMask1);
            Reg::Cast<float, T, castTraitOne>(vInFP32One, vInB16, dataMask1);
            if constexpr (HasSmooth) {
                Reg::DataCopy<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                    vSmooth0, vSmooth1, smoothLocalAddr, vlB32 * DIGIT_TWO);
                Reg::Mul(vInFP32Zero, vInFP32Zero, vSmooth0, maskAll);
                Reg::Mul(vInFP32One, vInFP32One, vSmooth1, maskAll);
            }
            Reg::Interleave(vSmooth0, vSmooth1, vInFP32Zero, vInFP32One);
            Reg::Abs(vInFP32Zero, vSmooth0, maskAll);
            Reg::Abs(vInFP32One, vSmooth1, maskAll);
            Reg::Max(vTileMax, vInFP32Zero, vInFP32One, maskAll);
            Reg::ReduceMax(vTileMax, vTileMax, dataMask2);
            Reg::Duplicate(vTileMax, vTileMax, maskAll);
            Reg::Div<float, &divMode>(vDynScale, vMaxVal, vTileMax, maskAll);
            Reg::Mul(vSmooth0, vSmooth0, vDynScale, maskAll);
            Reg::Mul(vSmooth1, vSmooth1, vDynScale, maskAll);

            if constexpr (Std::IsSame<U, int8_t>::value) {
                Reg::RegTensor<half> vHalf0;
                Reg::RegTensor<half> vHalf1;
                Reg::Cast<half, float, castTrait32tof16>(vHalf0, vSmooth0, maskAll);
                Reg::Cast<half, float, castTrait32tof16>(vHalf1, vSmooth1, maskAll);
                Reg::Cast<U, half, castTrait16toi8>(vOut0, vHalf0, maskAll);
                Reg::Cast<U, half, castTrait16toi8>(vOut1, vHalf1, maskAll);
            } else {
                Reg::Cast<U, float, castTrait32tof8>(vOut0, vSmooth0, dataMask2);
                Reg::Cast<U, float, castTrait32tof8>(vOut1, vSmooth1, dataMask3);
            }

            Reg::DataCopy<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (Reg::RegTensor<int8_t> &)vOut0, OUT_ELE_NUM_ONE_BLK, dataMask2);
            Reg::DataCopy<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                outLocalAddr, (Reg::RegTensor<int8_t> &)vOut1, OUT_ELE_NUM_ONE_BLK, dataMask3);

            Reg::Mul(vOutputScale, vTileMax, vInvMaxVal, maskOne);
            Reg::DataCopy<float, Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(scaleOutLocalAddr + i, vOutputScale,
                                                                                   maskOne);
        }
    }
}

} // namespace Quant
#endif
