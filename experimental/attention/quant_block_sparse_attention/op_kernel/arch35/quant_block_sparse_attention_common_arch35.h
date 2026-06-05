/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_block_sparse_attention_common_arch35.h
 * \brief
 */
#ifndef QUANT_BLOCK_SPARSE_ATTENTION_COMMON_ARCH35_H
#define QUANT_BLOCK_SPARSE_ATTENTION_COMMON_ARCH35_H
#include <type_traits>
#include "kernel_tiling/kernel_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "stdarg.h"
#include "common/pse.h"
#include "common/util_regbase.h"

using matmul::MatmulType;
using namespace AscendC;
using namespace regbaseutil;

constexpr uint32_t NEGATIVE_MIN_VALUE_FP32 = 0xFF7FFFFF;
constexpr uint32_t NEGATIVE_MIN_VALUE_FP16 = 0xFBFF;
constexpr uint32_t POSITIVE_MAX_VALUE_FP32 = 0x7F7FFFFF;
constexpr uint32_t POSITIVE_MAX_VALUE_FP16 = 0x7BFF;
constexpr int64_t FP8_QUANT_BLOCK_SIZE = 128;

constexpr uint16_t SHIFT_NUM_6 = 6;
constexpr uint16_t ADD_NUM_63 = 63;

constexpr uint32_t CV_RATIO = 2;
constexpr uint64_t SYNC_MODE = 4;
constexpr uint64_t KB_TO_BYTES = 1024;
constexpr uint64_t L0C_SIZE = 256;
constexpr uint64_t FLOAT_BYTES = 4;

enum class ImplModeEnum {
    AA_HIGH_PRECISION = 0,
    AA_HIGH_PERFORMANCE = 1,
    AA_INVALID_LINE_HIGH_PRECISION = 2
};

namespace BaseApi {
struct CubeCoordInfo {
    uint32_t curBIdx;
    uint32_t s1Coord;
    uint32_t s2Coord;
};

__aicore__ constexpr uint16_t Align64Func(uint16_t data) {
    return (data + ADD_NUM_63) >> SHIFT_NUM_6 << SHIFT_NUM_6;
}

template <typename INPUT_T>
__aicore__ constexpr bool IsFp8WithRope(bool hasRope) {
    if constexpr (!IsSameType<INPUT_T, fp8_e5m2_t>::value &&
                  !IsSameType<INPUT_T, fp8_e4m3fn_t>::value &&
                  !IsSameType<INPUT_T, hifloat8_t>::value) {
        return false;
    }
    if (hasRope) {
        return true;
    }
    return false;
}

template <typename INPUT_T>
__aicore__ constexpr bool IsInt8WithRope(bool hasRope) {
    if constexpr (!IsSameType<INPUT_T, int8_t>::value) {
        return false;
    }
    if (hasRope) {
        return true;
    }
    return false;
}

template <typename INPUT_T>
__aicore__ constexpr bool IsFp8OnlyWithAttenMask(
    regbaseutil::PseTypeEnum pseMode, bool hasAtten, bool hasDrop) {
    if constexpr (!IsSameType<INPUT_T, fp8_e5m2_t>::value &&
                  !IsSameType<INPUT_T, fp8_e4m3fn_t>::value &&
                  !IsSameType<INPUT_T, hifloat8_t>::value) {
        return false;
    }
    if (pseMode == regbaseutil::PseTypeEnum::PSE_NONE_TYPE && hasAtten && !hasDrop) {
        return true;
    }
    return false;
}

template <typename INPUT_T>
__aicore__ constexpr bool IsInt8OnlyWithAttenMask(
    regbaseutil::PseTypeEnum pseMode, bool hasAtten, bool hasDrop) {
    if constexpr (!IsSameType<INPUT_T, int8_t>::value) {
        return false;
    }
    if (pseMode == regbaseutil::PseTypeEnum::PSE_NONE_TYPE && hasAtten && !hasDrop) {
        return true;
    }
    return false;
}

__aicore__ constexpr bool ContainOptionalInput(
    regbaseutil::PseTypeEnum pseMode, bool hasAtten, bool hasDrop) {
    if (pseMode == regbaseutil::PseTypeEnum::PSE_NONE_TYPE && !hasAtten && !hasDrop) {
        return false;
    } else {
        return true;
    }
}

__aicore__ constexpr bool IsDn(
    bool isFp32, bool isValidFp8, regbaseutil::PseTypeEnum pseMode, bool hasAtten, bool hasDrop, bool isS1Base64,
    regbaseutil::DTemplateType dTemplateType, bool hasRope, bool enableKVPrefix, bool isInfer, bool isHiFp8) {
    if (enableKVPrefix) {
        return false;
    }
    if (!isInfer && isHiFp8) {
        return false;
    }
    if (((!isFp32 && !ContainOptionalInput(pseMode, hasAtten, hasDrop)) ||
        (isValidFp8 && !hasDrop && pseMode == regbaseutil::PseTypeEnum::PSE_NONE_TYPE)) && !isS1Base64 &&
        (uint16_t)dTemplateType <= (uint16_t)regbaseutil::DTemplateType::Aligned256 && !hasRope) {
        return true;
    }
    return false;
}

template <typename INPUT_T>
__aicore__ constexpr bool UbOutCondition(
    bool isFp32, regbaseutil::PseTypeEnum pseMode, bool hasAtten, bool hasDrop, bool hasRope, bool isS2Base64) {
    if (IsFp8WithRope<INPUT_T>(hasRope) || IsInt8WithRope<INPUT_T>(hasRope)) {
        return true;
    }
    if (IsFp8OnlyWithAttenMask<INPUT_T>(pseMode, hasAtten, hasDrop) || IsInt8OnlyWithAttenMask<INPUT_T>(pseMode, hasAtten, hasDrop)) {
        return true;
    }
    if (!ContainOptionalInput(pseMode, hasAtten, hasDrop)) {
        if (!isS2Base64 || isFp32) {
            return true;
        }
    }
    return false;
}

__aicore__ constexpr TPosition GetC2Position(regbaseutil::DTemplateType dTemplateType, bool ubOutCondition, bool isNdS2Size256, bool isMlaFullQuant, bool isMlaNoQuant = false) {
    if (isMlaNoQuant) {
        return TPosition::VECCALC;
    }
    if ((uint16_t)dTemplateType <= (uint16_t)regbaseutil::DTemplateType::Aligned128 ||
        (ubOutCondition && (uint16_t)dTemplateType <= (uint16_t)regbaseutil::DTemplateType::Aligned192) ||
        (ubOutCondition && isMlaFullQuant) ||
        isNdS2Size256) {
        return TPosition::VECCALC;
    } else {
        return TPosition::GM;
    }
}
}

#include "quant_block_sparse_attention_attenmask.h"
#include "../quant_block_sparse_attention_common.h"

constexpr static int64_t SPARSE_MODE_INT_DEFAULT = 2147483647;
constexpr ImplModeEnum kBsaImplMode = ImplModeEnum::AA_HIGH_PRECISION;
constexpr PseTypeEnum kBsaPseMode = PseTypeEnum::PSE_NONE_TYPE;
constexpr bool kBsaHasDrop = false;
constexpr bool kBsaHasRope = false;
constexpr bool kBsaIsInfer = true;
constexpr bool kBsaEnableKVPrefix = false;

// INPUT_T - means data type for input
// T       - means data type when calc

#define CHILD_SPEC_TEMPLATE_ARGS \
    INPUT_T, T, kBsaImplMode, q_layout, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, \
    kBsaPseMode, hasAtten, kBsaHasDrop, kBsaHasRope, OUTPUT_T, kBsaIsInfer, isPa

#define TEMPLATE_INTF \
    template <typename INPUT_T, typename T, ImplModeEnum implMode, LayOutTypeEnum layout, \
    S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType, \
    DTemplateType dVTemplateType, PseTypeEnum pseMode, bool hasAtten, bool hasDrop, bool hasRope,\
    typename OUTPUT_T, bool isInfer, bool isPa, bool useDn, bool enableKVPrefix>

#define TEMPLATE_INTF_ARGS \
    INPUT_T, T, implMode, layout, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, \
    pseMode, hasAtten, hasDrop, hasRope, OUTPUT_T, isInfer, isPa, useDn, enableKVPrefix

#define CUBE_BLOCK_TRAITS_TYPE_FIELDS(X) \
    X(INPUT_T) \
    X(T) \
    X(OUTPUT_T)

#define CUBE_BLOCK_TRAITS_CONST_FIELDS(X) \
    X(q_layout, LayOutTypeEnum, LayOutTypeEnum::None) \
    X(kv_layout, BSAKvLayout, BSAKvLayout::PA_BNSD) \
    X(s1TemplateType, S1TemplateType, S1TemplateType::Aligned128) \
    X(s2TemplateType, S2TemplateType, S2TemplateType::Aligned128) \
    X(dTemplateType, DTemplateType, DTemplateType::Aligned128) \
    X(dVTemplateType, DTemplateType, DTemplateType::Aligned128) \
    X(hasAtten, bool, false) \
    X(isPa, bool, false) \
    X(enableS1OutSplit, bool, false)

/* 1. 生成带默认值的模版Template */
#define GEN_TYPE_PARAM(name) typename name,
#define GEN_CONST_PARAM(name, type, default_val) type name = default_val,

#define TEMPLATES_DEF \
template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TYPE_PARAM) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_CONST_PARAM) bool end = true>

/* 2. 生成不带带默认值的模版Template */
#define GEN_TEMPLATE_TYPE_NODEF(name) typename name,
#define GEN_TEMPLATE_CONST_NODEF(name, type, default_val) type name,
#define TEMPLATES_DEF_NO_DEFAULT \
template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TEMPLATE_TYPE_NODEF) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TEMPLATE_CONST_NODEF) bool end>

/* 3. 生成有默认值, 不带ChildClass的Args */
#define GEN_ARG_NAME(name, ...) name,
#define TEMPLATE_ARGS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARG_NAME) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARG_NAME) end

/* 4. 生成BASE的有默认值的Template, BASE带ChildClass*/
#define TEMPLATES_DEF_BASE \
template < \
    typename ChildClass, \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TYPE_PARAM) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_CONST_PARAM) bool end = true>

/* 5. 生成BASE的没有默认值的Template, BASE带ChildClass */
#define TEMPLATES_DEF_BASE_NO_DEFAULT \
template <\
    typename ChildClass, \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TEMPLATE_TYPE_NODEF) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TEMPLATE_CONST_NODEF) bool end>

/* 6. 生成BASE的BaseArgs, BASE带ChildClass */
#define TEMPLATE_BASE_ARGS \
    ChildClass, \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARG_NAME) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARG_NAME) end
#endif  // QUANT_BLOCK_SPARSE_ATTENTION_COMMON_ARCH35_H
