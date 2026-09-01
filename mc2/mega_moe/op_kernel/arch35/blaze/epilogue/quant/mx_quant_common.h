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
 * \file mx_quant_common.h
 * \brief MX 量化公共参数、常量和转换属性。
 */

#ifndef MEGA_MOE_ARCH35_MX_QUANT_COMMON_H
#define MEGA_MOE_ARCH35_MX_QUANT_COMMON_H

#if defined(__DAV_C310__)
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace MegaMoeImpl {
namespace MxQuant {

constexpr int64_t DATA_ELEMENT_COUNT_PER_LOOP = 256;
constexpr int64_t SCALE_ELEMENT_COUNT_PER_DATA_LOOP = 8;

constexpr AscendC::Reg::CastTrait CAST_BF16_TO_FP32_ZERO_LAYOUT = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
constexpr AscendC::Reg::CastTrait CAST_BF16_TO_FP32_ONE_LAYOUT = {
    AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
constexpr AscendC::Reg::CastTrait CAST_FP32_TO_FP8_LAYOUT_ZERO = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
constexpr AscendC::Reg::CastTrait CAST_FP32_TO_FP8_LAYOUT_ONE = {
    AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
constexpr AscendC::Reg::CastTrait CAST_FP32_TO_FP8_LAYOUT_TWO = {
    AscendC::Reg::RegLayout::TWO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
constexpr AscendC::Reg::CastTrait CAST_FP32_TO_FP8_LAYOUT_THREE = {
    AscendC::Reg::RegLayout::THREE, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
} // namespace MxQuant
} // namespace MegaMoeImpl

#endif // defined(__DAV_C310__)
#endif // MEGA_MOE_ARCH35_MX_QUANT_COMMON_H
