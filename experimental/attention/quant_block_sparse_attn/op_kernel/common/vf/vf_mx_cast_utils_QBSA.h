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
 * \file vf_mx_cast_utils_QBSA.h
 * \brief QBSA MXFP8 Reg Cast traits and CTRL mode constants.
 */
#ifndef VF_MX_CAST_UTILS_QBSA_H
#define VF_MX_CAST_UTILS_QBSA_H

#include "vf_basic_block_utils.h"

namespace QbsaVectorApi {

// CTRL[60]=0 makes every Reg Cast use its own CastTrait::SatMode.
constexpr uint32_t CAST_GLOBAL_MODE_ENABLE_BIT = 60U;
constexpr int64_t CAST_USE_INSTRUCTION_SAT_MODE = 0;

// MX PScale FP32 -> BF16 intermediates preserve non-finite values.
constexpr static AscendC::Reg::CastTrait castTraitRintNoSatZero = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::Reg::CastTrait castTraitRintNoSatOne = {
    AscendC::Reg::RegLayout::ONE,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

// MX V2 FP32 -> BF16 output conversion preserves non-finite values.
constexpr static AscendC::Reg::CastTrait castTraitRoundNoSatZero = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::Reg::CastTrait castTraitRoundNoSatOne = {
    AscendC::Reg::RegLayout::ONE,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

} // namespace QbsaVectorApi

#endif // VF_MX_CAST_UTILS_QBSA_H
