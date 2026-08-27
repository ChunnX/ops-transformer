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
 * \file mx_quant.h
 * \brief MX 量化统一入口。
 */

#ifndef MEGA_MOE_ARCH35_MX_QUANT_H
#define MEGA_MOE_ARCH35_MX_QUANT_H

#if defined(__DAV_C310__)
#include "mx_quant_common.h"
#include "mx_quant_fp4.h"
#include "mx_quant_fp8.h"
#include "mx_scale.h"

namespace MegaMoeImpl {
namespace MxQuant {

template <typename OutputType>
__aicore__ inline void QuantizeMxData(const MxQuantArgs &args, uint16_t dataLoopCount)
{
    if constexpr (AscendC::IsSameType<OutputType, fp8_e4m3fn_t>::value ||
                  AscendC::IsSameType<OutputType, fp8_e5m2_t>::value) {
        QuantizeMxFp8Data<OutputType>(
            args.input, args.reciprocalScale, args.output, args.dataCount, dataLoopCount);
    }
    if constexpr (AscendC::IsSameType<OutputType, fp4x2_e2m1_t>::value ||
                  AscendC::IsSameType<OutputType, fp4x2_e1m2_t>::value) {
        QuantizeMxFp4Data<OutputType>(
            args.input, args.reciprocalScale, args.output, args.dataCount, dataLoopCount);
    }
}

template <typename OutputType>
__aicore__ inline void RunMxQuant(const MxQuantArgs &args)
{
    const uint16_t dataLoopCount = static_cast<uint16_t>(
        (args.dataCount + DATA_ELEMENT_COUNT_PER_LOOP - 1) / DATA_ELEMENT_COUNT_PER_LOOP);
    const uint16_t scaleLoopCount = static_cast<uint16_t>(
        (args.scaleCount + SCALE_ELEMENT_COUNT_PER_VECTOR - 1) / SCALE_ELEMENT_COUNT_PER_VECTOR);

    ComputeGroupMaxExp(args.input, args.maxExp, args.dataCount, dataLoopCount);
    ComputeMxScale<OutputType>(
        args.maxExp, args.outputScale, args.reciprocalScale, args.scaleCount, scaleLoopCount);
    QuantizeMxData<OutputType>(args, dataLoopCount);
}

} // namespace MxQuant
} // namespace MegaMoeImpl

#endif // defined(__DAV_C310__)
#endif // MEGA_MOE_ARCH35_MX_QUANT_H
