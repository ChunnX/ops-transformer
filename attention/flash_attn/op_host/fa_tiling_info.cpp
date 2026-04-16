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
 * \file fa_tiling_info.cpp
 * \brief
 */

#include "fa_tiling_info.h"


namespace optiling {

std::string LayoutToSerialString(FaLayout layout)
{
    const std::map<FaLayout, std::string> layout2Str = {
        { FaLayout::BSH, "BSH" },
        { FaLayout::BSND, "BSND" },
        { FaLayout::BNSD, "BNSD" },
        { FaLayout::NZ, "NZ" },
        { FaLayout::TND, "TND" },
        { FaLayout::NBSD, "NBSD" },
        { FaLayout::NTD, "NTD" },
        { FaLayout::S1S2, "S1S2" },
        { FaLayout::BS2, "BS2" },
        { FaLayout::BnBsH, "BnBsH" },
        { FaLayout::BnNBsD, "BnNBsD" },
        { FaLayout::BNS1S2, "BNS1S2" },
        { FaLayout::INS1S2, "1NS1S2" },
        { FaLayout::BNS11, "BNS11" },
        { FaLayout::TN1, "TN1" },
        { FaLayout::BS1S2, "BS1S2" },
        { FaLayout::B1S1S2, "B1S1S2" },
        { FaLayout::IS1S2, "1S1S2" },
        { FaLayout::I1S1S2, "11S1S2" }
    };

    if (layout2Str.find(layout) != layout2Str.end()) {
        return layout2Str.at(layout);
    }
    return "UNKNOWN";
}

std::string AxisToSerialString(FaAxis axis)
{
    switch (axis) {
        case FaAxis::B:
            return "B";
        case FaAxis::S:
            return "S";
        case FaAxis::N:
            return "N";
        case FaAxis::D:
            return "D";
        case FaAxis::H:
            return "H";
        case FaAxis::T:
            return "T";
        case FaAxis::D1:
            return "D1";
        case FaAxis::D0:
            return "D0";
        case FaAxis::S1:
            return "S1";
        case FaAxis::S2:
            return "S2";
        case FaAxis::Bn:
            return "Bn";
        case FaAxis::Bs:
            return "Bs";
        case FaAxis::CONST:
            return "CONST";
        default:
            return "UNKNOWN";
    }
}

std::string QuantModeToSerialString(FaQuantMode faQuantMode)
{
    switch (faQuantMode) {
        case FaQuantMode::NO_QUANT:
            return "NO_QUANT";
        case FaQuantMode::ANTI_QUANT:
            return "ANTI_QUANT";
        case FaQuantMode::FULL_QUANT:
            return "FULL_QUANT";
        default:
            return "UNKNOWN";
    }
}

// std::string SituationToSerialString(RopeMode ropeMode)
// {
//     if (ropeMode == RopeMode::ROPE_SPLIT) {
//         return "qkHeadDim = vHeadDim and rope exist";
//     } else {
//         return "rope not exist";
//     }
// }
} // namespace optiling