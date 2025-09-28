/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fia_tiling_info.cpp
 * \brief
 */

#include "fia_tiling_info.h"


namespace optiling {

std::string LayoutToSerialString(FiaLayout layout)
{
    switch (layout) {
        case FiaLayout::BSH:
            return "BSH";
        case FiaLayout::BSND:
            return "BSND";
        case FiaLayout::BNSD:
            return "BNSD";
        case FiaLayout::NZ:
            return "NZ";
        case FiaLayout::TND:
            return "TND";
        case FiaLayout::NBSD:
            return "NBSD";
        case FiaLayout::NTD:
            return "NTD";
        case FiaLayout::S1S2:
            return "SS";
        case FiaLayout::BS2:
            return "BS2";
        case FiaLayout::B1S2:
            return "B1S2";
        case FiaLayout::B11S2:
            return "B11S2";
        case FiaLayout::BnBsH:
            return "BnBsH";
        case FiaLayout::BnNBsD:
            return "BnNBsD";
        default:
            return "UNKNOWN";
    }
}

std::string AxisToSerialString(FiaAxis axis)
{
    switch (axis) {
        case FiaAxis::B:
            return "B";
        case FiaAxis::S:
            return "S";
        case FiaAxis::N:
            return "N";
        case FiaAxis::D:
            return "D";
        case FiaAxis::H:
            return "H";
        case FiaAxis::T:
            return "T";
        case FiaAxis::D1:
            return "D1";
        case FiaAxis::D0:
            return "D0";
        case FiaAxis::S1:
            return "S1";
        case FiaAxis::S2:
            return "S2";
        case FiaAxis::Bn:
            return "Bn";
        case FiaAxis::Bs:
            return "Bs";
        case FiaAxis::CONST:
            return "CONST";
        default:
            return "UNKNOWN";
    }
}

std::string QuantModeToSerialString(FiaQuantMode fiaQuantMode)
{
    switch (fiaQuantMode) {
        case FiaQuantMode::NO_QUANT:
            return "NO_QUANT";
        case FiaQuantMode::ANTI_QUANT:
            return "ANTI_QUANT";
        case FiaQuantMode::FULL_QUANT:
            return "FULL_QUANT";
        default:
            return "UNKNOWN";
    }
}
} // namespace optiling