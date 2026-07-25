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
 * \file fa_tiling_info.cpp
 * \brief
 */

#include "fa_tiling_info.h"


namespace optiling {
namespace flash_attn {

std::string LayoutToSerialString(FaLayout layout)
{
    const std::map<FaLayout, std::string> layout2Str = {{FaLayout::BSND, "BSND"},       {FaLayout::BNSD, "BNSD"},
                                                        {FaLayout::TND, "TND"},         {FaLayout::PA_BBND, "PA_BBND"},
                                                        {FaLayout::PA_BNBD, "PA_BNBD"}, {FaLayout::PA_NZ, "PA_NZ"},
                                                        {FaLayout::LSE_BNS, "LSE_BNS"}, {FaLayout::LSE_NT, "LSE_NT"}};

    if (layout2Str.find(layout) != layout2Str.end()) {
        return layout2Str.at(layout);
    }
    return "UNKNOWN";
}

static const std::string AXIS_SERIAL_STRINGS[] = {
    "B", "S", "N", "D", "H", "T", "D1", "D0", "S1", "S2", "Bn", "Bs", "CONST"
};

std::string AxisToSerialString(FaAxis axis)
{
    uint32_t idx = static_cast<uint32_t>(axis);
    return (idx < sizeof(AXIS_SERIAL_STRINGS) / sizeof(AXIS_SERIAL_STRINGS[0]))
        ? AXIS_SERIAL_STRINGS[idx] : "UNKNOWN";
}

} // namespace flash_attn
} // namespace optiling