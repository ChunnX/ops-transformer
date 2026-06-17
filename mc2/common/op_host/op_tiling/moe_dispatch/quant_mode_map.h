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
 * \file quant_mode_map.h
 * \brief MOE Distribute Dispatch QuantMode + dtype → RealMode 查表
 */

#ifndef MOE_DISPATCH_QUANT_MODE_MAP_H
#define MOE_DISPATCH_QUANT_MODE_MAP_H

#include <map>
#include <utility>
#include "defs.h"
#include "graph/utils/type_utils.h"

namespace moe_dispatch {

// 实际量化模式（考虑 dtype）
enum class RealMode : uint32_t {
    NO_SCALES = 0,
    STATIC_SCALES = 1,
    PERTOKEN_SCALES = 2,
    PERGROUP_SCALES = 3,
    MX_SCALES = 4,
    HIF8_SCALES = 5,
    INVALID_MODE
};

// QuantMode + xDtype → RealMode 查表
inline RealMode CheckQuantModeMap(moe_dispatch_defs::QuantMode quantMode, ge::DataType dtype)
{
    using moe_dispatch_defs::QuantMode;
    static const std::map<std::pair<QuantMode, ge::DataType>, RealMode> kQuantModeMap = {
        {{QuantMode::UNQUANT, ge::DT_BF16}, RealMode::NO_SCALES},
        {{QuantMode::UNQUANT, ge::DT_FLOAT16}, RealMode::NO_SCALES},
        {{QuantMode::UNQUANT, ge::DT_FLOAT8_E4M3FN}, RealMode::NO_SCALES},
        {{QuantMode::UNQUANT, ge::DT_FLOAT8_E5M2}, RealMode::NO_SCALES},
        {{QuantMode::UNQUANT, ge::DT_HIFLOAT8}, RealMode::NO_SCALES},
        {{QuantMode::UNQUANT, ge::DT_FLOAT4_E2M1}, RealMode::NO_SCALES},
        {{QuantMode::UNQUANT, ge::DT_FLOAT4_E1M2}, RealMode::NO_SCALES},
        {{QuantMode::STATIC_QUANT, ge::DT_INT8}, RealMode::STATIC_SCALES},
        {{QuantMode::STATIC_QUANT, ge::DT_HIFLOAT8}, RealMode::HIF8_SCALES},
        {{QuantMode::PERTOKEN_DYNAMIC_QUANT, ge::DT_INT8}, RealMode::PERTOKEN_SCALES},
        {{QuantMode::PERTOKEN_DYNAMIC_QUANT, ge::DT_FLOAT8_E5M2}, RealMode::PERTOKEN_SCALES},
        {{QuantMode::PERTOKEN_DYNAMIC_QUANT, ge::DT_FLOAT8_E4M3FN}, RealMode::PERTOKEN_SCALES},
        {{QuantMode::PERGROUP_DYNAMIC_QUANT, ge::DT_FLOAT8_E5M2}, RealMode::PERGROUP_SCALES},
        {{QuantMode::PERGROUP_DYNAMIC_QUANT, ge::DT_FLOAT8_E4M3FN}, RealMode::PERGROUP_SCALES},
        {{QuantMode::MX_QUANT, ge::DT_FLOAT8_E5M2}, RealMode::MX_SCALES},
        {{QuantMode::MX_QUANT, ge::DT_FLOAT8_E4M3FN}, RealMode::MX_SCALES},
        {{QuantMode::MX_QUANT, ge::DT_FLOAT4_E2M1}, RealMode::MX_SCALES},
        {{QuantMode::MX_QUANT, ge::DT_FLOAT4_E1M2}, RealMode::MX_SCALES},
    };
    auto it = kQuantModeMap.find({quantMode, dtype});
    return ((it != kQuantModeMap.end()) ? it->second : RealMode::INVALID_MODE);
}

} // namespace moe_dispatch

#endif // MOE_DISPATCH_QUANT_MODE_MAP_H
