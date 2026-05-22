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
 * \file quant_checker_factory.h
 * \brief MOE Distribute Dispatch 量化校验工厂函数
 */

#ifndef MOE_DISPATCH_QUANT_CHECKER_FACTORY_H
#define MOE_DISPATCH_QUANT_CHECKER_FACTORY_H

#include <memory>
#include "defs.h"
#include "quant_checker_base.h"
#include "quant_checker_unquant.h"
#include "quant_checker_static.h"
#include "quant_checker_pertoken.h"
#include "quant_checker_pergroup.h"
#include "quant_checker_mx.h"

namespace moe_dispatch {

// 根据 quantMode 创建对应的 checker
// mxDynScalesDtype: MX quant 的 dynamicScales dtype（v2-A5: e8m0, teardown: float）
inline std::unique_ptr<QuantCheckerBase> CreateQuantChecker(
    uint32_t quantMode,
    ge::DataType mxDynScalesDtype = ge::DT_FLOAT) {
    switch (quantMode) {
        case static_cast<uint32_t>(moe_dispatch_defs::QuantMode::UNQUANT):
            return std::make_unique<UnquantChecker>();
        case static_cast<uint32_t>(moe_dispatch_defs::QuantMode::STATIC_QUANT):
            return std::make_unique<StaticQuantChecker>();
        case static_cast<uint32_t>(moe_dispatch_defs::QuantMode::PERTOKEN_DYNAMIC_QUANT):
            return std::make_unique<PerTokenChecker>();
        case static_cast<uint32_t>(moe_dispatch_defs::QuantMode::PERGROUP_DYNAMIC_QUANT):
            return std::make_unique<PerGroupChecker>();
        case static_cast<uint32_t>(moe_dispatch_defs::QuantMode::MX_QUANT):
            return std::make_unique<MxQuantChecker>(mxDynScalesDtype);
        default:
            return nullptr;
    }
}

// 重载：支持 QuantMode 枚举参数
inline std::unique_ptr<QuantCheckerBase> CreateQuantChecker(
    moe_dispatch_defs::QuantMode quantMode,
    ge::DataType mxDynScalesDtype = ge::DT_FLOAT) {
    return CreateQuantChecker(static_cast<uint32_t>(quantMode), mxDynScalesDtype);
}

} // namespace moe_dispatch

#endif // MOE_DISPATCH_QUANT_CHECKER_FACTORY_H