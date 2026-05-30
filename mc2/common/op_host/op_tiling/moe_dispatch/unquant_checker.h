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
 * \file unquant_checker.h
 * \brief MOE Distribute Dispatch 非量化模式校验 (mode=0)
 */

#ifndef MOE_DISPATCH_QUANT_CHECKER_UNQUANT_H
#define MOE_DISPATCH_QUANT_CHECKER_UNQUANT_H

#include "quant_checker_base.h"
#include "defs.h"
#include "utils.h"

namespace moe_dispatch {

class UnquantChecker : public QuantCheckerBase {
public:
    UnquantChecker() : QuantCheckerBase(moe_dispatch_defs::QuantMode::UNQUANT)
    {
    }

    // scales 存在性判断
    // hif8/e4m3fn/e5m2/f4 类型需要 scales（静态缩放），bf16/fp16 不需要
    bool ShouldHaveScales(ge::DataType xDtype) const override
    {
        if (xDtype == ge::DT_HIFLOAT8 || xDtype == ge::DT_FLOAT8_E4M3FN || xDtype == ge::DT_FLOAT8_E5M2 ||
            xDtype == ge::DT_FLOAT4_E2M1 || xDtype == ge::DT_FLOAT4_E1M2) {
            return true;
        }
        return false;
    }

    // scales 维度数校验 - 非量化模式固定 2D
    uint32_t GetExpectedScalesDimNum(ge::DataType expandXDtype) const override
    {
        (void)expandXDtype;
        return moe_dispatch_defs::TWO_DIMS;
    }

    // scales 校验
    ge::graphStatus CheckScales(gert::TilingContext *ctx, const ScalesTensorInfo &info, uint32_t expectedDim0,
                                uint32_t h) const override
    {
        // scales 可选，如果存在则校验 dtype
        if (info.dtype == ge::DT_UNDEFINED) {
            return ge::GRAPH_SUCCESS; // 允许不存在
        }

        // dtype: float 或 e8m0
        if (info.dtype != ge::DT_FLOAT && info.dtype != ge::DT_FLOAT8_E8M0) {
            return ge::GRAPH_FAILED;
        }

        return ge::GRAPH_SUCCESS;
    }

    // dynamicScales 校验
    ge::graphStatus CheckDynamicScales(gert::TilingContext *ctx, const ScalesTensorInfo &info, uint32_t expectedDim0,
                                       uint32_t h) const override
    {
        // Unquant 模式不需要 dynamicScales
        return ge::GRAPH_SUCCESS;
    }

    // tokenMsgSize 计算
    uint32_t CalcTokenMsgSize(uint32_t h) const override
    {
        // Unquant: AlignUp(h, 256)
        return moe_dispatch_utils::AlignUp(h, moe_dispatch_defs::ALIGN_256);
    }

    // dynamicScales 不需要
    uint32_t GetDynamicScalesDimNum() const override
    {
        return 0U;
    }
};

} // namespace moe_dispatch

#endif // MOE_DISPATCH_QUANT_CHECKER_UNQUANT_H