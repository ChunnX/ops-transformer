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
 * \file dynamic_quant_checker_base.h
 * \brief MOE Distribute Dispatch 动态量化基类定义 (PerToken/PerGroup/MX 共有骨架)
 */

#ifndef MOE_DISPATCH_QUANT_CHECKER_DYNAMIC_BASE_H
#define MOE_DISPATCH_QUANT_CHECKER_DYNAMIC_BASE_H

#include "quant_checker_base.h"
#include "defs.h"
#include "utils.h"

namespace moe_dispatch {

// PerToken/PerGroup/MX 共有骨架
class DynamicQuantCheckerBase : public QuantCheckerBase {
public:
    explicit DynamicQuantCheckerBase(moe_dispatch_defs::QuantMode mode) : QuantCheckerBase(mode)
    {
    }

    // scales 存在性判断 - 动态量化 scales 可选，取决于 x dtype
    bool ShouldHaveScales(ge::DataType xDtype) const override
    {
        // 默认不需要，子类可 override
        return false;
    }

    // scales 维度数校验 - 动态量化固定 2D
    uint32_t GetExpectedScalesDimNum(ge::DataType expandXDtype) const
    {
        (void)expandXDtype; // 动态量化不依赖 expandX dtype
        return moe_dispatch_defs::TWO_DIMS;
    }

    // scales 校验
    ge::graphStatus CheckScales(gert::TilingContext *ctx, const ScalesTensorInfo &info, uint32_t expectedDim0,
                                uint32_t h) const override
    {
        // scales 可选
        if (info.dtype == ge::DT_UNDEFINED) {
            return ge::GRAPH_SUCCESS;
        }

        // dtype: float
        if (info.dtype != ge::DT_FLOAT) {
            return ge::GRAPH_FAILED;
        }

        return ge::GRAPH_SUCCESS;
    }

    // dynamicScales dim0 期望值计算（通用）
    uint32_t GetExpectedDynamicScalesDim0(uint32_t A, uint32_t tpWorldSize) const
    {
        return A * tpWorldSize;
    }

    // dynamicScales 默认维度数为 2D
    uint32_t GetDynamicScalesDimNum() const override
    {
        return moe_dispatch_defs::DYNAMIC_SCALE_TWO_DIM_NUM;
    }

    // dynamicScales 默认 dtype 为 float
    ge::DataType GetDynamicScalesDtype() const override
    {
        return ge::DT_FLOAT;
    }
};

} // namespace moe_dispatch

#endif // MOE_DISPATCH_QUANT_CHECKER_DYNAMIC_BASE_H