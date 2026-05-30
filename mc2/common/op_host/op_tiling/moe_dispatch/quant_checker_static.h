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
 * \file quant_checker_static.h
 * \brief MOE Distribute Dispatch 静态量化模式校验 (mode=1)
 */

#ifndef MOE_DISPATCH_QUANT_CHECKER_STATIC_H
#define MOE_DISPATCH_QUANT_CHECKER_STATIC_H

#include "quant_checker_base.h"
#include "defs.h"
#include "utils.h"

namespace moe_dispatch {

class StaticQuantChecker : public QuantCheckerBase {
public:
    StaticQuantChecker() : QuantCheckerBase(moe_dispatch_defs::QuantMode::STATIC_QUANT)
    {
    }

    // scales 存在性判断 - 静态量化必须有 scales
    bool ShouldHaveScales(ge::DataType xDtype) const override
    {
        return true;
    }

    // scales 维度数校验（静态量化特有）
    // expandXDtype: expandX 输出的 dtype，用于 hif8 特殊校验
    // 返回期望的维度数：hif8 时必须为 1D，其他为 1D 或 2D
    uint32_t GetExpectedScalesDimNum(ge::DataType expandXDtype) const
    {
        if (expandXDtype == ge::DT_HIFLOAT8) {
            return moe_dispatch_defs::ONE_DIM; // hif8 强制 1D
        }
        return 0U; // 返回 0 表示允许 1D 或 2D，由调用方进一步校验
    }

    // scales 是否允许多维度（静态量化特有：允许 1D 或 2D）
    bool AllowScalesMultiDim() const override
    {
        return true;
    }

    // scales 是否允许 2D（静态量化特有）
    bool AllowScalesTwoDim(ge::DataType expandXDtype) const
    {
        return expandXDtype != ge::DT_HIFLOAT8;
    }

    // scales 校验
    ge::graphStatus CheckScales(gert::TilingContext *ctx, const ScalesTensorInfo &info, uint32_t expectedDim0,
                                uint32_t h) const override
    {
        // 静态量化必须有 scales
        if (info.dtype == ge::DT_UNDEFINED) {
            return ge::GRAPH_FAILED;
        }

        // dtype: float
        if (info.dtype != ge::DT_FLOAT) {
            return ge::GRAPH_FAILED;
        }

        return ge::GRAPH_SUCCESS;
    }

    // dynamicScales 校验
    ge::graphStatus CheckDynamicScales(gert::TilingContext *ctx, const ScalesTensorInfo &info, uint32_t expectedDim0,
                                       uint32_t h) const override
    {
        // 静态量化不需要 dynamicScales
        return ge::GRAPH_SUCCESS;
    }

    // tokenMsgSize 计算
    uint32_t CalcTokenMsgSize(uint32_t h) const override
    {
        // Static quant: AlignUp(h, 256)
        return moe_dispatch_utils::AlignUp(h, moe_dispatch_defs::ALIGN_256);
    }

    // dynamicScales 不需要
    uint32_t GetDynamicScalesDimNum() const override
    {
        return 0U;
    }
};

} // namespace moe_dispatch

#endif // MOE_DISPATCH_QUANT_CHECKER_STATIC_H