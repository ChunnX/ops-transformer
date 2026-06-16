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
 * \file mx_quant_checker.h
 * \brief MOE Distribute Dispatch MX 量化校验 (mode=4)
 *        构造参数化：dynScalesDtype 由调用方传入
 */

#ifndef MOE_DISPATCH_QUANT_CHECKER_MX_H
#define MOE_DISPATCH_QUANT_CHECKER_MX_H

#include "dynamic_quant_checker_base.h"
#include "defs.h"
#include "utils.h"

namespace moe_dispatch {

class MxQuantChecker : public DynamicQuantCheckerBase {
public:
    // 构造参数化：dynScalesDtype 由调用方传入
    // v2-A5: DT_FLOAT8_E8M0
    // teardown: DT_FLOAT
    explicit MxQuantChecker(ge::DataType dynScalesDtype = ge::DT_FLOAT)
        : DynamicQuantCheckerBase(moe_dispatch_defs::QuantMode::MX_QUANT), dynScalesDtype_(dynScalesDtype)
    {
    }

    // scales 存在性判断 - MX quant 禁止 scales 输入
    bool ShouldHaveScales(ge::DataType xDtype) const override
    {
        return false;
    }

    // scales 校验 - MX quant 不允许 scales
    ge::graphStatus CheckScales(gert::TilingContext *ctx, const ScalesTensorInfo &info, uint32_t expectedDim0,
                                uint32_t h) const override
    {
        // MX quant 不允许 scales
        if (info.dtype != ge::DT_UNDEFINED) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    // dynamicScales 校验
    ge::graphStatus CheckDynamicScales(gert::TilingContext *ctx, const ScalesTensorInfo &info, uint32_t expectedDim0,
                                       uint32_t h) const override
    {
        // MX 必须有 dynamicScales
        if (info.dtype == ge::DT_UNDEFINED) {
            return ge::GRAPH_FAILED;
        }

        // dtype: 由构造参数决定
        if (info.dtype != dynScalesDtype_) {
            return ge::GRAPH_FAILED;
        }

        return ge::GRAPH_SUCCESS;
    }

    // dynamicScales 维度数 - 2D
    uint32_t GetDynamicScalesDimNum() const override
    {
        return moe_dispatch_defs::DYNAMIC_SCALE_TWO_DIM_NUM;
    }

    // dynamicScales dtype - 返回构造参数
    ge::DataType GetDynamicScalesDtype() const override
    {
        return dynScalesDtype_;
    }

    // dynamicScales dim1 计算
    // MX: CeilAlign(CeilDiv(h, 32), 2) = AlignUp(CeilDiv(h, 32), 2)
    uint32_t CalcDynamicScalesDim1(uint32_t h) const override
    {
        uint32_t rawDim1 = moe_dispatch_utils::CeilDiv(h, moe_dispatch_defs::MX_BLOCK_SIZE);
        return moe_dispatch_utils::AlignUp(rawDim1, moe_dispatch_defs::EVEN_ALIGN);
    }

    // tokenMsgSize 计算
    // MX: AlignUp(AlignUp(h, 256) + CeilAlign(CeilDiv(h, 32), 2), 512)
    uint32_t CalcTokenMsgSize(uint32_t h) const override
    {
        uint32_t alignedH = moe_dispatch_utils::AlignUp(h, moe_dispatch_defs::ALIGN_256);
        uint32_t scalesDim1 = moe_dispatch_utils::AlignUp(
            moe_dispatch_utils::CeilDiv(h, moe_dispatch_defs::MX_BLOCK_SIZE), moe_dispatch_defs::EVEN_ALIGN);
        uint32_t totalSize = alignedH + scalesDim1;
        return moe_dispatch_utils::AlignUp(totalSize, moe_dispatch_defs::ALIGN_512);
    }

private:
    ge::DataType dynScalesDtype_;
};

} // namespace moe_dispatch

#endif // MOE_DISPATCH_QUANT_CHECKER_MX_H