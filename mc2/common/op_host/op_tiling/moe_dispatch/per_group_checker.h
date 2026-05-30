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
 * \file per_group_checker.h
 * \brief MOE Distribute Dispatch PerGroup 动态量化校验 (mode=3)
 */

#ifndef MOE_DISPATCH_QUANT_CHECKER_PERGROUP_H
#define MOE_DISPATCH_QUANT_CHECKER_PERGROUP_H

#include "dynamic_quant_checker_base.h"
#include "defs.h"
#include "utils.h"

namespace moe_dispatch {

class PerGroupChecker : public DynamicQuantCheckerBase {
public:
    PerGroupChecker() : DynamicQuantCheckerBase(moe_dispatch_defs::QuantMode::PERGROUP_DYNAMIC_QUANT)
    {
    }

    // dynamicScales 校验
    ge::graphStatus CheckDynamicScales(gert::TilingContext *ctx, const ScalesTensorInfo &info, uint32_t expectedDim0,
                                       uint32_t h) const override
    {
        // PerGroup 必须有 dynamicScales
        if (info.dtype == ge::DT_UNDEFINED) {
            return ge::GRAPH_FAILED;
        }

        // dtype: float
        if (info.dtype != ge::DT_FLOAT) {
            return ge::GRAPH_FAILED;
        }

        return ge::GRAPH_SUCCESS;
    }

    // dynamicScales 维度数 - 2D
    uint32_t GetDynamicScalesDimNum() const override
    {
        return moe_dispatch_defs::DYNAMIC_SCALE_TWO_DIM_NUM;
    }

    // dynamicScales dtype
    ge::DataType GetDynamicScalesDtype() const override
    {
        return ge::DT_FLOAT;
    }

    // dynamicScales dim1 计算
    // PerGroup: CeilDiv(h, 128)
    uint32_t CalcDynamicScalesDim1(uint32_t h) const override
    {
        return moe_dispatch_utils::CeilDiv(h, moe_dispatch_defs::PERGROUP_BLOCK_SIZE);
    }

    // tokenMsgSize 计算
    // PerGroup: AlignUp(AlignUp(h, 128) + CeilDiv(h, 128), 512)
    uint32_t CalcTokenMsgSize(uint32_t h) const override
    {
        uint32_t alignedH = moe_dispatch_utils::AlignUp(h, moe_dispatch_defs::ALIGN_128);
        uint32_t scalesDim1 = moe_dispatch_utils::CeilDiv(h, moe_dispatch_defs::ALIGN_128);
        uint32_t totalSize = alignedH + scalesDim1;
        return moe_dispatch_utils::AlignUp(totalSize, moe_dispatch_defs::ALIGN_512);
    }
};

} // namespace moe_dispatch

#endif // MOE_DISPATCH_QUANT_CHECKER_PERGROUP_H