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
    StaticQuantChecker() : QuantCheckerBase(moe_dispatch_defs::QuantMode::STATIC_QUANT) {}

    // scales 存在性判断 - 静态量化必须有 scales
    bool ShouldHaveScales(ge::DataType xDtype) const override {
        return true;
    }

    // scales 校验
    ge::graphStatus CheckScales(
        gert::TilingContext* ctx,
        const ge::TensorDesc* desc,
        uint32_t expectedDim0,
        uint32_t h) const override {
        // 静态量化必须有 scales
        if (desc == nullptr) {
            return ge::GRAPH_FAILED;
        }

        // dtype: float
        ge::DataType dtype = desc->GetDataType();
        if (dtype != ge::DT_FLOAT) {
            return ge::GRAPH_FAILED;
        }

        // shape: 1D 或 2D
        const gert::StorageShape* shape = nullptr;
        if (ctx != nullptr) {
            // 这里假设 scales 是输入 tensor，具体索引由调用方传入
            // shape 校验在主流程中进行精确值校验
        }

        return ge::GRAPH_SUCCESS;
    }

    // dynamicScales 校验
    ge::graphStatus CheckDynamicScales(
        gert::TilingContext* ctx,
        const ge::TensorDesc* desc,
        uint32_t expectedDim0,
        uint32_t h) const override {
        // 静态量化不需要 dynamicScales
        return ge::GRAPH_SUCCESS;
    }

    // tokenMsgSize 计算
    uint32_t CalcTokenMsgSize(uint32_t h) const override {
        // Static quant: AlignUp(h, 256)
        return moe_dispatch_utils::AlignUp(h, moe_dispatch_defs::ALIGN_256);
    }

    // dynamicScales 不需要
    uint32_t GetDynamicScalesDimNum() const override {
        return 0U;
    }
};

} // namespace moe_dispatch

#endif // MOE_DISPATCH_QUANT_CHECKER_STATIC_H