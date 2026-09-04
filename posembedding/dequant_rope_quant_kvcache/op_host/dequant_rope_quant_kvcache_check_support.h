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
 * \file dequant_rope_quant_kvcache_check_support.h
 * \brief DequantRopeQuantKvcache input contract checks for Ascend 950.
 */

#ifndef DEQUANT_ROPE_QUANT_KVCACHE_CHECK_SUPPORT_H
#define DEQUANT_ROPE_QUANT_KVCACHE_CHECK_SUPPORT_H

#include <array>
#include <cstdint>

#include "graph/operator.h"
#include "register/op_def_registry.h"

namespace ops {
constexpr uint32_t DRQK_INPUT_NUM = 13U;
constexpr uint32_t DRQK_REQUIRED_INPUT_NUM = 8U;

inline bool IsDrqkInputPresent(const ge::TensorDesc &desc)
{
    return desc.GetDataType() != ge::DT_UNDEFINED;
}

inline bool IsDrqkDtypeSupported(const ge::Operator &op)
{
    const ge::DataType xDtype = op.GetInputDesc(0U).GetDataType();
    const ge::DataType cosDtype = op.GetInputDesc(1U).GetDataType();
    const ge::DataType sinDtype = op.GetInputDesc(2U).GetDataType();
    const bool ropeDtypeSupported =
        cosDtype == sinDtype && ((xDtype == ge::DT_FLOAT16 && cosDtype == ge::DT_FLOAT16) ||
                                 (xDtype == ge::DT_BF16 && cosDtype == ge::DT_BF16) ||
                                 (xDtype == ge::DT_INT32 && (cosDtype == ge::DT_FLOAT16 || cosDtype == ge::DT_BF16)));
    if (!ropeDtypeSupported || op.GetInputDesc(3U).GetDataType() != ge::DT_INT8 ||
        op.GetInputDesc(4U).GetDataType() != ge::DT_INT8 || op.GetInputDesc(5U).GetDataType() != ge::DT_INT32 ||
        op.GetInputDesc(6U).GetDataType() != ge::DT_FLOAT || op.GetInputDesc(7U).GetDataType() != ge::DT_FLOAT) {
        return false;
    }

    for (uint32_t index = 8U; index <= 11U; ++index) {
        const ge::TensorDesc desc = op.GetInputDesc(index);
        if (IsDrqkInputPresent(desc) && desc.GetDataType() != ge::DT_FLOAT) {
            return false;
        }
    }
    const ge::TensorDesc biasDesc = op.GetInputDesc(12U);
    if (IsDrqkInputPresent(biasDesc)) {
        const ge::DataType biasDtype = biasDesc.GetDataType();
        if (biasDtype != ge::DT_FLOAT && biasDtype != ge::DT_FLOAT16 && biasDtype != ge::DT_BF16 &&
            biasDtype != ge::DT_INT32) {
            return false;
        }
    }
    return true;
}

inline bool IsDrqkFormatSupported(const ge::Operator &op)
{
    for (uint32_t index = 0U; index < DRQK_INPUT_NUM; ++index) {
        const ge::TensorDesc desc = op.GetInputDesc(index);
        if (index >= DRQK_REQUIRED_INPUT_NUM && !IsDrqkInputPresent(desc)) {
            continue;
        }
        if (desc.GetOriginFormat() != ge::FORMAT_ND) {
            return false;
        }
    }
    return true;
}

inline ge::graphStatus CheckSupportForDequantRopeQuantKvcache(const ge::Operator &op, ge::AscendString &result)
{
    if (!IsDrqkFormatSupported(op)) {
        result = ge::AscendString(
            R"({"isSupported": "False", "dynamicCompileStatic": "True", "reason": "DequantRopeQuantKvcache supports ND input format only."})");
        return ge::GRAPH_FAILED;
    }
    if (!IsDrqkDtypeSupported(op)) {
        result = ge::AscendString(
            R"({"isSupported": "False", "dynamicCompileStatic": "True", "reason": "DequantRopeQuantKvcache input dtype combination is unsupported."})");
        return ge::GRAPH_FAILED;
    }
    result = ge::AscendString(
        R"({"isSupported": "True", "dynamicCompileStatic": "True", "reason": "DequantRopeQuantKvcache input contract passed."})");
    return ge::GRAPH_SUCCESS;
}
} // namespace ops

#endif // DEQUANT_ROPE_QUANT_KVCACHE_CHECK_SUPPORT_H
