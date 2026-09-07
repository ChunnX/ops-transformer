/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "onnx_attribute_utils.h"

namespace domi {
static Status ParseParamsMoeFinalizeRoutingV2(const ge::Operator& op_src, ge::Operator& op_dest)
{
    int drop_pad_mode = 0;
    if (ParseOnnxIntAttribute(op_src, "drop_pad_mode", drop_pad_mode) != SUCCESS) {
        return FAILED;
    }
    op_dest.SetAttr("drop_pad_mode", drop_pad_mode);
    return SUCCESS;
}

// register npu_moe_finalize_routing op info to GE
REGISTER_CUSTOM_OP("MoeFinalizeRoutingV2")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::NPUMoeFinalizeRoutingV2"),
                   ge::AscendString("ai.onnx::11::NPUMoeFinalizeRoutingV2"),
                   ge::AscendString("ai.onnx::12::NPUMoeFinalizeRoutingV2"),
                   ge::AscendString("ai.onnx::13::NPUMoeFinalizeRoutingV2"),
                   ge::AscendString("ai.onnx::14::NPUMoeFinalizeRoutingV2"),
                   ge::AscendString("ai.onnx::15::NPUMoeFinalizeRoutingV2"),
                   ge::AscendString("ai.onnx::16::NPUMoeFinalizeRoutingV2"),
                   ge::AscendString("ai.onnx::17::NPUMoeFinalizeRoutingV2"),
                   ge::AscendString("ai.onnx::18::NPUMoeFinalizeRoutingV2")})
    .ParseParamsByOperatorFn(ParseParamsMoeFinalizeRoutingV2)
    .ImplyType(ImplyType::TVM);
} // namespace domi
