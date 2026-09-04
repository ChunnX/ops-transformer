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
 * \file onnx_common.h
 * \brief
 */

#ifndef MATH_COMMON_ONNX_COMMON_H
#define MATH_COMMON_ONNX_COMMON_H

#include <string>
#include <vector>
#include <map>

#include "stub_ops.h"
#include "register/register.h"
#include "graph/operator.h"
#include "graph/graph.h"
#include "base/err_msg.h"
#include "log/log.h"
#include "plugin_util.h"
#include "onnx/proto/ge_onnx.pb.h"

namespace domi {
inline Status ChangeFormatFromOnnx(ge::Operator &op, const int idx, ge::Format format, bool is_input)
{
    if (is_input) {
        ge::TensorDesc org_tensor = op.GetInputDesc(idx);
        org_tensor.SetOriginFormat(format);
        org_tensor.SetFormat(format);
        auto ret = op.UpdateInputDesc(idx, org_tensor);
        if (ret != ge::GRAPH_SUCCESS) {
            OP_LOGE(GetOpName(op).c_str(), "change input format failed.");
            return FAILED;
        }
    } else {
        ge::TensorDesc org_tensor_y = op.GetOutputDesc(idx);
        org_tensor_y.SetOriginFormat(format);
        org_tensor_y.SetFormat(format);
        auto ret_y = op.UpdateOutputDesc(idx, org_tensor_y);
        if (ret_y != ge::GRAPH_SUCCESS) {
            OP_LOGE(GetOpName(op).c_str(), "change output format failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}
} // namespace domi

#endif //  MATH_COMMON_ONNX_COMMON_H
