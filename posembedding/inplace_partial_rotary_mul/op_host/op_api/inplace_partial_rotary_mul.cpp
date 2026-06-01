/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "inplace_partial_rotary_mul.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(InplacePartialRotaryMul);

const aclTensor *InplacePartialRotaryMul(const aclTensor *x, const aclTensor *cos, const aclTensor *sin,
    int64_t rotary_mode, const aclIntArray *partialSlice, aclOpExecutor *executor)
{
    L0_DFX(InplacePartialRotaryMul, x, cos, sin, rotary_mode, partialSlice);
    DataType outType = x->GetDataType();
    Format format = x->GetStorageFormat();
    auto output = executor->AllocTensor(outType, format, format);

    auto ret = INFER_SHAPE(InplacePartialRotaryMul,
        OP_INPUT(x, cos, sin), OP_OUTPUT(output),
        OP_ATTR(rotary_mode, partialSlice));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return nullptr, "InplacePartialRotaryMul InferShape failed.");
    ret = ADD_TO_LAUNCHER_LIST_AICORE(InplacePartialRotaryMul,
        OP_INPUT(x, cos, sin), OP_OUTPUT(output),
        OP_ATTR(rotary_mode, partialSlice));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
        "InplacePartialRotaryMul ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return output;
}
}
