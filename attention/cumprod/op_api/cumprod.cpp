/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cumprod.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(Cumprod);

// axis mirrors the built-in Cumprod IR (INPUT axis). The aclnn api exposes it as a
// scalar (aclnnCumprod's dim); the L0 layer materializes it into an int64 const
// tensor input via executor ConvertToTensor (official scalar->tensor-input pattern),
// so the op instance carries all three descriptors (x, axis, y) — binary selection,
// kernel argument binding and host-tiling const reads all key off them.
const aclTensor* Cumprod(const aclTensor* x, const aclScalar* axis, bool exclusive, bool reverse,
                         aclOpExecutor* executor)
{
    auto out = executor->AllocTensor(x->GetViewShape(), x->GetDataType(), Format::FORMAT_ND);
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor allocation failed.");
        return nullptr;
    }
    int64_t axisVal = axis->ToInt64();
    const aclTensor* axisTensor = executor->ConvertToTensor(&axisVal, 1, DataType::DT_INT64);
    if (axisTensor == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "materialize axis scalar into const tensor failed.");
        return nullptr;
    }
    L0_DFX(Cumprod, x, axis, exclusive, reverse, out);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        Cumprod,
        OP_INPUT(x, axisTensor),
        OP_OUTPUT(out),
        OP_ATTR(exclusive, reverse));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Cumprod ADD_TO_LAUNCHER_LIST_AICORE failed, ret:%d.", ret);
        return nullptr;
    }
    return out;
}
} // namespace l0op
