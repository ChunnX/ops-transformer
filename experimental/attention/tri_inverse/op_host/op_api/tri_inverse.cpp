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
 * \file tri_inverse.cpp
 * \brief
 */

#include "tri_inverse.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(TriInverse);

const aclTensor *TriInverse(const aclTensor *ccIn, aclOpExecutor *executor)
{
    L0_DFX(TriInverse, ccIn);

    DataType outType = ccIn->GetDataType();
    Format format = Format::FORMAT_ND;
    auto ccOut = executor->AllocTensor(outType, format, format);

    auto ret = INFER_SHAPE(TriInverse, OP_INPUT(ccIn), OP_OUTPUT(ccOut));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "TriInverse InferShape failed.");
        return nullptr;
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(TriInverse, OP_INPUT(ccIn), OP_OUTPUT(ccOut));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "TriInverse ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return ccOut;
}
} // namespace l0op
