/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "block_attn_res_update.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(BlockAttnResUpdate);

namespace {
static bool CheckNotNull(const aclTensor *partialBlock, const aclTensor *delta, const aclTensor *interNumerator,
                         const aclTensor *interMax, const aclTensor *interExpSum, const aclTensor *effectiveQuery,
                         const aclOpExecutor *executor)
{
    OP_CHECK(partialBlock != nullptr && delta != nullptr && interNumerator != nullptr && interMax != nullptr &&
                 interExpSum != nullptr && effectiveQuery != nullptr && executor != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "BlockAttnResUpdate input or executor is nullptr."), return false);
    return true;
}

static bool IsAiCoreSupport(const aclTensor *partialBlock, const aclTensor *delta, const aclTensor *interNumerator,
                            const aclTensor *interMax, const aclTensor *interExpSum,
                            const aclTensor *effectiveQuery)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(npuArch == NpuArch::DAV_3510,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BlockAttnResUpdate only supports Ascend950."), return false);

    OP_CHECK(partialBlock->GetDataType() == DataType::DT_FLOAT && delta->GetDataType() == DataType::DT_BF16 &&
                 interNumerator->GetDataType() == DataType::DT_FLOAT &&
                 interMax->GetDataType() == DataType::DT_FLOAT &&
                 interExpSum->GetDataType() == DataType::DT_FLOAT &&
                 effectiveQuery->GetDataType() == DataType::DT_FLOAT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "BlockAttnResUpdate input dtypes are not supported by the AiCore implementation."),
             return false);
    return true;
}
} // namespace

const aclTensor *BlockAttnResUpdate(
    aclTensor *partialBlock, const aclTensor *delta, const aclTensor *interNumerator, const aclTensor *interMax,
    const aclTensor *interExpSum, const aclTensor *effectiveQuery, float eps, aclOpExecutor *executor)
{
    L0_DFX(BlockAttnResUpdate, partialBlock, delta, interNumerator, interMax, interExpSum, effectiveQuery, eps);

    OP_CHECK(CheckNotNull(partialBlock, delta, interNumerator, interMax, interExpSum, effectiveQuery, executor),
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "BlockAttnResUpdate null pointer check failed."), return nullptr);
    OP_CHECK(IsAiCoreSupport(partialBlock, delta, interNumerator, interMax, interExpSum, effectiveQuery),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BlockAttnResUpdate IsAiCoreSupport check failed."), return nullptr);

    auto h = executor->AllocTensor(DataType::DT_BF16, Format::FORMAT_ND, Format::FORMAT_ND);
    OP_CHECK(h != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BlockAttnResUpdate AllocTensor failed."), return nullptr);

    auto ret = INFER_SHAPE(
        BlockAttnResUpdate, OP_INPUT(partialBlock, delta, interNumerator, interMax, interExpSum, effectiveQuery),
        OP_OUTPUT(partialBlock, h), OP_ATTR(eps));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return nullptr, "BlockAttnResUpdate InferShape failed.");

    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        BlockAttnResUpdate, OP_INPUT(partialBlock, delta, interNumerator, interMax, interExpSum, effectiveQuery),
        OP_OUTPUT(partialBlock, h), OP_ATTR(eps));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
                                         "BlockAttnResUpdate ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return h;
}

} // namespace l0op
