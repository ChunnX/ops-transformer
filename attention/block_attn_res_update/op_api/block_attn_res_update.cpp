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
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(BlockAttnResUpdate);

namespace {
static const std::initializer_list<op::DataType> LOW_PRECISION_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_BF16};

static bool CheckNotNull(const aclTensor *partialBlock, const aclTensor *partialDelta,
                         const aclTensor *effectiveQuery, const aclTensor *interMax,
                         const aclTensor *interExpSum, const aclTensor *interNumerator,
                         const aclOpExecutor *executor)
{
    OP_CHECK(partialBlock != nullptr && partialDelta != nullptr && effectiveQuery != nullptr &&
                 interMax != nullptr && interExpSum != nullptr && interNumerator != nullptr && executor != nullptr,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "BlockAttnResUpdate input or executor is nullptr."), return false);
    return true;
}

static bool BlockAttnResUpdateInferShape(const Shape &partialBlockShape, const Shape &partialDeltaShape,
                                         Shape &hShape, Shape &partialBlocksShape)
{
    hShape = partialDeltaShape;
    partialBlocksShape = partialBlockShape;
    return true;
}

static bool IsAiCoreSupport(const aclTensor *partialBlock, const aclTensor *partialDelta,
                            const aclTensor *effectiveQuery, const aclTensor *interMax,
                            const aclTensor *interExpSum, const aclTensor *interNumerator)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(npuArch == NpuArch::DAV_3510,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BlockAttnResUpdate only supports Ascend950."), return false);

    OP_CHECK(partialBlock->GetDataType() == DataType::DT_FLOAT &&
                 effectiveQuery->GetDataType() == DataType::DT_FLOAT &&
                 interMax->GetDataType() == DataType::DT_FLOAT &&
                 interExpSum->GetDataType() == DataType::DT_FLOAT &&
                 interNumerator->GetDataType() == DataType::DT_FLOAT &&
                 CheckType(partialDelta->GetDataType(), LOW_PRECISION_DTYPE_SUPPORT_LIST),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "BlockAttnResUpdate input dtypes are not supported by the AiCore implementation."),
             return false);
    return true;
}

static std::array<const aclTensor *, 2> BlockAttnResUpdateAiCore(
    const aclTensor *partialBlock, const aclTensor *partialDelta, const aclTensor *effectiveQuery,
    const aclTensor *interMax, const aclTensor *interExpSum, const aclTensor *interNumerator, float epsilon,
    const aclTensor *h, const aclTensor *partialBlocks, aclOpExecutor *executor)
{
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        BlockAttnResUpdate,
        OP_INPUT(partialBlock, partialDelta, effectiveQuery, interMax, interExpSum, interNumerator),
        OP_OUTPUT(h, partialBlocks), OP_ATTR(epsilon));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BlockAttnResUpdate ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return {nullptr, nullptr});
    return {h, partialBlocks};
}
} // namespace

std::array<const aclTensor *, 2> BlockAttnResUpdate(
    const aclTensor *partialBlock, const aclTensor *partialDelta, const aclTensor *effectiveQuery,
    const aclTensor *interMax, const aclTensor *interExpSum, const aclTensor *interNumerator, float epsilon,
    aclOpExecutor *executor)
{
    L0_DFX(BlockAttnResUpdate, partialBlock, partialDelta, effectiveQuery, interMax, interExpSum, interNumerator,
           epsilon);

    OP_CHECK(CheckNotNull(partialBlock, partialDelta, effectiveQuery, interMax, interExpSum, interNumerator, executor),
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "BlockAttnResUpdate null pointer check failed."),
             return {nullptr, nullptr});

    Shape hShape;
    Shape partialBlocksShape;
    OP_CHECK(BlockAttnResUpdateInferShape(partialBlock->GetViewShape(), partialDelta->GetViewShape(), hShape,
                                         partialBlocksShape),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BlockAttnResUpdate infer shape failed."),
             return {nullptr, nullptr});

    OP_CHECK(IsAiCoreSupport(partialBlock, partialDelta, effectiveQuery, interMax, interExpSum, interNumerator),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BlockAttnResUpdate IsAiCoreSupport check failed."),
             return {nullptr, nullptr});

    auto h = executor->AllocTensor(hShape, partialDelta->GetDataType(), Format::FORMAT_ND);
    auto partialBlocks = executor->AllocTensor(partialBlocksShape, DataType::DT_FLOAT, Format::FORMAT_ND);
    OP_CHECK(h != nullptr && partialBlocks != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BlockAttnResUpdate AllocTensor failed."),
             return {nullptr, nullptr});

    return BlockAttnResUpdateAiCore(partialBlock, partialDelta, effectiveQuery, interMax, interExpSum, interNumerator,
                                   epsilon, h, partialBlocks, executor);
}

} // namespace l0op
