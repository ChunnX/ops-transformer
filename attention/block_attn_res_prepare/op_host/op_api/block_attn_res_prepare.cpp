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
 * \file block_attn_res_prepare.cpp
 * \brief L0 operator implementation for BlockAttnResPrepare.
 */

#include "block_attn_res_prepare.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(BlockAttnResPrepare);

const std::array<const aclTensor *, 3> BlockAttnResPrepare(const aclTensor *blockResidual,
                                                           const aclTensor *effectiveQuery,
                                                           const aclTensor *validBlocks, float epsilon,
                                                           aclOpExecutor *executor)
{
    L0_DFX(BlockAttnResPrepare, blockResidual, effectiveQuery, validBlocks, epsilon);

    auto softmaxMax = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    auto weightedOutput = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    auto softmaxSum = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    const aclnnStatus ret = INFER_SHAPE(BlockAttnResPrepare,
                                        OP_INPUT(blockResidual, effectiveQuery, validBlocks),
                                        OP_OUTPUT(softmaxMax, weightedOutput, softmaxSum), OP_ATTR(epsilon));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BlockAttnResPrepare infer shape failed.");
        return {nullptr, nullptr, nullptr};
    }

    ADD_TO_LAUNCHER_LIST_AICORE(BlockAttnResPrepare, OP_INPUT(blockResidual, effectiveQuery, validBlocks),
                                OP_OUTPUT(softmaxMax, weightedOutput, softmaxSum), OP_ATTR(epsilon));
    return {softmaxMax, weightedOutput, softmaxSum};
}

} // namespace l0op
