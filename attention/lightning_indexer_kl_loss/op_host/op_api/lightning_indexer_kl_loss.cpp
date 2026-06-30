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
 * \file lightning_indexer_kl_loss.cpp
 * \brief
 */

#include "lightning_indexer_kl_loss.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/format_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(LightningIndexerKLLoss);

const std::array<const aclTensor *, LIGHTNING_INDEXER_KL_LOSS_OUTPUT_CNT> LightningIndexerKLLoss(
    const aclTensor *target_score, const aclTensor *index_probs,
    float eps, aclOpExecutor *executor)
{
    L0_DFX(LightningIndexerKLLoss, target_score, index_probs, eps);
    DataType outputDtype = target_score->GetDataType();
    auto loss = executor->AllocTensor(outputDtype, op::Format::FORMAT_ND, op::Format::FORMAT_ND);

    auto ret = INFER_SHAPE(LightningIndexerKLLoss,
                           OP_INPUT(target_score, index_probs),
                           OP_OUTPUT(loss),
                           OP_ATTR(eps));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "LightningIndexerKLLoss InferShape failed.");
        return {nullptr};
    }

    ret =
        ADD_TO_LAUNCHER_LIST_AICORE(LightningIndexerKLLoss,
                                    OP_INPUT(target_score, index_probs),
                                    OP_OUTPUT(loss),
                                    OP_ATTR(eps));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "LightningIndexerKLLoss launch kernel failed.");
        return {nullptr};
    }
    return {loss};
}

} // namespace l0op
