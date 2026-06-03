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
 * \file chunk_gated_delta_rule_neumann.cpp
 * \brief
 */

#include "chunk_gated_delta_rule_neumann.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "level0/fill.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkGatedDeltaRuleNeumann);

const std::array<const aclTensor *, 2>
ChunkGatedDeltaRuleNeumann(const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *beta,
                    const aclTensor *initialState, const aclTensor *actualSeqLengths, const aclTensor *gOptional,
                    float scaleValue, aclOpExecutor *executor)
{
    L0_DFX(ChunkGatedDeltaRuleNeumann, query, key, value, beta, initialState, actualSeqLengths, gOptional, scaleValue);

    DataType outType = value->GetDataType();
    Format format = Format::FORMAT_ND;
    auto out = executor->AllocTensor(outType, format, format);
    auto finalState = executor->AllocTensor(outType, format, format);

    auto ret =
        INFER_SHAPE(ChunkGatedDeltaRuleNeumann, OP_INPUT(query, key, value, beta, initialState, actualSeqLengths, gOptional),
                    OP_OUTPUT(out, finalState), OP_ATTR(scaleValue));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ChunkGatedDeltaRuleNeumann InferShape failed.");
        return {nullptr, nullptr};
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(ChunkGatedDeltaRuleNeumann,
                                      OP_INPUT(query, key, value, beta, initialState, actualSeqLengths, gOptional),
                                      OP_OUTPUT(out, finalState), OP_ATTR(scaleValue));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ChunkGatedDeltaRuleNeumann ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr};
    }

    return {out, finalState};
}
} // namespace l0op
