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
 * \file turbo_quant_attention_score.cpp
 * \brief L0-level API for TurboQuantAttentionScore.
 */

#include "turbo_quant_attention_score.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(TurboQuantAttentionScore);

const aclTensor *TurboQuantAttentionScore(const aclTensor *queryRotated, const aclTensor *queryQjl,
                                          const aclTensor *keyCacheIdx, const aclTensor *keyCacheQjl,
                                          const aclTensor *keyCacheNorm, const aclTensor *keyCacheGamma,
                                          const aclTensor *blockTable, const aclTensor *seqLens, int64_t mseBits,
                                          aclOpExecutor *executor)
{
    L0_DFX(TurboQuantAttentionScore, queryRotated, queryQjl, keyCacheIdx, keyCacheQjl, keyCacheNorm, keyCacheGamma,
           blockTable, seqLens, mseBits);

    const Format format = Format::FORMAT_ND;
    auto out = executor->AllocTensor(DataType::DT_FLOAT, format, format);
    OP_CHECK(out != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "attn_scores AllocTensor failed."), return nullptr);

    auto ret = INFER_SHAPE(
        TurboQuantAttentionScore,
        OP_INPUT(queryRotated, queryQjl, keyCacheIdx, keyCacheQjl, keyCacheNorm, keyCacheGamma, blockTable, seqLens),
        OP_OUTPUT(out), OP_ATTR(mseBits));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return nullptr, "TurboQuantAttentionScore InferShape failed.");

    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        TurboQuantAttentionScore,
        OP_INPUT(queryRotated, queryQjl, keyCacheIdx, keyCacheQjl, keyCacheNorm, keyCacheGamma, blockTable, seqLens),
        OP_OUTPUT(out), OP_ATTR(mseBits));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
                                         "TurboQuantAttentionScore ADD_TO_LAUNCHER_LIST_AICORE failed.");

    return out;
}

} // namespace l0op
