/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sliding_tile_attention.h"

#include "../sliding_tile_attention_case_guard.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(SlidingTileAttentionFai);

const aclTensor *SlidingTileAttention(const aclTensor *q, const aclTensor *k, const aclTensor *v, int64_t windowPattern,
                                      int64_t seqShape, aclTensor *output, aclOpExecutor *executor)
{
    L0_DFX(SlidingTileAttention, q, k, v, windowPattern, seqShape, output);
    const int64_t runtimeSignature = sliding_tile_attention::EncodeRuntimeSignature(seqShape, windowPattern);
    const op::Shape signatureShape({runtimeSignature});
    const aclTensor *signature = executor->AllocTensor(signatureShape, DataType::DT_UINT8);
    OP_CHECK(signature != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Failed to allocate runtime signature tensor."),
             return nullptr);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(SlidingTileAttentionFai, OP_INPUT(q, k, v, signature), OP_OUTPUT(output));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SlidingTileAttentionFai launch failed."),
             return nullptr);
    return output;
}

} // namespace l0op
