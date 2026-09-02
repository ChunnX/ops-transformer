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

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(SlidingTileAttentionFai);

const aclTensor *SlidingTileAttention(const aclTensor *q, const aclTensor *k, const aclTensor *v,
                                      const aclIntArray *windowSize, int64_t textLength, bool hasText, int64_t seqShape,
                                      const aclTensor *output, aclOpExecutor *executor)
{
    L0_DFX(SlidingTileAttention, q, k, v, windowSize, textLength, hasText, seqShape, output);
    const aclTensor *windows = executor->ConvertToTensor(windowSize, DataType::DT_INT64);
    OP_CHECK(windows != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Failed to convert windowSize to tensor."),
             return nullptr);
    const_cast<aclTensor *>(windows)->SetStorageFormat(Format::FORMAT_ND);
    const_cast<aclTensor *>(windows)->SetViewFormat(Format::FORMAT_ND);
    const_cast<aclTensor *>(windows)->SetOriginalFormat(Format::FORMAT_ND);
    const auto &shape = q->GetViewShape();
    const int64_t batch = shape.GetDim(0);
    const int64_t heads = shape.GetDim(1);
    const int64_t sequence = shape.GetDim(2);
    const int64_t headDim = shape.GetDim(3);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(SlidingTileAttentionFai, OP_INPUT(q, k, v, windows), OP_OUTPUT(output),
                                           OP_ATTR(batch, heads, sequence, headDim, textLength, hasText, seqShape));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SlidingTileAttentionFai launch failed."),
             return nullptr);
    return output;
}

} // namespace l0op
