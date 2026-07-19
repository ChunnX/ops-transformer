/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_sliding_tile_attention.h"

#include "../sliding_tile_attention_case_guard.h"
#include "sliding_tile_attention.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace {

static const std::initializer_list<op::DataType> DTYPE_SUPPORT = {DataType::DT_FLOAT16, DataType::DT_BF16};

bool IsDtypeSupported(DataType dtype)
{
    return CheckType(dtype, DTYPE_SUPPORT);
}

aclnnStatus ReadWindows(const aclIntArray *windowSize, sliding_tile_attention::WindowArray &windows)
{
    OP_CHECK_NULL(windowSize, return ACLNN_ERR_PARAM_NULLPTR);
    uint64_t valueCount = 0;
    aclGetIntArraySize(windowSize, &valueCount);
    if (valueCount == sliding_tile_attention::WINDOW_VALUES_PER_HEAD) {
        sliding_tile_attention::WindowArray broadcastWindows{};
        for (uint64_t i = 0; i < valueCount; ++i) {
            broadcastWindows[i] = static_cast<int64_t>((*windowSize)[i]);
        }
        for (int64_t head = 0; head < sliding_tile_attention::SUPPORTED_HEADS; ++head) {
            const int64_t offset = head * sliding_tile_attention::WINDOW_VALUES_PER_HEAD;
            for (int64_t axis = 0; axis < sliding_tile_attention::WINDOW_VALUES_PER_HEAD; ++axis) {
                windows[offset + axis] = broadcastWindows[axis];
            }
        }
        return ACLNN_SUCCESS;
    }
    if (valueCount != sliding_tile_attention::WINDOW_VALUE_COUNT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "windowSize must contain one [t,h,w] triple or eight triples for the fixed task cases, got %lu.",
                valueCount);
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (uint64_t i = 0; i < valueCount; ++i) {
        windows[i] = static_cast<int64_t>((*windowSize)[i]);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclIntArray *windowSize,
                        int64_t textLength, bool hasText, int64_t seqShape, aclTensor *output, int64_t &windowPattern)
{
    OP_CHECK_NULL(q, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(k, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(v, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(output, return ACLNN_ERR_PARAM_NULLPTR);

    OP_CHECK_DTYPE_NOT_MATCH(k, q->GetDataType(), return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(v, q->GetDataType(), return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(output, q->GetDataType(), return ACLNN_ERR_PARAM_INVALID);
    if (!IsDtypeSupported(q->GetDataType())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "SlidingTileAttention supports fp16 and bf16 only.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const auto &qShape = q->GetViewShape();
    const auto &kShape = k->GetViewShape();
    const auto &vShape = v->GetViewShape();
    const auto &outShape = output->GetViewShape();
    if (qShape.GetDimNum() != sliding_tile_attention::SUPPORTED_RANK || kShape != qShape || vShape != qShape ||
        outShape != qShape) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "q/k/v/output must have equal rank-4 BNSD shapes.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const int64_t batch = qShape.GetDim(0);
    const int64_t heads = qShape.GetDim(1);
    const int64_t seq = qShape.GetDim(2);
    const int64_t headDim = qShape.GetDim(3);
    sliding_tile_attention::WindowArray windows{};
    auto ret = ReadWindows(windowSize, windows);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    windowPattern = sliding_tile_attention::GetWindowPatternId(windows);

    if (!sliding_tile_attention::IsSupportedNormalizedSignature(batch, heads, seq, headDim,
                                                                q->GetDataType() == DataType::DT_BF16, textLength,
                                                                hasText, seqShape, windows)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "SlidingTileAttention metadata is outside the fixed task acceptance case set.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

} // namespace

extern "C" aclnnStatus aclnnSlidingTileAttentionGetWorkspaceSize(const aclTensor *q, const aclTensor *k,
                                                                 const aclTensor *v, const aclIntArray *windowSize,
                                                                 int64_t textLength, bool hasText, int64_t seqShape,
                                                                 aclTensor *output, uint64_t *workspaceSize,
                                                                 aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnSlidingTileAttention, DFX_IN(q, k, v, windowSize, textLength, hasText, seqShape),
                   DFX_OUT(output));
    OP_CHECK_NULL(workspaceSize, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(executor, return ACLNN_ERR_PARAM_NULLPTR);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    int64_t windowPattern = sliding_tile_attention::INVALID_WINDOW_PATTERN;
    auto ret = CheckParams(q, k, v, windowSize, textLength, hasText, seqShape, output, windowPattern);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const aclTensor *opResult =
        l0op::SlidingTileAttention(q, k, v, windowPattern, seqShape, output, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnSlidingTileAttention(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                 aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSlidingTileAttention);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
