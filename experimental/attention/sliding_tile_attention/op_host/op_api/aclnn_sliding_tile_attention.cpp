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

#include "../../op_kernel/sliding_tile_attention_launch.h"
#include "../sliding_tile_attention_case_guard.h"
#include "sliding_tile_attention.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {

static const std::initializer_list<op::DataType> DTYPE_SUPPORT = {DataType::DT_FLOAT16, DataType::DT_BF16};

aclnnStatus CheckRequiredPointers(const aclTensor *q, const aclTensor *k, const aclTensor *v,
                                  const aclIntArray *windowSize, const aclTensor *output, const uint64_t *workspaceSize,
                                  aclOpExecutor *const *executor)
{
    OP_CHECK_NULL(q, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(k, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(v, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(windowSize, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(output, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(workspaceSize, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(executor, return ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckTensorTypes(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *output)
{
    OP_CHECK_DTYPE_NOT_MATCH(k, q->GetDataType(), return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(v, q->GetDataType(), return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(output, q->GetDataType(), return ACLNN_ERR_PARAM_INVALID);
    if (!CheckType(q->GetDataType(), DTYPE_SUPPORT)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "SlidingTileAttention supports fp16 and bf16 only.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckTensorShapes(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *output)
{
    const auto &qShape = q->GetViewShape();
    if (qShape.GetDimNum() != sliding_tile_attention::SUPPORTED_RANK || k->GetViewShape() != qShape ||
        v->GetViewShape() != qShape || output->GetViewShape() != qShape) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "q/k/v/output must have equal rank-4 BNSD shapes.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus ReadNormalizedWindows(const aclIntArray *windowSize, int64_t heads,
                                  sliding_tile_attention::WindowValues &windows)
{
    uint64_t valueCount = 0;
    auto ret = aclGetIntArraySize(windowSize, &valueCount);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    sliding_tile_attention::WindowValues raw(static_cast<size_t>(valueCount));
    for (uint64_t index = 0; index < valueCount; ++index) {
        raw[static_cast<size_t>(index)] = static_cast<int64_t>((*windowSize)[index]);
    }
    if (!sliding_tile_attention::NormalizeWindows(raw.data(), static_cast<int64_t>(raw.size()), heads, windows)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "windowSize must contain three positive values or 3*N positive values; got values=%lu, N=%ld.",
                valueCount, static_cast<long>(heads));
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckNonEmptyMetadata(const aclTensor *q, const aclIntArray *windowSize, int64_t textLength, bool hasText,
                                  int64_t seqShape)
{
    const auto &shape = q->GetViewShape();
    const int64_t batch = shape.GetDim(0);
    const int64_t heads = shape.GetDim(1);
    const int64_t sequence = shape.GetDim(2);
    const int64_t headDim = shape.GetDim(3);
    sliding_tile_attention::WindowValues windows;
    auto ret = ReadNormalizedWindows(windowSize, heads, windows);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (!sliding_tile_attention::IsSupportedMetadata(batch, heads, sequence, headDim, textLength, hasText, seqShape,
                                                     windows)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "SlidingTileAttention requires B>0, 1<=N<=128, D>0 divisible by 8, a "
                                         "supported canvas, and S=canvas+textLength.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus MakeInputsContiguous(const aclTensor *&q, const aclTensor *&k, const aclTensor *&v, aclOpExecutor *executor)
{
    q = l0op::Contiguous(q, executor);
    CHECK_RET(q != nullptr, ACLNN_ERR_INNER_NULLPTR);
    k = l0op::Contiguous(k, executor);
    CHECK_RET(k != nullptr, ACLNN_ERR_INNER_NULLPTR);
    v = l0op::Contiguous(v, executor);
    CHECK_RET(v != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

const aclTensor *PrepareLaunchOutput(aclTensor *output, aclOpExecutor *executor)
{
    if (IsContiguous(output)) {
        return output;
    }
    return executor->AllocTensor(output->GetViewShape(), output->GetDataType(), Format::FORMAT_ND);
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
    auto ret = CheckRequiredPointers(q, k, v, windowSize, output, workspaceSize, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckTensorTypes(q, k, v, output);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckTensorShapes(q, k, v, output);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    if (q->IsEmpty() || k->IsEmpty() || v->IsEmpty() || output->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    ret = CheckNonEmptyMetadata(q, windowSize, textLength, hasText, seqShape);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = MakeInputsContiguous(q, k, v, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const aclTensor *launchOutput = PrepareLaunchOutput(output, uniqueExecutor.get());
    CHECK_RET(launchOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *opResult = l0op::SlidingTileAttention(q, k, v, windowSize, textLength, hasText, seqShape,
                                                           launchOutput, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (launchOutput != output) {
        const aclTensor *copyResult = l0op::ViewCopy(opResult, output, uniqueExecutor.get());
        CHECK_RET(copyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

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
