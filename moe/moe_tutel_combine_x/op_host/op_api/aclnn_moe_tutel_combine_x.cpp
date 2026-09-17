/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// L2 implementation for MoeTutelCombineX: stage 1 (plan) + stage 2 (run).
// Stage 1 validates the parameters, builds the executor plan around the L0
// call, and reports the workspace size. Stage 2 hands everything to the
// framework. L2 never launches kernels and never touches streams.
//
// Contract (docs/aclnnMoeTutelCombineX.md 参数说明/约束说明):
//   inputs : y_grad [EC,H]/[B,EC,H] {fp32,fp16,bf16}, gates same dtype
//            [S]/[B,S], indices/locations int32 same shape as gates
//   output : x_grad = gates.shape + (y_grad.shape[-1],), dtype = y_grad's
//   attrs  : none
// =============================================================================

#include "aclnn_moe_tutel_combine_x.h"
#include "moe_tutel_combine_x.h"

#include <initializer_list>

#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace {

// Dtype whitelist from the OpDef: y_grad / gates / x_grad share
// {DT_FLOAT, DT_FLOAT16, DT_BF16}; indices / locations are int32.
const std::initializer_list<op::DataType> MoeTutelCombineX_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

bool CheckNotNull(const aclTensor *yGrad, const aclTensor *gates, const aclTensor *indices, const aclTensor *locations,
                  const aclTensor *out)
{
    OP_CHECK_NULL(yGrad, return false);
    OP_CHECK_NULL(gates, return false);
    OP_CHECK_NULL(indices, return false);
    OP_CHECK_NULL(locations, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

bool CheckDtypeValid(const aclTensor *yGrad, const aclTensor *gates, const aclTensor *indices,
                     const aclTensor *locations, const aclTensor *out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(yGrad, MoeTutelCombineX_DTYPE_SUPPORT_LIST, return false);
    // No dtype promotion: gates and x_grad must follow y_grad.
    OP_CHECK_DTYPE_NOT_MATCH(gates, yGrad->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(indices, op::DataType::DT_INT32, return false);
    OP_CHECK_DTYPE_NOT_MATCH(locations, op::DataType::DT_INT32, return false);
    OP_CHECK_DTYPE_NOT_MATCH(out, yGrad->GetDataType(), return false);
    return true;
}

// ND-only layout contract (README.md / docs/aclnnMoeTutelCombineX.md 约束说明
// 「仅支持ND格式」), checked on every user-facing tensor including the output.
// In graph mode the ND-only contract is enforced by the InferShape gate
// (moe_tutel_combine_x_infershape.cpp); the tiling gate checks shapes and
// dtypes but not formats. In the aclnn path the L0 output is allocated
// internally (always ND) and the user's out only receives a ViewCopy — so a
// non-ND user output is invisible to the graph-mode checks and must be
// rejected here (the L2 aclnn self-check carries the negative behavior,
// OpDef NeedCheckSupportFlag(false)).
bool CheckFormat(const aclTensor *yGrad, const aclTensor *gates, const aclTensor *indices, const aclTensor *locations,
                 const aclTensor *out)
{
    for (const aclTensor *tensor : {yGrad, gates, indices, locations, out}) {
        if (tensor->GetStorageFormat() != op::Format::FORMAT_ND) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "MoeTutelCombineX only supports ND format, got format %d.",
                    static_cast<int32_t>(tensor->GetStorageFormat()));
            return false;
        }
    }
    return true;
}

bool CheckShape(const aclTensor *yGrad, const aclTensor *gates, const aclTensor *indices, const aclTensor *locations,
                const aclTensor *out)
{
    // Same derivation the L0 allocation uses — the shared helper (it also
    // enforces the structural contract: rank relation, same-shaped metadata,
    // batch dim).
    op::Shape expected;
    if (!l0op::MoeTutelCombineXOutShape(yGrad, gates, indices, locations, expected)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "MoeTutelCombineX output shape derivation failed.");
        return false;
    }
    if (out->GetViewShape() != expected) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out shape mismatch.");
        return false;
    }
    return true;
}

} // namespace

extern "C" {

aclnnStatus aclnnMoeTutelCombineXGetWorkspaceSize(const aclTensor *yGrad, const aclTensor *gates,
                                                  const aclTensor *indices, const aclTensor *locations,
                                                  const aclTensor *out, uint64_t *workspaceSize,
                                                  aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnMoeTutelCombineX, DFX_IN(yGrad, gates, indices, locations), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    OP_CHECK_NULL(workspaceSize, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(executor, return ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckNotNull(yGrad, gates, indices, locations, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(yGrad, gates, indices, locations, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(yGrad, gates, indices, locations, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(yGrad, gates, indices, locations, out), ACLNN_ERR_PARAM_INVALID);

    // Empty output (S=0 / H=0 / B=0): nothing to compute, nothing to launch.
    // Note: an empty y_grad with a non-empty out (EC=0 degradation, S>0) is
    // deliberately NOT early-returned — the kernel's zero-fill path must still
    // run, so only a truly empty output short-circuits here.
    if (out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // Non-contiguous inputs need contiguous copies first (one per input).
    auto yGradContiguous = l0op::Contiguous(yGrad, uniqueExecutor.get());
    CHECK_RET(yGradContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gatesContiguous = l0op::Contiguous(gates, uniqueExecutor.get());
    CHECK_RET(gatesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
    CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto locationsContiguous = l0op::Contiguous(locations, uniqueExecutor.get());
    CHECK_RET(locationsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Core computation via L0.
    auto result = l0op::MoeTutelCombineX(yGradContiguous, gatesContiguous, indicesContiguous, locationsContiguous,
                                         uniqueExecutor.get());
    CHECK_RET(result[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Copy the result into the user-provided (possibly non-contiguous) out.
    CHECK_RET(l0op::ViewCopy(result[0], out, uniqueExecutor.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeTutelCombineX(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMoeTutelCombineX);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

} // extern "C"
