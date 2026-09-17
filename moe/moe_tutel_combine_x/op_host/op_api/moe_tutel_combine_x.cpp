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
// L0 implementation for MoeTutelCombineX. Registers the op type, allocates the
// output on the executor, and records the AICore kernel in the executor's
// launch list. The kernel itself (op_kernel/) and its tiling (op_host/) are
// launched by the framework in stage 2 — this file never touches streams or
// memcpy.
// =============================================================================

#include "moe_tutel_combine_x.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(MoeTutelCombineX);

static std::array<const aclTensor *, 1> MoeTutelCombineXAiCore(const aclTensor *yGrad, const aclTensor *gates,
                                                               const aclTensor *indices, const aclTensor *locations,
                                                               const aclTensor *out, aclOpExecutor *executor)
{
    L0_DFX(MoeTutelCombineXAiCore, yGrad, gates, indices, locations, out);

    // MoeTutelCombineX declares no attrs (see op_graph/moe_tutel_combine_x_proto.h
    // REG_OP — four inputs, one output, no .Attr entries), so the launcher list
    // takes only the four inputs and the single output, in OpDef declaration
    // order (y_grad, gates, indices, locations -> x_grad).
    auto ret =
        ADD_TO_LAUNCHER_LIST_AICORE(MoeTutelCombineX, OP_INPUT(yGrad, gates, indices, locations), OP_OUTPUT(out));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return {},
                                         "MoeTutelCombineX ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return {out};
}

std::array<const aclTensor *, 1> MoeTutelCombineX(const aclTensor *yGrad, const aclTensor *gates,
                                                  const aclTensor *indices, const aclTensor *locations,
                                                  aclOpExecutor *executor)
{
    OP_CHECK_NULL(yGrad, return {});
    OP_CHECK_NULL(gates, return {});
    OP_CHECK_NULL(indices, return {});
    OP_CHECK_NULL(locations, return {});
    OP_CHECK_NULL(executor, return {});

    // Output shape: derived with the shared helper from the L0 header.
    op::Shape outShape;
    if (!MoeTutelCombineXOutShape(yGrad, gates, indices, locations, outShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "MoeTutelCombineX output shape derivation failed.");
        return {};
    }

    // x_grad dtype follows y_grad (dtype_policy: same_as_first_input).
    const aclTensor *out = executor->AllocTensor(outShape, yGrad->GetDataType());
    return MoeTutelCombineXAiCore(yGrad, gates, indices, locations, out, executor);
}

} // namespace l0op
