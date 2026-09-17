/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_MOE_TUTEL_COMBINE_X_H_
#define OP_API_INC_LEVEL0_MOE_TUTEL_COMBINE_X_H_

#include <array>

#include "aclnn/aclnn_base.h"
#include "opdev/shape_utils.h"

// =============================================================================
// L0 header for MoeTutelCombineX. Declares the l0op entry point and the ONE
// shared output-shape helper. The helper is used by BOTH the L0 allocation and
// the L2 parameter validation — keep it the single source of truth for the
// output shape.
//
// Output shape contract (spec.yaml outputs.x_grad / REQUIREMENTS §8.1):
//   x_grad.shape = gates.shape + (y_grad.shape[-1],)
//     - no batch: y_grad [EC, H], metadata [S]        -> x_grad [S, H]
//     - batch:    y_grad [B, EC, H], metadata [B, S]  -> x_grad [B, S, H]
//   Structural preconditions for a well-defined derivation:
//     - rank(y_grad) ∈ {2, 3} and rank(y_grad) == rank(gates) + 1
//     - gates / indices / locations are same-shaped
//     - batch form (rank 3): gates.dim0 == y_grad.dim0
// =============================================================================

namespace l0op {

// Derives the output shape from the inputs. Returns false when the structural
// shape contract is violated (bad rank relation, mismatched metadata shapes,
// or inconsistent batch dim); on success outShape is gates' view shape with
// y_grad's last dim appended.
inline bool MoeTutelCombineXOutShape(const aclTensor *yGrad, const aclTensor *gates, const aclTensor *indices,
                                     const aclTensor *locations, op::Shape &outShape)
{
    const op::Shape &yShape = yGrad->GetViewShape();
    const op::Shape &gShape = gates->GetViewShape();
    const size_t rankY = yShape.GetDimNum();
    const size_t rankG = gShape.GetDimNum();
    // rank(y_grad) ∈ {2, 3}; rank(gates) ∈ {1, 2}; rank(y_grad) == rank(gates) + 1
    if (rankY < 2 || rankY > 3 || rankY != rankG + 1) {
        return false;
    }
    // gates / indices / locations must be same-shaped.
    if ((gShape != indices->GetViewShape()) || (gShape != locations->GetViewShape())) {
        return false;
    }
    // Batch form: the micro-batch dim must agree.
    if (rankY == 3 && yShape.GetDim(0) != gShape.GetDim(0)) {
        return false;
    }
    outShape = gShape;
    outShape.AppendDim(yShape.GetDim(rankY - 1));
    return true;
}

// L0 entry point. Allocates x_grad and records the AICore kernel in the
// executor's launch list. Returns the allocated output tensor; all-null array
// on failure. Parameter order follows the OpDef input declaration order
// (y_grad, gates, indices, locations); the operator declares no attrs.
std::array<const aclTensor *, 1> MoeTutelCombineX(const aclTensor *yGrad, const aclTensor *gates,
                                                  const aclTensor *indices, const aclTensor *locations,
                                                  aclOpExecutor *executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_MOE_TUTEL_COMBINE_X_H_
