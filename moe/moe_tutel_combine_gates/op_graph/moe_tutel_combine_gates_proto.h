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
 * \file moe_tutel_combine_gates_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_MOETUTELCOMBINEGATES_H_
#define OPS_OP_PROTO_INC_MOETUTELCOMBINEGATES_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Compute the gate-weight gradient of a Tutel MoE combine step: gather one row of the dispatch
 * buffer `y_grad` per (batch, sample) task through `indices` / `locations`, and take the hidden-dimension
 * inner product with the token row of `x`.
 *
 * @par Inputs:
 * @li x: A 2D tensor, the token embedding matrix, with shape (S, H), format supports ND, and data type
 *  must be bfloat16, float16 or float32. S is the sample count and H the hidden size.
 * @li y_grad: A 3D tensor, the dispatch buffer holding the per-expert rows, with shape (B, C, H), format
 *  supports ND, and data type must be the same as that of x. B is the batch size, C the expert capacity
 *  and H the hidden size, which must equal x.shape[1].
 * @li indices: A 2D tensor, the expert id of every (batch, sample) task, with shape (B, S), format supports
 *  ND, and data type must be int32. It is consumed as a flat array of S*B elements.
 * @li locations: A 2D tensor, the slot position inside the expert of every (batch, sample) task, with shape
 *  (B, S), format supports ND, and data type must be int32. It is consumed as a flat array of S*B elements.
 *
 * @par Outputs:
 * @li gates_grad: A 2D tensor, the per-task gate gradient, with shape (B, S) equal to indices.shape, format
 *  supports ND, and data type must be the same as that of x. Tasks whose routing is out of range
 *  (indices < 0, indices >= B, locations < 0 or locations >= C) evaluate to exactly 0.
 *
 * @par Attributes:
 * No attributes: the capacity C is implied by y_grad.shape[1], and S / H / B are all derived from the
 * input shapes, so none of them can be configured separately.
 *
 * @par Restrictions:
 * Warning: This operator only supports Ascend 950PR / Ascend 950DT.
 */
REG_OP(MoeTutelCombineGates)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(y_grad, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(locations, TensorType({DT_INT32}))
    .OUTPUT(gates_grad, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(MoeTutelCombineGates)

} // namespace ge

#endif // OPS_OP_PROTO_INC_MOETUTELCOMBINEGATES_H_
