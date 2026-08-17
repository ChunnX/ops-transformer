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
 * \file moe_tutel_combine_x_proto.h
 * \brief
 */
#ifndef OPS_TRANSFORMER_MOE_TUTEL_COMBINE_X_PROTO_H_
#define OPS_TRANSFORMER_MOE_TUTEL_COMBINE_X_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Tutel combine function in moe. Gathers, for every token, the expert-output rows selected
 *        by the routing tables, scales them by the gate weights and reduces them over the top-k axis.

 * @par Inputs:
 * @li y_grad: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16. Shape is [E, C, H],
 *             where E is the expert number, C is the per-expert capacity and H is the hidden size.
 * @li gates: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16. Shape is [K, S],
 *            where K is topk's k size and S is the token size. Must have the same dtype as y_grad.
 * @li indices: A mutable Tensor of the type DT_INT32, for topk's k size. Shape is [K, S].
 * @li locations: A mutable Tensor of the type DT_INT32, for token size. Shape is [K, S].

 * @par Outputs:
 * @li x_grad: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16. Shape is [S, H].
 *             Has the same dtype as y_grad.\n

 * @attention Constraints:
 * @code{.c}
 *  - gates, indices and locations must have exactly the same shape.
 *  - A routing entry (k, s) contributes only when indices[k][s] >= 0 and locations[k][s] < C,
 *    which matches the drop / no-expert semantics of Tutel; entries that fail the check
 *    contribute zero. Values of indices beyond [0, E) or negative locations are not checked
 *    and lead to undefined behavior.
 *  - x_grad is fully written on every invocation. When K is 0 or every routing entry of a token
 *    is invalid, the corresponding row of x_grad is zero.
 * @endcode
 */
#ifndef OPS_PROTO_DEF_MOETUTELCOMBINEX
#define OPS_PROTO_DEF_MOETUTELCOMBINEX
REG_OP(MoeTutelCombineX)
    .INPUT(y_grad, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gates, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(locations, TensorType({DT_INT32}))
    .OUTPUT(x_grad, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(MoeTutelCombineX)
#endif // OPS_PROTO_DEF_MOETUTELCOMBINEX
} // namespace ge

#endif // OPS_TRANSFORMER_MOE_TUTEL_COMBINE_X_PROTO_H_
