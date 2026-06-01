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
 * \file inplace_partial_rotary_mul_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_INPLACE_PARTIAL_ROTARY_MUL_OPS_H_
#define OPS_OP_PROTO_INC_INPLACE_PARTIAL_ROTARY_MUL_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Apply inplace partial rotary position embedding for a single tensor.
 * @par Inputs:
 * @li x: A 4D tensor which rotary position embedding is applied inplace, format supports ND, and data type must be
 * float16, float or bfloat16.
 * @li cos: A 4D tensor which is "cos" in rotary position embedding, format supports ND, data type must be the same as
 * "x", and shape must be the same as "sin".
 * @li sin: A 4D tensor which is "sin" in rotary position embedding, format supports ND, data type must be the same as
 * "x", and shape must be the same as "cos".
 * @par Outputs:
 * x: A 4D tensor which is the result of inplace rotary position embedding, format supports ND, data type must be
 * the same as input "x", and shape must be the same as input "x".
 * @par Attributes:
 * rotary_mode: An optional attribute of type int, specifying the mode of rotary position embedding, must be 0-"half",
 * 1-"interleave", 2-"quarter" or 3-"interleave-half". Defaults to 0.
 * partial_slice: An optional attribute of type list int, specifying the slice range [start, end] for partial rotary.
 * Defaults to [0, 0].
 * @attention Constraints:
 * Let (B, S, N, D) represents the shape of the 4-D input "x". Under this representation, the shape constraints of each
 * parameter can be described as follows:
 * @li The D of "x", "cos", "sin" and output "x" must be equal.
 * @li In half, interleave and interleave-half mode, D must be a multiple of 2. In quarter mode, D must be a multiple
 * of 4.
 */
REG_OP(InplacePartialRotaryMul)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16, DT_FLOAT16, DT_BFLOAT16}))
    .INPUT(cos, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16, DT_FLOAT, DT_FLOAT}))
    .INPUT(sin, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16, DT_FLOAT16, DT_BFLOAT16}))
    .ATTR(rotary_mode, Int, 0)
    .ATTR(partial_slice, ListInt, {0, 0})
    .OP_END_FACTORY_REG(InplacePartialRotaryMul)

} // namespace ge

#endif
