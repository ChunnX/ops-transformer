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
#ifndef OPS_OP_PROTO_INC_MOE_TUTEL_COMBINE_X_OPS_H_
#define OPS_OP_PROTO_INC_MOE_TUTEL_COMBINE_X_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Scatter-combine gradient of Mixture-of-Experts backward: gather rows of the expert
      output gradient (y_grad) by the routing metadata (indices/locations), scale by gates and
      write them back to the original token positions of the input gradient (x_grad).
      Supports the no-batch (y_grad is 2D) and the batched (y_grad is 3D) tensor forms. 

* @par Inputs:
* @li y_grad: A 2D or 3D tensor of type BFloat16, Float16 or Float32, the expert output
      gradient in expert buffer layout. Shape supports (E*C, H) or (B, E*C, H). Format supports ND. 

* @li gates: A 1D or 2D tensor of type BFloat16, Float16 or Float32, the routing weight of each
      token copy. Shape supports (S) or (B, S). Format supports ND. 

* @li indices: A 1D or 2D tensor of type Int32, the expert id of each token copy.
      Shape supports (S) or (B, S). Format supports ND. 

* @li locations: A 1D or 2D tensor of type Int32, the offset of each token copy inside its
      expert buffer. Shape supports (S) or (B, S). Format supports ND. 

* @par Outputs:
* @li x_grad: A 2D or 3D tensor of type BFloat16, Float16 or Float32, the gradient with respect
      to the original token input. Shape supports (S, H) or (B, S, H). Format supports ND.
      x_grad.shape = gates.shape + (y_grad.shape[-1],). 

* @par Restrictions:
* @li y_grad, gates and x_grad must share the same data type; indices and locations are Int32.
      rank(y_grad) = rank(gates) + 1 and rank(y_grad) is 2 or 3. Tensors must be contiguous ND. 

* @par Third-party framework compatibility
* Compatible with the combine operation of the tutel MoE implementation (y_grad of experts in
      one expert-parallel buffer, gates/indices/locations of the token copies). 

*/
REG_OP(MoeTutelCombineX)
    .INPUT(y_grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(gates, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(locations, TensorType({DT_INT32}))
    .OUTPUT(x_grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(MoeTutelCombineX)

} // namespace ge

#endif
