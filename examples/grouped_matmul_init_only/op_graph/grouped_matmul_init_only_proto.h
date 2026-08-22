/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_init_only_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_GROUPED_MATMUL_INIT_ONLY_H_
#define OPS_OP_PROTO_INC_GROUPED_MATMUL_INIT_ONLY_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief GroupedMatmulInitOnly (init-only 头开销测量算子，不做真实 matmul 计算)。
*@par Inputs:
*Three inputs, including:
* @li x: A Tensor of type float16, shape [M, K].
* @li weight: A Tensor of type float16, shape [G, K, N].
* @li group_list: A Tensor of type int64, shape [G]. \n

*@par Outputs:
*y: A Tensor of type float16, shape [M, N].
*/
REG_OP(GroupedMatmulInitOnly)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(weight, TensorType({DT_FLOAT16}))
    .INPUT(group_list, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .OP_END_FACTORY_REG(GroupedMatmulInitOnly)

} // namespace ge

#endif // OPS_OP_PROTO_INC_GROUPED_MATMUL_INIT_ONLY_H_
