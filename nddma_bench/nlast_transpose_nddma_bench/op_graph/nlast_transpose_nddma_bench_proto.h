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
 * \file nlast_transpose_nddma_bench_proto.h
 * \brief NlastTransposeNddmaBench 图模式原型定义
 */

#ifndef OPS_OP_PROTO_INC_NLAST_TRANSPOSE_NDDMA_BENCH_OPS_H_
#define OPS_OP_PROTO_INC_NLAST_TRANSPOSE_NDDMA_BENCH_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief NlastTransposeNddmaBench: permute [0,1,3,2,4] via a single UB-internal 3D NDDMA transpose
*        (swap middle axes dim2/dim3, keep innermost contiguous axis dim4).

* @par Inputs:
* One input:
* @li x: A 5D tensor of shape (dim0, dim1, dim2, dim3, dim4). Type is float16, float32, int8. Format ND.

* @par Outputs:
* One output:
* @li y: A 5D tensor of shape (dim0, dim1, dim3, dim2, dim4). Type is float16, float32, int8. Format ND.

* @par Attributes:
* One attribute:
* @li is_mte3: An optional int, default 1. 1 = write the transposed result back to GM (accuracy),
*     0 = drop MTE3, benchmark NDDMA read bandwidth only.
*/
REG_OP(NlastTransposeNddmaBench)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8}))
    .ATTR(is_mte3, Int, 1)
    .OP_END_FACTORY_REG(NlastTransposeNddmaBench)

} // namespace ge

#endif // OPS_OP_PROTO_INC_NLAST_TRANSPOSE_NDDMA_BENCH_OPS_H_
