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
 * \file broadcast_nddma_bench_proto.h
 * \brief BroadcastNddmaBench 图模式原型定义
 */

#ifndef OPS_OP_PROTO_INC_BROADCAST_NDDMA_BENCH_OPS_H_
#define OPS_OP_PROTO_INC_BROADCAST_NDDMA_BENCH_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief BroadcastNddmaBench: broadcast along one/two axes via a single UB-internal 3D NDDMA
*        (srcStride=0 on broadcast axes). Output is always full 5D (dim0,dim1,dim2,dim3,dim4).

* @par Inputs:
* One input:
* @li x: A 5D tensor whose broadcast axes have length 1. Type is float16, float32, int8. Format ND.

* @par Outputs:
* One output:
* @li y: A full 5D tensor of shape (dim0, dim1, dim2, dim3, dim4). Type is float16, float32, int8. Format ND.

* @par Attributes:
* @li is_mte3: An optional int, default 1. 1 = write the broadcast result back to GM (accuracy),
*     0 = drop MTE3, benchmark NDDMA read bandwidth only.
* @li scene: An optional int, default 0. Broadcast scene: 0=AB / 1=BA / 2=BAB / 3=ABA.
* @li dim2: An optional int, default 0. Output axis-2 length (NDDMA outer axis).
* @li dim3: An optional int, default 0. Output axis-3 length (NDDMA middle axis).
* @li dim4: An optional int, default 0. Output axis-4 length (NDDMA innermost contiguous axis).
*/
REG_OP(BroadcastNddmaBench)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8}))
    .ATTR(is_mte3, Int, 1)
    .ATTR(scene, Int, 0)
    .ATTR(dim2, Int, 0)
    .ATTR(dim3, Int, 0)
    .ATTR(dim4, Int, 0)
    .OP_END_FACTORY_REG(BroadcastNddmaBench)

} // namespace ge

#endif // OPS_OP_PROTO_INC_BROADCAST_NDDMA_BENCH_OPS_H_
