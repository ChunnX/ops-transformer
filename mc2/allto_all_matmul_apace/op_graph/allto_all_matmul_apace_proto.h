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
 * \file allto_all_matmul_apace_proto.h
 * \brief Graph-mode operator prototype for AlltoAllMatmulApace (MX-quant FP8 only)
 */

#ifndef ALLTO_ALL_MATMUL_APACE_PROTO_H
#define ALLTO_ALL_MATMUL_APACE_PROTO_H

#include <graph/operator_reg.h>

namespace ge {
/**
 * @brief AlltoAll+Matmul fusion operator (apace UDMA path, MX-quant FP8 only).
 *
 * @par Inputs:
 * @li x1: Left matrix. Type: float8_e4m3fn, float8_e5m2. Shape: [M, K]. Format: ND.
 * @li x2: Right matrix. Type: float8_e4m3fn, float8_e5m2. Shape: [N, K]. Format: ND.
 * @li x1_scale: MX scale for x1. Type: float8_e8m0. Shape: [M, Ceil(K/64), 2]. Format: ND.
 * @li x2_scale: MX scale for x2. Type: float8_e8m0. Shape: [N, Ceil(K/64), 2]. Format: ND.
 *
 * @par Outputs:
 * @li y: Output matrix. Type: bf16. Shape: [M/world_size, N]. Format: ND.
 *
 * @par Attributes:
 * @li group: A string. Communication domain identifier.
 * @li world_size: An int. Number of ranks (2, 4, 8, 16).
 * @li transpose_x2: A bool. Whether x2 is transposed. Default: true.
 * @li hccl_buffer_size: An int. HCCL buffer size.
 */
REG_OP(AlltoAllMatmulApace)
    .INPUT(x1, TensorType({DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .INPUT(x2, TensorType({DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .INPUT(x1_scale, TensorType({DT_FLOAT8_E8M0}))
    .INPUT(x2_scale, TensorType({DT_FLOAT8_E8M0}))
    .OUTPUT(y, TensorType({DT_BF16}))
    .REQUIRED_ATTR(group, String)
    .REQUIRED_ATTR(world_size, Int)
    .ATTR(hccl_buffer_size, Int, 0)
    .ATTR(transpose_x2, Bool, true)
    .OP_END_FACTORY_REG(AlltoAllMatmulApace)
} // namespace ge

#endif // ALLTO_ALL_MATMUL_APACE_PROTO_H
