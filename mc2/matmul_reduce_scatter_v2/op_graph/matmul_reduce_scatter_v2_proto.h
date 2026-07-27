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
 * \file matmul_reduce_scatter_v2_proto.h
 * \brief
 */
#ifndef MATMUL_REDUCE_SCATTER_V2_PROTO_H_
#define MATMUL_REDUCE_SCATTER_V2_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Fusion op of matmul and reduce scatter v2.
* @par Inputs:
* six inputs, including:
* @li x1: A matrix tensor. The type supports float16, bfloat16, int8, float8_e4m3fn and float8_e5m2. The format supports
ND. The x1 only supports two dimensions in current version, for example (M, K). The x1 doesn't support transposed.
* @li x2: A matrix tensor. The type supports float16, bfloat16, int8, float8_e4m3fn and float8_e5m2. The x2 only
supports two dimensions in current version, for example (K, N). The x2 supports transposed and non-transposed. The K
value in x2 should be same as the K value in x1 when x2 is non-transposed, and the K value should be in range [256,
65535).
* @li bias: An optional tensor in ND format. For int8 inputs its type supports float16, bfloat16 and float32; for
float16 or bfloat16 inputs its type supports the corresponding input type or float32. For MXFP8 inputs it must be
float32. \n
* @li x1_scale: An optional float32 or float8_e8m0 tensor in ND format. In MXFP8 mode it is required and has shape
(M, ceil(K / 64), 2). \n
* @li x2_scale: An optional float32, int64 or float8_e8m0 tensor in ND format. In MXFP8 mode it is required and has
shape (ceil(K / 64), N, 2), or (N, ceil(K / 64), 2) when x2 is transposed. \n
* @li quant_scale: A matrix tensor. The type supports float32. The format supports ND. The quant_scale only supports one
dimension and only one element, for example (1,). The quant_scale only supports nullptr in current version. \n
*
* @par Attributes:
* @li group: A required String identifying the group of ranks
  participating in the op.
* @li reduce_op: A required string identifying the reduction operation to perform. Default: "sum". The reduce_op only
supports sum in current version.
* @li is_trans_a: A bool. If True, changes the shape of "x1" from [K, M] to [M, K] before multiplication.
* If False, don't change the shape of "x1" before multiplication. Default: "false". The is_trans_a only supports false
in current version.
* @li is_trans_b: A bool. If True, changes the shape of "x2" from [N, K] to [K, N] before multiplication.
* If False, don't change the shape of "x2" before multiplication. Default: "false".
* @li comm_turn: An int. Number of communications. Default: "0". The comm_turn only supports 0 in current
version.
* @li rank_size: An int. Number of rank num. Default: "0". The rank_size only supports 2/4/8 in current version.
* @li block_size: An int. Number of values in the M, N directions in y correspond to one dequantization coefficient.
Default: "0". The block_size only supports 0 in current version.
* @li group_size: An int. Reserved attribute. Default: "0". MXFP8 also accepts 4295032864, the packed value for
[1, 1, 32]. \n
* @li is_amax_out: A bool. whether the output of amax_out is supported . Default: "false". If True, supports output
amax_out.
* @li y_dtype: An int. The y_dtype only supports 1(float16)/27(bfloat16) in current version. \n
* @li comm_mode: A string. Communication mode. Default: "aiv". Only "aiv" is supported in current
version. \n
*
* @par Outputs:
* @li y: A matrix tensor. The type supports float16 and bfloat16. For float16/bfloat16 inputs, the type is the same as
x1. The format supports ND. The y is two dimensions, for example (M/rank_size, N).
* @li amax_out: A matrix tensor. The type supports float32. The format supports ND. The amax_out is 1 dimension and only
one element, for example (1,). The amax_out only supports nullptr in current version. \n
*/
REG_OP(MatmulReduceScatterV2)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(x1_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(x2_scale, TensorType({DT_FLOAT, DT_INT64, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(quant_scale, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(amax_out, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(group, String)
    .ATTR(reduce_op, String, "sum")
    .ATTR(is_trans_a, Bool, false)
    .ATTR(is_trans_b, Bool, false)
    .ATTR(comm_turn, Int, 0)
    .ATTR(rank_size, Int, 0)
    .ATTR(block_size, Int, 0)
    .ATTR(group_size, Int, 0)
    .ATTR(is_amax_out, Bool, false)
    .ATTR(y_dtype, Int, 0)
    .ATTR(comm_mode, String, "aiv")
    .OP_END_FACTORY_REG(MatmulReduceScatterV2)
} // namespace ge
#endif // MATMUL_REDUCE_SCATTER_V2_PROTO_H_
