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
 * \file matmul_all_reduce_v2_proto.h
 * \brief MatmulAllReduceV2 graph operator proto definition
 */

#ifndef MATMUL_ALL_REDUCE_V2_PROTO_H_
#define MATMUL_ALL_REDUCE_V2_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief Function MatmulAllReduceV2.
 * @par Inputs:
 * @li x1: A matrix tensor, left matrix of mm. The type support float16, bf16, int8, hifloat8,
     float8_e5m2, float8_e4m3, float4_e2m1.
 * @li x2: A matrix tensor, right matrix of mm. The type support float16, bf16, int8, int4, hifloat8, float8_e5m2,
     float8_e4m3, float4_e2m1.
 * @li bias: A matrix tensor, input biasOptional in the formula. The type support float16, bf16, int32, float32.
 * @li x3: A matrix tensor, input x3Optional in the formula. The type support float16, bf16, float32.
 * @li antiquant_scale: A matrix tensor. The type support float16, bf16, float32.
 * @li antiquant_offset: A matrix tensor. The type support float16, bf16.
 * @li dequant_scale: A matrix tensor. This parameter indicates the dequantization coefficient after the mm operation.
     The shape is (1) in the per-tensor scenario and (n)/(1, n) in the per-channel scenario.
     The type support float16, bf16, uint64, int64, float32, float8_e8m0.
 * @li pertoken_scale: A matrix tensor, the per-token dequantization coefficient after the MM computation.
     The type support float32, float8_e8m0, float16, bf16.
 * @li comm_quant_scale_1: A matrix tensor. The per-channel quantization coefficient after the matmulAdd computation.
     The type support float16, bf16, float32.
 * @li comm_quant_scale_2: A matrix tensor. The per-channel quantization coefficient after the allGather computation.
     The type support float16, bf16, float32. \n

 * @par Attributes:
 * @li group: A required String identifying the group of ranks participating in the op.
 * @li reduce_op: A required string identifying the reduction operation to perform. support "sum", "min", "max", "prod", 
     currently only support "sum".
 * @li is_trans_a: A bool. If True, changes the shape of "x1" from [K, M] to [M, K] before multiplication. Default: false.
 * @li is_trans_b: A bool. If True, changes the shape of "x2" from [N, K] to [K, N] before multiplication. Default: false.
 * @li comm_turn: An int. Number of communications with AICPU. Default: 0.
 * @li antiquant_group_size: An int. Number of per-group for quant. Default: 0.
 * @li group_size: An int. group_size = group size K | group size N << 16 | group size M << 32. Default: 0.
 * @li y_dtype: An int. The y_dtype only support 0(float32)/1(float16)/27(bfloat16) in current version.
     Default: 28(ge::DataType::DT_UNDEFINED).
 * @li comm_quant_mode: An int. Number of low-bits communicate mode. Static quant: 0. Dynamic quant: 1. Default: 0.
 * @li comm_mode: A string. Communication mode. Default: "ccu". The comm_mode only supports "default", "ccu" or "aicpu" 
     in current version. \n 

 * @par Outputs:
 * y: A matrix tensor. The type support float16, bf16. \n

 * @attention Constraints:
 * MatmulAllReduceV2 has the same constraints as MatmulAllReduce.
 * - Functionality is identical to MatmulAllReduce.
 * - Three interfaces are supported: aclnn_matmul_all_reduce_v3, aclnn_quant_matmul_all_reduce_v5, 
     aclnn_weight_quant_matmul_all_reduce_v2.
 */

REG_OP(MatmulAllReduceV2)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16, DT_INT32, DT_FLOAT}))
    .OPTIONAL_INPUT(x3, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(dequant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_UINT64, DT_INT64, DT_FLOAT, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(pertoken_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(comm_quant_scale_1, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(comm_quant_scale_2, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .REQUIRED_ATTR(group, String)
    .ATTR(reduce_op, String, "sum")
    .ATTR(is_trans_a, Bool, false)
    .ATTR(is_trans_b, Bool, false)
    .ATTR(comm_turn, Int, 0)
    .ATTR(antiquant_group_size, Int, 0)
    .ATTR(group_size, Int, 0)
    .ATTR(y_dtype, Int, 28)
    .ATTR(comm_quant_mode, Int, 0)
    .ATTR(comm_mode, String, "")
    .OP_END();

} // namespace ge

#endif // MATMUL_ALL_REDUCE_V2_PROTO_H_