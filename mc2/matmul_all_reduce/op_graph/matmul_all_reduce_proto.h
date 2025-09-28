/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_all_reduce_proto.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_PROTO_H_
#define MATMUL_ALL_REDUCE_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Function MatmulAllReduce.
* @par Inputs:
* @li x1: A matrix Tensor. The type support float16, bf16, int8.
* @li x2: A matrix Tensor. The type support float16, bf16, int8, int4.
* @li bias: A matrix Tensor. The type support float16, bf16, int32.
* @li x3: A matrix Tensor. The type support float16, bf16.
* @li antiquant_scale: A matrix Tensor. The type support float16, bf16.
* @li antiquant_offset: A matrix Tensor. The type support float16, bf16.
* @li dequant_scale: A matrix Tensor. The type support float16, bf16, uint64, int64, float32.
* @li pertoken_scale: A matrix Tensor. The type support float32.
* @li comm_quant_scale_1: A matrix Tensor. The type support float16, bf16.
* @li comm_quant_scale_2: A matrix Tensor. The type support float16, bf16. \n

* @par Attributes:
* @li group: A required String identifying the group of ranks
*  participating in the op.
* @li reduce_op: A required string identifying the reduction operation to
*  perform. support "sum", "min", "max", "prod", currently only support "sum".
* @li is_trans_a: A bool. If True, changes the shape of "x1" from [K, M] to
*  [M, K] before multiplication. Default: false.
* @li is_trans_b: A bool. If True, changes the shape of "x2" from [N, K] to
*  [K, N] before multiplication. Default: false.
* @li comm_turn: An int. Number of communications with AICPU. Default: 0.
* @li antiquant_group_size: An int. Number of per-group for quant. Default: 0. \n

* @par Outputs:
* y: A matrix Tensor. The type support float16, bf16. \n

* @attention Constraints:
* - Constraints for MatmulAllreudce:
* @li MatmulAllReduce has poor performance when the product of the 1th dimension(b) and 2st dimension(s) of input x1 is
small.
* @li x1 can be 2-dimensional or 3-dimensional, and the dimension is (b, s, k) or (m, k). x2 must be
*  2-dimensional and its dimension is (k, n). The axis meets the input parameter requirements of the mm operator,
*  and their k axes are equal. If bias is not empty, it is 1-dimensional.
* @li Dimensions except the last one of output are the same as those of x1. The last dimension is the same as
*  that of x2. If bias is not empty, the shape size is the same as the last dimension of output.
* @li The input data type of x1, x2 and bias (if supported) computation must be the same as the output data
*  type of output computation.
* @li The x2 matrix can be transposed or not transposed. The x1 matrix cannot be transposed.
* @li The Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component support
1, 2, 4, and 8 cards.
*
* - Constraints for WeightQuantMatmulAllreudce:
* @li WeightQuantMatmulAllreudce has poor performance when the product of the 1th dimension(b) and 2st dimension(s) of
input x1 is small.
* @li x1 can be 2-dimensional or 3-dimensional, and the dimension is (b, s, k) or (m, k). x2 must be
*  2-dimensional. Its dimension is (k, n). The k axis meets the input parameter requirements of the matmul operator.
*  Their k axes are equal. The range of k and n is [1, 65535].
* @li The passed x1, x2, antiquant_scale, or output cannot be a null pointer.
* @li Dimensions except the last one of x3 (non-empty) and output are the same as those of x1. The last
*  dimension of x3 (non-empty) and output are the same as that of x2. If bias is not empty, the shape
*  size is the same as the last dimension of output. The shape of antiquant_scale is [1] in the per-tensor
*  scenario, [1,n]\[n] in the per-channel scenario, and [ceil(k,antiquant_group_size),n] in the per-group scenario. If
*  `n` is 1, there is only one element in both per-tensor and per-channel scenarios, and the per-channel scenario
*  equals the per-tensor scenario. If antiquantOffset is not empty, the shape is the same as that of antiquant_scale.
* @li The data types and data formats of x1, x2, x3 (non-empty), antiquant_scale,
*  antiquantOffset (non-empty), output, and bias (non-empty) must be supported.
* @li The output data types of x1, antiquant_scale, antiquantOffset (non-empty), x3 (non-empty), and
*  bias (non-empty) must be the same.
* @li The value of antiquant_group_size must be within the value range and be a multiple of 32.
* @li The x2 matrix can be transposed or not transposed. The x1 matrix cannot be transposed.
* @li In the long sequence scenario, as b/s or m increases, OOM or computation timeout may occur.
* @li When the format of x2 is FRACTAL_NZ, only two dimensions are supported. CalculateMatmulWeightSizeV2
*  TransMatmulWeightGetWorkspaceSize/TransMatmulWeight needs to be used to convert the format ND into NZ.
* @li The Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component support
1, 2, 4, and 8 cards.
*
* - Constraints for QuantMatmulAllreudce:
* @li QuantMatmulAllreudce has poor performance when the product of the 1th dimension(b) and 2st dimension(s) of input
x1 is small.
* @li x1 can be a 2-dimensional or 3-dimensional tensor and cannot be empty. The dimension of x1 is (b, s, k)
*  or (m, k). x2 must be 2-dimensional. Its dimension is (k, n). The k axis meets the input parameter
*  requirements of the mm operator, and their k axes are equal.
* @li Dimensions except the last one of output are the same as those of x1. The last dimension is the same as
*  that of x2. If bias is not empty, the shape size is the same as the last dimension of output. If x3
*  is not empty, the shape size is the same as that of output.
* @li The passed x1, x2, dequantScale, or output cannot be a null pointer.
* @li The data types and data formats of x1, x2, dequantScale, output, bias (non-empty),
*  and x3 (non-empty) must be within the supported ranges.
* @li If output is of FLOAT16 type, the type of dequantScale is INT64 or UINT64 (x3 is not supported in
*  this case). If  output is of BFLOAT16 type, the types of dequantScaleand x3 both are BFLOAT16.
* @li The value of reduce_op must be within the available range. Currently, only sum is supported.
* @li The x2 matrix can be transposed or not transposed. The x1 matrix cannot be transposed.
* @li The Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component support
1, 2, 4, and 8 cards.
*/
REG_OP(MatmulAllReduce)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_INT4}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16, DT_INT32}))
    .OPTIONAL_INPUT(x3, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(dequant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_UINT64, DT_INT64, DT_FLOAT}))
    .OPTIONAL_INPUT(pertoken_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(comm_quant_scale_1, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(comm_quant_scale_2, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(group, String)
    .ATTR(reduce_op, String, "sum")
    .ATTR(is_trans_a, Bool, false)
    .ATTR(is_trans_b, Bool, false)
    .ATTR(comm_turn, Int, 0)
    .ATTR(antiquant_group_size, Int, 0)
    .OP_END_FACTORY_REG(MatmulAllReduce)

} // namespace ge

#endif // MATMUL_ALL_REDUCE_PROTO_H_