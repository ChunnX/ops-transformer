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
 * \file qbmm_reduce_scatter_add_rms_norm_cast_proto.h
 * \brief A5 GE图IR原型 - MX量化版本
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_QBMM_REDUCE_SCATTER_ADD_RMS_NORM_CAST_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_QBMM_REDUCE_SCATTER_ADD_RMS_NORM_CAST_PROTO_H_

#include "graph/operator_reg.h"
namespace ge {

REG_OP(QbmmReduceScatterAddRmsNormCast)
    .INPUT(x1, TensorType({DT_FLOAT8_E4M3FN}))
    .INPUT(x2, TensorType({DT_FLOAT8_E4M3FN}))
    .INPUT(y, TensorType({DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(pertoken_scale, TensorType({DT_FLOAT8_E8M0}))
    .OUTPUT(y1, TensorType({DT_FLOAT}))
    .OUTPUT(y2, TensorType({DT_BF16}))
    .OUTPUT(x, TensorType({DT_BF16}))
    .REQUIRED_ATTR(group, String)
    .ATTR(rank_size, Int, 0)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(dtype, Int, 0)
    .ATTR(epsilon, Float, 1e-6f)
    .OP_END_FACTORY_REG(QbmmReduceScatterAddRmsNormCast)

}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_QBMM_REDUCE_SCATTER_ADD_RMS_NORM_CAST_PROTO_H_
