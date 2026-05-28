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
 * \file add_rms_norm_dynamic_quant_all_gather_qbmm_proto.h
 * \brief GE图IR原型 - AddRmsNorm + DynamicQuant(MX) + AllGather + Qbmm 融合算子
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_ADD_RMS_NORM_DYNAMIC_QUANT_ALL_GATHER_QBMM_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_ADD_RMS_NORM_DYNAMIC_QUANT_ALL_GATHER_QBMM_PROTO_H_

#include "graph/operator_reg.h"
namespace ge {

REG_OP(AddRmsNormDynamicQuantAllGatherQbmm)
    .INPUT(x1, TensorType({DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .INPUT(residual, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(y, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT}))
    .INPUT(scaleInv2, TensorType({DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(smooth_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_INT32}))
    .OUTPUT(output, TensorType({DT_BF16, DT_FLOAT16}))
    .OUTPUT(z, TensorType({DT_BF16, DT_FLOAT16}))
    .OUTPUT(dynamic_quant_out, TensorType({DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .OUTPUT(all_gather_data_out, TensorType({DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .OUTPUT(all_gather_scales_out, TensorType({DT_FLOAT8_E8M0}))
    .REQUIRED_ATTR(group, String)
    .ATTR(rank_size, Int, 0)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(dtype, Int, 0)
    .ATTR(residual_norm_mode, Int, 0)
    .ATTR(optional_output, Bool, false)
    .OP_END_FACTORY_REG(AddRmsNormDynamicQuantAllGatherQbmm)

}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_ADD_RMS_NORM_DYNAMIC_QUANT_ALL_GATHER_QBMM_PROTO_H_
