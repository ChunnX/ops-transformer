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
 * \file allto_all_matmul_v2_proto.h
 * \brief Graph-mode operator prototype for AlltoAllMatmulV2 (MX-quant FP8 only)
 */

#ifndef ALLTO_ALL_MATMUL_APACE_PROTO_H
#define ALLTO_ALL_MATMUL_APACE_PROTO_H

#include <graph/operator_reg.h>

namespace ge {
/**
 * @brief AlltoAll+Matmul fusion operator (apace UDMA path, MX-quant FP8 only).
 */
REG_OP(AlltoAllMatmulV2)
    .INPUT(context, TensorType({DT_INT32, DT_UINT8}))
    .INPUT(x1, TensorType({DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .INPUT(x2, TensorType({DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(x1_scale, TensorType({DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(x2_scale, TensorType({DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(comm_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(x1_offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(x2_offset, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16}))
    .OUTPUT(all2all_out, TensorType({DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .REQUIRED_ATTR(group, String)
    .REQUIRED_ATTR(world_size, Int)
    .REQUIRED_ATTR(hccl_buffer_size, Int)
    .ATTR(all2all_axes, ListInt, {-2, -1})
    .ATTR(y_dtype, Int, 27)
    .ATTR(x1_quant_mode, Int, 0)
    .ATTR(x2_quant_mode, Int, 0)
    .ATTR(comm_quant_mode, Int, 0)
    .ATTR(x1_quant_dtype, Int, 28)
    .ATTR(comm_quant_dtype, Int, 28)
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, true)
    .ATTR(group_size, Int, 0)
    .ATTR(alltoall_out_flag, Bool, false)
    .ATTR(comm_mode, String, "urma")
    .ATTR(precision_mode, Int, 0)
    .OP_END_FACTORY_REG(AlltoAllMatmulV2)
} // namespace ge

#endif // ALLTO_ALL_MATMUL_APACE_PROTO_H
