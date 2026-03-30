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
 * \file mhc_pre_sinkhorn_proto.h
 * \brief MhcPreSinkhorn operator proto definition
 */
 
#ifndef OPS_BUILT_IN_OP_PROTO_INC_MHC_PRE_SINKHORN_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_MHC_PRE_SINKHORN_PROTO_H_
#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief Perform the Sinkhorn normalization on the input tensor h_res to generate
 * a doubly stochastic matrix, and compute related normalization and summation results.
 */
REG_OP(MhcPreSinkhorn)
    .INPUT(h_res, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OUTPUT(norm_out, TensorType({DT_FLOAT}))
    .OUTPUT(sum_out, TensorType({DT_FLOAT}))
    .ATTR(eps, Float, 1.00e-06)
    .ATTR(num_iters, Int, 20)
    .ATTR(out_flag, Int, 0)
    .OP_END_FACTORY_REG(MhcPreSinkhorn)

} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_MHC_PRE_SINKHORN_PROTO_H_
