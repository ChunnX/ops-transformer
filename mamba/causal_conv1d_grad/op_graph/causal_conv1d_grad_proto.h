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
 * \file causal_conv1d_grad_proto.h
 * \brief CausalConv1dGrad 图模式 GE IR 定义
 */
#ifndef OPS_OP_PROTO_INC_CAUSAL_CONV1D_GRAD_H_
#define OPS_OP_PROTO_INC_CAUSAL_CONV1D_GRAD_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 *@brief CausalConv1dGrad: 因果一维深度卷积 + SiLU 的反向算子。
 *@par Inputs:
 * @li grad_y: 上游梯度 [B, T, D]，float16/bfloat16
 * @li x: 前向输入 [B, T, D]
 * @li weight: 卷积权重 [kW, D]，kW=4
 * @li bias: 偏置 [D]
 *@par Outputs:
 * @li dx: 输入梯度 [B, T, D]
 * @li dweight: 权重梯度 [kW, D]
 * @li dbias: 偏置梯度 [D]
 */
REG_OP(CausalConv1dGrad)
    .INPUT(grad_y, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(weight, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(bias, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(dx, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(dweight, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(dbias, TensorType({DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(CausalConv1dGrad)

} // namespace ge

#endif // OPS_OP_PROTO_INC_CAUSAL_CONV1D_GRAD_H_
