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
 * \file apply_rotary_pos_emb_grad_proto.h
 * \brief apply_rotary_pos_emb_grad 算子 proto 注册
 */
#ifndef OPS_OP_PROTO_INC_APPLY_ROTARY_POS_EMB_GRAD_OPS_H_
#define OPS_OP_PROTO_INC_APPLY_ROTARY_POS_EMB_GRAD_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief 双路反向旋转位置编码。同时计算 query 和 key 的 rope 反向梯度。
 * @par Inputs:
 * @li dy_q: 来自 query 输出端的上游梯度，3D 或 4D tensor，format ND，支持 float16/float32/bfloat16。
 * @li dy_k: 来自 key 输出端的上游梯度，shape 和 dtype 与 dy_q 相同或满足 broadcast 关系。
 * @li cos: 正向 cos 输入，format ND。
 * @li sin: 正向 sin 输入，format ND。
 * @li x_q: 可选输入。正向 query 输入。非空时需要同时提供 x_k，用于计算 dcos/dsin。
 * @li x_k: 可选输入。正向 key 输入。非空时需要同时提供 x_q，用于计算 dcos/dsin。
 * @par Outputs:
 * @li dx_q: query 梯度输出。
 * @li dx_k: key 梯度输出。
 * @li dcos: cos 梯度输出（仅 x_q/x_k 非空时有效）。
 * @li dsin: sin 梯度输出（仅 x_q/x_k 非空时有效）。
 * @par Attributes:
 * mode: 可选属性 int，旋转模式。当前仅支持 0 (half)。默认 0。
 * @par Constraints:
 * - 仅 Ascend 950 支持。
 * - D <= 1024，D 必须能被 2 整除。
 * - dy_q 和 dx_q 的 shape 必须相同，dy_k 和 dx_k 的 shape 必须相同。
 * - cos 和 sin 的 shape 必须相同。
 */
REG_OP(ApplyRotaryPosEmbGrad)
    .INPUT(dy_q, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(dy_k, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(cos, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(sin, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OPTIONAL_INPUT(x_q, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OPTIONAL_INPUT(x_k, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(dx_q, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(dx_k, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(dcos, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(dsin, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .ATTR(mode, Int, 0)
    .OP_END_FACTORY_REG(ApplyRotaryPosEmbGrad)

} // namespace ge

#endif
