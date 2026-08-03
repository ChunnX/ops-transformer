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
 *
 * torch API:
 *   torch.ops.custom.npu_apply_rotary_pos_emb_backward(
 *     grad_query_embed, grad_key_embed, cos, sin,
 *     *, query=None, key=None, rotary_mode="half", layout=1)
 *
 * @par Inputs:
 * @li grad_query_embed: Q 路输出端上游梯度，3D(TND) 或 4D，format ND，支持 float16/float32/bfloat16。
 * @li grad_key_embed: K 路输出端上游梯度，除 N 维度外与 grad_query_embed 各维度相同。
 * @li cos: forward cos 输入，N 维度必须 = 1，其余维度与 grad_query_embed 一致。
 * @li sin: forward sin 输入，与 cos shape 相同。
 * @li query: 可选。forward query 输入。shape 与 grad_query_embed 一致。
 * @li key: 可选。forward key 输入。shape 与 grad_key_embed 一致。
 *   (query 和 key 必须同时非空或同时为空)
 *
 * @par Outputs:
 * @li grad_query: Q 路梯度。
 * @li grad_key: K 路梯度。
 * @li grad_cos: cos 梯度。
 * @li grad_sin: sin 梯度。
 *
 * @par Attributes:
 * @li rotary_mode: string, 旋转模式。仅支持 "half"。默认 "half"。
 * @li layout: int, layout 类型。1=BSND, 2=SBND, 3=BNSD(预留), 4=TND。默认 1。
 *
 * @par Constraints:
 * - 仅 Ascend 950 支持。
 * - D <= 1024，D 能被 2 整除。
 * - cos N 维必须 = 1; cos B ∈ {1, grad_query_embed.B}; S/D 与 grad_query_embed 一致。
 */
REG_OP(ApplyRotaryPosEmbGrad)
    .INPUT(grad_query_embed, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(grad_key_embed, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(cos, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(sin, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OPTIONAL_INPUT(query, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OPTIONAL_INPUT(key, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(grad_query, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OUTPUT(grad_key, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OPTIONAL_OUTPUT(grad_cos, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OPTIONAL_OUTPUT(grad_sin, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .ATTR(rotary_mode, String, "half")
    .ATTR(layout, Int, 1)
    .OP_END_FACTORY_REG(ApplyRotaryPosEmbGrad)

} // namespace ge

#endif
