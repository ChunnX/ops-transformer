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
 * \file op_transformer_proto_extend.h
 * \brief
 */
#ifndef OPS_OP_MATH_PROTO_EXTEND_H_
#define OPS_OP_MATH_PROTO_EXTEND_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief swin_transformer model specific structure.Operator only supports swin_transformer.

* @par Inputs:
* Three inputs, including:
* @li x: An ND Tensor. Must be one of the following types: float16, float, bfloat16,
         the shape should be (B*W, N, S1, S2) or (B, W, N, S1, S2).
* @li atten_mask: An ND Tensor. Must be one of the following types: float16, float, bfloat16,
                  the shape should be (W, S1, S2) or (W, 1, S1, S2) or (1, W, 1, S1, S2)
* @li relative_pos_bias: An ND Tensor. Must be one of the following types: float16, float, bfloat16.
                         the shape sholud be (N, S1, S2) or (1, N, S1, S2) or (1, 1, N, S1, S2)

* @par Attributes:
* @li scale_value: A optional attribute, the type is float. Defaults to 1.0.
* @li inner_precision_mode: A optional attribute, the type is int. Defaults to 0, reserved field.

* @par Outputs:
* One output, including:
* @li y: An ND Tensor. Must be one of the following types: float16, float, bfloat16,
         the shape should be same with x.
*/
REG_OP(MaskedSoftmaxWithRelPosBias)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BFLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_FLOAT16, DT_BFLOAT16, DT_FLOAT}))
    .INPUT(relative_pos_bias, TensorType({DT_FLOAT16, DT_BFLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BFLOAT16, DT_FLOAT}))
    .ATTR(scale_value, Float, 1.0)
    .ATTR(inner_precision_mode, Int, 0)
    .OP_END_FACTORY_REG(MaskedSoftmaxWithRelPosBias)

    /**
    * @brief AttentionScore's forward calculation.

    * @par Inputs:
    * six inputs, including:
    * @li query: A matrix Tensor. The type only support float16. Enter a 4D Tensor.
    * @li key: A matrix Tensor. The type only support float16. Enter a 4D Tensor.
    * @li value: A matrix Tensor. The type only support float16. Enter a 4D Tensor.
    * @li padding_mask: A matrix Tensor. The type only support float16. Enter a 4D Tensor.
    * @li scale: A scalar. The type only support float16. Enter a 4D Tensor.
    * @li drop_mask: A matrix Tensor. An optional input parameter. The type only support uint8. Enter a 4D Tensor.

    * @par Attributes:
    * @li keep_prob: A float. The keep probability of dropout. Default: 1.0.
    * @li query_transpose: A bool. If True, changes the shape of "query" from [B, N, S, D] to [B, N, D, S].
    * Default: false.
    * @li key_transpose: A bool. If True, changes the shape of "key" from [B, N, S, D] to [B, N, D, S].
    * Default: false.
    * @li bmm_score_transpose_a: A bool. If True, changes the shape of "mid_data" from [B, N, S, D] to [B, N, D, S].
    * Default: false.
    * @li bmm_score_transpose_b: A bool. If True, changes the shape of "value" from [B, N, S, D] to [B, N, D, S].
    * Default: false.
    * @li softmax_axes: A list of int. The dimension softmax would be performed on. Defaults to "[-1]".

    * @par Outputs:
    * attention_score: The result matrix Tensor. The type only support float16. The output shape is the same as query.
    * softmax_output: The result matrix Tensor. The type only support float16. The output shape is the same as query.

    * @par Restrictions:
    * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
    */
    REG_OP(AttentionScore)
    .INPUT(query, TensorType({DT_FLOAT16}))
    .INPUT(key, TensorType({DT_FLOAT16}))
    .INPUT(value, TensorType({DT_FLOAT16}))
    .INPUT(padding_mask, TensorType({DT_FLOAT16}))
    .INPUT(scale, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(drop_mask, TensorType({DT_INT8}))
    .OUTPUT(attention_score, TensorType({DT_FLOAT16}))
    .OUTPUT(softmax_output, TensorType({DT_FLOAT16}))
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(query_transpose, Bool, false)
    .ATTR(key_transpose, Bool, false)
    .ATTR(bmm_score_transpose_a, Bool, false)
    .ATTR(bmm_score_transpose_b, Bool, false)
    .ATTR(softmax_axes, ListInt, {-1})
    .OP_END_FACTORY_REG(AttentionScore)
    /**
    * @brief
                / (MatMul -> ConfusionTransposeD).
      LayerNorm - (MatMul -> ConfusionTransposeD).
                \ (MatMul -> ConfusionTransposeD).
    * @par Inputs:
    * Nine inputs, including:
    * @li x: A matrix Tensor. Must be one of the following types: float16. Enter a 4D Tensor.
    * @li kernel_query: A matrix Tensor. Must be one of the following types: float16. Enter a 4D Tensor.
    * @li kernel_key: A matrix Tensor. Must be one of the following types: float16. Enter a 4D Tensor.
    * @li kernel_value: A matrix Tensor. Must be one of the following types: float16. Enter a 4D Tensor.
    * @li gamma: A matrix Tensor. Must be one of the following types: float16. Enter a 4D Tensor.
    * @li beta: A matrix Tensor. Must be one of the following types: float16. Enter a 4D Tensor.
    * @li bias_query: A matrix Tensor. An optional input parameter.
    * Must be one of the following types: float16. Enter a 4D Tensor.
    * @li bias_key: A matrix Tensor. An optional input parameter.
    * Must be one of the following types: float16. Enter a 4D Tensor.
    * @li bias_value: A matrix Tensor. An optional input parameter.
    * Must be one of the following types: float16. Enter a 4D Tensor.

    * @par Attributes:
    * @li epsilon: A optional attribute, the type is float32. Defaults to 1e-7.
    * @li trans_a: A optional attribute, the type is bool. Defaults to False.
    * @li trans_b: A optional attribute, the type is bool. Defaults to False.

    * @par Outputs:
    * Six outputs, including:
    * @li norm: A matrix Tensor. Must be one of the following types: float16. The output shape is the same as
    kernel_query.
    * @li query_output: A matrix Tensor. Must be one of the following types: float16.
    * The output shape is the same as kernel_query.
    * @li key_output: A matrix Tensor. Must be one of the following types: float16.
    * The output shape is the same as kernel_query.
    * @li value_output: A matrix Tensor. Must be one of the following types: float16.
    * The output shape is the same as kernel_query.
    * @li mean: A matrix Tensor. Must be one of the following types: float16.
    * The output shape is the same as kernel_query.
    * @li variance: A matrix Tensor. Must be one of the following types: float16.
    * The output shape is the same as kernel_query.

    * @par Restrictions:
    * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
    */
    REG_OP(AttentionLnQKV)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(kernel_query, TensorType({DT_FLOAT16}))
    .INPUT(kernel_key, TensorType({DT_FLOAT16}))
    .INPUT(kernel_value, TensorType({DT_FLOAT16}))
    .INPUT(gamma, TensorType({DT_FLOAT16}))
    .INPUT(beta, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(bias_query, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(bias_key, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(bias_value, TensorType({DT_FLOAT16}))
    .OUTPUT(norm, TensorType({DT_FLOAT16}))
    .OUTPUT(query_output, TensorType({DT_FLOAT16}))
    .OUTPUT(key_output, TensorType({DT_FLOAT16}))
    .OUTPUT(value_output, TensorType({DT_FLOAT16}))
    .OUTPUT(mean, TensorType({DT_FLOAT16}))
    .OUTPUT(variance, TensorType({DT_FLOAT16}))
    .ATTR(epsilon, Float, 0.0000001f)
    .ATTR(trans_a, Bool, false)
    .ATTR(trans_b, Bool, false)
    .OP_END_FACTORY_REG(AttentionLnQKV)

    /**
    * @brief Backprop W of AttentionLnQKV + ReduceSumD \n
    * @par Inputs:
    * Four inputs, including:
    * @li x: A Tensor. Must be one of the following types: float16.
    * @li query_dx: A Tensor. Must be one of the following types: float16.
    * @li key_dw: A Tensor. Must be one of the following types: float16.
    * @li value_dw: A Tensor. Must be one of the following types: float16.

    * @par Attributes:
    * @li trans_a: A optional attribute, the type is bool. Defaults to True.
    * @li trans_b: A optional attribute, the type is bool. Defaults to False.
    * @li trans_dw: A optional attribute, the type is bool. Defaults to False. \n

    * @par Outputs:
    * Six outputs, including:
    * @li dw_query: A Tensor. Must be one of the following types: float16.
    * @li dw_key: A Tensor. Must be one of the following types: float16.
    * @li dw_value: A Tensor. Must be one of the following types: float16.
    * @li dbias_query: A Tensor. Must be one of the following types: float16.
    * @li dbias_key: A Tensor. Must be one of the following types: float16.
    * @li dbias_value: A Tensor. Must be one of the following types: float16. \n

    * @par Restrictions:
    * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use. \n
    */
    REG_OP(AttentionQKVGradW)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(query_dx, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(key_dw, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(value_dw, TensorType({DT_FLOAT16}))
    .OUTPUT(dw_query, TensorType({DT_FLOAT16}))
    .OUTPUT(dw_key, TensorType({DT_FLOAT16}))
    .OUTPUT(dw_value, TensorType({DT_FLOAT16}))
    .OUTPUT(dbias_query, TensorType({DT_FLOAT16}))
    .OUTPUT(dbias_key, TensorType({DT_FLOAT16}))
    .OUTPUT(dbias_value, TensorType({DT_FLOAT16}))
    .ATTR(trans_a, Bool, true)
    .ATTR(trans_b, Bool, false)
    .ATTR(trans_dw, Bool, false)
    .OP_END_FACTORY_REG(AttentionQKVGradW)

    /**
    * @brief Backprop X of AttentionLnQKV + AddN \n
    * @par Inputs:
    * Seven inputs, including:
    * @li ln_dx: A Tensor. Must be one of the following types: float16.
    * @li query_dx: A Tensor. Must be one of the following types: float16.
    * @li key_dw: A Tensor. Must be one of the following types: float16.
    * @li value_dw: A Tensor. Must be one of the following types: float16.
    * @li kernel_query: A Tensor. Must be one of the following types: float16.
    * @li kernel_key: A Tensor. Must be one of the following types: float16.
    * @li kernel_value: A Tensor. Must be one of the following types: float16. \n

    * @par Attributes:
    * @li trans_a: A optional attribute, the type is bool. Defaults to False.
    * @li trans_b: A optional attribute, the type is bool. Defaults to True. \n

    * @par Outputs:
    * One outputs, including:
    * @li dx: A Tensor. Must be one of the following types: float16. \n

    * @par Restrictions:
    * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use. \n
    */
    REG_OP(AttentionQKVGradX)
    .OPTIONAL_INPUT(ln_dx, TensorType({DT_FLOAT16}))
    .INPUT(query_dx, TensorType({DT_FLOAT16}))
    .INPUT(key_dw, TensorType({DT_FLOAT16}))
    .INPUT(value_dw, TensorType({DT_FLOAT16}))
    .INPUT(kernel_query, TensorType({DT_FLOAT16}))
    .INPUT(kernel_key, TensorType({DT_FLOAT16}))
    .INPUT(kernel_value, TensorType({DT_FLOAT16}))
    .OUTPUT(dx, TensorType({DT_FLOAT16}))
    .ATTR(trans_a, Bool, false)
    .ATTR(trans_b, Bool, true)
    .OP_END_FACTORY_REG(AttentionQKVGradX)

    /**
    * @brief Function AttentionScoreGrad. \n

    * @par Inputs:
    * seven inputs, including:
    * @li attention_score: A matrix Tensor. The type only support float16.
    * @li dx: A matrix Tensor. The type only support float16.
    * @li query: A matrix Tensor. The type only support float16.
    * @li key: A matrix Tensor. The type only support float16.
    * @li value: A matrix Tensor. The type only support float16.
    * @li scale: A scalar. The type only support float16.
    * @li drop_mask: A matrix Tensor. The type only support uint8. \n

    * @par Attributes:
    * @li keep_prob: A mutable Tensor. Must met all of the following rules:
     shape of "keep_prob" should be (1,) or [1,].
    * @li query_transpose: A bool. If True, changes the shape of "query" from [K, M] to
     [M, K].
    * @li key_transpose: A bool. If True, changes the shape of "key" from [N, K] to
     [K, N].
    * @li value_transpose: A bool. If True, changes the shape of "mid_data" from [K, M] to
     [M, K].
    * @li dx_transpose: A bool. If True, changes the shape of "value" from [N, K] to
     [K, N].
    * @li softmax_axes: A int. The dimension softmax would be performed on. Defaults
     to "-1" . \n

    * @par Outputs:
    * value_dw: The result matrix Tensor. The type only support float16.
    * query_dx: The result matrix Tensor. The type only support float16.
    * key_dw: The result matrix Tensor. The type only support float16.

    * @par Restrictions:
    * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
    */
    REG_OP(AttentionScoreGrad)
    .INPUT(attention_score, TensorType({DT_FLOAT16}))
    .INPUT(dx, TensorType({DT_FLOAT16}))
    .INPUT(query, TensorType({DT_FLOAT16}))
    .INPUT(key, TensorType({DT_FLOAT16}))
    .INPUT(value, TensorType({DT_FLOAT16}))
    .INPUT(scale, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(drop_mask, TensorType({DT_INT8}))
    .OUTPUT(value_dw, TensorType({DT_FLOAT16}))
    .OUTPUT(query_dx, TensorType({DT_FLOAT16}))
    .OUTPUT(key_dw, TensorType({DT_FLOAT16}))
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(query_transpose, Bool, false)
    .ATTR(key_transpose, Bool, false)
    .ATTR(value_transpose, Bool, false)
    .ATTR(dx_transpose, Bool, false)
    .ATTR(softmax_axes, Int, -1)
    .OP_END_FACTORY_REG(AttentionScoreGrad)

    /**
     * @brief Tutel combine function in moe.
     *
     * @par Inputs:
     * @li x: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.
     * @li y_grad: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.
     * @li indices: A mutable Tensor of the type DT_INT32, for topk's k size.
     * @li locations: A mutable Tensor of the type DT_INT32, for token size.
     *
     * @par Outputs:
     * gates_grad: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.\n
     */
    REG_OP(MoeTutelCombineGates)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(y_grad, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(locations, TensorType({DT_INT32}))
    .OUTPUT(gates_grad, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(MoeTutelCombineGates)

    /**
     * @brief Tutel combine function in moe.
     *
     * @par Inputs:
     * @li y_grad: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.
     * @li gates: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.
     * @li indices: A mutable Tensor of the type DT_INT32, for topk's k size.
     * @li locations: A mutable Tensor of the type DT_INT32, for token size.
     *
     * @par Outputs:
     * @li x_grad: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.\n
     */
    REG_OP(MoeTutelCombineX)
    .INPUT(y_grad, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gates, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(locations, TensorType({DT_INT32}))
    .OUTPUT(x_grad, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(MoeTutelCombineX)

    /**
     * @brief Tutel dispatch function in moe.
     *
     * @par Inputs:
     * @li x: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.
     * @li gates: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.
     * @li indices: A mutable Tensor of the type DT_INT32, for topk's k size.
     * @li locations: A mutable Tensor of the type DT_INT32, for token size.
     *
     * @par Attributes:
     * capacity: expert capacity.
     *
     * @par Outputs:
     * y: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.\n
     */
    REG_OP(MoeTutelDispatch)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gates, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(locations, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(capacity, Int)
    .OP_END_FACTORY_REG(MoeTutelDispatch)

    /**
     * @brief MultiHeadAttention.
     * @par Inputs:
     * thirteen input, including:
     * @li query: A Tensor. Query of Attention. Support float16
     * @li key: A Tensor. Key of Attention. Support float16
     * @li value: A Tensor. Value of Attention. Support float16
     * @li query_weight: A Tensor. QueryWeight of Attention. Support float16
     * @li key_weight: A Tensor. KeyWeight of Attention. Support float16
     * @li value_weight: A Tensor. ValueWeight of Attention. Support float16
     * @li attn_mask: A Tensor. AttentionMask of Attention. Support float16
     * @li out_proj_weight: A Tensor. OutProjWeight of Attention. Support float16
     * @li query_bias: Optional Tensor. QueryBias of Attention. Support float16
     * @li key_bias: Optional Tensor. KeyBias of Attention. Support float16
     * @li value_bias: Optional Tensor. ValueBias of Attention. Support float16
     * @li out_proj_bias: Optional Tensor. OutProjBias of Attention. Support float16
     * @li dropout_mask_input: Optional Tensor. DropOutMask of Attention. Support uint8 \n

     * @par Attributes:
     * @li attn_head_num: Attention Head numbers, Support int
     * @li attn_dim_per_head: Attention dim of a Head, Support int
     * @li src_len: source length, Support int
     * @li tgt_len: target length, Support int
     * @li keep_prob: dropout keep probability, Support float
     * @li softmax_use_float: SoftMax Use Float32 to keep precision, Support bool \n

     * @par Outputs:
     * Eight output, including:
     * @li y: A Tensor. Result of Attention. Support float16
     * @li dropout_mask: DropOutMask of Attention. Support uint8
     * @li query_res: Query Result of Attention. Support float16
     * @li key_res: Key Result of Attention. Support float16
     * @li value_res: Value Result of Attention. Support float16
     * @li attn_scores: Attention Scores of SoftMax. Support float16, float
     * @li attn_res: Attention Result of SoftMax. Support float16
     * @li context: Context of Attention. Support float16

     * @par Restrictions:
     * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
     */
    REG_OP(MultiHeadAttention)
    .INPUT(query, TensorType({DT_FLOAT16}))
    .INPUT(key, TensorType({DT_FLOAT16}))
    .INPUT(value, TensorType({DT_FLOAT16}))
    .INPUT(query_weight, TensorType({DT_FLOAT16}))
    .INPUT(key_weight, TensorType({DT_FLOAT16}))
    .INPUT(value_weight, TensorType({DT_FLOAT16}))
    .INPUT(attn_mask, TensorType({DT_FLOAT16}))
    .INPUT(out_proj_weight, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(query_bias, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(key_bias, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(value_bias, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(out_proj_bias, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(dropout_mask_input, TensorType({DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .OUTPUT(dropout_mask, TensorType({DT_UINT8}))
    .OUTPUT(query_res, TensorType({DT_FLOAT16}))
    .OUTPUT(key_res, TensorType({DT_FLOAT16}))
    .OUTPUT(value_res, TensorType({DT_FLOAT16}))
    .OUTPUT(attn_scores, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(attn_res, TensorType({DT_FLOAT16}))
    .OUTPUT(context, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(attn_head_num, Int)
    .REQUIRED_ATTR(attn_dim_per_head, Int)
    .REQUIRED_ATTR(src_len, Int)
    .REQUIRED_ATTR(tgt_len, Int)
    .REQUIRED_ATTR(keep_prob, Float)
    .REQUIRED_ATTR(softmax_use_float, Bool)
    .OP_END_FACTORY_REG(MultiHeadAttention)

    /**
     * @brief MultiHeadAttentionGrad.
     * @par Inputs:
     * thirteen input, including:
     * @li query: A Tensor. Query of Attention. Support float16
     * @li key: A Tensor. Key of Attention. Support float16
     * @li value: A Tensor. Value of Attention. Support float16
     * @li query_weight: A Tensor. QueryWeight of Attention. Support float16
     * @li key_weight: A Tensor. KeyWeight of Attention. Support float16
     * @li value_weight: A Tensor. ValueWeight of Attention. Support float16
     * @li out_proj_weight: A Tensor. OutProjWeight of Attention. Support float16
     * @li query_res: A Tensor. Query Result of Attention. Support float16
     * @li key_res: A Tensor. Key Result of Attention. Support float16
     * @li value_res: A Tensor. Value Result of Attention. Support float16
     * @li attn_scores: A Tensor. Attention Scores of Attention. Support float16, float
     * @li attn_res: A Tensor. Attention Result of Attention. Support float16
     * @li context: A Tensor. Context of Attention. Support float16
     * @li y_grad: A Tensor. Grad of Attention. Support float16
     * @li dropout_mask: : A Tensor. Query Result of Attention. Support uint8 \n

     * @par Attributes:
     * @li attn_head_num: Attention Head numbers, Support int
     * @li attn_dim_per_head: Attention dim of a Head, Support int
     * @li src_len: source length, Support int
     * @li tgt_len: target length, Support int
     * @li keep_prob: dropout keep probability, Support float
     * @li softmax_use_float: SoftMax Use Float32 to keep precision, Support bool
     * @li bias_grad_mask: mask for attention has bias grad, Support list bool  \n

     * @par Outputs:
     * Eight output, including:
     * @li query_weight_grad: QueryWeight Grad of Attention. Support float16
     * @li key_weight_grad: KeyWeight Grad of Attention. Support float16
     * @li value_weight_grad: ValueWeight Grad of Attention. Support float16
     * @li out_proj_weight_grad: OutProjWeight Grad of Attention. Support float16
     * @li query_grad: Query Grad of Attention. Support float16
     * @li key_grad: Key Grad of Attention. Support float16
     * @li value_grad: Value Grad of Attention. Support float16
     * @li query_bias_grad: QueryBias Grad of Attention. Support float16
     * @li key_bias_grad: KeyBias Grad of Attention. Support float16
     * @li value_bias_grad: ValueBias Grad of Attention. Support float16
     * @li out_proj_bias_grad: OutProjBias Grad of Attention. Support float16

     * @par Restrictions:
     * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
     */
    REG_OP(MultiHeadAttentionGrad)
    .INPUT(query, TensorType({DT_FLOAT16}))
    .INPUT(key, TensorType({DT_FLOAT16}))
    .INPUT(value, TensorType({DT_FLOAT16}))
    .INPUT(query_weight, TensorType({DT_FLOAT16}))
    .INPUT(key_weight, TensorType({DT_FLOAT16}))
    .INPUT(value_weight, TensorType({DT_FLOAT16}))
    .INPUT(out_proj_weight, TensorType({DT_FLOAT16}))
    .INPUT(query_res, TensorType({DT_FLOAT16}))
    .INPUT(key_res, TensorType({DT_FLOAT16}))
    .INPUT(value_res, TensorType({DT_FLOAT16}))
    .INPUT(attn_scores, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(attn_res, TensorType({DT_FLOAT16}))
    .INPUT(context, TensorType({DT_FLOAT16}))
    .INPUT(y_grad, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(dropout_mask, TensorType({DT_UINT8}))
    .OUTPUT(query_weight_grad, TensorType({DT_FLOAT16}))
    .OUTPUT(key_weight_grad, TensorType({DT_UINT8}))
    .OUTPUT(value_weight_grad, TensorType({DT_FLOAT16}))
    .OUTPUT(out_proj_weight_grad, TensorType({DT_FLOAT16}))
    .OUTPUT(query_grad, TensorType({DT_FLOAT16}))
    .OUTPUT(key_grad, TensorType({DT_FLOAT16}))
    .OUTPUT(value_grad, TensorType({DT_FLOAT16}))
    .OUTPUT(query_bias_grad, TensorType({DT_FLOAT16}))
    .OUTPUT(key_bias_grad, TensorType({DT_FLOAT16}))
    .OUTPUT(value_bias_grad, TensorType({DT_FLOAT16}))
    .OUTPUT(out_proj_bias_grad, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(attn_head_num, Int)
    .REQUIRED_ATTR(attn_dim_per_head, Int)
    .REQUIRED_ATTR(src_len, Int)
    .REQUIRED_ATTR(tgt_len, Int)
    .REQUIRED_ATTR(keep_prob, Float)
    .REQUIRED_ATTR(softmax_use_float, Bool)
    .REQUIRED_ATTR(bias_grad_mask, ListBool)
    .OP_END_FACTORY_REG(MultiHeadAttentionGrad)

    /**
    * @brief Function SwinAttentionScore. \n

    * @par Inputs:
    * six inputs, including:
    * @li query: A matrix Tensor. The type only support float16.
    * @li key: A matrix Tensor. The type only support float16.
    * @li value: A matrix Tensor. The type only support float16.
    * @li padding_mask1: A matrix Tensor. The type only support float16.
    * @li padding_mask2: A matrix Tensor. The type only support float16.
    * @li scale: A scalar. The type only support float16.
    * @li drop_mask: A matrix Tensor. The type only support uint8. \n

    * @par Attributes:
    * @li keep_prob: A mutable Tensor. Must met all of the following rules:
     shape of "keep_prob" should be (1,) or [1,].
    * @li query_transpose: A bool. If True, changes the shape of "query" from [K, M] to
     [M, K].
    * @li key_transpose: A bool. If True, changes the shape of "key" from [N, K] to
     [K, N].
    * @li bmm_score_transpose_a: A bool. If True, changes the shape of "mid_data" from [K, M] to
     [M, K].
    * @li bmm_score_transpose_b: A bool. If True, changes the shape of "value" from [N, K] to
     [K, N].
    * @li softmax_axes: A list of int. The dimension softmax would be performed on. Defaults
     to "[]" . \n

    * @par Outputs:
    * attention_score: The result matrix Tensor. The type only support float16.
    * softmax: The result matrix Tensor. The type only support float16.

    * @par Restrictions:
    * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
    */
    REG_OP(SwinAttentionScore)
    .INPUT(query, TensorType({DT_FLOAT16}))
    .INPUT(key, TensorType({DT_FLOAT16}))
    .INPUT(value, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(padding_mask1, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(padding_mask2, TensorType({DT_FLOAT16}))
    .INPUT(scale, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(drop_mask, TensorType({DT_INT8}))
    .OUTPUT(attention_score, TensorType({DT_FLOAT16}))
    .OUTPUT(softmax, TensorType({DT_FLOAT16}))
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(query_transpose, Bool, false)
    .ATTR(key_transpose, Bool, false)
    .ATTR(bmm_score_transpose_a, Bool, false)
    .ATTR(bmm_score_transpose_b, Bool, false)
    .ATTR(softmax_axes, ListInt, {})
    .OP_END_FACTORY_REG(SwinAttentionScore)

    /**
    * @brief Fusion op for FFN. This op supports to compute MoeFFN(Mixture-of-Experts) or FFN.
    * @par Inputs:
    * fourteen inputs, including:
    * @li x: A matrix Tensor. The type support int8, float16, bfloat16.
    * Format support ND, FRACTAL_NZ. Shape supports at least 2 dimensions (M,K1), and at most 8 dimensions.
    * @li weight1: A matrix Tensor for weight of the first matmul. The type support int4, int8, float16, bfloat16.
    * Format support ND, FRACTAL_NZ. When having experts/having no expert, shape should be (E, K1, N1)/(K1, N1).
    * @li weight2: A matrix Tensor for weight of the second matmul. The type support int4, int8, float16, bfloat16.
    * Format support ND, FRACTAL_NZ. When having experts/having no expert, shape should be (E, K2, N2)/(K2, N2).
    * @li expert_tokens: A matrix Tensor. Indicating num of tokens in each of experts. If having experts, expert_tokens
    should be passed; if having no experts, expert_tokens should not be passed.
    * The type support int64. Format support ND. If not null, shape should be (E) and should satisfy E <= 256.
    * @li bias1: A matrix Tensor for bias of the first matmul. The type support int32, float16, float32. Format support
    ND.
    * When having experts/having no expert, shape should be (E, N1)/(N1).
    * @li bias2: A matrix Tensor for bias of the secend matmul. The type support int32, float16, float32. Format support
    ND.
    * When having experts/having no expert, shape should be (E, N2)/(N2).
    * @li scale: A matrix Tensor. Indicating scaling factor of quantization parameter.
    * The type support float32. Format support ND. In per-tensor quantization cases, when having experts/having no
    expert, shape should be (E)/(1).
    * In per-channel quantization cases, when having experts/having no expert, shape should be (E, N1)/(N1).
    * @li offset: A matrix Tensor. Indicating the offset of the quantization parameter.
    * The type support float32. Format support ND. When having experts/having no expert, shape should be (E)/(1).
    * @li deq_scale1: A matrix Tensor. Indicating scaling factor of dequantization parameter for the first matmul.
    * The type support uint64, int64, float32, bfloat16. Format support ND. When having experts/having no expert, shape
    should be (E, N1)/(N1).
    * @li deq_scale2: A matrix Tensor. Indicating scaling factor of dequantization parameter for the second matmul.
    * The type support uint64, int64, float32, bfloat16. Format support ND. When having experts/having no expert, shape
    should be (E, N2)/(N2).
    * @li antiquant_scale1: A matrix Tensor. Indicating the scaling factor of the fake-quantization parameter for the
    first matmul.
    * The type support float16, bfloat16. Format support ND. In per-channel fake-quantization cases, when having
    experts/having no expert, shape should be (E, N1)/(N1).
    * In per-in-group fake-quantization cases, when having experts/having no expert, shape should be (E, G1, N1)/(G1,
    N1).
    * @li antiquant_scale2: A matrix Tensor. Indicating the scaling factor of the fake-quantization parameter for the
    second matmul.
    * The type support float16, bfloat16. Format support ND. In per-channel fake-quantization cases, when having
    experts/having no expert, shape should be (E, N1)/(N1).
    * In per-in-group fake-quantization cases, when having experts/having no expert, shape should be (E, G2, N2)/(G2,
    N2).
    * @li antiquant_offset1: A matrix Tensor. Indicating the offset of the fake-quantization parameter for the first
    matmul.
    * The type support float16, bfloat16. Format support ND. In per-channel fake-quantization cases, when having
    experts/having no expert, shape should be (E, N2)/(N2).
    * In per-in-group fake-quantization cases, when having experts/having no expert, shape should be (E, G1, N1)/(G1,
    N1).
    * @li antiquant_offset2: A matrix Tensor. Indicating the offset of the fake-quantization parameter for the second
    matmul.
    * The type support float16, bfloat16. Format support ND. In per-channel fake-quantization cases, when having
    experts/having no expert, shape should be (E, N2)/(N2).
    * In per-in-group fake-quantization cases, when having experts/having no expert, shape should be (E, G2, N2)/(G2,
    N2).

    * @par Attributes:
    * @li activation: A string. The type of activation. Support fastgelu, gelu, relu, silu, geglu, swiglu and reglu.
    * @li inner_precise: An int. 0, fp16 high precision. 1, high performance. Default value: 0
    * @li output_dtype: An int. -1, output data type is float16. 0, output data type is float16. 1, output data type is
    bfloat16. Default -1.
    * @li tokens_index_flag: A bool. false, values in expert_tokens are values. true, values in expert_tokens are
    indices. Default value: false
    *
    * @par Outputs:
    * y: A matrix Tensor. The type support float16, bfloat16.
    * Format support ND, FRACTAL_NZ. Num of dimension should be same as x.
    *\n
    *\n
    * The following are the supported data formats and data types (for Atlas A2 Training Series Product/Atlas 800I A2
    Inference Product/A200I A2 Box Heterogeneous Component):
    *\n
    | Tensor    | x       | weight1/weight2 | bias1/bias2 | scale/offset | deq_scale1/deq_scale2 |
    antiquant_scale1/antiquant_scale2  | antiquant_offset1/antiquant_offset2 | y       | | :-------: | :-----: |
    :-------------: | :---------: | :----------: | :-------------------: | :--------------------------------: |
    :---------------------------------: | :-----: | | Format1   | ND      | ND              | ND          | ND | ND | ND
    | ND                                  | ND      | | Data Type | float16 | float16         | float16     | - | - | -
    | -                                   | float16 | |           | bfloat16| bfloat16        | float32     | - | - | -
    | -                                   | bfloat16| |           | int8    | int8            | int32       | float32 |
    uint64                | -                                  | -                                   | float16 | | |
    int8    | int8            | int32       | float32      | bfloat16              | - | - | bfloat16| |           |
    int8    | int8            | int32       | float32      | int64                 | - | - | float16 | |           |
    int8    | int8            | int32       | float32      | float32               | - | - | float16 | |           |
    float16 | int8            | float16     | -            | -                     | float16 | float16 | float16 | | |
    bfloat16| int8            | float32     | -            | -                     | bfloat16 | bfloat16 | bfloat16| |
    | float16 | int4            | float16     | -            | -                     | float16 | float16 | float16 | |
    | bfloat16| int4            | float32     | -            | -                     | bfloat16 | bfloat16 | bfloat16|
    *\n
    * The following are the supported data formats and data types (for Atlas Inference Series Product):
    *\n
    | Tensor    | x          | weight1/weight2 | bias1/bias2 | scale/offset | deq_scale1/deq_scale2 |
    antiquant_scale1/antiquant_scale2  | antiquant_offset1/antiquant_offset2 | y          | | :-------: | :--------: |
    :-------------: | :---------: | :----------: | :-------------------: | :--------------------------------: |
    :---------------------------------: | :--------: | | Format1   | FRACTAL_NZ | FRACTAL_NZ      | ND          | ND |
    ND                    | ND                                 | ND                                  | FRACTAL_NZ | |
    Data Type | float16    | float16         | float16     | -            | -                     | - | - | float16    |
    *\n
    * @attention Constraints:
    * @li Atlas Inference Series Product only support non-quantization high performance no-expert cases; x and y must
    have two dimensions; activation only supports gelu/fastgelu/relu/silu.
    * @li If expert_tokens is passed, when tokens_index_flag is true, it must be a non-negative monotone non-decreasing
    array; when tokens_index_flag is false, it must be a non-negative array.
    * @li If expert_tokens is passed, when tokens_index_flag is false, sum of expert_tokens should be equal to the first
    dim M of x; when tokens_index_flag is true, the last value in expert_tokens should be equal to the first dim M of x.
    * @li If activation is geglu/swiglu/reglu, only supporting float16 (data type of required inputs are all float16)
    high performance no-expert cases, and should satisfy N1=2*K2.
    * @li If activation is gelu/fastgelu/relu/silu, should satisfy N1=K2.
    * @li All cases should satisfy K1=N2, K1<65536, K2<65536, and M should be less than 2147483547 after aligning to 32
    byte.
    * @li Non-quantization cases should not pass quantization or fake-quantization related optional inputs, quantization
    cases should not pass fake-quantization related optional inputs, fake-quantization cases should not pass
    quantization related optional inputs.
    * @li Per-tensor quantization cases support data type template with deq_scale1/deq_scale2 float32, while per-channel
    quantization cases do not support this data type template.
    * @li If data type of weight1 and weight2 is int4, the last dimension of weight1 and weight2 must be even.
    * @li In per-in-group fake-quantization cases, group num G1 of antiquant_scale1 and antiquant_offset1 must be
    divisible by K1, group num G2 of antiquant_scale2 and antiquant_offset2 must be divisible by K2.
    * @li Attr inner_precise is only valid in non-quantization cases. In non-quantization cases, if data type of
    required inputs are all bfloat16, inner_precise only supports 0; if data type of required inputs are all float16,
    inner_precise can pass 0 or 1.
    * @li Attr output_dtype is only valid in quantization cases.
    */
    REG_OP(FFN)
    .INPUT(x, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))
    .INPUT(weight1, TensorType({DT_INT8, DT_FLOAT16, DT_BF16, DT_INT4}))
    .INPUT(weight2, TensorType({DT_INT8, DT_FLOAT16, DT_BF16, DT_INT4}))
    .OPTIONAL_INPUT(expert_tokens, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(bias1, TensorType({DT_INT32, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias2, TensorType({DT_INT32, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(deq_scale1, TensorType({DT_UINT64, DT_BF16, DT_INT64, DT_FLOAT}))
    .OPTIONAL_INPUT(deq_scale2, TensorType({DT_UINT64, DT_BF16, DT_INT64, DT_FLOAT}))
    .OPTIONAL_INPUT(antiquant_scale1, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_scale2, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_offset1, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_offset2, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(activation, String)
    .ATTR(inner_precise, Int, 0)
    .ATTR(output_dtype, Int, -1)
    .ATTR(tokens_index_flag, Bool, false)
    .OP_END_FACTORY_REG(FFN)

} // namespace ge

#endif