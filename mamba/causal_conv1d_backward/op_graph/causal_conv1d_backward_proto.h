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
 * \file causal_conv1d_backward_proto.h
 * \brief CausalConv1dBackward 图模式 GE IR 定义
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_CAUSAL_CONV1D_BACKWARD_H_
#define OPS_BUILT_IN_OP_PROTO_INC_CAUSAL_CONV1D_BACKWARD_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Gradients of the causal 1D depthwise convolution (CausalConv1d, prefill
* mode, bias=None, cold start) w.r.t. its input x and its kernel weight. \n
*
* Forward (prefill, cold start, no bias): \n
*   pre[t, d] = sum_{k=0}^{W-1} weight[k, d] * x[t-(W-1-k), d] * m(t, t-(W-1-k)) \n
*   y[t, d]   = Act(pre[t, d]) \n
* m(t, u) is 1 when token t and token u lie in the same sequence and u >= 0. Sequences
* are the [query_start_loc[i], query_start_loc[i+1]) ranges of the flattened [batch*seqlen]
* token axis; when query_start_loc is None the whole axis is one sequence. \n
* weight[k, d] multiplies x[t-(W-1-k), d], i.e. weight[W-1] pairs with the current token
* x[t] and weight[0] with the oldest one x[t-(W-1)]. This is the same layout and the same
* weight convention as CausalConv1d (vllm's causal_conv1d_fn agrees).
*
* Backward: \n
*   dout[t, d]        = grad_y[t, d] * Act'(pre[t, d]) \n
*   grad_x[s, d]      = sum_{j=0}^{W-1} weight[W-1-j, d] * m(s+j, s) * dout[s+j, d] \n
*   grad_weight[k, d] = sum_{t} m(t, t-(W-1-k)) * dout[t, d] * x[t-(W-1-k), d] \n
* Act'(pre) and all accumulation are evaluated in FP32; the results are cast to the
* input dtype once. grad_weight accumulates over up to batch*seqlen terms, so it must
* not be accumulated in fp16/bfloat16.
*
* @par Inputs:
* @li grad_y: Gradient of the convolution output. 2D varlen shape [cu_seq_len, D] or 3D
* batch shape [batch, seqlen, D]. Same shape as x. Supports float16, bfloat16.
* @li x: Convolution input. Same shape and dtype as grad_y. Required by grad_weight
* (and to recompute pre, hence Act'(pre)).
* @li weight: Convolution kernel, shape [W, D], W in {2, 3, 4} ([4, 12288] at the
* reference training config). Same dtype as grad_y. May be non-contiguous (a common
* case: the transpose(0, 1) view of a [C, 1, W] parameter).
* @li query_start_loc: Optional. Cumulative token offsets of the sequences on the
* flattened [batch*seqlen] axis. shape [num_seq+1], int32, non-decreasing,
* query_start_loc[0] = 0, query_start_loc[-1] = batch*seqlen. Required when grad_y is 2D.
* When None the whole token axis is treated as one sequence (no document boundary).
*
* @par Attributes:
* @li activation_mode: An optional string. Activation that was applied in the forward:
* "silu" or "none". Default: "silu".
*
* @par Outputs:
* @li grad_x: Gradient w.r.t. x. Same shape and dtype as x.
* @li grad_weight: Gradient w.r.t. weight. shape [W, D], same dtype as weight.
*/
REG_OP(CausalConv1dBackward)
    .INPUT(grad_y, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(weight, TensorType({DT_BF16, DT_FLOAT16}))
    .OPTIONAL_INPUT(query_start_loc, TensorType({DT_INT32}))
    .ATTR(activation_mode, String, "silu")
    .OUTPUT(grad_x, TensorType({DT_BF16, DT_FLOAT16}))
    .OUTPUT(grad_weight, TensorType({DT_BF16, DT_FLOAT16}))
    .OP_END_FACTORY_REG(CausalConv1dBackward)

} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_CAUSAL_CONV1D_BACKWARD_H_
