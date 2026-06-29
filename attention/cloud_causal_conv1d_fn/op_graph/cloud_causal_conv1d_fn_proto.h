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
 * \file cloud_causal_conv1d_fn_proto.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_CLOUD_CAUSAL_CONV1D_FN_H_
#define OPS_BUILT_IN_OP_PROTO_INC_CLOUD_CAUSAL_CONV1D_FN_H_

#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief Function CloudCausalConv1dFn.
 *
 * @par Inputs:
 * @li x: Input sequence tensor. shape [cu_seq_len, dim] or [batch, seqlen, dim]. Supports float16, bfloat16.
 * @li weight: Convolution kernel of shape [K, dim], K fixed to 4. Same type as x.
 * @li bias: Optional. Convolution bias of shape [dim]. Same type as x.
 * @li conv_states: Cache state tensor storing historical tokens per sequence, updated in-place. Same type as x.
 * @li query_start_loc: Optional. Start offset of each sequence in x. shape [batch+1]. int32.
 * @li cache_indices: Optional. Index mapping each sequence to its cache slot. 1D [batch]. int32.
 * @li num_accepted_tokens: Optional. Number of accepted tokens per sequence in speculative decoding. shape [batch].
 * int32.
 *
 * @par Attributes:
 * @li activation_mode: A optional int. Activation function type: 0 (None), 1 (silu), 2 (swish).
 * @li pad_slot_id: A optional int. Slot ID used to skip padding batches.
 *
 * @par Outputs:
 * @li y: Output sequence tensor. Same shape and type as x.
 */
REG_OP(CloudCausalConv1dFn)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(weight, TensorType({DT_BF16, DT_FLOAT16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(conv_states, TensorType({DT_BF16, DT_FLOAT16}))
    .OPTIONAL_INPUT(query_start_loc, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(cache_indices, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(num_accepted_tokens, TensorType({DT_INT32}))
    .ATTR(activation_mode, Int, 0)
    .ATTR(pad_slot_id, Int, -1)
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16}))
    .OP_END_FACTORY_REG(CloudCausalConv1dFn)

} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_CLOUD_CAUSAL_CONV1D_FN_H_
