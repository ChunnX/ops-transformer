/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_attention_grad_proto.h
 * \brief
 */
#ifndef SPARSE_FLASH_ATTENTION_GRAD_PROTO_H_
#define SPARSE_FLASH_ATTENTION_GRAD_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Backwards calculation of SparseFlashAttention.
*
* @par Inputs:
* @li query: A matrix tensor. The type support float16, bfloat16.
* Query for attention structure.
* @li key: A matrix tensor. The type support float16, bfloat16.
* Key for attention structure.
* @li sparse_indices: A matrix tensor. The type support int32.
* The indices of sparse kv cache.
* @li d_out: A matrix tensor. The type support float16, bfloat16.
* Gradient of attention_out.
* @li out: A matrix tensor. The type support float16, bfloat16.
* Attention output of the forward computation.
* @li softmax_max: A matrix tensor. The type support float32.
* Max intermediate result of the forward computation.
* @li softmax_sum: A matrix tensor. The type support float32.
* Sum intermediate result of the forward computation.
* @li sinks: A tensor. The type support float.
* Learnable sinks of the attention structure.
* @li value: A matrix tensor. The type support float16, bfloat16.
* Value for attention structure.
* @li actual_seq_lengths_query: A matrix tensor. The type support int32.
* Efective sequence length of query in different batches.
* @li actual_seq_lengths_kv: A matrix tensor. The type support int32.
* Effective sequence length of key and value in different batches.
* @li query_rope: A tensor. The type support float16, bfloat16.
* @li key_rope: A tensor. The type support float16, bfloat16.
*
* @par Attributes:
* @li scale_value: A float. A required attribute.
* @li sparse_block_size: An int. An optional attribute. Max value: 64. Default: 1.
* @li layout: A string. An optional attribute. Specifies the layout of inputs, the value must be one of
* ["BSND", "TND"]. Default: "BSND".
* @li sparse_mode: An int. Sparse mode. Default: 3.
* - 0: default mask
* - 3: rightDownCausal make
* - 4: band mask
* @li pre_tokens: An int. Previous tokens. Default: 2147483647.
* @li next_tokens: An int. Next tokens. Default: 2147483647.
* @li deterministic: A bool. An optional attribute. Default: false.
*
* @par Outputs:
* @li d_query: A matrix tensor. The type support float16, bfloat16. Gradient of query.
* @li d_key: A matrix tensor. The type support float16, bfloat16. Gradient of key.
* @li d_value: A matrix tensor. The type support float16, bfloat16. Gradient of value.
* @li d_query_rope: A matrix tensor. The type support float16, bfloat16. Gradient of query_rope.
* @li d_key_rope: A matrix tensor. The type support float16, bfloat16. Gradient of key_rope.
* @li d_sinks: A matrix tensor. The type support float32. Gradient of sinks.
*/
REG_OP(SparseFlashAttentionGrad)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(sparse_indices, TensorType({DT_INT32}))
    .INPUT(d_out, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(out, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(softmax_max, TensorType({DT_FLOAT32}))
    .INPUT(softmax_sum, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(sinks, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(value, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(actual_seq_lengths_query, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(actual_seq_lengths_kv, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(query_rope, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(key_rope, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(d_query, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(d_key, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(d_value, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(d_query_rope, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(d_key_rope, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(d_sinks, TensorType({DT_FLOAT32}))
    .REQUIRED_ATTR(scale_value, Float)
    .ATTR(sparse_block_size, Int, 1)
    .ATTR(layout, String, "BSND")
    .ATTR(sparse_mode, Int, 3)
    .ATTR(pre_tokens, Int, 2147483647)
    .ATTR(next_tokens, Int, 2147483647)
    .ATTR(deterministic, Bool, false)
    .OP_END_FACTORY_REG(SparseFlashAttentionGrad)

} // namespace ge
#endif // SPARSE_FLASH_ATTENTION_GRAD_PROTO_H_
