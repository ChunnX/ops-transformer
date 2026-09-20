/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_PROTO_INC_GenericLightningIndexer_H_
#define OPS_OP_PROTO_INC_GenericLightningIndexer_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
 * @brief GenericLightningIndexer unifies indexer score + TopK.
 *        Matmul(Q, K^T) → act → head reduce → mask → TopK → optional Expand/AppendTail.
 *        Q/K preprocess is out of this op; compressed keys are consumed via cmp_ratio.
 *
 * @par Inputs:
 * @li query / key / weights / actual_seq_q / actual_seq_k / seqused_q / seqused_k /
 *     cmp_residual_k / block_table / q_descale / k_descale / output_idx_offset /
 *     metadata / start_loc.
 *
 * @par Attributes: topk, cmp_ratio, layout_q, layout_k, sparse_mode, act_type,
 *        has_weighted_sum, expand_indices, append_tail, quant_mode, return_value,
 *        select_mode, key_reduce_mode, score_scale, key_stride0, k_descale_stride0,
 *        max_seqlen_q, init_blocks, local_blocks.
 *
 * @par Outputs:
 * @li sparse_indices: int32 selected indices. Invalid slots are -1.
 * @li sparse_values: float scores when return_value=true, else shape (0,).
 */
REG_OP(GenericLightningIndexer)
    .INPUT(query,
           TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E4M3FN, DT_INT8, DT_HIFLOAT8, DT_FLOAT4_E2M1, DT_FLOAT8_E5M2}))
    .INPUT(key,
           TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E4M3FN, DT_INT8, DT_HIFLOAT8, DT_FLOAT4_E2M1, DT_FLOAT8_E5M2}))
    .OPTIONAL_INPUT(weights, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(actual_seq_q, TensorType({DT_INT32, DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_k, TensorType({DT_INT32, DT_INT64}))
    .OPTIONAL_INPUT(seqused_q, TensorType({DT_INT32, DT_INT64}))
    .OPTIONAL_INPUT(seqused_k, TensorType({DT_INT32, DT_INT64}))
    .OPTIONAL_INPUT(cmp_residual_k, TensorType({DT_INT32, DT_INT64}))
    .OPTIONAL_INPUT(block_table, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(q_descale, TensorType({DT_FLOAT, DT_FLOAT16, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(k_descale, TensorType({DT_FLOAT, DT_FLOAT16, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(output_idx_offset, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(metadata, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(start_loc, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(sparse_indices, TensorType({DT_INT32}))
    .OUTPUT(sparse_values, TensorType({DT_FLOAT}))
    .ATTR(topk, Int, 2048)
    .ATTR(cmp_ratio, Int, 1)
    .ATTR(layout_q, String, "BSND")
    .ATTR(layout_k, String, "BSND")
    .ATTR(sparse_mode, Int, 0)
    .ATTR(act_type, Int, 0)
    .ATTR(has_weighted_sum, Bool, true)
    .ATTR(expand_indices, Bool, false)
    .ATTR(append_tail, Bool, false)
    .ATTR(quant_mode, Int, -1)
    .ATTR(return_value, Bool, false)
    .ATTR(select_mode, Int, 0)
    .ATTR(key_reduce_mode, Int, 0)
    .ATTR(score_scale, Float, 1.0)
    .ATTR(key_stride0, Int, -1)
    .ATTR(k_descale_stride0, Int, -1)
    .ATTR(max_seqlen_q, Int, -1)
    .ATTR(init_blocks, Int, 0)
    .ATTR(local_blocks, Int, 0)
    .OP_END_FACTORY_REG(GenericLightningIndexer)
} // namespace ge

#endif
