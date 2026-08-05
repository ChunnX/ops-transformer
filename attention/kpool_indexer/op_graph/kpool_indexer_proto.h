/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kpool_indexer_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_KpoolIndexer_H_
#define OPS_OP_PROTO_INC_KpoolIndexer_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief KpoolIndexer packs consecutive tokens into pools, computes attention
 *        relevance scores per pool, selects top-k pools and expands them into
 *        sparse token indices, so as to reduce indexing overhead while keeping
 *        the benefit of sparse attention. \n

 * @par Inputs:
 * Inputs including:
 * @li query: The Query tensor used for pool score computation. \n
 *   - Data types: float16, bfloat16, float8_e4m3fn.
 *   - format: ND
 * @li pool_key: The pooled Key tensor used for pool score computation. \n
 *   - Data types: float16, bfloat16, float8_e4m3fn.
 *   - format: ND
 * @li weights: The multi-head weight projection matrix for score aggregation. \n
 *   - Data types: float16, bfloat16.
 *   - format: ND
 * @li pool_tail_k: The number of valid tokens in the tail (incomplete) pool of each batch. \n
 *   - Data types: int32.
 *   - format: ND
 * @li actual_seq_q: Optional. The valid token count of Query per batch. \n
 *   - Data types: int32.
 *   - format: ND
 * @li actual_seq_k: Optional. The valid token count of Key per batch. \n
 *   - Data types: int32.
 *   - format: ND
 * @li block_table: Optional. The block mapping table used in PageAttention KV storage. \n
 *   - Data types: int32.
 *   - format: ND
 * @li q_descale: Optional. The dequantization scale of Query. \n
 *   - Data types: float32, float8_e8m0.
 *   - format: ND
 * @li k_descale: Optional. The dequantization scale of Key. \n
 *   - Data types: float32, float8_e8m0.
 *   - format: ND

 * @par Attributes:
 * @li topk: A required int attribute. The number of tokens to keep after pool
 *   expansion. k_pool = floor(topk / pool_size) pools are selected.
 * @li pool_size: A required int attribute. The number of tokens packed into one pool.
 * @li layout_q: An optional string attribute. The layout of Query, "BSND" or "TND".
 *   Defaults to "BSND".
 * @li layout_k: An optional string attribute. The layout of Key, "BSND", "TND" or "PA_BSND".
 *   Defaults to "BSND".
 * @li mask_mode: An optional int attribute. The mask mode.
 *   - 0: defaultMask
 *   - 3: rightDownCausal
 *   Defaults to 3.
 * @li quant_mode: An optional int attribute. The quantization mode of Query/Key.
 *   - -1: no quantization (default)
 *   - 0: FP8 per-token-head quantization (scale stored as float32)
 *   - 1: MX-FP8 quantization (scale stored as float8_e8m0)
 * @li return_value: An optional int attribute. Whether to output sparse_values.
 *   - 0: do not output (default)
 *   - 1: output sparse_values

 * @par Outputs:
 * @li sparse_indices: The expanded sparse token indices.
 *   - Data types: int32.
 * @li sparse_values: The pool scores of the selected top-k pools.
 *   - Data types: float32.

 * @attention Constraints:
 * @code{.c}
 *  - topk must be divisible by pool_size (topk % pool_size == 0).
 *  - pool_size supports [1, 128]; topk supports [1, 2048] and 3072/4096/5120/6144/7168/8192.
 *  - head_dim supports 128; query head num (N1) supports [1, 64]; pool_key head num (N2) supports 1.
 *  - block_size must be a multiple of 16 and no greater than 1024.
 *  - Padding invalid data on S1/S2 of query/pool_key is not supported.
 *  - pool_key supports non-contiguous on axis 0.
 *  - Ascend950PR/Ascend950DT, Atlas A3 training/inference products.
 *  - FP8 (float8_e4m3fn) and MX-FP8 (float8_e8m0 scale) data types are only supported
 *    on Ascend950PR/Ascend950DT and Atlas A3 training/inference products.
 * @endcode
 */
REG_OP(KpoolIndexer)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E4M3FN}))
    .INPUT(pool_key, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E4M3FN}))
    .INPUT(weights, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(pool_tail_k, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(actual_seq_q, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(actual_seq_k, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(block_table, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(q_descale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(k_descale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OUTPUT(sparse_indices, TensorType({DT_INT32}))
    .OUTPUT(sparse_values, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(topk, Int)
    .REQUIRED_ATTR(pool_size, Int)
    .ATTR(layout_q, String, "BSND")
    .ATTR(layout_k, String, "BSND")
    .ATTR(mask_mode, Int, 3)
    .ATTR(quant_mode, Int, -1)
    .ATTR(return_value, Int, 0)
.OP_END_FACTORY_REG(KpoolIndexer)
} // namespace ge

#endif // OPS_OP_PROTO_INC_KpoolIndexer_H_
