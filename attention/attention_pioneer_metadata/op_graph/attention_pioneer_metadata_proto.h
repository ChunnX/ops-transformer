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
 * \file attention_pioneer_metadata_proto.h
 * \brief Op prototype registration for AttentionPioneerMetadata AICPU operator.
 */
#ifndef ATTENTION_PIONEER_METADATA_PROTO_H
#define ATTENTION_PIONEER_METADATA_PROTO_H

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

REG_OP(AttentionPioneerMetadata)
    .OPTIONAL_INPUT(actual_seq_lengths_q, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_lengths_kv, TensorType({DT_INT64}))
    .OUTPUT(metaData_out, TensorType({DT_INT32}))
    .REQUIRED_ATTR(num_heads_q, Int)
    .REQUIRED_ATTR(num_heads_kv, Int)
    .REQUIRED_ATTR(head_dim_qk, Int)
    .REQUIRED_ATTR(head_dim_v, Int)
    .ATTR(batch_size, Int, 0)
    .ATTR(sparse_mode, Int, 0)
    .ATTR(pre_tokens, Int, 2147483647)
    .ATTR(next_tokens, Int, 2147483647)
    .ATTR(input_layout, String, "TND")
    .ATTR(k_sink_num, Int, 0)
    .ATTR(rope_head_dim, Int, 0)
    .ATTR(block_size, Int, 0)
    .REQUIRED_ATTR(soc_version, String)
    .REQUIRED_ATTR(aic_core_num, Int)
    .REQUIRED_ATTR(aiv_core_num, Int)
    .OP_END_FACTORY_REG(AttentionPioneerMetadata)

} // namespace ge

#endif // ATTENTION_PIONEER_METADATA_PROTO_H
