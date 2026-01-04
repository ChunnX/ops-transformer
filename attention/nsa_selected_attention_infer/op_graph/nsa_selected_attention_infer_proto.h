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
 * \file nsa_selected_attention_infer_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_NSA_SELECTED_ATTENTION_INFER_H_
#define OPS_OP_PROTO_INC_NSA_SELECTED_ATTENTION_INFER_H_

#include "graph/operator_reg.h"

namespace ge {

REG_OP(NsaSelectedAttentionInfer)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(topk_indices, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_BOOL, DT_UINT8}))
    .OPTIONAL_INPUT(block_table, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(actual_q_seq_lengths, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_kv_seq_lengths, TensorType({DT_INT64}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(NsaSelectedAttentionInfer)

} // namespace ge

#endif // OPS_OP_PROTO_INC_NSA_SELECTED_ATTENTION_INFER_H_