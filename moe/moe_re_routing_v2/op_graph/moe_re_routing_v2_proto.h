/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_PROTO_INC_MOE_RE_ROUTING_V2_OPS_H_
#define OPS_OP_PROTO_INC_MOE_RE_ROUTING_V2_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
REG_OP(MoeReRoutingV2)
    .INPUT(tokens, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8,
                               DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))
    .INPUT(expert_token_num_per_rank, TensorType({DT_INT32, DT_INT64}))
    .OPTIONAL_INPUT(per_token_scales, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(expert_topk_weight, TensorType({DT_FLOAT}))
    .OUTPUT(permute_tokens, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8,
                                        DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))
    .OUTPUT(permute_per_token_scales, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OUTPUT(permute_token_idx, TensorType({DT_INT32}))
    .OUTPUT(expert_token_num, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(permute_topk_weight, TensorType({DT_FLOAT}))
    .ATTR(expert_token_num_type, Int, 1)
    .ATTR(idx_type, Int, 0)
    .OP_END_FACTORY_REG(MoeReRoutingV2)

} // namespace ge

#endif
