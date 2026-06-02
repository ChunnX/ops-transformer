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
 * \file dispatch_layout_proto.h
 * \brief GE proto for DispatchLayout (MoE EP layout).
 *
 * Outputs align with op_host/dispatch_layout_infershape.cpp:
 *   numTokensPerExpert: [round * num_experts]
 *   isTokenInRank: [round * per_round_tokens, num_ranks], DT_INT32 (0/1)
 *   ext_handle_data[0] (notifySendData): [round * num_local_experts], kernel scratch
 *   ext_handle_data[1] (sendTokenIdxSmall): [num_tokens * num_topk]
 */
#ifndef DISPATCH_LAYOUT_PROTO_H_
#define DISPATCH_LAYOUT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

REG_OP(DispatchLayout)
    .INPUT(topkIdx, TensorType({DT_INT64}))
    .OUTPUT(numTokensPerRank, TensorType({DT_INT32}))
    .OUTPUT(numTokensPerExpert, TensorType({DT_INT32}))
    .OUTPUT(isTokenInRank, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(ext_handle_data, TensorType({DT_INT32}))
    .REQUIRED_ATTR(num_tokens, Int)
    .REQUIRED_ATTR(num_ranks, Int)
    .REQUIRED_ATTR(num_experts, Int)
    .REQUIRED_ATTR(num_topk, Int)
    .REQUIRED_ATTR(local_ranksize, Int)
    .REQUIRED_ATTR(per_round_tokens, Int)
    .REQUIRED_ATTR(rank_id, Int)
    .OP_END_FACTORY_REG(DispatchLayout)

} // namespace ge

#endif // DISPATCH_LAYOUT_PROTO_H_
