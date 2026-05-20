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
 * \file combine_normal_proto.h
 * \brief GE proto registration for CombineNormal.
 */
#ifndef COMBINE_NORMAL_PROTO_H_
#define COMBINE_NORMAL_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

REG_OP(CombineNormal)
    .INPUT(context, TensorType({DT_INT32}))
    .INPUT(recv_x, TensorType({DT_BF16, DT_FLOAT16, DT_INT32}))
    .INPUT(token_src_info, TensorType({DT_INT32}))
    .INPUT(ep_recv_counts, TensorType({DT_INT32}))
    .INPUT(recv_topk_weights, TensorType({DT_FLOAT}))
    .INPUT(token_idx, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(tp_recv_counts, TensorType({DT_INT32}))
    .OUTPUT(x, TensorType({DT_BF16, DT_FLOAT16}))
    .OUTPUT(combine_send_cost_stats, TensorType({DT_INT32}))
    .REQUIRED_ATTR(ep_world_size, Int)
    .REQUIRED_ATTR(ep_rank_id, Int)
    .REQUIRED_ATTR(moe_expert_num, Int)
    .REQUIRED_ATTR(ccl_buffer_size, Int)
    .ATTR(tp_world_size, Int, 0)
    .ATTR(tp_rank_id, Int, 0)
    .ATTR(real_max_bs, Int, 0)
    .ATTR(round, Int, 4)
    .ATTR(per_round_tokens, Int, 1024)
    .OP_END_FACTORY_REG(CombineNormal)

} // namespace ge

#endif // COMBINE_NORMAL_PROTO_H_
