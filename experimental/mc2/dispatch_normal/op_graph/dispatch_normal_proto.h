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
 * \file dispatch_normal_proto.h
 * \brief GE proto registration for DispatchNormal.
 */
#ifndef DISPATCH_NORMAL_PROTO_H_
#define DISPATCH_NORMAL_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

REG_OP(DispatchNormal)
    .INPUT(context, TensorType({DT_INT32}))
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(topk_idx, TensorType({DT_INT32}))
    .INPUT(send_offset, TensorType({DT_INT32}))
    .INPUT(send_tokenIdx, TensorType({DT_INT32}))
    .DYNAMIC_INPUT(recv_offset, TensorType({DT_INT32}))
    .DYNAMIC_INPUT(expert_output_offset, TensorType({DT_INT32}))
    .OUTPUT(recv_x, TensorType({DT_BF16, DT_INT8, DT_FLOAT16}))
    .OUTPUT(x_scales, TensorType({DT_FLOAT}))
    .OUTPUT(assist_info_for_combine, TensorType({DT_INT32}))
    .OUTPUT(dispatch_wait_recv_cost_stats, TensorType({DT_INT32}))
    .OUTPUT(ext_handle_data, TensorType({DT_INT32}))
    .REQUIRED_ATTR(ep_world_size, Int)
    .REQUIRED_ATTR(ep_rank_id, Int)
    .REQUIRED_ATTR(moe_expert_num, Int)
    .REQUIRED_ATTR(ccl_buffer_size, Int)
    .ATTR(quant_mode, Int, 0)
    .ATTR(real_max_bs, Int, 0)
    .ATTR(global_bs, Int, 0)
    .ATTR(round, Int, 4)
    .ATTR(per_round_tokens, Int, 1024)
    .OP_END_FACTORY_REG(DispatchNormal)

} // namespace ge

#endif // DISPATCH_NORMAL_PROTO_H_
