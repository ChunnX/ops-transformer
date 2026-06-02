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
 * \file notify_dispatch_proto.h
 * \brief GE proto registration for NotifyDispatch.
 */
#ifndef NOTIFY_DISPATCH_PROTO_H_
#define NOTIFY_DISPATCH_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

REG_OP(NotifyDispatch)
    .INPUT(context, TensorType({DT_INT32}))
    .INPUT(sendData, TensorType({DT_INT32}))
    .INPUT(tokenPerExpertData, TensorType({DT_INT32}))
    .OUTPUT(sendDataOffset, TensorType({DT_INT32}))
    .OUTPUT(recvData, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(recv_offset, TensorType({DT_INT32}))
    .DYNAMIC_OUTPUT(expert_output_offset, TensorType({DT_INT32}))
    .OUTPUT(totalRecvTokens, TensorType({DT_INT32}))
    .OUTPUT(maxBs, TensorType({DT_INT32}))
    .OUTPUT(recvTokensPerExpert, TensorType({DT_INT32}))
    .OUTPUT(ext_handle_data, TensorType({DT_INT32}))
    .REQUIRED_ATTR(sendCount, Int)
    .REQUIRED_ATTR(num_tokens, Int)
    .REQUIRED_ATTR(ep_world_size, Int)
    .REQUIRED_ATTR(ep_rank_id, Int)
    .REQUIRED_ATTR(local_rank_size, Int)
    .REQUIRED_ATTR(local_rank_id, Int)
    .REQUIRED_ATTR(ccl_buffer_size, Int)
    .REQUIRED_ATTR(round, Int)
    .REQUIRED_ATTR(perRoundTokens, Int)
    .OP_END_FACTORY_REG(NotifyDispatch)

} // namespace ge

#endif // NOTIFY_DISPATCH_PROTO_H_
