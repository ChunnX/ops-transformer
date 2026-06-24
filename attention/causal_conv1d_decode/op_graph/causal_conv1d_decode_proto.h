/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_CAUSAL_CONV1D_DECODE_H_
#define OPS_BUILT_IN_OP_PROTO_INC_CAUSAL_CONV1D_DECODE_H_

#include "graph/operator_reg.h"

namespace ge {

REG_OP(CausalConv1dDecode)
    .INPUT(x, TensorType({DT_BF16}))
    .INPUT(weight, TensorType({DT_BF16}))
    .INPUT(conv_states, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(query_start_loc, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(cache_indices, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(bias, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(num_accepted_tokens, TensorType({DT_INT32}))
    .ATTR(activation_mode, Int, 0)
    .ATTR(pad_slot_id, Int, -1)
    .ATTR(conv_mode, Int, 0)
    .OUTPUT(y, TensorType({DT_BF16}))
    .OP_END_FACTORY_REG(CausalConv1dDecode)

} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_CAUSAL_CONV1D_DECODE_H_
