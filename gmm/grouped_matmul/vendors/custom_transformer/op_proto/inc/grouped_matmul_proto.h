#ifndef GROUPED_MATMUL_PROTO_H_
#define GROUPED_MATMUL_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(GroupedMatmul)
    .DYNAMIC_INPUT(x, ge::TensorType::ALL())
    .DYNAMIC_INPUT(weight, ge::TensorType::ALL())
    .DYNAMIC_INPUT(bias, ge::TensorType::ALL())
    .DYNAMIC_INPUT(scale, ge::TensorType::ALL())
    .DYNAMIC_INPUT(offset, ge::TensorType::ALL())
    .DYNAMIC_INPUT(antiquant_scale, ge::TensorType::ALL())
    .DYNAMIC_INPUT(antiquant_offset, ge::TensorType::ALL())
    .OPTIONAL_INPUT(group_list, ge::TensorType::ALL())
    .OPTIONAL_INPUT(per_token_scale, ge::TensorType::ALL())
    .DYNAMIC_OUTPUT(y, ge::TensorType::ALL())
    .ATTR(split_item, Int, 0)
    .ATTR(dtype, Int, 0)
    .ATTR(transpose_weight, Bool, false)
    .ATTR(transpose_x, Bool, false)
    .ATTR(group_type, Int, -1)
    .ATTR(group_list_type, Int, 0)
    .ATTR(act_type, Int, 0)
    .ATTR(tuning_config, ListInt, {0})
    .OP_END_FACTORY_REG(GroupedMatmul);

}

#endif
