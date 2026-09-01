#ifndef ALL_GATHER_MATMUL_V2_PROTO_H_
#define ALL_GATHER_MATMUL_V2_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(AllGatherMatmulV2)
    .INPUT(x1, ge::TensorType::ALL())
    .INPUT(x2, ge::TensorType::ALL())
    .OPTIONAL_INPUT(bias, ge::TensorType::ALL())
    .OPTIONAL_INPUT(x1_scale, ge::TensorType::ALL())
    .OPTIONAL_INPUT(x2_scale, ge::TensorType::ALL())
    .OPTIONAL_INPUT(quant_scale, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OUTPUT(gather_out, ge::TensorType::ALL())
    .OUTPUT(amax_out, ge::TensorType::ALL())
    .REQUIRED_ATTR(group, String)
    .ATTR(is_trans_a, Bool, false)
    .ATTR(is_trans_b, Bool, false)
    .ATTR(gather_index, Int, 0)
    .ATTR(comm_turn, Int, 0)
    .ATTR(rank_size, Int, 0)
    .ATTR(block_size, Int, 0)
    .ATTR(group_size, Int, 0)
    .ATTR(is_gather_out, Bool, true)
    .ATTR(is_amax_out, Bool, false)
    .ATTR(y_dtype, Int, 28)
    .REQUIRED_ATTR(comm_mode, String)
    .OP_END_FACTORY_REG(AllGatherMatmulV2);

}

#endif
