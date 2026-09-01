#ifndef ALL_GATHER_MATMUL_V3_PROTO_H_
#define ALL_GATHER_MATMUL_V3_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(AllGatherMatmulV3)
    .INPUT(context, ge::TensorType::ALL())
    .INPUT(x1, ge::TensorType::ALL())
    .INPUT(x2, ge::TensorType::ALL())
    .OPTIONAL_INPUT(bias, ge::TensorType::ALL())
    .OPTIONAL_INPUT(x1_scale, ge::TensorType::ALL())
    .OPTIONAL_INPUT(x2_scale, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OUTPUT(gather_out, ge::TensorType::ALL())
    .OUTPUT(amax_out, ge::TensorType::ALL())
    .REQUIRED_ATTR(group, String)
    .REQUIRED_ATTR(hccl_buffer_size, Int)
    .ATTR(is_trans_a, Bool, false)
    .ATTR(is_trans_b, Bool, false)
    .ATTR(rank_size, Int, 0)
    .ATTR(group_size, Int, 0)
    .ATTR(y_dtype, Int, 28)
    .ATTR(comm_mode, String, "ai_cpu")
    .OP_END_FACTORY_REG(AllGatherMatmulV3);

}

#endif
