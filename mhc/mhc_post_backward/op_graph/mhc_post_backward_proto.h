

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
 * \file mhc_post_backward_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_MHCPOSTBACKWARD_H_
#define OPS_OP_PROTO_INC_MHCPOSTBACKWARD_H_

#include "graph/operator_reg.h"

namespace ge {
REG_OP(MhcPostBackward)
    .INPUT(grad_y, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(h_res, TensorType({DT_FLOAT}))
    .INPUT(h_out, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(h_post, TensorType({DT_FLOAT}))
    .OUTPUT(grad_x, TensorType({DT_BF16, DT_FLOAT16}))
    .OUTPUT(grad_h_res, TensorType({DT_FLOAT}))
    .OUTPUT(grad_h_out, TensorType({DT_BF16, DT_FLOAT16}))
    .OUTPUT(grad_h_post, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(MhcPostBackward)

} // namespace ge

#endif // OPS_OP_PROTO_INC_MHCPOSTBackward_H_
