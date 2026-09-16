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
 * \file add_lora_proto.h
 * \brief
 */
#ifndef ADD_LORA_PROTO_H
#define ADD_LORA_PROTO_H

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

REG_OP(AddLora)
    .INPUT(y, TensorType({DT_FLOAT16}))
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(weightB, TensorType({DT_FLOAT16}))
    .INPUT(indices, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(weightA, TensorType({DT_FLOAT16}))
    .OUTPUT(y_out, TensorType({DT_FLOAT16}))
    .ATTR(layer_idx, Int, 0)
    .ATTR(scale, Float, 1e-3f)
    .ATTR(y_offset, Int, 0)
    .ATTR(y_slice_size, Int, -1)
    .OP_END_FACTORY_REG(AddLora)

} // namespace ge

#endif // ADD_LORA_PROTO_H
