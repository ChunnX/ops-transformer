/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file BatchToSpace_proto.h
 * \brief BatchToSpace operator definition
 */
#ifndef OPS_OP_PROTO_INC_BATCHTOSPACE_H_
#define OPS_OP_PROTO_INC_BATCHTOSPACE_H_

#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief BatchToSpace operation for NHWC format tensors.
 * @par Inputs:
 * @li x: A 4D Tensor with NHWC format. Type is BasicType.
 *        Shape is [batch, height, width, depth].
 * @li crops: A 2D Tensor with shape [2, 2] of type int32 or int64.
 *            crops = [[crop_top, crop_bottom], [crop_left, crop_right]]
 * @par Attributes:
 * @li block_size: An int attribute. Must be >= 1.
 *                 The size of the spatial block.
 * @par Outputs:
 * @li y: A 4D Tensor with NHWC format. Type is same as x.
 *        Shape is [batch / (block_size * block_size), height * block_size - crop_top - crop_bottom,
 *                  width * block_size - crop_left - crop_right, depth]
 * @par Constraints:
 * @li Only support NHWC format
 * @li batch must be divisible by block_size * block_size
 * @li crop_top + crop_bottom must be <= block_size * height
 * @li crop_left + crop_right must be <= block_size * width
 */
REG_OP(BatchToSpace)
    .INPUT(x, TensorType::BasicType())
    .INPUT(crops, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(block_size, Int)
    .OP_END_FACTORY_REG(BatchToSpace)

} // namespace ge

#endif // OPS_OP_PROTO_INC_BATCHTOSPACE_H_