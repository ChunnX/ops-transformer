/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_init_routing_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_MOE_INIT_ROUTING_OPS_H_
#define OPS_OP_PROTO_INC_MOE_INIT_ROUTING_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief compute init routing for moe.
* @par Inputs:
* @li x: A 2D tensor. Shape is: (B*S, H). Type is:Int8, BFloat16, Float16 or Float32. Format support ND.
* @li expert_idx: A 2D tensor. Shape is: (B*S, K). Type is:Int32. Expert index. Format support ND.
* @li scale: A 1D or 2D tensor. Shape is: (B*S) or (B*S, H). Type is:Float32. Format support ND.
* @li offset: A 2D tensor. Shape is: (expert_end - expert_start, 1) or (expert_end - expert_start, H).
               Type is:Float32. Format support ND.
* @par Outputs:
* @li expanded_x: A 2D tensor. Shape is: (B*S*K, H). Type is:Int8, BFloat16, Float16 or Float32.
                  The data type must be the same as that of x. Format support ND.
* @li expanded_row_idx: A 1D tensor. Shape is: (B*S*K). Type is:Int32. Format support ND.
* @li expert_tokens_count_or_cumsum: A 1D tensor. represents the number of tokens processed by each expert and the
                                       cumulative value. The value is controlled by expert_tokens_num_flag to output.
                                       Type is:Int64. shape is (expert_end - expert_start, ). Format support ND.
* @li expanded_scale: A 1D tensor. Shape is: (B*S*K). Type is:Float32.
                        The data type must be the same as that of scale. Format support ND.
* @par Attributes:
* @li active_num: Optional parameter. Type is:Int32. identify activate scenario. The value 0 indicates a non-active
*                 scenario, and a value greater than 0 indicates an active scenario. In the active scenario, the size
*                 of axis 0 of grad_expanded_x must be equal to the value of active_num. Default: -1.
* @li expert_capacity: Optional parameter. Type is:Int32. The max tokens count of every expert. Default: -1.
* @li expert_num: Optional parameter. Type is:Int32. Number of experts. Default: -1.
* @li drop_pad_mode: Optional parameter. Type is:Int32. The value is 0(dropless) or 1(dropPad). Default: 0.
* @li expert_tokens_num_type: Optional parameter. Type is:Int32. The value is 0(compute tokens cumsum) or
                              1(compute tokens count), which in dropPad scenario. Default: 0.
* @li expert_tokens_num_flag: Optional parameter. Type is:Bool. The value is true (compute tokens) or
                              false(do not compute tokens), which in dropPad scenario. Default: false.
* @li quant_mode: Optional parameter. Type is:Int. The value is -1(no quant) or 0(static quant) or 1(dynamic quant). Default: -1.
* @li active_expert_range: Optional parameter. Type is:ListInt. Like [expert_start, expert_end].
                           expert_start must be greater than or equal to 0, expert_end must be less than or equal to 10240,
                           expert_start must be less than expert_end. Default: [].
* @li row_idx_type: Optional parameter. Type is:Int. The value is 0(gather) or 1(scatter). Default: 0.
*/
REG_OP(MoeInitRoutingV3)
.INPUT(x, TensorType({DT_INT8, DT_FLOAT16, DT_FLOAT, DT_BF16}))
.INPUT(expert_idx, TensorType({DT_INT32}))
.OPTIONAL_INPUT(scale, TensorType({DT_FLOAT}))
.OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
.OUTPUT(expanded_x, TensorType({DT_INT8, DT_FLOAT16, DT_FLOAT, DT_BF16}))
.OUTPUT(expanded_row_idx, TensorType({DT_INT32}))
.OUTPUT(expert_tokens_count_or_cumsum, TensorType({DT_INT64}))
.OUTPUT(expanded_scale, TensorType({DT_FLOAT}))
.ATTR(active_num, Int, -1)
.ATTR(expert_capacity, Int, -1)
.ATTR(expert_num, Int, -1)
.ATTR(drop_pad_mode, Int, 0)
.ATTR(expert_tokens_num_type, Int, 0)
.ATTR(expert_tokens_num_flag, Bool, false)
.ATTR(quant_mode, Int, -1)
.ATTR(active_expert_range, ListInt, {})
.ATTR(row_idx_type, Int, 0)
.OP_END_FACTORY_REG(MoeInitRoutingV3)

} // namespace ge

#endif