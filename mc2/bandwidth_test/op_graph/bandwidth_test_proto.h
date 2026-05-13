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
 * \file bandwidth_test_proto.h
 * \brief Bandwidth test operator for measuring communication bandwidth
 */
#ifndef BANDWIDTH_TEST_PROTO_H_
#define BANDWIDTH_TEST_PROTO_H_

#include "graph/operator_reg.h"
namespace ge {

/**
* @brief BandwidthTest operator for testing NPU communication bandwidth.

* @par Inputs
* Three inputs, including:
* @li context: A tensor. Support dtype: int32. Shape supports (world_size), support format: ND.
* @li x: A tensor. Support dtype: float16. Shape supports (BS, H), support format: ND.
* @li dstrank_id: A tensor. Support dtype: int32, indicates target rank for each token. Shape supports (BS, ), support format: ND.

* @par Attributes
* @li group: Required. Communication group name, dtype: String.
* @li world_size: Required. Number of ranks in the group, dtype: Int.
* @li ccl_buffer_size: Required. CCL buffer size, dtype: Int.
* @li max_bs: Optional. Maximum batch size per rank, dtype: Int.
* @li mode: Optional. mode, dtype: Int.
* @li comm_alg: Optional. Communication type, dtype: Int.

* @par Outputs
* Two outputs, including:
* @li y: A tensor. Received data. Support dtype: float16. Shape supports (max_recv_cnt, H), support format: ND.
* @li receive_cnt: A tensor. Received token count from each rank. Support dtype: int32, support format: ND.
*/
REG_OP(BandwidthTest)
    .INPUT(context, TensorType({DT_INT32}))
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(dstrank_id, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .OUTPUT(receive_cnt, TensorType({DT_INT32}))
    .REQUIRED_ATTR(group, String)
    .REQUIRED_ATTR(world_size, Int)
    .REQUIRED_ATTR(ccl_buffer_size, Int)
    .ATTR(max_bs, Int, 0)
    .ATTR(mode, Int, 0)
    .ATTR(comm_alg, Int, 0)
    .OP_END_FACTORY_REG(BandwidthTest)

}  // namespace ge
#endif  // BANDWIDTH_TEST_PROTO_H_