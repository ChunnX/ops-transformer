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
 * \file urma_p2p_benchmark_proto.h
 * \brief URMA点对点通信基准测试算子的proto定义
 */
#ifndef URMA_P2P_BENCHMARK_PROTO_H_
#define URMA_P2P_BENCHMARK_PROTO_H_

#include "graph/operator_reg.h"
namespace ge {

/**
* @brief UrmaP2pBenchmark operator for testing URMA point-to-point communication performance on NPU.

* @par Inputs
* Two inputs, including:
* @li x: A tensor. Support dtype: float16. Shape supports (BS, H), support format: ND.
* @li dst_rank_id: A tensor. Support dtype: int32, indicates target rank for each token. Shape supports (BS, ),
        support format: ND.

* @par Attributes
* @li group: Required. Communication group name, dtype: String.
* @li world_size: Required. Number of ranks in the group, dtype: Int.
* @li max_bs: Required. Maximum batch size per rank, dtype: Int.
* @li mode: Required. Test mode: 0=overhead, 1=latency, 2=bandwidth, dtype: Int.
* @li ccl_buffer_size: Required. HCCL buffer size, dtype: Int.
* @li dst_rank_id: Optional. Point-to-point target rank ID, -1 means all ranks, dtype: Int.

* @par Outputs
* Two outputs, including:
* @li y: A tensor. Received data. Support dtype: float16. Shape supports (max_recv_cnt, H),
        support format: ND.
* @li receive_cnt: A tensor. Received token count from each rank. Support dtype: int32, support format: ND.
*/
REG_OP(UrmaP2pBenchmark)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(dst_rank_id, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .OUTPUT(receive_cnt, TensorType({DT_INT32}))
    .REQUIRED_ATTR(group, String)
    .REQUIRED_ATTR(world_size, Int)
    .REQUIRED_ATTR(max_bs, Int)
    .REQUIRED_ATTR(mode, Int)
    .REQUIRED_ATTR(ccl_buffer_size, Int)
    .ATTR(dst_rank_id, Int, -1)
    .ATTR(aiv_num, Int, -1)
    .OP_END_FACTORY_REG(UrmaP2pBenchmark)

}  // namespace ge
#endif  // URMA_P2P_BENCHMARK_PROTO_H_