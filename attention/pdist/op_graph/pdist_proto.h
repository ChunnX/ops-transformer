/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.cann.huawei.com/CANN_OPEN_SOURCE_LICENSE_ACK
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is
 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */

/*!
 * \file pdist_proto.h
 * \brief GE IR registration for Pdist (graph-mode / RunGraph entry).
 */
#ifndef OPS_OP_PROTO_INC_Pdist_H_
#define OPS_OP_PROTO_INC_Pdist_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Computes the p-norm distance between every pair of row vectors. \n
 *        Output is the upper-triangular row-major vector y[idx], where
 *        idx = i*(2*N-i-1)/2 + (j-i-1) for every 0 <= i < j <= N-1. \n

 * @par Inputs:
 * Inputs including:
 * @li x: Input tensor of shape [N, M]. Row i is compared against every row j > i.
 *   - Data types: float16, float32.
 *   - format: ND (row-major contiguous)

 * @par Attributes:
 * @li p: An optional float attribute. The norm order.
 *   - p >= 0 or +inf; p = 0 counts mismatched (including NaN) dimensions,
 *     p = +inf takes the maximum absolute difference, 0 < p < inf uses the
 *     p-power sum followed by the 1/p power. Defaults to 2.0 (Euclidean).

 * @par Outputs:
 * @li y: Pairwise distance vector of shape [N*(N-1)/2], same dtype as x.
 *   - Data types: float16, float32.
 *   - format: ND

 * @attention Constraints:
 * @code{.c}
 *  - 2 <= N <= 65535, 1 <= M <= 65535.
 *  - x must be 2-D row-major contiguous; y dtype equals x dtype.
 *  - fp16 inputs accumulate in fp32 on chip and narrow at the exit.
 * @endcode
 */

REG_OP(Pdist)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(p, Float, 2.0)
    .OP_END_FACTORY_REG(Pdist)

} // namespace ge

#endif // OPS_OP_PROTO_INC_Pdist_H_
