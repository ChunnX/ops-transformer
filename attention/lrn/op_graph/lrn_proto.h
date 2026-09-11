/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lrn_proto.h
 * \brief LRN operator prototype definition: TF across-channels local
 *        response normalization on 4-D NCHW tensors.
 */
#ifndef LRN_PROTO_H
#define LRN_PROTO_H

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief LRN. \n
 *
 * Formula (TF across-channels semantics):
 * @code
 * y[n, c, h, w] = x[n, c, h, w]
 *                 / (bias + alpha * sum_{j in win(c)} x[n, j, h, w]^2)^beta
 * win(c) = [max(0, c - depth_radius), min(C - 1, c + depth_radius)]
 * @endcode
 * The window is clipped at both channel boundaries; alpha is NOT divided
 * by the window width (TF/GEIR semantics). norm > 0 always holds because
 * bias > 0, so the power runs the stable exp(-beta * ln(norm)) path.
 *
 * @par Inputs:
 * 1 input tensor:
 * @li x: Input feature map. Shape is [N, C, H, W] (4-D NCHW).
 *        Supported data types: DT_FLOAT16 or DT_FLOAT (same-dtype only).
 *
 * @par Attributes:
 * @li depth_radius: Half-width of the channel summation window r; the
 *                   effective window is [max(0, c-r), min(C-1, c+r)].
 *                   Optional, Int, default 5, range [0, 48].
 * @li bias: Additive constant k of the norm (norm = bias + alpha * S).
 *           Optional, Float, default 1.0, must be > 0.
 * @li alpha: Scale of the window square-sum (NOT divided by the window
 *            width). Optional, Float, default 1.0, must be > 0.
 * @li beta: Exponent of the norm. Optional, Float, default 0.5.
 * @li norm_region: Normalization region; only "ACROSS_CHANNELS" is
 *                  supported (WITHIN_CHANNEL is rejected).
 *
 * @par Outputs:
 * 1 output tensor:
 * @li y: Normalized feature map. Same shape and data type as x.
 *
 * @attention Constraints:
 * @code{.c}
 * - x must be 4-D NCHW: N in [1, 4096], C in [1, 4096],
 *   H in [1, 8192], W in [1, 8192].
 * - depth_radius in [0, 48]; bias > 0 and alpha > 0 (norm > 0).
 * - norm_region must be "ACROSS_CHANNELS".
 * @endcode
 */

REG_OP(LRN)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(depth_radius, Int, 5)
    .ATTR(bias, Float, 1.0)
    .ATTR(alpha, Float, 1.0)
    .ATTR(beta, Float, 0.5)
    .ATTR(norm_region, String, "ACROSS_CHANNELS")
    .OP_END_FACTORY_REG(LRN)

} // namespace ge

#endif // LRN_PROTO_H
