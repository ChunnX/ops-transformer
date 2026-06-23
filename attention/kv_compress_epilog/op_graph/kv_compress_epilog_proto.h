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
 * \file kv_compress_epilog_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_KvCompressEpilog_H_
#define OPS_OP_PROTO_INC_KvCompressEpilog_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Compresses cached key/value data in the KV-cache epilog stage by applying
 *        group-wise dynamic quantization (FP8 group / E8M0 / HiFloat8) with scale factors. \n
 *        The quantized rows are scattered into cache for each valid slot. \n

 * @par Inputs:
 * Inputs including:
 * @li cache: The existing key/value cache tensor to be updated in-place. \n
 *   - Data types: uint8.
 *   - format: ND
 * @li x: Input activation tensor to be quantized and written into the cache.
 *   - Data types: bfloat16.
 *   - format: ND
 * @li slot_mapping: Token index mapping indicating which cache slot each token maps to.
 *   - Data types: int32, int64.
 *   - format: ND

 * @par Attributes:
 * @li quant_group_size: An optional int attribute. Quant group size. Defaults to 64.
 * @li quant_mode: An optional int attribute. Quantization mode.
 *   - 0: group quantization, scale stored as bfloat16
 *   - 1: group quantization, scale stored as float8_e8m0 (default)
 *   - 2: HiFloat8 quantization (whole-row, scale attribute applied)
 * @li round_scale: An optional bool attribute. Whether to round the group scale value. Defaults to true.
 * @li x_scale: An optional float attribute. Global scale multiplier for HiFloat8 mode. Defaults to 1.0.

 * @par Outputs:
 * @li cache: Updated key/value cache after in-place compression.

 * @attention Constraints:
 * @code{.c}
 *  - cache is both input and output (in-place update).
 *  - slot_mapping dimensions should equal x dimensions minus 1.
 *  - The last dimension (d) of x must be 64-aligned (d % 64 == 0) and greater than 64.
 *  - Ascend950PR/Ascend950DT.
 * @endcode
 */
REG_OP(KvCompressEpilog)
    .INPUT(cache, TensorType({DT_UINT8}))
    .INPUT(x, TensorType({DT_BF16}))
    .INPUT(slot_mapping, TensorType::IndexNumberType())
    .OUTPUT(cache, TensorType({DT_UINT8}))
    .ATTR(quant_group_size, Int, 64)
    .ATTR(quant_mode, Int, 1)
    .ATTR(round_scale, Bool, true)
    .ATTR(x_scale, Float, 1.0)
.OP_END_FACTORY_REG(KvCompressEpilog)
}  // namespace ge

#endif  // OPS_OP_PROTO_INC_KvCompressEpilog_H_