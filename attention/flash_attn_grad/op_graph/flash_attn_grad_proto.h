/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_GRAD_PROTO_H
#define FLASH_ATTN_GRAD_PROTO_H

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

REG_OP(FlashAttnGrad)
    .INPUT(q, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(k, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(v, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(dout, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(attn_out, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(softmax_lse, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(cu_seqlens_q, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(cu_seqlens_kv, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(seqused_q, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(seqused_kv, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(sinks, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(attn_mask, TensorType({DT_INT8}))
    .OPTIONAL_INPUT(metadata, TensorType({DT_INT32}))
    .OUTPUT(dq, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(dk, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(dv, TensorType({DT_FLOAT16, DT_BF16}))
    .ATTR(softmax_scale, Float, 0.0)
    .ATTR(mask_mode, Int, 0)
    .ATTR(win_left, Int, -1)
    .ATTR(win_right, Int, -1)
    .ATTR(max_seqlen_q, Int, -1)
    .ATTR(max_seqlen_kv, Int, -1)
    .ATTR(layout_q, String, "BSND")
    .ATTR(layout_kv, String, "BSND")
    .ATTR(layout_out, String, "BSND")
    .OP_END_FACTORY_REG(FlashAttnGrad)

} // namespace ge

#endif // FLASH_ATTN_GRAD_PROTO_H
