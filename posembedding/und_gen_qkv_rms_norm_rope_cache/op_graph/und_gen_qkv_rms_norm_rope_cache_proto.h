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
 * \file und_gen_qkv_rms_norm_rope_cache_proto.h
 * \brief
 */

#ifndef OPS_OP_PROTO_INC_UND_GEN_QKV_RMS_NORM_ROPE_CACHE_H_
#define OPS_OP_PROTO_INC_UND_GEN_QKV_RMS_NORM_ROPE_CACHE_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
 * @par Restrictions:
 * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
 */

REG_OP(UndGenQkvRmsNormRopeCache)
    .INPUT(und_qkv, TensorType({DT_BF16}))            // [und_len, N, D], N = Hq + Hk + Hv
    .INPUT(und_weights_q, TensorType({DT_BF16}))      // [D]
    .INPUT(und_weights_k, TensorType({DT_BF16}))      // [D]
    .INPUT(cos_sin_cache, TensorType({DT_FLOAT}))     // [max_pos, D], 前半 cos 后半 sin
    .INPUT(k_cache, TensorType({DT_BF16}))            // [Bn, Bs, Hk, D], 原地写入
    .INPUT(v_cache, TensorType({DT_BF16}))            // [Bn, Bs, Hv, D], 原地写入
    .INPUT(slot_mapping, TensorType({DT_INT64}))      // [T], slot = block_num * Bs + row_idx
    .INPUT(positions, TensorType({DT_INT64}))         // [3, T], MRoPE 三轴位置
    // NOTE: 以下 4 个可选输入当前实现要求必须全部提供（缺省的退化路径暂不支持，tiling 会拦截），
    //       IR 保留 OPTIONAL 语义以便后续放开。
    .OPTIONAL_INPUT(gen_qkv, TensorType({DT_BF16}))   // [gen_len, N, D]
    .OPTIONAL_INPUT(gen_weights_q, TensorType({DT_BF16}))  // [D]
    .OPTIONAL_INPUT(gen_weights_k, TensorType({DT_BF16}))  // [D]
    .OPTIONAL_INPUT(cat_indices, TensorType({DT_INT64}))   // [T], out_t -> src_t
    .OUTPUT(q, TensorType({DT_BF16}))                 // [T, Hq, D]
    .OUTPUT(k_cache, TensorType({DT_BF16}))           // [Bn, Bs, Hk, D], 与输入同地址
    .OUTPUT(v_cache, TensorType({DT_BF16}))           // [Bn, Bs, Hv, D], 与输入同地址
    .REQUIRED_ATTR(num_heads_q, Int)
    .REQUIRED_ATTR(num_heads_k, Int)
    .REQUIRED_ATTR(num_heads_v, Int)
    .ATTR(norm_eps, Float, 1e-6f)
    .ATTR(mrope_section, ListInt, {})
    .OP_END_FACTORY_REG(UndGenQkvRmsNormRopeCache)
} // namespace ge

#endif // OPS_OP_PROTO_INC_UND_GEN_QKV_RMS_NORM_ROPE_CACHE_H_
