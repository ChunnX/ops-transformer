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
 * \file scatter_kv_cache_with_k_scale_proto.h
 * \brief scatter_kv_cache_with_k_scale算子原型定义
 */

#ifndef SCATTER_KV_CACHE_WITH_K_SCALE_PROTO_H_
#define SCATTER_KV_CACHE_WITH_K_SCALE_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief Scatter KV cache with key scale. 将key、value和key_scale更新到对应的cache中。 \n
 *
 * @par 功能说明:
 * 该算子用于将FP8格式的key/value更新到KV cache中，同时将对应的key_scale更新到key_scale_cache中。
 * 支持FP8_E5M2和FP8_E4M3FN两种数据类型，适用于KV cache量化场景。 \n
 *
 * @par Inputs:
 * 7个输入tensor：
 * @li key: 需要更新的key值，FP8格式。形状为[num_tokens, num_head, k_head_size]。
 *          支持数据类型：DT_FLOAT8_E5M2 或 DT_FLOAT8_E4M3FN。
 * @li value: 需要更新的value值，FP8格式。形状为[num_tokens, num_head, v_head_size]。
 *            支持数据类型：DT_FLOAT8_E5M2 或 DT_FLOAT8_E4M3FN。
 * @li key_cache: 需要被key更新的cache，FP8格式。形状为[num_blocks, num_head, block_size, k_head_size]。
 *                支持数据类型：DT_FLOAT8_E5M2 或 DT_FLOAT8_E4M3FN。
 * @li value_cache: 需要被value更新的cache，FP8格式。形状为[num_blocks, num_head, block_size, v_head_size]。
 *                  支持数据类型：DT_FLOAT8_E5M2 或 DT_FLOAT8_E4M3FN。
 * @li slot_mapping: 每个token在cache中的偏移量。形状为[num_tokens]。
 *                   支持数据类型：DT_INT32 或 DT_INT64。
 * @li key_scale: 需要更新的key缩放因子，FP32格式。形状为[num_tokens, num_head]。
 *                支持数据类型：DT_FLOAT。
 * @li key_scale_cache: 需要被key_scale更新的cache，FP32格式。
 *                      形状为[num_blocks, num_head, block_size, 1]。
 *                      支持数据类型：DT_FLOAT。
 *
 * @par Attributes:
 * @li cache_layout: 可选属性，描述cache的布局格式。默认值为"BNBD"。
 *                   当前版本仅支持"BNBD"布局。
 *
 * @par Outputs:
 * 3个输出tensor（inplace更新）：
 * @li key_cache: 更新后的key_cache，与输入key_cache形状和数据类型相同。
 * @li value_cache: 更新后的value_cache，与输入value_cache形状和数据类型相同。
 * @li k_scale_cache: 更新后的key_scale_cache，与输入key_scale_cache形状和数据类型相同。
 *
 * @par Constraints:
 * @li key、value、key_cache、value_cache必须具有相同的数据类型。
 * @li key_scale和key_scale_cache必须为DT_FLOAT类型。
 * @li slot_mapping必须为DT_INT32或DT_INT64类型。
 * @li slot_mapping中的值必须在[0, num_blocks * block_size - 1]范围内。
 * @li num_tokens <= num_blocks * block_size。
 * @li 当k_head_size == v_head_size时，使用特化模式（SPECIALIZED）。
 * @li 当k_head_size != v_head_size时，使用泛化模式（GENERALIZED）。
 *
 * @par 支持的平台:
 * Ascend950
 *
 * @par 示例:
 * @code{.cpp}
 * // 示例shape配置：
 * key:           [128, 32, 128]           // num_tokens=128, num_head=32, k_head_size=128
 * value:         [128, 32, 128]           // num_tokens=128, num_head=32, v_head_size=128
 * key_cache:     [1024, 32, 16, 128]      // num_blocks=1024, num_head=32, block_size=16
 * value_cache:   [1024, 32, 16, 128]      // num_blocks=1024, num_head=32, block_size=16
 * slot_mapping:  [128]                    // num_tokens=128
 * key_scale:     [128, 32]                // num_tokens=128, num_head=32
 * key_scale_cache: [1024, 32, 16, 1]      // num_blocks=1024, num_head=32, block_size=16
 * @endcode
 */

REG_OP(ScatterKvCacheWithKScale)
    .INPUT(key, "T")
    .INPUT(value, "T")
    .INPUT(key_cache, "T")
    .INPUT(value_cache, "T")
    .INPUT(slot_mapping, TensorType::IndexNumberType())
    .INPUT(key_scale, "FP32")
    .INPUT(key_scale_cache, "FP32")
    .OUTPUT(key_cache, "T")
    .OUTPUT(value_cache, "T")
    .OUTPUT(k_scale_cache, "FP32")
    .ATTR(cache_layout, String, "BNBD")
    .DATATYPE(T, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OP_END_FACTORY_REG(ScatterKvCacheWithKScale)

} // namespace ge
#endif // SCATTER_KV_CACHE_WITH_K_SCALE_PROTO_H_