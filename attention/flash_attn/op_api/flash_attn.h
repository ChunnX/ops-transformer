/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_FLASH_ATTN_H_
#define OP_API_INC_LEVEL0_FLASH_ATTN_H_

#include <array>
#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief FlashAttn level-0 operator。
 *        封装FlashAttn算子的底层调度，完成InferShape与Kernel Launch注册。
 *        该接口为内部接口，仅供aclnn层调用。
 *
 * @param q                   query tensor
 * @param k                   key tensor
 * @param v                   value tensor
 * @param blockTableOptional  分页KV缓存块映射表（可选，INT32）
 * @param cuSeqlensQOptional  query累积序列长度tensor（可选，INT32）
 * @param cuSeqlensKvOptional kv累积序列长度tensor（可选，INT32）
 * @param sequsedQOptional    query各batch实际序列长度tensor（可选，INT32）
 * @param sequsedKvOptional   kv各batch实际序列长度tensor（可选，INT32）
 * @param sinksOptional       可学习sink权重（可选，FLOAT32）
 * @param metadataOptional    预计算tiling元数据（可选，INT32）
 * @param softmaxMode         softmax缩放系数（float）
 * @param maskMode            掩码模式（int64_t）
 * @param winLeft             左窗口大小（int64_t）
 * @param winRight            右窗口大小（int64_t）
 * @param layoutQ             query布局字符串
 * @param layoutKv            kv布局字符串
 * @param layoutOut           输出布局字符串
 * @param returnSoftmaxLse    是否输出softmax_lse（int64_t）
 * @param deterministic       是否确定性计算（int64_t）
 * @param executor            op执行器
 * @return std::array<const aclTensor*, 2> [attentionOut, softmaxLse]
 *         任意元素为nullptr表示对应输出的InferShape或Launch失败。
 */
const std::array<const aclTensor *, 2> FlashAttn(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *blockTableOptional,
    const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensKvOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKvOptional,
    const aclTensor *sinksOptional,
    const aclTensor *metadataOptional,
    float softmaxMode,
    int64_t maskMode,
    int64_t winLeft,
    int64_t winRight,
    const char *layoutQ,
    const char *layoutKv,
    const char *layoutOut,
    int64_t returnSoftmaxLse,
    int64_t deterministic,
    aclOpExecutor *executor);

}  // namespace l0op

#endif  // OP_API_INC_LEVEL0_FLASH_ATTN_H_
