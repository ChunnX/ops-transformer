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
 * \file moe_tutel_dispatch_common.h
 * \brief
 */

#ifndef MOE_TUTEL_DISPATCH_COMMON_H_
#define MOE_TUTEL_DISPATCH_COMMON_H_

#include "kernel_operator.h"
#include "moe_tutel_dispatch_tiling_data.h"

namespace MoeTutelDispatchNs {
using namespace AscendC;

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t PAYLOAD_BUFFER_NUM = 2;
constexpr int64_t PAYLOAD_QUEUE_DEPTH = 2;

// x / gates / y 一律按同宽无符号整型搬运：
// - 本算子是纯 bit 搬运，不做任何算术，用整型容器与用 float16/bfloat16/float 完全等价；
// - 规避 bfloat16_t 的 cast / 构造在 arch35 上的兼容性问题。
// 与 attention/scatter_pa_kv_cache 的 arch35 内核同法（它把 payload 模板成 int8/int16/int32）。

// 「值为零」的位型判据：除符号位外全 0。fp16 / bf16 / fp32 皆成立，
// 与 A2 kernel 的 `static_cast<float>(gates_) != 0` 逐位等价（NaN / Inf / denormal 判为非零）。
template <typename TByte>
__aicore__ inline TByte GateAbsMask()
{
    return static_cast<TByte>(sizeof(TByte) == sizeof(uint32_t) ? 0x7FFFFFFFU : 0x00007FFFU);
}

__aicore__ inline int64_t CeilDivInt64(int64_t a, int64_t b)
{
    return (b == 0) ? 0 : ((a + b - 1) / b);
}

__aicore__ inline int64_t CeilAlignInt64(int64_t a, int64_t align)
{
    return (align == 0) ? a : (CeilDivInt64(a, align) * align);
}

__aicore__ inline int64_t MinInt64(int64_t a, int64_t b)
{
    return a < b ? a : b;
}
} // namespace MoeTutelDispatchNs

#endif // MOE_TUTEL_DISPATCH_COMMON_H_
