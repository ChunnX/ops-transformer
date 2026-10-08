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
 * \file compressor_grad_common_arch22.h
 * \brief CompressorGrad arch22 kernel helpers shared by the cast/scatter and
 *        gemm kernel headers (single definition point, both headers are
 *        compiled into one translation unit by op_kernel/compressor_grad.cpp).
 */

#ifndef COMPRESSOR_GRAD_COMMON_ARCH22_H
#define COMPRESSOR_GRAD_COMMON_ARCH22_H

#include "kernel_operator.h"
// MIX kernel 的生成包装层会调用 matmul::clearWorkspace（kfc），与 compressor arch22 同样引入
#include "lib/matmul_intf.h"

using namespace AscendC;

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b)
{
    return (b == 0U) ? 0U : (a + b - 1U) / b;
}

__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}

__aicore__ inline uint64_t MinU64(uint64_t a, uint64_t b)
{
    return a < b ? a : b;
}

// Set-then-wait on an intra-core pipe event: blocks until all prior
// reads/writes of the source pipe are complete (TPipe-managed event id).
template <AscendC::HardEvent evt>
__aicore__ inline void SetWaitFlag()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    AscendC::SetFlag<evt>(eventId);
    AscendC::WaitFlag<evt>(eventId);
}

#endif // COMPRESSOR_GRAD_COMMON_ARCH22_H
