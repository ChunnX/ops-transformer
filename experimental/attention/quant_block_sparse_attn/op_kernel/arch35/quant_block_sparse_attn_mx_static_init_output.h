/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_BLOCK_SPARSE_ATTN_MX_STATIC_INIT_OUTPUT_H_
#define QUANT_BLOCK_SPARSE_ATTN_MX_STATIC_INIT_OUTPUT_H_

#include "kernel_operator.h"
#include "quant_block_sparse_attn_mx_static.h"

namespace BaseApi {
namespace MxStatic {

constexpr uint32_t INIT_OUTPUT_MAX_REPEAT = 255U;
constexpr uint32_t INIT_OUTPUT_BLOCK_BYTES = 32U;
constexpr uint32_t INIT_OUTPUT_SCRATCH_BYTES = INIT_OUTPUT_MAX_REPEAT * INIT_OUTPUT_BLOCK_BYTES;

template <typename T>
__aicore__ inline void InitOutput(AscendC::GlobalTensor<T> dst, uint64_t elementCount, T value,
                                  const AscendC::LocalTensor<uint8_t> &scratchBytes)
{
    if (elementCount == 0U) {
        return;
    }
    constexpr uint32_t scratchElements = INIT_OUTPUT_SCRATCH_BYTES / sizeof(T);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MxStatic::INIT_OUTPUT_FREE_EVENT_ID);
    AscendC::LocalTensor<T> scratch = scratchBytes.template ReinterpretCast<T>();
    AscendC::Duplicate(scratch, value, scratchElements);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(MxStatic::INIT_OUTPUT_READY_EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(MxStatic::INIT_OUTPUT_READY_EVENT_ID);

    uint64_t copied = 0U;
    while (copied < elementCount) {
        const uint64_t remain = elementCount - copied;
        const uint32_t copyElements =
            remain > static_cast<uint64_t>(scratchElements) ? scratchElements : static_cast<uint32_t>(remain);
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1U;
        copyParams.blockLen = copyElements * sizeof(T);
        copyParams.srcStride = 0U;
        copyParams.dstStride = 0U;
        AscendC::DataCopyPad(dst[copied], scratch, copyParams);
        copied += copyElements;
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MxStatic::INIT_OUTPUT_FREE_EVENT_ID);
}

} // namespace MxStatic
} // namespace BaseApi

#endif // QUANT_BLOCK_SPARSE_ATTN_MX_STATIC_INIT_OUTPUT_H_
