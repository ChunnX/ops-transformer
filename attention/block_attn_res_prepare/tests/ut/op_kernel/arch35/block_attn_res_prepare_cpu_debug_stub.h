/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_ATTN_RES_PREPARE_CPU_DEBUG_STUB_H
#define BLOCK_ATTN_RES_PREPARE_CPU_DEBUG_STUB_H

#include <cstdint>

#include "blaze_kernel_stub.h"
#include "tikicpulib.h"

inline int64_t ld_dev(int64_t *src, int16_t offset)
{
    return static_cast<int64_t>(ld_dev(reinterpret_cast<uint64_t *>(src), offset));
}

inline void vdup(vector_f8e8m0 &, fp8_e8m0_t, vector_bool, int32_t)
{}

inline void vdup(vector_f8e8m0 &, vector_f8e8m0, vector_bool, int32_t, int32_t)
{
}

#include "kernel_operator.h"
#include "tensor_api/tensor.h"

namespace BlockAttnResPrepareOps::CpuDebug {

template <typename T>
__ubuf__ T *ToPhysicalUbAddress(uint64_t byteOffset)
{
    const uint32_t offset = static_cast<uint32_t>(byteOffset);
    AscendC::LocalTensor<uint8_t> localTensor(AscendC::TPosition::VECCALC, offset, AscendC::TOTAL_UB_SIZE - offset);
    return reinterpret_cast<__ubuf__ T *>(localTensor.GetPhyAddr());
}

} // namespace BlockAttnResPrepareOps::CpuDebug

namespace AscendC::Te {

template <typename PtrPattern, typename DataType, typename Addr, EnableMakePtrByTrait<PtrPattern, Addr> = 0>
__aicore__ inline auto BlockAttnResPrepareMakeMemPtr(Addr byteOffset)
{
    auto *physicalAddress = BlockAttnResPrepareOps::CpuDebug::ToPhysicalUbAddress<DataType>(byteOffset);
    return MakeMemPtr<PtrPattern>(physicalAddress);
}

template <typename PtrPattern, typename Iterator, EnableMakeHardwarePtr<PtrPattern, Iterator> = 0>
__aicore__ inline constexpr auto BlockAttnResPrepareMakeMemPtr(Iterator iterator)
{
    return MakeMemPtr<PtrPattern>(iterator);
}

} // namespace AscendC::Te

#define MakeMemPtr BlockAttnResPrepareMakeMemPtr

#endif // BLOCK_ATTN_RES_PREPARE_CPU_DEBUG_STUB_H
