/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "../common/constants.h"
#include "../common/types.h"
#include "sync.h"

#if defined(__CCE__)
#include "kernel_operator.h"

namespace sage::hardware {

// ============================================================================
// Global <-> Local Tensor Row Transfer Helpers with Auto-Padding
// ============================================================================

template <class T>
__aicore__ inline void ReadRow(AscendC::LocalTensor<T> dst, AscendC::GlobalTensor<T> src,
                               uint64_t off, uint32_t d, uint32_t dp) {
    AscendC::Duplicate(dst, T(0), dp);
    Sync<AscendC::HardEvent::V_MTE2>();
    AscendC::DataCopyExtParams copy{1, d * uint32_t(sizeof(T)), 0, 0, 0};
    constexpr uint32_t blockElements = 32 / sizeof(T);
    uint8_t rightPadding = static_cast<uint8_t>((blockElements - d % blockElements) % blockElements);
    AscendC::DataCopyPadExtParams<T> pad{true, 0, rightPadding, T(0)};
    AscendC::DataCopyPad(dst, src[off], copy, pad);
    Sync<AscendC::HardEvent::MTE2_V>();
}

template <class T>
__aicore__ inline void WriteRow(AscendC::GlobalTensor<T> dst, uint64_t off,
                                AscendC::LocalTensor<T> src, uint32_t n) {
    Sync<AscendC::HardEvent::V_MTE3>();
    AscendC::DataCopyExtParams copy{1, n * uint32_t(sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(dst[off], src, copy);
}

// ============================================================================
// Catlass Native zN Fractal Layout Direct Transfer
// ============================================================================

// Store an 8x8 FP8 tile fragment across two 32-element NZ planes without bus transpose
template <uint32_t STRIDE>
__simd_callee__ inline void StoreNZPlanePair(__ubuf__ float8_e4m3_t *out, uint32_t plane, uint32_t row,
                                             AscendC::MicroAPI::RegTensor<float8_e4m3_t> &packed,
                                             AscendC::MicroAPI::MaskReg &first32,
                                             AscendC::MicroAPI::MaskReg &last32) {
    using namespace AscendC::MicroAPI;
    StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + plane * STRIDE + row * 32, packed, first32);
    StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + (plane + 1) * STRIDE + row * 32 - 32, packed, last32);
}

} // namespace sage::hardware

namespace sage {
using hardware::ReadRow;
using hardware::WriteRow;
using hardware::StoreNZPlanePair;
} // namespace sage

#endif // __CCE__
