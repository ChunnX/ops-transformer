/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "../common/constants.h"
#include "../common/types.h"
#include "../common/math.h"
#include "../hardware/memory.h"
#include "../hardware/sync.h"
#include "../hardware/layout.h"

namespace sage::compute {

#if defined(__CCE__)
using AscendC::RoundMode;
using AscendC::CMPMODE;

// ============================================================================
// Channel-wise Reduction Microkernel (4-Way Unrolled)
// ============================================================================

template <class T, uint32_t DP>
__simd_callee__ inline void LoadLaneF32(AscendC::MicroAPI::RegTensor<float> &x, __ubuf__ T *addr,
                                        uint32_t row, uint32_t col) {
    using namespace AscendC::MicroAPI;
    auto mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<T> raw;
    static constexpr CastTrait cast{RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    LoadAlign<T, LoadDist::DIST_UNPACK_B16>(raw, addr + row * DP + col);
    Cast<float, T, cast>(x, raw, mask);
}

template <class T, uint32_t DP>
__simd_vf__ inline void StatsSumImpl(__ubuf__ T *rows, __ubuf__ float *state, uint32_t count) {
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto mask = CreateMask<float, MaskPattern::ALL>();
    for (uint32_t col = 0; col < DP; col += 64) {
        RegTensor<float> acc0, acc1, acc2, acc3;
        LoadAlign(acc0, state + col);
        Duplicate(acc1, 0.0f);
        Duplicate(acc2, 0.0f);
        Duplicate(acc3, 0.0f);
        uint32_t row = 0;
        for (; row + 3 < count; row += 4) {
            RegTensor<float> x0, x1, x2, x3;
            RegTensor<T> raw0, raw1, raw2, raw3;
            static constexpr CastTrait cast{RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(raw0, rows + (row + 0) * DP + col);
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(raw1, rows + (row + 1) * DP + col);
            Cast<float, T, cast>(x0, raw0, mask);
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(raw2, rows + (row + 2) * DP + col);
            Cast<float, T, cast>(x1, raw1, mask);
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(raw3, rows + (row + 3) * DP + col);
            Cast<float, T, cast>(x2, raw2, mask);
            Add(acc0, acc0, x0, mask);
            Cast<float, T, cast>(x3, raw3, mask);
            Add(acc1, acc1, x1, mask);
            Add(acc2, acc2, x2, mask);
            Add(acc3, acc3, x3, mask);
        }
        for (; row + 1 < count; row += 2) {
            RegTensor<float> x0, x1;
            LoadLaneF32<T, DP>(x0, rows, row, col);
            LoadLaneF32<T, DP>(x1, rows, row + 1, col);
            Add(acc0, acc0, x0, mask);
            Add(acc1, acc1, x1, mask);
        }
        if (row < count) {
            RegTensor<float> x_tail;
            LoadLaneF32<T, DP>(x_tail, rows, row, col);
            Add(acc0, acc0, x_tail, mask);
        }
        Add(acc0, acc0, acc1, mask);
        Add(acc2, acc2, acc3, mask);
        Add(acc0, acc0, acc2, mask);
        StoreAlign(state + col, acc0, mask);
    }
}

template <class T>
__aicore__ inline void StatsSumVF(AscendC::LocalTensor<T> rows, AscendC::LocalTensor<float> state,
                                  uint32_t count, uint32_t dp) {
    auto input = (__ubuf__ T *)rows.GetPhyAddr();
    auto sum = (__ubuf__ float *)state.GetPhyAddr();
    if (dp == 128)
        StatsSumImpl<T, 128>(input, sum, count);
    else
        StatsSumImpl<T, 64>(input, sum, count);
}

// ============================================================================
// Smooth-V Mean Centering Microkernel
// ============================================================================

template <class T, uint32_t DP>
__simd_vf__ inline void CenterVRowsImpl(__ubuf__ T *v, __ubuf__ float *meanV, uint32_t rows) {
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto f = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait toFloat{RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr CastTrait back{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    for (uint16_t col = 0; col < DP; col += 64) {
        RegTensor<float> mv;
        LoadAlign(mv, meanV + col);
        for (uint16_t row = 0; row < rows; ++row) {
            RegTensor<T> raw, packed;
            RegTensor<float> x;
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(raw, v + row * DP + col);
            Cast<float, T, toFloat>(x, raw, f);
            Sub(x, x, mv, f);
            Cast<T, float, back>(packed, x, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(v + row * DP + col, packed, f);
        }
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

template <class T>
__aicore__ inline void CenterVRowsVF(__ubuf__ T *v, __ubuf__ float *meanV, uint32_t dp, uint32_t rows) {
    if (dp == 128)
        CenterVRowsImpl<T, 128>(v, meanV, rows);
    else
        CenterVRowsImpl<T, 64>(v, meanV, rows);
}

#endif // __CCE__

} // namespace sage::compute
