/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef MEGA_MHC_PROJECTION_STAGE_H
#define MEGA_MHC_PROJECTION_STAGE_H
#include "rms_norm_common.h"

namespace AiInfraMegaMhc::Compute {
/** Convert an entire AIV tile and accumulate Pre statistics in the same VF.
 * Formula: stage[r,c] = FP32(x2[r,c]); squares[r] += sum_c stage[r,c]^2.
 * Input is packed by valid K; stage uses the host-planned L1 row stride.
 * Each lane adds at most 16 squares for host K<=1024, then reduces in fixed order.
 * COMPENSATE retains the compensated variant for numerical/performance comparisons.
 */
template <typename T, bool COMPENSATE = false>
__aicore__ inline void PrepareProjectionTile(__ubuf__ T *input, uint32_t rows,
    uint32_t kCount, uint32_t stageStride, __ubuf__ float *stage, __ubuf__ float *squares)
{
    const uint16_t rowCount = static_cast<uint16_t>(rows);
    const uint16_t repeats = static_cast<uint16_t>((kCount + 63U) / 64U);
    __VEC_SCOPE__
    {
        Reg::MaskReg fullMask = MaskForCount(64U);
        Reg::MaskReg scalarMask = MaskForCount(1U);
        for (uint16_t row = 0; row < rowCount; ++row) {
            Reg::RegTensor<float> sum;
            Reg::RegTensor<float> compensation;
            Reg::RegTensor<float> value;
            Reg::RegTensor<float> square;
            Reg::RegTensor<float> corrected;
            Reg::RegTensor<float> next;
            Reg::RegTensor<float> difference;
            Reg::RegTensor<float> partial;
            Reg::RegTensor<float> previous;
            Reg::RegTensor<T> raw;
            Reg::Duplicate(sum, 0.0f, fullMask);
            Reg::Duplicate(compensation, 0.0f, fullMask);
            uint32_t remaining = kCount;
            for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
                const uint32_t column = repeat * 64U;
                Reg::MaskReg mask = Reg::UpdateMask<float>(remaining);
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(raw, input + row * kCount + column);
                Reg::Cast<float, T, kCastInputToFloat>(value, raw, mask);
                Reg::StoreAlign<float>(stage + row * stageStride + column, value, mask);
                Reg::Mul(square, value, value, fullMask);
                if constexpr (COMPENSATE) {
                    Reg::Sub(corrected, square, compensation, fullMask);
                    Reg::Add(next, sum, corrected, fullMask);
                    Reg::Sub(difference, next, sum, fullMask);
                    Reg::Sub(compensation, difference, corrected, fullMask);
                    Reg::Adds(sum, next, 0.0f, fullMask);
                } else {
                    Reg::Add(sum, sum, square, fullMask);
                }
            }
            Reg::Reduce<Reg::ReduceType::SUM>(partial, sum, fullMask);
            Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(previous, squares + row);
            Reg::Add(partial, previous, partial, scalarMask);
            Reg::Store<float>(squares + row, partial, 1U);
        }
    }
}
} // namespace AiInfraMegaMhc::Compute
#endif
