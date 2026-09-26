/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is distributed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_RMS0_STATISTICS_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_RMS0_STATISTICS_H

#include "../compute/rms_norm_0.h"
#include "../mega_mhc_kernel_context.h"

namespace AiInfraMegaMhc::Schedule {
using namespace AscendC;

/** Assign complete RMS0 rows to AIVs and write inverse RMS scalars for the caller's global boundary. */
template <typename T>
__aicore__ inline void RunRms0Statistics(Kernel::Tensors<T> &tensors, const MegaMhcTilingData &tiling,
                                         TPipe *pipe)
{
    if ASCEND_IS_AIV {
        TBuf<TPosition::VECCALC> hOutBuffer;
        TBuf<TPosition::VECCALC> statsBuffer;
        Kernel::InitUbBuffer(pipe, hOutBuffer, tiling.d * sizeof(T));
        Kernel::InitUbBuffer(pipe, statsBuffer, 2U * sizeof(float));
        LocalTensor<T> hOut = hOutBuffer.Get<T>();
        LocalTensor<float> stats = statsBuffer.Get<float>();
        __ubuf__ float *statsPtr = reinterpret_cast<__ubuf__ float *>(stats.GetPhyAddr());

        const uint32_t aivIndex = GetBlockIdx();
        for (uint64_t row = aivIndex; row < tiling.totalRows; row += tiling.activeAiv) {
            Kernel::CopyGlobalToUb<T>(hOut, tensors.hOut, static_cast<uint64_t>(row) * tiling.d, tiling.d);
            Compute::ComputeRmsNorm0Statistics<T>(
                reinterpret_cast<__ubuf__ T *>(hOut.GetPhyAddr()), tiling.d, tiling.normEps,
                statsPtr + 1, statsPtr);
            Kernel::CopyUbToGlobal<float>(tensors.invRms0, row, stats, 1U);
        }
    }
}

}  // namespace AiInfraMegaMhc::Schedule

#endif  // OPS_TRANSFORMER_MHC_MEGA_MHC_RMS0_STATISTICS_H
