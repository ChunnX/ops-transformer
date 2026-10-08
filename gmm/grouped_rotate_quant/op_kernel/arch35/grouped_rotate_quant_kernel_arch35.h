/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef GROUPED_ROTATE_QUANT_KERNEL_ARCH35_H
#define GROUPED_ROTATE_QUANT_KERNEL_ARCH35_H

#include "kernel_tiling/kernel_tiling.h"
#include "grouped_rotate_quant_tensor_api.h"

namespace GroupedRotateQuant {
template <typename Backend>
__aicore__ inline void RunGroupedRotateQuant(
    GM_ADDR x, GM_ADDR rotation, GM_ADDR alpha, GM_ADDR groupList, GM_ADDR y, GM_ADDR scale,
    const TilingData &tilingData)
{
    // Fail at compile time until a real grouped Tensor API backend is bound.
    // Never publish an empty device implementation that reports success.
    static_assert(Backend::AVAILABLE,
        "GroupedRotateQuant: grouped Tensor API backend is not integrated; "
        "complete arch35/grouped_rotate_quant_tensor_api.h before building this operator.");
    if constexpr (Backend::AVAILABLE) {
        ASSERT(tilingData.version == TILING_VERSION);
        if (tilingData.version != TILING_VERSION || tilingData.totalElements == 0) {
            return;
        }
        // runtimeM may have tileCount==0 for NONEMPTY input. Both AIC and AIV
        // enter the same backend invocation; no per-group calls or AIV return.
        const KernelArguments args{x, rotation, tilingData.hasAlpha != 0 ? alpha : nullptr,
            tilingData.hasGroupList != 0 ? groupList : nullptr, y, scale, tilingData};
        typename Backend::Kernel kernel;
        kernel(args);
    }
}

template <typename XType, typename YType>
__aicore__ inline void GroupedRotateQuantKernel(
    GM_ADDR x, GM_ADDR rotation, GM_ADDR alpha, GM_ADDR groupList, GM_ADDR y, GM_ADDR scale, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(TilingData, tilingData, tiling);
    RunGroupedRotateQuant<TensorApiBackend<XType, YType>>(
        x, rotation, alpha, groupList, y, scale, tilingData);
}
} // namespace GroupedRotateQuant
#endif
