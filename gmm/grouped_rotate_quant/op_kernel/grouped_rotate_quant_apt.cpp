/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is licensed under the CANN Open Software License Agreement Version 2.0.
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "arch35/grouped_rotate_quant_kernel_arch35.h"

namespace {
template <typename Left, typename Right>
constexpr bool SAME_DTYPE = false;
template <typename Type>
constexpr bool SAME_DTYPE<Type, Type> = true;
} // namespace

// OpDef order: four inputs, two dynamic outputs, workspace, tiling.
extern "C" __global__ __aicore__ void grouped_rotate_quant(
    GM_ADDR x, GM_ADDR rotation, GM_ADDR alpha, GM_ADDR groupList, GM_ADDR y, GM_ADDR scale,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GroupedRotateQuant::TilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    // v1 reserves system workspace only. Do not give its base address to the
    // backend as user scratch; introducing scratch requires a host ABI update.
    (void)workspace;
    static_assert(GroupedRotateQuant::TILING_KEY == 0, "Update the GroupedRotateQuant entry dispatch");
    static_assert(SAME_DTYPE<DTYPE_X, DTYPE_ROTATION>, "x and rotation must have the same dtype");
    static_assert(SAME_DTYPE<DTYPE_SCALE, AscendC::fp8_e8m0_t>, "scale must be E8M0");
    if (TILING_KEY_IS(0)) {
        GroupedRotateQuant::GroupedRotateQuantKernel<DTYPE_X, DTYPE_Y>(
            x, rotation, alpha, groupList, y, scale, tiling);
    }
}
