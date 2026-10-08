/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef GROUPED_ROTATE_QUANT_KERNEL_ARGS_H
#define GROUPED_ROTATE_QUANT_KERNEL_ARGS_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "grouped_rotate_quant_tiling_data.h"

namespace GroupedRotateQuant {
// Operator-local integration contract, NOT an existing Blaze Params type.
// The four list addresses point to TensorList descriptors, including lists of
// length one. The backend owns descriptor decoding and per-group views.
struct KernelArguments {
    GM_ADDR xList;
    GM_ADDR rotationList;
    GM_ADDR alpha; // BF16[1], nullptr when absent
    GM_ADDR groupList; // INT64[groupNum], nullptr when absent
    GM_ADDR yList;
    GM_ADDR scaleList; // separate TensorList descriptor, never derived from yList
    const TilingData &tiling;
};
} // namespace GroupedRotateQuant
#endif
