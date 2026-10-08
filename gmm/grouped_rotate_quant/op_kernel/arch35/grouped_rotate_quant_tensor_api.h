/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef GROUPED_ROTATE_QUANT_TENSOR_API_H
#define GROUPED_ROTATE_QUANT_TENSOR_API_H

#include "grouped_rotate_quant_kernel_args.h"

namespace GroupedRotateQuant {
// Integration point for the future GROUPED Tensor API implementation.
// Replace/specialize this binding once its BlockMmad, Epilogue and Scheduler
// interfaces are available. An available binding must provide:
//   static constexpr bool AVAILABLE = true;
//   using Kernel = ...;
// Kernel must synchronously implement operator()(const KernelArguments &).
// Assemble the actual Blaze components and translate their Params here; do not
// reinterpret TilingData as the single-group RotateQuant tiling structure.
template <typename XType, typename YType>
struct TensorApiBackend {
    static constexpr bool AVAILABLE = false;
};
} // namespace GroupedRotateQuant
#endif
