// Copyright (c) 2026, edconeone
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Adapted from DeepEP; original copyright and license notices follow.

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#pragma once

#include <type_traits>
#include "kernel_operator.h"

namespace MegaMoeImpl::GmmKernel {
// Both Ascend FP4 encodings pack two logical elements into one byte.
template <typename T>
__aicore__ inline constexpr bool IsFp4()
{
    // CANN compiler aliases may be global; do not require AscendC aliases.
    using namespace AscendC;
    return std::is_same_v<T, fp4x2_e2m1_t> || std::is_same_v<T, fp4x2_e1m2_t>;
}

template <typename T>
__aicore__ inline constexpr T ScalarMin(T a, T b)
{
    return a < b ? a : b;
}
} // namespace MegaMoeImpl::GmmKernel
