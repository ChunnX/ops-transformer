/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dispatch_policy_frag.h
 * \brief Dispatch policy for FragmentTensor-aware matmul (AllToAll+Matmul fusion).
 */
#pragma once

#include "blaze/gemm/policy/dispatch_policy.h"

namespace Blaze {
namespace Gemm {

template <uint64_t FULL_LOAD_MODE_ = 0, bool ATOMIC_ADD = false>
struct MatmulWithScaleMxFrag {
    using ScheduleType = KernelMmadWithScaleMx;
    constexpr static uint64_t fullLoadMode = FULL_LOAD_MODE_;
    constexpr static bool isAtomicAdd = ATOMIC_ADD;
};

} // namespace Gemm
} // namespace Blaze
