/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_FT_CATLASS_ARCH_CROSS_CORE_SYNC_AIV_HPP
#define MATMUL_FT_CATLASS_ARCH_CROSS_CORE_SYNC_AIV_HPP

#include "catlass/arch/cross_core_sync.hpp"

namespace FTSelf {

template <uint8_t MODE, pipe_t PIPE>
CATLASS_DEVICE
void CrossCoreBarrierAIC()
{
    constexpr Catlass::Arch::FlagID flagId = Catlass::Arch::BarrierFlag<MODE, AscendC::AIC>::ID;
    AscendC::CrossCoreSetFlag<MODE, PIPE>(flagId);
    AscendC::CrossCoreWaitFlag(flagId);
}

template <uint8_t MODE, pipe_t PIPE>
CATLASS_DEVICE
void CrossCoreBarrierAIV()
{
    constexpr Catlass::Arch::FlagID flagId = Catlass::Arch::BarrierFlag<MODE, AscendC::AIV>::ID;
    AscendC::CrossCoreSetFlag<MODE, PIPE>(flagId);
    AscendC::CrossCoreWaitFlag(flagId);
}

}  // namespace FTSelf

#endif  // MATMUL_FT_CATLASS_ARCH_CROSS_CORE_SYNC_AIV_HPP
