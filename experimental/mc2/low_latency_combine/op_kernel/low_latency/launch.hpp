// Copyright (c) 2026, edconeone
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once
#include <cstdint>
#include <type_traits>
namespace deepep {
struct EpServerDedupCombineKernelArgs {
    uint8_t *commArgs, *expertOut, *destinationIndex, *sendCounts, *expertScales;
    uint8_t *tokenType, *relayReadIndex, *sourceMask, *yOut, *tilingData;
};
static_assert(std::is_standard_layout<EpServerDedupCombineKernelArgs>::value, "combine launch ABI");
static_assert(sizeof(EpServerDedupCombineKernelArgs) == 80, "combine launch ABI");
} // namespace deepep
extern "C" void combine_server_dedup_kernel_do(void* stream, const deepep::EpServerDedupCombineKernelArgs* args);
