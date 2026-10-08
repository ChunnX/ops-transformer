// Copyright (c) 2026, edconeone
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "low_latency/combine_layout.h"
#include "low_latency/launch.hpp"
#include <array>
#undef NDEBUG
#include <cassert>

int main()
{
    CombineDedup::Layout layout{};
    for (int world : {16, 32, 64, 128}) {
        assert(CombineDedup::BuildLayout(world, 8, 7168, 8, 128, layout));
        assert(layout.capacityRows == static_cast<uint64_t>(world * 8 * (128 / world)));
        assert(layout.requiredWindowBytes <= DispatchDedup::kEpServerDedupWindowUsableBytes);
    }
    for (auto shape : {std::array<int64_t, 5>{8, 8, 7168, 8, 128},
                       {16, 0, 7168, 8, 128},
                       {16, 8, 999999, 8, 128},
                       {16, 8, 7168, 0, 128},
                       {16, 8, 7168, 8, 127}}) {
        assert(!CombineDedup::BuildLayout(shape[0], shape[1], shape[2], shape[3], shape[4], layout));
        assert(layout.capacityRows == 0);
    }
}
