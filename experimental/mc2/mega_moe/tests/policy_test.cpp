// Copyright (c) 2026, edconeone
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Test production scheduling decisions at the Decode/Prefill boundary on CPU.
#define __aicore__
#include "kernels/mega_moe/catlass_gmm/a8w4_k_plan.hpp"
#include "kernels/mega_moe/common/mega_moe_shared_policy.hpp"
#include <initializer_list>
#undef NDEBUG
#include <cassert>

int main()
{
    using namespace MegaMoeImpl;
    for (unsigned shared : {0U, 1U, 2U}) {
        auto decode = SelectSharedExpertPolicy(true, shared, 256);
        auto prefill = SelectSharedExpertPolicy(true, shared, 257);
        assert(decode.decode == (shared > 0));
        assert(!prefill.decode);
        assert(prefill.routeAiv1Only == (shared > 0));
        assert(!decode.routeAiv1Only);
        assert(decode.gmm1TileN == (shared ? 192U : 256U));
        assert(prefill.gmm1TileN == 256U);
    }
    assert(UseDecodeInputReady(true, 256));
    assert(!UseDecodeInputReady(true, 257));
    assert(!UseDecodeInputReady(false, 256));
    assert(!SelectSharedExpertPolicy(false, 1, 256).decode);
    using GmmKernel::MakeA8W4KPlan;
    assert(MakeA8W4KPlan(128, 128, true).kb == 512);
    assert(MakeA8W4KPlan(257, 128, true).kb == 256);
    assert(MakeA8W4KPlan(128, 192, true).kb == 256);
    assert(MakeA8W4KPlan(128, 128, false).kb == 256);
}
