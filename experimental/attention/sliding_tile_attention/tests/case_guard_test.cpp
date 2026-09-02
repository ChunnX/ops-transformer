/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <iostream>
#include <limits>

#include "sliding_tile_attention_case_guard.h"

namespace {

bool Check(bool condition, const char *message)
{
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        return false;
    }
    return true;
}

} // namespace

int main()
{
    using namespace sliding_tile_attention;
    bool ok = true;

    for (int64_t heads : {1, 7, 8, 9, 127, 128}) {
        int64_t broadcast[] = {3, 5, 7};
        WindowValues windows;
        ok &= Check(NormalizeWindows(broadcast, WINDOW_VALUES_PER_HEAD, heads, windows),
                    "broadcast window must normalize for every supported head count");
        ok &= Check(static_cast<int64_t>(windows.size()) == heads * WINDOW_VALUES_PER_HEAD,
                    "normalized window length must be 3*N");
        ok &= Check(IsSupportedMetadata(3, heads, STEPVIDEO_SEQUENCE, 64, 0, false, STEPVIDEO_SEQ_SHAPE_ID, windows),
                    "positive batch and supported N must be accepted");
    }

    ok &= Check(!IsSupportedHeads(0), "N=0 must be rejected");
    ok &= Check(!IsSupportedHeads(129), "N=129 must be rejected");
    for (int64_t headDim : {8, 16, 56, 64, 72, 128, 256}) {
        ok &= Check(IsSupportedHeadDim(headDim), "positive multiples of eight must be accepted");
    }
    ok &= Check(!IsSupportedHeadDim(0), "D=0 must be rejected for non-empty tensors");
    ok &= Check(!IsSupportedHeadDim(7), "D not divisible by eight must be rejected");

    WindowValues perHead;
    int64_t perHeadValues[] = {1, 3, 5, 7, 9, 11};
    ok &= Check(NormalizeWindows(perHeadValues, 6, 2, perHead), "arbitrary per-head windows must be accepted");
    ok &= Check(perHead[0] == 1 && perHead[5] == 11, "per-head windows must preserve values");
    int64_t invalidWindow[] = {3, 0, 3};
    ok &= Check(!NormalizeWindows(invalidWindow, 3, 1, perHead), "non-positive window values must be rejected");
    int64_t evenWindow[] = {3, 4, 5};
    ok &= Check(!NormalizeWindows(evenWindow, 3, 1, perHead), "even window values must be rejected");
    int64_t wideWindow[] = {(1LL << 32) + 1, std::numeric_limits<int64_t>::max(), 3};
    ok &= Check(NormalizeWindows(wideWindow, 3, 1, perHead),
                "positive odd int64 windows must be preserved for device-side extent saturation");
    ok &= Check(perHead[0] == wideWindow[0] && perHead[1] == wideWindow[1],
                "window normalization must not narrow int64 values");
    ok &= Check(!NormalizeWindows(perHeadValues, 6, 3, perHead), "window length other than 3 or 3*N must fail");

    WindowValues oneHead = {3, 3, 3};
    ok &= Check(!IsSupportedMetadata(0, 1, STEPVIDEO_SEQUENCE, 8, 0, false, STEPVIDEO_SEQ_SHAPE_ID, oneHead),
                "non-empty execution metadata requires positive batch");
    ok &= Check(!IsSupportedMetadata(1, 1, STEPVIDEO_SEQUENCE, 8, 1, false, STEPVIDEO_SEQ_SHAPE_ID, oneHead),
                "hasText must match textLength");
    ok &= Check(!IsSupportedMetadata(1, 1, STEPVIDEO_SEQUENCE + 8, 8, 0, false, STEPVIDEO_SEQ_SHAPE_ID, oneHead),
                "sequence must match the selected canvas plus textLength");
    ok &=
        Check(IsSupportedMetadata(1, 1, FASTVIDEO_IMAGE_SEQUENCE + 256, 8, 256, true, FASTVIDEO_SEQ_SHAPE_ID, oneHead),
              "text storage capacity may exceed the original 128-token task case");

    if (!ok) {
        return EXIT_FAILURE;
    }
    std::cout << "ALL_SLIDING_TILE_ATTENTION_CASE_GUARD_TESTS_PASSED dynamic_heads=1..128 head_dim_multiple=8\n";
    return EXIT_SUCCESS;
}
