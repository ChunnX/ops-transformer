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

#include "sliding_tile_attention_case_guard.h"

namespace {

using sliding_tile_attention::WindowArray;

bool Accepted(int64_t batch, int64_t sequence, bool bf16, int64_t textLength, bool hasText, int64_t seqShape,
              const WindowArray &windows)
{
    return sliding_tile_attention::IsSupportedNormalizedSignature(batch, sliding_tile_attention::SUPPORTED_HEADS,
                                                                  sequence, sliding_tile_attention::SUPPORTED_HEAD_DIM,
                                                                  bf16, textLength, hasText, seqShape, windows);
}

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

    for (bool bf16 : {false, true}) {
        ok &= Check(Accepted(1, FASTVIDEO_SEQUENCE, bf16, FASTVIDEO_TEXT_LENGTH, true, FASTVIDEO_SEQ_SHAPE_ID,
                             SOURCE_WINDOW_PATTERN),
                    "FastVideo source window must be accepted");
        ok &= Check(Accepted(1, FASTVIDEO_SEQUENCE, bf16, FASTVIDEO_TEXT_LENGTH, true, FASTVIDEO_SEQ_SHAPE_ID,
                             WIDE_WINDOW_PATTERN),
                    "FastVideo wide window must be accepted");
        ok &= Check(Accepted(1, FASTVIDEO_SEQUENCE, bf16, FASTVIDEO_TEXT_LENGTH, true, FASTVIDEO_SEQ_SHAPE_ID,
                             UNIFORM_333_WINDOW_PATTERN),
                    "FastVideo uniform window must be accepted");
        ok &= Check(Accepted(1, STEPVIDEO_SEQUENCE, bf16, 0, false, STEPVIDEO_SEQ_SHAPE_ID, SOURCE_WINDOW_PATTERN),
                    "StepVideo source window must be accepted");
        ok &= Check(Accepted(1, STEPVIDEO_SEQUENCE, bf16, 0, false, STEPVIDEO_SEQ_SHAPE_ID, WIDE_WINDOW_PATTERN),
                    "StepVideo wide window must be accepted");
        ok &= Check(Accepted(1, STEPVIDEO_SEQUENCE, bf16, 0, false, STEPVIDEO_SEQ_SHAPE_ID, UNIFORM_333_WINDOW_PATTERN),
                    "StepVideo uniform window must be accepted");
        ok &= Check(Accepted(1, WAN_SEQUENCE, bf16, 0, false, WAN_SEQ_SHAPE_ID, UNIFORM_333_WINDOW_PATTERN),
                    "Wan uniform window must be accepted");
        ok &= Check(Accepted(1, WAN_SEQUENCE, bf16, 0, false, WAN_SEQ_SHAPE_ID, WAN_MIXED_WINDOW_PATTERN),
                    "Wan mixed window must be accepted");
    }
    ok &= Check(Accepted(2, WAN_SEQUENCE, false, 0, false, WAN_SEQ_SHAPE_ID, UNIFORM_333_WINDOW_PATTERN),
                "Wan batch-two fp16 case must be accepted");

    ok &= Check(!Accepted(2, WAN_SEQUENCE, true, 0, false, WAN_SEQ_SHAPE_ID, UNIFORM_333_WINDOW_PATTERN),
                "Wan batch-two bf16 must be rejected");
    ok &= Check(!Accepted(1, FASTVIDEO_SEQUENCE, false, 256, true, FASTVIDEO_SEQ_SHAPE_ID, SOURCE_WINDOW_PATTERN),
                "textLength=256 is outside the fixed task cases");
    ok &= Check(!Accepted(1, FASTVIDEO_SEQUENCE, false, FASTVIDEO_TEXT_LENGTH, false, FASTVIDEO_SEQ_SHAPE_ID,
                          SOURCE_WINDOW_PATTERN),
                "FastVideo without text must be rejected");
    ok &= Check(!Accepted(1, STEPVIDEO_SEQUENCE, false, 0, false, STEPVIDEO_SEQ_SHAPE_ID, WAN_MIXED_WINDOW_PATTERN),
                "StepVideo mixed Wan window must be rejected");
    ok &= Check(!IsSupportedNormalizedSignature(1, 16, STEPVIDEO_SEQUENCE, SUPPORTED_HEAD_DIM, false, 0, false,
                                                STEPVIDEO_SEQ_SHAPE_ID, UNIFORM_333_WINDOW_PATTERN),
                "N=16 must be rejected");
    ok &= Check(!IsSupportedNormalizedSignature(1, SUPPORTED_HEADS, STEPVIDEO_SEQUENCE, 128, false, 0, false,
                                                STEPVIDEO_SEQ_SHAPE_ID, UNIFORM_333_WINDOW_PATTERN),
                "D=128 must be rejected");

    int64_t seqShape = 0;
    int64_t windowPattern = 0;
    const int64_t runtimeSignature = EncodeRuntimeSignature(FASTVIDEO_SEQ_SHAPE_ID, SOURCE_WINDOW_PATTERN_ID);
    ok &= Check(DecodeRuntimeSignature(runtimeSignature, seqShape, windowPattern) &&
                    seqShape == FASTVIDEO_SEQ_SHAPE_ID && windowPattern == SOURCE_WINDOW_PATTERN_ID,
                "runtime signature round trip must pass");
    ok &= Check(!DecodeRuntimeSignature(0, seqShape, windowPattern), "zero runtime signature must fail");
    WindowArray broadcastWindows{};
    for (int64_t head = 0; head < SUPPORTED_HEADS; ++head) {
        const int64_t offset = head * WINDOW_VALUES_PER_HEAD;
        broadcastWindows[offset] = 3;
        broadcastWindows[offset + 1] = 3;
        broadcastWindows[offset + 2] = 3;
    }
    ok &= Check(WindowsEqual(broadcastWindows, UNIFORM_333_WINDOW_PATTERN),
                "single [3,3,3] window must normalize to the fixed eight-head pattern");

    if (!ok) {
        return EXIT_FAILURE;
    }
    std::cout << "ALL_SLIDING_TILE_ATTENTION_CASE_GUARD_TESTS_PASSED signatures=17 cases=18\n";
    return EXIT_SUCCESS;
}
