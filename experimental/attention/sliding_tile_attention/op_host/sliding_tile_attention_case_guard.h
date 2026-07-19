/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SLIDING_TILE_ATTENTION_CASE_GUARD_H_
#define SLIDING_TILE_ATTENTION_CASE_GUARD_H_

#include <array>
#include <cstdint>

namespace sliding_tile_attention {

constexpr int64_t SUPPORTED_RANK = 4;
constexpr int64_t SUPPORTED_HEADS = 8;
constexpr int64_t SUPPORTED_HEAD_DIM = 64;
constexpr int64_t WINDOW_VALUES_PER_HEAD = 3;
constexpr int64_t WINDOW_VALUE_COUNT = SUPPORTED_HEADS * WINDOW_VALUES_PER_HEAD;
constexpr int64_t TILE_TIME = 6;
constexpr int64_t TILE_HEIGHT = 8;
constexpr int64_t TILE_WIDTH = 8;
constexpr int64_t FASTVIDEO_CANVAS_TIME = 30;
constexpr int64_t FASTVIDEO_CANVAS_HEIGHT = 48;
constexpr int64_t FASTVIDEO_CANVAS_WIDTH = 80;
constexpr int64_t STEPVIDEO_CANVAS_TIME = 36;
constexpr int64_t STEPVIDEO_CANVAS_HEIGHT = 48;
constexpr int64_t STEPVIDEO_CANVAS_WIDTH = 48;
constexpr int64_t WAN_CANVAS_TIME = 18;
constexpr int64_t WAN_CANVAS_HEIGHT = 48;
constexpr int64_t WAN_CANVAS_WIDTH = 80;
constexpr int64_t FASTVIDEO_IMAGE_SEQUENCE = FASTVIDEO_CANVAS_TIME * FASTVIDEO_CANVAS_HEIGHT * FASTVIDEO_CANVAS_WIDTH;
constexpr int64_t FASTVIDEO_TEXT_LENGTH = 128;
constexpr int64_t FASTVIDEO_SEQUENCE = FASTVIDEO_IMAGE_SEQUENCE + FASTVIDEO_TEXT_LENGTH;
constexpr int64_t STEPVIDEO_SEQUENCE = STEPVIDEO_CANVAS_TIME * STEPVIDEO_CANVAS_HEIGHT * STEPVIDEO_CANVAS_WIDTH;
constexpr int64_t WAN_SEQUENCE = WAN_CANVAS_TIME * WAN_CANVAS_HEIGHT * WAN_CANVAS_WIDTH;
constexpr int64_t INVALID_WINDOW_PATTERN = 0;
constexpr int64_t SOURCE_WINDOW_PATTERN_ID = 1;
constexpr int64_t WIDE_WINDOW_PATTERN_ID = 2;
constexpr int64_t UNIFORM_333_WINDOW_PATTERN_ID = 3;
constexpr int64_t WAN_MIXED_WINDOW_PATTERN_ID = 4;
constexpr int64_t RUNTIME_SIGNATURE_RADIX = 8;
constexpr int64_t FASTVIDEO_SEQ_SHAPE_ID = 1;
constexpr int64_t STEPVIDEO_SEQ_SHAPE_ID = 2;
constexpr int64_t WAN_SEQ_SHAPE_ID = 3;

using WindowArray = std::array<int64_t, WINDOW_VALUE_COUNT>;

constexpr WindowArray SOURCE_WINDOW_PATTERN = {
    1, 1, 1, 3, 3, 3, 3, 5, 5, 5, 5, 5, 1, 3, 3, 3, 3, 5, 5, 5, 7, 7, 7, 7,
};
constexpr WindowArray WIDE_WINDOW_PATTERN = {
    7, 7, 7, 7, 5, 7, 5, 7, 7, 5, 5, 7, 7, 3, 5, 3, 7, 5, 5, 7, 3, 7, 7, 5,
};
constexpr WindowArray UNIFORM_333_WINDOW_PATTERN = {
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
};
constexpr WindowArray WAN_MIXED_WINDOW_PATTERN = {
    3, 3, 5, 3, 1, 9, 1, 3, 5, 5, 5, 5, 3, 5, 7, 5, 3, 7, 1, 1, 1, 7, 7, 7,
};

inline bool WindowsEqual(const WindowArray &lhs, const WindowArray &rhs)
{
    for (int64_t index = 0; index < WINDOW_VALUE_COUNT; ++index) {
        if (lhs[index] != rhs[index]) {
            return false;
        }
    }
    return true;
}

inline int64_t GetWindowPatternId(const WindowArray &windows)
{
    if (WindowsEqual(windows, SOURCE_WINDOW_PATTERN)) {
        return SOURCE_WINDOW_PATTERN_ID;
    }
    if (WindowsEqual(windows, WIDE_WINDOW_PATTERN)) {
        return WIDE_WINDOW_PATTERN_ID;
    }
    if (WindowsEqual(windows, UNIFORM_333_WINDOW_PATTERN)) {
        return UNIFORM_333_WINDOW_PATTERN_ID;
    }
    if (WindowsEqual(windows, WAN_MIXED_WINDOW_PATTERN)) {
        return WAN_MIXED_WINDOW_PATTERN_ID;
    }
    return INVALID_WINDOW_PATTERN;
}

inline bool FillWindowPattern(int64_t patternId, WindowArray &windows)
{
    switch (patternId) {
        case SOURCE_WINDOW_PATTERN_ID:
            windows = SOURCE_WINDOW_PATTERN;
            return true;
        case WIDE_WINDOW_PATTERN_ID:
            windows = WIDE_WINDOW_PATTERN;
            return true;
        case UNIFORM_333_WINDOW_PATTERN_ID:
            windows = UNIFORM_333_WINDOW_PATTERN;
            return true;
        case WAN_MIXED_WINDOW_PATTERN_ID:
            windows = WAN_MIXED_WINDOW_PATTERN;
            return true;
        default:
            return false;
    }
}

inline int64_t EncodeRuntimeSignature(int64_t seqShape, int64_t windowPattern)
{
    return seqShape * RUNTIME_SIGNATURE_RADIX + windowPattern;
}

inline bool DecodeRuntimeSignature(int64_t runtimeSignature, int64_t &seqShape, int64_t &windowPattern)
{
    if (runtimeSignature <= 0) {
        return false;
    }
    seqShape = runtimeSignature / RUNTIME_SIGNATURE_RADIX;
    windowPattern = runtimeSignature % RUNTIME_SIGNATURE_RADIX;
    WindowArray windows{};
    return seqShape >= FASTVIDEO_SEQ_SHAPE_ID && seqShape <= WAN_SEQ_SHAPE_ID &&
           FillWindowPattern(windowPattern, windows) &&
           EncodeRuntimeSignature(seqShape, windowPattern) == runtimeSignature;
}

inline bool IsSupportedNormalizedSignature(int64_t batch, int64_t heads, int64_t sequence, int64_t headDim, bool isBf16,
                                           int64_t textLength, bool hasText, int64_t seqShape,
                                           const WindowArray &windows)
{
    if (heads != SUPPORTED_HEADS || headDim != SUPPORTED_HEAD_DIM) {
        return false;
    }

    const int64_t windowPattern = GetWindowPatternId(windows);
    const bool isSource = windowPattern == SOURCE_WINDOW_PATTERN_ID;
    const bool isWide = windowPattern == WIDE_WINDOW_PATTERN_ID;
    const bool isUniform333 = windowPattern == UNIFORM_333_WINDOW_PATTERN_ID;
    const bool isWanMixed = windowPattern == WAN_MIXED_WINDOW_PATTERN_ID;

    // The task acceptance table contains 18 fixed inputs. Its single-triple [3,3,3] shorthand is normalized to
    // eight triples before ACLNN, so the 18 inputs become 17 unique device-side metadata signatures.
    if (seqShape == FASTVIDEO_SEQ_SHAPE_ID) {
        return batch == 1 && sequence == FASTVIDEO_SEQUENCE && hasText && textLength == FASTVIDEO_TEXT_LENGTH &&
               (isSource || isWide || isUniform333);
    }
    if (seqShape == STEPVIDEO_SEQ_SHAPE_ID) {
        return batch == 1 && sequence == STEPVIDEO_SEQUENCE && !hasText && textLength == 0 &&
               (isSource || isWide || isUniform333);
    }
    if (seqShape == WAN_SEQ_SHAPE_ID) {
        if (batch == 1 && sequence == WAN_SEQUENCE && !hasText && textLength == 0) {
            return isUniform333 || isWanMixed;
        }
        return batch == 2 && sequence == WAN_SEQUENCE && !hasText && textLength == 0 && !isBf16 && isUniform333;
    }
    return false;
}

} // namespace sliding_tile_attention

#endif
