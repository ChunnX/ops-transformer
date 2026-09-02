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

#include <cstdint>
#include <vector>

namespace sliding_tile_attention {

constexpr int64_t SUPPORTED_RANK = 4;
constexpr int64_t MIN_HEADS = 1;
constexpr int64_t MAX_HEADS = 128;
constexpr int64_t HEAD_DIM_ALIGNMENT = 8;
constexpr int64_t WINDOW_VALUES_PER_HEAD = 3;
constexpr int64_t MAX_WINDOW_VALUE_COUNT = MAX_HEADS * WINDOW_VALUES_PER_HEAD;
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
constexpr int64_t FASTVIDEO_MAX_TEXT_LENGTH = 512;
constexpr int64_t FASTVIDEO_SEQUENCE = FASTVIDEO_IMAGE_SEQUENCE + FASTVIDEO_TEXT_LENGTH;
constexpr int64_t STEPVIDEO_SEQUENCE = STEPVIDEO_CANVAS_TIME * STEPVIDEO_CANVAS_HEIGHT * STEPVIDEO_CANVAS_WIDTH;
constexpr int64_t WAN_SEQUENCE = WAN_CANVAS_TIME * WAN_CANVAS_HEIGHT * WAN_CANVAS_WIDTH;
constexpr int64_t FASTVIDEO_SEQ_SHAPE_ID = 1;
constexpr int64_t STEPVIDEO_SEQ_SHAPE_ID = 2;
constexpr int64_t WAN_SEQ_SHAPE_ID = 3;

using WindowValues = std::vector<int64_t>;

inline bool IsSupportedHeads(int64_t heads)
{
    return heads >= MIN_HEADS && heads <= MAX_HEADS;
}

inline bool IsSupportedHeadDim(int64_t headDim)
{
    return headDim > 0 && headDim % HEAD_DIM_ALIGNMENT == 0;
}

inline bool IsValidWindowValue(int64_t value)
{
    return value > 0 && (value & 1) != 0;
}

inline bool IsSupportedSeqShape(int64_t seqShape)
{
    return seqShape >= FASTVIDEO_SEQ_SHAPE_ID && seqShape <= WAN_SEQ_SHAPE_ID;
}

inline bool IsValidSequenceMetadata(int64_t sequence, int64_t textLength, bool hasText, int64_t seqShape)
{
    switch (seqShape) {
        case FASTVIDEO_SEQ_SHAPE_ID:
            return hasText && textLength >= 0 && textLength <= FASTVIDEO_MAX_TEXT_LENGTH &&
                   sequence >= FASTVIDEO_IMAGE_SEQUENCE &&
                   sequence - FASTVIDEO_IMAGE_SEQUENCE <= FASTVIDEO_MAX_TEXT_LENGTH &&
                   textLength <= sequence - FASTVIDEO_IMAGE_SEQUENCE;
        case STEPVIDEO_SEQ_SHAPE_ID:
            return sequence == STEPVIDEO_SEQUENCE && textLength == 0 && !hasText;
        case WAN_SEQ_SHAPE_ID:
            return sequence == WAN_SEQUENCE && textLength == 0 && !hasText;
        default:
            return false;
    }
}

inline bool NormalizeWindows(const int64_t *values, int64_t valueCount, int64_t heads, WindowValues &windows)
{
    if (values == nullptr || !IsSupportedHeads(heads)) {
        return false;
    }
    const int64_t normalizedCount = heads * WINDOW_VALUES_PER_HEAD;
    if (valueCount != WINDOW_VALUES_PER_HEAD && valueCount != normalizedCount) {
        return false;
    }
    windows.resize(static_cast<size_t>(normalizedCount));
    for (int64_t head = 0; head < heads; ++head) {
        for (int64_t axis = 0; axis < WINDOW_VALUES_PER_HEAD; ++axis) {
            const int64_t value =
                values[valueCount == WINDOW_VALUES_PER_HEAD ? axis : head * WINDOW_VALUES_PER_HEAD + axis];
            if (!IsValidWindowValue(value)) {
                return false;
            }
            windows[static_cast<size_t>(head * WINDOW_VALUES_PER_HEAD + axis)] = value;
        }
    }
    return true;
}

inline bool GetCanvasMetadata(int64_t seqShape, int64_t &canvasT, int64_t &canvasH, int64_t &canvasW)
{
    switch (seqShape) {
        case FASTVIDEO_SEQ_SHAPE_ID:
            canvasT = FASTVIDEO_CANVAS_TIME;
            canvasH = FASTVIDEO_CANVAS_HEIGHT;
            canvasW = FASTVIDEO_CANVAS_WIDTH;
            return true;
        case STEPVIDEO_SEQ_SHAPE_ID:
            canvasT = STEPVIDEO_CANVAS_TIME;
            canvasH = STEPVIDEO_CANVAS_HEIGHT;
            canvasW = STEPVIDEO_CANVAS_WIDTH;
            return true;
        case WAN_SEQ_SHAPE_ID:
            canvasT = WAN_CANVAS_TIME;
            canvasH = WAN_CANVAS_HEIGHT;
            canvasW = WAN_CANVAS_WIDTH;
            return true;
        default:
            return false;
    }
}

inline bool IsSupportedMetadata(int64_t batch, int64_t heads, int64_t sequence, int64_t headDim, int64_t textLength,
                                bool hasText, int64_t seqShape, const WindowValues &windows)
{
    if (batch <= 0 || !IsSupportedHeads(heads) || !IsSupportedHeadDim(headDim) || !IsSupportedSeqShape(seqShape) ||
        static_cast<int64_t>(windows.size()) != heads * WINDOW_VALUES_PER_HEAD) {
        return false;
    }
    for (int64_t value : windows) {
        if (!IsValidWindowValue(value)) {
            return false;
        }
    }
    return IsValidSequenceMetadata(sequence, textLength, hasText, seqShape);
}

} // namespace sliding_tile_attention

#endif
