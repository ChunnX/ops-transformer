/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#ifndef SLIDING_TILE_ATTENTION_TILING_DATA_H
#define SLIDING_TILE_ATTENTION_TILING_DATA_H

#include <cstdint>

constexpr int64_t SLIDING_TILE_ATTENTION_MAX_HEAD_NUM = 64;

struct SlidingTileAttentionTilingData {
    int64_t batchSize;
    int64_t headNum;
    int64_t seqLen;
    int64_t headDim;
    float scale;
    int64_t maxQueryRows;
    int64_t maxVisibleKvTokens;
    int64_t queryTaskNum;
    int64_t headDimAlign;
    int64_t cubeTaskNum;
    int64_t cubeWorkspacePerCore;
    int64_t scoreWorkspaceElems;
    int64_t probWorkspaceElems;
    int64_t outWorkspaceElems;
    int64_t scoreWorkspaceBytes;
    int64_t probWorkspaceBytes;
    int64_t outWorkspaceBytes;
    int64_t aicCoreNum;
    int64_t aivCoreNum;
    int64_t imageSeqLen;
    int64_t textLength;
    int64_t hasText;
    int64_t seqT;
    int64_t seqH;
    int64_t seqW;
    int64_t tileT;
    int64_t tileH;
    int64_t tileW;
    int64_t tileTokenNum;
    int64_t tileNumT;
    int64_t tileNumH;
    int64_t tileNumW;
    int64_t tileNumHW;
    int64_t imageTileNum;
    int64_t validSeqLen;
    int64_t totalQueryNum;
    int64_t windowT[SLIDING_TILE_ATTENTION_MAX_HEAD_NUM];
    int64_t windowH[SLIDING_TILE_ATTENTION_MAX_HEAD_NUM];
    int64_t windowW[SLIDING_TILE_ATTENTION_MAX_HEAD_NUM];
};

#endif // SLIDING_TILE_ATTENTION_TILING_DATA_H
