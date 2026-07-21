/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * ...
 */

#ifndef ROTARY_POSITION_EMBEDDING3D_TILING_DATA_H
#define ROTARY_POSITION_EMBEDDING3D_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

struct RotaryPositionEmbedding3dTilingData {
    int64_t totalLength;
    int64_t headDim;
    int64_t seqLen;
    int64_t T;          // video frames
    int64_t H;          // video height tokens
    int64_t W;          // video width tokens
    int64_t tBand;      // full temporal band dim (d0)
    int64_t hBand;      // full height band dim (d1)
    int64_t wBand;      // full width band dim (d2)
    int64_t tileNum;
    int64_t blockLength;
    float freqBase;
    float rT;           // freqBase^(-2/tBand)
    float rH;           // freqBase^(-2/hBand)
    float rW;           // freqBase^(-2/wBand)
};
#endif
