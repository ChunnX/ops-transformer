/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Device-side tiling data for the Ascend 950 MXFP8 weight-NZ V3 path.
 *
 * The field order and types must stay byte-for-byte compatible with
 * op_host/op_tiling/grouped_matmul_swiglu_quant_weight_nz_v3_tiling.h.
 * This is deliberately a plain POD definition: device compilation cannot
 * depend on the host-side TilingDef registration classes.
 */

#ifndef GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_V3_TILING_DATA_H
#define GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_V3_TILING_DATA_H

#include <cstdint>

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(push, 8)
struct GMMSwigluQuantV3Params {
    uint32_t groupNum = 0;
    uint8_t groupListType = 0;
    uint8_t quantDtype = 0;
    uint8_t isMxWeightNzMultiTensor = 0;
    uint8_t dequantDtype = 0;
    int64_t swigluMode = 2;
    float clampLimit = 7.0F;
    float gluAlpha = 1.702F;
    float gluBias = 1.0F;
    uint8_t scaleAlg = 0;
    uint8_t reserved0 = 0;
    uint16_t reserved1 = 0;
    float dstTypeMax = 0.0F;
    uint8_t roundMode = 0;
    uint8_t reserved2 = 0;
    uint16_t reserved3 = 0;
};

struct GMMSwigluQuantV3TilingDataParams {
    GMMSwigluQuantV3Params gmmSwigluQuantParams;
    TCubeTiling mmTilingData;
};
#pragma pack(pop)

#endif // GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_V3_TILING_DATA_H
