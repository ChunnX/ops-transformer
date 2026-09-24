/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_swiglu_quant_swiglu_params.h
 * \brief Shared SwiGLU attributes appended to the two Ascend 950 tiling payloads.
 */

#ifndef GROUPED_MATMUL_SWIGLU_QUANT_SWIGLU_PARAMS_H
#define GROUPED_MATMUL_SWIGLU_QUANT_SWIGLU_PARAMS_H

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

#pragma pack(push, 8)
struct GMMSwigluQuantSwigluParams {
    int64_t swigluMode = 0;
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
#pragma pack(pop)

static_assert(sizeof(GMMSwigluQuantSwigluParams) == 32, "SwiGLU tiling attributes must occupy 32 bytes");

#endif // GROUPED_MATMUL_SWIGLU_QUANT_SWIGLU_PARAMS_H
