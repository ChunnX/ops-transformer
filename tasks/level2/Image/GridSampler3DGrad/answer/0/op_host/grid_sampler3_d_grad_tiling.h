/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grid_sampler3_d_grad_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_GRID_SAMPLER_3D_GRAD_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_GRID_SAMPLER_3D_GRAD_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"

#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGI(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) { \
        std::printf("nullptr error!"); \
        return ge::GRAPH_FAILED; \
    }
#define unlikely(x) __builtin_expect((x), 0)
#define OP_CHECK_IF(condition, log, return_expr) \
    do { \
        if (unlikely(condition)) { \
            log; \
            return_expr; \
        } \
    } while (0)

namespace optiling {

struct InputParamsInfo {
    uint32_t batch = 0;
    uint32_t channel = 0;
    uint32_t xD = 0;
    uint32_t xH = 0;
    uint32_t xW = 0;
    uint32_t gridD = 0;
    uint32_t gridH = 0;
    uint32_t gridW = 0;
    uint32_t tilingKey = 0;
    uint32_t interpolation = 0;
    uint32_t padding = 0;
    uint32_t alignCorners = 0;
    bool regBase = false;
};

constexpr uint16_t MAX_CORE_CONT = 50;

BEGIN_TILING_DATA_DEF(GridSampler3DGradTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, channel);
TILING_DATA_FIELD_DEF(uint32_t, xD);
TILING_DATA_FIELD_DEF(uint32_t, xH);
TILING_DATA_FIELD_DEF(uint32_t, xW);
TILING_DATA_FIELD_DEF(uint32_t, gridD);
TILING_DATA_FIELD_DEF(uint32_t, gridH);
TILING_DATA_FIELD_DEF(uint32_t, gridW);
TILING_DATA_FIELD_DEF(uint32_t, interpolation);
TILING_DATA_FIELD_DEF(uint32_t, padding);
TILING_DATA_FIELD_DEF(uint32_t, alignCorners);
TILING_DATA_FIELD_DEF(uint32_t, blockNum);
TILING_DATA_FIELD_DEF(uint32_t, pNumPerCore);
TILING_DATA_FIELD_DEF(uint32_t, tailPNum);
TILING_DATA_FIELD_DEF(uint32_t, group);
TILING_DATA_FIELD_DEF(uint32_t, ubFactorElement);
TILING_DATA_FIELD_DEF(uint32_t, isDeterministic);
TILING_DATA_FIELD_DEF(uint32_t, tailBNum);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GridSampler3DGrad, GridSampler3DGradTilingData)
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_GRID_SAMPLER_3D_GRAD_TILING_H
