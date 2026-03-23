/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ai_infra_mhc_post_grad_tiling.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>

namespace optiling {
BEGIN_TILING_DATA_DEF(AiInfraMhcPostGradTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalItems);
TILING_DATA_FIELD_DEF(uint32_t, itemsPerCore);
TILING_DATA_FIELD_DEF(uint32_t, remainderItems);
TILING_DATA_FIELD_DEF(uint32_t, usedCores);
TILING_DATA_FIELD_DEF(uint32_t, S);
TILING_DATA_FIELD_DEF(uint32_t, n);
TILING_DATA_FIELD_DEF(uint32_t, D);
TILING_DATA_FIELD_DEF(uint32_t, tileD);
TILING_DATA_FIELD_DEF(uint32_t, nTilesD);
TILING_DATA_FIELD_DEF(uint32_t, alignedD);
TILING_DATA_FIELD_DEF(uint32_t, lastTileD);
TILING_DATA_FIELD_DEF(uint32_t, alignedN);      // n aligned to 8 for float32 vector ops
TILING_DATA_FIELD_DEF(uint32_t, alignedNN);     // n*n aligned to 8 for float32 vector ops
TILING_DATA_FIELD_DEF(uint32_t, isNAligned);    // n == 8 (32 bytes for DataCopy)
TILING_DATA_FIELD_DEF(uint32_t, isNNAligned);   // (n*n) % 8 == 0
TILING_DATA_FIELD_DEF(uint32_t, isDAligned);    // D % 16 == 0 && nTilesD == 1
TILING_DATA_FIELD_DEF(uint32_t, itemsPerAic);
TILING_DATA_FIELD_DEF(uint32_t, remainderItemsAic);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(AiInfraMhcPostGrad, AiInfraMhcPostGradTilingData)

struct AiInfraMhcPostGradCompileInfo {
    uint32_t aicNum;
    uint32_t aivNum;
};
} // namespace optiling
