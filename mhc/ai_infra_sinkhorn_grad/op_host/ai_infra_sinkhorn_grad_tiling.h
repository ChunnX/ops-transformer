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
 * \file ai_infra_sinkhorn_grad_tiling.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>

namespace optiling {
BEGIN_TILING_DATA_DEF(AiInfraSinkhornGradTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalLength);
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, numIters);
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, perCoreElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreElements);
TILING_DATA_FIELD_DEF(int64_t, perCoreLoops);
TILING_DATA_FIELD_DEF(int64_t, perCorePerLoopElements);
TILING_DATA_FIELD_DEF(int64_t, perCoreLastLoopElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreLoops);
TILING_DATA_FIELD_DEF(int64_t, lastCorePerLoopElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreLastLoopElements);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(AiInfraSinkhornGrad, AiInfraSinkhornGradTilingData)

struct AiInfraSinkhornGradCompileInfo {
    uint32_t aicNum;
    uint32_t aivNum;
};
} // namespace optiling
