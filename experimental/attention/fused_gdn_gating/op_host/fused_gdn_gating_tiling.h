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
 * \file fused_gdn_gating_tiling.h
 * \brief Function-style tiling declaration for FusedGdnGating.
 */

#ifndef FUSED_GDN_GATING_TILING_H
#define FUSED_GDN_GATING_TILING_H

#include <cstdint>
#include <exe_graph/runtime/tiling_context.h>
#include <exe_graph/runtime/tiling_parse_context.h>
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(FusedGdnGatingTilingData)

    TILING_DATA_FIELD_DEF(uint32_t, numHeads);
    TILING_DATA_FIELD_DEF(float, beta);
    TILING_DATA_FIELD_DEF(uint32_t, numBatches);
    TILING_DATA_FIELD_DEF(uint32_t, rowsPerIter);
    TILING_DATA_FIELD_DEF(uint32_t, useBulkDma);
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, alignedLength);
    TILING_DATA_FIELD_DEF(uint32_t, tailLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileRows);
    TILING_DATA_FIELD_DEF(float, inv_beta);
    TILING_DATA_FIELD_DEF(float, threshold);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FusedGdnGating, FusedGdnGatingTilingData)

struct FusedGdnGatingCompileInfo {
    uint32_t coreNum;
    uint64_t ubSizePlatForm;
};

ge::graphStatus FusedGdnGatingTilingFunc(gert::TilingContext *context);

ge::graphStatus TilingPrepareForFusedGdnGating(gert::TilingParseContext *context);

} // namespace optiling

#endif // FUSED_GDN_GATING_TILING_H
