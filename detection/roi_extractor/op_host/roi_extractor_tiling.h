/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the LICENSE file in the root of the software repository for the full text of the License.
 */

/*!
 * \file roi_extractor_tiling.h
 * \brief Host-side tiling for the PyPTO Pro roi_extractor kernel.
 *
 * RoiExtractorTilingDataTmp is the registration/serialisation view; the
 * field-identical RoiExtractorTiling class is generated from the kernel's
 * Python dataclass by pypto codegen (force-include, three-side contract in
 * docs/MDE.md section 2).
 */
#ifndef ROI_EXTRACTOR_TILING_H_
#define ROI_EXTRACTOR_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(RoiExtractorTilingDataTmp)
TILING_DATA_FIELD_DEF(int64_t, numLevels)
TILING_DATA_FIELD_DEF(int64_t, roisNum)
TILING_DATA_FIELD_DEF(int64_t, batchSize)
TILING_DATA_FIELD_DEF(int64_t, channels)
TILING_DATA_FIELD_DEF(int64_t, pooledHeight)
TILING_DATA_FIELD_DEF(int64_t, pooledWidth)
TILING_DATA_FIELD_DEF(int64_t, sampleNum)
TILING_DATA_FIELD_DEF(int64_t, finestScale)
TILING_DATA_FIELD_DEF(float, roiScaleFactor)
TILING_DATA_FIELD_DEF(int64_t, aligned)
TILING_DATA_FIELD_DEF(int64_t, hasIndex)
TILING_DATA_FIELD_DEF(int64_t, colOffset)
TILING_DATA_FIELD_DEF(float, spatialScale0)
TILING_DATA_FIELD_DEF(float, spatialScale1)
TILING_DATA_FIELD_DEF(float, spatialScale2)
TILING_DATA_FIELD_DEF(float, spatialScale3)
TILING_DATA_FIELD_DEF(float, spatialScale4)
TILING_DATA_FIELD_DEF(int64_t, levelH0)
TILING_DATA_FIELD_DEF(int64_t, levelH1)
TILING_DATA_FIELD_DEF(int64_t, levelH2)
TILING_DATA_FIELD_DEF(int64_t, levelH3)
TILING_DATA_FIELD_DEF(int64_t, levelH4)
TILING_DATA_FIELD_DEF(int64_t, levelW0)
TILING_DATA_FIELD_DEF(int64_t, levelW1)
TILING_DATA_FIELD_DEF(int64_t, levelW2)
TILING_DATA_FIELD_DEF(int64_t, levelW3)
TILING_DATA_FIELD_DEF(int64_t, levelW4)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(RoiExtractor, RoiExtractorTilingDataTmp)

struct RoiExtractorCompileInfo {
    uint32_t coreNum = 50;
    uint64_t ubSize = 262144;
    bool is_ascendc = true;

    static ge::graphStatus ParamCheck(gert::TilingContext *context);
};

} // namespace optiling

#endif // ROI_EXTRACTOR_TILING_H_
