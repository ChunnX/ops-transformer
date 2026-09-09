/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file roi_extractor_tiling.h
 * \brief RoiExtractor tiling data definition (mirror of the RoiExtractorTiling
 *        dataclass in op_kernel/roi_extractor.py; field order/type/layout must
 *        stay identical — the build force-includes the pypto-generated header).
 */

#ifndef ROI_EXTRACTOR_TILING_H_
#define ROI_EXTRACTOR_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(RoiExtractorTilingData)
TILING_DATA_FIELD_DEF(int64_t, r)
TILING_DATA_FIELD_DEF(int64_t, n)
TILING_DATA_FIELD_DEF(int64_t, c)
TILING_DATA_FIELD_DEF(int64_t, l)
TILING_DATA_FIELD_DEF(int64_t, pooled_h)
TILING_DATA_FIELD_DEF(int64_t, pooled_w)
TILING_DATA_FIELD_DEF(int64_t, sample_num)
TILING_DATA_FIELD_DEF(int64_t, finest_scale)
TILING_DATA_FIELD_DEF(int64_t, aligned)
TILING_DATA_FIELD_DEF(int64_t, is_fp16)
TILING_DATA_FIELD_DEF_ARR(int64_t, 8, feat_h)
TILING_DATA_FIELD_DEF_ARR(int64_t, 8, feat_w)
TILING_DATA_FIELD_DEF_ARR(float, 8, spatial_scale)
TILING_DATA_FIELD_DEF_ARR(float, 9, b_pow2)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(RoiExtractor, RoiExtractorTilingData)

constexpr int64_t ROI_EXTRACTOR_MAX_LEVEL = 8; // canndev MAX_NUM_FEATURE_MAP
constexpr int64_t ROI_EXTRACTOR_CH_BLOCK = 64; // kernel CH_BLOCK

struct RoiExtractorCompileInfo {
    static ge::graphStatus ParamCheck(gert::TilingContext *context);
};

} // namespace optiling

#endif // ROI_EXTRACTOR_TILING_H_
