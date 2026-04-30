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
 * \file weight_quant_batch_matmul_v2_tiling_registry.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "weight_quant_batch_matmul_v2_tiling_custom.h"
#include "weight_quant_batch_matmul_v2_tiling_custom_nz_splitk.h"
#include "weight_quant_batch_matmul_v2_tiling_fixpipe.h"
#include "weight_quant_batch_matmul_v2_tiling_msd.h"
#include "weight_quant_batch_matmul_v2_tiling_msd_group.h"
#include "weight_quant_batch_matmul_v2_tiling_splitk.h"
#include "weight_quant_batch_matmul_v2_weight_nz_tiling.h"
#include "op_cache_tiling.h"
#include "platform/platform_infos_def.h"

using Ops::NN::Optiling::TilingRegistry;
using Ops::NN::TilingPrepareForOpCache;

namespace optiling {

constexpr int32_t SPLIT_K_PRIORITY = 0;
constexpr int32_t MSD_GROUP_PRIORITY = 1;
constexpr int32_t MSD_PRIORITY = 2;
constexpr int32_t CUSTOM_SPLITK_PRIORITY = 3;
constexpr int32_t CUSTOM_PRIORITY = 4;
constexpr int32_t FIXPIPE_PRIORITY = 5;
constexpr int32_t WEIGHT_NZ_PRIORITY = 6;

REGISTER_TILING_TEMPLATE("WeightQuantBatchMatmulV2", WeightQuantBatchMatmulV2TilingSplitK, SPLIT_K_PRIORITY);
REGISTER_TILING_TEMPLATE("WeightQuantBatchMatmulV2", WeightQuantBatchMatmulV2TilingMsdGroup, MSD_GROUP_PRIORITY);
REGISTER_TILING_TEMPLATE("WeightQuantBatchMatmulV2", WeightQuantBatchMatmulV2Msd, MSD_PRIORITY);
REGISTER_TILING_TEMPLATE("WeightQuantBatchMatmulV2", WeightQuantBatchMatmulV2CustomNzSplitK, CUSTOM_SPLITK_PRIORITY);
REGISTER_TILING_TEMPLATE("WeightQuantBatchMatmulV2", WeightQuantBatchMatmulV2TilingCustom, CUSTOM_PRIORITY);
REGISTER_TILING_TEMPLATE("WeightQuantBatchMatmulV2", WeightQuantBatchMatmulV2TilingFixpipe, FIXPIPE_PRIORITY);
REGISTER_TILING_TEMPLATE("WeightQuantBatchMatmulV2", WeightQuantBatchMatmulV2WeightNz, WEIGHT_NZ_PRIORITY);

static ge::graphStatus WeightQuantBatchMatmulV2TilingFunc(gert::TilingContext* context)
{
    platform_ascendc::SocVersion socVersion;
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "WeightQuantBatchMatmulV2", "tilingContext is null");
    auto compileInfoPtr = reinterpret_cast<const WeightQuantBatchMatmulV2CompileInfo*>(context->GetCompileInfo());
    if (compileInfoPtr == nullptr) {
        auto platformInfoPtr = context->GetPlatformInfo();
        OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "platformInfoPtr is null");
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        socVersion = ascendcPlatform.GetSocVersion();
    } else {
        socVersion = compileInfoPtr->socVersion;
    }
    OP_TILING_CHECK(
        CheckPara(context, socVersion) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "WeightQuantBatchMatMul para is illegal"),
        return ge::GRAPH_FAILED);
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

IMPL_OP_OPTILING(WeightQuantBatchMatmulV2)
    .Tiling(WeightQuantBatchMatmulV2TilingFunc);
} // namespace optiling
