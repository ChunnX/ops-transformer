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
 * \file quant_batch_matmul_v4_tiling_registry.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "quant_batch_matmul_v4_msd_tiling.h"
#include "quant_batch_matmul_v4_perblock_tiling.h"
#include "quant_batch_matmul_v4_pergroup_tiling.h"
#include "quant_batch_matmul_v4_compile_info.h"
#include "common/error_util.h"
#include "platform/platform_infos_def.h"

namespace optiling {
using Ops::NN::Optiling::TilingRegistry;

constexpr int32_t MSD_PRIORITY = 2;
constexpr int32_t PERBLOCK_PRIORITY = 3;
constexpr int32_t PERGROUP_PRIORITY = 4;

REGISTER_TILING_TEMPLATE("QuantBatchMatmulV4", QuantBatchMatmulV4MsdTiling, MSD_PRIORITY);
REGISTER_TILING_TEMPLATE("QuantBatchMatmulV4", QuantBatchMatmulV4PerblockTiling, PERBLOCK_PRIORITY);
REGISTER_TILING_TEMPLATE("QuantBatchMatmulV4", QuantBatchMatmulV4PergroupTiling, PERGROUP_PRIORITY);


ge::graphStatus QuantBatchMatmulV4TilingFunc(gert::TilingContext *context)
{
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "QuantBatchMatmulV4", "tilingContext is null");
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "platformInfoPtr is null");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    auto socVersion = ascendcPlatform.GetSocVersion();

    // 910B/910_93: use MSD, Perblock, Pergroup strategies
    if (socVersion == platform_ascendc::SocVersion::ASCEND910B ||
            socVersion == platform_ascendc::SocVersion::ASCEND910_93) {
        std::vector<int32_t> registerList = {MSD_PRIORITY, PERBLOCK_PRIORITY, PERGROUP_PRIORITY};
        auto ret = TilingRegistry::GetInstance().DoTilingImpl(context, registerList);
        return ret;
    }
    OP_LOGE(context->GetNodeName(), "Unsupported platform for QuantBatchMatmulV4");
    return ge::GRAPH_FAILED;
}

IMPL_OP_OPTILING(QuantBatchMatmulV4)
    .Tiling(QuantBatchMatmulV4TilingFunc);
} // namespace optiling
