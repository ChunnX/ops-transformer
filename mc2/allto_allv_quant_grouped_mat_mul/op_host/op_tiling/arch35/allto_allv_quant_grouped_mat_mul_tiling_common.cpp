/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You can not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file allto_allv_quant_grouped_mat_mul_tiling_common.cpp
 * \brief V1 only overrides SetHcclTiling to fix CCU mode
 */

#include "allto_allv_quant_grouped_mat_mul_tiling_common.h"
#include "mc2/allto_allv_quant_grouped_mat_mul_v2/op_host/op_tiling/arch35/allto_allv_quant_grouped_mat_mul_v2_tiling_common.h"

using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;

namespace optiling {

ge::graphStatus AlltoAllvQuantGmmTilingCommon::SetHcclTiling() const
{
    uint32_t alltoAllvCmd = 8U;
    std::string alltoAllvConfig = "AlltoAll=level0:fullmesh;level1:pairwise";

    const uint32_t alltoAllvReduceType = 0u;
    mc2tiling::HcclDataType alltoallvHcclDataType = mc2tiling::HcclDataType::HCCL_DATA_TYPE_RESERVED;
    if (gmmXDataType_ == ge::DataType::DT_FLOAT4_E2M1) {
        alltoallvHcclDataType = mc2tiling::HcclDataType::HCCL_DATA_TYPE_INT8;
    } else {
        alltoallvHcclDataType = mc2tiling::ConvertGeTypeToHcclType(context_->GetNodeName(), gmmXDataType_);
    }
    OP_TILING_CHECK(alltoallvHcclDataType == mc2tiling::HcclDataType::HCCL_DATA_TYPE_RESERVED,
        OP_LOGE(context_->GetNodeName(), "Ge datatype gmmXDataType(%s) is not found in HCCL_DATA_TYPE.",
        ge::TypeUtils::DataTypeToSerialString(gmmXDataType_).c_str()), return ge::GRAPH_FAILED);
    uint8_t alltoAllvDataType = static_cast<uint8_t>(alltoallvHcclDataType);

    Mc2CcTilingConfig hcclCcTilingConfig(group_, alltoAllvCmd, alltoAllvConfig, alltoAllvReduceType, alltoAllvDataType,
        alltoAllvDataType);
    // V1 has no comm_mode attr, fixed to use CCU (default)
    // Do NOT call SetCommEngine, use default CCU mode

    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(tilingData->hcclA2avTilingInfo.hcclInitTiling) != 0,
        OP_LOGE(context_->GetNodeName(),
        "mc2CcTilingConfig mc2tiling GetTiling hcclA2avTilingInfo.hcclInitTiling failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(tilingData->hcclA2avTilingInfo.a2avCcTiling) != 0,
        OP_LOGE(context_->GetNodeName(),
        "mc2CcTilingConfig mc2tiling GetTiling hcclA2avTilingInfo.a2avCcTiling failed"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling