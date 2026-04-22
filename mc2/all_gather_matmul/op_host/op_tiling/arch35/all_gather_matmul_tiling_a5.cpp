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
 * \file all_gather_matmul_tiling_a5.cpp
 * \brief
 */
#include "../all_gather_matmul_tiling_common.h"
#include "all_gather_formulaic_tiling_a5.h"

namespace optiling {
AllGatherPlusMM* GetTileFormulate(const mc2tiling::TilingArgs& args, uint32_t inputRankDim,
        KernelType inputKernelType)
{
    SocVersion inputSocVersion = SocVersion::SOC950;
    AllGatherPlusMMA5* tileFormulate =  new AllGatherPlusMMA5(args, args.rankDim, inputKernelType, inputSocVersion);
    return tileFormulate;
}

ge::graphStatus CheckValidRank(Mc2Tiling::AllGatherMatmulTilingData* tilingData,
        const std::map<uint32_t, std::vector<uint32_t>>&VALID_RANK, gert::TilingContext *context)
{
    auto it = std::find(VALID_RANK.at(0).begin(),
    VALID_RANK.at(0).end(), rankSize);
    OP_TILING_CHECK(it == VALID_RANK.at(0).end(),
    VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(),
    "world_size value is %u, which is illegal.", rankSize), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void SetSocParam(Mc2Tiling::AllGatherMatmulTilingData* tilingData, const char* group)
{
    tilingData->socParam.isA3 = 0U;
    tilingData->socParam.isStep = 0U;   
    tilingData->socParam.isND2NZ = 1U; 
}

std::string GetAlgConfig(Mc2Tiling::AllGatherMatmulTilingData* tilingData)
{
    return "AllGather=level0:fullmesh";
}
}