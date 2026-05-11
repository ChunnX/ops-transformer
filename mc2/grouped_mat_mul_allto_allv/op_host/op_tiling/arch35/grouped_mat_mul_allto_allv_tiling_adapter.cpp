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
 * \file grouped_mat_mul_allto_allv_tiling_adapter.cpp
 * \brief
 */

#include "common/utils/op_mc2.h"
#include "mc2_log.h"
#include "grouped_mat_mul_allto_allv_tiling_adapter.h"

using namespace Mc2Log;
using namespace AscendC;
using namespace optiling;
using namespace Mc2GroupedMatmulTiling::GmmConstant;
using namespace optiling::Mc2GroupedMatmul;

namespace optiling {

const std::vector<uint32_t> NON_QUANT_MODE_MP = {0, 0, 0, 0, 0, 0, 0};

ge::graphStatus
Mc2GroupedMatmul::GroupedMatmulAllToAllvAdapter::SetCommonInputParams(const GmmAlltoAllvParamsInfo &params)
{
    GetPlatformInfo();
    inputParams_.opName = params.opName;
    inputParams_.kernelType = 0UL;
    inputParams_.splitItem = 2;
    inputParams_.actType = 0;
    inputParams_.aFormat = ge::FORMAT_ND;
    inputParams_.bFormat = ge::FORMAT_ND;
    inputParams_.cFormat = ge::FORMAT_ND;
    inputParams_.transA = false;
    inputParams_.hasBias = false;
    inputParams_.isSingleX = true;
    inputParams_.isSingleW = true;
    inputParams_.isSingleY = true;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus
Mc2GroupedMatmul::GroupedMatmulAllToAllvAdapter::SetGroupExpertInputParameters(const GmmAlltoAllvParamsInfo &params,
                                                                               uint64_t gmmX)
{
    inputParams_.mSize = gmmX;
    inputParams_.kSize = params.H1;
    inputParams_.nSize = params.N1;
    inputParams_.groupNum = 1;
    inputParams_.aQuantMode = QuantMode::NO_QUANT;
    inputParams_.bQuantMode = QuantMode::NO_QUANT;
    inputParams_.groupType = optiling::Mc2GroupedMatmul::SPLIT_M;
    inputParams_.groupListType = 1;
    inputParams_.aDtype = params.gmmXDtype;
    inputParams_.bDtype = params.gmmWeightDtype;
    inputParams_.cDtype = params.gmmYDtype;
    inputParams_.biasDtype = ge::DT_INT32;
    inputParams_.scaleDtype = ge::DT_FLOAT;
    inputParams_.perTokenScaleDtype = ge::DT_FLOAT;

    inputParams_.transB = params.isGmmWeightTrans;
    mList_[0] = static_cast<int32_t>(gmmX);
    kList_[0] = static_cast<int32_t>(params.H1);
    nList_[0] = static_cast<int32_t>(params.N1);
    SetKernelType();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus
Mc2GroupedMatmul::GroupedMatmulAllToAllvAdapter::SetSharedExpertInputParameters(const GmmAlltoAllvParamsInfo &params)
{
    inputParams_.mSize = params.Bs;
    inputParams_.kSize = params.H2;
    inputParams_.nSize = params.N2;
    inputParams_.groupNum = 1;
    inputParams_.aQuantMode = QuantMode::NO_QUANT;
    inputParams_.bQuantMode = QuantMode::NO_QUANT;
    inputParams_.groupType = optiling::Mc2GroupedMatmul::SPLIT_M;
    inputParams_.groupListType = 1;
    inputParams_.aDtype = params.mmXDtype;
    inputParams_.bDtype = params.mmWeightDtype;
    inputParams_.cDtype = params.mmYDtype;
    inputParams_.biasDtype = ge::DT_INT32;
    inputParams_.scaleDtype = ge::DT_FLOAT;
    inputParams_.perTokenScaleDtype = ge::DT_FLOAT;

    inputParams_.transB = params.isMmWeightTrans;
    mList_[0] = static_cast<int32_t>(params.Bs);
    kList_[0] = static_cast<int32_t>(params.H2);
    nList_[0] = static_cast<int32_t>(params.N2);
    SetKernelType();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2GroupedMatmul::GroupedMatmulAllToAllvAdapter::Process()
{
    GE_ASSERT_GRAPH_SUCCESS(DoOpTiling());
    GE_ASSERT_GRAPH_SUCCESS(DoLibApiTiling());
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling