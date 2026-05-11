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
 * \file grouped_mat_mul_allto_allv_tiling_common.h
 * \brief
 */

#ifndef GROUPED_MAT_MUL_ALLTO_ALLV_TILING_COMMON_H
#define GROUPED_MAT_MUL_ALLTO_ALLV_TILING_COMMON_H

#include "securec.h"
#include "../grouped_mat_mul_allto_allv_tiling_base.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "tiling/tiling_api.h"
#include "op_host/op_tiling/new_mc2_tiling_utils.h"
#include "../../../../3rd/grouped_matmul/op_tiling/gmm_qbmm_tiling.h"
#include "../../../../3rd/grouped_matmul/op_tiling/grouped_matmul_host_util.h"
#include "../../../../3rd/grouped_matmul/op_tiling/grouped_matmul_tiling.h"
#include "../../../op_kernel/arch35/grouped_mat_mul_allto_allv_tiling.h"
#include "register/tilingdata_base.h"

namespace optiling {

using namespace AscendC;
using namespace Mc2GroupedMatmulTiling::GmmConstant;
using Mc2GroupedMatmulTiling::Mc2GroupedQbmmTiling;
using Mc2GroupedMatmulTiling::QuantMode;

struct GmmAlltoAllvParamsInfo {
    const char *opName = nullptr;
    ge::DataType gmmXDtype;
    ge::DataType gmmWeightDtype;
    ge::DataType yDtype;
    ge::DataType gmmYDtype;
    ge::DataType mmXDtype;
    ge::DataType mmWeightDtype;
    ge::DataType mmYDtype;

    uint64_t A = 0;
    uint64_t H1 = 0;
    uint64_t ep = 0;
    uint64_t gmmWeightDim1 = 0;
    uint64_t gmmWeightDim2 = 0;
    uint64_t BsK = 0;
    uint64_t N1 = 0;
    uint64_t Bs = 0;
    uint64_t H2 = 0;
    uint64_t mmWeightDim0 = 0;
    uint64_t mmWeightDim1 = 0;
    uint64_t N2 = 0;

    bool hasSharedMm = false;
    bool isGmmWeightTrans = false;
    bool isMmWeightTrans = false;
    uint64_t epWorldSize = 0;
    uint64_t groupSize = 0;

    uint64_t aivCoreNum = 0;
    uint64_t aicCoreNum = 0;
};

struct TilingInferredInfo {
    uint64_t gmmResultLen = 0UL;
    uint64_t commLen = 0UL;
    uint64_t permuteLen = 0UL;
    uint32_t biasLen = 0UL;
};

class GroupedMatmulAllToAllvTilingCommon : public GmmAlltoAllvTilingBase {
public:
    explicit GroupedMatmulAllToAllvTilingCommon(gert::TilingContext *context) : GmmAlltoAllvTilingBase(context)
    {
    }

    ~GroupedMatmulAllToAllvTilingCommon() override = default;

    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

    void PrintGmmA2avTilingData(GmmA2avTilingData &outTilingData);

protected:
    ge::graphStatus CheckAndSetInputOutputInfo();
    ge::graphStatus CheckAndSetLocalParams();
    ge::graphStatus CheckParamsRelationAndSetLocalParams();
    ge::graphStatus DoOpTiling();
    ge::graphStatus SetTilingCommonInfo();
    ge::graphStatus SetGmmA2avWorkspaceInfo();
    ge::graphStatus DoGMMTiling();
    ge::graphStatus SetHcclTiling();
    ge::graphStatus GetPlatformInfo() override;

    bool IsContains(const std::vector<uint32_t> &list, uint32_t value);
    ge::graphStatus CheckShapeDimensions(const gert::StorageShape *shape, uint64_t dims, const char *shapeName);

    GmmAlltoAllvParamsInfo localParams_;
    GmmA2avTilingData localTilingData_;
    TilingInferredInfo inferredInfo_;
    uint64_t workSpaceSize_ = 0;
    uint64_t libApiWorkSpaceSize_ = 0;
    const char *opName_ = nullptr;
};

void PrintGmmA2avWorkspaceInfo(const GmmA2avWorkspaceInfo &workspaceInfo, const char *opName_);
void PrintTaskTilingInfo(const MC2KernelTemplate::TaskTilingInfo &taskTilingInfo,
                         const GmmAlltoAllvParamsInfo &localParams, const char *opName_);
void PrintGMMTilingData(const MC2KernelTemplate::GMMQuantTilingData &data, const char *opName_);

} // namespace optiling

#endif // GROUPED_MAT_MUL_ALLTO_ALLV_TILING_COMMON_H