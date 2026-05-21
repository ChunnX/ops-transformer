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
 * \file grouped_mat_mul_allto_allv_tiling.h
 * \brief 重构后A3/A5通用tiling结构体，产出QuantGmmA2avTilingData(量化参数=0)
 */
#ifndef MC2_GROUPED_MATMUL_ALLTO_ALLV_TILING_STRUCT_H
#define MC2_GROUPED_MATMUL_ALLTO_ALLV_TILING_STRUCT_H

#include "grouped_mat_mul_allto_allv_tiling_base.h"
#include "../../quant_grouped_mat_mul_allto_allv/op_host/op_tiling/quant_grouped_mat_mul_allto_allv_tiling_base.h"
#include "../../quant_grouped_mat_mul_allto_allv/op_kernel/arch35/quant_grouped_mat_mul_allto_allv_tiling.h"

namespace optiling {
class GmmAlltoAllvTilingStruct : public GmmAlltoAllvTilingBase {
public:
    explicit GmmAlltoAllvTilingStruct(gert::TilingContext *context) : GmmAlltoAllvTilingBase(context) {};

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    ge::graphStatus CheckAndSetInputOutputInfo();
    ge::graphStatus CheckAndSetLocalParams();
    ge::graphStatus CheckAndSetLocalParamsGmm();
    ge::graphStatus CheckAndSetLocalParamsMm();
    ge::graphStatus CheckAndSetLocalParamsAttr();
    ge::graphStatus CheckParamsRelationGmmTransShape();
    ge::graphStatus CheckParamsRelationGmm();
    ge::graphStatus CheckParamsRelationMmTransShape();
    ge::graphStatus CheckParamsRelationMm();
    ge::graphStatus CheckParamsAttrEpAndSetLocalParams();
    ge::graphStatus CheckAndSetSendRecvCountsAttr();
    ge::graphStatus CheckLocalParams();
    ge::graphStatus CheckParamsRelationAndSetLocalParams();
    ge::graphStatus SetTilingCommonInfo();
    ge::graphStatus SetGmmA2avWorkspaceInfo();
    ge::graphStatus DoGMMTiling();
    ge::graphStatus SetHcclTiling();
    ge::graphStatus CheckShapeDimensions(const gert::StorageShape *shape, uint64_t dims, const char *shapeName);
    bool IsContains(const std::vector<uint32_t> &list, uint32_t value);

    QuantGmmA2avTilingData localTilingData_;
    uint32_t libApiWorkSpaceSize_{0};
    uint32_t workSpaceSize_{0};
    uint64_t gmmResultLen_{0};

    uint64_t A_ = 0;
    uint64_t H1_ = 0;
    uint64_t ep_ = 0;
    uint64_t BsK_ = 0;
    uint64_t N1_ = 0;
    uint64_t Bs_ = 0;
    uint64_t H2_ = 0;
    uint64_t N2_ = 0;
    uint64_t epWorldSize_ = 0;
    uint64_t aivCoreNum_ = 0;
    uint64_t aicCoreNum_ = 0;
    uint64_t gmmWeightDim1_ = 0;
    uint64_t gmmWeightDim2_ = 0;
    uint64_t mmWeightDim0_ = 0;
    uint64_t mmWeightDim1_ = 0;
    bool hasSharedMm_ = false;
    bool isGmmWeightTrans_ = false;
    bool isMmWeightTrans_ = false;
    ge::DataType gmmXDtype_ = ge::DT_UNDEFINED;
    ge::DataType gmmYDtype_ = ge::DT_UNDEFINED;
    ge::DataType yDtype_ = ge::DT_UNDEFINED;
};
} // namespace optiling
#endif