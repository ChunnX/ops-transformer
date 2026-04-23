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
 * \file grouped_matmul_swiglu_quant_v2_basic_tiling.h
 * \brief
 */

#ifndef GROUPED_MATMUL_SWIGLU_QUANT_V2_BASIC_TILING_H
#define GROUPED_MATMUL_SWIGLU_QUANT_V2_BASIC_TILING_H

#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>
#include "../../../../grouped_matmul/op_host/op_tiling/arch35/grouped_quant_matmul_tiling.h"
#include "../../grouped_matmul_swiglu_quant_v2_host_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
#include "../grouped_matmul_swiglu_quant_v2_tiling.h"

namespace optiling {
using namespace Ops::Transformer::OpTiling;

constexpr size_t NZ_WEIGHT_DIM_LIMIT_950 = 5UL;
constexpr size_t NZ_WEIGHT_MULTI_TENSOR_DIM_950 = 4UL;
constexpr size_t NZ_ALIGN_K_950 = 16UL;
constexpr size_t NZ_ALIGN_N_950 = 32UL;

class GroupedMatmulSwigluQuantV2Tiling950 : public GroupedQmmTiling {
public:
    explicit GroupedMatmulSwigluQuantV2Tiling950(gert::TilingContext *context) : GroupedQmmTiling(context)
    {
        Reset();
    }
    ~GroupedMatmulSwigluQuantV2Tiling950() override = default;

    void Reset(gert::TilingContext *context) override
    {
        GroupedQmmTiling::Reset(context);
        Reset();
    }

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;
    void Reset() override;
    void SetKernelType();
    ge::graphStatus GetWorkspaceSize() override;

private:
    bool AnalyzeAttrs() override;
    bool AnalyzeDtype() override;
    bool AnalyzeInputs() override;
    void PrintQuantParams() override;
    bool SetQuantModeForGMMSwigluQuant(const gert::Shape &wScaleShape, const gert::Shape &xScaleShape);
    bool CheckShapeForMxQuant(const gert::Shape &x1ScaleShape, const gert::Shape &x2ScaleShape);
    bool CheckDtype();
    bool CheckDims() const;
    bool CheckNzWeightShape();
    bool IsFp4(ge::DataType dtype) const;
    bool IsFp8(ge::DataType dtype) const;
    bool IsFp4Input() const;
    bool IsFp8Input();
    bool IsWeightNz() const;
    bool AnalyzeAttrsPertoken();
    bool IsB8(ge::DataType dtype);
    bool CheckDtypePertoken();
    bool AnalyzeInputsPertoken();
    ge::graphStatus DoOpTilingPertoken();
    void PrintPertokenQuantParams();
    bool CheckCoreNum() const override;
    bool SetMKNForNzWeight(const gert::Shape &xShape, const gert::Shape &wShape);
    GMMSwigluQuantTilingDataParams tilingData_;

    const std::vector<ge::DataType> quantDtypeSupportList = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                                                             ge::DT_FLOAT4_E2M1};
    bool isWeightNz_ = false;
};
} // namespace optiling

#endif