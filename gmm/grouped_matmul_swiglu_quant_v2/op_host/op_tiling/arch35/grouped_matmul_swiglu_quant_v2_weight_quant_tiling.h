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
 * \file grouped_matmul_swiglu_quant_v2_weight_quant_tiling.h
 * \brief GMMSQ MxA8W4 weight quant tiling template
 */
#ifndef GROUPED_MATMUL_SWIGLU_QUANT_V2_WEIGHT_QUANT_TILING_H
#define GROUPED_MATMUL_SWIGLU_QUANT_V2_WEIGHT_QUANT_TILING_H

#include <graph/utils/type_utils.h>
#include <vector>
#include <functional>
#include <string>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "../grouped_matmul_swiglu_quant_v2_tiling.h"
#include "../../../op_kernel/arch35/weight_quant_basic_block/grouped_matmul_swiglu_quant_v2_weight_quant_tiling_data.h"

namespace optiling {
using namespace Ops::Transformer::OpTiling;

namespace GMMSQArch35WeightQuantTiling {

enum class ScenarioType {
    NONE = 0,
    MX_A8W4_WEIGHT_NZ = 1,
};

struct GMMSQWeightQuantInputParams {
    std::string opName;
    ge::DataType xDtype = ge::DT_INT8;
    ge::DataType wDtype = ge::DT_INT8;
    ge::Format wFormat = ge::FORMAT_ND;
    bool xTrans = false;
    bool wTrans = false;
    int64_t mSize = -1;
    int64_t kSize = -1;
    int64_t nSize = -1;
    int64_t groupListType = -1;
    int64_t groupNum = -1;
    int64_t outputBS = -1;
    bool hasBias = false;
};
} // namespace GMMSQArch35WeightQuantTiling

class GroupedMatmulSwigluQuantV2WeightQuantTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit GroupedMatmulSwigluQuantV2WeightQuantTiling(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
        Reset();
    }
    ~GroupedMatmulSwigluQuantV2WeightQuantTiling() override = default;

    void Reset(gert::TilingContext *context) override
    {
        Ops::Transformer::OpTiling::TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void Reset();

    GMMSQArch35WeightQuantTiling::GMMSQWeightQuantInputParams inputParams_;

private:
    const GMMSwigluV2CompileInfo *compileInfoPtr_;
    GMMSQArch35WeightQuantTiling::ScenarioType scenarioType_ = ScenarioType::NONE;
    std::vector<std::function<bool(gert::TilingContext *context)>> checkConditionFuncs_;
    std::vector<std::function<void(gert::TilingContext *context, GMMSQWeightQuantInputParams &inputParams_)>>
        SetInputFuncs_;
    GMMSQArch35Tiling::GMMSQWeightQuantTilingData tilingData_;

    void RunSetInputFunc();
    bool SetMxA8W4NzInputFunc();
    bool SetMxA8W4NzConditionFunc();
    bool RunCheckFunc();
    bool InferScenario();
    bool AnalyzeAttrs();
    bool AnalyzeDtype();
    bool AnalyzeInputs();
    bool CheckShapeForMxQuant(const gert::Shape &wScaleShape, const gert::Shape &xScaleShape);

    int64_t coreNum_ = 0;
};
} // namespace optiling

#endif // GROUPED_MATMUL_SWIGLU_QUANT_V2_WEIGHT_QUANT_TILING_H
