/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file grouped_matmul_swiglu_quant_v2_weight_quant_asw_tiling.h
 * \brief
 */

#ifndef GROUPED_MATMUL_SWIGLU_QUANT_V2_WEIGHT_QUANT_ASW_TILING_H
#define GROUPED_MATMUL_SWIGLU_QUANT_V2_WEIGHT_QUANT_ASW_TILING_H

#include <graph/utils/type_utils.h>
#include <vector>
#include <string>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "../grouped_matmul_swiglu_quant_v2_tiling.h"
#include "grouped_matmul_swiglu_quant_v2_weight_quant_checker.h"
#include "../../../op_kernel/arch35/weight_quant_asw/grouped_matmul_swiglu_quant_v2_weight_quant_asw_tiling_data.h"

namespace optiling {
using namespace Ops::Transformer::OpTiling;

class GroupedMatmulSwigluQuantV2WeightQuantASWTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit GroupedMatmulSwigluQuantV2WeightQuantASWTiling(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
        Reset();
    }
    ~GroupedMatmulSwigluQuantV2WeightQuantASWTiling() override = default;

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

    GMMSQWeightQuantChecker::GMMSQWeightQuantInputParams inputParams_;

private:
    const GMMSwigluV2CompileInfo *compileInfoPtr_ = nullptr;
    GMMSQArch35Tiling::GMMSQWeightQuantAswTilingData tilingData_;
};
} // namespace optiling

#endif // GROUPED_MATMUL_SWIGLU_QUANT_V2_WEIGHT_QUANT_ASW_TILING_H