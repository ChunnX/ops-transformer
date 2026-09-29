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
 * \file grouped_quant_basic_api_matmul_tiling.h
 * \brief
 */
#ifndef GROUPED_MATMUL_SWIGLU_QUANT_V2_HIF4_TILING_H
#define GROUPED_MATMUL_SWIGLU_QUANT_V2_HIF4_TILING_H

#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>
#include "../../../../grouped_matmul/op_host/op_tiling/arch35/grouped_quant_basic_api_matmul_tiling.h"
#include "grouped_matmul_swiglu_quant_v2_basic_tiling.h"
#include "../../grouped_matmul_swiglu_quant_v2_host_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "../grouped_matmul_swiglu_quant_v2_tiling.h"

namespace optiling {

class GroupedMatmulSwigluQuantV2Hif4Tiling : public GroupedQmmBasicApiTiling {
public:
    explicit GroupedMatmulSwigluQuantV2Hif4Tiling(gert::TilingContext *context) : GroupedQmmBasicApiTiling(context)
    {
        Reset();
    }
    ~GroupedMatmulSwigluQuantV2Hif4Tiling() override = default;

    void Reset(gert::TilingContext *context) override
    {
        GroupedQmmBasicApiTiling::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;
    void PrintQuantParams() override;
    virtual void Reset();
    
private:
    bool AnalyzeAttrs() override;
    bool AnalyzeDtype() override;
    bool AnalyzeInputs() override;
    bool CheckCoreNum() const override;
    bool CheckDtype() const;
    bool SetQuantModeForGMMSwigluQuant();
    bool IsHifp4(ge::DataType dtype) const;
    bool IsHifp4Input() const;
    bool IsHif4Scale() const;
    bool CheckQuantParamsForHif4TypeM(const gert::Shape &xScaleShape, const gert::Shape &wScaleShape) const;
    bool CheckHif4Dims() const;
    ge::graphStatus GetWorkspaceSize() override;
    template <typename TTilingData>
    ge::graphStatus SaveTilingDataToContext(const TTilingData &tilingData)
    {
        context_->SetBlockDim(aicoreParams_.aicNum);
        context_->SetScheduleMode(1);
        OP_CHECK_IF(
            tilingData_.GetDataSize() % sizeof(uint64_t) != 0,
            OP_LOGE(context_->GetNodeName(), "Tiling data size[%zu] is not aligned to 8", tilingData_.GetDataSize()),
            return ge::GRAPH_FAILED);
        tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }
    void LogQuantParams(const GMMSwigluQuantParams &params) const;
    GMMSwigluQuantBasicApiTilingDataParams tilingData_;
};
} // namespace optiling

#endif // GROUPED_QUANT_BASIC_API_MATMUL_TILING_H
