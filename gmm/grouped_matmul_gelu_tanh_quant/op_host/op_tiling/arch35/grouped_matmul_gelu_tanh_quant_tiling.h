/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GROUPED_MATMUL_GELU_TANH_QUANT_TILING_H
#define GROUPED_MATMUL_GELU_TANH_QUANT_TILING_H

#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>
#include "../../../../grouped_matmul/op_host/op_tiling/arch35/grouped_quant_matmul_tiling.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GMMGeluTanhQuantParams)
TILING_DATA_FIELD_DEF(uint32_t, groupNum);
TILING_DATA_FIELD_DEF(uint8_t, groupListType);
TILING_DATA_FIELD_DEF(uint8_t, quantDtype);
TILING_DATA_FIELD_DEF(uint8_t, roundMode);
TILING_DATA_FIELD_DEF(uint8_t, dequantDtype);
TILING_DATA_FIELD_DEF(uint8_t, scaleAlg);
TILING_DATA_FIELD_DEF(uint32_t, rowLen);
TILING_DATA_FIELD_DEF(uint32_t, ubAvail);
TILING_DATA_FIELD_DEF(float, dstTypeMax);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GMMGeluTanhQuantParamsOp, GMMGeluTanhQuantParams)

BEGIN_TILING_DATA_DEF(GMMGeluTanhQuantTilingDataParams)
TILING_DATA_FIELD_DEF_STRUCT(GMMGeluTanhQuantParams, gmmGeluTanhQuantParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupedMatmulGeluTanhQuant_0, GMMGeluTanhQuantTilingDataParams)
REGISTER_TILING_DATA_CLASS(GroupedMatmulGeluTanhQuant_1, GMMGeluTanhQuantTilingDataParams)

class GroupedMatmulGeluTanhQuantTiling950 : public GroupedQmmTiling {
public:
    explicit GroupedMatmulGeluTanhQuantTiling950(gert::TilingContext *context) : GroupedQmmTiling(context)
    {
        Reset();
    }
    ~GroupedMatmulGeluTanhQuantTiling950() override = default;

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
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void Reset() override;

private:
    bool AnalyzeAttrs() override;
    bool AnalyzeDtype() override;
    bool AnalyzeInputs() override;
    bool CheckCoreNum() const override;
    bool CheckDtype() const;
    bool CheckMxScaleShape(const gert::Shape &xScaleShape, const gert::Shape &wScaleShape) const;
    bool CheckWeightNzShape(const gert::Shape &wStorageShape) const;
    bool IsFp8(ge::DataType dtype) const;
    void FillMmTilingData();

    GMMGeluTanhQuantTilingDataParams tilingData_;
    uint8_t roundMode_ = 4;
    uint8_t scaleAlg_ = 0;
    float dstTypeMax_ = 0.0f;
    ge::DataType quantDtype_ = ge::DT_FLOAT8_E4M3FN;
};
} // namespace optiling

#endif
