/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GROUPED_MATMUL_ACTIVATION_QUANT_TILING_H
#define GROUPED_MATMUL_ACTIVATION_QUANT_TILING_H

#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>
#include "../../../../grouped_matmul/op_host/op_tiling/arch35/grouped_quant_basic_api_matmul_tiling.h"
#include "../../../../grouped_matmul/op_kernel/arch35/grouped_matmul_tiling_data_apt.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
// BlockMmadMx consumes compact low-level BasicAPI tiling fields directly.
// TCubeTiling is only needed when the kernel uses the high-level matmul tiling adapter path.
using QuantBasicApiMMTiling = GroupedMatmulTilingData::QuantBasicApiMMTiling;

BEGIN_TILING_DATA_DEF(GMMActivationQuantParams)
TILING_DATA_FIELD_DEF(uint32_t, groupNum);
TILING_DATA_FIELD_DEF(uint8_t, groupListType);
TILING_DATA_FIELD_DEF(uint8_t, activationType);
TILING_DATA_FIELD_DEF(uint8_t, quantDtype);
TILING_DATA_FIELD_DEF(uint8_t, roundMode);
TILING_DATA_FIELD_DEF(uint8_t, dequantDtype);
TILING_DATA_FIELD_DEF(uint8_t, scaleAlg);
TILING_DATA_FIELD_DEF(uint32_t, rowLen);
TILING_DATA_FIELD_DEF(uint32_t, ubAvail);
TILING_DATA_FIELD_DEF(float, dstTypeMax);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GMMActivationQuantParamsOp, GMMActivationQuantParams)

BEGIN_TILING_DATA_DEF(GMMActivationQuantTilingDataParams)
TILING_DATA_FIELD_DEF_STRUCT(GMMActivationQuantParams, gmmActivationQuantParams);
TILING_DATA_FIELD_DEF_STRUCT(QuantBasicApiMMTiling, mmTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupedMatmulActivationQuant_0, GMMActivationQuantTilingDataParams)
REGISTER_TILING_DATA_CLASS(GroupedMatmulActivationQuant_1, GMMActivationQuantTilingDataParams)

class GroupedMatmulActivationQuantTiling950 : public GroupedQmmBasicApiTiling {
public:
    explicit GroupedMatmulActivationQuantTiling950(gert::TilingContext *context) : GroupedQmmBasicApiTiling(context)
    {
        Reset();
    }
    ~GroupedMatmulActivationQuantTiling950() override = default;

    void Reset(gert::TilingContext *context) override
    {
        GroupedQmmBasicApiTiling::Reset(context);
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
    static constexpr uint8_t DEFAULT_ROUND_MODE_RINT = 4;
    static constexpr uint8_t DEFAULT_SCALE_ALG_OCP = 0;
    static constexpr uint8_t DEFAULT_ACTIVATION_TYPE_GELU_TANH = 2;
    static constexpr float DEFAULT_DST_TYPE_MAX = 0.0f;

    bool AnalyzeAttrs() override;
    bool AnalyzeDtype() override;
    bool AnalyzeInputs() override;
    bool CheckCoreNum() const override;
    bool CheckDtype() const;
    bool CheckMxScaleShape(const gert::Shape &xScaleShape, const gert::Shape &wScaleShape) const;
    bool CheckWeightNzShape(const gert::Shape &wStorageShape) const;
    bool IsFp8(ge::DataType dtype) const;

    GMMActivationQuantTilingDataParams tilingData_;
    uint8_t roundMode_ = DEFAULT_ROUND_MODE_RINT;
    uint8_t scaleAlg_ = DEFAULT_SCALE_ALG_OCP;
    uint8_t activationType_ = DEFAULT_ACTIVATION_TYPE_GELU_TANH;
    float dstTypeMax_ = DEFAULT_DST_TYPE_MAX;
    ge::DataType quantDtype_ = ge::DT_FLOAT8_E4M3FN;
};
} // namespace optiling

#endif
