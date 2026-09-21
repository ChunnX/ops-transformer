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
 * \file grouped_matmul_swiglu_quant_weight_nz_v3_tiling.h
 * \brief Ascend 950 MXFP8 weight-NZ tiling data for GroupedMatmulSwigluQuantV3.
 */

#ifndef GROUPED_MATMUL_SWIGLU_QUANT_V3_TILING_H
#define GROUPED_MATMUL_SWIGLU_QUANT_V3_TILING_H

#include <cstdint>
#include <cmath>
#include <string>

#include "op_host/tiling_base.h"
#include "tiling/tiling_api.h"
#include "grouped_matmul_swiglu_quant_v2_tiling.h"
#include "../../../grouped_matmul/op_host/op_tiling/arch35/grouped_quant_matmul_tiling.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GMMSwigluQuantV3Params)
TILING_DATA_FIELD_DEF(uint32_t, groupNum);
TILING_DATA_FIELD_DEF(uint8_t, groupListType);
TILING_DATA_FIELD_DEF(uint8_t, quantDtype);
TILING_DATA_FIELD_DEF(uint8_t, isMxWeightNzMultiTensor);
TILING_DATA_FIELD_DEF(uint8_t, dequantDtype);
TILING_DATA_FIELD_DEF(int64_t, swigluMode);
TILING_DATA_FIELD_DEF(float, clampLimit);
TILING_DATA_FIELD_DEF(float, gluAlpha);
TILING_DATA_FIELD_DEF(float, gluBias);
TILING_DATA_FIELD_DEF(uint8_t, scaleAlg);
TILING_DATA_FIELD_DEF(uint8_t, reserved0);
TILING_DATA_FIELD_DEF(uint16_t, reserved1);
TILING_DATA_FIELD_DEF(float, dstTypeMax);
TILING_DATA_FIELD_DEF(uint8_t, roundMode);
TILING_DATA_FIELD_DEF(uint8_t, reserved2);
TILING_DATA_FIELD_DEF(uint16_t, reserved3);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GMMSwigluQuantV3ParamsOp, GMMSwigluQuantV3Params)

BEGIN_TILING_DATA_DEF(GMMSwigluQuantV3TilingDataParams)
TILING_DATA_FIELD_DEF_STRUCT(GMMSwigluQuantV3Params, gmmSwigluQuantParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupedMatmulSwigluQuantV2_32, GMMSwigluQuantV3TilingDataParams)
REGISTER_TILING_DATA_CLASS(GroupedMatmulSwigluQuantV2_33, GMMSwigluQuantV3TilingDataParams)
REGISTER_TILING_DATA_CLASS(GroupedMatmulSwigluQuantV2_48, GMMSwigluQuantV3TilingDataParams)
REGISTER_TILING_DATA_CLASS(GroupedMatmulSwigluQuantV2_49, GMMSwigluQuantV3TilingDataParams)

namespace GroupedMatmulSwigluQuantV3Tiling {
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t X_SCALE_INDEX = 1;
constexpr uint32_t GROUPLIST_INDEX = 2;
constexpr uint32_t WEIGHT_INDEX = 3;
constexpr uint32_t WEIGHT_SCALE_INDEX = 4;
constexpr uint32_t WEIGHT_ASSIST_MATRIX_INDEX = 5;
constexpr uint32_t BIAS_INDEX = 6;
constexpr uint32_t SMOOTH_SCALE_INDEX = 7;
constexpr uint32_t Y_INDEX = 0;
constexpr uint32_t Y_SCALE_INDEX = 1;

constexpr uint32_t ATTR_INDEX_DEQUANT_MODE = 0;
constexpr uint32_t ATTR_INDEX_DEQUANT_DTYPE = 1;
constexpr uint32_t ATTR_INDEX_QUANT_MODE = 2;
constexpr uint32_t ATTR_INDEX_QUANT_DTYPE = 3;
constexpr uint32_t ATTR_INDEX_TRANSPOSE_WEIGHT = 4;
constexpr uint32_t ATTR_INDEX_GROUPLIST_TYPE = 5;
constexpr uint32_t ATTR_INDEX_TUNING_CONFIG = 6;
constexpr uint32_t ATTR_INDEX_SWIGLU_MODE = 7;
constexpr uint32_t ATTR_INDEX_CLAMP_LIMIT = 8;
constexpr uint32_t ATTR_INDEX_GLU_ALPHA = 9;
constexpr uint32_t ATTR_INDEX_GLU_BIAS = 10;
constexpr uint32_t ATTR_INDEX_ROUND_MODE = 11;
constexpr uint32_t ATTR_INDEX_SCALE_ALG = 12;
constexpr uint32_t ATTR_INDEX_DST_TYPE_MAX = 13;

constexpr int64_t MX_QUANT_MODE = 2;
constexpr int64_t FLOAT_DEQUANT_DTYPE = 0;
constexpr int64_t MX_GROUP_SIZE_V3 = 64;
constexpr int64_t NZ_C0_SIZE = 32;
constexpr int64_t NZ_K0_SIZE = 16;
constexpr int64_t SWIGLU_SPLIT = 2;
// This iteration intentionally exposes only the mode-2 SwiGLU formula.
constexpr int64_t SWIGLU_MODE = 2;
constexpr uint8_t SCALE_ALG_OCP = 0;
constexpr uint8_t SCALE_ALG_CUBLAS = 1;
constexpr float DEFAULT_DST_TYPE_MAX = 0.0F;
constexpr char DEFAULT_ROUND_MODE[] = "rint";
constexpr int64_t MAX_GROUP_NUM = 1024;
constexpr size_t SINGLE_TENSOR_COUNT = 1;
constexpr size_t X_DIM_NUM = 2;
constexpr size_t X_SCALE_DIM_NUM = 3;
constexpr size_t WEIGHT_VIEW_DIM_NUM = 3;
constexpr size_t WEIGHT_STORAGE_DIM_NUM = 5;
constexpr size_t WEIGHT_SCALE_DIM_NUM = 4;
constexpr size_t GROUP_LIST_DIM_NUM = 1;
constexpr size_t OUTPUT_DIM_NUM = 2;
constexpr size_t OUTPUT_SCALE_DIM_NUM = 3;
constexpr size_t SYS_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;

class GroupedMatmulSwigluQuantV3Tiling950 : public GroupedQmmTiling {
public:
    explicit GroupedMatmulSwigluQuantV3Tiling950(gert::TilingContext *context)
        : GroupedQmmTiling(context)
    {
        Reset();
    }
    ~GroupedMatmulSwigluQuantV3Tiling950() override = default;

    void Reset(gert::TilingContext *context) override
    {
        GroupedQmmTiling::Reset(context);
        Reset();
    }

protected:
    ge::graphStatus GetPlatformInfo() override;
    const char *GetOpType() const override
    {
        return "GroupedMatmulSwigluQuantV2";
    }
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
    bool CheckShapeAndScale(const gert::Shape &xShape, const gert::Shape &xScaleShape, const gert::Shape &weightShape,
                            const gert::Shape &weightStorageShape, const gert::Shape &weightScaleShape,
                            const gert::Shape &groupListShape) const;
    bool CheckOutputShape() const;
    bool IsTensorApiCapable() const;
    bool CheckCoreNum() const;
    bool GetAttrInt(const gert::RuntimeAttrs *attrs, uint32_t index, int64_t &value) const;
    bool GetAttrBool(const gert::RuntimeAttrs *attrs, uint32_t index, bool &value) const;
    bool GetAttrFloat(const gert::RuntimeAttrs *attrs, uint32_t index, float &value) const;
    bool GetAttrString(const gert::RuntimeAttrs *attrs, uint32_t index, std::string &value) const;

    GMMSwigluQuantV3TilingDataParams tilingData_;
    bool transposeWeight_ = false;
    int64_t swigluMode_ = SWIGLU_MODE;
    float clampLimit_ = 7.0F;
    float gluAlpha_ = 1.702F;
    float gluBias_ = 1.0F;
    uint8_t scaleAlg_ = SCALE_ALG_OCP;
    float dstTypeMax_ = DEFAULT_DST_TYPE_MAX;
    uint8_t roundMode_ = 0;
    uint32_t aivNum_ = 0;
    bool useTensorApi_ = false;
};

} // namespace GroupedMatmulSwigluQuantV3Tiling
} // namespace optiling

#endif // GROUPED_MATMUL_SWIGLU_QUANT_V3_TILING_H
