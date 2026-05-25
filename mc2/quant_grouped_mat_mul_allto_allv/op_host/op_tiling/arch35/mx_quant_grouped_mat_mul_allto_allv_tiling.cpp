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
 * \file mx_quant_grouped_mat_mul_allto_allv_tiling.cpp
 * \brief
 */

#include "op_mc2.h"
#include "mc2_log.h"
#include "mx_quant_grouped_mat_mul_allto_allv_tiling.h"
#include "quant_grouped_mat_mul_allto_allv_tiling_adapter.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include <tiling/tiling_api.h>
#include "mc2_comm_utils.h"
#include <numeric>

using namespace Mc2Log;
using namespace AscendC;
using namespace Mc2Tiling;
using namespace Mc2Tiling::Mc2GroupedMatmul;

namespace Mc2Tiling {

const std::vector<uint32_t> MX_QUANT_GMM_X_DTYPE_LIST = {
    ge::DT_FLOAT8_E5M2,
    ge::DT_FLOAT8_E4M3FN,
    ge::DT_FLOAT4_E2M1,
};
const std::vector<uint32_t> MX_QUANT_GMM_WEIGHT_DTYPE_LIST = {
    ge::DT_FLOAT8_E5M2,
    ge::DT_FLOAT8_E4M3FN,
    ge::DT_FLOAT4_E2M1,
};
const std::vector<uint32_t> MX_QUANT_GMM_X_SCALE_DTYPE_LIST = {
    ge::DT_FLOAT8_E8M0,
};
const std::vector<uint32_t> MX_QUANT_GMM_WEIGHT_SCALE_DTYPE_LIST = {
    ge::DT_FLOAT8_E8M0,
};
const std::vector<uint32_t> MX_QUANT_GMM_Y_DTYPE_LIST = {
    ge::DT_FLOAT16,
    ge::DT_BF16,
};
const std::set<int64_t> SUPPORT_RANK_SIZE{2, 4, 8, 16, 32, 64, 128, 256};
constexpr int64_t RANK_DEFAULT_NUM = -1;

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::MxCheckShapeDimensions(const gert::StorageShape *shape,
                                                                            uint64_t dims, const char *shapeName)
{
    uint64_t dimNum = shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK((dimNum != dims),
                    OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, shapeName,
                        (std::to_string(dimNum) + "D").c_str(), (std::to_string(dims) + "D").c_str()),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool MxQuantGroupedMatmulAllToAllvTiling::IsCapable()
{
    QuantModePair mode = GetQuantMode(context_, opName_);
    OP_TILING_CHECK(mode == QUANT_PAIR_ERROR,
                    OP_LOGE_WITH_INVALID_ATTR(opName_, "quant mode", std::to_string(static_cast<int>(mode)).c_str(),
                                              "valid quant mode"),
                    return false);
    OP_LOGD(opName_, "QuantMode=%d, expected MX mode=%d", mode, QUANT_PAIR_MX);
    if (mode == QUANT_PAIR_MX) {
        OP_LOGD(opName_, "MxQuantGroupedMatmulAllToAllvTiling MX mode capable.");
        return true;
    }
    OP_LOGD(opName_, "Skip MxQuantGroupedMatmulAllToAllvTiling, mode mismatch.");
    return false;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::CheckAndSetLocalParamsGmm()
{
    localParams_.gmmXDtype = context_->GetInputDesc(GMM_X_INDEX)->GetDataType();
    localParams_.gmmWeightDtype = context_->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType();
    OP_TILING_CHECK(
        !IsContains(MX_QUANT_GMM_X_DTYPE_LIST, localParams_.gmmXDtype),
        OP_LOGE_FOR_INVALID_DTYPE(opName_, "gmmX"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        !IsContains(MX_QUANT_GMM_WEIGHT_DTYPE_LIST, localParams_.gmmWeightDtype),
        OP_LOGE_FOR_INVALID_DTYPE(opName_, "gmmWeight"),
        return ge::GRAPH_FAILED);
    localParams_.yDtype = context_->GetOutputDesc(OUTPUT_Y_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(MX_QUANT_GMM_Y_DTYPE_LIST, localParams_.yDtype),
                    OP_LOGE_FOR_INVALID_DTYPE(opName_, "y"),
                    return ge::GRAPH_FAILED);
    localParams_.gmmYDtype = localParams_.yDtype;

    const gert::StorageShape *gmmXStorageShape = context_->GetInputShape(GMM_X_INDEX);
    OP_TILING_CHECK(gmmXStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "gmmXStorageShape"),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *gmmWeightStorageShape = context_->GetInputShape(GMM_WEIGHT_INDEX);
    OP_TILING_CHECK(gmmWeightStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "gmmWeightStorageShape"),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *yStorageShape = context_->GetOutputShape(OUTPUT_Y_INDEX);
    OP_TILING_CHECK(yStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "yStorageShape"),
                    return ge::GRAPH_FAILED);

    auto status = MxCheckShapeDimensions(gmmXStorageShape, DIM_TWO, "gmmXShape");
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    status = MxCheckShapeDimensions(gmmWeightStorageShape, DIM_THREE, "gmmWeightShape");
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    status = MxCheckShapeDimensions(yStorageShape, DIM_TWO, "yShape");
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    localParams_.A = gmmXStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.H1 = gmmXStorageShape->GetStorageShape().GetDim(DIM_ONE);

    localParams_.ep = gmmWeightStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.BsK = yStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.N1 = yStorageShape->GetStorageShape().GetDim(DIM_ONE);

    localParams_.gmmWeightDim1 = gmmWeightStorageShape->GetStorageShape().GetDim(DIM_ONE);
    localParams_.gmmWeightDim2 = gmmWeightStorageShape->GetStorageShape().GetDim(DIM_TWO);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::CheckAndSetLocalParamsMm()
{
    if (!localParams_.hasSharedMm) {
        return ge::GRAPH_SUCCESS;
    }
    localParams_.mmXDtype = context_->GetOptionalInputDesc(MM_X_OPTIONAL_INDEX)->GetDataType();
    localParams_.mmWeightDtype = context_->GetOptionalInputDesc(MM_WEIGHT_OPTIONAL_INDEX)->GetDataType();
    OP_TILING_CHECK(
        !IsContains(MX_QUANT_GMM_X_DTYPE_LIST, localParams_.mmXDtype),
        OP_LOGE_FOR_INVALID_DTYPE(opName_, "mmX"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        !IsContains(MX_QUANT_GMM_WEIGHT_DTYPE_LIST, localParams_.mmWeightDtype),
        OP_LOGE_FOR_INVALID_DTYPE(opName_, "mmWeight"),
        return ge::GRAPH_FAILED);
    localParams_.mmYDtype = context_->GetOutputDesc(OUTPUT_MM_Y_OPTIONAL_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(MX_QUANT_GMM_Y_DTYPE_LIST, localParams_.mmYDtype),
                    OP_LOGE_FOR_INVALID_DTYPE(opName_, "mmY"),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *mmXStorageShape = context_->GetOptionalInputShape(MM_X_OPTIONAL_INDEX);
    OP_TILING_CHECK(mmXStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "mmXStorageShape"),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *mmWeightStorageShape = context_->GetOptionalInputShape(MM_WEIGHT_OPTIONAL_INDEX);
    OP_TILING_CHECK(mmWeightStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "mmWeightStorageShape"),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *mmYStorageShape = context_->GetOutputShape(OUTPUT_MM_Y_OPTIONAL_INDEX);
    OP_TILING_CHECK(mmYStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "mmYStorageShape"),
                    return ge::GRAPH_FAILED);

    auto status = MxCheckShapeDimensions(mmXStorageShape, DIM_TWO, "mmXShape");
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    status = MxCheckShapeDimensions(mmWeightStorageShape, DIM_TWO, "mmWeightShape");
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    status = MxCheckShapeDimensions(mmYStorageShape, DIM_TWO, "mmYShape");
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    localParams_.Bs = mmXStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.H2 = mmXStorageShape->GetStorageShape().GetDim(DIM_ONE);

    uint64_t mmYDim0 = mmYStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    OP_TILING_CHECK(localParams_.Bs != mmYDim0,
                    OP_LOGE_FOR_INVALID_SHAPE(opName_, "mmX and mmY"),
                    return ge::GRAPH_FAILED);

    localParams_.N2 = mmYStorageShape->GetStorageShape().GetDim(DIM_ONE);

    localParams_.mmWeightDim0 = mmWeightStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.mmWeightDim1 = mmWeightStorageShape->GetStorageShape().GetDim(DIM_ONE);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::CheckParamsRelationGmm()
{
    auto gmmXScaleDesc = context_->GetRequiredInputDesc(GMM_X_SCALE_INDEX);
    auto gmmWeightScaleDesc = context_->GetRequiredInputDesc(GMM_WEIGHT_SCALE_INDEX);
    OP_TILING_CHECK(gmmXScaleDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "gmmXScaleDesc"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gmmWeightScaleDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "gmmWeightScaleDesc"),
                    return ge::GRAPH_FAILED);

    localParams_.gmmXScaleDtype = gmmXScaleDesc->GetDataType();
    localParams_.gmmWeightScaleDtype = gmmWeightScaleDesc->GetDataType();
    OP_TILING_CHECK(!IsContains(MX_QUANT_GMM_X_SCALE_DTYPE_LIST, localParams_.gmmXScaleDtype),
                    OP_LOGE_FOR_INVALID_DTYPE(opName_, "gmmXScale"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsContains(MX_QUANT_GMM_WEIGHT_SCALE_DTYPE_LIST, localParams_.gmmWeightScaleDtype),
                    OP_LOGE_FOR_INVALID_DTYPE(opName_, "gmmWeightScale"),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *gmmXScaleStorageShape = context_->GetRequiredInputShape(GMM_X_SCALE_INDEX);
    const gert::StorageShape *gmmWeightScaleStorageShape = context_->GetRequiredInputShape(GMM_WEIGHT_SCALE_INDEX);
    OP_TILING_CHECK(gmmXScaleStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "gmmXScaleStorageShape"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gmmWeightScaleStorageShape == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(opName_, "gmmWeightScaleStorageShape"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(localParams_.gmmXQuantMode != QUANT_MX,
                    OP_LOGE_FOR_INVALID_VALUE(opName_, "gmmXQuantMode", localParams_.gmmXQuantMode),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(localParams_.gmmWeightQuantMode != QUANT_MX,
                    OP_LOGE_FOR_INVALID_VALUE(opName_, "gmmWeightQuantMode", localParams_.gmmWeightQuantMode),
                    return ge::GRAPH_FAILED);
    ge::graphStatus status = MxCheckShapeDimensions(gmmXScaleStorageShape, DIM_THREE, "gmmXScaleShape");
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = MxCheckShapeDimensions(gmmWeightScaleStorageShape, DIM_FOUR, "gmmWeightScaleShape");
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);

    localParams_.gmmQuantSuit = QUANT_PAIR_MX;
    if (!localParams_.isGmmWeightTrans) {
        OP_TILING_CHECK(localParams_.H1 != localParams_.gmmWeightDim1,
                        OP_LOGE_FOR_INVALID_SHAPE(opName_, "gmmX and gmmWeight (non-transposed)"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(localParams_.N1 != localParams_.gmmWeightDim2,
                        OP_LOGE_FOR_INVALID_SHAPE(opName_, "y and gmmWeight (non-transposed)"),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(localParams_.H1 != localParams_.gmmWeightDim2,
                        OP_LOGE_FOR_INVALID_SHAPE(opName_, "gmmX and gmmWeight (transposed)"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(localParams_.N1 != localParams_.gmmWeightDim1,
                        OP_LOGE_FOR_INVALID_SHAPE(opName_, "y and gmmWeight (transposed)"),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::CheckParamsRelationMm()
{
    if (!localParams_.hasSharedMm) {
        return ge::GRAPH_SUCCESS;
    }
    auto mmXScaleDesc = context_->GetOptionalInputDesc(MM_X_SCALE_OPTIONAL_INDEX);
    auto mmWeightScaleDesc = context_->GetOptionalInputDesc(MM_WEIGHT_SCALE_OPTIONAL_INDEX);
    OP_TILING_CHECK(mmXScaleDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "mmXScaleDesc"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mmWeightScaleDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "mmWeightScaleDesc"),
                    return ge::GRAPH_FAILED);

    localParams_.mmXScaleDtype = mmXScaleDesc->GetDataType();
    localParams_.mmWeightScaleDtype = mmWeightScaleDesc->GetDataType();
    OP_TILING_CHECK(!IsContains(MX_QUANT_GMM_X_SCALE_DTYPE_LIST, localParams_.mmXScaleDtype),
                    OP_LOGE_FOR_INVALID_DTYPE(opName_, "mmXScale"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsContains(MX_QUANT_GMM_WEIGHT_SCALE_DTYPE_LIST, localParams_.mmWeightScaleDtype),
                    OP_LOGE_FOR_INVALID_DTYPE(opName_, "mmWeightScale"),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *mmXScaleStorageShape = context_->GetOptionalInputShape(MM_X_SCALE_OPTIONAL_INDEX);
    const gert::StorageShape *mmWeightScaleStorageShape =
        context_->GetOptionalInputShape(MM_WEIGHT_SCALE_OPTIONAL_INDEX);
    OP_TILING_CHECK(mmXScaleStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "mmXScaleStorageShape"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mmWeightScaleStorageShape == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(opName_, "mmWeightScaleStorageShape"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(localParams_.mmXQuantMode != QUANT_MX,
                    OP_LOGE_FOR_INVALID_VALUE(opName_, "mmXQuantMode", localParams_.mmXQuantMode),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(localParams_.mmWeightQuantMode != QUANT_MX,
                    OP_LOGE_FOR_INVALID_VALUE(opName_, "mmWeightQuantMode", localParams_.mmWeightQuantMode),
                    return ge::GRAPH_FAILED);
    ge::graphStatus status = MxCheckShapeDimensions(mmXScaleStorageShape, DIM_THREE, "mmXScaleShape");
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = MxCheckShapeDimensions(mmWeightScaleStorageShape, DIM_THREE, "mmWeightScaleShape");
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);

    localParams_.mmQuantSuit = QUANT_PAIR_MX;
    if (!localParams_.isMmWeightTrans) {
        OP_TILING_CHECK(localParams_.H2 != localParams_.mmWeightDim0,
                        OP_LOGE_FOR_INVALID_SHAPE(opName_, "mmX and mmWeight (non-transposed)"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(localParams_.N2 != localParams_.mmWeightDim1,
                        OP_LOGE_FOR_INVALID_SHAPE(opName_, "mmY and mmWeight (non-transposed)"),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(localParams_.H2 != localParams_.mmWeightDim1,
                        OP_LOGE_FOR_INVALID_SHAPE(opName_, "mmX and mmWeight (transposed)"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(localParams_.N2 != localParams_.mmWeightDim0,
                        OP_LOGE_FOR_INVALID_SHAPE(opName_, "mmY and mmWeight (transposed)"),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::CheckParamsAttrEpAndSetLocalParams()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "attrs"), return ge::GRAPH_FAILED);
    const char *group = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    OP_TILING_CHECK(group == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "group"), return ge::GRAPH_FAILED);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "epWorldSize"),
                    return ge::GRAPH_FAILED);

    int64_t rankDim = 0;
    std::string supportRankSizeRange;
    for (const auto &v : SUPPORT_RANK_SIZE) {
        supportRankSizeRange += (std::to_string(v) + " ");
    }
    if (*epWorldSizePtr == RANK_DEFAULT_NUM) {
        OP_TILING_CHECK(!mc2tiling::GetRankSize(opName_, group, rankDim), OP_LOGE(opName_, "GetRankSize failed."),
                        return ge::GRAPH_FAILED);
    } else {
        rankDim = *epWorldSizePtr;
    }
    OP_TILING_CHECK(
        SUPPORT_RANK_SIZE.find(rankDim) == SUPPORT_RANK_SIZE.end(),
        OP_LOGE_FOR_INVALID_VALUE(opName_, "epWorldSize", rankDim),
        return ge::GRAPH_FAILED);
    localParams_.epWorldSize = rankDim;

    bool isEpNumMatch = (localParams_.ep > 0) && (localParams_.ep < 33);
    if (!isEpNumMatch) {
        OP_LOGE_FOR_INVALID_VALUE(opName_, "ep", std::to_string(localParams_.ep).c_str(), "(0, 32]");
        return ge::GRAPH_FAILED;
    } else {
        uint64_t expertNum = localParams_.ep * rankDim;
        OP_TILING_CHECK(expertNum > MAX_EXPERT_NUM,
                        OP_LOGE_FOR_INVALID_VALUE(opName_, "expertNum", expertNum),
                        return ge::GRAPH_FAILED);
    }

    auto groupSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_GROUP_SIZE_OPTIONAL_INDEX);
    OP_TILING_CHECK(groupSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "groupSize"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*groupSizePtr < 0, OP_LOGE_FOR_INVALID_VALUE(opName_, "groupSize", *groupSizePtr),
                    return ge::GRAPH_FAILED);

    localParams_.groupSize = *groupSizePtr;
    uint64_t groupSizeK = static_cast<uint64_t>(*groupSizePtr) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeN = (static_cast<uint64_t>(*groupSizePtr) >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeM = (static_cast<uint64_t>(*groupSizePtr) >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;

    OP_TILING_CHECK(((groupSizeM != MX_GROUP_SIZE_M) && (groupSizeM != 0)) ||
                        ((groupSizeN != MX_GROUP_SIZE_N) && (groupSizeN != 0)) ||
                        ((groupSizeK != MX_GROUP_SIZE_K) && (groupSizeK != 0)),
                    OP_LOGE_FOR_INVALID_VALUE(opName_, "groupSize", *groupSizePtr),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::CheckMxQuantDtypeConstraints()
{
    if (!localParams_.hasSharedMm) {
        return ge::GRAPH_SUCCESS;
    }
    OP_TILING_CHECK(localParams_.gmmXDtype != localParams_.mmXDtype,
                    OP_LOGE_FOR_INVALID_DTYPE(opName_, "mmX (should match gmmX)"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(localParams_.gmmWeightDtype != localParams_.mmWeightDtype,
                    OP_LOGE_FOR_INVALID_DTYPE(opName_, "mmWeight (should match gmmWeight)"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        localParams_.yDtype != localParams_.mmYDtype,
        OP_LOGE_FOR_INVALID_DTYPE(opName_, "mmY (should match y)"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::CheckMxQuantGmmScaleShapes()
{
    bool TransGmmWeightFlag = false;
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "attrs"), return ge::GRAPH_FAILED);
    const bool *isTransGmmWeight = attrs->GetAttrPointer<bool>(ATTR_TRANS_GMM_WEIGHT_INDEX);
    if (isTransGmmWeight) {
        TransGmmWeightFlag = *isTransGmmWeight;
    }
    const gert::StorageShape *gmmXScaleShape = context_->GetRequiredInputShape(GMM_X_SCALE_INDEX);
    const gert::StorageShape *gmmWeightScaleShape = context_->GetRequiredInputShape(GMM_WEIGHT_SCALE_INDEX);
    OP_TILING_CHECK((gmmXScaleShape == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName_, "gmmXScaleShape"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((gmmWeightScaleShape == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName_, "gmmWeightScale"),
                    return ge::GRAPH_FAILED);

    uint64_t gmmXScaleDim0 = gmmXScaleShape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t gmmXScaleDim1 = gmmXScaleShape->GetStorageShape().GetDim(DIM_ONE);
    uint64_t gmmXScaleDim2 = gmmXScaleShape->GetStorageShape().GetDim(DIM_TWO);
    uint64_t gmmWeightScaleDim0 = gmmWeightScaleShape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t gmmWeightScaleDim1 = gmmWeightScaleShape->GetStorageShape().GetDim(DIM_ONE);
    uint64_t gmmWeightScaleDim2 = gmmWeightScaleShape->GetStorageShape().GetDim(DIM_TWO);
    uint64_t gmmWeightScaleDim3 = gmmWeightScaleShape->GetStorageShape().GetDim(DIM_THREE);

    uint64_t gmmxDivH1 = (localParams_.H1 + MX_SCALE_GROUP - 1) / MX_SCALE_GROUP;
    OP_TILING_CHECK(
        (localParams_.A != gmmXScaleDim0) || (gmmxDivH1 != gmmXScaleDim1) || (gmmXScaleDim2 != EVEN_ALIGN),
        OP_LOGE_FOR_INVALID_SHAPE(opName_, "gmmXScale"),
        return ge::GRAPH_FAILED);

    if (localParams_.isGmmWeightTrans) { // Transposed Scenario
        OP_TILING_CHECK((gmmWeightScaleDim0 != localParams_.ep) || (gmmWeightScaleDim1 != localParams_.N1) ||
                            (gmmWeightScaleDim2 != gmmxDivH1) || (gmmWeightScaleDim3 != EVEN_ALIGN),
                        OP_LOGE_FOR_INVALID_SHAPE(opName_, "gmmWeightScale (transposed)"),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(
            (gmmWeightScaleDim0 != localParams_.ep) || (gmmWeightScaleDim1 != gmmxDivH1) ||
                (gmmWeightScaleDim2 != localParams_.N1) || (gmmWeightScaleDim3 != EVEN_ALIGN),
            OP_LOGE_FOR_INVALID_SHAPE(opName_, "gmmWeightScale"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::CheckMxQuantMmScaleShapes()
{
    if (!localParams_.hasSharedMm) {
        return ge::GRAPH_SUCCESS;
    }
    bool TransmmWeightFlag = false;
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "attrs"), return ge::GRAPH_FAILED);
    const bool *isTransmmWeight = attrs->GetAttrPointer<bool>(ATTR_TRANS_MM_WEIGHT_INDEX);
    if (isTransmmWeight) {
        TransmmWeightFlag = *isTransmmWeight;
    }
    const gert::StorageShape *mmXScaleShape = context_->GetOptionalInputShape(MM_X_SCALE_OPTIONAL_INDEX);
    const gert::StorageShape *mmWeightScaleShape = context_->GetOptionalInputShape(MM_WEIGHT_SCALE_OPTIONAL_INDEX);
    OP_TILING_CHECK((mmXScaleShape == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName_, "mmXScaleShape"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((mmWeightScaleShape == nullptr), OP_LOGE_WITH_INVALID_INPUT(opName_, "mmWeightScaleShape"),
                    return ge::GRAPH_FAILED);

    uint64_t mmXScaleDim0 = mmXScaleShape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t mmXScaleDim1 = mmXScaleShape->GetStorageShape().GetDim(DIM_ONE);
    uint64_t mmXScaleDim2 = mmXScaleShape->GetStorageShape().GetDim(DIM_TWO);
    uint64_t mmWeightScaleDim0 = mmWeightScaleShape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t mmWeightScaleDim1 = mmWeightScaleShape->GetStorageShape().GetDim(DIM_ONE);
    uint64_t mmWeightScaleDim2 = mmWeightScaleShape->GetStorageShape().GetDim(DIM_TWO);

    uint64_t mmxDivH2 = (localParams_.H2 + MX_SCALE_GROUP - 1) / MX_SCALE_GROUP;
    OP_TILING_CHECK((localParams_.Bs != mmXScaleDim0) || (mmxDivH2 != mmXScaleDim1) || (mmXScaleDim2 != EVEN_ALIGN),
                    OP_LOGE_FOR_INVALID_SHAPE(opName_, "mmXScale"),
                    return ge::GRAPH_FAILED);

    if (localParams_.isMmWeightTrans) { // Transposed Scenario
        OP_TILING_CHECK(
            (mmWeightScaleDim0 != localParams_.N2) || (mmWeightScaleDim1 != mmxDivH2) ||
                (mmWeightScaleDim2 != EVEN_ALIGN),
            OP_LOGE_FOR_INVALID_SHAPE(opName_, "mmWeightScale (transposed)"),
            return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(
            (mmWeightScaleDim0 != mmxDivH2) || (mmWeightScaleDim1 != localParams_.N2) ||
                (mmWeightScaleDim2 != EVEN_ALIGN),
            OP_LOGE_FOR_INVALID_SHAPE(opName_, "mmWeightScale"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::CheckMxfp4SpecificConstraints()
{
    if (localParams_.gmmXDtype != ge::DT_FLOAT4_E2M1 && localParams_.gmmWeightDtype != ge::DT_FLOAT4_E2M1) {
        return ge::GRAPH_SUCCESS;
    }
    // In MXFP4 mode，the data types of x and weight must be consistent
    OP_TILING_CHECK(localParams_.gmmXDtype != localParams_.gmmWeightDtype,
                    OP_LOGE_FOR_INVALID_DTYPE(opName_, "gmmX and gmmWeight (should match in MXFP4)"),
                    return ge::GRAPH_FAILED);
    OP_LOGD(opName_, "MXFP4 mode enabled: gmmXDtype=%s", Ops::Base::ToString(localParams_.gmmXDtype).c_str());
    // non-transposed: weight shape is(e,H1,N1)，N must be even
    if (!localParams_.isGmmWeightTrans) {
        OP_TILING_CHECK(
            localParams_.N1 % 2 != 0,
            OP_LOGE_FOR_INVALID_VALUE(opName_, "N1 (must be even in MXFP4)", localParams_.N1),
            return ge::GRAPH_FAILED);
    }
    // H1 must be even and not equal to 2
    OP_TILING_CHECK((localParams_.H1 % 2 != 0) || (localParams_.H1 == 2),
                    OP_LOGE_FOR_INVALID_VALUE(opName_, "H1 (even and !=2 in MXFP4)", localParams_.H1),
                    return ge::GRAPH_FAILED);
    // exist mm
    if (localParams_.hasSharedMm) {
        // non-transposed: weight shape is(H2,N2)
        if (!localParams_.isMmWeightTrans) {
            OP_TILING_CHECK(localParams_.N2 % 2 != 0,
                            OP_LOGE_FOR_INVALID_VALUE(opName_, "N2 (must be even in MXFP4)", localParams_.N2),
                            return ge::GRAPH_FAILED);
        }
        // H2 must be even and not equal to 2
        OP_TILING_CHECK(
            (localParams_.H2 % 2 != 0) || (localParams_.H2 == 2),
            OP_LOGE_FOR_INVALID_VALUE(opName_, "H2 (even and !=2 in MXFP4)", localParams_.H2),
            return ge::GRAPH_FAILED);
    }
    OP_LOGD(opName_, "MXFP4 specific constraints check passed.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::CheckAndSetInputOutputInfo()
{
    auto status = CheckOpInputSingleParamsTensor();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    status = CheckAndSetLocalParams();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    status = CheckParamsRelationAndSetLocalParams();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    status = CheckMxQuantDtypeConstraints();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    status = CheckMxQuantGmmScaleShapes();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    status = CheckMxQuantMmScaleShapes();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    status = CheckMxfp4SpecificConstraints();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MxQuantGroupedMatmulAllToAllvTiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(opName_, "get workspace failed"), return ge::GRAPH_FAILED);
    workspaces[0] = workSpaceSize_;
    OP_LOGD(opName_, "Workspaces[0] size=%ld", workspaces[0]);

    return ge::GRAPH_SUCCESS;
}

uint64_t MxQuantGroupedMatmulAllToAllvTiling::GetTilingKey() const
{
    uint8_t commMode = Mc2Comm::GetCommModeFromEnv();
    const uint64_t tilingKey =
        GET_TPL_TILING_KEY(localParams_.hasSharedMm, localParams_.isGmmWeightTrans, localParams_.isMmWeightTrans,
                           commMode);
    OP_LOGD(opName_, "GET_TPL_TILING_KEY: [%d,%d,%d,%d], TilingKey is [%lu].", localParams_.hasSharedMm,
            localParams_.isGmmWeightTrans, localParams_.isMmWeightTrans, commMode, tilingKey);
    return tilingKey;
}

// 注册tiling类
REGISTER_OPS_TILING_TEMPLATE(QuantGroupedMatMulAlltoAllv, MxQuantGroupedMatmulAllToAllvTiling, 1);

}
