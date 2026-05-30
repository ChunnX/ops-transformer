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
 * \file grouped_mat_mul_allto_allv_tiling.cpp
 */

#include "grouped_mat_mul_allto_allv_tiling.h"
#include "op_mc2.h"
#include "mc2_log.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "mc2_comm_utils.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "graph/utils/type_utils.h"

using namespace Mc2Log;
using namespace AscendC;
using namespace ge;
using namespace Ops::Transformer::OpTiling;
using namespace Mc2Tiling;
using namespace Mc2Tiling::Mc2GroupedMatmul;

namespace optiling {

const std::vector<uint32_t> GroupedMatmulAllToAllvTiling::GMM_X_DTYPE_LIST = {ge::DT_FLOAT16, ge::DT_BF16};
const std::vector<uint32_t> GroupedMatmulAllToAllvTiling::GMM_WEIGHT_DTYPE_LIST = {ge::DT_FLOAT16, ge::DT_BF16};
const std::vector<uint32_t> GroupedMatmulAllToAllvTiling::GMM_Y_DTYPE_LIST = {ge::DT_FLOAT16, ge::DT_BF16};
const std::set<int64_t> GroupedMatmulAllToAllvTiling::A5_SUPPORT_RANK_SIZE{2, 4, 8, 16, 32, 64};
const std::set<int64_t> GroupedMatmulAllToAllvTiling::A3_SUPPORT_RANK_SIZE{8, 16, 32, 64, 128};

ge::graphStatus GroupedMatmulAllToAllvTiling::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    localParams_.opName = opName_;
    return ge::GRAPH_SUCCESS;
}

bool GroupedMatmulAllToAllvTiling::IsCapable()
{
    return true;
}

ge::graphStatus GroupedMatmulAllToAllvTiling::CheckOpInputSingleParamsTensorNotSupport()
{
    auto sendCountsTensorShape = context_->GetOptionalInputShape(SEND_COUNTS_TENSOR_OPTIONAL_INDEX);
    auto recvCountsTensorShape = context_->GetOptionalInputShape(RECV_COUNTS_TENSOR_OPTIONAL_INDEX);
    OP_TILING_CHECK(sendCountsTensorShape != nullptr || recvCountsTensorShape != nullptr,
                    OP_LOGE(opName_, "sendCountsTensor and recvCountsTensor should all be nullptr now!"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTiling::CheckOpInputSingleParamsTensorSupport()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTiling::CheckOpInputSingleParamsTensorMM()
{
    auto mmXTensorShape = context_->GetOptionalInputShape(MM_X_OPTIONAL_INDEX);
    auto mmWeightTensorShape = context_->GetOptionalInputShape(MM_WEIGHT_OPTIONAL_INDEX);
    auto mmYShape = context_->GetOutputShape(OUTPUT_MM_Y_OPTIONAL_INDEX);

    bool isMmXNull = (mmXTensorShape == nullptr);
    bool isMmWeightNull = (mmWeightTensorShape == nullptr);
    bool isMmYNull = (mmYShape == nullptr);
    if (!isMmYNull) {
        isMmYNull = mmYShape->GetStorageShape().GetDimNum() == 0;
    }
    bool allNull = isMmXNull && isMmWeightNull && isMmYNull;
    bool allNotNull = !isMmXNull && !isMmWeightNull && !isMmYNull;
    OP_TILING_CHECK(!allNull && !allNotNull,
                    OP_LOGE(opName_, "mmXTensor, mmWeightTensor, mmYTensor must exist or not exist at same time."),
                    return ge::GRAPH_FAILED);
    if (!isMmXNull) {
        localParams_.hasSharedMm = true;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTiling::CheckAndSetLocalParamsGmm()
{
    auto gmmXDesc = context_->GetInputDesc(GMM_X_INDEX);
    OP_TILING_CHECK(gmmXDesc == nullptr, OP_LOGE(opName_, "GetInputDesc gmmX returned null."), return ge::GRAPH_FAILED);
    localParams_.gmmXDtype = gmmXDesc->GetDataType();
    OP_TILING_CHECK(!IsContains(GMM_X_DTYPE_LIST, localParams_.gmmXDtype),
                    OP_LOGE(opName_, "The Input gmmX Dtype should be in (DT_FLOAT16, DT_BF16), but gmmX is %s.",
                            Ops::Base::ToString(localParams_.gmmXDtype).c_str()),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *gmmWeightStorageShape = context_->GetInputShape(GMM_WEIGHT_INDEX);
    OP_TILING_CHECK(gmmWeightStorageShape == nullptr, OP_LOGE(opName_, "gmmWeightStorageShape is null!"),
                    return ge::GRAPH_FAILED);
    auto gmmWeightDesc = context_->GetInputDesc(GMM_WEIGHT_INDEX);
    OP_TILING_CHECK(gmmWeightDesc == nullptr, OP_LOGE(opName_, "GetInputDesc gmmWeight returned null."),
                    return ge::GRAPH_FAILED);
    localParams_.gmmWeightDtype = gmmWeightDesc->GetDataType();
    OP_TILING_CHECK(!IsContains(GMM_WEIGHT_DTYPE_LIST, localParams_.gmmWeightDtype),
                    OP_LOGE(opName_,
                            "The Input gmmWeight Dtype should be in (DT_FLOAT16, DT_BF16), but gmmWeight is %s.",
                            Ops::Base::ToString(localParams_.gmmWeightDtype).c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(localParams_.gmmXDtype != localParams_.gmmWeightDtype,
                    OP_LOGE(opName_, "gmmX dtype %s should be equal to gmmWeight dtype %s.",
                            Ops::Base::ToString(localParams_.gmmXDtype).c_str(),
                            Ops::Base::ToString(localParams_.gmmWeightDtype).c_str()),
                    return ge::GRAPH_FAILED);

    auto yDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_TILING_CHECK(yDesc == nullptr, OP_LOGE(opName_, "GetOutputDesc y returned null."), return ge::GRAPH_FAILED);
    localParams_.yDtype = yDesc->GetDataType();
    OP_TILING_CHECK(!IsContains(GMM_Y_DTYPE_LIST, localParams_.yDtype),
                    OP_LOGE(opName_, "The Output y Dtype should be in (DT_FLOAT16, DT_BF16), but y Dtype is %s.",
                            Ops::Base::ToString(localParams_.yDtype).c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(localParams_.gmmXDtype != localParams_.yDtype,
                    OP_LOGE(opName_, "gmmX dtype %s should be equal to y dtype %s.",
                            Ops::Base::ToString(localParams_.gmmXDtype).c_str(),
                            Ops::Base::ToString(localParams_.yDtype).c_str()),
                    return ge::GRAPH_FAILED);
    localParams_.gmmYDtype = localParams_.yDtype;

    const gert::StorageShape *gmmXStorageShape = context_->GetInputShape(GMM_X_INDEX);
    const gert::StorageShape *yStorageShape = context_->GetOutputShape(OUTPUT_Y_INDEX);
    OP_TILING_CHECK(gmmXStorageShape == nullptr, OP_LOGE(opName_, "gmmXStorageShape is null!"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(yStorageShape == nullptr, OP_LOGE(opName_, "yStorageShape is null!"), return ge::GRAPH_FAILED);

    auto status = CheckShapeDimensions(gmmXStorageShape, DIM_TWO, "gmmXShape");
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckShapeDimensions(gmmWeightStorageShape, DIM_THREE, "gmmWeightShape");
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckShapeDimensions(yStorageShape, DIM_TWO, "yShape");
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);

    localParams_.A = gmmXStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.H1 = gmmXStorageShape->GetStorageShape().GetDim(DIM_ONE);
    localParams_.ep = gmmWeightStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.gmmWeightDim1 = gmmWeightStorageShape->GetStorageShape().GetDim(DIM_ONE);
    localParams_.gmmWeightDim2 = gmmWeightStorageShape->GetStorageShape().GetDim(DIM_TWO);
    localParams_.BsK = yStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.N1 = yStorageShape->GetStorageShape().GetDim(DIM_ONE);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTiling::CheckAndSetLocalParamsMm()
{
    if (!localParams_.hasSharedMm) {
        return ge::GRAPH_SUCCESS;
    }
    auto mmXDesc = context_->GetOptionalInputDesc(MM_X_OPTIONAL_INDEX);
    auto mmWeightDesc = context_->GetOptionalInputDesc(MM_WEIGHT_OPTIONAL_INDEX);
    OP_TILING_CHECK(mmXDesc == nullptr, OP_LOGE(opName_, "GetOptionalInputDesc mmX returned null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mmWeightDesc == nullptr, OP_LOGE(opName_, "GetOptionalInputDesc mmWeight returned null."),
                    return ge::GRAPH_FAILED);
    localParams_.mmXDtype = mmXDesc->GetDataType();
    localParams_.mmWeightDtype = mmWeightDesc->GetDataType();
    OP_TILING_CHECK(!IsContains(GMM_X_DTYPE_LIST, localParams_.mmXDtype),
                    OP_LOGE(opName_, "The Input mmX Dtype should be in (DT_FLOAT16, DT_BF16), but mmX is %s.",
                            Ops::Base::ToString(localParams_.mmXDtype).c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsContains(GMM_WEIGHT_DTYPE_LIST, localParams_.mmWeightDtype),
                    OP_LOGE(opName_, "The Input mmWeight Dtype should be in (DT_FLOAT16, DT_BF16), but mmWeight is %s.",
                            Ops::Base::ToString(localParams_.mmWeightDtype).c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(localParams_.gmmXDtype != localParams_.mmXDtype,
                    OP_LOGE(opName_, "mmX dtype %s must be the same as gmmX dtype %s.",
                            Ops::Base::ToString(localParams_.mmXDtype).c_str(),
                            Ops::Base::ToString(localParams_.gmmXDtype).c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(localParams_.gmmXDtype != localParams_.mmWeightDtype,
                    OP_LOGE(opName_, "mmWeight dtype %s must be the same as gmmX dtype %s.",
                            Ops::Base::ToString(localParams_.mmWeightDtype).c_str(),
                            Ops::Base::ToString(localParams_.gmmXDtype).c_str()),
                    return ge::GRAPH_FAILED);
    auto mmYDesc = context_->GetOutputDesc(OUTPUT_MM_Y_OPTIONAL_INDEX);
    OP_TILING_CHECK(mmYDesc == nullptr, OP_LOGE(opName_, "GetOutputDesc mmY returned null."), return ge::GRAPH_FAILED);
    localParams_.mmYDtype = mmYDesc->GetDataType();

    const gert::StorageShape *mmXStorageShape = context_->GetOptionalInputShape(MM_X_OPTIONAL_INDEX);
    const gert::StorageShape *mmWeightStorageShape = context_->GetOptionalInputShape(MM_WEIGHT_OPTIONAL_INDEX);
    const gert::StorageShape *mmYStorageShape = context_->GetOutputShape(OUTPUT_MM_Y_OPTIONAL_INDEX);
    OP_TILING_CHECK(mmXStorageShape == nullptr, OP_LOGE(opName_, "mmXStorageShape is null!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mmWeightStorageShape == nullptr, OP_LOGE(opName_, "mmWeightStorageShape is null!"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mmYStorageShape == nullptr, OP_LOGE(opName_, "mmYStorageShape is null!"), return ge::GRAPH_FAILED);

    auto status = CheckShapeDimensions(mmXStorageShape, DIM_TWO, "mmXShape");
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckShapeDimensions(mmWeightStorageShape, DIM_TWO, "mmWeightShape");
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckShapeDimensions(mmYStorageShape, DIM_TWO, "mmYShape");
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);

    localParams_.Bs = mmXStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.H2 = mmXStorageShape->GetStorageShape().GetDim(DIM_ONE);
    localParams_.mmWeightDim0 = mmWeightStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.mmWeightDim1 = mmWeightStorageShape->GetStorageShape().GetDim(DIM_ONE);
    uint64_t mmYDim0 = mmYStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    OP_TILING_CHECK(localParams_.Bs != mmYDim0,
                    OP_LOGE(opName_, "mmX DIM0 %lu and mmY DIM0 %lu is not valid!", localParams_.Bs, mmYDim0),
                    return ge::GRAPH_FAILED);
    localParams_.N2 = mmYStorageShape->GetStorageShape().GetDim(DIM_ONE);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTiling::CheckAndSetLocalParamsAttr()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(opName_, "Failed to get attrs."), return ge::GRAPH_FAILED);

    auto transGmmWeightPtr = attrs->GetAttrPointer<bool>(ATTR_TRANS_GMM_WEIGHT_INDEX);
    OP_TILING_CHECK(transGmmWeightPtr == nullptr, OP_LOGE(opName_, "transGmmWeightPtr is null."),
                    return ge::GRAPH_FAILED);
    localParams_.isGmmWeightTrans = *transGmmWeightPtr;

    auto transMmWeightPtr = attrs->GetAttrPointer<bool>(ATTR_TRANS_MM_WEIGHT_INDEX);
    OP_TILING_CHECK(transMmWeightPtr == nullptr, OP_LOGE(opName_, "transMmWeightPtr is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*transMmWeightPtr == true && !localParams_.hasSharedMm,
                    OP_LOGE(opName_,
                            "transMmWeight should not be true when mmX is null! transMmWeight=%d, hasSharedMm=%d.",
                            *transMmWeightPtr, localParams_.hasSharedMm),
                    return ge::GRAPH_FAILED);
    localParams_.isMmWeightTrans = *transMmWeightPtr;

    localParams_.gmmXScaleDtype = localParams_.gmmXDtype;
    localParams_.gmmWeightScaleDtype = localParams_.gmmXDtype;
    localParams_.mmXScaleDtype = localParams_.gmmXDtype;
    localParams_.mmWeightScaleDtype = localParams_.gmmXDtype;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTiling::CheckAndSetLocalParams()
{
    auto status = CheckAndSetLocalParamsGmm();
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckAndSetLocalParamsMm();
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckAndSetLocalParamsAttr();
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTiling::CheckParamsRelationGmm()
{
    return CheckParamsRelationGmmTransShape();
}

ge::graphStatus GroupedMatmulAllToAllvTiling::CheckParamsRelationMm()
{
    return CheckParamsRelationMmTransShape();
}

ge::graphStatus GroupedMatmulAllToAllvTiling::CheckParamsAttrEpAndSetLocalParams()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    const char *group = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    OP_TILING_CHECK(group == nullptr, OP_LOGE(opName_, "group is null."), return ge::GRAPH_FAILED);

    int64_t rankDim = 0;
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(opName_, "epWorldSizePtr is null."), return ge::GRAPH_FAILED);
    if (*epWorldSizePtr == RANK_DEFAULT_NUM) {
        OP_TILING_CHECK(!mc2tiling::GetRankSize(opName_, group, rankDim), OP_LOGE(opName_, "GetRankSize failed."),
                        return ge::GRAPH_FAILED);
    } else {
        rankDim = *epWorldSizePtr;
    }

    std::string supportRankSizeRange;
    const std::set<int64_t> &supportRankSize =
        (npuArch_ == NpuArch::DAV_3510) ? A5_SUPPORT_RANK_SIZE : A3_SUPPORT_RANK_SIZE;
    for (const auto &v : supportRankSize) {
        supportRankSizeRange += (std::to_string(v) + " ");
    }
    OP_TILING_CHECK(
        supportRankSize.find(rankDim) == supportRankSize.end(),
        OP_LOGE(opName_, "World_size should be %s, but the actual value is %ld.", supportRankSizeRange, rankDim),
        return ge::GRAPH_FAILED);
    localParams_.epWorldSize = rankDim;

    bool isEpNumMatch = (localParams_.ep > 0) && (localParams_.ep < 33);
    if (!isEpNumMatch) {
        OP_LOGE(opName_, "ep (experts per rank) should be in range (0, %ld], but got %lu.", MAX_EXPERT_NUM_PER_RANK,
                localParams_.ep);
        return ge::GRAPH_FAILED;
    }
    uint64_t expertNum = localParams_.ep * rankDim;
    OP_TILING_CHECK(
        expertNum > MAX_EXPERT_NUM,
        OP_LOGE(opName_, "experts(ep * epWorldSize) max is %lu, but now is %lu !", MAX_EXPERT_NUM, expertNum),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTiling::GetPlatformInfo()
{
    auto status = Mc2Tiling::Mc2GroupedMatmul::QuantGroupedMatmulAllToAllvTilingCommon::GetPlatformInfo();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, OP_LOGE(opName_, "Failed to get platform info."), return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    npuArch_ = ascendcPlatform.GetCurNpuArch();
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedMatmulAllToAllvTiling::GetTilingKey() const
{
    uint8_t commMode = Mc2Comm::GetCommModeFromEnv();
    if (npuArch_ != NpuArch::DAV_3510) {
        commMode = Mc2Comm::COMM_MODE_AICPU;
    }
    const uint64_t tilingKey = GET_TPL_TILING_KEY(localParams_.hasSharedMm, localParams_.isGmmWeightTrans,
                                                  localParams_.isMmWeightTrans, commMode);
    OP_LOGD(opName_, "GET_TPL_TILING_KEY: [%d,%d,%d,%d], TilingKey is [%lu].", localParams_.hasSharedMm,
            localParams_.isGmmWeightTrans, localParams_.isMmWeightTrans, commMode, tilingKey);
    return tilingKey;
}

static ge::graphStatus GroupedMatMulAlltoAllvTilingFunc(gert::TilingContext *context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

struct GroupedMatMulAlltoAllvCompileInfo {};
static ge::graphStatus TilingParseForGroupedMatMulAlltoAllv(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupedMatMulAlltoAllv)
    .Tiling(GroupedMatMulAlltoAllvTilingFunc)
    .TilingParse<GroupedMatMulAlltoAllvCompileInfo>(TilingParseForGroupedMatMulAlltoAllv);
} // namespace optiling