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
 * \file grouped_mat_mul_allto_allv_tiling_common.cpp
 * \brief
 */

#include "common/utils/op_mc2.h"
#include "mc2_log.h"
#include "grouped_mat_mul_allto_allv_tiling_common.h"
#include "grouped_mat_mul_allto_allv_tiling_adapter.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include <tiling/tiling_api.h>
#include <numeric>
#include <sstream>

using namespace Mc2Log;
using namespace AscendC;
using namespace optiling;
using namespace optiling::Mc2GroupedMatmul;

namespace optiling {

constexpr uint32_t GMM_X_INDEX = 0;
constexpr uint32_t GMM_WEIGHT_INDEX = 1;
constexpr uint32_t SEND_COUNTS_TENSOR_OPTIONAL_INDEX = 2;
constexpr uint32_t RECV_COUNTS_TENSOR_OPTIONAL_INDEX = 3;
constexpr uint32_t MM_X_OPTIONAL_INDEX = 4;
constexpr uint32_t MM_WEIGHT_OPTIONAL_INDEX = 5;

constexpr uint32_t OUTPUT_Y_INDEX = 0;
constexpr uint32_t OUTPUT_MM_Y_OPTIONAL_INDEX = 1;

constexpr uint32_t ATTR_GROUP_INDEX = 0;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_SEND_COUNTS_INDEX = 2;
constexpr uint32_t ATTR_RECV_COUNTS_INDEX = 3;
constexpr uint32_t ATTR_TRANS_GMM_WEIGHT_INDEX = 4;
constexpr uint32_t ATTR_TRANS_MM_WEIGHT_INDEX = 5;

constexpr uint32_t DIM_ZERO = 0;
constexpr uint32_t DIM_ONE = 1;
constexpr uint32_t DIM_TWO = 2;
constexpr uint32_t DIM_THREE = 3;

constexpr int64_t NUM_ZERO = 0;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t NUM_EIGHT = 8;

constexpr uint64_t MAX_EXPERT_NUM = 256;
constexpr int64_t MAX_EXPERT_NUM_PER_RANK = 32;
constexpr int64_t MAX_DIM_VALUE = 65536;
constexpr uint64_t MAX_H1_VALUE = 65536;
constexpr uint64_t MAX_N1_VALUE = 65536;
constexpr uint64_t MAX_N2_VALUE = 65536;
constexpr uint64_t MIN_K_VALUE = 2;
constexpr uint64_t MAX_K_VALUE = 8;
constexpr uint32_t MAX_SHARED_H_SHAPE_SIZE = 12288;
constexpr int64_t MAX_BSK_VALUE = 52428800;
constexpr int64_t RANK_DEFAULT_NUM = -1;

const std::vector<uint32_t> GMM_X_DTYPE_LIST = {ge::DT_FLOAT16, ge::DT_BF16};
const std::vector<uint32_t> GMM_WEIGHT_DTYPE_LIST = {ge::DT_FLOAT16, ge::DT_BF16};
const std::vector<uint32_t> GMM_Y_DTYPE_LIST = {ge::DT_FLOAT16, ge::DT_BF16};
const std::set<int64_t> SUPPORT_RANK_SIZE{2, 4, 8, 16, 32, 64, 128, 256};

bool GroupedMatmulAllToAllvTilingCommon::IsContains(const std::vector<uint32_t> &list, uint32_t value)
{
    return std::count(list.begin(), list.end(), value) > 0;
}

ge::graphStatus GroupedMatmulAllToAllvTilingCommon::CheckShapeDimensions(const gert::StorageShape *shape, uint64_t dims,
                                                                         const char *shapeName)
{
    uint64_t dimNum = shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK((dimNum != dims),
                    OP_LOGE(opName_, "The %s dimNum should be %lu, but got %lu.", shapeName, dims, dimNum),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTilingCommon::GetShapeAttrsInfo()
{
    auto status = GmmAlltoAllvTilingBase::GetShapeAttrsInfo();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK((opName_ == nullptr), OP_LOGE("GMMALLTOALLV", "The opName_ is null."), return ge::GRAPH_FAILED);
    localParams_.opName = opName_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTilingCommon::CheckAndSetLocalParams()
{
    localParams_.gmmXDtype = context_->GetInputDesc(GMM_X_INDEX)->GetDataType();
    localParams_.gmmWeightDtype = context_->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(GMM_X_DTYPE_LIST, localParams_.gmmXDtype),
                    OP_LOGE(opName_, "The Input gmmX Dtype should be in (DT_FLOAT16, DT_BF16), but gmmX is %s.",
                            Ops::Base::ToString(localParams_.gmmXDtype).c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsContains(GMM_WEIGHT_DTYPE_LIST, localParams_.gmmWeightDtype),
                    OP_LOGE(opName_,
                            "The Input gmmWeight Dtype should be in (DT_FLOAT16, DT_BF16), but gmmWeight is %s.",
                            Ops::Base::ToString(localParams_.gmmWeightDtype).c_str()),
                    return ge::GRAPH_FAILED);
    localParams_.yDtype = context_->GetOutputDesc(OUTPUT_Y_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(GMM_Y_DTYPE_LIST, localParams_.yDtype),
                    OP_LOGE(opName_, "The Output y Dtype should be in (DT_FLOAT16, DT_BF16), but y Dtype is %s.",
                            Ops::Base::ToString(localParams_.yDtype).c_str()),
                    return ge::GRAPH_FAILED);
    localParams_.gmmYDtype = localParams_.yDtype;

    const gert::StorageShape *gmmXStorageShape = context_->GetInputShape(GMM_X_INDEX);
    const gert::StorageShape *gmmWeightStorageShape = context_->GetInputShape(GMM_WEIGHT_INDEX);
    const gert::StorageShape *yStorageShape = context_->GetOutputShape(OUTPUT_Y_INDEX);
    OP_TILING_CHECK(gmmXStorageShape == nullptr, OP_LOGE(opName_, "gmmXStorageShape is null!"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gmmWeightStorageShape == nullptr, OP_LOGE(opName_, "gmmWeightStorageShape is null!"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(yStorageShape == nullptr, OP_LOGE(opName_, "yStorageShape is null!"), return ge::GRAPH_FAILED);

    auto status = CheckShapeDimensions(gmmXStorageShape, DIM_TWO, "gmmXShape");
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    status = CheckShapeDimensions(gmmWeightStorageShape, DIM_THREE, "gmmWeightShape");
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    status = CheckShapeDimensions(yStorageShape, DIM_TWO, "yShape");
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    localParams_.A = gmmXStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.H1 = gmmXStorageShape->GetStorageShape().GetDim(DIM_ONE);
    localParams_.ep = gmmWeightStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.gmmWeightDim1 = gmmWeightStorageShape->GetStorageShape().GetDim(DIM_ONE);
    localParams_.gmmWeightDim2 = gmmWeightStorageShape->GetStorageShape().GetDim(DIM_TWO);
    localParams_.BsK = yStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.N1 = yStorageShape->GetStorageShape().GetDim(DIM_ONE);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTilingCommon::CheckAndSetLocalParamsMm()
{
    if (!localParams_.hasSharedMm) {
        return ge::GRAPH_SUCCESS;
    }
    localParams_.mmXDtype = context_->GetOptionalInputDesc(MM_X_OPTIONAL_INDEX)->GetDataType();
    localParams_.mmWeightDtype = context_->GetOptionalInputDesc(MM_WEIGHT_OPTIONAL_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(GMM_X_DTYPE_LIST, localParams_.mmXDtype),
                    OP_LOGE(opName_, "The Input mmX Dtype should be in (DT_FLOAT16, DT_BF16), but mmX is %s.",
                            Ops::Base::ToString(localParams_.mmXDtype).c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsContains(GMM_WEIGHT_DTYPE_LIST, localParams_.mmWeightDtype),
                    OP_LOGE(opName_, "The Input mmWeight Dtype should be in (DT_FLOAT16, DT_BF16), but mmWeight is %s.",
                            Ops::Base::ToString(localParams_.mmWeightDtype).c_str()),
                    return ge::GRAPH_FAILED);
    localParams_.mmYDtype = context_->GetOutputDesc(OUTPUT_MM_Y_OPTIONAL_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(GMM_Y_DTYPE_LIST, localParams_.mmYDtype),
                    OP_LOGE(opName_, "The Output mmY Dtype should be in (DT_FLOAT16, DT_BF16), but mmY is %s.",
                            Ops::Base::ToString(localParams_.mmYDtype).c_str()),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *mmXStorageShape = context_->GetOptionalInputShape(MM_X_OPTIONAL_INDEX);
    const gert::StorageShape *mmWeightStorageShape = context_->GetOptionalInputShape(MM_WEIGHT_OPTIONAL_INDEX);
    const gert::StorageShape *mmYStorageShape = context_->GetOutputShape(OUTPUT_MM_Y_OPTIONAL_INDEX);
    OP_TILING_CHECK(mmXStorageShape == nullptr, OP_LOGE(opName_, "mmXStorageShape is null!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mmWeightStorageShape == nullptr, OP_LOGE(opName_, "mmWeightStorageShape is null!"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mmYStorageShape == nullptr, OP_LOGE(opName_, "mmYStorageShape is null!"), return ge::GRAPH_FAILED);

    auto status = CheckShapeDimensions(mmXStorageShape, DIM_TWO, "mmXShape");
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    status = CheckShapeDimensions(mmWeightStorageShape, DIM_TWO, "mmWeightShape");
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    status = CheckShapeDimensions(mmYStorageShape, DIM_TWO, "mmYShape");
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

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

ge::graphStatus GroupedMatmulAllToAllvTilingCommon::CheckAndSetLocalParamsAttr()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(opName_, "Failed to get attrs."), return ge::GRAPH_FAILED);

    auto gmmTransWeightptr = attrs->GetAttrPointer<bool>(ATTR_TRANS_GMM_WEIGHT_INDEX);
    OP_TILING_CHECK(gmmTransWeightptr == nullptr, OP_LOGE(opName_, "gmmTransWeightptr is null."),
                    return ge::GRAPH_FAILED);
    localParams_.isGmmWeightTrans = *gmmTransWeightptr;

    auto mmTransWeightptr = attrs->GetAttrPointer<bool>(ATTR_TRANS_MM_WEIGHT_INDEX);
    OP_TILING_CHECK(mmTransWeightptr == nullptr, OP_LOGE(opName_, "mmTransWeightptr is null."),
                    return ge::GRAPH_FAILED);
    localParams_.isMmWeightTrans = *mmTransWeightptr;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTilingCommon::CheckParamsRelationGmmTransShape()
{
    if (localParams_.isGmmWeightTrans) {
        OP_TILING_CHECK(localParams_.H1 != localParams_.gmmWeightDim2,
                        OP_LOGE(opName_, "gmmX shape K %lu not match gmmWeight shape K %lu !", localParams_.H1,
                                localParams_.gmmWeightDim2),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(localParams_.N1 != localParams_.gmmWeightDim1,
                        OP_LOGE(opName_, "y shape N %lu not match gmmWeight shape N %lu !", localParams_.N1,
                                localParams_.gmmWeightDim1),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(localParams_.H1 != localParams_.gmmWeightDim1,
                        OP_LOGE(opName_, "gmmX shape %lu not match gmmWeight shape %lu !", localParams_.H1,
                                localParams_.gmmWeightDim1),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(localParams_.N1 != localParams_.gmmWeightDim2,
                        OP_LOGE(opName_, "y shape N %lu not match gmmWeight shape N %lu !", localParams_.N1,
                                localParams_.gmmWeightDim2),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTilingCommon::CheckParamsRelationMmTransShape()
{
    if (!localParams_.hasSharedMm) {
        return ge::GRAPH_SUCCESS;
    }
    if (localParams_.isMmWeightTrans) {
        OP_TILING_CHECK(localParams_.H2 != localParams_.mmWeightDim1,
                        OP_LOGE(opName_, "mmX shape %lu not match mmWeight shape %lu !", localParams_.H2,
                                localParams_.mmWeightDim1),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(localParams_.N2 != localParams_.mmWeightDim0,
                        OP_LOGE(opName_, "mmY shape N %lu not match mmWeight shape N %lu !", localParams_.N2,
                                localParams_.mmWeightDim0),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(localParams_.H2 != localParams_.mmWeightDim0,
                        OP_LOGE(opName_, "mmX shape %lu not match mmWeight shape %lu !", localParams_.H2,
                                localParams_.mmWeightDim0),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(localParams_.N2 != localParams_.mmWeightDim1,
                        OP_LOGE(opName_, "mmY shape N %lu not match mmWeight shape N %lu !", localParams_.N2,
                                localParams_.mmWeightDim1),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTilingCommon::CheckParamsAttrEpAndSetLocalParams()
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
    for (const auto &v : SUPPORT_RANK_SIZE) {
        supportRankSizeRange += (std::to_string(v) + " ");
    }
    OP_TILING_CHECK(
        SUPPORT_RANK_SIZE.find(rankDim) == SUPPORT_RANK_SIZE.end(),
        OP_LOGE(opName_, "World_size should be %s, but the actual value is %ld.", supportRankSizeRange, rankDim),
        return ge::GRAPH_FAILED);
    localParams_.epWorldSize = rankDim;

    bool isEpNumMatch = (localParams_.ep > 0) && (localParams_.ep < 33);
    if (isEpNumMatch) {
        uint64_t expertNum = localParams_.ep * rankDim;
        OP_TILING_CHECK(
            expertNum > MAX_EXPERT_NUM,
            OP_LOGE(opName_, "experts(ep * epWorldSize) max is %lu, but now is %lu !", MAX_EXPERT_NUM, expertNum),
            return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(opName_, "ep (experts per rank) should be in range (0, %ld], but got %lu.", MAX_EXPERT_NUM_PER_RANK,
                localParams_.ep);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTilingCommon::CheckAndSetSendRecvCountsAttr()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    uint64_t expertNum = localParams_.ep * localParams_.epWorldSize;
    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(sendCountsPtr == nullptr, OP_LOGE(opName_, "sendCountsPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(recvCountsPtr == nullptr, OP_LOGE(opName_, "recvCountsPtr is null."), return ge::GRAPH_FAILED);
    uint64_t sendCountsSize = sendCountsPtr->GetSize();
    uint64_t recvCountsSize = recvCountsPtr->GetSize();
    OP_TILING_CHECK(sendCountsSize != recvCountsSize,
                    OP_LOGE(opName_, "length of sendCounts %lu should be equal to length of recvCount %lu !",
                            sendCountsSize, recvCountsSize),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sendCountsSize != expertNum,
                    OP_LOGE(opName_, "length of sendCounts %lu should be equal to experts(ep * epWorldSize) %lu !",
                            sendCountsSize, expertNum),
                    return ge::GRAPH_FAILED);

    const int64_t *sendCounts = static_cast<const int64_t *>(sendCountsPtr->GetData());
    const int64_t *recvCounts = static_cast<const int64_t *>(recvCountsPtr->GetData());
    uint64_t sendCountsSum = std::accumulate(sendCounts, sendCounts + sendCountsSize, 0ULL);
    OP_TILING_CHECK(sendCountsSum != localParams_.A,
                    OP_LOGE(opName_, "sendCountsSum %lu should be equal to A %lu !", sendCountsSum, localParams_.A),
                    return ge::GRAPH_FAILED);
    uint64_t recvCountsSum = std::accumulate(recvCounts, recvCounts + recvCountsSize, 0ULL);
    OP_TILING_CHECK(recvCountsSum != localParams_.BsK,
                    OP_LOGE(opName_, "recvCountsSum %lu should be equal to BSK %lu !", recvCountsSum, localParams_.BsK),
                    return ge::GRAPH_FAILED);

    auto taskTilingInfoPtr = &localTilingData_.taskTilingInfo;
    uint64_t maxCountsSize = std::min<uint64_t>(expertNum, MAX_EXPERT_NUM);
    for (uint64_t i = 0; i < maxCountsSize; i++) {
        OP_TILING_CHECK(sendCounts[i] < 0, OP_LOGE(opName_, "sendCounts value %ld should not be < 0 !", sendCounts[i]),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(recvCounts[i] < 0, OP_LOGE(opName_, "recvCounts value %ld should not be < 0 !", recvCounts[i]),
                        return ge::GRAPH_FAILED);
        taskTilingInfoPtr->sendCnt[i] = static_cast<int32_t>(sendCounts[i]);
        taskTilingInfoPtr->recvCnt[i] = static_cast<int32_t>(recvCounts[i]);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTilingCommon::CheckLocalParams()
{
    OP_TILING_CHECK((localParams_.H1 == 0) || (localParams_.H1 >= MAX_H1_VALUE),
                    OP_LOGE(opName_, "H1 should be in range (0, %lu), but got %lu.", MAX_H1_VALUE, localParams_.H1),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((localParams_.BsK >= MAX_BSK_VALUE),
                    OP_LOGE(opName_, "BSK should be in range (0, %lu), but got %lu.", MAX_BSK_VALUE, localParams_.BsK),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((localParams_.N1 == 0) || (localParams_.N1 >= MAX_N1_VALUE),
                    OP_LOGE(opName_, "N1 should be in range (0, %lu), but got %lu.", MAX_N1_VALUE, localParams_.N1),
                    return ge::GRAPH_FAILED);

    if (!localParams_.hasSharedMm) {
        return ge::GRAPH_SUCCESS;
    }
    OP_TILING_CHECK((localParams_.Bs == 0) || (localParams_.BsK % localParams_.Bs != 0),
                    OP_LOGE(opName_, "BSK should be an integer multiple of BS, but got BSK[%lu] and BS[%lu].",
                            localParams_.BsK, localParams_.Bs),
                    return ge::GRAPH_FAILED);
    uint64_t k = localParams_.BsK / localParams_.Bs;
    OP_TILING_CHECK((k < MIN_K_VALUE) || (k > MAX_K_VALUE),
                    OP_LOGE(opName_, "K (BSK / BS) should be in [%lu, %lu], but got %lu.", MIN_K_VALUE, MAX_K_VALUE, k),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (localParams_.H2 == 0) || (localParams_.H2 > MAX_SHARED_H_SHAPE_SIZE),
        OP_LOGE(opName_, "H2 should be in range (0, %lu), but got %lu.", MAX_SHARED_H_SHAPE_SIZE, localParams_.H2),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((localParams_.N2 == 0) || (localParams_.N2 >= MAX_N2_VALUE),
                    OP_LOGE(opName_, "N2 should be in range (0, %lu), but got %lu.", MAX_N2_VALUE, localParams_.N2),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTilingCommon::CheckParamsRelationAndSetLocalParams()
{
    auto status = CheckParamsRelationGmmTransShape();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    status = CheckParamsRelationMmTransShape();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    status = CheckParamsAttrEpAndSetLocalParams();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    status = CheckAndSetSendRecvCountsAttr();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    status = CheckLocalParams();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckOpInputSingleParamsTensorNotSupport(gert::TilingContext *context_, const char *opName_)
{
    auto sendCountsTensorShape = context_->GetOptionalInputShape(SEND_COUNTS_TENSOR_OPTIONAL_INDEX);
    auto recvCountsTensorShape = context_->GetOptionalInputShape(RECV_COUNTS_TENSOR_OPTIONAL_INDEX);
    bool sendCountsTensorShapeNotNull = sendCountsTensorShape != nullptr;
    bool recvCountsTensorShapeNotNull = recvCountsTensorShape != nullptr;
    OP_TILING_CHECK(sendCountsTensorShapeNotNull || recvCountsTensorShapeNotNull,
                    OP_LOGE(opName_,
                            "sendCountsTensorNotNull = %d and recvCountsTensorNotNull = %d, should all be nullptr now!",
                            sendCountsTensorShapeNotNull, recvCountsTensorShapeNotNull),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTilingCommon::CheckAndSetInputOutputInfo()
{
    auto status = CheckOpInputSingleParamsTensorNotSupport(context_, opName_);
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto mmXTensorShape = context_->GetOptionalInputShape(MM_X_OPTIONAL_INDEX);
    auto mmWeightTensorShape = context_->GetOptionalInputShape(MM_WEIGHT_OPTIONAL_INDEX);
    auto mmYShape = context_->GetOutputShape(OUTPUT_MM_Y_OPTIONAL_INDEX);

    bool isMmXNull = (mmXTensorShape == nullptr);
    bool isMmWeightNull = (mmWeightTensorShape == nullptr);
    bool isMmYNull = (mmYShape == nullptr);
    if (!isMmYNull) {
        auto mmYDimNum = mmYShape->GetStorageShape().GetDimNum();
        isMmYNull = mmYDimNum == 0;
    }
    bool isAllSame = (isMmXNull == isMmWeightNull) && (isMmWeightNull == isMmYNull);
    OP_TILING_CHECK(!isAllSame,
                    OP_LOGE(opName_, "mmXTensor, mmWeightTensor, mmYTensor must exist or not exist at same time."),
                    return ge::GRAPH_FAILED);
    if (!isMmXNull) {
        localParams_.hasSharedMm = true;
    }

    status = CheckAndSetLocalParams();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    status = CheckAndSetLocalParamsMm();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    status = CheckAndSetLocalParamsAttr();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTilingCommon::GetPlatformInfo()
{
    fe::PlatFormInfos *platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, OP_LOGE(opName_, "Failed to get platform info."), return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    libApiWorkSpaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    localParams_.aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    localParams_.aicCoreNum = ascendcPlatform.GetCoreNumAic();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTilingCommon::SetTilingCommonInfo()
{
    auto taskTilingInfoPtr = &localTilingData_.taskTilingInfo;
    taskTilingInfoPtr->BSK = localParams_.BsK;
    taskTilingInfoPtr->BS = localParams_.Bs;
    taskTilingInfoPtr->H1 = localParams_.H1;
    taskTilingInfoPtr->H2 = localParams_.H2;
    taskTilingInfoPtr->A = localParams_.A;
    taskTilingInfoPtr->N1 = localParams_.N1;
    taskTilingInfoPtr->N2 = localParams_.N2;
    taskTilingInfoPtr->epWorldSize = localParams_.epWorldSize;
    taskTilingInfoPtr->e = localParams_.ep;

    taskTilingInfoPtr->mainLoopExpertNum = 1;
    taskTilingInfoPtr->tailLoopExpertNum = 1;
    taskTilingInfoPtr->totalLoopCount = localParams_.ep;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTilingCommon::SetGmmA2avWorkspaceInfo()
{
    constexpr uint64_t alignAddrLen = 512;
    auto gmmYDtypeSize = mc2tiling::GetDataTypeSize(opName_, localParams_.gmmYDtype);
    inferredInfo_.gmmResultLen = mc2tiling::AlignUp(localParams_.A * localParams_.N1 * gmmYDtypeSize, alignAddrLen);
    localTilingData_.workspaceInfo.wsGmmOutputSize = inferredInfo_.gmmResultLen;
    constexpr uint64_t ptrTableSize = 128;
    uint64_t gmmComputeWsSize = localParams_.ep * sizeof(int64_t) + ptrTableSize;
    localTilingData_.workspaceInfo.wsGmmComputeWorkspaceSize = mc2tiling::AlignUp(gmmComputeWsSize, alignAddrLen);
    localTilingData_.workspaceInfo.wsSharedGmmComputeWorkspaceSize = mc2tiling::AlignUp(gmmComputeWsSize, alignAddrLen);
    workSpaceSize_ = libApiWorkSpaceSize_ + inferredInfo_.gmmResultLen +
                     localTilingData_.workspaceInfo.wsGmmComputeWorkspaceSize +
                     localTilingData_.workspaceInfo.wsSharedGmmComputeWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTilingCommon::DoGMMTiling()
{
    Mc2GroupedMatmul::GroupedMatmulAllToAllvAdapter gmmTile(context_);
    GE_ASSERT_GRAPH_SUCCESS(gmmTile.SetCommonInputParams(localParams_));
    uint64_t mMaxSize = 0;
    uint64_t mSize = 0;
    for (uint64_t expertIdx = 0; expertIdx < localParams_.ep; expertIdx++) {
        mSize = 0;
        for (uint64_t i = 0; i < localParams_.epWorldSize; i++) {
            mSize += localTilingData_.taskTilingInfo.sendCnt[i * localParams_.ep + expertIdx];
        }
        mMaxSize = std::max(mSize, mMaxSize);
    }
    GE_ASSERT_GRAPH_SUCCESS(gmmTile.SetGroupExpertInputParameters(localParams_, mMaxSize));
    GE_ASSERT_GRAPH_SUCCESS(gmmTile.Process());
    localTilingData_.gmmBaseTiling = gmmTile.GetGmmQuantTilingAdapterData();

    if (!localParams_.hasSharedMm) {
        return ge::GRAPH_SUCCESS;
    }
    GE_ASSERT_GRAPH_SUCCESS(gmmTile.SetSharedExpertInputParameters(localParams_));
    GE_ASSERT_GRAPH_SUCCESS(gmmTile.Process());
    localTilingData_.sharedGmmTiling = gmmTile.GetGmmQuantTilingAdapterData();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTilingCommon::SetHcclTiling()
{
    uint32_t alltoAllvCmd = 8U;
    std::string alltoAllvConfig = "AlltoAll=level0:fullmesh;level1:pairwise";

    auto attrs = context_->GetAttrs();
    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);

    const uint32_t alltoAllvReduceType = 0u;
    auto outputDataType = context_->GetOutputDesc(OUTPUT_Y_INDEX)->GetDataType();
    OP_TILING_CHECK(
        mc2tiling::HCCL_DATA_TYPE.find(outputDataType) == mc2tiling::HCCL_DATA_TYPE.end(),
        OP_LOGE(opName_, "Output data type %s is not supported yet.", Ops::Base::ToString(outputDataType).c_str()),
        return ge::GRAPH_FAILED);

    auto alltoAllvDstDataType = static_cast<uint8_t>(mc2tiling::HCCL_DATA_TYPE.find(outputDataType)->second);
    auto alltoAllvSrcDataType = static_cast<uint8_t>(mc2tiling::HCCL_DATA_TYPE.find(outputDataType)->second);

    Mc2CcTilingConfig hcclCcTilingConfig(groupEpPtr, alltoAllvCmd, alltoAllvConfig, alltoAllvReduceType,
                                         alltoAllvDstDataType, alltoAllvSrcDataType);
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(localTilingData_.hcclA2avTiling.hcclInitTiling) != 0,
                    OP_LOGE(opName_, "mc2CcTilingConfig mc2tiling GetTiling hcclInitTiling failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(localTilingData_.hcclA2avTiling.a2avCcTiling) != 0,
                    OP_LOGE(opName_, "mc2CcTilingConfig mc2tiling GetTiling alltoAllvCcTiling failed"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTilingCommon::DoOpTiling()
{
    GE_ASSERT_GRAPH_SUCCESS(CheckAndSetInputOutputInfo());
    GE_ASSERT_GRAPH_SUCCESS(CheckParamsRelationAndSetLocalParams());
    GE_ASSERT_GRAPH_SUCCESS(SetTilingCommonInfo());
    GE_ASSERT_GRAPH_SUCCESS(DoGMMTiling());
    GE_ASSERT_GRAPH_SUCCESS(SetHcclTiling());
    GE_ASSERT_GRAPH_SUCCESS(SetGmmA2avWorkspaceInfo());
    return ge::GRAPH_SUCCESS;
}

void PrintGmmA2avWorkspaceInfo(const GmmA2avWorkspaceInfo &workspaceInfo, const char *opName_)
{
    std::stringstream ss;
    ss << "workspaceInfo: ";
    ss << "wsGmmOutputSize=" << workspaceInfo.wsGmmOutputSize
       << ", wsGmmComputeWorkspaceSize=" << workspaceInfo.wsGmmComputeWorkspaceSize
       << ", wsSharedGmmComputeWorkspaceSize=" << workspaceInfo.wsSharedGmmComputeWorkspaceSize;
    OP_LOGD(opName_, "%s", ss.str().c_str());
}

void PrintTaskTilingInfo(const MC2KernelTemplate::TaskTilingInfo &taskTilingInfo,
                         const GmmAlltoAllvParamsInfo &localParams, const char *opName_)
{
    std::stringstream ss;
    ss << "TaskTilingInfo: ";
    ss << "BSK=" << taskTilingInfo.BSK << ", BS=" << taskTilingInfo.BS << ", H1=" << taskTilingInfo.H1
       << ", H2=" << taskTilingInfo.H2 << ", A=" << taskTilingInfo.A << ", N1=" << taskTilingInfo.N1
       << ", N2=" << taskTilingInfo.N2;
    ss << ", epWorldSize=" << taskTilingInfo.epWorldSize << ", e=" << taskTilingInfo.e;
    ss << ", mainLoopExpertNum=" << taskTilingInfo.mainLoopExpertNum
       << ", tailLoopExpertNum=" << taskTilingInfo.tailLoopExpertNum
       << ", totalLoopCount=" << taskTilingInfo.totalLoopCount;
    ss << "\nSendCounts: ";
    for (int64_t i = 0; i < localParams.ep * localParams.epWorldSize; i++) {
        if (taskTilingInfo.sendCnt[i] != 0) {
            if (i != 0) {
                ss << " ,";
            }
            ss << taskTilingInfo.sendCnt[i];
        }
    }
    ss << "\nRecvCounts: ";
    for (int64_t i = 0; i < localParams.ep * localParams.epWorldSize; i++) {
        if (taskTilingInfo.recvCnt[i] != 0) {
            if (i != 0) {
                ss << " ,";
            }
            ss << taskTilingInfo.recvCnt[i];
        }
    }
    OP_LOGI(opName_, "%s", ss.str().c_str());
}

void PrintGMMTilingData(const MC2KernelTemplate::GMMQuantTilingData &data, const char *opName_)
{
    const auto &mm = data.mmTilingData;
    const auto &quantParams = data.gmmQuantParams;
    const auto &gmmArray = data.gmmArray;

    std::stringstream ss;
    ss << "MM Tiling: M=" << mm.M << ", N=" << mm.N << ", K=" << mm.Ka << ", usedCoreNum=" << mm.usedCoreNum
       << ", baseM=" << mm.baseM << ", baseN=" << mm.baseN << ", baseK=" << mm.baseK
       << ", singleCoreM=" << mm.singleCoreM << ", singleCoreN=" << mm.singleCoreN << ", singleCoreK=" << mm.singleCoreK
       << ", dbL0C=" << mm.dbL0C << ", depthA1=" << mm.depthA1 << ", depthB1=" << mm.depthB1 << ", stepKa=" << mm.stepKa
       << ", stepKb=" << mm.stepKb << ", stepM=" << mm.stepM << ", stepN=" << mm.stepN
       << ", iterateOrder=" << mm.iterateOrder;

    ss << "\nQuant Params: groupNum=" << quantParams.groupNum << ", activeType=" << quantParams.activeType
       << ", aQuantMode=" << quantParams.aQuantMode << ", bQuantMode=" << quantParams.bQuantMode
       << ", singleX=" << static_cast<int32_t>(quantParams.singleX)
       << ", singleW=" << static_cast<int32_t>(quantParams.singleW)
       << ", singleY=" << static_cast<int32_t>(quantParams.singleY)
       << ", groupType=" << static_cast<int32_t>(quantParams.groupType)
       << ", groupListType=" << static_cast<uint32_t>(quantParams.groupListType)
       << ", hasBias=" << static_cast<int32_t>(quantParams.hasBias) << ", reserved=" << quantParams.reserved;

    ss << "\nArray: mList[0]=" << gmmArray.mList[0] << ", kList[0]=" << gmmArray.kList[0]
       << ", nList[0]=" << gmmArray.nList[0];

    OP_LOGI(opName_, "GmmA2AvTiling TilingParams:\n%s", ss.str().c_str());
}

void GroupedMatmulAllToAllvTilingCommon::PrintGmmA2avTilingData(GmmA2avTilingData &outTilingData)
{
    PrintGmmA2avWorkspaceInfo(outTilingData.workspaceInfo, opName_);
    PrintTaskTilingInfo(outTilingData.taskTilingInfo, localParams_, opName_);
    OP_LOGD(opName_, "------------- PrintGMMTilingData -------------------");
    PrintGMMTilingData(outTilingData.gmmBaseTiling, opName_);
    OP_LOGD(opName_, "------------- PrintGMMSharedTilingData -------------");
    PrintGMMTilingData(outTilingData.sharedGmmTiling, opName_);
}

ge::graphStatus GroupedMatmulAllToAllvTilingCommon::PostTiling()
{
    PrintGmmA2avTilingData(localTilingData_);
    context_->SetBlockDim(localParams_.aicCoreNum);
    GmmA2avTilingData *outTilingData = context_->GetTilingData<GmmA2avTilingData>();
    size_t tilingBufCap = context_->GetRawTilingData()->GetCapacity();
    OP_TILING_CHECK((outTilingData == nullptr), OP_LOGE(opName_, "Failed to get tiling data from context"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (tilingBufCap < sizeof(localTilingData_)),
        OP_LOGE(opName_, "TilingBuffer too small, capacity = %zu, need = %zu.", tilingBufCap, sizeof(localTilingData_)),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sizeof(localTilingData_) % sizeof(uint64_t) != 0,
                    OP_LOGE(opName_, "Tiling data size[%zu] is not aligned to 8", sizeof(localTilingData_)),
                    return ge::GRAPH_FAILED);
    errno_t ret =
        memcpy_s(outTilingData, tilingBufCap, reinterpret_cast<void *>(&localTilingData_), sizeof(localTilingData_));
    if (ret != EOK) {
        OP_LOGE(opName_, "postTiling: memcpy_s failed with ret=%d.", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(sizeof(localTilingData_));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTilingCommon::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(opName_, "Failed to get workspace."), return ge::GRAPH_FAILED);
    workspaces[0] = workSpaceSize_;
    OP_LOGD(opName_, "Workspaces[0] size=%ld", workspaces[0]);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulAllToAllvTilingCommon::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedMatmulAllToAllvTilingCommon::GetTilingKey() const
{
    const uint64_t tilingKey =
        GET_TPL_TILING_KEY(localParams_.hasSharedMm, localParams_.isGmmWeightTrans, localParams_.isMmWeightTrans, 0, 0);
    OP_LOGD(opName_, "GET_TPL_TILING_KEY: [%d,%d,%d,%d,%d], TilingKey is [%lu].", localParams_.hasSharedMm,
            localParams_.isGmmWeightTrans, localParams_.isMmWeightTrans, 0, 0, tilingKey);
    return tilingKey;
}

} // namespace optiling