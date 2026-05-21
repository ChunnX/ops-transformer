/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the software repository for the full text of the License.
 */

/*!
 * \file grouped_mat_mul_allto_allv_tiling.cpp
 * \brief 重构后tiling实现，A3/A5通用，产出QuantGmmA2avTilingData(量化参数=0)
 *        参考quant_grouped_mat_mul_allto_allv tiling代码模式
 */

#include "grouped_mat_mul_allto_allv_tiling.h"
#include <string>
#include <numeric>
#include <vector>
#include "op_mc2.h"
#include "mc2_log.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "op_host/op_tiling/new_mc2_tiling_utils.h"
#include "op_host/op_tiling/hccl_formulaic_tiling.h"
#include "mc2_hcom_topo_info.h"
#include "mc2_comm_utils.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "securec.h"
#include "../../quant_grouped_mat_mul_allto_allv/op_host/op_tiling/arch35/quant_grouped_mat_mul_allto_allv_tiling_adapter.h"

using namespace Mc2Log;
using namespace AscendC;
using namespace ge;
using namespace Ops::Transformer::OpTiling;
using namespace Mc2Tiling;
using namespace Mc2Tiling::Mc2GroupedMatmul;

namespace optiling {

constexpr uint32_t GMM_X_INDEX = 0;
constexpr uint32_t GMM_WEIGHT_INDEX = 1;
constexpr uint32_t SEND_COUNTS_TENSOR_OPTIONAL_INDEX = 2;
constexpr uint32_t RECV_COUNTS_TENSOR_OPTIONAL_INDEX = 3;
constexpr uint32_t MM_X_OPTIONAL_INDEX = 4;
constexpr uint32_t MM_WEIGHT_OPTIONAL_INDEX = 5;
constexpr uint32_t OUTPUT_Y_INDEX = 0;
constexpr uint32_t OUTPUT_MM_Y_OPTIONAL_INDEX = 1;

constexpr uint32_t DIM_ZERO = 0;
constexpr uint32_t DIM_ONE = 1;
constexpr uint32_t DIM_TWO = 2;
constexpr uint32_t DIM_THREE = 3;

constexpr uint32_t ATTR_GROUP_INDEX = 0;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
constexpr uint32_t ATTR_SEND_COUNTS_INDEX = 2;
constexpr uint32_t ATTR_RECV_COUNTS_INDEX = 3;
constexpr uint32_t ATTR_TRANS_GMM_WEIGHT_INDEX = 4;
constexpr uint32_t ATTR_TRANS_MM_WEIGHT_INDEX = 5;

constexpr uint64_t MAX_EXPERT_NUM = 256;
constexpr int64_t MAX_EXPERT_NUM_PER_RANK = 32;
constexpr int64_t MAX_DIM_VALUE = 65536;
constexpr uint32_t MAX_SHARED_H_SHAPE_SIZE = 12288;
constexpr int64_t MAX_BSK_VALUE = 52428800;
constexpr uint64_t MIN_K_VALUE = 2;
constexpr uint64_t MAX_K_VALUE = 8;
constexpr uint64_t MAX_H1_VALUE = 65536;
constexpr uint64_t MAX_N1_VALUE = 65536;
constexpr uint64_t MAX_N2_VALUE = 65536;

const std::vector<uint32_t> GMM_X_DTYPE_LIST = {ge::DT_FLOAT16, ge::DT_BF16};
const std::vector<uint32_t> GMM_WEIGHT_DTYPE_LIST = {ge::DT_FLOAT16, ge::DT_BF16};
const std::vector<uint32_t> GMM_Y_DTYPE_LIST = {ge::DT_FLOAT16, ge::DT_BF16};
const std::set<int64_t> SUPPORT_RANK_SIZE{2, 4, 8, 16, 32, 64, 128, 256};
constexpr int64_t RANK_DEFAULT_NUM = -1;
constexpr uint64_t ALIGN_ADDR_LEN = 512;

bool GmmAlltoAllvTilingStruct::IsContains(const std::vector<uint32_t> &list, uint32_t value)
{
    return std::count(list.begin(), list.end(), value) > 0;
}

ge::graphStatus GmmAlltoAllvTilingStruct::CheckShapeDimensions(const gert::StorageShape *shape, uint64_t dims,
                                                               const char *shapeName)
{
    uint64_t dimNum = shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK((dimNum != dims),
                    OP_LOGE(opName_, "The %s dimNum should be %lu, but got %lu.", shapeName, dims, dimNum),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GmmAlltoAllvTilingStruct::CheckAndSetLocalParamsGmm()
{
    gmmXDtype_ = context_->GetInputDesc(GMM_X_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(GMM_X_DTYPE_LIST, gmmXDtype_),
                    OP_LOGE(opName_, "The Input gmmX Dtype should be in (DT_FLOAT16, DT_BF16), but gmmX is %s.",
                            Ops::Base::ToString(gmmXDtype_).c_str()),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *gmmWeightStorageShape = context_->GetInputShape(GMM_WEIGHT_INDEX);
    OP_TILING_CHECK(gmmWeightStorageShape == nullptr, OP_LOGE(opName_, "gmmWeightStorageShape is null!"),
                    return ge::GRAPH_FAILED);
    auto gmmWeightDtype = context_->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(GMM_WEIGHT_DTYPE_LIST, gmmWeightDtype),
                    OP_LOGE(opName_,
                            "The Input gmmWeight Dtype should be in (DT_FLOAT16, DT_BF16), but gmmWeight is %s.",
                            Ops::Base::ToString(gmmWeightDtype).c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gmmXDtype_ != gmmWeightDtype,
                    OP_LOGE(opName_, "gmmX dtype %s should be equal to gmmWeight dtype %s.",
                            Ops::Base::ToString(gmmXDtype_).c_str(), Ops::Base::ToString(gmmWeightDtype).c_str()),
                    return ge::GRAPH_FAILED);

    yDtype_ = context_->GetOutputDesc(OUTPUT_Y_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(GMM_Y_DTYPE_LIST, yDtype_),
                    OP_LOGE(opName_, "The Output y Dtype should be in (DT_FLOAT16, DT_BF16), but y Dtype is %s.",
                            Ops::Base::ToString(yDtype_).c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gmmXDtype_ != yDtype_,
                    OP_LOGE(opName_, "gmmX dtype %s should be equal to y dtype %s.",
                            Ops::Base::ToString(gmmXDtype_).c_str(), Ops::Base::ToString(yDtype_).c_str()),
                    return ge::GRAPH_FAILED);
    gmmYDtype_ = yDtype_;

    const gert::StorageShape *gmmXStorageShape = context_->GetInputShape(GMM_X_INDEX);
    OP_TILING_CHECK(gmmXStorageShape == nullptr, OP_LOGE(opName_, "gmmXStorageShape is null!"),
                    return ge::GRAPH_FAILED);
    const gert::StorageShape *yStorageShape = context_->GetOutputShape(OUTPUT_Y_INDEX);
    OP_TILING_CHECK(yStorageShape == nullptr, OP_LOGE(opName_, "yStorageShape is null!"), return ge::GRAPH_FAILED);

    auto status = CheckShapeDimensions(gmmXStorageShape, DIM_TWO, "gmmXShape");
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckShapeDimensions(gmmWeightStorageShape, DIM_THREE, "gmmWeightShape");
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckShapeDimensions(yStorageShape, DIM_TWO, "yShape");
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);

    A_ = gmmXStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    H1_ = gmmXStorageShape->GetStorageShape().GetDim(DIM_ONE);
    ep_ = gmmWeightStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    gmmWeightDim1_ = gmmWeightStorageShape->GetStorageShape().GetDim(DIM_ONE);
    gmmWeightDim2_ = gmmWeightStorageShape->GetStorageShape().GetDim(DIM_TWO);
    BsK_ = yStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    N1_ = yStorageShape->GetStorageShape().GetDim(DIM_ONE);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GmmAlltoAllvTilingStruct::CheckAndSetLocalParamsMm()
{
    if (!hasSharedMm_) {
        return ge::GRAPH_SUCCESS;
    }
    auto mmXDtype = context_->GetOptionalInputDesc(MM_X_OPTIONAL_INDEX)->GetDataType();
    auto mmWeightDtype = context_->GetOptionalInputDesc(MM_WEIGHT_OPTIONAL_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(GMM_X_DTYPE_LIST, mmXDtype),
                    OP_LOGE(opName_, "The Input mmX Dtype should be in (DT_FLOAT16, DT_BF16), but mmX is %s.",
                            Ops::Base::ToString(mmXDtype).c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsContains(GMM_WEIGHT_DTYPE_LIST, mmWeightDtype),
                    OP_LOGE(opName_, "The Input mmWeight Dtype should be in (DT_FLOAT16, DT_BF16), but mmWeight is %s.",
                            Ops::Base::ToString(mmWeightDtype).c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gmmXDtype_ != mmXDtype,
                    OP_LOGE(opName_, "mmX dtype %s must be the same as gmmX dtype %s.",
                            Ops::Base::ToString(mmXDtype).c_str(), Ops::Base::ToString(gmmXDtype_).c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gmmXDtype_ != mmWeightDtype,
                    OP_LOGE(opName_, "mmWeight dtype %s must be the same as gmmX dtype %s.",
                            Ops::Base::ToString(mmWeightDtype).c_str(), Ops::Base::ToString(gmmXDtype_).c_str()),
                    return ge::GRAPH_FAILED);

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

    Bs_ = mmXStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    H2_ = mmXStorageShape->GetStorageShape().GetDim(DIM_ONE);
    mmWeightDim0_ = mmWeightStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    mmWeightDim1_ = mmWeightStorageShape->GetStorageShape().GetDim(DIM_ONE);
    uint64_t mmYDim0 = mmYStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    OP_TILING_CHECK(Bs_ != mmYDim0, OP_LOGE(opName_, "mmX DIM0 %lu and mmY DIM0 %lu is not valid!", Bs_, mmYDim0),
                    return ge::GRAPH_FAILED);
    N2_ = mmYStorageShape->GetStorageShape().GetDim(DIM_ONE);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GmmAlltoAllvTilingStruct::CheckAndSetLocalParamsAttr()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(opName_, "Failed to get attrs."), return ge::GRAPH_FAILED);

    auto transGmmWeightPtr = attrs->GetAttrPointer<bool>(ATTR_TRANS_GMM_WEIGHT_INDEX);
    OP_TILING_CHECK(transGmmWeightPtr == nullptr, OP_LOGE(opName_, "transGmmWeightPtr is null."),
                    return ge::GRAPH_FAILED);
    isGmmWeightTrans_ = *transGmmWeightPtr;

    auto transMmWeightPtr = attrs->GetAttrPointer<bool>(ATTR_TRANS_MM_WEIGHT_INDEX);
    OP_TILING_CHECK(transMmWeightPtr == nullptr, OP_LOGE(opName_, "transMmWeightPtr is null."),
                    return ge::GRAPH_FAILED);
    isMmWeightTrans_ = *transMmWeightPtr;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GmmAlltoAllvTilingStruct::CheckAndSetLocalParams()
{
    auto status = CheckAndSetLocalParamsGmm();
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckAndSetLocalParamsMm();
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckAndSetLocalParamsAttr();
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GmmAlltoAllvTilingStruct::CheckParamsRelationGmmTransShape()
{
    if (isGmmWeightTrans_) {
        OP_TILING_CHECK(H1_ != gmmWeightDim2_,
                        OP_LOGE(opName_, "gmmX shape K %lu not match gmmWeight shape K %lu !", H1_, gmmWeightDim2_),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(N1_ != gmmWeightDim1_,
                        OP_LOGE(opName_, "y shape N %lu not match gmmWeight shape N %lu !", N1_, gmmWeightDim1_),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(H1_ != gmmWeightDim1_,
                        OP_LOGE(opName_, "gmmX shape %lu not match gmmWeight shape %lu !", H1_, gmmWeightDim1_),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(N1_ != gmmWeightDim2_,
                        OP_LOGE(opName_, "y shape N %lu not match gmmWeight shape N %lu !", N1_, gmmWeightDim2_),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GmmAlltoAllvTilingStruct::CheckParamsRelationGmm()
{
    return CheckParamsRelationGmmTransShape();
}

ge::graphStatus GmmAlltoAllvTilingStruct::CheckParamsRelationMmTransShape()
{
    if (!hasSharedMm_) {
        return ge::GRAPH_SUCCESS;
    }
    if (isMmWeightTrans_) {
        OP_TILING_CHECK(H2_ != mmWeightDim1_,
                        OP_LOGE(opName_, "mmX shape %lu not match mmWeight shape %lu !", H2_, mmWeightDim1_),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(N2_ != mmWeightDim0_,
                        OP_LOGE(opName_, "mmY shape N %lu not match mmWeight shape N %lu !", N2_, mmWeightDim0_),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(H2_ != mmWeightDim0_,
                        OP_LOGE(opName_, "mmX shape %lu not match mmWeight shape %lu !", H2_, mmWeightDim0_),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(N2_ != mmWeightDim1_,
                        OP_LOGE(opName_, "mmY shape N %lu not match mmWeight shape N %lu !", N2_, mmWeightDim1_),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GmmAlltoAllvTilingStruct::CheckParamsRelationMm()
{
    return CheckParamsRelationMmTransShape();
}

ge::graphStatus GmmAlltoAllvTilingStruct::CheckParamsAttrEpAndSetLocalParams()
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
    epWorldSize_ = rankDim;

    bool isEpNumMatch = (ep_ > 0) && (ep_ < 33);
    if (!isEpNumMatch) {
        OP_LOGE(opName_, "ep (experts per rank) should be in range (0, %ld], but got %lu.", MAX_EXPERT_NUM_PER_RANK,
                ep_);
        return ge::GRAPH_FAILED;
    }
    uint64_t expertNum = ep_ * rankDim;
    OP_TILING_CHECK(
        expertNum > MAX_EXPERT_NUM,
        OP_LOGE(opName_, "experts(ep * epWorldSize) max is %lu, but now is %lu !", MAX_EXPERT_NUM, expertNum),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GmmAlltoAllvTilingStruct::CheckAndSetSendRecvCountsAttr()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    uint64_t expertNum = ep_ * epWorldSize_;
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
    OP_TILING_CHECK(sendCountsSum != A_,
                    OP_LOGE(opName_, "sendCountsSum %lu should be equal to A %lu !", sendCountsSum, A_),
                    return ge::GRAPH_FAILED);
    uint64_t recvCountsSum = std::accumulate(recvCounts, recvCounts + recvCountsSize, 0ULL);
    OP_TILING_CHECK(recvCountsSum != BsK_,
                    OP_LOGE(opName_, "recvCountsSum %lu should be equal to BSK %lu !", recvCountsSum, BsK_),
                    return ge::GRAPH_FAILED);

    auto gmmQTilingCommonInfoPtr = &localTilingData_.taskTilingInfo;
    uint64_t maxCountsSize = std::min<uint64_t>(expertNum, MAX_EXPERT_NUM);
    for (uint64_t i = 0; i < maxCountsSize; i++) {
        OP_TILING_CHECK(sendCounts[i] < 0, OP_LOGE(opName_, "sendCounts value %ld should not be < 0 !", sendCounts[i]),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(recvCounts[i] < 0, OP_LOGE(opName_, "recvCounts value %ld should not be < 0 !", recvCounts[i]),
                        return ge::GRAPH_FAILED);
        gmmQTilingCommonInfoPtr->sendCnt[i] = static_cast<int32_t>(sendCounts[i]);
        gmmQTilingCommonInfoPtr->recvCnt[i] = static_cast<int32_t>(recvCounts[i]);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GmmAlltoAllvTilingStruct::CheckLocalParams()
{
    OP_TILING_CHECK((H1_ == 0) || (H1_ >= MAX_H1_VALUE),
                    OP_LOGE(opName_, "H1 should be in range (0, %lu), but got %lu.", MAX_H1_VALUE, H1_),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((BsK_ >= MAX_BSK_VALUE),
                    OP_LOGE(opName_, "BSK should be in range (0, %lu), but got %lu.", MAX_BSK_VALUE, BsK_),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((N1_ == 0) || (N1_ >= MAX_N1_VALUE),
                    OP_LOGE(opName_, "N1 should be in range (0, %lu), but got %lu.", MAX_N1_VALUE, N1_),
                    return ge::GRAPH_FAILED);
    if (!hasSharedMm_) {
        return ge::GRAPH_SUCCESS;
    }
    OP_TILING_CHECK(
        (Bs_ == 0) || (BsK_ % Bs_ != 0),
        OP_LOGE(opName_, "BSK should be an integer multiple of BS, but got BSK[%lu] and BS[%lu].", BsK_, Bs_),
        return ge::GRAPH_FAILED);
    uint64_t k = BsK_ / Bs_;
    OP_TILING_CHECK((k < MIN_K_VALUE) || (k > MAX_K_VALUE),
                    OP_LOGE(opName_, "K (BSK / BS) should be in [%lu, %lu], but got %lu.", MIN_K_VALUE, MAX_K_VALUE, k),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((H2_ == 0) || (H2_ > MAX_SHARED_H_SHAPE_SIZE),
                    OP_LOGE(opName_, "H2 should be in range (0, %lu), but got %lu.", MAX_SHARED_H_SHAPE_SIZE, H2_),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((N2_ == 0) || (N2_ >= MAX_N2_VALUE),
                    OP_LOGE(opName_, "N2 should be in range (0, %lu), but got %lu.", MAX_N2_VALUE, N2_),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GmmAlltoAllvTilingStruct::CheckParamsRelationAndSetLocalParams()
{
    auto status = CheckParamsRelationGmm();
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckParamsRelationMm();
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckParamsAttrEpAndSetLocalParams();
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckAndSetSendRecvCountsAttr();
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckLocalParams();
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GmmAlltoAllvTilingStruct::CheckAndSetInputOutputInfo()
{
    auto sendCountsTensorShape = context_->GetOptionalInputShape(SEND_COUNTS_TENSOR_OPTIONAL_INDEX);
    auto recvCountsTensorShape = context_->GetOptionalInputShape(RECV_COUNTS_TENSOR_OPTIONAL_INDEX);
    OP_TILING_CHECK(sendCountsTensorShape != nullptr || recvCountsTensorShape != nullptr,
                    OP_LOGE(opName_, "sendCountsTensor and recvCountsTensor should all be nullptr now!"),
                    return ge::GRAPH_FAILED);

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
    hasSharedMm_ = !isMmXNull;

    auto status = CheckAndSetLocalParams();
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckParamsRelationAndSetLocalParams();
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GmmAlltoAllvTilingStruct::SetTilingCommonInfo()
{
    auto gmmQTilingCommonInfoPtr = &localTilingData_.taskTilingInfo;
    gmmQTilingCommonInfoPtr->BSK = BsK_;
    gmmQTilingCommonInfoPtr->BS = Bs_;
    gmmQTilingCommonInfoPtr->H1 = H1_;
    gmmQTilingCommonInfoPtr->H2 = H2_;
    gmmQTilingCommonInfoPtr->A = A_;
    gmmQTilingCommonInfoPtr->N1 = N1_;
    gmmQTilingCommonInfoPtr->N2 = N2_;
    gmmQTilingCommonInfoPtr->epWorldSize = epWorldSize_;
    gmmQTilingCommonInfoPtr->e = ep_;
    gmmQTilingCommonInfoPtr->mainLoopExpertNum = 1;
    gmmQTilingCommonInfoPtr->tailLoopExpertNum = 1;
    gmmQTilingCommonInfoPtr->totalLoopCount = ep_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GmmAlltoAllvTilingStruct::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, OP_LOGE(opName_, "Failed to get platform info."), return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    npuArch_ = ascendcPlatform.GetCurNpuArch();
    libApiWorkSpaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
    aicCoreNum_ = ascendcPlatform.GetCoreNumAic();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GmmAlltoAllvTilingStruct::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GmmAlltoAllvTilingStruct::SetGmmA2avWorkspaceInfo()
{
    auto gmmYDtypeSize = mc2tiling::GetDataTypeSize(opName_, gmmYDtype_);
    gmmResultLen_ = mc2tiling::AlignUp(A_ * N1_ * gmmYDtypeSize, ALIGN_ADDR_LEN);
    localTilingData_.workspaceInfo.wsGmmOutputSize = gmmResultLen_;
    constexpr uint64_t ptrTableSize = 128;
    uint64_t gmmComputeWsSize = ep_ * sizeof(int64_t) + ptrTableSize;
    localTilingData_.workspaceInfo.wsGmmComputeWorkspaceSize = mc2tiling::AlignUp(gmmComputeWsSize, ALIGN_ADDR_LEN);
    localTilingData_.workspaceInfo.wsSharedGmmComputeWorkspaceSize =
        mc2tiling::AlignUp(gmmComputeWsSize, ALIGN_ADDR_LEN);
    workSpaceSize_ = libApiWorkSpaceSize_ + gmmResultLen_ + localTilingData_.workspaceInfo.wsGmmComputeWorkspaceSize +
                     localTilingData_.workspaceInfo.wsSharedGmmComputeWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GmmAlltoAllvTilingStruct::DoGMMTiling()
{
    Mc2Tiling::Mc2GroupedMatmul::QuantGmmAlltoAllvParamsInfo gmmParams;
    gmmParams.opName = opName_;
    gmmParams.A = A_;
    gmmParams.H1 = H1_;
    gmmParams.ep = ep_;
    gmmParams.BsK = BsK_;
    gmmParams.N1 = N1_;
    gmmParams.Bs = Bs_;
    gmmParams.H2 = H2_;
    gmmParams.N2 = N2_;
    gmmParams.epWorldSize = epWorldSize_;
    gmmParams.aivCoreNum = aivCoreNum_;
    gmmParams.aicCoreNum = aicCoreNum_;
    gmmParams.gmmWeightDim1 = gmmWeightDim1_;
    gmmParams.gmmWeightDim2 = gmmWeightDim2_;
    gmmParams.mmWeightDim0 = mmWeightDim0_;
    gmmParams.mmWeightDim1 = mmWeightDim1_;
    gmmParams.gmmXQuantMode = 0;
    gmmParams.gmmWeightQuantMode = 0;
    gmmParams.mmXQuantMode = 0;
    gmmParams.mmWeightQuantMode = 0;
    gmmParams.gmmQuantSuit = 0;
    gmmParams.mmQuantSuit = 0;
    gmmParams.commQuantMode = 0;
    gmmParams.commQuantDtype = 0;
    gmmParams.attrGmmYDtype = 0;
    gmmParams.attrMmYDtype = 0;
    gmmParams.groupSize = 0;
    gmmParams.hasSharedMm = hasSharedMm_;
    gmmParams.isGmmWeightTrans = isGmmWeightTrans_;
    gmmParams.isMmWeightTrans = isMmWeightTrans_;
    gmmParams.gmmXDtype = gmmXDtype_;
    gmmParams.gmmWeightDtype = gmmXDtype_;
    gmmParams.gmmXScaleDtype = gmmXDtype_;
    gmmParams.gmmWeightScaleDtype = gmmXDtype_;
    gmmParams.mmXDtype = gmmXDtype_;
    gmmParams.mmWeightDtype = gmmXDtype_;
    gmmParams.mmXScaleDtype = gmmXDtype_;
    gmmParams.mmWeightScaleDtype = gmmXDtype_;
    gmmParams.gmmYDtype = gmmYDtype_;
    gmmParams.mmYDtype = gmmYDtype_;
    gmmParams.yDtype = yDtype_;

    QuantGroupedMatmulAllToAllvAdapter gmmTile(context_);
    GE_ASSERT_GRAPH_SUCCESS(gmmTile.SetCommonInputParams(gmmParams));

    uint64_t mMaxSize = 0;
    uint64_t mSize = 0;
    for (uint64_t expertIdx = 0; expertIdx < ep_; expertIdx++) {
        mSize = 0;
        for (uint64_t i = 0; i < epWorldSize_; i++) {
            mSize += localTilingData_.taskTilingInfo.sendCnt[i * ep_ + expertIdx];
        }
        mMaxSize = std::max(mSize, mMaxSize);
    }

    GE_ASSERT_GRAPH_SUCCESS(gmmTile.SetGroupExpertInputParameters(gmmParams, mMaxSize));
    GE_ASSERT_GRAPH_SUCCESS(gmmTile.Process());
    localTilingData_.gmmBaseTiling = gmmTile.GetGmmQuantTilingAdapterData();

    if (!hasSharedMm_) {
        return ge::GRAPH_SUCCESS;
    }
    GE_ASSERT_GRAPH_SUCCESS(gmmTile.SetSharedExpertInputParameters(gmmParams));
    GE_ASSERT_GRAPH_SUCCESS(gmmTile.Process());
    localTilingData_.sharedGmmTiling = gmmTile.GetGmmQuantTilingAdapterData();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GmmAlltoAllvTilingStruct::SetHcclTiling()
{
    uint32_t alltoAllvCmd = 8U;
    std::string alltoAllvConfig = "AlltoAll=level0:fullmesh;level1:pairwise";
    auto attrs = context_->GetAttrs();
    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    const uint32_t alltoAllvReduceType = 0u;
    auto outputDataType = context_->GetOutputDesc(OUTPUT_Y_INDEX)->GetDataType();
    OP_TILING_CHECK(mc2tiling::HCCL_DATA_TYPE.find(outputDataType) == mc2tiling::HCCL_DATA_TYPE.end(),
                    OP_LOGE(opName_, "%s is Unsupported outputdata type!", Ops::Base::ToString(outputDataType).c_str()),
                    return ge::GRAPH_FAILED);
    auto alltoAllvDstDataType = static_cast<uint8_t>(mc2tiling::HCCL_DATA_TYPE.find(outputDataType)->second);
    auto alltoAllvSrcDataType = static_cast<uint8_t>(mc2tiling::HCCL_DATA_TYPE.find(outputDataType)->second);
    Mc2CcTilingConfig hcclCcTilingConfig(groupEpPtr, alltoAllvCmd, alltoAllvConfig, alltoAllvReduceType,
                                         alltoAllvDstDataType, alltoAllvSrcDataType);
    uint8_t commMode = Mc2Comm::GetCommModeFromEnv();
    if (commMode == Mc2Comm::COMM_MODE_AICPU) {
        hcclCcTilingConfig.SetCommEngine(Mc2Comm::ENGINE_AICPU);
    }
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(localTilingData_.hcclA2avTiling.hcclInitTiling) != 0,
                    OP_LOGE(opName_, "mc2CcTilingConfig GetTiling hcclInitTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(localTilingData_.hcclA2avTiling.a2avCcTiling) != 0,
                    OP_LOGE(opName_, "mc2CcTilingConfig GetTiling alltoAllvCcTiling failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool GmmAlltoAllvTilingStruct::IsCapable()
{
    return true;
}

ge::graphStatus GmmAlltoAllvTilingStruct::DoOpTiling()
{
    opName_ = context_->GetNodeName();
    GE_ASSERT_GRAPH_SUCCESS(CheckAndSetInputOutputInfo());
    GE_ASSERT_GRAPH_SUCCESS(SetTilingCommonInfo());
    GE_ASSERT_GRAPH_SUCCESS(DoGMMTiling());
    GE_ASSERT_GRAPH_SUCCESS(SetHcclTiling());
    GE_ASSERT_GRAPH_SUCCESS(SetGmmA2avWorkspaceInfo());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GmmAlltoAllvTilingStruct::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(opName_, "get workspace failed"), return ge::GRAPH_FAILED);
    workspaces[0] = workSpaceSize_;
    OP_LOGD(opName_, "Workspaces[0] size=%ld", workspaces[0]);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GmmAlltoAllvTilingStruct::PostTiling()
{
    context_->SetBlockDim(aicCoreNum_);
    QuantGmmA2avTilingData *outTilingData = context_->GetTilingData<QuantGmmA2avTilingData>();
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

uint64_t GmmAlltoAllvTilingStruct::GetTilingKey() const
{
    uint8_t commMode = Mc2Comm::GetCommModeFromEnv();
    auto platformInfo = context_->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    NpuArch npuArch = ascendcPlatform.GetCurNpuArch();
    if (npuArch != NpuArch::DAV_3510) {
        commMode = Mc2Comm::COMM_MODE_AICPU;
    }
    const uint64_t tilingKey = GET_TPL_TILING_KEY(hasSharedMm_, isGmmWeightTrans_, isMmWeightTrans_, commMode);
    OP_LOGD(opName_, "GET_TPL_TILING_KEY: [%d,%d,%d,%d], TilingKey is [%lu].", hasSharedMm_, isGmmWeightTrans_,
            isMmWeightTrans_, commMode, tilingKey);
    return tilingKey;
}

ge::graphStatus GmmAlltoAllvTilingStruct::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
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