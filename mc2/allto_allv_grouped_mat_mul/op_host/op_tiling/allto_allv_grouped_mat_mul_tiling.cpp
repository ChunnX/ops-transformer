/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file allto_allv_grouped_mat_mul_tiling.cc
 * \brief
 */

#include "allto_allv_grouped_mat_mul_tiling.h"
#include <algorithm>
#include <cstddef>
#include <numeric>
#include "../../allto_allv_quant_grouped_mat_mul/op_host/op_tiling/allto_allv_quant_grouped_mat_mul_tiling_base.h"
#include "../../op_kernel/allto_allv_grouped_mat_mul_tiling_key.h"
#include "mc2_comm_utils.h"

namespace optiling {
constexpr uint32_t SINGLE_GROUP_NUM = 1;
constexpr uint32_t GMM_ACT_TYPE_NONE = 0;

using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;
namespace {
    static const char* A_INNER_DEBUG = "AlltoAllvGroupedMatMul Tiling";
}
constexpr uint32_t OUTPUT_Y_INDEX = 0U;
constexpr uint32_t NUM_ZERO = 0;
constexpr uint32_t NUM_ONE = 1;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t E_MAX_VALUE_NON_QUANT = 48;
constexpr uint32_t MAX_BSK = 52428800;
constexpr uint32_t MAX_SHAPE_SIZE = 65536;

ge::graphStatus AlltoAllvGmmTiling::GetContextAttr(const gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(A_INNER_DEBUG, "Attrs is null, expected non-null."),
        return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    OP_TILING_CHECK((groupEpPtr == nullptr) || (epWorldSizePtr == nullptr),
        OP_LOGE(A_INNER_DEBUG, "Group or epWorldSize attr is null, expected non-null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*epWorldSizePtr == 0, OP_LOGE(A_INNER_DEBUG, "EpWorldSize is 0, expected positive value."),
        return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    std::vector<int64_t> validEpWorldSizeValues;
    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        validEpWorldSizeValues = {2, 4, 8, 16, 32, 64, 128, 256};
    } else {
        validEpWorldSizeValues = {8, 16, 32, 64, 128};
    }
    OP_TILING_CHECK(std::find(validEpWorldSizeValues.begin(), validEpWorldSizeValues.end(),
        *epWorldSizePtr) == validEpWorldSizeValues.end(),
        OP_LOGE(A_INNER_DEBUG, "EpWorldSize[%lld] is not in valid set for this platform.", *epWorldSizePtr),
        return ge::GRAPH_FAILED);

    auto mmXStorageShape = context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX);
    tilingData->isPermuteOut = (mmXStorageShape != nullptr);

    groupPtr_ = groupEpPtr;
    group_ = groupEpPtr;
    epWorldSizePtr_ = epWorldSizePtr;

    auto transGmmWeightPtr = attrs->GetAttrPointer<bool>(NON_QUANT_ATTR_TRANS_GMM_WEIGHT_INDEX);
    OP_TILING_CHECK(transGmmWeightPtr == nullptr,
        OP_LOGE(A_INNER_DEBUG, "TransGmmWeight attr is null, expected non-null."), return ge::GRAPH_FAILED);
    transGmmWeight_ = *transGmmWeightPtr;

    auto transMmWeightPtr = attrs->GetAttrPointer<bool>(NON_QUANT_ATTR_TRANS_MM_WEIGHT_INDEX);
    OP_TILING_CHECK(transMmWeightPtr == nullptr,
        OP_LOGE(A_INNER_DEBUG, "TransMmWeight attr is null, expected non-null."), return ge::GRAPH_FAILED);
    transMmWeight_ = (mmXStorageShape != nullptr) ? *transMmWeightPtr : false;

    OP_LOGI(A_INNER_DEBUG, "EpGroup is %s, epWorldSize is %lld.", groupPtr_, *epWorldSizePtr_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::GetShapeAndFormat(const gert::TilingContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckMKN(const gert::TilingContext* context)
{
    (void)context;
    OP_TILING_CHECK(
        mmDataTypeSize == 0,
        OP_LOGE(A_INNER_DEBUG, "Matmul dtype[%s] size is 0, expected positive size.",
            TypeUtils::DataTypeToAscendString(mmXDataType_).GetString()),
        return ge::GRAPH_FAILED);
    uint32_t numInOneBlk = ONE_BLK_SIZE / mmDataTypeSize;
    OP_TILING_CHECK(numInOneBlk == 0,
        OP_LOGE(A_INNER_DEBUG, "NumInOneBlk is 0, expected positive value."), return ge::GRAPH_FAILED);
    int64_t maxMKN = INT_MAX / numInOneBlk * numInOneBlk;
    OP_TILING_CHECK(
        maxM_ > maxMKN || maxN_ > maxMKN || maxK_ > maxMKN,
        OP_LOGE(A_INNER_DEBUG, "Gmm M[%d], N[%d], or K[%d] exceeds 32B-aligned max[%lld], expected within range.",
            maxM_, maxN_, maxK_, maxMKN),
        return ge::GRAPH_FAILED);
    if (context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX) != nullptr) {
        OP_TILING_CHECK(
            maxMForMM_ > maxMKN || maxNForMM_ > maxMKN || maxKForMM_ > maxMKN,
            OP_LOGE(A_INNER_DEBUG, "Mm M[%d], N[%d], or K[%d] exceeds 32B-aligned max[%lld], expected within range.",
                maxMForMM_, maxNForMM_, maxKForMM_, maxMKN),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckSendRecvDataVolumn(const gert::TilingContext* context) const
{
    uint64_t eExpert = tilingData->taskTilingInfo.e;
    uint64_t epWorldSize = tilingData->taskTilingInfo.epWorldSize;

    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(A_INNER_DEBUG, "Attrs is null, expected non-null."),
        return ge::GRAPH_FAILED);

    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK((sendCountsPtr == nullptr) || (recvCountsPtr == nullptr),
        OP_LOGE(A_INNER_DEBUG, "SendCounts or recvCounts attr is null, expected non-null."), return ge::GRAPH_FAILED);

    const int64_t* sendCountsLocal = static_cast<const int64_t*>(sendCountsPtr->GetData());
    const int64_t* recvCountsLocal = static_cast<const int64_t*>(recvCountsPtr->GetData());
    uint64_t recvSum = 0U;
    uint64_t sendSum = 0U;
    uint64_t H1 = tilingData->taskTilingInfo.H1;
    uint64_t bsk = context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(0);
    uint64_t a = context->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape().GetDim(0);
    uint64_t sendCountsSize = sendCountsPtr->GetSize();
    uint64_t recvCountsSize = recvCountsPtr->GetSize();
    auto platformInfo = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    if (NeedToCheckCounts()) {
        for (uint64_t i = 1U; i <= epWorldSize; i++) {
            recvSum = 0U;
            sendSum = 0U;
            for (uint64_t j = (i - 1U) * eExpert; j <= i * eExpert - 1U; j++) {
                OP_TILING_CHECK((sendCountsLocal[j] < NUM_ZERO) || (sendCountsLocal[j] > static_cast<int64_t>(bsk)),
                    OP_LOGE(A_INNER_DEBUG, "SendCounts[%llu] value is %lld, expected in range [0, BSK[%llu]].",
                        j, sendCountsLocal[j], bsk),
                    return ge::GRAPH_FAILED);
                OP_TILING_CHECK((recvCountsLocal[j] < NUM_ZERO) || (recvCountsLocal[j] > static_cast<int64_t>(a)),
                    OP_LOGE(A_INNER_DEBUG, "RecvCounts[%llu] value is %lld, expected in range [0, A[%llu]].",
                        j, recvCountsLocal[j], a),
                    return ge::GRAPH_FAILED);
                recvSum += static_cast<uint64_t>(recvCountsLocal[j]) * H1 * 2U;
                sendSum += static_cast<uint64_t>(sendCountsLocal[j]) * H1 * 2U;
            }
        }
    }

    uint64_t sendCountsSum = std::accumulate(sendCountsLocal, sendCountsLocal + sendCountsSize, 0ULL);
    OP_TILING_CHECK(sendCountsSum != bsk,
        OP_LOGE(A_INNER_DEBUG, "Sum of sendCounts is %llu, expected equal to BSK[%llu].", sendCountsSum, bsk),
        return ge::GRAPH_FAILED);
    uint64_t recvCountsSum = std::accumulate(recvCountsLocal, recvCountsLocal + recvCountsSize, 0ULL);
    OP_TILING_CHECK(recvCountsSum != a,
        OP_LOGE(A_INNER_DEBUG, "Sum of recvCounts is %llu, expected equal to A[%llu].", recvCountsSum, a),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckShapeSize(const gert::TilingContext* context) const
{
    OP_TILING_CHECK(
        (context->GetInputShape(GMM_X_INDEX) == nullptr) || (context->GetInputShape(GMM_WEIGHT_INDEX) == nullptr),
        OP_LOGE(A_INNER_DEBUG, "GmmX or gmmWeight input shape is null, expected non-null."), return ge::GRAPH_FAILED);

    uint64_t BSK = context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(0);
    if (BSK <= NUM_ZERO || BSK >= MAX_BSK) {
        OP_LOGE(A_INNER_DEBUG, "BSK value is %llu, expected in range [1, %llu).", BSK, static_cast<uint64_t>(MAX_BSK));
        return ge::GRAPH_FAILED;
    }

    uint64_t H1 = context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(1);
    if (H1 <= NUM_ZERO || H1 >= MAX_SHAPE_SIZE) {
        OP_LOGE(A_INNER_DEBUG, "H1 value is %llu, expected in range [1, %llu).",
            H1, static_cast<uint64_t>(MAX_SHAPE_SIZE));
        return ge::GRAPH_FAILED;
    }

    uint64_t N1 = context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(1);
    if (N1 <= NUM_ZERO || N1 >= MAX_SHAPE_SIZE) {
        OP_LOGE(A_INNER_DEBUG, "N1 value is %llu, expected in range [1, %llu).",
            N1, static_cast<uint64_t>(MAX_SHAPE_SIZE));
        return ge::GRAPH_FAILED;
    }

    if (context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX) != nullptr) {
        uint64_t BS = context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX)->GetStorageShape().GetDim(0);
        if (BS <= NUM_ZERO || BS >= MAX_BSK) {
            OP_LOGE(A_INNER_DEBUG, "BS value is %llu, expected in range [1, %llu).",
                BS, static_cast<uint64_t>(MAX_BSK));
            return ge::GRAPH_FAILED;
        }
        uint64_t H2 = context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX)->GetStorageShape().GetDim(1);
        if (H2 <= NUM_ZERO || H2 > H2_MAX_VALUE) {
            OP_LOGE(A_INNER_DEBUG, "H2 value is %llu, expected in range [1, %llu].",
                H2, H2_MAX_VALUE);
            return ge::GRAPH_FAILED;
        }
        uint64_t n2DimIndex = transMmWeight_ ? DIM_ZERO : DIM_ONE;
        uint64_t N2 = context->GetInputShape(NON_QUANT_MM_WEIGHT_INDEX)->GetStorageShape().GetDim(n2DimIndex);
        if (N2 <= NUM_ZERO || N2 >= MAX_SHAPE_SIZE) {
            OP_LOGE(A_INNER_DEBUG, "N2 value is %llu, expected in range [1, %llu).",
                N2, static_cast<uint64_t>(MAX_SHAPE_SIZE));
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckAttrsShapeSize(const gert::TilingContext* context) const
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(A_INNER_DEBUG, "Attrs is null, expected non-null."),
        return ge::GRAPH_FAILED);

    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(sendCountsPtr == nullptr, OP_LOGE(A_INNER_DEBUG, "SendCounts attr is null, expected non-null."),
        return ge::GRAPH_FAILED);

    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(recvCountsPtr == nullptr, OP_LOGE(A_INNER_DEBUG, "RecvCounts attr is null, expected non-null."),
        return ge::GRAPH_FAILED);

    uint64_t sendCountsSize = sendCountsPtr->GetSize();
    uint64_t recvCountsSize = recvCountsPtr->GetSize();
    if (sendCountsSize != recvCountsSize) {
        OP_LOGE(A_INNER_DEBUG, "SendCounts size is %llu, recvCounts size is %llu, expected equal.",
            sendCountsSize, recvCountsSize);
        return ge::GRAPH_FAILED;
    }

    if (sendCountsSize >= MC2KernelTemplate::MAX_EXPERT_SIZE) {
        OP_LOGE(A_INNER_DEBUG, "SendCounts size is %llu, expected less than %u.",
            sendCountsSize, MC2KernelTemplate::MAX_EXPERT_SIZE);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckAttrsShapeRelation(const gert::TilingContext* context) const
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(A_INNER_DEBUG, "Attrs is null, expected non-null."),
        return ge::GRAPH_FAILED);

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    OP_TILING_CHECK(epWorldSizePtr == nullptr,
        OP_LOGE(A_INNER_DEBUG, "EpWorldSize attr is null, expected non-null."), return ge::GRAPH_FAILED);

    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(sendCountsPtr == nullptr,
        OP_LOGE(A_INNER_DEBUG, "SendCounts attr is null, expected non-null."), return ge::GRAPH_FAILED);

    const int64_t* sendCountsAttr = static_cast<const int64_t*>(sendCountsPtr->GetData());
    OP_TILING_CHECK(sendCountsAttr == nullptr,
        OP_LOGE(A_INNER_DEBUG, "SendCounts data is null, expected non-null."), return ge::GRAPH_FAILED);
    uint64_t sendCountsSize = sendCountsPtr->GetSize();
    uint64_t eExpert = sendCountsSize / (*epWorldSizePtr);

    if (sendCountsSize != e_ * (*epWorldSizePtr)) {
        OP_LOGE(A_INNER_DEBUG, "SendCounts size is %llu, expected equal to e[%llu] * epWorldSize[%lld].",
            sendCountsSize, e_, *epWorldSizePtr);
        return ge::GRAPH_FAILED;
    }

    if (eExpert <= NUM_ZERO || eExpert >= E_MAX_VALUE_NON_QUANT) {
        OP_LOGE(A_INNER_DEBUG, "Expert num per rank is %llu, expected in range [1, %u).",
            eExpert, E_MAX_VALUE_NON_QUANT);
        return ge::GRAPH_FAILED;
    }

    for (uint64_t i = 0; i < sendCountsSize; i++) {
        if (sendCountsAttr[i] < NUM_ZERO) {
            OP_LOGE(A_INNER_DEBUG, "SendCounts[%llu] value is negative, expected non-negative.", i);
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckShapeRelation(const gert::TilingContext* context) const
{
    auto gmmXShape = context->GetInputShape(GMM_X_INDEX)->GetStorageShape();
    auto gmmWeightShape = context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape();
    uint64_t H1 = gmmXShape.GetDim(1);
    uint64_t h1DimIndex = transGmmWeight_ ? DIM_TWO : DIM_ONE;
    uint64_t H2 = gmmWeightShape.GetDim(h1DimIndex);
    if (H1 != H2) {
        OP_LOGE(A_INNER_DEBUG, "GmmX H1 is %llu, gmmWeight H1 is %llu, expected equal.",
            H1, H2);
        return ge::GRAPH_FAILED;
    }

    // Check gmmY output: N1 dimension should match gmmWeight
    OP_TILING_CHECK(context->GetOutputShape(OUTPUT_GMM_Y_INDEX) == nullptr,
        OP_LOGE(A_INNER_DEBUG, "GmmY output shape is null, expected non-null."),
        return ge::GRAPH_FAILED);
    auto gmmYShape = context->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape();
    uint64_t n1DimIndex = transGmmWeight_ ? DIM_ONE : DIM_TWO;
    uint64_t N1 = gmmWeightShape.GetDim(n1DimIndex);
    uint64_t gmmYDim1 = gmmYShape.GetDim(DIM_ONE);
    if (gmmYDim1 != N1) {
        OP_LOGE(A_INNER_DEBUG, "GmmY N1 dim is %llu, gmmWeight N1 is %llu, expected equal.",
            gmmYDim1, N1);
        return ge::GRAPH_FAILED;
    }

    // Check permuteOut output when permuteOutFlag is true
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(A_INNER_DEBUG, "Attrs is null, expected non-null."),
        return ge::GRAPH_FAILED);
    auto permuteOutFlagPtr = attrs->GetAttrPointer<bool>(NON_QUANT_ATTR_PERMUTE_OUT_FLAG_INDEX);
    if (permuteOutFlagPtr != nullptr && *permuteOutFlagPtr) {
        OP_TILING_CHECK(context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX) == nullptr,
            OP_LOGE(A_INNER_DEBUG,
                "PermuteOut output shape is null when permuteOutFlag is true."),
            return ge::GRAPH_FAILED);
        auto permuteOutShape =
            context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX)->GetStorageShape();
        uint64_t permuteOutA = permuteOutShape.GetDim(DIM_ZERO);
        uint64_t gmmYA = gmmYShape.GetDim(DIM_ZERO);
        if (permuteOutA != gmmYA) {
            OP_LOGE(A_INNER_DEBUG,
                "PermuteOut A dim is %llu, gmmY A is %llu, expected equal.",
                permuteOutA, gmmYA);
            return ge::GRAPH_FAILED;
        }
        uint64_t permuteOutH1 = permuteOutShape.GetDim(DIM_ONE);
        if (permuteOutH1 != H1) {
            OP_LOGE(A_INNER_DEBUG,
                "PermuteOut H1 dim is %llu, gmmX H1 is %llu, expected equal.",
                permuteOutH1, H1);
            return ge::GRAPH_FAILED;
        }
    }

    if (tilingData->isPermuteOut) {
        auto mmXShape = context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX)->GetStorageShape();
        auto mmWeightShape = context->GetInputShape(NON_QUANT_MM_WEIGHT_INDEX)->GetStorageShape();
        uint64_t BS = mmXShape.GetDim(DIM_ZERO);
        uint64_t H = mmXShape.GetDim(1);
        uint64_t h2DimIndex = transMmWeight_ ? DIM_ONE : DIM_ZERO;
        uint64_t H2MM = mmWeightShape.GetDim(h2DimIndex);
        if (H != H2MM) {
            OP_LOGE(A_INNER_DEBUG, "MmX H2 is %llu, mmWeight H2 is %llu, expected equal.",
                H, H2MM);
            return ge::GRAPH_FAILED;
        }

        uint64_t BSK = gmmXShape.GetDim(0);
        if (BSK % BS != 0) {
            OP_LOGE(A_INNER_DEBUG, "BSK[%llu] is not divisible by BS[%llu].", BSK, BS);
            return ge::GRAPH_FAILED;
        }
        uint64_t K = BSK / BS;
        if (K < K_MIN_VALUE || K > K_MAX_VALUE) {
            OP_LOGE(A_INNER_DEBUG, "K value is %llu, expected in range [%llu, %llu].",
                K, K_MIN_VALUE, K_MAX_VALUE);
            return ge::GRAPH_FAILED;
        }

        // Check mmY output when mmX is present
        OP_TILING_CHECK(context->GetOutputShape(OUTPUT_MM_Y_INDEX) == nullptr,
            OP_LOGE(A_INNER_DEBUG,
                "MmY output shape is null when mmX is present, expected non-null."),
            return ge::GRAPH_FAILED);
        auto mmYShape = context->GetOutputShape(OUTPUT_MM_Y_INDEX)->GetStorageShape();
        uint64_t mmYBS = mmYShape.GetDim(DIM_ZERO);
        if (mmYBS != BS) {
            OP_LOGE(A_INNER_DEBUG,
                "MmY BS dim is %llu, mmX BS is %llu, expected equal.",
                mmYBS, BS);
            return ge::GRAPH_FAILED;
        }
        uint64_t n2DimIndex = transMmWeight_ ? DIM_ZERO : DIM_ONE;
        uint64_t N2 = mmWeightShape.GetDim(n2DimIndex);
        uint64_t mmYDim1 = mmYShape.GetDim(DIM_ONE);
        if (mmYDim1 != N2) {
            OP_LOGE(A_INNER_DEBUG,
                "MmY N2 dim is %llu, mmWeight N2 is %llu, expected equal.",
                mmYDim1, N2);
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckShapeDims(const gert::TilingContext* context)
{
    auto gmmXShape = context->GetInputShape(GMM_X_INDEX)->GetStorageShape();
    OP_TILING_CHECK(gmmXShape.GetDimNum() != DIM_TWO,
        OP_LOGE(A_INNER_DEBUG, "GmmX dim num is %llu, expected 2.", gmmXShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    uint64_t gmmXDim0 = gmmXShape.GetDim(0);
    OP_TILING_CHECK(gmmXDim0 > static_cast<uint64_t>(INT32_MAX),
        OP_LOGE(A_INNER_DEBUG, "GmmX dim0(BSK) is %llu, expected less than 2147483647.", gmmXDim0),
        return ge::GRAPH_FAILED);
    maxM_ = static_cast<int32_t>(gmmXDim0);
    uint64_t gmmXDim1 = gmmXShape.GetDim(1);
    maxK_ = static_cast<int32_t>(gmmXDim1);

    auto gmmWeightShape = context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape();
    OP_TILING_CHECK(gmmWeightShape.GetDimNum() != DIM_THREE,
        OP_LOGE(A_INNER_DEBUG, "GmmWeight dim num is %llu, expected 3.", gmmWeightShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    e_ = gmmWeightShape.GetDim(0);
    uint64_t h1DimIndex = transGmmWeight_ ? DIM_TWO : DIM_ONE;
    uint64_t n1DimIndex = transGmmWeight_ ? DIM_ONE : DIM_TWO;
    maxK_ = static_cast<int32_t>(gmmWeightShape.GetDim(h1DimIndex));
    maxN_ = static_cast<int32_t>(gmmWeightShape.GetDim(n1DimIndex));

    mmXDataType_ = context->GetInputDesc(GMM_X_INDEX)->GetDataType();
    gmmXDataType_ = context->GetInputDesc(GMM_X_INDEX)->GetDataType();
    gmmWeightDataType_ = context->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType();
    mmDataTypeSize = GetSizeByDataType(mmXDataType_);

    if (context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX) != nullptr) {
        auto mmXShape = context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX)->GetStorageShape();
        OP_TILING_CHECK(mmXShape.GetDimNum() != DIM_TWO,
            OP_LOGE(A_INNER_DEBUG, "MmX dim num is %llu, expected 2.", mmXShape.GetDimNum()),
            return ge::GRAPH_FAILED);
        uint64_t mmXDim0 = mmXShape.GetDim(0);
        uint64_t mmXDim1 = mmXShape.GetDim(1);
        OP_TILING_CHECK(mmXDim0 > static_cast<uint64_t>(INT32_MAX),
            OP_LOGE(A_INNER_DEBUG, "MmX dim0(BS) is %llu, expected less than 2147483647.", mmXDim0),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(mmXDim1 > static_cast<uint64_t>(INT32_MAX),
            OP_LOGE(A_INNER_DEBUG, "MmX dim1(H2) is %llu, expected less than 2147483647.", mmXDim1),
            return ge::GRAPH_FAILED);
        maxMForMM_ = static_cast<int32_t>(mmXDim0);
        maxKForMM_ = static_cast<int32_t>(mmXDim1);
        mmXDataType_ = context->GetInputDesc(NON_QUANT_MM_X_INDEX)->GetDataType();
        mmWeightDataType_ = context->GetInputDesc(NON_QUANT_MM_WEIGHT_INDEX)->GetDataType();

        auto mmWeightShape = context->GetInputShape(NON_QUANT_MM_WEIGHT_INDEX)->GetStorageShape();
        OP_TILING_CHECK(mmWeightShape.GetDimNum() != DIM_TWO,
            OP_LOGE(A_INNER_DEBUG, "MmWeight dim num is %llu, expected 2.", mmWeightShape.GetDimNum()),
            return ge::GRAPH_FAILED);
        uint64_t kDimIndex = transMmWeight_ ? DIM_ONE : DIM_ZERO;
        uint64_t nDimIndex = transMmWeight_ ? DIM_ZERO : DIM_ONE;
        uint64_t mmKDim = mmWeightShape.GetDim(kDimIndex);
        uint64_t mmNDim = mmWeightShape.GetDim(nDimIndex);
        OP_TILING_CHECK(mmKDim > static_cast<uint64_t>(INT32_MAX),
            OP_LOGE(A_INNER_DEBUG, "MmWeight K dim is %llu, expected less than 2147483647.", mmKDim),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(mmNDim > static_cast<uint64_t>(INT32_MAX),
            OP_LOGE(A_INNER_DEBUG, "MmWeight N dim is %llu, expected less than 2147483647.", mmNDim),
            return ge::GRAPH_FAILED);
        maxKForMM_ = static_cast<int32_t>(mmKDim);
        maxNForMM_ = static_cast<int32_t>(mmNDim);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckDType(const gert::TilingContext* context) const
{
    ge::DataType gmmXDtype = context->GetInputDesc(GMM_X_INDEX)->GetDataType();
    ge::DataType gmmWeightDtype = context->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType();
    if (gmmXDtype != gmmWeightDtype) {
        OP_LOGE(A_INNER_DEBUG, "GmmX dtype is %s, gmmWeight dtype is %s, expected same dtype.",
            ge::TypeUtils::DataTypeToSerialString(gmmXDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(gmmWeightDtype).c_str());
        return ge::GRAPH_FAILED;
    }

    if (gmmXDtype != ge::DT_FLOAT16 && gmmXDtype != ge::DT_BF16) {
        OP_LOGE(A_INNER_DEBUG, "GmmX dtype is %s, expected FLOAT16 or BFLOAT16.",
            ge::TypeUtils::DataTypeToSerialString(gmmXDtype).c_str());
        return ge::GRAPH_FAILED;
    }

    if (context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX) != nullptr) {
        ge::DataType mmXDtype = context->GetInputDesc(NON_QUANT_MM_X_INDEX)->GetDataType();
        ge::DataType mmWeightDtype = context->GetInputDesc(NON_QUANT_MM_WEIGHT_INDEX)->GetDataType();
        if (mmXDtype != mmWeightDtype) {
            OP_LOGE(A_INNER_DEBUG, "MmX dtype is %s, mmWeight dtype is %s, expected same dtype.",
                ge::TypeUtils::DataTypeToSerialString(mmXDtype).c_str(),
                ge::TypeUtils::DataTypeToSerialString(mmWeightDtype).c_str());
            return ge::GRAPH_FAILED;
        }

        if (mmXDtype != ge::DT_FLOAT16 && mmXDtype != ge::DT_BF16) {
            OP_LOGE(A_INNER_DEBUG, "MmX dtype is %s, expected FLOAT16 or BFLOAT16.",
                ge::TypeUtils::DataTypeToSerialString(mmXDtype).c_str());
            return ge::GRAPH_FAILED;
        }

        if (gmmXDtype != mmXDtype) {
            OP_LOGE(A_INNER_DEBUG, "GmmX dtype is %s, mmX dtype is %s, expected same dtype.",
                ge::TypeUtils::DataTypeToSerialString(gmmXDtype).c_str(),
                ge::TypeUtils::DataTypeToSerialString(mmXDtype).c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckMmShapeDims(const gert::TilingContext* context) const
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::SetHcclTiling(const gert::TilingContext* context) const
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(A_INNER_DEBUG, "Attrs is null, expected non-null."),
        return ge::GRAPH_FAILED);

    auto tilingData = context_->GetTilingData<AlltoAllvGmmTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(A_INNER_DEBUG, "TilingData is null, expected non-null."),
        return ge::GRAPH_FAILED);

    uint32_t alltoAllvCmd = 8U;
    std::string alltoAllvConfig = "AlltoAll=level0:fullmesh;level1:pairwise";
    const uint32_t alltoAllvReduceType = 0u;

    mc2tiling::HcclDataType alltoallvHcclDataType =
        mc2tiling::ConvertGeTypeToHcclType(context_->GetNodeName(), gmmXDataType_);
    OP_TILING_CHECK(alltoallvHcclDataType == mc2tiling::HcclDataType::HCCL_DATA_TYPE_RESERVED,
        OP_LOGE(context_->GetNodeName(),
            "GmmX dtype[%s] is unsupported by HCCL, "
            "expected INT8/UINT8/INT16/UINT16/INT32/UINT32/"
            "FLOAT16/FLOAT/BFLOAT16/HIFLOAT8/FLOAT8_E4M3FN/FLOAT8_E5M2.",
            ge::TypeUtils::DataTypeToSerialString(gmmXDataType_).c_str()),
        return ge::GRAPH_FAILED);
    uint8_t alltoAllvDataType = static_cast<uint8_t>(alltoallvHcclDataType);

    Mc2CcTilingConfig hcclCcTilingConfig(group_, alltoAllvCmd, alltoAllvConfig, alltoAllvReduceType, alltoAllvDataType,
        alltoAllvDataType);

    uint8_t commMode = Mc2Comm::COMM_MODE_AICPU;
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        commMode = Mc2Comm::GetCommModeFromEnv();
    }
    OP_LOGD(context->GetNodeName(), "CommMode is %u.", commMode);
    if (commMode == Mc2Comm::COMM_MODE_AICPU) {
        hcclCcTilingConfig.SetCommEngine(Mc2Comm::ENGINE_AICPU);
    }

    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(tilingData->hcclA2avTilingInfo.hcclInitTiling) != 0,
        OP_LOGE(A_INNER_DEBUG, "HCCL init tiling config failed, expected success."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(tilingData->hcclA2avTilingInfo.a2avCcTiling) != 0,
        OP_LOGE(A_INNER_DEBUG, "HCCL alltoallv tiling config failed, expected success."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::setNumBlocks(gert::TilingContext* context)
{
    if (GetCommonPlatformInfo() != ge::GRAPH_SUCCESS) {
        OP_LOGE(A_INNER_DEBUG, "Failed to get common platform info.");
        return ge::GRAPH_FAILED;
    }
    if (CheckCommonPlatformInfo() != ge::GRAPH_SUCCESS) {
        OP_LOGE(A_INNER_DEBUG, "Common platform info check failed, expected valid UB/L1/L0C sizes.");
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    uint64_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t numBlocks = mc2tiling::GetNumBlocks(aicNum, aivNum, A_INNER_DEBUG);

    tilingData->taskTilingInfo.ubSize = ubSize_;
    context->SetBlockDim(static_cast<uint32_t>(numBlocks));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::Init()
{
    tilingData = context_->GetTilingData<AlltoAllvGmmTilingData>();
    OP_TILING_CHECK(
        GetContextAttr(context_) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Failed to get context attr."),
        return ge::GRAPH_FAILED);

    if (context_->GetOptionalInputShape(NON_QUANT_MM_X_INDEX) != nullptr) {
        OP_TILING_CHECK(context_->GetOutputShape(OUTPUT_MM_Y_INDEX) == nullptr,
            OP_LOGE(A_INNER_DEBUG, "MmY output shape is null when mmX is present, expected non-null."),
            return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK(
        CheckShapeDims(context_) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Shape dim check failed."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckDType(context_) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Dtype check failed."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckShapeRelation(context_) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Shape relation check failed."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        GetShapeAndFormat(context_) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Failed to get shape and format."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckShapeSize(context_) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Shape size check failed."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckAttrsShapeSize(context_) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Attrs shape size check failed."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckAttrsShapeRelation(context_) != ge::GRAPH_SUCCESS,
        OP_LOGE(A_INNER_DEBUG, "Attrs shape relation check failed."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckSendRecvDataVolumn(context_) != ge::GRAPH_SUCCESS,
        OP_LOGE(A_INNER_DEBUG, "Send/recv data volume check failed."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckMKN(context_) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "MKN check failed."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckMmShapeDims(context_) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Mm shape dims check failed."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

uint64_t AlltoAllvGmmTiling::GetTilingKey() const
{
    uint8_t commMode = Mc2Comm::COMM_MODE_AICPU;
    auto platformInfo = context_->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        commMode = Mc2Comm::GetCommModeFromEnv();
    }
    bool tilingekyGmmTrans = transGmmWeight_;
    bool tilingekyMmTrans = transMmWeight_;
    uint64_t tilingKey = GET_TPL_TILING_KEY(tilingekyGmmTrans, tilingekyMmTrans, commMode);
    OP_LOGD(A_INNER_DEBUG, "TilingKey is %llu, gmmTrans=%d, mmTrans=%d, commMode=%u.",
            tilingKey, tilingekyGmmTrans, tilingekyMmTrans, commMode);
    return tilingKey;
}

ge::graphStatus AlltoAllvGmmTiling::RunFusionKernelTiling(gert::TilingContext* context)
{
    OP_TILING_CHECK(
        SetHcclTiling(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "HCCL tiling config failed."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        setNumBlocks(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Set numBlocks failed."),
        return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(A_INNER_DEBUG, "Attrs is null, expected non-null."),
        return ge::GRAPH_FAILED);

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    OP_TILING_CHECK(epWorldSizePtr == nullptr,
        OP_LOGE(A_INNER_DEBUG, "EpWorldSize attr is null, expected non-null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*epWorldSizePtr == 0,
        OP_LOGE(A_INNER_DEBUG, "EpWorldSize is 0, expected positive value."), return ge::GRAPH_FAILED);
    epWorldSize_ = *epWorldSizePtr;

    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK((sendCountsPtr == nullptr) || (recvCountsPtr == nullptr),
        OP_LOGE(A_INNER_DEBUG, "SendCounts or recvCounts attr is null, expected non-null."), return ge::GRAPH_FAILED);

    sendCounts = static_cast<const int64_t*>(sendCountsPtr->GetData());
    OP_TILING_CHECK(sendCounts == nullptr,
        OP_LOGE(A_INNER_DEBUG, "SendCounts data is null, expected non-null."), return ge::GRAPH_FAILED);
    recvCounts = static_cast<const int64_t*>(recvCountsPtr->GetData());
    OP_TILING_CHECK(recvCounts == nullptr,
        OP_LOGE(A_INNER_DEBUG, "RecvCounts data is null, expected non-null."), return ge::GRAPH_FAILED);
    uint64_t sendCountsSize = sendCountsPtr->GetSize();
    e_ = sendCountsSize / epWorldSize_;

    auto gmmXShape = context->GetInputShape(GMM_X_INDEX)->GetStorageShape();
    bsk_ = gmmXShape.GetDim(0);
    h1_ = gmmXShape.GetDim(1);

    auto transGmmWeightPtr = attrs->GetAttrPointer<bool>(NON_QUANT_ATTR_TRANS_GMM_WEIGHT_INDEX);
    auto transMmWeightPtr = attrs->GetAttrPointer<bool>(NON_QUANT_ATTR_TRANS_MM_WEIGHT_INDEX);
    OP_TILING_CHECK((transGmmWeightPtr == nullptr) || (transMmWeightPtr == nullptr),
        OP_LOGE(A_INNER_DEBUG, "TransGmmWeight or transMmWeight attr is null, expected non-null."),
        return ge::GRAPH_FAILED);
    transGmmWeight_ = *transGmmWeightPtr;
    transMmWeight_ = (context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX) != nullptr) ? *transMmWeightPtr : false;

    auto gmmWeightShape = context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape();
    e_ = gmmWeightShape.GetDim(0);
    uint64_t n1DimIndex = transGmmWeight_ ? DIM_ONE : DIM_TWO;
    n1_ = gmmWeightShape.GetDim(n1DimIndex);

    auto gmmYShape = context->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape();
    a_ = gmmYShape.GetDim(0);

    if (context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX) != nullptr) {
        auto mmXShape = context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX)->GetStorageShape();
        bs_ = mmXShape.GetDim(0);
        h2_ = mmXShape.GetDim(1);
        hasSharedExpertFlag_ = true;
        auto mmWeightShape = context->GetInputShape(NON_QUANT_MM_WEIGHT_INDEX)->GetStorageShape();
        uint64_t n2DimIndex = transMmWeight_ ? DIM_ZERO : DIM_ONE;
        n2_ = mmWeightShape.GetDim(n2DimIndex);
    }

    OP_TILING_CHECK(
        DoAiCoreTiling(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "AI core tiling failed."),
        return ge::GRAPH_FAILED);

    uint64_t tilingKey = GetTilingKey();
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::DoAiCoreTiling(const gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(recvCountsPtr == nullptr,
        OP_LOGE(A_INNER_DEBUG, "RecvCounts attr is null, expected non-null."), return ge::GRAPH_FAILED);
    recvCounts = static_cast<const int64_t*>(recvCountsPtr->GetData());
    OP_TILING_CHECK(recvCounts == nullptr,
        OP_LOGE(A_INNER_DEBUG, "RecvCounts data is null, expected non-null."), return ge::GRAPH_FAILED);
    uint64_t recvCountsSize = recvCountsPtr->GetSize();
    uint64_t maxMSize = 0;
    OP_TILING_CHECK(e_ * epWorldSize_ > recvCountsSize,
        OP_LOGE(A_INNER_DEBUG, "ExpertNum[%llu] * epWorldSize[%llu] exceeds recvCounts size[%llu].",
            e_, epWorldSize_, recvCountsSize),
        return ge::GRAPH_FAILED);
    for (uint64_t expertIdx = 0; expertIdx < e_; expertIdx++) {
        uint64_t mSize = 0;
        for (uint64_t rankIdx = 0; rankIdx < epWorldSize_; rankIdx++) {
            mSize += recvCounts[rankIdx * e_ + expertIdx];
        }
        maxMSize = std::max(mSize, maxMSize);
    }

    if (maxMSize != 0) {
        AlltoAllvGmmTilingHelper gmmHelper(*this);
        GE_ASSERT_GRAPH_SUCCESS(gmmHelper.SetInputParams(maxMSize, n1_, h1_, transGmmWeight_,
            gmmXDataType_, gmmWeightDataType_, gmmXDataType_));
        GE_ASSERT_GRAPH_SUCCESS(gmmHelper.Process());
        tilingData->gmmQuantTilingData = gmmHelper.GetAlltoAllvQuantHelperData();
    }

    if (bs_ != 0) {
        AlltoAllvGmmTilingHelper mmHelper(*this);
        GE_ASSERT_GRAPH_SUCCESS(mmHelper.SetInputParams(bs_, n2_, h2_, transMmWeight_,
            mmXDataType_, mmWeightDataType_, mmXDataType_));
        GE_ASSERT_GRAPH_SUCCESS(mmHelper.Process());
        tilingData->mmQuantTilingData = mmHelper.GetAlltoAllvQuantHelperData();
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingHelper::SetInputParams(uint64_t M, uint64_t N, uint64_t K, bool transB,
    ge::DataType aDtype, ge::DataType bDtype, ge::DataType cDtype)
{
    GetPlatformInfo();
    inputParams_.opName = context_->GetNodeName();
    inputParams_.kernelType = 0UL;
    inputParams_.splitItem = 0;
    inputParams_.actType = GMM_ACT_TYPE_NONE;
    inputParams_.aFormat = ge::FORMAT_ND;
    inputParams_.bFormat = ge::FORMAT_ND;
    inputParams_.cFormat = ge::FORMAT_ND;
    inputParams_.transA = false;
    inputParams_.transB = transB;
    inputParams_.hasBias = false;
    inputParams_.isSingleX = false;
    inputParams_.isSingleW = false;
    inputParams_.isSingleY = false;

    inputParams_.mSize = M;
    inputParams_.kSize = K;
    inputParams_.nSize = N;
    inputParams_.groupNum = SINGLE_GROUP_NUM;
    inputParams_.aQuantMode = Mc2GroupedMatmulTiling::QuantMode::DEFAULT;
    inputParams_.bQuantMode = Mc2GroupedMatmulTiling::QuantMode::DEFAULT;
    inputParams_.groupType = Mc2GroupedMatmul::SPLIT_M;
    inputParams_.groupListType = 1;

    inputParams_.aDtype = aDtype;
    inputParams_.bDtype = bDtype;
    inputParams_.cDtype = cDtype;
    inputParams_.biasDtype = ge::DT_INT32;
    inputParams_.scaleDtype = ge::DT_FLOAT;
    inputParams_.perTokenScaleDtype = ge::DT_FLOAT;

    OP_TILING_CHECK(M > static_cast<uint64_t>(INT32_MAX),
        OP_LOGE(A_INNER_DEBUG, "M value is %llu, expected less than 2147483647.", M),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(K > static_cast<uint64_t>(INT32_MAX),
        OP_LOGE(A_INNER_DEBUG, "K value is %llu, expected less than 2147483647.", K),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(N > static_cast<uint64_t>(INT32_MAX),
        OP_LOGE(A_INNER_DEBUG, "N value is %llu, expected less than 2147483647.", N),
        return ge::GRAPH_FAILED);
    mList_[0] = static_cast<int32_t>(M);
    kList_[0] = static_cast<int32_t>(K);
    nList_[0] = static_cast<int32_t>(N);
    SetKernelType();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingHelper::Process()
{
    GE_ASSERT_GRAPH_SUCCESS(DoOpTiling());
    GE_ASSERT_GRAPH_SUCCESS(DoLibApiTiling());
    return ge::GRAPH_SUCCESS;
}

uint64_t AlltoAllvGmmTilingStruct::GetTilingKey() const
{
    const uint64_t tilingKey = context_->GetTilingKey();
    OP_LOGD(A_INNER_DEBUG, "TilingKey is %llu.", tilingKey);
    return tilingKey;
}

bool AlltoAllvGmmTilingStruct::IsCapable()
{
    return true;
}

ge::graphStatus AlltoAllvGmmTilingBase::GetPlatformInfo()
{
    if (GetCommonPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckCommonPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::GetShapeAttrsInfo()
{
    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(A_INNER_DEBUG, "Attrs is null, expected non-null."),
        return ge::GRAPH_FAILED);
    groupPtr_ = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    OP_TILING_CHECK(groupPtr_ == nullptr, OP_LOGE(A_INNER_DEBUG, "Group attr is null, expected non-null."),
        return ge::GRAPH_FAILED);
    group_ = groupPtr_;
    epWorldSizePtr_ = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    OP_TILING_CHECK(epWorldSizePtr_ == nullptr,
        OP_LOGE(A_INNER_DEBUG, "EpWorldSize attr is null, expected non-null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*epWorldSizePtr_ == 0,
        OP_LOGE(A_INNER_DEBUG, "EpWorldSize is 0, expected positive value."), return ge::GRAPH_FAILED);
    epWorldSize_ = *epWorldSizePtr_;
    sendCountsPtr_ = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(sendCountsPtr_ == nullptr,
        OP_LOGE(A_INNER_DEBUG, "SendCounts attr is null, expected non-null."), return ge::GRAPH_FAILED);
    sendCounts = static_cast<const int64_t*>(sendCountsPtr_->GetData());
    OP_TILING_CHECK(sendCounts == nullptr,
        OP_LOGE(A_INNER_DEBUG, "SendCounts data is null, expected non-null."), return ge::GRAPH_FAILED);
    recvCountsPtr_ = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(recvCountsPtr_ == nullptr,
        OP_LOGE(A_INNER_DEBUG, "RecvCounts attr is null, expected non-null."), return ge::GRAPH_FAILED);
    recvCounts = static_cast<const int64_t*>(recvCountsPtr_->GetData());
    OP_TILING_CHECK(recvCounts == nullptr,
        OP_LOGE(A_INNER_DEBUG, "RecvCounts data is null, expected non-null."), return ge::GRAPH_FAILED);
    transGmmWeightPtr_ = attrs->GetAttrPointer<bool>(NON_QUANT_ATTR_TRANS_GMM_WEIGHT_INDEX);
    if (transGmmWeightPtr_ != nullptr) {
        transGmmWeight_ = *transGmmWeightPtr_;
    }
    transMmWeightPtr_ = attrs->GetAttrPointer<bool>(NON_QUANT_ATTR_TRANS_MM_WEIGHT_INDEX);
    if (transMmWeightPtr_ != nullptr) {
        transMmWeight_ = *transMmWeightPtr_;
    }
    if (context_->GetOptionalInputShape(NON_QUANT_MM_X_INDEX) == nullptr) {
        transMmWeight_ = false;
    }
    permuteOutFlagPtr_ = attrs->GetAttrPointer<bool>(NON_QUANT_ATTR_PERMUTE_OUT_FLAG_INDEX);
    if (permuteOutFlagPtr_ != nullptr) {
        permuteOutFlag_ = *permuteOutFlagPtr_;
    }

    if (context_->GetOptionalInputShape(NON_QUANT_MM_X_INDEX) != nullptr) {
        auto mmXShape = context_->GetOptionalInputShape(NON_QUANT_MM_X_INDEX)->GetStorageShape();
        bs_ = mmXShape.GetDim(0);
        h2_ = mmXShape.GetDim(1);
        hasSharedExpertFlag_ = true;
        mmXDataType_ = context_->GetInputDesc(NON_QUANT_MM_X_INDEX)->GetDataType();
        mmWeightDataType_ = context_->GetInputDesc(NON_QUANT_MM_WEIGHT_INDEX)->GetDataType();
        auto mmWeightShape = context_->GetInputShape(NON_QUANT_MM_WEIGHT_INDEX)->GetStorageShape();
        uint64_t n2DimIndex = transMmWeight_ ? DIM_ZERO : DIM_ONE;
        n2_ = mmWeightShape.GetDim(n2DimIndex);
    } else {
        transMmWeight_ = false;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::GetShapeInfo()
{
    OP_TILING_CHECK(context_->GetInputShape(GMM_X_INDEX) == nullptr,
        OP_LOGE(A_INNER_DEBUG, "GmmX input shape is null, expected non-null."), return ge::GRAPH_FAILED);
    bsk_ = context_->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    h1_ = context_->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    gmmXDataType_ = context_->GetInputDesc(GMM_X_INDEX)->GetDataType();

    OP_TILING_CHECK(context_->GetInputShape(GMM_WEIGHT_INDEX) == nullptr,
        OP_LOGE(A_INNER_DEBUG, "GmmWeight input shape is null, expected non-null."), return ge::GRAPH_FAILED);
    e_ = context_->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t n1DimIndex = transGmmWeight_ ? DIM_ONE : DIM_TWO;
    n1_ = context_->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(n1DimIndex);
    gmmWeightDataType_ = context_->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType();

    OP_TILING_CHECK(context_->GetOutputShape(OUTPUT_GMM_Y_INDEX) == nullptr,
        OP_LOGE(A_INNER_DEBUG, "GmmY output shape is null, expected non-null."), return ge::GRAPH_FAILED);
    a_ = context_->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape().GetDim(DIM_ZERO);

    if (context_->GetOptionalInputShape(NON_QUANT_MM_X_INDEX) != nullptr) {
        hasSharedExpertFlag_ = true;
        bs_ = context_->GetOptionalInputShape(NON_QUANT_MM_X_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
        h2_ = context_->GetOptionalInputShape(NON_QUANT_MM_X_INDEX)->GetStorageShape().GetDim(DIM_ONE);
        mmXDataType_ = context_->GetInputDesc(NON_QUANT_MM_X_INDEX)->GetDataType();
    }

    if (context_->GetOptionalInputShape(NON_QUANT_MM_WEIGHT_INDEX) != nullptr) {
        auto mmWeightShape = context_->GetInputShape(NON_QUANT_MM_WEIGHT_INDEX)->GetStorageShape();
        uint64_t n2DimIndex = transMmWeight_ ? DIM_ZERO : DIM_ONE;
        n2_ = mmWeightShape.GetDim(n2DimIndex);
        mmWeightDataType_ = context_->GetInputDesc(NON_QUANT_MM_WEIGHT_INDEX)->GetDataType();
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::DoLibApiTiling()
{
    return GetShapeInfo();
}

ge::graphStatus AlltoAllvGmmTilingBase::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr,
        OP_LOGE(context_->GetNodeName(), "Workspace is null, expected non-null."), return ge::GRAPH_FAILED);
    const uint64_t tensorListSize = 512;
    uint64_t groupListSize = sizeof(int64_t) * e_;
    uint64_t xDataTypeSize = GetSizeByDataType(gmmXDataType_);
    uint64_t permuteOutSize = permuteOutFlag_ ? 0 : Ops::Base::CeilAlign(a_ * h1_ * xDataTypeSize, tensorListSize);
    uint64_t permuteScaleOutSize = 0;
    workspaces[0] = libApiWorkSpaceSize_ + permuteOutSize + permuteScaleOutSize + groupListSize + tensorListSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::PostTiling()
{
    auto tilingData = context_->GetTilingData<AlltoAllvGmmTilingData>();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, tilingData);

    tilingData->taskTilingInfo.BSK = bsk_;
    tilingData->taskTilingInfo.BS = bs_;
    tilingData->taskTilingInfo.H1 = h1_;
    tilingData->taskTilingInfo.H2 = h2_;
    tilingData->taskTilingInfo.A = a_;
    tilingData->taskTilingInfo.N1 = n1_;
    tilingData->taskTilingInfo.N2 = n2_;
    tilingData->taskTilingInfo.epWorldSize = epWorldSize_;
    tilingData->taskTilingInfo.e = e_;
    tilingData->taskTilingInfo.ubSize = ubSize_;
    tilingData->taskTilingInfo.mainLoopExpertNum = e_;
    tilingData->taskTilingInfo.tailLoopExpertNum = 0;
    tilingData->taskTilingInfo.totalLoopCount = e_;

    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(A_INNER_DEBUG, "Attrs is null, expected non-null."),
        return ge::GRAPH_FAILED);
    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(sendCountsPtr == nullptr,
        OP_LOGE(A_INNER_DEBUG, "SendCounts attr is null, expected non-null."), return ge::GRAPH_FAILED);
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(recvCountsPtr == nullptr,
        OP_LOGE(A_INNER_DEBUG, "RecvCounts attr is null, expected non-null."), return ge::GRAPH_FAILED);
    const int64_t* sendCountsData = static_cast<const int64_t*>(sendCountsPtr->GetData());
    OP_TILING_CHECK(sendCountsData == nullptr,
        OP_LOGE(A_INNER_DEBUG, "SendCounts data is null, expected non-null."), return ge::GRAPH_FAILED);
    const int64_t* recvCountsData = static_cast<const int64_t*>(recvCountsPtr->GetData());
    OP_TILING_CHECK(recvCountsData == nullptr,
        OP_LOGE(A_INNER_DEBUG, "RecvCounts data is null, expected non-null."), return ge::GRAPH_FAILED);

    for (uint32_t i = 0; i < e_ * epWorldSize_; i++) {
        tilingData->taskTilingInfo.sendCnt[i] = sendCountsData[i];
        tilingData->taskTilingInfo.recvCnt[i] = recvCountsData[i];
    }

    tilingData->isPermuteOut = permuteOutFlag_;
    tilingData->isNeedMM = hasSharedExpertFlag_;
    tilingData->isFp16 = (gmmXDataType_ == ge::DT_FLOAT16);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AlltoAllvGmmTilingFunc(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

struct AlltoAllvGmmCompileInfo {
};
static ge::graphStatus TilingParseForAlltoAllvGmm(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<AlltoAllvGmmCompileInfo>();
    OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AlltoAllvGroupedMatMul)
    .Tiling(AlltoAllvGmmTilingFunc)
    .TilingParse<AlltoAllvGmmCompileInfo>(TilingParseForAlltoAllvGmm);
} // namespace optiling