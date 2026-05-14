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
 * \file allto_allv_grouped_mat_mul_tiling.cc
 * \brief
 */

#include "allto_allv_grouped_mat_mul_tiling.h"
#include "../../allto_allv_quant_grouped_mat_mul/op_host/op_tiling/allto_allv_quant_grouped_mat_mul_tiling_base.h"
#include "../../op_kernel/allto_allv_grouped_mat_mul_tiling_key.h"
#include "mc2_comm_utils.h"

using namespace ge;
using namespace AscendC;
using namespace Ops::Transformer::OpTiling;
namespace {
    static const char* A_INNER_DEBUG = "AlltoAllvGroupedMatMul Tiling";
}
namespace optiling {
// Constants unique to non-quant version (not in allto_allv_quant_grouped_mat_mul_tiling_base.h)
constexpr uint32_t OUTPUT_Y_INDEX = 0U;
constexpr uint32_t NUM_ZERO = 0;
constexpr uint32_t NUM_ONE = 1;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t NUM_THREE = 3;
constexpr uint32_t NUM_EIGHT = 8;
constexpr uint32_t NUM_SIXTEEN = 16;
constexpr uint32_t E_MAX_VALUE_NON_QUANT = 48;
constexpr uint32_t NUM_SIXTYFOUR = 64;
constexpr uint32_t MAX_EXPERT_NUM = 384;
constexpr uint32_t MAX_BSK = 52428800;
constexpr uint32_t MAX_SHAPE_SIZE = 65536;
constexpr uint32_t MAX_SHARED_H_SHAPE_SIZE = 12288;

constexpr int64_t BEST_L1_PARTA = 256 * 1024;
constexpr int64_t BEST_L1_PARTB = 128 * 1024;
constexpr int64_t BEST_BASE_N = 256;
constexpr uint32_t UB_DIVIDE_NUM = 2;
constexpr uint32_t UB_CALSIZE_PER_BLOCK = 16 * 1024;
constexpr uint64_t DOUBLE_BUFFER_L0A_L0B = 2;
constexpr uint64_t DOUBLE_BUFFER_STEPKA_STEPKB = 2;
constexpr uint32_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t MAX_TURN_NUM = 24;
constexpr int32_t MAX_BASE_K = 128;
constexpr uint64_t COMM_TILE = 8;

static inline uint32_t SixteenAlign(uint32_t a, bool up = false)
{
    if (up) {
        a += 15;
    }
    return a & ~15;
}

static inline uint32_t Ceil(uint32_t a, uint32_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

static uint64_t GMMGetSizePlatForm(
    const platform_ascendc::CoreMemType memType, platform_ascendc::PlatformAscendC ascendcPlatform)
{
    uint64_t size = 0;
    ascendcPlatform.GetCoreMemSize(memType, size);
    return size;
}

struct PlatFormMemSize {
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0CSize;
    uint64_t l0ASize;
    uint64_t l0BSize;

    explicit PlatFormMemSize(platform_ascendc::PlatformAscendC ascendcPlatform)
        : ubSize(GMMGetSizePlatForm(platform_ascendc::CoreMemType::UB, ascendcPlatform)),
          l1Size(GMMGetSizePlatForm(platform_ascendc::CoreMemType::L1, ascendcPlatform)),
          l0CSize(GMMGetSizePlatForm(platform_ascendc::CoreMemType::L0_C, ascendcPlatform)),
          l0ASize(GMMGetSizePlatForm(platform_ascendc::CoreMemType::L0_A, ascendcPlatform)),
          l0BSize(GMMGetSizePlatForm(platform_ascendc::CoreMemType::L0_B, ascendcPlatform))
    {}
};

ge::graphStatus AlltoAllvGmmTiling::GetContextAttr(const gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(A_INNER_DEBUG, "GetAttrs returned null."), return ge::GRAPH_FAILED);
    
    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    OP_TILING_CHECK((groupEpPtr == nullptr) || (epWorldSizePtr == nullptr),
        OP_LOGE(A_INNER_DEBUG, "groupEpPtr or epWorldSizePtr is null."), return ge::GRAPH_FAILED);
    
    auto mmXStorageShape = context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX);
    tilingData->isPermuteOut = (mmXStorageShape != nullptr);
    
    groupPtr_ = groupEpPtr;
    epWorldSizePtr_ = epWorldSizePtr;
    
    auto transGmmWeightPtr = attrs->GetAttrPointer<bool>(NON_QUANT_ATTR_TRANS_GMM_WEIGHT_INDEX);
    OP_TILING_CHECK(transGmmWeightPtr == nullptr,
        OP_LOGE(A_INNER_DEBUG, "transGmmWeightPtr is null."), return ge::GRAPH_FAILED);
    transGmmWeight_ = *transGmmWeightPtr;
    
    OP_LOGI(A_INNER_DEBUG, "epGroup is %s, epWorldSize is %lu.", groupPtr_, *epWorldSizePtr_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::GetShapeAndFormat(const gert::TilingContext* context)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckMKN(const gert::TilingContext* context)
{
    (void)context;
    OP_TILING_CHECK(
        mmDataTypeSize == 0,
        OP_LOGE(
            A_INNER_DEBUG, "GMM get matmul dtype[%s] size is 0.",
            TypeUtils::DataTypeToAscendString(mmXDataType_).GetString()),
        return ge::GRAPH_FAILED);
    uint32_t numInOneBlk = ONE_BLK_SIZE / mmDataTypeSize;
    OP_TILING_CHECK(numInOneBlk == 0, OP_LOGE(A_INNER_DEBUG, "GMM numInOneBlk cannot be 0."), return ge::GRAPH_FAILED);
    int64_t maxMKN = INT_MAX / numInOneBlk * numInOneBlk;
    OP_TILING_CHECK(
        maxM_ > maxMKN || maxN_ > maxMKN || maxK_ > maxMKN,
        OP_LOGE(A_INNER_DEBUG, "32B-aligned m, n or k axis for gmm is out of range int32!"), return ge::GRAPH_FAILED);
    if (context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX) != nullptr) {
        OP_TILING_CHECK(
            maxMForMM_ > maxMKN || maxNForMM_ > maxMKN || maxKForMM_ > maxMKN,
            OP_LOGE(A_INNER_DEBUG, "32B-aligned m, n or k axis for mm is out of range int32!"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckSendRecvDataVolumn(const gert::TilingContext* context) const
{
    uint64_t eExpert = tilingData->taskTilingInfo.e;
    uint64_t epWorldSize = tilingData->taskTilingInfo.epWorldSize;
    uint64_t recvSendMin = static_cast<uint64_t>(2U * 1024U * 1024U);

    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(A_INNER_DEBUG, "GetAttrs returned null."), return ge::GRAPH_FAILED);

    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK((sendCountsPtr == nullptr) || (recvCountsPtr == nullptr),
        OP_LOGE(A_INNER_DEBUG, "sendCountsPtr or recvCountsPtr is null."), return ge::GRAPH_FAILED);

    const uint64_t* sendCounts = static_cast<const uint64_t*>(sendCountsPtr->GetData());
    const uint64_t* recvCounts = static_cast<const uint64_t*>(recvCountsPtr->GetData());
    uint64_t recvSum = 0U;
    uint64_t sendSum = 0U;
    uint64_t H1 = tilingData->taskTilingInfo.H1;
    uint64_t bsk = context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(0);
    uint64_t a = context->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape().GetDim(0);
    auto platformInfo = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    if (NeedToCheckCounts()) {
        for (uint64_t i = 1U; i <= epWorldSize; i++) {
            recvSum = 0U;
            sendSum = 0U;
            for (uint64_t j = (i - 1U) * eExpert; j <= i * eExpert - 1U; j++) {
                OP_TILING_CHECK((sendCounts[j] < NUM_ZERO) || (sendCounts[j] > bsk),
                    OP_LOGE(A_INNER_DEBUG, "sendCounts[%lu] should be in [0, bsK[%lu]], but get %lu",j, bsk, sendCounts[j]),
                    return ge::GRAPH_FAILED);
                OP_TILING_CHECK((recvCounts[j] < NUM_ZERO) || (recvCounts[j] > a),
                    OP_LOGE(A_INNER_DEBUG, "recvCounts[%lu] should be in [0, a[%lu]], but get %lu",j, a, recvCounts[j]),
                    return ge::GRAPH_FAILED);
                recvSum += recvCounts[j] * H1 * 2U;
                sendSum += sendCounts[j] * H1 * 2U;
            }
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckShapeSize(const gert::TilingContext* context) const
{
    OP_TILING_CHECK(
        (context->GetInputShape(GMM_X_INDEX) == nullptr) || (context->GetInputShape(GMM_WEIGHT_INDEX) == nullptr),
        OP_LOGE(A_INNER_DEBUG, "GetInputShape gmmX or gmmWeight returned null."), return ge::GRAPH_FAILED);

    uint64_t BSK = context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(0);
    if (BSK <= NUM_ZERO || BSK >= MAX_BSK) {
        OP_LOGE(A_INNER_DEBUG, "BSK is out of range [1, %lu).", MAX_BSK);
        return ge::GRAPH_FAILED;
    }

    uint64_t H1 = context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(1);
    if (H1 <= NUM_ZERO || H1 >= MAX_SHAPE_SIZE) {
        OP_LOGE(A_INNER_DEBUG, "H1 is out of range [1, %lu).", MAX_SHAPE_SIZE);
        return ge::GRAPH_FAILED;
    }

    uint64_t N1 = context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(1);
    if (N1 <= NUM_ZERO || N1 >= MAX_SHAPE_SIZE) {
        OP_LOGE(A_INNER_DEBUG, "N1 is out of range [1, %lu).", MAX_SHAPE_SIZE);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckAttrsShapeSize(const gert::TilingContext* context) const
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(A_INNER_DEBUG, "GetAttrs returned null."), return ge::GRAPH_FAILED);

    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(sendCountsPtr == nullptr, OP_LOGE(A_INNER_DEBUG, "GetAttrPointer sendCounts returned null."),
        return ge::GRAPH_FAILED);

    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(recvCountsPtr == nullptr, OP_LOGE(A_INNER_DEBUG, "GetAttrPointer recvCounts returned null."),
        return ge::GRAPH_FAILED);

    const uint64_t* sendCounts = static_cast<const uint64_t*>(sendCountsPtr->GetData());
    const uint64_t* recvCounts = static_cast<const uint64_t*>(recvCountsPtr->GetData());
    uint64_t sendCountsSize = sendCountsPtr->GetSize();
    uint64_t recvCountsSize = recvCountsPtr->GetSize();

    if (sendCountsSize != recvCountsSize) {
        OP_LOGE(A_INNER_DEBUG, "sendCountsSize is not equal to recvCountsSize.");
        return ge::GRAPH_FAILED;
    }

    if (sendCountsSize >= MC2KernelTemplate::MAX_EXPERT_SIZE) {
        OP_LOGE(A_INNER_DEBUG, "sendCountsSize is too large, max is %lu.", MC2KernelTemplate::MAX_EXPERT_SIZE);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckAttrsShapeRelation(const gert::TilingContext* context) const
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(A_INNER_DEBUG, "GetAttrs returned null."), return ge::GRAPH_FAILED);

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(A_INNER_DEBUG, "GetAttrPointer epWorldSize returned null."),
        return ge::GRAPH_FAILED);

    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(sendCountsPtr == nullptr, OP_LOGE(A_INNER_DEBUG, "GetAttrPointer sendCounts returned null."),
        return ge::GRAPH_FAILED);

    const uint64_t* sendCounts = static_cast<const uint64_t*>(sendCountsPtr->GetData());
    uint64_t sendCountsSize = sendCountsPtr->GetSize();
    uint64_t eExpert = sendCountsSize / (*epWorldSizePtr);

    if (eExpert <= NUM_ZERO || eExpert >= E_MAX_VALUE_NON_QUANT) {
        OP_LOGE(A_INNER_DEBUG, "eExpert is out of range [1, %lu).", E_MAX_VALUE_NON_QUANT);
        return ge::GRAPH_FAILED;
    }

    for (uint64_t i = 0; i < sendCountsSize; i++) {
        if (sendCounts[i] < NUM_ZERO) {
            OP_LOGE(A_INNER_DEBUG, "sendCounts[%lu] is negative.", i);
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckShapeRelation(const gert::TilingContext* context) const
{
    auto gmmXShape = context->GetInputShape(GMM_X_INDEX)->GetStorageShape();
    auto gmmWeightShape = context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape();
    uint64_t BSK = gmmXShape.GetDim(0);
    uint64_t H1 = gmmXShape.GetDim(1);
    uint64_t e = gmmWeightShape.GetDim(0);
    uint64_t h1DimIndex = transGmmWeight_ ? DIM_TWO : DIM_ONE;
    uint64_t n1DimIndex = transGmmWeight_ ? DIM_ONE : DIM_TWO;
    uint64_t H2 = gmmWeightShape.GetDim(h1DimIndex);
    uint64_t N1 = gmmWeightShape.GetDim(n1DimIndex);

    if (H1 != H2) {
        OP_LOGE(A_INNER_DEBUG, "H1 is not equal to H2.");
        return ge::GRAPH_FAILED;
    }

    if (tilingData->isPermuteOut) {
        auto mmXShape = context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX)->GetStorageShape();
        auto mmWeightShape = context->GetInputShape(NON_QUANT_MM_WEIGHT_INDEX)->GetStorageShape();
        uint64_t BS = mmXShape.GetDim(0);
        uint64_t H = mmXShape.GetDim(1);
        uint64_t H2MM = mmWeightShape.GetDim(0);
        uint64_t N2 = mmWeightShape.GetDim(1);

        if (H != H2MM) {
            OP_LOGE(A_INNER_DEBUG, "mmX H is not equal to mmWeight H2.");
            return ge::GRAPH_FAILED;
        }

        if (N1 != H) {
            OP_LOGE(A_INNER_DEBUG, "N1 is not equal to mmX H.");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckShapeDims(const gert::TilingContext* context)
{
    auto gmmXShape = context->GetInputShape(GMM_X_INDEX)->GetStorageShape();
    OP_TILING_CHECK(gmmXShape.GetDimNum() != DIM_TWO,
        OP_LOGE(A_INNER_DEBUG, "gmmX shape dim num should be 2."), return ge::GRAPH_FAILED);
    maxM_ = gmmXShape.GetDim(0);
    maxK_ = gmmXShape.GetDim(1);

    auto gmmWeightShape = context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape();
    OP_TILING_CHECK(gmmWeightShape.GetDimNum() != DIM_THREE,
        OP_LOGE(A_INNER_DEBUG, "gmmWeight shape dim num should be 3."), return ge::GRAPH_FAILED);
    e_ = gmmWeightShape.GetDim(0);
    uint64_t h1DimIndex = transGmmWeight_ ? DIM_TWO : DIM_ONE;
    uint64_t n1DimIndex = transGmmWeight_ ? DIM_ONE : DIM_TWO;
    maxK_ = gmmWeightShape.GetDim(h1DimIndex);
    maxN_ = gmmWeightShape.GetDim(n1DimIndex);

    mmXDataType_ = context->GetInputDesc(GMM_X_INDEX)->GetDataType();
    gmmXDataType_ = context->GetInputDesc(GMM_X_INDEX)->GetDataType();
    mmDataTypeSize = GetSizeByDataType(mmXDataType_);

    if (context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX) != nullptr) {
        auto mmXShape = context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX)->GetStorageShape();
        OP_TILING_CHECK(mmXShape.GetDimNum() != DIM_TWO,
            OP_LOGE(A_INNER_DEBUG, "mmX shape dim num should be 2."), return ge::GRAPH_FAILED);
        maxMForMM_ = mmXShape.GetDim(0);
        maxKForMM_ = mmXShape.GetDim(1);

        auto mmWeightShape = context->GetInputShape(NON_QUANT_MM_WEIGHT_INDEX)->GetStorageShape();
        OP_TILING_CHECK(mmWeightShape.GetDimNum() != DIM_TWO,
            OP_LOGE(A_INNER_DEBUG, "mmWeight shape dim num should be 2."), return ge::GRAPH_FAILED);
        maxKForMM_ = mmWeightShape.GetDim(0);
        maxNForMM_ = mmWeightShape.GetDim(1);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckDType(const gert::TilingContext* context) const
{
    ge::DataType gmmXDtype = context->GetInputDesc(GMM_X_INDEX)->GetDataType();
    ge::DataType gmmWeightDtype = context->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType();

    if (gmmXDtype != gmmWeightDtype) {
        OP_LOGE(A_INNER_DEBUG, "gmmXDtype is not equal to gmmWeightDtype.");
        return ge::GRAPH_FAILED;
    }

    if (gmmXDtype != ge::DT_FLOAT16 && gmmXDtype != ge::DT_BF16) {
        OP_LOGE(A_INNER_DEBUG, "gmmXDtype should be FP16 or BF16.");
        return ge::GRAPH_FAILED;
    }

    if (context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX) != nullptr) {
        ge::DataType mmXDtype = context->GetInputDesc(NON_QUANT_MM_X_INDEX)->GetDataType();
        ge::DataType mmWeightDtype = context->GetInputDesc(NON_QUANT_MM_WEIGHT_INDEX)->GetDataType();

        if (mmXDtype != mmWeightDtype) {
            OP_LOGE(A_INNER_DEBUG, "mmXDtype is not equal to mmWeightDtype.");
            return ge::GRAPH_FAILED;
        }

        if (mmXDtype != ge::DT_FLOAT16 && mmXDtype != ge::DT_BF16) {
            OP_LOGE(A_INNER_DEBUG, "mmXDtype should be FP16 or BF16.");
            return ge::GRAPH_FAILED;
        }

        if (gmmXDtype != mmXDtype) {
            OP_LOGE(A_INNER_DEBUG, "gmmXDtype is not equal to mmXDtype.");
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
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(A_INNER_DEBUG, "GetAttrs returned null."), return ge::GRAPH_FAILED);

    auto tilingData = context_->GetTilingData<AlltoAllvGmmTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(A_INNER_DEBUG, "GetTilingData returned null."), return ge::GRAPH_FAILED);

    uint32_t alltoAllvCmd = 8U;
    std::string alltoAllvConfig = "AlltoAll=level0:fullmesh;level1:pairwise";
    const uint32_t alltoAllvReduceType = 0u;

    mc2tiling::HcclDataType alltoallvHcclDataType = mc2tiling::ConvertGeTypeToHcclType(context_->GetNodeName(), gmmXDataType_);
    OP_TILING_CHECK(alltoallvHcclDataType == mc2tiling::HcclDataType::HCCL_DATA_TYPE_RESERVED, 
        OP_LOGE(context_->GetNodeName(), "Ge datatype gmmXDataType(%s) is not found in HCCL_DATA_TYPE.", 
        ge::TypeUtils::DataTypeToSerialString(gmmXDataType_).c_str()), 
        return ge::GRAPH_FAILED);
    uint8_t alltoAllvDataType = static_cast<uint8_t>(alltoallvHcclDataType);

    Mc2CcTilingConfig hcclCcTilingConfig(group_, alltoAllvCmd, alltoAllvConfig, alltoAllvReduceType, alltoAllvDataType,
        alltoAllvDataType);

    uint8_t commMode = Mc2Comm::GetCommModeFromEnv();
    if (commMode == Mc2Comm::COMM_MODE_AICPU) {
        hcclCcTilingConfig.SetCommEngine(Mc2Comm::ENGINE_AICPU);
    }

    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(tilingData->hcclA2avTilingInfo.hcclInitTiling) != 0,
        OP_LOGE(A_INNER_DEBUG, "mc2CcTilingConfig GetTiling hcclInitTiling failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(tilingData->hcclA2avTilingInfo.a2avCcTiling) != 0,
        OP_LOGE(A_INNER_DEBUG, "mc2CcTilingConfig GetTiling a2avCcTiling failed"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::setNumBlocks(gert::TilingContext* context)
{
    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    uint64_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    static const PlatFormMemSize PLATFORM_SIZE(ascendcPlatform);
    libApiWorkSpaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint64_t numBlocks = mc2tiling::GetNumBlocks(aicNum, aivNum, A_INNER_DEBUG);
    OP_TILING_CHECK(
        (PLATFORM_SIZE.ubSize == 0U) || (PLATFORM_SIZE.l1Size == 0U) || (PLATFORM_SIZE.l0CSize == 0U) ||
        (PLATFORM_SIZE.l0ASize == 0U) || (PLATFORM_SIZE.l0BSize == 0U),
        OP_LOGE(
            A_INNER_DEBUG,
            "platform info is invalid, ubSize=%lu, l1Size=%lu, l0CSize=%lu, l0ASize=%lu, l0BSize=%lu",
            PLATFORM_SIZE.ubSize, PLATFORM_SIZE.l1Size, PLATFORM_SIZE.l0CSize,
            PLATFORM_SIZE.l0ASize, PLATFORM_SIZE.l0BSize),
        return ge::GRAPH_FAILED);

    tilingData->taskTilingInfo.ubSize = PLATFORM_SIZE.ubSize;
    ubSize_ = PLATFORM_SIZE.ubSize;
    context->SetBlockDim(static_cast<uint32_t>(numBlocks));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::Init()
{
    tilingData = context_->GetTilingData<AlltoAllvGmmTilingData>();
    OP_TILING_CHECK(
        GetContextAttr(context_) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Get context attr failed!"),
        return ge::GRAPH_FAILED);

    if (context_->GetOptionalInputShape(NON_QUANT_MM_X_INDEX) != nullptr) {
        OP_TILING_CHECK(context_->GetOptionalInputShape(NON_QUANT_MM_X_INDEX) == nullptr,
            OP_LOGE(A_INNER_DEBUG, "GetOptionalInputShape of mm_x returns null."),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(context_->GetOutputShape(OUTPUT_MM_Y_INDEX) == nullptr,
            OP_LOGE(A_INNER_DEBUG, "GetOutputShape of mm_y returns null."),
            return ge::GRAPH_FAILED);
    }
    OP_TILING_CHECK(
        CheckShapeDims(context_) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Check shape dim failed!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckDType(context_) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Check dtype failed!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckShapeRelation(context_) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Check shape relation failed!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        GetShapeAndFormat(context_) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Get shape and format failed!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckShapeSize(context_) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Check shape size failed!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckAttrsShapeSize(context_) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Check attrs shape size failed!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckAttrsShapeRelation(context_) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Check attrs shape relation failed!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckSendRecvDataVolumn(context_) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Check send recv data volume failed!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckMKN(context_) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Check mkn failed!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckMmShapeDims(context_) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Check mm shape dims failed!"),
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
    OP_LOGD(A_INNER_DEBUG, "AlltoAllvGmm GetTilingKey: gmmTrans=%d, mmTrans=%d, commMode=%u, tilingKey=%lu",
            tilingekyGmmTrans, tilingekyMmTrans, commMode, tilingKey);
    return tilingKey;
}

ge::graphStatus AlltoAllvGmmTiling::RunFusionKernelTiling(gert::TilingContext* context)
{
    OP_LOGD(A_INNER_DEBUG, "begin RunFusionKernelTiling.");

    OP_TILING_CHECK(
        SetHcclTiling(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "set hccl tiling failed!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        setNumBlocks(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "set numBlocks failed!"),
        return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(A_INNER_DEBUG, "GetAttrs returned null."), return ge::GRAPH_FAILED);

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(A_INNER_DEBUG, "GetAttrPointer epWorldSize returned null."),
        return ge::GRAPH_FAILED);
    epWorldSize_ = *epWorldSizePtr;

    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK((sendCountsPtr == nullptr) || (recvCountsPtr == nullptr),
        OP_LOGE(A_INNER_DEBUG, "sendCountsPtr or recvCountsPtr is null."), return ge::GRAPH_FAILED);

    const uint64_t* sendCounts = static_cast<const uint64_t*>(sendCountsPtr->GetData());
    const uint64_t* recvCounts = static_cast<const uint64_t*>(recvCountsPtr->GetData());
    uint64_t sendCountsSize = sendCountsPtr->GetSize();
    e_ = sendCountsSize / epWorldSize_;

    auto gmmXShape = context->GetInputShape(GMM_X_INDEX)->GetStorageShape();
    bsk_ = gmmXShape.GetDim(0);
    h1_ = gmmXShape.GetDim(1);

    auto transGmmWeightPtr = attrs->GetAttrPointer<bool>(NON_QUANT_ATTR_TRANS_GMM_WEIGHT_INDEX);
    auto transMmWeightPtr = attrs->GetAttrPointer<bool>(NON_QUANT_ATTR_TRANS_MM_WEIGHT_INDEX);
    OP_TILING_CHECK((transGmmWeightPtr == nullptr) || (transMmWeightPtr == nullptr),
        OP_LOGE(A_INNER_DEBUG, "transGmmWeightPtr or transMmWeightPtr is null."), return ge::GRAPH_FAILED);
    transGmmWeight_ = *transGmmWeightPtr;
    transMmWeight_ = *transMmWeightPtr;

    auto gmmWeightShape = context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape();
    e_ = gmmWeightShape.GetDim(0);
    uint64_t n1DimIndex = transGmmWeight_ ? DIM_ONE : DIM_TWO;
    n1_ = gmmWeightShape.GetDim(n1DimIndex);

    auto gmmYShape = context->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape();
    a_ = gmmYShape.GetDim(0);

    if (context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX) != nullptr) {
        auto mmXShape = context->GetOptionalInputShape(NON_QUANT_MM_X_INDEX)->GetStorageShape();
        bs_ = mmXShape.GetDim(0);
    }

    OP_TILING_CHECK(
        DoAiCoreTiling(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "GMMAlltoAllv DoAiCoreTiling failed."),
        return ge::GRAPH_FAILED);

    uint64_t tilingKey = GetTilingKey();
    context->SetTilingKey(tilingKey);

    OP_LOGD(A_INNER_DEBUG, "end RunFusionKernelTiling, tilingKey is %lu", tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::DoAiCoreTiling(const gert::TilingContext* context)
{
    OP_LOGD(A_INNER_DEBUG, "begin DoAiCoreTiling.");
    auto dTypeForMM = matmul_tiling::DataType::DT_FLOAT16;
    if (mmXDataType_ == ge::DT_BF16) {
        dTypeForMM = matmul_tiling::DataType::DT_BF16;
    }
    OP_LOGD(
        A_INNER_DEBUG, "mmXDataType_ is %d, dTypeForMM is %d.", static_cast<int>(mmXDataType_), static_cast<int>(dTypeForMM));
    OP_LOGD(A_INNER_DEBUG, "end DoAiCoreTiling.");
    return ge::GRAPH_SUCCESS;
}

uint64_t AlltoAllvGmmTilingStruct::GetTilingKey() const
{
    const uint64_t tilingKey = context_->GetTilingKey();
    OP_LOGD(A_INNER_DEBUG, "AlltoAllvGmmTiling get tiling key %lu", tilingKey);
    return tilingKey;
}

bool AlltoAllvGmmTilingStruct::IsCapable()
{
    return true;
}

ge::graphStatus AlltoAllvGmmTilingBase::GetPlatformInfo()
{
    OP_LOGD(context_->GetNodeName(), "start GetPlatformInfo.");
    if (GetCommonPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckCommonPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context_->GetNodeName(), "end GetPlatformInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::GetShapeAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "start GetShapeAttrsInfo.");
    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(A_INNER_DEBUG, "GetAttrs returned null."), return ge::GRAPH_FAILED);
    groupPtr_ = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    OP_TILING_CHECK(groupPtr_ == nullptr, OP_LOGE(A_INNER_DEBUG, "group attr can not be null."),
        return ge::GRAPH_FAILED);
    group_ = groupPtr_;
    epWorldSizePtr_ = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    OP_TILING_CHECK(epWorldSizePtr_ == nullptr, OP_LOGE(A_INNER_DEBUG, "epWorldSize attr can not be null."),
        return ge::GRAPH_FAILED);
    epWorldSize_ = *epWorldSizePtr_;
    sendCountsPtr_ = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    OP_TILING_CHECK(sendCountsPtr_ == nullptr, OP_LOGE(A_INNER_DEBUG, "sendCounts attr can not be null."),
        return ge::GRAPH_FAILED);
    sendCounts = static_cast<const int64_t*>(sendCountsPtr_->GetData());
    recvCountsPtr_ = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(recvCountsPtr_ == nullptr, OP_LOGE(A_INNER_DEBUG, "recvCounts attr can not be null."),
        return ge::GRAPH_FAILED);
    recvCounts = static_cast<const int64_t*>(recvCountsPtr_->GetData());
    transGmmWeightPtr_ = attrs->GetAttrPointer<bool>(NON_QUANT_ATTR_TRANS_GMM_WEIGHT_INDEX);
    if (transGmmWeightPtr_ != nullptr) {
        transGmmWeight_ = *transGmmWeightPtr_;
    }
    transMmWeightPtr_ = attrs->GetAttrPointer<bool>(NON_QUANT_ATTR_TRANS_MM_WEIGHT_INDEX);
    if (transMmWeightPtr_ != nullptr) {
        transMmWeight_ = *transMmWeightPtr_;
    }
    permuteOutFlagPtr_ = attrs->GetAttrPointer<bool>(NON_QUANT_ATTR_PERMUTE_OUT_FLAG_INDEX);
    if (permuteOutFlagPtr_ != nullptr) {
        permuteOutFlag_ = *permuteOutFlagPtr_;
    }
    OP_LOGD(context_->GetNodeName(), "end GetShapeAttrsInfo.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTilingBase::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "start PostTiling.");
    auto tilingData = context_->GetTilingData<AlltoAllvGmmTilingData>();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, tilingData);
    
    tilingData->taskTilingInfo.BSK = bsk_;
    tilingData->taskTilingInfo.BS = bs_;
    tilingData->taskTilingInfo.H1 = h1_;
    tilingData->taskTilingInfo.A = a_;
    tilingData->taskTilingInfo.N1 = n1_;
    tilingData->taskTilingInfo.epWorldSize = epWorldSize_;
    tilingData->taskTilingInfo.e = e_;
    tilingData->taskTilingInfo.ubSize = ubSize_;
    tilingData->taskTilingInfo.mainLoopExpertNum = e_;
    tilingData->taskTilingInfo.tailLoopExpertNum = 0;
    tilingData->taskTilingInfo.totalLoopCount = e_;
    
    auto attrs = context_->GetAttrs();
    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    const int64_t* sendCountsData = static_cast<const int64_t*>(sendCountsPtr->GetData());
    const int64_t* recvCountsData = static_cast<const int64_t*>(recvCountsPtr->GetData());
    
    for (uint32_t i = 0; i < e_ * epWorldSize_; i++) {
        tilingData->taskTilingInfo.sendCnt[i] = sendCountsData[i];
        tilingData->taskTilingInfo.recvCnt[i] = recvCountsData[i];
    }
    
    tilingData->isPermuteOut = permuteOutFlag_;
    
    OP_LOGD(context_->GetNodeName(), "end PostTiling.");
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