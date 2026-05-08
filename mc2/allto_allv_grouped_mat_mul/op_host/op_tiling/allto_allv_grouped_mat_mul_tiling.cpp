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
constexpr uint64_t COMM_TILE = 8; // 每卡数据分配几次计算

static inline uint32_t SixteenAlign(uint32_t a, bool up = false)
{
    if (up) {
        a += 15; // 15: 16 bytes up-align
    }
    return a & ~15; // ~15: 16 bytes down-align
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

// 已删除PrintTilingDataGMM/MM函数：TCubeTiling结构不存在



// 获取入参参数
ge::graphStatus AlltoAllvGmmTiling::GetContextAttr(const gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(A_INNER_DEBUG, "GetAttrs returned null."), return ge::GRAPH_FAILED);
    
    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    OP_TILING_CHECK((groupEpPtr == nullptr) || (epWorldSizePtr == nullptr),
        OP_LOGE(A_INNER_DEBUG, "groupEpPtr or epWorldSizePtr is null."), return ge::GRAPH_FAILED);
    
    auto mmXStorageShape = context->GetOptionalInputShape(MM_X_INDEX);
    tilingData->isPermuteOut = (mmXStorageShape != nullptr);
    
    group_ = groupEpPtr;
    epWorldSize_ = *epWorldSizePtr;
    
    OP_LOGI(A_INNER_DEBUG, "epGroup is %s, epWorldSize is %lu.", group_, epWorldSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::GetShapeAndFormat(const gert::TilingContext* context)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckMKN(const gert::TilingContext* context)
{
    (void)context; // Unused
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
    if (context->GetOptionalInputShape(MM_X_INDEX) != nullptr) {
        OP_TILING_CHECK(
            maxMForMM_ > maxMKN || maxNForMM_ > maxMKN || maxKForMM_ > maxMKN,
            OP_LOGE(A_INNER_DEBUG, "32B-aligned m, n or k axis for mm is out of range int32!"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckSendRecvDataVolumn(const gert::TilingContext* context) const
{
    // 单卡之间通信数据 [2M,100M]
    uint64_t eExpert = tilingData->taskTilingInfo.e;
    uint64_t epWorldSize = tilingData->taskTilingInfo.epWorldSize;
    uint64_t recvSendMin = static_cast<uint64_t>(2U * 1024U * 1024U);       // 通信量下限 2MB=2*1024*1024

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
                sendSum += sendCounts[j] * H1 * 2U; // /sizeof(gmmX) = 2U
            }
        }
    }

    return ge::GRAPH_SUCCESS;
}

// 检查入参 shape size
ge::graphStatus AlltoAllvGmmTiling::CheckShapeSize(const gert::TilingContext* context) const
{
    OP_TILING_CHECK(
        (context->GetInputShape(GMM_X_INDEX) == nullptr) || (context->GetInputShape(GMM_WEIGHT_INDEX) == nullptr),
        OP_LOGE(A_INNER_DEBUG, "GetInputShape gmmX or gmmWeight returned null."), return ge::GRAPH_FAILED);

    uint64_t BSK = context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(0);
    if (BSK <= NUM_ZERO || BSK >= MAX_BSK) {
        OP_LOGE(A_INNER_DEBUG, "BSK should be in (0, 52428800), but got %lu!", BSK);
        return ge::GRAPH_FAILED;
    }
    uint64_t H1 = context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(
        (H1 <= NUM_ZERO) || (H1 >= MAX_SHAPE_SIZE),
        OP_LOGE(A_INNER_DEBUG, "H1 should be in (0, 65536), but got %lu!", H1), return ge::GRAPH_FAILED);

    uint64_t N1 = transGmmWeight_ ?
                      context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(1) :
                      context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(NUM_TWO);
    OP_TILING_CHECK(
        N1 <= NUM_ZERO || N1 >= MAX_SHAPE_SIZE, OP_LOGE(A_INNER_DEBUG, "N1 should be in (0, 65536), but got %lu!", N1),
        return ge::GRAPH_FAILED);

    if (context->GetOptionalInputShape(MM_X_INDEX) != nullptr) {
        uint64_t BS = context->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDim(0);
        if (BS <= NUM_ZERO) {
            OP_LOGE(A_INNER_DEBUG, "BS should be larger than 0, but got %lu!", BS);
            return ge::GRAPH_FAILED;
        }
        uint64_t H2 = context->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDim(1);
        if (H2 <= NUM_ZERO || H2 > MAX_SHARED_H_SHAPE_SIZE) {
            OP_LOGE(A_INNER_DEBUG, "H2 should be in (0, 12288], but got %lu!", H2);
            return ge::GRAPH_FAILED;
        }
        uint64_t N2 = transMmWeight_ ?
                          context->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDim(0) :
                          context->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDim(1);
        if (N2 <= NUM_ZERO || N2 >= MAX_SHAPE_SIZE) {
            OP_LOGE(A_INNER_DEBUG, "N2 should be in (0, 65536), but got %lu!", N2);
            return ge::GRAPH_FAILED;
        }
        if (BS == 0) {
            OP_LOGE(A_INNER_DEBUG, "BS can not be zero, but got %lu!", BS);
            return ge::GRAPH_FAILED;
        }
        uint64_t topK = BSK / BS;
        if (topK < NUM_TWO || topK > NUM_EIGHT) {
            OP_LOGE(A_INNER_DEBUG, "topK should be in [2, 8], but got %lu!", topK);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckAttrsShapeSize(const gert::TilingContext* context) const
{
    uint64_t E_ep = tilingData->taskTilingInfo.e;
    if (E_ep <= NUM_ZERO || E_ep > E_MAX_VALUE_NON_QUANT) {
        OP_LOGE(A_INNER_DEBUG, "E_ep should be in (0, %d], but got %lu!", E_MAX_VALUE_NON_QUANT, E_ep);
        return ge::GRAPH_FAILED;
    }
    uint64_t epWorldSize = tilingData->taskTilingInfo.epWorldSize;
    auto platformInfo = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    std::vector<int64_t> epWorldSizeOptional;
    std::string epWorldSizeNum;
    epWorldSizeOptional = GetEpWorldSizeOptional();
    for (size_t i = 0; i < epWorldSizeOptional.size(); i++) {
        epWorldSizeNum += (std::to_string(epWorldSizeOptional[i]) + " ");
    }
    OP_TILING_CHECK(
        std::find(epWorldSizeOptional.begin(), epWorldSizeOptional.end(), epWorldSize) == epWorldSizeOptional.end(),
        OP_LOGE(A_INNER_DEBUG, "epWorldSize[%lu] should be %s!", epWorldSize, epWorldSizeNum.c_str()), return ge::GRAPH_FAILED);
    // 对sendCounts和recvCounts校验
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(A_INNER_DEBUG, "GetAttrs returned null."), return ge::GRAPH_FAILED);
    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK((sendCountsPtr == nullptr) || (recvCountsPtr == nullptr),
        OP_LOGE(A_INNER_DEBUG, "sendCountsPtr or recvCountsPtr is null."), return ge::GRAPH_FAILED);
    uint64_t sendCountsSize = sendCountsPtr->GetSize();
    uint64_t recvCountsSize = recvCountsPtr->GetSize();
    OP_TILING_CHECK(sendCountsSize != recvCountsSize,
        OP_LOGE(
            A_INNER_DEBUG, "The size of sendCounts(e*ep) %lu should be equal to recvCounts(e*ep) %lu !", sendCountsSize,
            recvCountsSize),
        return ge::GRAPH_FAILED);
    if (E_ep * epWorldSize != sendCountsSize) {
        OP_LOGE(A_INNER_DEBUG,
            "The first dim of gmmWeight(e, H1, N1) %lu  multi epWorldSize %lu shoubl be equal to the size of "
            "sendCounts(e*ep) %lu!", E_ep, epWorldSize, sendCountsSize);
        return ge::GRAPH_FAILED;
    }
    if ((E_ep * epWorldSize <= NUM_ZERO) || (E_ep * epWorldSize > MAX_EXPERT_NUM)) {
        OP_LOGE(
            A_INNER_DEBUG, "The size of send_counts(e*ep) and recv_counts(e*ep) should be in (0, %d], but got %lu!",
            MAX_EXPERT_NUM, E_ep * epWorldSize);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 检查sendcounts recvcounts shape 之间的关系
ge::graphStatus AlltoAllvGmmTiling::CheckAttrsShapeRelation(const gert::TilingContext* context) const
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(A_INNER_DEBUG, "GetAttrs returned null."), return ge::GRAPH_FAILED);

    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(
        (sendCountsPtr == nullptr) || (recvCountsPtr == nullptr),
        OP_LOGE(A_INNER_DEBUG, "sendCountsPtr or recvCountsPtr is null."), return ge::GRAPH_FAILED);

    uint64_t sendCountsSize = sendCountsPtr->GetSize();
    uint64_t recvCountsSize = recvCountsPtr->GetSize();

    for (size_t i = 0; i < sendCountsPtr->GetSize() && i < MC2KernelTemplate::MAX_EXPERT_SIZE; i++) {
        tilingData->taskTilingInfo.sendCnt[i] = static_cast<int32_t>(
            static_cast<const uint64_t*>(sendCountsPtr->GetData())[i]);
    }
    for (size_t i = 0; i < recvCountsPtr->GetSize() && i < MC2KernelTemplate::MAX_EXPERT_SIZE; i++) {
        tilingData->taskTilingInfo.recvCnt[i] = static_cast<int32_t>(
            static_cast<const uint64_t*>(recvCountsPtr->GetData())[i]);
    }

    const uint64_t* sendCounts = static_cast<const uint64_t*>(sendCountsPtr->GetData());
    uint64_t sendCountsSum = std::accumulate(sendCounts, sendCounts + sendCountsSize, 0ULL);
    OP_TILING_CHECK(
        sendCountsSum != tilingData->taskTilingInfo.BSK,
        OP_LOGE(
            A_INNER_DEBUG, "The sum of sendCounts %lu should be equal to BSK %lu!", sendCountsSum,
            tilingData->taskTilingInfo.BSK),
        return ge::GRAPH_FAILED);

    const uint64_t* recvCounts = static_cast<const uint64_t*>(recvCountsPtr->GetData());
    uint64_t recvCountsSum = std::accumulate(recvCounts, recvCounts + recvCountsSize, 0ULL);
    OP_TILING_CHECK(
        recvCountsSum != tilingData->taskTilingInfo.A,
        OP_LOGE(
            A_INNER_DEBUG, "The sum of recvCounts %lu should be equal to A %lu!", recvCountsSum,
            tilingData->taskTilingInfo.A),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 检查入参 shape 之间的关系
ge::graphStatus AlltoAllvGmmTiling::CheckShapeRelation(const gert::TilingContext* context) const
{
    OP_TILING_CHECK((context->GetInputShape(GMM_WEIGHT_INDEX) == nullptr) || (context->GetInputShape(GMM_X_INDEX) == nullptr),
                     OP_LOGE(A_INNER_DEBUG, "GetInputShape gmmX or gmmWeight returned nullptr."), return ge::GRAPH_FAILED);

    uint64_t gmmWeightH1 = transGmmWeight_ ?
                               context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(NUM_TWO) :
                               context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDim(1);
    uint64_t gmmXH1 = context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(gmmXH1 != gmmWeightH1, OP_LOGE(A_INNER_DEBUG,
                    "The H1 %lu of gmmX(BSK, H1) should be equal to the H1 %lu of gmmWeight(e, H1, N1) !", gmmXH1, gmmWeightH1),
                    return ge::GRAPH_FAILED);

    if (context->GetOptionalInputShape(MM_X_INDEX) != nullptr) {
        uint64_t mmXH2 = context->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDim(1);
        uint64_t mmWeightH2 = transMmWeight_ ?
                                  context->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDim(1) :
                                  context->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(mmXH2 != mmWeightH2, OP_LOGE(A_INNER_DEBUG,
                        "The H2 %lu of mmX(BS, H2) should be equal to the H2 %lu of mmWeight(H2, N2)!", mmXH2, mmWeightH2),
                        return ge::GRAPH_FAILED);

        uint64_t mmXBS = context->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDim(0);
        uint64_t mmYBS = context->GetOutputShape(OUTPUT_MM_Y_INDEX)->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(mmXBS != mmYBS, OP_LOGE(A_INNER_DEBUG,
                        "The BS %lu of mmX(BS, H2) should be equal to the BS %lu of mmY(BS, N2)!", mmXBS, mmYBS),
                        return ge::GRAPH_FAILED);
    }

    if (tilingData->isPermuteOut) {
        OP_TILING_CHECK((context->GetOutputShape(OUTPUT_GMM_Y_INDEX) == nullptr) ||
                        (context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX) == nullptr),
                        OP_LOGE(A_INNER_DEBUG, "GetPermuteOutputShape GmmY or permuteOut returned null."),
                        return ge::GRAPH_FAILED);
        uint64_t gmmYA = context->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape().GetDim(0);
        uint64_t permuteA = context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX)->GetStorageShape().GetDim(0);
        uint64_t permuteH1 = context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX)->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(gmmXH1 != permuteH1, OP_LOGE(A_INNER_DEBUG,
                        "The H1 %lu of gmmX(BSK, H1) should be equal to the H1 %lu of permuteOut(A, H1)!", gmmXH1, permuteH1),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(gmmYA != permuteA, OP_LOGE(A_INNER_DEBUG,
                        "The A %lu of gmmY(A, H1) should be equal to the A %lu of permuteOut(A, H1)!", gmmYA, permuteA),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckMmShapeDims(const gert::TilingContext* context) const
{
    if (context->GetOptionalInputShape(MM_X_INDEX) != nullptr) {
        if (context->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDimNum() != NUM_TWO) {
            OP_LOGE(
                A_INNER_DEBUG, "The dim of mmX(BS, H2) should be 2, but got %lu!",
                context->GetOptionalInputShape(MM_X_INDEX)->GetStorageShape().GetDimNum());
            return ge::GRAPH_FAILED;
        }
        if (context->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDimNum() != NUM_TWO) {
            OP_LOGE(
                A_INNER_DEBUG, "The dim of mmWeight(H2, N2) should be 2, but got %lu!",
                context->GetOptionalInputShape(MM_WEIGHT_INDEX)->GetStorageShape().GetDimNum());
            return ge::GRAPH_FAILED;
        }
        if (context->GetOutputShape(OUTPUT_MM_Y_INDEX)->GetStorageShape().GetDimNum() != NUM_TWO) {
            OP_LOGE(
                A_INNER_DEBUG, "The dim of mmY(BS, N2) should be 2, but got %lu!",
                context->GetOutputShape(OUTPUT_MM_Y_INDEX)->GetStorageShape().GetDimNum());
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_TILING_CHECK(
            (context->GetOutputShape(OUTPUT_MM_Y_INDEX) != nullptr &&
             context->GetOutputShape(OUTPUT_MM_Y_INDEX)->GetStorageShape().GetDimNum() != NUM_ZERO),
            OP_LOGE(A_INNER_DEBUG, "The mmY should be null when mmX and mmWeight are null!"), return ge::GRAPH_FAILED);
        if (transMmWeight_) {
            OP_LOGE(A_INNER_DEBUG, "The trans_mm_weight should be false when mmX mmWeight mmY is null!");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

// 检查入参 shape 维度
ge::graphStatus AlltoAllvGmmTiling::CheckShapeDims(const gert::TilingContext* context)
{
    OP_TILING_CHECK(
        (context->GetInputShape(GMM_X_INDEX) == nullptr) || (context->GetInputShape(GMM_WEIGHT_INDEX) == nullptr),
        OP_LOGE(A_INNER_DEBUG, "GetInputShape gmmX or gmmWeight returned null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (context->GetOutputShape(OUTPUT_GMM_Y_INDEX) == nullptr),
        OP_LOGE(A_INNER_DEBUG, "GetOutputShape gmmY returned null."), return ge::GRAPH_FAILED);

    if (context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDimNum() != NUM_TWO) {
        OP_LOGE(A_INNER_DEBUG, "The dim of gmmX(BSK, H1) should be 2, but got %lu!",
            context->GetInputShape(GMM_X_INDEX)->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDimNum() != NUM_THREE) {
        OP_LOGE(A_INNER_DEBUG, "The dim of gmmWeight(e, H1, N1) should be 3, but got %lu!",
            context->GetInputShape(GMM_WEIGHT_INDEX)->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }
    if (context->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape().GetDimNum() != NUM_TWO) {
        OP_LOGE(A_INNER_DEBUG, "The dim of gmmY(A, N1) should be 2, but got %lu!",
            context->GetOutputShape(OUTPUT_GMM_Y_INDEX)->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }
    if (context->GetOptionalInputShape(MM_X_INDEX) != nullptr) {
        OP_TILING_CHECK(context->GetOptionalInputShape(MM_WEIGHT_INDEX) == nullptr,
            OP_LOGE(A_INNER_DEBUG, "GetOptionalInputShape of mm_weight is null."),
            return ge::GRAPH_FAILED);
    }

    OP_TILING_CHECK(
        CheckMmShapeDims(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Check mm shape dim failed!"),
        return ge::GRAPH_FAILED);

    if (tilingData->isPermuteOut) {
        if (context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX) == nullptr ||
            context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX)->GetStorageShape().GetDimNum() == NUM_ZERO) {
            OP_LOGE(A_INNER_DEBUG, "The permuteOut should not be null when permuteOutFlag is true!");
            return ge::GRAPH_FAILED;
        }
        if (context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX)->GetStorageShape().GetDimNum() != NUM_TWO) {
            OP_LOGE(A_INNER_DEBUG, "The dim of permuteOut(A, H1) should be 2, but got %lu!",
                context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX)->GetStorageShape().GetDimNum());
            return ge::GRAPH_FAILED;
        }
    } else {
        if (context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX) != nullptr &&
            context->GetOutputShape(OUTPUT_PERMUTE_OUT_INDEX)->GetStorageShape().GetDimNum() != NUM_ZERO) {
            OP_LOGE(A_INNER_DEBUG, "The permuteOut should be null when permuteOutFlag is false!");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::SetHcclTiling(const gert::TilingContext* context) const
{
    (void)context; // Unused
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(A_INNER_DEBUG, "Tiling Data is null!"), return ge::GRAPH_FAILED);

    uint32_t alltoAllvCmd = 8U;
    std::string alltoAllvConfig = "AlltoAll=level0:fullmesh;level1:pairwise";

    const uint32_t alltoAllvReduceType = 0u;
    auto outputDataType = context->GetOutputDesc(OUTPUT_Y_INDEX)->GetDataType();
    auto inputDataType = context->GetInputDesc(GMM_X_INDEX)->GetDataType();
    OP_TILING_CHECK(
        mc2tiling::HCCL_DATA_TYPE.find(outputDataType) == mc2tiling::HCCL_DATA_TYPE.end(),
        OP_LOGE(A_INNER_DEBUG, "%s is Unsupported outputdata type!", Ops::Base::ToString(outputDataType).c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        mc2tiling::HCCL_DATA_TYPE.find(inputDataType) == mc2tiling::HCCL_DATA_TYPE.end(),
        OP_LOGE(A_INNER_DEBUG, "%s is Unsupported inputdata type!", Ops::Base::ToString(inputDataType).c_str()),
        return ge::GRAPH_FAILED);

    auto alltoAllvDstDataType = static_cast<uint8_t>(mc2tiling::HCCL_DATA_TYPE.find(outputDataType)->second);
    auto alltoAllvSrcDataType = static_cast<uint8_t>(mc2tiling::HCCL_DATA_TYPE.find(inputDataType)->second);

    Mc2CcTilingConfig hcclCcTilingConfig(group_, alltoAllvCmd, alltoAllvConfig,
                                         alltoAllvReduceType, alltoAllvDstDataType, alltoAllvSrcDataType);
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(tilingData->hcclA2avTilingInfo.hcclInitTiling) != 0,
        OP_LOGE(A_INNER_DEBUG, "mc2CcTilingConfig mc2tiling GetTiling hcclInitTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(tilingData->hcclA2avTilingInfo.a2avCcTiling) != 0,
        OP_LOGE(A_INNER_DEBUG, "mc2CcTilingConfig mc2tiling GetTiling a2avCcTiling failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::CheckDType(const gert::TilingContext* context) const
{
    OP_TILING_CHECK(
        (context->GetInputDesc(GMM_X_INDEX) == nullptr) || (context->GetInputDesc(GMM_WEIGHT_INDEX) == nullptr),
        OP_LOGE(A_INNER_DEBUG, "GetInputDesc gmmX or gmmWeight returned null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        context->GetOutputDesc(OUTPUT_Y_INDEX) == nullptr, OP_LOGE(A_INNER_DEBUG, "GetOutputDesc y returned null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (context->GetInputDesc(GMM_X_INDEX)->GetDataType() != ge::DT_FLOAT16) &&
            (context->GetInputDesc(GMM_X_INDEX)->GetDataType() != ge::DT_BF16),
        OP_LOGE(A_INNER_DEBUG, "Unsupported dataType, gmmx only support float16 and bfloat16!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (context->GetInputDesc(GMM_X_INDEX)->GetDataType() != context->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType()) ||
            (context->GetInputDesc(GMM_X_INDEX)->GetDataType() !=
             context->GetOutputDesc(OUTPUT_Y_INDEX)->GetDataType()),
        OP_LOGE(A_INNER_DEBUG, "The dataType of gmmWeight and gmmY should be the same with gmmX."), return ge::GRAPH_FAILED);
    if (context->GetOptionalInputShape(MM_X_INDEX) != nullptr) {
        auto mmXDex = context->GetOptionalInputDesc(MM_X_INDEX);
        OP_TILING_CHECK(mmXDex == nullptr, OP_LOGE(A_INNER_DEBUG, "Flag isNeedMM is True, but MM_X is null."), return ge::GRAPH_FAILED);
        auto mmWeightDesc = context->GetOptionalInputDesc(MM_WEIGHT_INDEX);
        OP_TILING_CHECK(
            mmWeightDesc == nullptr, OP_LOGE(A_INNER_DEBUG, "Flag isNeedMM is True, MM_WEIGHT is null."), return ge::GRAPH_FAILED);
        auto mmYDesc = context->GetOutputDesc(OUTPUT_MM_Y_INDEX);
        OP_TILING_CHECK(mmYDesc == nullptr, OP_LOGE(A_INNER_DEBUG, "GetOutputDesc mmY returned null."), return ge::GRAPH_FAILED);

        OP_TILING_CHECK(
            (context->GetOptionalInputDesc(MM_X_INDEX)->GetDataType() != ge::DT_FLOAT16) &&
                (context->GetOptionalInputDesc(MM_X_INDEX)->GetDataType() != ge::DT_BF16),
            OP_LOGE(A_INNER_DEBUG, "Unsupported dataType, mmx only support float16 and bfloat16!"), return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            (context->GetOptionalInputDesc(MM_X_INDEX)->GetDataType() !=
             context->GetOptionalInputDesc(MM_WEIGHT_INDEX)->GetDataType()) ||
                (context->GetOptionalInputDesc(MM_X_INDEX)->GetDataType() !=
                 context->GetOutputDesc(OUTPUT_MM_Y_INDEX)->GetDataType()),
            OP_LOGE(A_INNER_DEBUG, "The dataType of mmWeight and mmY should be the same with mmX."), return ge::GRAPH_FAILED);
        // 校验mmdataType和gmmdataType一致
        OP_TILING_CHECK(context->GetOptionalInputDesc(MM_X_INDEX)->GetDataType() != context->GetInputDesc(GMM_X_INDEX)->GetDataType(),
 	            OP_LOGE(context->GetNodeName(), "mmX data type (%s) must be the same as gmmX data type (%s) when shared expert is enabled.",
 	                ge::TypeUtils::DataTypeToSerialString(context->GetOptionalInputDesc(MM_X_INDEX)->GetDataType()).c_str(),
 	                ge::TypeUtils::DataTypeToSerialString(context->GetInputDesc(GMM_X_INDEX)->GetDataType()).c_str()),
 	            return ge::GRAPH_FAILED);
 	    OP_TILING_CHECK(context->GetOptionalInputDesc(MM_WEIGHT_INDEX)->GetDataType() != context->GetInputDesc(GMM_X_INDEX)->GetDataType(),
 	        OP_LOGE(context->GetNodeName(), "mmWeight data type (%s) must be the same as gmmX data type (%s) when shared expert is enabled.",
 	            ge::TypeUtils::DataTypeToSerialString(context->GetOptionalInputDesc(MM_WEIGHT_INDEX)->GetDataType()).c_str(),
 	            ge::TypeUtils::DataTypeToSerialString(context->GetInputDesc(GMM_X_INDEX)->GetDataType()).c_str()),
 	        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::Init()
{
    tilingData = context_->GetTilingData<AlltoAllvGmmTilingData>();
    OP_TILING_CHECK(
        GetContextAttr(context_) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Get context attr failed!"),
        return ge::GRAPH_FAILED);

    if (context_->GetOptionalInputShape(MM_X_INDEX) != nullptr) {
        OP_TILING_CHECK(context_->GetOptionalInputShape(MM_X_INDEX) == nullptr,
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
        CheckAttrsShapeSize(context_) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "Check Attrs shape size failed!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckAttrsShapeRelation(context_) != ge::GRAPH_SUCCESS,
        OP_LOGE(A_INNER_DEBUG, "Check Attrs Shape Relation failed!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckSendRecvDataVolumn(context_) != ge::GRAPH_SUCCESS,
        OP_LOGE(A_INNER_DEBUG, "Check Send Recv Data Volumn failed!"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        CheckMKN(context_) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "GMM CheckMKN failed."),
        return ge::GRAPH_FAILED);
    OP_LOGI(
        A_INNER_DEBUG,
        "AlltoAllvGmmTiling: maxM_ is %d, maxK_ is %d, maxN_ is %d, maxMForMM_ is %d, maxKForMM_ is %d, maxNForMM_ "
        "is %d.",
        maxM_, maxK_, maxN_, maxMForMM_, maxKForMM_, maxNForMM_);
    return ge::GRAPH_SUCCESS;
}

uint64_t AlltoAllvGmmTiling::GetTilingKey() const
{
    uint32_t templateMmDType = 0;  // ADD_TPL_FP16
    bool tilingkeyMm = false;
    bool tilingekyGmmTrans = false;
    bool tilingekyMmTrans = false;
    if (context_->GetInputDesc(GMM_X_INDEX)->GetDataType() == ge::DT_FLOAT16) {
        templateMmDType = 0;  // ADD_TPL_FP16
    } else if (context_->GetInputDesc(GMM_X_INDEX)->GetDataType() == ge::DT_BF16) {
        templateMmDType = 1;  // ADD_TPL_BP16
    }
    if (context_->GetOptionalInputShape(MM_X_INDEX) != nullptr) {
        tilingkeyMm = true;
    } else {
        tilingkeyMm = false;
    }
    if (transGmmWeight_) {
        tilingekyGmmTrans = true;
    } else {
        tilingekyGmmTrans = false;
    }
    if (transMmWeight_) {
        tilingekyMmTrans = true;
    } else {
        tilingekyMmTrans = false;
    }
    uint64_t tilingKey = GET_TPL_TILING_KEY(templateMmDType, tilingkeyMm,
                                    tilingekyGmmTrans, tilingekyMmTrans);

    OP_LOGD(A_INNER_DEBUG, "end RunFusionKernelTiling, tilingKey is %lu", tilingKey);
    return tilingKey;
}

ge::graphStatus AlltoAllvGmmTiling::setNumBlocks(gert::TilingContext* context){
    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    // 设置 CV 核数
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    uint64_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    static const PlatFormMemSize PLATFORM_SIZE(ascendcPlatform);
    static const platform_ascendc::SocVersion SOC_VERSION = ascendcPlatform.GetSocVersion();
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
    context->SetBlockDim(static_cast<uint32_t>(numBlocks));           // 通算融合场景 AIC_NUM:AIV_NUM = 1:2 默认启动

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGmmTiling::RunFusionKernelTiling(gert::TilingContext* context)
{
    OP_LOGD(A_INNER_DEBUG, "begin RunFusionKernelTiling.");

    OP_TILING_CHECK(
        SetHcclTiling(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "set hccl tiling failed!"),
        return ge::GRAPH_FAILED);

    // aicore tiling
    OP_TILING_CHECK(
        DoAiCoreTiling(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "GMMAlltoAllv DoAiCoreTiling failed."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        setNumBlocks(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "GMMAlltoAllv setNumBlocks failed."),
        return ge::GRAPH_FAILED);

    // set workspaces
    size_t* workspaces = context->GetWorkspaceSizes(1); // 1: fixed value
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(A_INNER_DEBUG, "get workspace failed"), return ge::GRAPH_FAILED);

    uint64_t commOut = a_ * h1_ * mmDataTypeSize;
    uint64_t permuteOut = permuteOutFlag_ ? 0 : (a_ * h1_ * mmDataTypeSize);
    workspaces[0] = libApiWorkSpaceSize_ + commOut + permuteOut;
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
    // 注意：使用量化scheduler后，不再需要CalMMTiling/SetMMTiling（GMMQuantTilingData自动填充）
    // 旧TCubeTiling相关代码已删除：commonTilingInfo/gmmTilingData/mmTilingData字段不存在
    OP_LOGD(A_INNER_DEBUG, "end DoAiCoreTiling.");
    return ge::GRAPH_SUCCESS;
}

// 已删除CalMMTiling函数：量化scheduler使用GMMQuantTilingData，不需要手动CalMMTiling
// 已删除SetMMTiling函数：量化scheduler使用GMMQuantTilingData，不需要手动SetMMTiling

bool AlltoAllvGmmTilingStruct::IsCapable()
{
    return true;
}

uint64_t AlltoAllvGmmTilingStruct::GetTilingKey() const
{
    const uint64_t tilingKey = context_->GetTilingKey();
    OP_LOGD(A_INNER_DEBUG, "AlltoAllvGmmTiling get tiling key %lu", tilingKey);
    return tilingKey;
}

ge::graphStatus AlltoAllvGmmTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(
        platformInfo == nullptr, VECTOR_INNER_ERR_REPORT_TILING(A_INNER_DEBUG, "fail to get platform info"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Every thing is done by DoOptiling.
ge::graphStatus AlltoAllvGmmTilingBase::GetShapeAttrsInfo()
{
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
    
    for (uint32_t i = 0; i < e_ * epWorldSize_; i++) {
        tilingData->taskTilingInfo.sendCnt[i] = sendCounts[i];
        tilingData->taskTilingInfo.recvCnt[i] = recvCounts[i];
    }
    
    tilingData->isPermuteOut = false;
    
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
    .TilingParse<AlltoAllvGmmCompileInfo>(TilingParseForAlltoAllvGmm); // 向框架注册入口函数
} // namespace optiling