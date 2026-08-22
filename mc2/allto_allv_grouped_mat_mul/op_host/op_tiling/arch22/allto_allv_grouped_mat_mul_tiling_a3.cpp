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
 * \file allto_allv_grouped_mat_mul_tiling_a3.cpp
 * \brief
 */

#include "allto_allv_grouped_mat_mul_tiling_a3.h"
#include "../../../op_kernel/allto_allv_grouped_mat_mul_aiv_comm.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>

#include "mc2_comm_utils.h"

namespace {
const char *A_INNER_DEBUG = "AlltoAllvGroupedMatMul Tiling";
constexpr size_t AIV_ATTR_COMM_MODE_INDEX = 7U;
constexpr uint32_t AIV_SCHEDULE_MODE = 1U;
constexpr uint32_t AIV_SYSTEM_WORKSPACE_SIZE = 16U * 1024U * 1024U;
constexpr uint64_t AIV_WORKSPACE_ALIGN = 512U;
constexpr uint32_t AIV_ALLTOALL_OP_TYPE = 8U;
constexpr uint32_t BF16_BYTES = 2U;
constexpr uint32_t UB_MOVE_CAPACITY = 16U * 1024U;

bool IsShapePresentAiv(const gert::StorageShape *shape)
{
    return shape != nullptr && shape->GetStorageShape().GetDimNum() > 0;
}

bool IsPositiveInt32(int64_t value)
{
    return value > 0 && value <= static_cast<int64_t>(std::numeric_limits<int32_t>::max());
}

uint64_t AlignWorkspace(uint64_t value)
{
    return (value + AIV_WORKSPACE_ALIGN - 1U) / AIV_WORKSPACE_ALIGN * AIV_WORKSPACE_ALIGN;
}

void SetCocTiling(uint32_t m, uint32_t k, uint32_t n, uint64_t ubSize, AlltoAllvGmmCoCTiling &tiling)
{
    if (m > 64U) {
        tiling.m0 = 128U;
        tiling.k0 = k < 64U ? 32U : 64U;
        tiling.n0 = 128U;
    } else if (k >= 64U && n >= 128U) {
        tiling.m0 = 64U;
        tiling.k0 = 64U;
        tiling.n0 = 128U;
    } else {
        tiling.m0 = 64U;
        tiling.k0 = 32U;
        tiling.n0 = 64U;
    }
    uint64_t moveCapacity = std::min<uint64_t>(UB_MOVE_CAPACITY, ubSize / (BF16_BYTES * 2U));
    tiling.ubMoveNum = static_cast<uint32_t>(moveCapacity / 16U * 16U);
    if (tiling.ubMoveNum == 0U) {
        tiling.ubMoveNum = 16U;
    }
    tiling.swizzlCount = 1U;
    tiling.swizzlDirect = 0U;
}

bool IsAivMode(const gert::TilingContext *context)
{
    if (context == nullptr || context->GetAttrs() == nullptr) {
        return false;
    }
    const char *commMode = context->GetAttrs()->GetAttrPointer<char>(AIV_ATTR_COMM_MODE_INDEX);
    return commMode != nullptr && std::strcmp(commMode, "aiv") == 0;
}

ge::graphStatus FillAivTiling(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(context->SetScheduleMode(AIV_SCHEDULE_MODE) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Failed to set AIV schedule mode."), return ge::GRAPH_FAILED);

    auto *tilingData = context->GetTilingData<AlltoAllvGmmAivTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"),
                    return ge::GRAPH_FAILED);
    *tilingData = {};

    const auto *attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);
    const char *group = attrs->GetAttrPointer<char>(optiling::ATTR_GROUP_INDEX);
    const int64_t *epWorldSize = attrs->GetAttrPointer<int64_t>(optiling::ATTR_EP_WORLD_SIZE_INDEX);
    const auto *sendCountsVector =
        attrs->GetAttrPointer<gert::ContinuousVector>(optiling::ATTR_SEND_COUNTS_INDEX);
    const auto *recvCountsVector =
        attrs->GetAttrPointer<gert::ContinuousVector>(optiling::ATTR_RECV_COUNTS_INDEX);
    const bool *transGmmWeight =
        attrs->GetAttrPointer<bool>(optiling::NON_QUANT_ATTR_TRANS_GMM_WEIGHT_INDEX);
    const bool *transMmWeight =
        attrs->GetAttrPointer<bool>(optiling::NON_QUANT_ATTR_TRANS_MM_WEIGHT_INDEX);
    const bool *permuteOutFlag =
        attrs->GetAttrPointer<bool>(optiling::NON_QUANT_ATTR_PERMUTE_OUT_FLAG_INDEX);
    OP_TILING_CHECK(group == nullptr || epWorldSize == nullptr || sendCountsVector == nullptr ||
                        recvCountsVector == nullptr || transGmmWeight == nullptr || transMmWeight == nullptr ||
                        permuteOutFlag == nullptr,
                    OP_LOGE(nodeName, "AIV attributes are incomplete."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*epWorldSize != 2 && *epWorldSize != 4 && *epWorldSize != 8,
                    OP_LOGE(nodeName, "AIV epWorldSize must be 2, 4 or 8, but got %lld.", *epWorldSize),
                    return ge::GRAPH_FAILED);

    int64_t rankSize = 0;
    OP_TILING_CHECK(!mc2tiling::GetRankSize(nodeName, group, rankSize),
                    OP_LOGE(nodeName, "Failed to get rank size for group %s.", group), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(rankSize != *epWorldSize,
                    OP_LOGE(nodeName, "epWorldSize %lld does not match HCCL rank size %lld.", *epWorldSize, rankSize),
                    return ge::GRAPH_FAILED);

    const auto *gmmXShape = context->GetInputShape(optiling::GMM_X_INDEX);
    const auto *gmmWeightShape = context->GetInputShape(optiling::GMM_WEIGHT_INDEX);
    const auto *gmmYShape = context->GetOutputShape(optiling::OUTPUT_GMM_Y_INDEX);
    const auto *gmmXDesc = context->GetInputDesc(optiling::GMM_X_INDEX);
    const auto *gmmWeightDesc = context->GetInputDesc(optiling::GMM_WEIGHT_INDEX);
    const auto *gmmYDesc = context->GetOutputDesc(optiling::OUTPUT_GMM_Y_INDEX);
    OP_TILING_CHECK(gmmXShape == nullptr || gmmWeightShape == nullptr || gmmYShape == nullptr ||
                        gmmXDesc == nullptr || gmmWeightDesc == nullptr || gmmYDesc == nullptr,
                    OP_LOGE(nodeName, "Required GMM tensors are incomplete."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gmmXDesc->GetDataType() != ge::DT_BF16 || gmmWeightDesc->GetDataType() != ge::DT_BF16 ||
                        gmmYDesc->GetDataType() != ge::DT_BF16,
                    OP_LOGE(nodeName, "AIV mode supports BF16 GMM tensors only."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gmmXDesc->GetStorageFormat() != ge::FORMAT_ND ||
                        gmmWeightDesc->GetStorageFormat() != ge::FORMAT_ND ||
                        gmmYDesc->GetStorageFormat() != ge::FORMAT_ND,
                    OP_LOGE(nodeName, "AIV mode supports ND GMM tensors only."), return ge::GRAPH_FAILED);

    const auto &xShape = gmmXShape->GetStorageShape();
    const auto &weightShape = gmmWeightShape->GetStorageShape();
    const auto &yShape = gmmYShape->GetStorageShape();
    OP_TILING_CHECK(xShape.GetDimNum() != 2 || weightShape.GetDimNum() != 3 || yShape.GetDimNum() != 2,
                    OP_LOGE(nodeName, "AIV GMM expects x/y to be 2D and weight to be 3D."),
                    return ge::GRAPH_FAILED);

    const int64_t inputM = xShape.GetDim(0);
    const int64_t inputK = xShape.GetDim(1);
    const int64_t expertPerRank = weightShape.GetDim(0);
    const int64_t weightK = weightShape.GetDim(*transGmmWeight ? 2 : 1);
    const int64_t weightN = weightShape.GetDim(*transGmmWeight ? 1 : 2);
    const int64_t outputM = yShape.GetDim(0);
    const int64_t outputN = yShape.GetDim(1);
    OP_TILING_CHECK(!IsPositiveInt32(inputM) || !IsPositiveInt32(inputK) ||
                        !IsPositiveInt32(expertPerRank) || !IsPositiveInt32(weightK) ||
                        !IsPositiveInt32(weightN) || !IsPositiveInt32(outputM),
                    OP_LOGE(nodeName, "AIV GMM dimensions must be positive int32 values."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(inputK != weightK || outputN != weightN,
                    OP_LOGE(nodeName, "AIV GMM K/N dimensions are inconsistent."), return ge::GRAPH_FAILED);

    const uint64_t countNum = sendCountsVector->GetSize();
    OP_TILING_CHECK(countNum != recvCountsVector->GetSize() ||
                        countNum != static_cast<uint64_t>(*epWorldSize) * static_cast<uint64_t>(expertPerRank) ||
                        countNum > A2AVGMM_MAX_COUNT_NUM,
                    OP_LOGE(nodeName, "AIV counts length must equal rankSize * expertPerRank and be <= %u.",
                            A2AVGMM_MAX_COUNT_NUM),
                    return ge::GRAPH_FAILED);
    const auto *sendCounts = static_cast<const int64_t *>(sendCountsVector->GetData());
    const auto *recvCounts = static_cast<const int64_t *>(recvCountsVector->GetData());
    OP_TILING_CHECK(sendCounts == nullptr || recvCounts == nullptr,
                    OP_LOGE(nodeName, "AIV counts data is null."), return ge::GRAPH_FAILED);

    uint64_t sendSum = 0U;
    uint64_t recvSum = 0U;
    for (uint64_t i = 0U; i < countNum; ++i) {
        OP_TILING_CHECK(sendCounts[i] < 0 || sendCounts[i] > inputM ||
                            recvCounts[i] < 0 || recvCounts[i] > outputM,
                        OP_LOGE(nodeName, "AIV counts[%lu] is out of range.", i), return ge::GRAPH_FAILED);
        tilingData->sendCounts[i] = sendCounts[i];
        tilingData->recvCounts[i] = recvCounts[i];
        sendSum += static_cast<uint64_t>(sendCounts[i]);
        recvSum += static_cast<uint64_t>(recvCounts[i]);
    }
    OP_TILING_CHECK(sendSum != static_cast<uint64_t>(inputM) || recvSum != static_cast<uint64_t>(outputM),
                    OP_LOGE(nodeName, "AIV counts sums must match input/output token counts."),
                    return ge::GRAPH_FAILED);

    uint64_t maxExpertM = 0U;
    for (int64_t expert = 0; expert < expertPerRank; ++expert) {
        uint64_t expertM = 0U;
        for (int64_t rank = 0; rank < rankSize; ++rank) {
            expertM += static_cast<uint64_t>(recvCounts[rank * expertPerRank + expert]);
        }
        maxExpertM = std::max(maxExpertM, expertM);
    }

    const auto *mmXShape = context->GetOptionalInputShape(optiling::NON_QUANT_MM_X_INDEX);
    const auto *mmWeightShape = context->GetOptionalInputShape(optiling::NON_QUANT_MM_WEIGHT_INDEX);
    const auto *mmYShape = context->GetOutputShape(optiling::OUTPUT_MM_Y_INDEX);
    const bool hasMmX = IsShapePresentAiv(mmXShape);
    const bool hasMmWeight = IsShapePresentAiv(mmWeightShape);
    const bool hasMmY = IsShapePresentAiv(mmYShape);
    OP_TILING_CHECK((hasMmX || hasMmWeight || hasMmY) && !(hasMmX && hasMmWeight && hasMmY),
                    OP_LOGE(nodeName, "mmX, mmWeight and mmY must be provided together."), return ge::GRAPH_FAILED);

    uint32_t mmM = 0U;
    uint32_t mmK = 0U;
    uint32_t mmN = 0U;
    if (hasMmX) {
        const auto *mmXDesc = context->GetOptionalInputDesc(optiling::NON_QUANT_MM_X_INDEX);
        const auto *mmWeightDesc = context->GetOptionalInputDesc(optiling::NON_QUANT_MM_WEIGHT_INDEX);
        const auto *mmYDesc = context->GetOutputDesc(optiling::OUTPUT_MM_Y_INDEX);
        OP_TILING_CHECK(mmXDesc == nullptr || mmWeightDesc == nullptr || mmYDesc == nullptr ||
                            mmXDesc->GetDataType() != ge::DT_BF16 ||
                            mmWeightDesc->GetDataType() != ge::DT_BF16 ||
                            mmYDesc->GetDataType() != ge::DT_BF16 ||
                            mmXDesc->GetStorageFormat() != ge::FORMAT_ND ||
                            mmWeightDesc->GetStorageFormat() != ge::FORMAT_ND ||
                            mmYDesc->GetStorageFormat() != ge::FORMAT_ND,
                        OP_LOGE(nodeName, "AIV shared MM supports BF16 ND tensors only."),
                        return ge::GRAPH_FAILED);
        const auto &mx = mmXShape->GetStorageShape();
        const auto &mw = mmWeightShape->GetStorageShape();
        const auto &my = mmYShape->GetStorageShape();
        OP_TILING_CHECK(mx.GetDimNum() != 2 || mw.GetDimNum() != 2 || my.GetDimNum() != 2,
                        OP_LOGE(nodeName, "AIV shared MM tensors must be 2D."), return ge::GRAPH_FAILED);
        const int64_t localMmM = mx.GetDim(0);
        const int64_t localMmK = mx.GetDim(1);
        const int64_t localWeightK = mw.GetDim(*transMmWeight ? 1 : 0);
        const int64_t localWeightN = mw.GetDim(*transMmWeight ? 0 : 1);
        OP_TILING_CHECK(!IsPositiveInt32(localMmM) || !IsPositiveInt32(localMmK) ||
                            !IsPositiveInt32(localWeightK) || !IsPositiveInt32(localWeightN) ||
                            localMmK != localWeightK || my.GetDim(0) != localMmM || my.GetDim(1) != localWeightN,
                        OP_LOGE(nodeName, "AIV shared MM dimensions are inconsistent."), return ge::GRAPH_FAILED);
        mmM = static_cast<uint32_t>(localMmM);
        mmK = static_cast<uint32_t>(localMmK);
        mmN = static_cast<uint32_t>(localWeightN);
    }

    const auto *permuteShape = context->GetOutputShape(optiling::OUTPUT_PERMUTE_OUT_INDEX);
    const bool hasPermute = IsShapePresentAiv(permuteShape);
    OP_TILING_CHECK(*permuteOutFlag != hasPermute,
                    OP_LOGE(nodeName, "permuteOutFlag must match permuteOut presence."), return ge::GRAPH_FAILED);
    if (hasPermute) {
        const auto *permuteDesc = context->GetOutputDesc(optiling::OUTPUT_PERMUTE_OUT_INDEX);
        const auto &shape = permuteShape->GetStorageShape();
        OP_TILING_CHECK(permuteDesc == nullptr || permuteDesc->GetDataType() != ge::DT_BF16 ||
                            permuteDesc->GetStorageFormat() != ge::FORMAT_ND || shape.GetDimNum() != 2 ||
                            shape.GetDim(0) != outputM || shape.GetDim(1) != inputK,
                        OP_LOGE(nodeName, "AIV permuteOut must be BF16 ND with shape [outputM, K]."),
                        return ge::GRAPH_FAILED);
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aicNum = platform.GetCoreNumAic();
    const uint32_t aivNum = platform.GetCoreNumAiv();
    uint64_t ubSize = 0U;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint32_t blockDim = platform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    OP_TILING_CHECK(blockDim == 0U, OP_LOGE(nodeName, "AIV mixed block dimension is zero."),
                    return ge::GRAPH_FAILED);
    context->SetBlockDim(blockDim);

    auto &gmmInfo = tilingData->gmmInfo;
    gmmInfo.M = static_cast<uint32_t>(maxExpertM);
    gmmInfo.K = static_cast<uint32_t>(inputK);
    gmmInfo.N = static_cast<uint32_t>(weightN);
    gmmInfo.rankSize = static_cast<uint32_t>(rankSize);
    gmmInfo.expertPerRank = static_cast<uint32_t>(expertPerRank);
    gmmInfo.maxOutputSize = static_cast<uint32_t>(outputM);
    gmmInfo.isTransposeB = static_cast<uint32_t>(*transGmmWeight);
    gmmInfo.hasSharedExpert = static_cast<uint32_t>(hasMmX);
    gmmInfo.hasPermuteOut = static_cast<uint32_t>(hasPermute);
    gmmInfo.aivCoreNum = aivNum;
    gmmInfo.aicCoreNum = aicNum;
    gmmInfo.totalUbSize = static_cast<uint32_t>(ubSize);

    if (hasMmX) {
        tilingData->mmInfo.M = mmM;
        tilingData->mmInfo.K = mmK;
        tilingData->mmInfo.N = mmN;
        tilingData->mmInfo.rankSize = static_cast<uint32_t>(rankSize);
        tilingData->mmInfo.expertPerRank = 1U;
        tilingData->mmInfo.maxOutputSize = mmM;
        tilingData->mmInfo.isTransposeB = static_cast<uint32_t>(*transMmWeight);
        tilingData->mmInfo.hasSharedExpert = 1U;
        tilingData->mmInfo.aivCoreNum = aivNum;
        tilingData->mmInfo.aicCoreNum = aicNum;
        tilingData->mmInfo.totalUbSize = static_cast<uint32_t>(ubSize);
    }

    SetCocTiling(gmmInfo.M, gmmInfo.K, gmmInfo.N, ubSize, tilingData->gmmCocTiling);
    if (hasMmX) {
        SetCocTiling(mmM, mmK, mmN, ubSize, tilingData->mmCocTiling);
    }
    tilingData->countNum = static_cast<uint32_t>(countNum);

    const uint64_t recvBytes =
        AlignWorkspace(static_cast<uint64_t>(outputM) * static_cast<uint64_t>(inputK) * BF16_BYTES);
    const uint64_t countBytes = AlignWorkspace(countNum * sizeof(AlltoAllvGmmExpertSourceMeta));
    const uint64_t metaBytes =
        AlignWorkspace(static_cast<uint64_t>(expertPerRank) * sizeof(AlltoAllvGmmExpertMeta));
    const uint64_t readyBytes = AlignWorkspace(AlltoAllvGroupedMatMulAiv::ExpertReadyBytes(static_cast<uint32_t>(expertPerRank)));
    tilingData->recvTokenOffset = 0U;
    tilingData->countTableOffset = recvBytes;
    tilingData->expertMetaOffset = recvBytes + countBytes;
    tilingData->expertReadyOffset = recvBytes + countBytes + metaBytes;
    tilingData->userWorkspaceSize = recvBytes + countBytes + metaBytes + readyBytes;

    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaceSizes == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "workspace"),
                    return ge::GRAPH_FAILED);
    workspaceSizes[0] = AIV_SYSTEM_WORKSPACE_SIZE + tilingData->userWorkspaceSize;

    const uint64_t tilingKey =
        GET_TPL_TILING_KEY(*transGmmWeight, hasMmX ? *transMmWeight : false, Mc2Comm::COMM_MODE_AIV);
    context->SetTilingKey(tilingKey);

    std::string algConfig = "AlltoAll=level0:fullmesh;level1:pairwise";
    AscendC::Mc2CcTilingConfig commConfig(group, AIV_ALLTOALL_OP_TYPE, algConfig);
    OP_TILING_CHECK(commConfig.SetCommEngine(Mc2Comm::ENGINE_MTE) != 0U,
                    OP_LOGE(nodeName, "Failed to select MTE communication engine."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commConfig.GetTiling(tilingData->hcclInitTiling) != 0U ||
                        commConfig.GetTiling(tilingData->hcclCcTiling) != 0U,
                    OP_LOGE(nodeName, "Failed to generate AIV HCCL tiling."), return ge::GRAPH_FAILED);

    OP_LOGI(nodeName,
            "AIV tiling: rank=%u, expert=%u, maxExpertM=%u, K=%u, N=%u, blockDim=%u, workspace=%lu.",
            gmmInfo.rankSize, gmmInfo.expertPerRank, gmmInfo.M, gmmInfo.K, gmmInfo.N, blockDim,
            tilingData->userWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}
} // namespace

namespace optiling {

std::vector<int64_t> AlltoAllvGmmTilingA3::GetEpWorldSizeOptional() const
{
    return {8, 16, 32, 64, 128};
}

bool AlltoAllvGmmTilingA3::NeedToCheckCounts() const
{
    return true;
}

ge::graphStatus AlltoAllvGmmTilingFuncA3::AlltoAllvGmmOpTilingFunc(gert::TilingContext *context)
{
    AlltoAllvGmmTilingA3 tiling(context);
    OP_TILING_CHECK(tiling.Init(context) != ge::GRAPH_SUCCESS, OP_LOGE(A_INNER_DEBUG, "GMM tiling init failed."),
                    return ge::GRAPH_FAILED);
    return tiling.RunFusionKernelTiling(context);
}

ge::graphStatus AlltoAllvGmmTilingStructA3::DoOpTiling()
{
    if (IsAivMode(context_)) {
        return FillAivTiling(context_);
    }
    AlltoAllvGmmTilingFuncA3 funcA3;
    return funcA3.AlltoAllvGmmOpTilingFunc(context_);
}

ge::graphStatus AlltoAllvGmmTilingStructA3::GetShapeAttrsInfo()
{
    if (IsAivMode(context_)) {
        return ge::GRAPH_SUCCESS;
    }
    return AlltoAllvGmmTilingBase::GetShapeAttrsInfo();
}

ge::graphStatus AlltoAllvGmmTilingStructA3::DoLibApiTiling()
{
    if (IsAivMode(context_)) {
        return ge::GRAPH_SUCCESS;
    }
    return AlltoAllvGmmTilingBase::DoLibApiTiling();
}

ge::graphStatus AlltoAllvGmmTilingStructA3::GetWorkspaceSize()
{
    if (IsAivMode(context_)) {
        return ge::GRAPH_SUCCESS;
    }
    return AlltoAllvGmmTilingBase::GetWorkspaceSize();
}

ge::graphStatus AlltoAllvGmmTilingStructA3::PostTiling()
{
    if (IsAivMode(context_)) {
        return ge::GRAPH_SUCCESS;
    }
    return AlltoAllvGmmTilingBase::PostTiling();
}

REGISTER_OPS_TILING_TEMPLATE(AlltoAllvGroupedMatMul, AlltoAllvGmmTilingStructA3, 0);
} // namespace optiling
