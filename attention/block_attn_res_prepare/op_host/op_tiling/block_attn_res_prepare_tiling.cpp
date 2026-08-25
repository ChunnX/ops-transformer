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
 * \file block_attn_res_prepare_tiling.cpp
 * \brief Common and Vector tiling for BlockAttnResPrepare.
 */

#include "block_attn_res_prepare_tiling.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

#include "log/log.h"
#include "securec.h"

namespace optiling {
namespace {

constexpr const char *OP_NAME = "BlockAttnResPrepare";
constexpr size_t BLOCK_RES_INDEX = 0;
constexpr size_t VALID_BLOCKS_INDEX = 1;
constexpr size_t PSEUDO_QUERY_INDEX = 2;
constexpr size_t EPS_ATTR_INDEX = 0;
constexpr size_t BLOCK_RES_RANK = 3;
constexpr size_t PSEUDO_QUERY_RANK = 2;
constexpr size_t T_DIM_INDEX = 0;
constexpr size_t N_DIM_INDEX = 1;
constexpr size_t D_DIM_INDEX = 2;
constexpr size_t S_DIM_INDEX = 0;
constexpr size_t PSEUDO_QUERY_D_DIM_INDEX = 1;

constexpr uint64_t FP32_BYTES = sizeof(float);
constexpr uint64_t UB_RESERVED_BYTES = 8UL * 1024UL;
constexpr uint32_t FP32_REG_ELEMS = 64U;
constexpr uint32_t FP32_32B_ALIGN_ELEMS = 8U;
constexpr uint64_t MAX_HEAD_DIM = 8192UL;
constexpr uint32_t MAX_BLOCK_NUM = FP32_REG_ELEMS;
constexpr uint32_t STAT_VECTOR_COUNT = 2U;
constexpr uint32_t STAT_BUFFER_NUM = 2U;
constexpr uint32_t SCALAR_BLOCK_ELEMS = 8U;
constexpr uint32_t STAT_UB_ELEMS = STAT_VECTOR_COUNT * (FP32_REG_ELEMS + SCALAR_BLOCK_ELEMS);
constexpr uint32_t VECTOR_SCHEDULE_MODE = 1U;
constexpr uint64_t DOUBLE_BUFFER_NUM = 2UL;
constexpr uint64_t VECTOR_TILING_PRIORITY = 1000UL;
constexpr uint64_t VECTOR_BASE_D_LARGE = 1024UL;
constexpr uint64_t VECTOR_BASE_D_MEDIUM = 512UL;
constexpr uint64_t VECTOR_BASE_D_SMALL = 256UL;
constexpr size_t VECTOR_BASE_D_CANDIDATE_NUM = 4;
constexpr uint64_t MAX_ALIGNED_D =
    static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) - (FP32_32B_ALIGN_ELEMS - 1U);

template <typename T>
constexpr T CeilDiv(T value, T factor)
{
    return (value + factor - 1) / factor;
}

template <typename T>
constexpr T AlignUp(T value, T factor)
{
    return CeilDiv(value, factor) * factor;
}

bool IsPositiveShape(const gert::Shape &shape)
{
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) <= 0) {
            return false;
        }
    }
    return true;
}

} // namespace

bool BlockAttnResPrepareTiling::IsCapable()
{
    // The vector template is the fallback.  Cube has its own registered template with a
    // higher priority and is considered before this class; returning true here keeps the
    // fallback available when the Cube resource/capability checks reject a shape.
    return true;
}

ge::graphStatus BlockAttnResPrepareTilingBase::GetPlatformInfo()
{
    auto *platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        const auto *compileInfo = reinterpret_cast<const BlockAttnResPrepareCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"),
                    return ge::GRAPH_FAILED);
        aicCoreNum_ = compileInfo->aicCoreNum;
        aivCoreNum_ = compileInfo->aivCoreNum;
        ubSize_ = compileInfo->ubSize;
        l1Size_ = compileInfo->l1Size;
        l0ASize_ = compileInfo->l0ASize;
        l0BSize_ = compileInfo->l0BSize;
        l0CSize_ = compileInfo->l0CSize;
        systemWorkspaceSize_ = compileInfo->systemWorkspaceSize;
    } else {
        const auto platform = platform_ascendc::PlatformAscendC(platformInfo);
        aicCoreNum_ = platform.GetCoreNumAic();
        aivCoreNum_ = platform.GetCoreNumAiv();
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0ASize_);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0BSize_);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize_);
        systemWorkspaceSize_ = platform.GetLibApiWorkSpaceSize();
    }
    OP_CHECK_IF(aivCoreNum_ == 0, OP_LOGE(context_->GetNodeName(), "AIV core number is zero"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize_ <= UB_RESERVED_BYTES, OP_LOGE(context_->GetNodeName(), "UB size is too small: %lu", ubSize_),
                return ge::GRAPH_FAILED);
    workspaceSize_ = systemWorkspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResPrepareTilingBase::CheckInputShapes()
{
    const auto *blockResStorageShape = context_->GetInputShape(BLOCK_RES_INDEX);
    const auto *validStorageShape = context_->GetInputShape(VALID_BLOCKS_INDEX);
    const auto *pseudoQueryStorageShape = context_->GetInputShape(PSEUDO_QUERY_INDEX);
    OP_CHECK_IF(blockResStorageShape == nullptr || validStorageShape == nullptr || pseudoQueryStorageShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "input shape is null"), return ge::GRAPH_FAILED);

    const gert::Shape &blockResShape = blockResStorageShape->GetOriginShape();
    const gert::Shape &validShape = validStorageShape->GetOriginShape();
    const gert::Shape &pseudoQueryShape = pseudoQueryStorageShape->GetOriginShape();
    OP_CHECK_IF(blockResShape.GetDimNum() != BLOCK_RES_RANK,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "block_res",
                                              (std::to_string(blockResShape.GetDimNum()) + "D").c_str(), "3D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(pseudoQueryShape.GetDimNum() != PSEUDO_QUERY_RANK,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "pseudo_query",
                                              (std::to_string(pseudoQueryShape.GetDimNum()) + "D").c_str(), "2D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(validShape.GetDimNum() != 1 || validShape.GetDim(0) != 1,
                OP_LOGE_FOR_INVALID_SHAPE(context_->GetNodeName(), "valid_blocks",
                                          Ops::Base::ToString(validShape).c_str(), "1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsPositiveShape(blockResShape) || !IsPositiveShape(pseudoQueryShape),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->GetNodeName(), "block_res, pseudo_query",
                    (Ops::Base::ToString(blockResShape) + ", " + Ops::Base::ToString(pseudoQueryShape)).c_str(),
                    "all dimensions of block_res and pseudo_query must be positive"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockResShape.GetDim(D_DIM_INDEX) != pseudoQueryShape.GetDim(PSEUDO_QUERY_D_DIM_INDEX),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context_->GetNodeName(), "block_res.shape[2], pseudo_query.shape[1]",
                    (std::to_string(blockResShape.GetDim(D_DIM_INDEX)) + ", " +
                     std::to_string(pseudoQueryShape.GetDim(PSEUDO_QUERY_D_DIM_INDEX)))
                        .c_str(),
                    "block_res.shape[2] must equal pseudo_query.shape[1]"),
                return ge::GRAPH_FAILED);

    totalT_ = static_cast<uint64_t>(blockResShape.GetDim(T_DIM_INDEX));
    totalN_ = static_cast<uint64_t>(blockResShape.GetDim(N_DIM_INDEX));
    totalD_ = static_cast<uint64_t>(blockResShape.GetDim(D_DIM_INDEX));
    totalS_ = static_cast<uint64_t>(pseudoQueryShape.GetDim(S_DIM_INDEX));
    OP_CHECK_IF(totalT_ > std::numeric_limits<uint32_t>::max() || totalN_ > std::numeric_limits<uint32_t>::max() ||
                    totalS_ > std::numeric_limits<uint32_t>::max(),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context_->GetNodeName(), "block_res.shape[0], block_res.shape[1], pseudo_query.shape[0]",
                    (std::to_string(totalT_) + ", " + std::to_string(totalN_) + ", " + std::to_string(totalS_)).c_str(),
                    "the dimensions T, N and S must be within the uint32 range"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(totalN_ > MAX_BLOCK_NUM,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "block_res.shape[1]",
                                                       std::to_string(totalN_).c_str(),
                                                       "block_res.shape[1] must be less than or equal to 64"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(totalD_ > MAX_HEAD_DIM,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "block_res.shape[2]",
                                                       std::to_string(totalD_).c_str(),
                                                       "block_res.shape[2] must be less than or equal to 8192"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(totalD_ > MAX_ALIGNED_D,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "block_res.shape[2]", std::to_string(totalD_).c_str(),
                    "block_res.shape[2] must support uint32-aligned tiling without overflow"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(totalT_ > std::numeric_limits<uint64_t>::max() / totalS_,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context_->GetNodeName(), "block_res.shape[0], pseudo_query.shape[0]",
                    (std::to_string(totalT_) + ", " + std::to_string(totalS_)).c_str(),
                    "the product of block_res.shape[0] and pseudo_query.shape[0] must be within the uint64 range"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResPrepareTilingBase::CheckInputDtypes() const
{
    const auto *blockResDesc = context_->GetInputDesc(BLOCK_RES_INDEX);
    const auto *validDesc = context_->GetInputDesc(VALID_BLOCKS_INDEX);
    const auto *pseudoQueryDesc = context_->GetInputDesc(PSEUDO_QUERY_INDEX);
    OP_CHECK_IF(blockResDesc == nullptr || validDesc == nullptr || pseudoQueryDesc == nullptr,
                OP_LOGE(context_->GetNodeName(), "input descriptor is null"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockResDesc->GetDataType() != ge::DT_FLOAT || pseudoQueryDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context_->GetNodeName(), "block_res, pseudo_query",
                    (ge::TypeUtils::DataTypeToSerialString(blockResDesc->GetDataType()) + ", " +
                     ge::TypeUtils::DataTypeToSerialString(pseudoQueryDesc->GetDataType()))
                        .c_str(),
                    "the dtypes of block_res and pseudo_query must both be FLOAT"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(validDesc->GetDataType() != ge::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "valid_blocks",
                                          ge::TypeUtils::DataTypeToSerialString(validDesc->GetDataType()).c_str(),
                                          "INT64"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResPrepareTilingBase::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE(OP_NAME, "tiling context is null"), return ge::GRAPH_FAILED);
    if (CheckInputShapes() != ge::GRAPH_SUCCESS || CheckInputDtypes() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const auto *attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is null"), return ge::GRAPH_FAILED);
    const float *eps = attrs->GetFloat(EPS_ATTR_INDEX);
    OP_CHECK_IF(eps == nullptr || !std::isfinite(*eps) || *eps <= 0.0F,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "eps", eps == nullptr ? "nullptr" : std::to_string(*eps).c_str(),
                    "eps must exist, be finite and be greater than zero"),
                return ge::GRAPH_FAILED);
    eps_ = *eps;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResPrepareTilingBase::CalculateWorkDistribution(uint64_t totalWorkUnits, uint64_t maxCoreNum,
                                                                         const char *templateName,
                                                                         WorkDistribution &distribution) const
{
    OP_CHECK_IF(totalWorkUnits > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(context_->GetNodeName(), "%s work units exceed uint32 tiling range", templateName),
                return ge::GRAPH_FAILED);

    const uint64_t usedCoreNum = std::min(maxCoreNum, totalWorkUnits);
    OP_CHECK_IF(usedCoreNum == 0U, OP_LOGE(context_->GetNodeName(), "%s used core number is zero", templateName),
                return ge::GRAPH_FAILED);
    const uint64_t workUnitsPerCore = totalWorkUnits / usedCoreNum;
    OP_CHECK_IF(workUnitsPerCore > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(context_->GetNodeName(), "%s work units per core exceed uint32 range", templateName),
                return ge::GRAPH_FAILED);

    distribution.usedCoreNum = static_cast<uint32_t>(usedCoreNum);
    distribution.blockFactor = static_cast<uint32_t>(workUnitsPerCore);
    distribution.bigCoreNum = static_cast<uint32_t>(totalWorkUnits % usedCoreNum);
    distribution.tailBlockFactor = distribution.blockFactor + ((distribution.bigCoreNum > 0U) ? 1U : 0U);
    return ge::GRAPH_SUCCESS;
}

bool BlockAttnResPrepareTiling::SelectVectorBaseD(bool hasMultipleWorkRounds)
{
    const uint64_t usableUbBytes = ubSize_ - UB_RESERVED_BYTES;
    const uint64_t dAlign = AlignUp<uint64_t>(totalD_, FP32_32B_ALIGN_ELEMS);

    const std::array<uint64_t, VECTOR_BASE_D_CANDIDATE_NUM> candidates = {
        totalD_, std::min<uint64_t>(totalD_, VECTOR_BASE_D_LARGE), std::min<uint64_t>(totalD_, VECTOR_BASE_D_MEDIUM),
        std::min<uint64_t>(totalD_, VECTOR_BASE_D_SMALL)};
    uint64_t previous = 0;
    for (const uint64_t candidate : candidates) {
        if (candidate == 0 || candidate == previous) {
            continue;
        }
        previous = candidate;
        // Vector UB rows only require the hardware's 32B address granularity.
        // VF keeps its 64-element register width internally and masks tail
        // elements during arithmetic; it must not inflate every UB row pitch.
        const uint64_t baseDAlign = AlignUp<uint64_t>(candidate, FP32_32B_ALIGN_ELEMS);
        const uint64_t dTileNum = CeilDiv(totalD_, candidate);
        const uint64_t qBufferNum = (dTileNum > 1 || hasMultipleWorkRounds) ? DOUBLE_BUFFER_NUM : 1UL;
        const uint64_t oBufferNum = (dTileNum > 1) ? DOUBLE_BUFFER_NUM : 1UL;
        OP_CHECK_IF(dTileNum > std::numeric_limits<uint64_t>::max() / totalN_,
                    OP_LOGE(context_->GetNodeName(), "D tile count times N overflows uint64"), return false);
        const uint64_t vLoopNum = dTileNum * totalN_;
        const uint64_t vBufferNum = (vLoopNum > 1) ? DOUBLE_BUFFER_NUM : 1UL;
        const uint64_t fixedElems = (qBufferNum + vBufferNum + oBufferNum) * baseDAlign +
                                    static_cast<uint64_t>(STAT_BUFFER_NUM) * STAT_UB_ELEMS;
        const uint64_t fixedBytes = fixedElems * FP32_BYTES;
        if (fixedBytes > usableUbBytes) {
            continue;
        }

        baseD_ = static_cast<uint32_t>(candidate);
        qBufferNum_ = static_cast<uint32_t>(qBufferNum);
        vBufferNum_ = static_cast<uint32_t>(vBufferNum);
        oBufferNum_ = static_cast<uint32_t>(oBufferNum);

        const uint64_t cacheRowBytes = dAlign * FP32_BYTES;
        const uint64_t cacheCapacity = (usableUbBytes - fixedBytes) / cacheRowBytes;
        vCacheRows_ = static_cast<uint32_t>(std::min<uint64_t>(totalN_, cacheCapacity));
        return true;
    }
    return false;
}

ge::graphStatus BlockAttnResPrepareTiling::DoOpTiling()
{
    const uint64_t totalWorkUnits = totalT_ * totalS_;
    WorkDistribution distribution;
    if (CalculateWorkDistribution(totalWorkUnits, aivCoreNum_, "Vector", distribution) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    usedCoreNum_ = distribution.usedCoreNum;
    const bool hasMultipleWorkRounds = distribution.tailBlockFactor > 1U;

    OP_CHECK_IF(!SelectVectorBaseD(hasMultipleWorkRounds),
                OP_LOGE(context_->GetNodeName(), "no Vector baseD satisfies UB capacity"), return ge::GRAPH_FAILED);

    tilingData_.totalT = static_cast<uint32_t>(totalT_);
    tilingData_.totalN = static_cast<uint32_t>(totalN_);
    tilingData_.totalS = static_cast<uint32_t>(totalS_);
    tilingData_.totalWorkUnits = static_cast<uint32_t>(totalWorkUnits);
    tilingData_.totalD = totalD_;
    tilingData_.usedCoreNum = usedCoreNum_;
    tilingData_.bigCoreNum = distribution.bigCoreNum;
    tilingData_.blockFactor = distribution.blockFactor;
    tilingData_.tailBlockFactor = distribution.tailBlockFactor;
    tilingData_.baseD = baseD_;
    tilingData_.vCacheRows = static_cast<uint8_t>(vCacheRows_);
    tilingData_.qBufferNum = static_cast<uint8_t>(qBufferNum_);
    tilingData_.vBufferNum = static_cast<uint8_t>(vBufferNum_);
    tilingData_.oBufferNum = static_cast<uint8_t>(oBufferNum_);
    tilingData_.statUbElems = STAT_UB_ELEMS;
    tilingData_.workUbElems = 0U;
    tilingData_.eps = eps_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResPrepareTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t BlockAttnResPrepareTiling::GetTilingKey() const
{
    return BLOCK_ATTN_RES_PREPARE_VECTOR_KEY;
}

ge::graphStatus BlockAttnResPrepareTilingBase::GetWorkspaceSize()
{
    size_t *workspaceSizes = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaceSizes == nullptr, OP_LOGE(context_->GetNodeName(), "workspace size address is null"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(workspaceSize_ > std::numeric_limits<size_t>::max(),
                OP_LOGE(context_->GetNodeName(), "workspace exceeds size_t range"), return ge::GRAPH_FAILED);
    workspaceSizes[0] = static_cast<size_t>(workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResPrepareTilingBase::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    context_->SetTilingKey(GetTilingKey());
    context_->SetScheduleMode(VECTOR_SCHEDULE_MODE);

    const TilingDataView tilingDataView = GetTilingDataView();
    const char *templateName = tilingDataView.templateName == nullptr ? "unknown" : tilingDataView.templateName;
    OP_CHECK_IF(tilingDataView.data == nullptr || tilingDataView.size == 0U,
                OP_LOGE(context_->GetNodeName(), "%s tiling data is empty", templateName), return ge::GRAPH_FAILED);
    auto *rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(rawTilingData == nullptr, OP_LOGE(context_->GetNodeName(), "raw tiling data is null"),
                return ge::GRAPH_FAILED);
    auto *rawData = rawTilingData->GetData();
    OP_CHECK_IF(rawData == nullptr, OP_LOGE(context_->GetNodeName(), "raw tiling data buffer is null"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(rawTilingData->GetCapacity() < tilingDataView.size,
                OP_LOGE(context_->GetNodeName(), "raw %s tiling data capacity is insufficient", templateName),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(memcpy_s(rawData, rawTilingData->GetCapacity(), tilingDataView.data, tilingDataView.size) != EOK,
                OP_LOGE(context_->GetNodeName(), "copy %s tiling data failed", templateName), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(tilingDataView.size);
    return ge::GRAPH_SUCCESS;
}

BlockAttnResPrepareTilingBase::TilingDataView BlockAttnResPrepareTiling::GetTilingDataView() const
{
    return {&tilingData_, sizeof(tilingData_), "Vector"};
}

void BlockAttnResPrepareTiling::DumpTilingInfo()
{
    OP_LOGI(context_->GetNodeName(),
            "BlockAttnResPrepare tiling: T=%lu N=%lu S=%lu D=%lu cores=%u baseD=%u "
            "q/v/o buffers=%u/%u/%u vCacheRows=%u",
            totalT_, totalN_, totalS_, totalD_, usedCoreNum_, baseD_, qBufferNum_, vBufferNum_, oBufferNum_,
            vCacheRows_);
}

REGISTER_OPS_TILING_TEMPLATE(BlockAttnResPrepare, BlockAttnResPrepareCubeTiling, 0);
REGISTER_OPS_TILING_TEMPLATE(BlockAttnResPrepare, BlockAttnResPrepareTiling, VECTOR_TILING_PRIORITY);

ge::graphStatus TilingBlockAttnResPrepare(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(OP_NAME, "tiling context is null"), return ge::GRAPH_FAILED);
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareBlockAttnResPrepare(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(OP_NAME, "tiling parse context is null"), return ge::GRAPH_FAILED);
    auto *compileInfo = context->GetCompiledInfo<BlockAttnResPrepareCompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(context->GetNodeName(), "compiled info is null"),
                return ge::GRAPH_FAILED);
    auto *platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context->GetNodeName(), "platform info is null"),
                return ge::GRAPH_FAILED);
    const auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->aicCoreNum = platform.GetCoreNumAic();
    compileInfo->aivCoreNum = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfo->l1Size);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfo->l0ASize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfo->l0BSize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfo->l0CSize);
    compileInfo->systemWorkspaceSize = platform.GetLibApiWorkSpaceSize();
    OP_CHECK_IF(compileInfo->aicCoreNum == 0 || compileInfo->aivCoreNum == 0 || compileInfo->ubSize == 0 ||
                    compileInfo->l1Size == 0 || compileInfo->l0ASize == 0 || compileInfo->l0BSize == 0 ||
                    compileInfo->l0CSize == 0,
                OP_LOGE(context->GetNodeName(), "failed to query Ascend 950 core memory information"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BlockAttnResPrepare)
    .Tiling(TilingBlockAttnResPrepare)
    .TilingParse<BlockAttnResPrepareCompileInfo>(TilingPrepareBlockAttnResPrepare);

} // namespace optiling
