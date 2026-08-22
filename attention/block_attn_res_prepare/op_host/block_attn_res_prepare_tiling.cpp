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
 * \brief Ascend 950 tiling for BlockAttnResPrepare.
 */

#include "block_attn_res_prepare_tiling.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

#include "log/log.h"
#include "securec.h"

namespace optiling {
namespace {

constexpr const char *OP_NAME = "BlockAttnResPrepare";
constexpr size_t BLOCK_RES_INDEX = 0;
constexpr size_t VALID_BLOCKS_INDEX = 1;
constexpr size_t PSEUDO_QUERY_INDEX = 2;
constexpr size_t EPS_ATTR_INDEX = 0;

constexpr uint64_t FP32_BYTES = sizeof(float);
constexpr uint64_t UB_RESERVED_BYTES = 8UL * 1024UL;
constexpr uint64_t L1_RESERVED_BYTES = 8UL * 1024UL;
constexpr uint32_t FP32_REG_ELEMS = 64U;
constexpr uint32_t FP32_32B_ALIGN_ELEMS = 8U;
constexpr uint32_t CUBE_BLOCK = 16U;
constexpr uint32_t CUBE_MAX_BASE_S = 32U;
constexpr uint32_t CUBE_MIN_T = 32U;
constexpr uint32_t CUBE_MIN_S = 16U;
constexpr uint32_t CUBE_MIN_D = 256U;
constexpr uint64_t MAX_HEAD_DIM = 8192UL;
constexpr uint32_t MAX_BLOCK_NUM = FP32_REG_ELEMS;
constexpr uint32_t STAT_VECTOR_COUNT = 2U;
constexpr uint32_t SCALAR_BLOCK_ELEMS = 8U;
constexpr uint32_t STAT_UB_ELEMS = STAT_VECTOR_COUNT * FP32_REG_ELEMS + 2U * SCALAR_BLOCK_ELEMS;
constexpr uint32_t VECTOR_SCHEDULE_MODE = 1U;
constexpr uint64_t MAX_ALIGNED_D = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) -
                                   (FP32_32B_ALIGN_ELEMS - 1U);

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

ge::graphStatus BlockAttnResPrepareTiling::GetPlatformInfo()
{
    auto *platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        const auto *compileInfo =
            reinterpret_cast<const BlockAttnResPrepareCompileInfo *>(context_->GetCompileInfo());
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
    OP_CHECK_IF(aivCoreNum_ == 0, OP_LOGE(context_->GetNodeName(), "AIV core number is zero"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize_ <= UB_RESERVED_BYTES, OP_LOGE(context_->GetNodeName(), "UB size is too small: %lu", ubSize_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResPrepareTiling::CheckInputShapes()
{
    const auto *blockResStorageShape = context_->GetInputShape(BLOCK_RES_INDEX);
    const auto *validStorageShape = context_->GetInputShape(VALID_BLOCKS_INDEX);
    const auto *pseudoQueryStorageShape = context_->GetInputShape(PSEUDO_QUERY_INDEX);
    OP_CHECK_IF(blockResStorageShape == nullptr || validStorageShape == nullptr || pseudoQueryStorageShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "input shape is null"), return ge::GRAPH_FAILED);

    const gert::Shape &blockResShape = blockResStorageShape->GetOriginShape();
    const gert::Shape &validShape = validStorageShape->GetOriginShape();
    const gert::Shape &pseudoQueryShape = pseudoQueryStorageShape->GetOriginShape();
    OP_CHECK_IF(blockResShape.GetDimNum() != 3, OP_LOGE(context_->GetNodeName(), "block_res must be [T,N,D]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(pseudoQueryShape.GetDimNum() != 2, OP_LOGE(context_->GetNodeName(), "pseudo_query must be [S,D]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(validShape.GetDimNum() != 1 || validShape.GetDim(0) != 1,
                OP_LOGE(context_->GetNodeName(), "valid_blocks must have shape [1]"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsPositiveShape(blockResShape) || !IsPositiveShape(pseudoQueryShape),
                OP_LOGE(context_->GetNodeName(), "all input dimensions must be positive"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockResShape.GetDim(2) != pseudoQueryShape.GetDim(1),
                OP_LOGE(context_->GetNodeName(), "block_res.D and pseudo_query.D must be equal"),
                return ge::GRAPH_FAILED);

    totalT_ = static_cast<uint64_t>(blockResShape.GetDim(0));
    totalN_ = static_cast<uint64_t>(blockResShape.GetDim(1));
    totalD_ = static_cast<uint64_t>(blockResShape.GetDim(2));
    totalS_ = static_cast<uint64_t>(pseudoQueryShape.GetDim(0));
    OP_CHECK_IF(totalT_ > std::numeric_limits<uint32_t>::max() || totalN_ > std::numeric_limits<uint32_t>::max() ||
                    totalS_ > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(context_->GetNodeName(), "T/N/S exceed uint32 tiling range"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(totalN_ > MAX_BLOCK_NUM,
                OP_LOGE(context_->GetNodeName(), "N=%lu exceeds the current single-VReg softmax limit %u", totalN_,
                        MAX_BLOCK_NUM),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(totalD_ > MAX_HEAD_DIM,
                OP_LOGE(context_->GetNodeName(), "D=%lu exceeds the interface limit %lu", totalD_, MAX_HEAD_DIM),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(totalD_ > MAX_ALIGNED_D,
                OP_LOGE(context_->GetNodeName(), "D=%lu is too large for uint32-aligned tiling", totalD_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(totalT_ > std::numeric_limits<uint64_t>::max() / totalS_,
                OP_LOGE(context_->GetNodeName(), "T*S overflows uint64"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResPrepareTiling::CheckInputDtypes() const
{
    const auto *blockResDesc = context_->GetInputDesc(BLOCK_RES_INDEX);
    const auto *validDesc = context_->GetInputDesc(VALID_BLOCKS_INDEX);
    const auto *pseudoQueryDesc = context_->GetInputDesc(PSEUDO_QUERY_INDEX);
    OP_CHECK_IF(blockResDesc == nullptr || validDesc == nullptr || pseudoQueryDesc == nullptr,
                OP_LOGE(context_->GetNodeName(), "input descriptor is null"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockResDesc->GetDataType() != ge::DT_FLOAT || pseudoQueryDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE(context_->GetNodeName(), "block_res and pseudo_query must be FP32"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(validDesc->GetDataType() != ge::DT_INT64,
                OP_LOGE(context_->GetNodeName(), "valid_blocks must be INT64"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResPrepareTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE(OP_NAME, "tiling context is null"), return ge::GRAPH_FAILED);
    if (CheckInputShapes() != ge::GRAPH_SUCCESS || CheckInputDtypes() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const auto *attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs is null"), return ge::GRAPH_FAILED);
    const float *eps = attrs->GetFloat(EPS_ATTR_INDEX);
    OP_CHECK_IF(eps == nullptr || !std::isfinite(*eps) || *eps <= 0.0F,
                OP_LOGE(context_->GetNodeName(), "eps must be finite and greater than zero"),
                return ge::GRAPH_FAILED);
    eps_ = *eps;
    return ge::GRAPH_SUCCESS;
}

bool BlockAttnResPrepareTiling::SelectVectorBaseD(bool hasMultipleWorkRounds)
{
    const uint64_t usableUbBytes = ubSize_ - UB_RESERVED_BYTES;
    const uint64_t dAlign = AlignUp<uint64_t>(totalD_, FP32_32B_ALIGN_ELEMS);

    const std::array<uint64_t, 4> candidates = {totalD_, std::min<uint64_t>(totalD_, 1024UL),
                                                std::min<uint64_t>(totalD_, 512UL),
                                                std::min<uint64_t>(totalD_, 256UL)};
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
        const uint64_t qBufferNum = (dTileNum > 1 || hasMultipleWorkRounds) ? 2UL : 1UL;
        const uint64_t oBufferNum = (dTileNum > 1) ? 2UL : 1UL;
        OP_CHECK_IF(dTileNum > std::numeric_limits<uint64_t>::max() / totalN_,
                    OP_LOGE(context_->GetNodeName(), "D tile count times N overflows uint64"), return false);
        const uint64_t vLoopNum = dTileNum * totalN_;
        const uint64_t vBufferNum = (vLoopNum > 1) ? 2UL : 1UL;
        const uint64_t fixedElems =
            (qBufferNum + vBufferNum + oBufferNum) * baseDAlign + static_cast<uint64_t>(STAT_UB_ELEMS);
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
    OP_CHECK_IF(totalWorkUnits > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(context_->GetNodeName(), "Vector work units exceed uint32 tiling range"),
                return ge::GRAPH_FAILED);
    usedCoreNum_ = static_cast<uint32_t>(std::min<uint64_t>(aivCoreNum_, totalWorkUnits));
    OP_CHECK_IF(usedCoreNum_ == 0, OP_LOGE(context_->GetNodeName(), "used core number is zero"),
                return ge::GRAPH_FAILED);
    const uint64_t workUnitsPerCore = totalWorkUnits / usedCoreNum_;
    OP_CHECK_IF(workUnitsPerCore > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(context_->GetNodeName(), "work units per core exceed uint32 range"),
                return ge::GRAPH_FAILED);

    const uint32_t blockFactor = static_cast<uint32_t>(workUnitsPerCore);
    const uint32_t bigCoreNum = static_cast<uint32_t>(totalWorkUnits % usedCoreNum_);
    const uint32_t tailBlockFactor = blockFactor + ((bigCoreNum > 0) ? 1U : 0U);
    const bool hasMultipleWorkRounds = tailBlockFactor > 1U;

    OP_CHECK_IF(!SelectVectorBaseD(hasMultipleWorkRounds),
                OP_LOGE(context_->GetNodeName(), "no Vector baseD satisfies UB capacity"),
                return ge::GRAPH_FAILED);

    tilingData_.totalT = static_cast<uint32_t>(totalT_);
    tilingData_.totalN = static_cast<uint32_t>(totalN_);
    tilingData_.totalS = static_cast<uint32_t>(totalS_);
    tilingData_.totalWorkUnits = static_cast<uint32_t>(totalWorkUnits);
    tilingData_.totalD = totalD_;
    tilingData_.usedCoreNum = usedCoreNum_;
    tilingData_.bigCoreNum = bigCoreNum;
    tilingData_.blockFactor = blockFactor;
    tilingData_.tailBlockFactor = tailBlockFactor;
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

ge::graphStatus BlockAttnResPrepareTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t BlockAttnResPrepareTiling::GetTilingKey() const
{
    return BLOCK_ATTN_RES_PREPARE_VECTOR_KEY;
}

ge::graphStatus BlockAttnResPrepareTiling::GetWorkspaceSize()
{
    size_t *workspaceSizes = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaceSizes == nullptr, OP_LOGE(context_->GetNodeName(), "workspace size address is null"),
                return ge::GRAPH_FAILED);
    workspaceSizes[0] = static_cast<size_t>(systemWorkspaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResPrepareTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    context_->SetTilingKey(GetTilingKey());
    context_->SetScheduleMode(VECTOR_SCHEDULE_MODE);
    auto *rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(rawTilingData == nullptr, OP_LOGE(context_->GetNodeName(), "raw tiling data is null"),
                return ge::GRAPH_FAILED);
    auto *rawData = rawTilingData->GetData();
    OP_CHECK_IF(rawData == nullptr, OP_LOGE(context_->GetNodeName(), "raw tiling data buffer is null"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(rawTilingData->GetCapacity() < sizeof(BlockAttnResPrepareTilingData),
                OP_LOGE(context_->GetNodeName(), "raw tiling data capacity is insufficient"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(memcpy_s(rawData, rawTilingData->GetCapacity(), &tilingData_, sizeof(tilingData_)) != EOK,
                OP_LOGE(context_->GetNodeName(), "copy vector tiling data failed"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(BlockAttnResPrepareTilingData));
    return ge::GRAPH_SUCCESS;
}

void BlockAttnResPrepareTiling::DumpTilingInfo()
{
    OP_LOGI(context_->GetNodeName(),
            "BlockAttnResPrepare tiling: T=%lu N=%lu S=%lu D=%lu cores=%u baseD=%u "
            "q/v/o buffers=%u/%u/%u vCacheRows=%u",
            totalT_, totalN_, totalS_, totalD_, usedCoreNum_, baseD_, qBufferNum_, vBufferNum_, oBufferNum_,
            vCacheRows_);
}

bool BlockAttnResPrepareCubeTiling::IsCapable()
{
    if (totalT_ < CUBE_MIN_T || totalS_ < CUBE_MIN_S || totalD_ < CUBE_MIN_D || aicCoreNum_ == 0U ||
        aivCoreNum_ < 2U || l1Size_ <= L1_RESERVED_BYTES || l0ASize_ == 0U || l0BSize_ == 0U || l0CSize_ == 0U) {
        return false;
    }
    // MM2 Fixpipe writes a strided [S, D] tile directly into O[S, T, D]. The destination row stride is T * D
    // elements and is encoded by the hardware interface as uint32_t.
    if (totalT_ > std::numeric_limits<uint32_t>::max() / totalD_) {
        return false;
    }
    return SelectCubeTileShape();
}

bool BlockAttnResPrepareCubeTiling::SelectCubeTileShape()
{
    const uint64_t usableUbBytes = ubSize_ - UB_RESERVED_BYTES;
    const uint64_t usableL1Bytes = l1Size_ - L1_RESERVED_BYTES;
    nAlign_ = static_cast<uint32_t>(AlignUp<uint64_t>(totalN_, CUBE_BLOCK));
    dAlign_ = static_cast<uint32_t>(AlignUp<uint64_t>(totalD_, CUBE_BLOCK));

    // Pack adjacent token matrices into MM1's N dimension until one 16-column Cube tile is filled. The
    // load-balance guard keeps the maximum number of token/S tiles assigned to one core no larger than baseT=1,
    // because MM2 and its AIV epilogue still execute once per token.
    const std::array<uint64_t, 5> sCandidates = {
        totalS_, std::min<uint64_t>(totalS_, 128UL), std::min<uint64_t>(totalS_, 64UL),
        std::min<uint64_t>(totalS_, CUBE_MAX_BASE_S), std::min<uint64_t>(totalS_, CUBE_BLOCK)};
    const std::array<uint64_t, 4> tCandidates = {8UL, 4UL, 2UL, 1UL};
    const std::array<uint64_t, 3> dCandidates = {std::min<uint64_t>(totalD_, 512UL),
                                                 std::min<uint64_t>(totalD_, 256UL),
                                                 std::min<uint64_t>(totalD_, 128UL)};
    uint64_t previousS = 0U;
    for (const uint64_t sCandidate : sCandidates) {
        if (sCandidate == 0U || sCandidate == previousS) {
            continue;
        }
        previousS = sCandidate;
        const uint32_t candidateSAlign = static_cast<uint32_t>(AlignUp<uint64_t>(sCandidate, CUBE_BLOCK));
        const uint64_t candidateSTileNum = CeilDiv(totalS_, sCandidate);
        const uint64_t mixedCoreNum = std::min<uint64_t>(aicCoreNum_, aivCoreNum_ / 2U);
        const uint64_t baselineWorkUnits = totalT_ * candidateSTileNum;
        const uint64_t baselineUsedCores = std::min<uint64_t>(mixedCoreNum, baselineWorkUnits);
        const uint64_t baselineTokenWork = CeilDiv(baselineWorkUnits, baselineUsedCores);
        for (const uint64_t candidateT : tCandidates) {
            if (candidateT > totalT_ || candidateT * totalN_ > CUBE_BLOCK) {
                continue;
            }
            const uint64_t candidateTGroups = CeilDiv(totalT_, candidateT);
            const uint64_t groupedWorkUnits = candidateTGroups * candidateSTileNum;
            const uint64_t groupedUsedCores = std::min<uint64_t>(mixedCoreNum, groupedWorkUnits);
            const uint64_t groupedTokenWork = CeilDiv(groupedWorkUnits, groupedUsedCores) * candidateT;
            if (groupedTokenWork > baselineTokenWork) {
                continue;
            }
            const uint64_t candidateMm1NAlign = AlignUp<uint64_t>(candidateT * totalN_, CUBE_BLOCK);
            uint64_t previousD = 0U;
            for (const uint64_t candidateD : dCandidates) {
                if (candidateD == 0U || candidateD == previousD) {
                    continue;
                }
                previousD = candidateD;
                const uint64_t baseDAlign = AlignUp<uint64_t>(candidateD, CUBE_BLOCK);
                const uint64_t dTileNum = CeilDiv(totalD_, candidateD);
                const uint64_t qL1BufferNum = dTileNum > 1U ? 2U : 1U;
                const uint64_t vL1BufferNum = dTileNum > 1U ? 2U : 1U;
                const uint64_t vUbBufferNum = 2U;
                const uint64_t qL1Elems = static_cast<uint64_t>(candidateSAlign) * baseDAlign;
                const uint64_t vL1Elems = candidateMm1NAlign * baseDAlign;
                const uint64_t eL1Elems = static_cast<uint64_t>(candidateSAlign) * nAlign_;
                const uint64_t vUbElems = static_cast<uint64_t>(nAlign_) * baseDAlign;
                const uint64_t dotUbElems = static_cast<uint64_t>(candidateSAlign) * nAlign_ + FP32_REG_ELEMS;
                const uint64_t reduceUbElems = FP32_REG_ELEMS;
                const uint64_t softmaxUbElems = static_cast<uint64_t>(2U) * candidateSAlign;

                const uint64_t l1Bytes =
                    (qL1BufferNum * qL1Elems + vL1BufferNum * vL1Elems + eL1Elems) * FP32_BYTES;
                const uint64_t ubBytes =
                    (vUbBufferNum * vUbElems + dotUbElems + reduceUbElems + softmaxUbElems) * FP32_BYTES;
                const uint64_t mm1BaseK = std::min<uint64_t>(baseDAlign, 64U);
                const uint64_t mm1L0ABytes = static_cast<uint64_t>(candidateSAlign) * mm1BaseK * FP32_BYTES;
                const uint64_t mm1L0BBytes = candidateMm1NAlign * mm1BaseK * FP32_BYTES;
                const uint64_t mm1L0CBytes =
                    static_cast<uint64_t>(candidateSAlign) * candidateMm1NAlign * FP32_BYTES;
                const uint64_t mm2L0ABytes = static_cast<uint64_t>(candidateSAlign) * nAlign_ * FP32_BYTES;
                const uint64_t mm2L0BBytes = static_cast<uint64_t>(nAlign_) * baseDAlign * FP32_BYTES;
                const uint64_t mm2L0CBytes = static_cast<uint64_t>(candidateSAlign) * baseDAlign * FP32_BYTES;
                const uint64_t mm1L1SlotBytes =
                    (static_cast<uint64_t>(candidateSAlign) + candidateMm1NAlign) * baseDAlign * FP32_BYTES;
                const uint64_t mm2L1SlotBytes = (static_cast<uint64_t>(candidateSAlign) * nAlign_ +
                                                 static_cast<uint64_t>(nAlign_) * baseDAlign) *
                                                FP32_BYTES;

                if (l1Bytes > usableL1Bytes || ubBytes > usableUbBytes || mm1L1SlotBytes > l1Size_ / 4U ||
                    mm2L1SlotBytes > l1Size_ / 4U || mm1L0ABytes > l0ASize_ || mm2L0ABytes > l0ASize_ ||
                    mm1L0BBytes > l0BSize_ || mm2L0BBytes > l0BSize_ || mm1L0CBytes > l0CSize_ ||
                    mm2L0CBytes > l0CSize_) {
                    continue;
                }

                baseT_ = static_cast<uint32_t>(candidateT);
                baseS_ = static_cast<uint32_t>(sCandidate);
                sAlign_ = candidateSAlign;
                baseD_ = static_cast<uint32_t>(candidateD);
                baseDAlign_ = static_cast<uint32_t>(baseDAlign);
                dTileNum_ = static_cast<uint32_t>(dTileNum);
                mm1NAlign_ = static_cast<uint32_t>(candidateMm1NAlign);
                qL1BufferNum_ = static_cast<uint32_t>(qL1BufferNum);
                vL1BufferNum_ = static_cast<uint32_t>(vL1BufferNum);
                vUbBufferNum_ = static_cast<uint32_t>(vUbBufferNum);
                qL1Elems_ = qL1Elems;
                vL1Elems_ = vL1Elems;
                eL1Elems_ = eL1Elems;
                vUbElems_ = vUbElems;
                dotUbElems_ = dotUbElems;
                reduceUbElems_ = reduceUbElems;
                softmaxUbElems_ = softmaxUbElems;
                return true;
            }
        }
    }
    return false;
}

ge::graphStatus BlockAttnResPrepareCubeTiling::DoOpTiling()
{
    OP_CHECK_IF(baseT_ == 0U || baseD_ == 0U, OP_LOGE(context_->GetNodeName(), "Cube tile was not selected"),
                return ge::GRAPH_FAILED);

    sTileNum_ = static_cast<uint32_t>(CeilDiv<uint64_t>(totalS_, baseS_));
    const uint64_t tTileNum = CeilDiv<uint64_t>(totalT_, baseT_);
    OP_CHECK_IF(tTileNum > std::numeric_limits<uint64_t>::max() / sTileNum_,
                OP_LOGE(context_->GetNodeName(), "Cube work count overflows uint64"), return ge::GRAPH_FAILED);
    const uint64_t totalWorkUnits = tTileNum * sTileNum_;
    OP_CHECK_IF(totalWorkUnits > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(context_->GetNodeName(), "Cube work units exceed uint32 tiling range"),
                return ge::GRAPH_FAILED);
    const uint64_t mixedCoreNum = std::min<uint64_t>(aicCoreNum_, aivCoreNum_ / 2U);
    usedCoreNum_ = static_cast<uint32_t>(std::min<uint64_t>(mixedCoreNum, totalWorkUnits));
    OP_CHECK_IF(usedCoreNum_ == 0U || totalWorkUnits / usedCoreNum_ > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(context_->GetNodeName(), "invalid Cube work distribution"), return ge::GRAPH_FAILED);
    const uint32_t blockFactor = static_cast<uint32_t>(totalWorkUnits / usedCoreNum_);
    const uint32_t bigCoreNum = static_cast<uint32_t>(totalWorkUnits % usedCoreNum_);

    // MM1 produces [baseS, baseT * N]. AIV and AIC use two E slots to overlap the next token's softmax with the
    // current token's MM2. MM2 Fixpipe writes O directly and therefore needs no output workspace.
    const uint64_t dotWorkspaceElems = static_cast<uint64_t>(baseS_) * baseT_ * totalN_;
    const uint64_t eWorkspaceElems = static_cast<uint64_t>(sAlign_) * nAlign_;
    const uint64_t eWorkspaceBufferNum = baseT_ > 1U ? 2U : 1U;
    OP_CHECK_IF(eWorkspaceElems > std::numeric_limits<uint64_t>::max() / eWorkspaceBufferNum ||
                    dotWorkspaceElems > std::numeric_limits<uint64_t>::max() -
                                            eWorkspaceBufferNum * eWorkspaceElems,
                OP_LOGE(context_->GetNodeName(), "Cube per-core workspace size overflows uint64"),
                return ge::GRAPH_FAILED);
    workspacePerCoreElems_ = dotWorkspaceElems + eWorkspaceBufferNum * eWorkspaceElems;
    OP_CHECK_IF(workspacePerCoreElems_ > std::numeric_limits<uint64_t>::max() / usedCoreNum_ / FP32_BYTES,
                OP_LOGE(context_->GetNodeName(), "Cube workspace size overflows uint64"), return ge::GRAPH_FAILED);
    const uint64_t cubeWorkspaceSize = workspacePerCoreElems_ * usedCoreNum_ * FP32_BYTES;
    OP_CHECK_IF(systemWorkspaceSize_ > std::numeric_limits<uint64_t>::max() - cubeWorkspaceSize,
                OP_LOGE(context_->GetNodeName(), "Cube total workspace size overflows uint64"),
                return ge::GRAPH_FAILED);
    workspaceSize_ = systemWorkspaceSize_ + cubeWorkspaceSize;

    tilingData_.totalT = static_cast<uint32_t>(totalT_);
    tilingData_.totalN = static_cast<uint32_t>(totalN_);
    tilingData_.totalS = static_cast<uint32_t>(totalS_);
    tilingData_.totalWorkUnits = static_cast<uint32_t>(totalWorkUnits);
    tilingData_.totalD = totalD_;
    tilingData_.usedCoreNum = usedCoreNum_;
    tilingData_.aicCoreNum = static_cast<uint32_t>(aicCoreNum_);
    tilingData_.aivCoreNum = static_cast<uint32_t>(aivCoreNum_);
    tilingData_.bigCoreNum = bigCoreNum;
    tilingData_.blockFactor = blockFactor;
    tilingData_.tailBlockFactor = blockFactor + (bigCoreNum > 0U ? 1U : 0U);
    tilingData_.baseT = baseT_;
    tilingData_.baseS = baseS_;
    tilingData_.baseD = baseD_;
    tilingData_.baseDAlign = baseDAlign_;
    tilingData_.sTileNum = sTileNum_;
    tilingData_.dTileNum = dTileNum_;
    tilingData_.sAlign = sAlign_;
    tilingData_.nAlign = nAlign_;
    tilingData_.mm1NAlign = mm1NAlign_;
    tilingData_.dAlign = dAlign_;
    tilingData_.qL1BufferNum = static_cast<uint8_t>(qL1BufferNum_);
    tilingData_.vL1BufferNum = static_cast<uint8_t>(vL1BufferNum_);
    tilingData_.vUbBufferNum = static_cast<uint8_t>(vUbBufferNum_);
    tilingData_.l1CacheMode = static_cast<uint32_t>(BlockAttnResPrepareVCacheMode::RELOAD);
    tilingData_.vL1CacheRows = static_cast<uint8_t>(0U);
    tilingData_.qL1Elems = qL1Elems_;
    tilingData_.vL1Elems = vL1Elems_;
    tilingData_.eL1Elems = eL1Elems_;
    tilingData_.vUbElems = vUbElems_;
    tilingData_.dotUbElems = dotUbElems_;
    tilingData_.reduceUbElems = reduceUbElems_;
    tilingData_.softmaxUbElems = softmaxUbElems_;
    tilingData_.workspacePerCoreElems = workspacePerCoreElems_;
    tilingData_.eps = eps_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResPrepareCubeTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t BlockAttnResPrepareCubeTiling::GetTilingKey() const
{
    return BLOCK_ATTN_RES_PREPARE_MIX_CUBE_KEY;
}

ge::graphStatus BlockAttnResPrepareCubeTiling::GetWorkspaceSize()
{
    size_t *workspaceSizes = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaceSizes == nullptr, OP_LOGE(context_->GetNodeName(), "workspace size address is null"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(workspaceSize_ > std::numeric_limits<size_t>::max(),
                OP_LOGE(context_->GetNodeName(), "Cube workspace exceeds size_t"), return ge::GRAPH_FAILED);
    workspaceSizes[0] = static_cast<size_t>(workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockAttnResPrepareCubeTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    context_->SetTilingKey(GetTilingKey());
    context_->SetScheduleMode(VECTOR_SCHEDULE_MODE);
    auto *rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(rawTilingData == nullptr || rawTilingData->GetData() == nullptr,
                OP_LOGE(context_->GetNodeName(), "raw Cube tiling data is null"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(rawTilingData->GetCapacity() < sizeof(BlockAttnResPrepareCubeTilingData),
                OP_LOGE(context_->GetNodeName(), "raw Cube tiling data capacity is insufficient"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(), &tilingData_, sizeof(tilingData_)) !=
                    EOK,
                OP_LOGE(context_->GetNodeName(), "copy Cube tiling data failed"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(BlockAttnResPrepareCubeTilingData));
    return ge::GRAPH_SUCCESS;
}

void BlockAttnResPrepareCubeTiling::DumpTilingInfo()
{
    OP_LOGI(context_->GetNodeName(),
            "BlockAttnResPrepare Cube tiling: T=%lu N=%lu S=%lu D=%lu cores=%u baseS=%u baseD=%u "
            "baseT=%u sLoops=%u dLoops=%u qL1/vL1/vUb buffers=%u/%u/%u nAlign/mm1NAlign=%u/%u "
            "workspacePerCore=%lu workspace=%lu",
            totalT_, totalN_, totalS_, totalD_, usedCoreNum_, baseS_, baseD_, baseT_, sTileNum_, dTileNum_,
            qL1BufferNum_, vL1BufferNum_, vUbBufferNum_, nAlign_, mm1NAlign_, workspacePerCoreElems_, workspaceSize_);
}

REGISTER_OPS_TILING_TEMPLATE(BlockAttnResPrepare, BlockAttnResPrepareCubeTiling, 0);
REGISTER_OPS_TILING_TEMPLATE(BlockAttnResPrepare, BlockAttnResPrepareTiling, 1000);

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
