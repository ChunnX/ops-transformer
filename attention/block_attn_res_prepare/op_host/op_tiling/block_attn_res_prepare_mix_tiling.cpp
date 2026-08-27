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
 * \file block_attn_res_prepare_mix_tiling.cpp
 * \brief Mixed Cube/Vector tiling for BlockAttnResPrepare.
 */

#include "block_attn_res_prepare_mix_tiling.h"

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

#include "log/log.h"

namespace {

const std::vector<int32_t> supportedNpuArch = {static_cast<int32_t>(NpuArch::DAV_3510)};
constexpr int32_t TILING_PRIORITY = 0;
} // namespace

namespace optiling {
namespace {

constexpr uint64_t FP32_BYTES = sizeof(float);
constexpr uint64_t UB_RESERVED_BYTES = 8UL * 1024UL;
constexpr uint64_t L1_RESERVED_BYTES = 8UL * 1024UL;
constexpr uint32_t FP32_REG_ELEMS = 64U;
constexpr uint32_t CUBE_BLOCK = 16U;
constexpr uint32_t CUBE_MAX_BASE_S = 32U;
constexpr uint32_t CUBE_MIN_T = 32U;
constexpr uint32_t CUBE_MIN_S = 16U;
constexpr uint32_t CUBE_MIN_D = 256U;
constexpr uint32_t STAT_VECTOR_COUNT = 2U;
constexpr uint64_t DOUBLE_BUFFER_NUM = 2UL;
constexpr uint64_t AIV_CORE_NUM_PER_AIC = 2UL;
constexpr uint64_t L1_SLOT_NUM = 4UL;
constexpr uint64_t CUBE_BASE_S_LARGE = 128UL;
constexpr uint64_t CUBE_BASE_S_MEDIUM = 64UL;
constexpr uint64_t CUBE_BASE_T_LARGE = 8UL;
constexpr uint64_t CUBE_BASE_T_MEDIUM = 4UL;
constexpr uint64_t CUBE_BASE_T_SMALL = 2UL;
constexpr uint64_t CUBE_BASE_D_LARGE = 512UL;
constexpr uint64_t CUBE_BASE_D_MEDIUM = 256UL;
constexpr uint64_t CUBE_BASE_D_SMALL = 128UL;
constexpr uint64_t CUBE_MM1_MAX_BASE_K = 64UL;
constexpr size_t CUBE_BASE_S_CANDIDATE_NUM = 5;
constexpr size_t CUBE_BASE_T_CANDIDATE_NUM = 4;
constexpr size_t CUBE_BASE_D_CANDIDATE_NUM = 3;

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

} // namespace

bool BlockAttnResPrepareMixTiling::IsCapable()
{
    if (totalT_ < CUBE_MIN_T || totalS_ < CUBE_MIN_S || totalD_ < CUBE_MIN_D) {
        return false;
    }
    // MM2 Fixpipe writes a strided [S, D] tile directly into O[S, T, D]. The destination row stride is T * D
    // elements and is encoded by the hardware interface as uint32_t.
    if (totalT_ > std::numeric_limits<uint32_t>::max() / totalD_) {
        return false;
    }
    return SelectMixTileShape();
}

bool BlockAttnResPrepareMixTiling::SelectMixTileShape()
{
    const uint64_t usableUbBytes = ubSize_ - UB_RESERVED_BYTES;
    const uint64_t usableL1Bytes = l1Size_ - L1_RESERVED_BYTES;
    const uint64_t mixedCoreNum = std::min<uint64_t>(aicCoreNum_, aivCoreNum_ / AIV_CORE_NUM_PER_AIC);
    if (mixedCoreNum == 0U) {
        return false;
    }
    nAlign_ = static_cast<uint32_t>(AlignUp<uint64_t>(totalN_, CUBE_BLOCK));
    dAlign_ = static_cast<uint32_t>(AlignUp<uint64_t>(totalD_, CUBE_BLOCK));

    // Pack adjacent token matrices into MM1's N dimension until one 16-column Cube tile is filled. The
    // load-balance guard keeps the maximum number of token/S tiles assigned to one core no larger than baseT=1,
    // because MM2 and its AIV epilogue still execute once per token.
    const std::array<uint64_t, CUBE_BASE_S_CANDIDATE_NUM> sCandidates = {
        totalS_, std::min<uint64_t>(totalS_, CUBE_BASE_S_LARGE), std::min<uint64_t>(totalS_, CUBE_BASE_S_MEDIUM),
        std::min<uint64_t>(totalS_, CUBE_MAX_BASE_S), std::min<uint64_t>(totalS_, CUBE_BLOCK)};
    const std::array<uint64_t, CUBE_BASE_T_CANDIDATE_NUM> tCandidates = {CUBE_BASE_T_LARGE, CUBE_BASE_T_MEDIUM,
                                                                         CUBE_BASE_T_SMALL, 1UL};
    const std::array<uint64_t, CUBE_BASE_D_CANDIDATE_NUM> dCandidates = {
        std::min<uint64_t>(totalD_, CUBE_BASE_D_LARGE), std::min<uint64_t>(totalD_, CUBE_BASE_D_MEDIUM),
        std::min<uint64_t>(totalD_, CUBE_BASE_D_SMALL)};
    uint64_t previousS = 0U;
    for (const uint64_t sCandidate : sCandidates) {
        if (sCandidate == 0U || sCandidate == previousS) {
            continue;
        }
        previousS = sCandidate;
        const uint32_t candidateSAlign = static_cast<uint32_t>(AlignUp<uint64_t>(sCandidate, CUBE_BLOCK));
        const uint64_t candidateSTileNum = CeilDiv(totalS_, sCandidate);
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
                const uint64_t qL1BufferNum = dTileNum > 1U ? DOUBLE_BUFFER_NUM : 1U;
                const uint64_t vL1BufferNum = dTileNum > 1U ? DOUBLE_BUFFER_NUM : 1U;
                const uint64_t vUbBufferNum = DOUBLE_BUFFER_NUM;
                const uint64_t qL1Elems = static_cast<uint64_t>(candidateSAlign) * baseDAlign;
                const uint64_t vL1Elems = candidateMm1NAlign * baseDAlign;
                const uint64_t eL1Elems = static_cast<uint64_t>(candidateSAlign) * nAlign_;
                const uint64_t vUbElems = static_cast<uint64_t>(nAlign_) * baseDAlign;
                const uint64_t dotUbElems = static_cast<uint64_t>(candidateSAlign) * nAlign_ + FP32_REG_ELEMS;
                const uint64_t reduceUbElems = FP32_REG_ELEMS;
                const uint64_t softmaxUbElems = static_cast<uint64_t>(STAT_VECTOR_COUNT) * candidateSAlign;

                const uint64_t l1Bytes = (qL1BufferNum * qL1Elems + vL1BufferNum * vL1Elems + eL1Elems) * FP32_BYTES;
                const uint64_t ubBytes =
                    (vUbBufferNum * vUbElems + dotUbElems + reduceUbElems + softmaxUbElems) * FP32_BYTES;
                const uint64_t mm1BaseK = std::min<uint64_t>(baseDAlign, CUBE_MM1_MAX_BASE_K);
                const uint64_t mm1L0ABytes = static_cast<uint64_t>(candidateSAlign) * mm1BaseK * FP32_BYTES;
                const uint64_t mm1L0BBytes = candidateMm1NAlign * mm1BaseK * FP32_BYTES;
                const uint64_t mm1L0CBytes = static_cast<uint64_t>(candidateSAlign) * candidateMm1NAlign * FP32_BYTES;
                const uint64_t mm2L0ABytes = static_cast<uint64_t>(candidateSAlign) * nAlign_ * FP32_BYTES;
                const uint64_t mm2L0BBytes = static_cast<uint64_t>(nAlign_) * baseDAlign * FP32_BYTES;
                const uint64_t mm2L0CBytes = static_cast<uint64_t>(candidateSAlign) * baseDAlign * FP32_BYTES;
                const uint64_t mm1L1SlotBytes =
                    (static_cast<uint64_t>(candidateSAlign) + candidateMm1NAlign) * baseDAlign * FP32_BYTES;
                const uint64_t mm2L1SlotBytes =
                    (static_cast<uint64_t>(candidateSAlign) * nAlign_ + static_cast<uint64_t>(nAlign_) * baseDAlign) *
                    FP32_BYTES;

                if (l1Bytes > usableL1Bytes || ubBytes > usableUbBytes || mm1L1SlotBytes > l1Size_ / L1_SLOT_NUM ||
                    mm2L1SlotBytes > l1Size_ / L1_SLOT_NUM || mm1L0ABytes > l0ASize_ || mm2L0ABytes > l0ASize_ ||
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

ge::graphStatus BlockAttnResPrepareMixTiling::DoOpTiling()
{
    OP_CHECK_IF(baseT_ == 0U || baseD_ == 0U,
                OP_LOGE(context_->GetNodeName(),
                        "Mix tile was not selected: T=%lu, N=%lu, S=%lu, D=%lu, baseT=%u, baseS=%u, baseD=%u", totalT_,
                        totalN_, totalS_, totalD_, baseT_, baseS_, baseD_),
                return ge::GRAPH_FAILED);

    sTileNum_ = static_cast<uint32_t>(CeilDiv<uint64_t>(totalS_, baseS_));
    const uint64_t tTileNum = CeilDiv<uint64_t>(totalT_, baseT_);
    OP_CHECK_IF(tTileNum > std::numeric_limits<uint64_t>::max() / sTileNum_,
                OP_LOGE(context_->GetNodeName(), "Mix work count overflows uint64: tTileNum=%lu, sTileNum=%u", tTileNum,
                        sTileNum_),
                return ge::GRAPH_FAILED);
    const uint64_t totalWorkUnits = tTileNum * sTileNum_;
    const uint64_t mixedCoreNum = std::min<uint64_t>(aicCoreNum_, aivCoreNum_ / AIV_CORE_NUM_PER_AIC);
    WorkDistribution distribution;
    if (CalculateWorkDistribution(totalWorkUnits, mixedCoreNum, "Mix", distribution) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    usedCoreNum_ = distribution.usedCoreNum;

    // MM1 writes rows with mm1NAlign pitch, which also reserves capacity for runtime token grouping. AIV and AIC use
    // two E slots when more than one token exists, even if host tiling selects baseT=1, because the kernel may expand
    // baseT after reading validN. MM2 Fixpipe writes O directly and therefore needs no output workspace.
    const uint64_t dotWorkspaceElems = static_cast<uint64_t>(baseS_) * mm1NAlign_;
    const uint64_t eWorkspaceElems = static_cast<uint64_t>(sAlign_) * nAlign_;
    const uint64_t eWorkspaceBufferNum = totalT_ > 1U ? DOUBLE_BUFFER_NUM : 1U;
    OP_CHECK_IF(eWorkspaceElems > std::numeric_limits<uint64_t>::max() / eWorkspaceBufferNum ||
                    dotWorkspaceElems > std::numeric_limits<uint64_t>::max() - eWorkspaceBufferNum * eWorkspaceElems,
                OP_LOGE(context_->GetNodeName(),
                        "Mix per-core workspace size overflows uint64: dotWorkspaceElems=%lu, "
                        "eWorkspaceElems=%lu, eWorkspaceBufferNum=%lu",
                        dotWorkspaceElems, eWorkspaceElems, eWorkspaceBufferNum),
                return ge::GRAPH_FAILED);
    workspacePerCoreElems_ = dotWorkspaceElems + eWorkspaceBufferNum * eWorkspaceElems;
    OP_CHECK_IF(workspacePerCoreElems_ > std::numeric_limits<uint64_t>::max() / usedCoreNum_ / FP32_BYTES,
                OP_LOGE(context_->GetNodeName(),
                        "Mix workspace size overflows uint64: workspacePerCoreElems=%lu, usedCoreNum=%u, "
                        "elementBytes=%lu",
                        workspacePerCoreElems_, usedCoreNum_, FP32_BYTES),
                return ge::GRAPH_FAILED);
    const uint64_t mixWorkspaceSize = workspacePerCoreElems_ * usedCoreNum_ * FP32_BYTES;
    OP_CHECK_IF(systemWorkspaceSize_ > std::numeric_limits<uint64_t>::max() - mixWorkspaceSize,
                OP_LOGE(context_->GetNodeName(),
                        "Mix total workspace size overflows uint64: systemWorkspaceSize=%lu, mixWorkspaceSize=%lu",
                        systemWorkspaceSize_, mixWorkspaceSize),
                return ge::GRAPH_FAILED);
    workspaceSize_ = systemWorkspaceSize_ + mixWorkspaceSize;

    tilingData_.totalT = static_cast<uint32_t>(totalT_);
    tilingData_.totalN = static_cast<uint8_t>(totalN_);
    tilingData_.totalS = static_cast<uint32_t>(totalS_);
    tilingData_.totalWorkUnits = static_cast<uint32_t>(totalWorkUnits);
    tilingData_.totalD = static_cast<uint32_t>(totalD_);
    tilingData_.usedCoreNum = static_cast<uint16_t>(usedCoreNum_);
    tilingData_.aicCoreNum = static_cast<uint16_t>(aicCoreNum_);
    tilingData_.aivCoreNum = static_cast<uint16_t>(aivCoreNum_);
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
    tilingData_.l1CacheMode = static_cast<uint8_t>(BlockAttnResPrepareVCacheMode::RELOAD);
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

uint64_t BlockAttnResPrepareMixTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(BLOCK_ATTN_RES_PREPARE_TPL_MIX_CUBE);
}

BlockAttnResPrepareBaseTiling::TilingDataView BlockAttnResPrepareMixTiling::GetTilingDataView() const
{
    return {&tilingData_, sizeof(tilingData_), "Mix"};
}

void BlockAttnResPrepareMixTiling::DumpTilingInfo()
{
    OP_LOGI(context_->GetNodeName(),
            "BlockAttnResPrepare Mix tiling: T=%lu N=%lu S=%lu D=%lu cores=%u baseS=%u baseD=%u "
            "baseT=%u sLoops=%u dLoops=%u qL1/vL1/vUb buffers=%u/%u/%u nAlign/mm1NAlign=%u/%u "
            "workspacePerCore=%lu workspace=%lu",
            totalT_, totalN_, totalS_, totalD_, usedCoreNum_, baseS_, baseD_, baseT_, sTileNum_, dTileNum_,
            qL1BufferNum_, vL1BufferNum_, vUbBufferNum_, nAlign_, mm1NAlign_, workspacePerCoreElems_, workspaceSize_);
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(BlockAttnResPrepare, BlockAttnResPrepareMixTiling, supportedNpuArch,
                                   TILING_PRIORITY);

} // namespace optiling
