/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You can not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_gather_hccl_utils.cpp
 * \brief HCCL limit adjustment utilities implementation
 */

#include "all_gather_hccl_utils.h"
#include "mc2_log.h"
#include <algorithm>
#include <cmath>

namespace optiling {

uint64_t CalcMaxTileMFromHcclLimit(const CutResult& cutRes,
                                   uint64_t kValue, uint64_t dtypeSize, uint64_t rankDim,
                                   const std::string& opName)
{
    uint64_t originalCommSize = cutRes.longTileLen * kValue * dtypeSize * rankDim;
    double originalCommMB = static_cast<double>(originalCommSize) / (1024.0 * 1024.0);
    OP_LOGD(opName.c_str(), "Formulaic original cut: tileM=%lu, tileCnt=%lu, tailM=%lu, "
            "shortTileAtBack=%d, commSize=%.2fMB",
            cutRes.longTileLen, cutRes.numLongTile, cutRes.shortTileLen,
            cutRes.shortTileAtBack, originalCommMB);

    if (originalCommSize <= HCCL_MEM_LIMIT) {
        OP_LOGD(opName.c_str(), "HCCL limit satisfied, no adjustment needed");
        return HCCL_NO_ADJUSTMENT;
    }

    uint64_t maxTileM = HCCL_MEM_LIMIT / (kValue * dtypeSize * rankDim);
    if (maxTileM < ALIGN_LEN) {
        OP_LOGE(opName.c_str(),
                "Shape too small: maxTileM=%lu < minAlignLen=%lu\n"
                "  Cannot satisfy HCCL 256MB limit\n"
                "  This shape is NOT SUPPORTED",
                maxTileM, ALIGN_LEN);
        return HCCL_UNSUPPORTED;
    }

    OP_LOGD(opName.c_str(), "HCCL limit requires adjustment: maxTileM=%lu", maxTileM);
    return maxTileM;
}

uint64_t SelectOptimalCandidateTileM(uint64_t maxTileM)
{
    for (uint64_t candidate : TILE_M_CANDIDATES) {
        if (candidate <= maxTileM) {
            return candidate;
        }
    }

    return (maxTileM / ALIGN_LEN) * ALIGN_LEN;
}

uint64_t DetermineFinalTileMWith63Limit(uint64_t mValue, uint64_t candidateTileM, uint64_t maxTileM,
                                        uint64_t kValue, uint64_t dtypeSize, uint64_t rankDim,
                                        const std::string& opName)
{
    uint64_t totalTileCnt = (mValue + candidateTileM - 1) / candidateTileM;

    if (totalTileCnt <= HCCL_MAX_TOTAL_TILES) {
        OP_LOGD(opName.c_str(), "Candidate tileM=%lu satisfies limit: totalTileCnt=%lu ≤ %lu",
                candidateTileM, totalTileCnt, HCCL_MAX_TOTAL_TILES);
        return candidateTileM;
    }

    OP_LOGD(opName.c_str(), "Candidate tileM=%lu requires too many tiles: totalTileCnt=%lu > %lu, "
            "recalculating...",
            candidateTileM, totalTileCnt, HCCL_MAX_TOTAL_TILES);

    uint64_t minTileM = (mValue + HCCL_MAX_TOTAL_TILES - 1) / HCCL_MAX_TOTAL_TILES;
    uint64_t minTileM_align = ((minTileM + ALIGN_LEN - 1) / ALIGN_LEN) * ALIGN_LEN;

    uint64_t finalTileM = HCCL_UNSUPPORTED;
    for (size_t i = 0; i < sizeof(TILE_M_CANDIDATES) / sizeof(TILE_M_CANDIDATES[0]); i++) {
        if (TILE_M_CANDIDATES[i] >= minTileM_align) {
            finalTileM = TILE_M_CANDIDATES[i];
            break;
        }
    }

    if (finalTileM == HCCL_UNSUPPORTED) {
        finalTileM = minTileM_align;
    }

    if (finalTileM > maxTileM) {
        uint64_t requiredCommMB = static_cast<double>(finalTileM * kValue * dtypeSize * rankDim)
                                  / (1024.0 * 1024.0);
        OP_LOGE(opName.c_str(),
                "HCCL limit cannot be satisfied:\n"
                "  Input: M=%lu, K=%lu, dtypeSize=%lu, rankDim=%lu\n"
                "  HCCL maxTileM=%lu, required minTileM=%lu\n"
                "  Even with %lu tiles (max limit), each tile=%.2fMB > 256MB\n"
                "  This shape is NOT SUPPORTED",
                mValue, kValue, dtypeSize, rankDim,
                maxTileM, finalTileM,
                HCCL_MAX_TOTAL_TILES, requiredCommMB);
        return HCCL_UNSUPPORTED;
    }

    OP_LOGD(opName.c_str(), "Recalculated tileM=%lu to satisfy %lu tile limit",
            finalTileM, HCCL_MAX_TOTAL_TILES);
    return finalTileM;
}

void ApplyTileSplit(CutResult& cutRes, uint64_t mValue, uint64_t tileM,
                    uint64_t kValue, uint64_t dtypeSize, uint64_t rankDim,
                    const std::string& opName)
{
    uint64_t numLongTile = mValue / tileM;
    uint64_t tailM = mValue - numLongTile * tileM;

    if (numLongTile == 0) {
        numLongTile = 1;
        tileM = mValue;
        tailM = 0;

        uint64_t newCommSize = tileM * kValue * dtypeSize * rankDim;
        if (newCommSize > HCCL_MEM_LIMIT) {
            OP_LOGE(opName.c_str(),
                    "Shape too small to satisfy HCCL limit:\n"
                    "  M=%lu < tileM=%lu\n"
                    "  Cannot split, single tile exceeds 256MB\n"
                    "  This shape is NOT SUPPORTED",
                    mValue, tileM);
            return;
        }
    }

    cutRes.longTileLen = tileM;
    cutRes.numLongTile = numLongTile;
    cutRes.shortTileLen = tailM;
    cutRes.numShortTile = (tailM > 0) ? 1 : 0;
    cutRes.totalTileCnt = numLongTile + cutRes.numShortTile;

    uint64_t newCommSize = tileM * kValue * dtypeSize * rankDim;
    double newCommMB = static_cast<double>(newCommSize) / (1024.0 * 1024.0);
    OP_LOGD(opName.c_str(),
            "HCCL adjusted cut: tileM=%lu, numLongTile=%lu, tailM=%lu, "
            "totalTileCnt=%lu, shortTileAtBack=%d, commSize=%.2fMB",
            tileM, numLongTile, tailM, cutRes.totalTileCnt, cutRes.shortTileAtBack, newCommMB);

    if (tailM == 0) {
        OP_LOGD(opName.c_str(), "Perfect split: no tail block, all tiles aligned");
    }
}

void AdjustCutResultForHCCL(CutResult& cutRes,
                            uint64_t mValue, uint64_t kValue,
                            uint64_t dtypeSize, uint64_t rankDim,
                            const std::string& opName)
{
    uint64_t maxTileM = CalcMaxTileMFromHcclLimit(cutRes, kValue, dtypeSize, rankDim, opName);

    if (maxTileM == HCCL_NO_ADJUSTMENT) {
        return;
    }

    if (maxTileM == HCCL_UNSUPPORTED) {
        return;
    }

    uint64_t candidateTileM = SelectOptimalCandidateTileM(maxTileM);
    OP_LOGD(opName.c_str(), "Selected optimal candidate tileM=%lu", candidateTileM);

    uint64_t tileM = DetermineFinalTileMWith63Limit(mValue, candidateTileM, maxTileM,
                                                    kValue, dtypeSize, rankDim, opName);

    if (tileM == HCCL_UNSUPPORTED) {
        return;
    }

    ApplyTileSplit(cutRes, mValue, tileM, kValue, dtypeSize, rankDim, opName);
}

}  // namespace optiling