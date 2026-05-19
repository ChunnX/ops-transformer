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
 * \file all_gather_hccl_utils.cpp
 * \brief HCCL limit adjustment utilities implementation
 */

#include "all_gather_hccl_utils.h"
#include "mc2_log.h"
#include <algorithm>
#include <cmath>

namespace optiling {

void AdjustCutResultForHCCL(CutResult& cutRes,
                            uint64_t mValue, uint64_t kValue,
                            uint64_t dtypeSize, uint64_t rankDim,
                            const std::string& opName)
{
    // 原始切分
    uint64_t originalCommSize = cutRes.longTileLen * kValue * dtypeSize * rankDim;
    double originalCommMB = static_cast<double>(originalCommSize) / (1024.0 * 1024.0);
    OP_LOGD(opName.c_str(), "Formulaic original cut: tileM=%lu, tileCnt=%lu, tailM=%lu, "
            "shortTileAtBack=%d, commSize=%.2fMB",
            cutRes.longTileLen, cutRes.numLongTile, cutRes.shortTileLen,
            cutRes.shortTileAtBack, originalCommMB);
    // 检查是否需要修正（通信量未超限）
    if (originalCommSize <= HCCL_MEM_LIMIT) {
        OP_LOGD(opName.c_str(), "HCCL limit satisfied, no adjustment needed");
        return;
    }
    
    // 计算HCCL限制下的最大tileM
    uint64_t hcclMaxTileM = HCCL_MEM_LIMIT / (kValue * dtypeSize * rankDim);
    // 选择256对齐的候选tileM
    uint64_t candidateTileM = (hcclMaxTileM / ALIGN_LEN) * ALIGN_LEN;
    if (candidateTileM == 0) {
        candidateTileM = ALIGN_LEN;
    }
    OP_LOGD(opName.c_str(), "HCCL limit: M=%lu, K=%lu, dtypeSize=%lu, rankDim=%lu, "
            "hcclMaxTileM=%lu, candidateTileM=%lu(256-aligned)",
            mValue, kValue, dtypeSize, rankDim, hcclMaxTileM, candidateTileM);
    
    // 预估切分轮次，并进行可行性检查
    uint64_t tileCnt = (mValue + candidateTileM - 1) / candidateTileM;
    if (tileCnt > HCCL_MAX_TILE_CNT) {
        double requiredCommMB = static_cast<double>(candidateTileM * kValue * dtypeSize * rankDim)
                                / (1024.0 * 1024.0);
        OP_LOGE(opName.c_str(),
                "HCCL limit cannot be satisfied:\n"
                "  Input: M=%lu, K=%lu, dtypeSize=%lu, rankDim=%lu\n"
                "  HCCL maxTileM=%lu, 256-aligned candidate=%lu\n"
                "  Required tileCnt=%lu > maxTileCnt=%lu\n"
                "  Original: tileM=%lu, tileCnt=%lu, commSize=%.2fMB > 256MB\n"
                "  Candidate commSize=%.2fMB\n"
                "  This shape is NOT SUPPORTED",
                mValue, kValue, dtypeSize, rankDim,
                hcclMaxTileM, candidateTileM,
                tileCnt, HCCL_MAX_TILE_CNT,
                cutRes.longTileLen, cutRes.numLongTile, originalCommMB,
                requiredCommMB);
        return;  // 按照调整策略无法支持，保持原始切分（后续会报错）
    }

    // 确定tileM（使用256对齐候选值）
    uint64_t tileM = candidateTileM;
    // 计算切分
    uint64_t numLongTile = mValue / tileM;
    uint64_t tailM = mValue - numLongTile * tileM;
    // 保证至少一个主块
    if (numLongTile == 0) {
        numLongTile = 1;
        tileM = mValue;
        tailM = 0;
        // 再次检查HCCL限制
        uint64_t newCommSize = tileM * kValue * dtypeSize * rankDim;
        if (newCommSize > HCCL_MEM_LIMIT) {
            OP_LOGE(opName.c_str(),
                    "Shape too small to satisfy HCCL limit:\n"
                    "  M=%lu < minTileLen=%lu\n"
                    "  Cannot split, single tile exceeds HCCL limit\n"
                    "  This shape is NOT SUPPORTED",
                    mValue, candidateTileM);
            return;
        }
    }

    // 更新切分结果
    cutRes.longTileLen = tileM;
    cutRes.numLongTile = numLongTile;
    cutRes.shortTileLen = tailM;
    cutRes.numShortTile = (tailM > 0) ? 1 : 0;
    cutRes.totalTileCnt = numLongTile + cutRes.numShortTile;
    // shortTileAtBack保持原始值
    uint64_t newCommSize = tileM * kValue * dtypeSize * rankDim;
    double newCommMB = static_cast<double>(newCommSize) / (1024.0 * 1024.0);
    // 调整后的切分
    OP_LOGD(opName.c_str(),
            "HCCL adjusted cut: tileM=%lu(256-aligned), numLongTile=%lu, tailM=%lu, "
            "shortTileAtBack=%d, commSize=%.2fMB",
            tileM, numLongTile, tailM, cutRes.shortTileAtBack, newCommMB);
    if (tailM == 0) {
        OP_LOGD(opName.c_str(), "Perfect split: no tail block, all tiles aligned");
    }
}

}  // namespace optiling