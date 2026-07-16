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
 * \file mc2_fit_based_balance_tiling.cpp
 * \brief
 */
#include "mc2_log.h"
#include "mc2_fit_based_balance_tiling.h"

CutResult Mc2FitBasedBalanceTiling::GetTiling()
{
    const uint64_t FORCE_TILE_CNT = 1; // 消融实验: 0=正常tiling, 1/2/4=强制M轴均分
    if (FORCE_TILE_CNT != 0 && mmInfo_.mValue % FORCE_TILE_CNT == 0) {
        uint64_t tileLen = mmInfo_.mValue / FORCE_TILE_CNT;
        tilingM_.cutRes.shortTileAtBack = true;
        tilingM_.cutRes.longTileLen = tileLen;
        tilingM_.cutRes.numLongTile = FORCE_TILE_CNT;
        tilingM_.cutRes.shortTileLen = 0U;
        tilingM_.cutRes.numShortTile = 0U;
        tilingM_.cutRes.totalTileCnt = FORCE_TILE_CNT;
        OPS_LOG_D("Mc2FitBasedBalanceTiling",
                  "Force uniform split: tileCnt=%lu, tileLen=%lu, mValue=%lu",
                  FORCE_TILE_CNT, tileLen, mmInfo_.mValue);
        return tilingM_.cutRes;
    }

    EstimateMMCommTime();
    SetShortTileLen();
    if (tilingM_.totalLen < tilingM_.cutRes.shortTileLen * 2) { // do not cut if mValue is too small
        tilingM_.NoCutTiling();
    } else {
        SetLongTileLen();
        AdjustLongShortTileLen();
    }
    OPS_LOG_D("Mc2FitBasedBalanceTiling",
              "Input shape {M, N, K} = {%lu, %lu, %lu}, Final cut: shortTileAtBack %d,"
              " longTileLen %lu, numLongTile %lu, shortTileLen %lu, numShortTile %lu",
              mmInfo_.mValue, mmInfo_.nValue, mmInfo_.kValue, tilingM_.cutRes.shortTileAtBack,
              tilingM_.cutRes.longTileLen, tilingM_.cutRes.numLongTile, tilingM_.cutRes.shortTileLen,
              tilingM_.cutRes.numShortTile);

    return tilingM_.cutRes;
}
