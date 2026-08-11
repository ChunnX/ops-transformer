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
#include <cstdlib>

#include "mc2_log.h"
#include "mc2_fit_based_balance_tiling.h"

namespace {
// 消融实验: 通过环境变量强制M轴均分切分块数, 无需重编算子包。
// 使能方式:
//   export MC2_FORCE_TILE_CNT=2   # 强制均分2块 (取值 1/2/4, 需 mValue 能被该值整除)
//   unset MC2_FORCE_TILE_CNT      # 恢复原有自适应tiling (未设置或0=自适应)
constexpr char FORCE_TILE_CNT_ENV[] = "MC2_FORCE_TILE_CNT";
} // namespace

CutResult Mc2FitBasedBalanceTiling::GetTiling()
{
    const char *forceTileEnv = getenv(FORCE_TILE_CNT_ENV);
    const uint64_t forceTileCnt = (forceTileEnv != nullptr) ? std::strtoul(forceTileEnv, nullptr, 10) : 0U;
    if (forceTileCnt != 0 && mmInfo_.mValue % forceTileCnt == 0) {
        uint64_t tileLen = mmInfo_.mValue / forceTileCnt;
        tilingM_.cutRes.shortTileAtBack = true;
        tilingM_.cutRes.longTileLen = tileLen;
        tilingM_.cutRes.numLongTile = forceTileCnt;
        tilingM_.cutRes.shortTileLen = 0U;
        tilingM_.cutRes.numShortTile = 0U;
        tilingM_.cutRes.totalTileCnt = forceTileCnt;
        OPS_LOG_D("Mc2FitBasedBalanceTiling",
                  "Force uniform split: tileCnt=%lu, tileLen=%lu, mValue=%lu",
                  forceTileCnt, tileLen, mmInfo_.mValue);
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
