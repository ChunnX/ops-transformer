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
 * \file mega_moe_tile_info.h
 * \brief TileInfo and shared state types for single-core GMM1/GMM2 modules.
 *        TileInfo replaces all blockIdx-derived values; TilingByTile replaces
 *        TilingByCore which internally calls GetBlockIdx()/GetBlockNum().
 */

#ifndef MEGA_MOE_MODULES_TILE_INFO_H
#define MEGA_MOE_MODULES_TILE_INFO_H

#include "kernel_operator.h"
#include "mega_moe_base.h"
#include "mega_moe_workspace_info.h"

namespace MegaMoeModules {

struct TileInfo {
    uint32_t tileIdx;
    uint32_t totalTiles;
    uint32_t aivTileIdx;
    uint32_t aivTotalTiles;
    uint32_t subTileIdx;
};

using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t,
                                    int64_t, int64_t, int64_t, int64_t, int64_t,
                                    int64_t, int64_t, int64_t, int64_t>;

struct ExpertLoopState {
    TupleShape problemShape;
    BlockOffset baseOffset;
    uint32_t expertBeforeCnt = 0;
};

__aicore__ inline void TilingByTile(int32_t totalLen, int32_t &tileLen,
    int32_t &tileOffset, uint32_t tileIdx, uint32_t totalTiles, int32_t align = ALIGN_32)
{
    int32_t lenPerTile = Ops::Base::CeilDiv(static_cast<uint32_t>(totalLen), static_cast<uint32_t>(totalTiles));
    int32_t lenPerTileAlign = Ops::Base::CeilAlign(static_cast<uint32_t>(lenPerTile), static_cast<uint32_t>(align));
    tileLen = lenPerTileAlign;
    tileOffset = tileIdx * lenPerTileAlign;
    if (tileOffset + tileLen >= totalLen) {
        tileLen = totalLen - tileOffset;
    }
    if (tileOffset >= totalLen) {
        tileLen = 0;
    }
}

} // namespace MegaMoeModules

#endif
