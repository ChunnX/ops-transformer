/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mega_moe_tile_info.h
 * \brief TileInfo: single-core identification passed explicitly to all module functions,
 *        replacing all blockIdx-derived values. Also provides TilingByTile to replace
 *        TilingByCore which internally calls GetBlockIdx()/GetBlockNum().
 */

#ifndef MEGA_MOE_TILE_INFO_H
#define MEGA_MOE_TILE_INFO_H

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
