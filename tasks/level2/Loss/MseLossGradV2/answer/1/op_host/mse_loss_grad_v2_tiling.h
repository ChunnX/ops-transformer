#ifndef MSE_LOSS_GRAD_V2_TILING_H
#define MSE_LOSS_GRAD_V2_TILING_H

#pragma once
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MseLossGradTilingData)
TILING_DATA_FIELD_DEF(float, cof);
TILING_DATA_FIELD_DEF(uint64_t, totalLength);
TILING_DATA_FIELD_DEF(uint64_t, tileNum);
TILING_DATA_FIELD_DEF(uint64_t, padLength);
TILING_DATA_FIELD_DEF(uint64_t, blockLength);
TILING_DATA_FIELD_DEF(uint32_t, usedDb);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MseLossGradV2, MseLossGradTilingData)

void GetTilingKey(const uint32_t dtypeKey, uint32_t& tilingKey);
} // namespace optiling
#endif