#ifndef INPLACE_SCATTER_ADD_TILING_H
#define INPLACE_SCATTER_ADD_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(InplaceScatterAddTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, M);
  TILING_DATA_FIELD_DEF(uint32_t, N);
  TILING_DATA_FIELD_DEF(uint32_t, K);
  TILING_DATA_FIELD_DEF(uint32_t, NAligned);
  TILING_DATA_FIELD_DEF(uint32_t, frontCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, frontCoreIndicesNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailCoreIndicesNum);
  TILING_DATA_FIELD_DEF(uint32_t, ubSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(InplaceScatterAdd, InplaceScatterAddTilingData)
}
#endif