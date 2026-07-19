/**
 * @file gen_position_ids_from_mask_tiling.h
 */
#ifndef GEN_POSITION_IDS_FROM_MASK_TILING_H_
#define GEN_POSITION_IDS_FROM_MASK_TILING_H_

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GenPositionIdsFromMaskTilingData)
    TILING_DATA_FIELD_DEF(int64_t, b);                  // batch
    TILING_DATA_FIELD_DEF(int64_t, s);                  // seq len
    TILING_DATA_FIELD_DEF(int64_t, paddingFillValue);   // padding 填充值
    TILING_DATA_FIELD_DEF(int64_t, coreNum);            // 实际使用核数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GenPositionIdsFromMask, GenPositionIdsFromMaskTilingData)

}  // namespace optiling

#endif  // GEN_POSITION_IDS_FROM_MASK_TILING_H_