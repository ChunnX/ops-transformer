/*!
 * \file apply_top_k_top_p_with_sorted_tiling.h
 * \brief
 * ATTENTION: MAKE SURE 'BEGIN_TILING_DATA_DEF' STAY IN THE SAME LINE (28) USING BLANK LINES.
 * 
 * 
 * 
 * 
 * 
 */
#ifndef __APPLY_TOP_K_TOP_P_WITH_SORTED_TILINGDATA_H__
#define __APPLY_TOP_K_TOP_P_WITH_SORTED_TILINGDATA_H__

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ApplyTopKTopPWithSortedTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, vocabSize);
    TILING_DATA_FIELD_DEF(uint32_t, batchPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, tailBatch);
    TILING_DATA_FIELD_DEF(uint32_t, blockNum);
    TILING_DATA_FIELD_DEF(uint32_t, dataNumInit);
    TILING_DATA_FIELD_DEF(uint32_t, dataNumInitAligned);
    TILING_DATA_FIELD_DEF(uint32_t, ubFactorElement);
    TILING_DATA_FIELD_DEF(uint32_t, ubFactorElementAligned);
    TILING_DATA_FIELD_DEF(uint32_t, tailUbFactorElement);
    TILING_DATA_FIELD_DEF(uint32_t, tailUbFactorElementAligned);
    TILING_DATA_FIELD_DEF(uint32_t, calUbSize);
    TILING_DATA_FIELD_DEF(uint32_t, iterateTimes);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ApplyTopKTopPWithSorted, ApplyTopKTopPWithSortedTilingData)

}  // namespace optiling
#endif  // __APPLY_TOP_K_TOP_P_WITH_SORTED_TILINGDATA_H__