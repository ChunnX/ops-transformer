/*!
 * \file embedding_bag_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_V2_EMBEDDING_BAG_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_V2_EMBEDDING_BAG_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(EmbeddingBagTilingData)
TILING_DATA_FIELD_DEF(int64_t, formerOffsetNum);
TILING_DATA_FIELD_DEF(int64_t, tailOffsetNum);
TILING_DATA_FIELD_DEF(int64_t, numEmbeddings);
TILING_DATA_FIELD_DEF(int64_t, computeRepTime);
TILING_DATA_FIELD_DEF(int64_t, indicesMaxMoveLength);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, paddingIdx);
TILING_DATA_FIELD_DEF(int64_t, numIndices);
TILING_DATA_FIELD_DEF(int64_t, mode);
TILING_DATA_FIELD_DEF(int64_t, tilingKey);
TILING_DATA_FIELD_DEF(int64_t, hasPerSampleWeights);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(EmbeddingBag, EmbeddingBagTilingData)

ge::graphStatus TilingEmbeddingBag(gert::TilingContext* context);
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_V2_EMBEDDING_BAG_H