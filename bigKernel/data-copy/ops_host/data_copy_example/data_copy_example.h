#include "register/tilingdata_base.h"

namespace optiling{
BEGIN_TILING_DATA_DEF(DataCopyExampleTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, shapeT);
    TILING_DATA_FIELD_DEF(uint64_t, shapeN);
    TILING_DATA_FIELD_DEF(uint64_t, shapeD);
    TILING_DATA_FIELD_DEF(uint64_t, B); // seq个数，前缀和序列长度为B+1
    TILING_DATA_FIELD_DEF(uint64_t, shapeTAlign);
    TILING_DATA_FIELD_DEF(uint64_t, shapeNDAlign);
    TILING_DATA_FIELD_DEF(uint64_t, useCoreNum); // 完成总任务实际需要使用的核数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DataCopyExample, DataCopyExampleTilingData)
}