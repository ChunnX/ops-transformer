#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_SEGSUM_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_SEGSUM_TILING_H

#include "register/tilingdata_base.h"
#include "common/log.h"
#include "register/op_impl_registry.h"
#include "common/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

constexpr uint16_t MAX_CORE_CONT = 50;

struct SegsumCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSize = 0;
};

BEGIN_TILING_DATA_DEF(SegsumTilingData)

TILING_DATA_FIELD_DEF(int64_t, dataType);
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, batches);
TILING_DATA_FIELD_DEF(int64_t, tailDimSize);
TILING_DATA_FIELD_DEF(int64_t, slideSize);

TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_CONT, batchStart);
TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_CONT, batchEnd);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Segsum, SegsumTilingData)
} // namespace optiling
#endif