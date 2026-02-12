/*!
 * \file adaptive_avg_pool3d_tiling.h
 * \brief
 * ATTENTION: MAKE SURE 'BEGIN_TILING_DATA_DEF' STAY IN THE SAME LINE (28) USING BLANK LINES.
 */

#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ADAPTIVE_AVG_POOL3D_TILING_H_
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ADAPTIVE_AVG_POOL3D_TILING_H_

#include "common/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "common/tiling_base.h"
#include "common/tiling_templates_registry.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AdaptiveAvgPool3dTilingData)
TILING_DATA_FIELD_DEF(uint64_t, dimC);
TILING_DATA_FIELD_DEF(uint64_t, cTileLength);
TILING_DATA_FIELD_DEF(uint64_t, inD);
TILING_DATA_FIELD_DEF(uint64_t, inH);
TILING_DATA_FIELD_DEF(uint64_t, inW);
TILING_DATA_FIELD_DEF(uint64_t, outD);
TILING_DATA_FIELD_DEF(uint64_t, outH);
TILING_DATA_FIELD_DEF(uint64_t, outW);
TILING_DATA_FIELD_DEF(uint64_t, formerLength);
TILING_DATA_FIELD_DEF(uint64_t, formerNum);
TILING_DATA_FIELD_DEF(uint64_t, tailLength);
TILING_DATA_FIELD_DEF(uint64_t, tailNum);
TILING_DATA_FIELD_DEF(uint64_t, indexBufLen);
TILING_DATA_FIELD_DEF(uint64_t, windowWNum);
TILING_DATA_FIELD_DEF(uint64_t, maxWindowWLength);
TILING_DATA_FIELD_DEF(uint64_t, inputTileNum);
TILING_DATA_FIELD_DEF(uint64_t, atomicAddNum);
END_TILING_DATA_DEF;

struct AdaptiveAvgPool3dCompileInfo {
    int32_t totalCoreNum = 0;
    uint32_t sysWorkspaceSize = 0;
    uint64_t ubSizePlatForm = 0;
};

REGISTER_TILING_DATA_CLASS(AdaptiveAvgPool3d, AdaptiveAvgPool3dTilingData)
} // namespace optiling

#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ADAPTIVE_AVG_POOL3D_TILING_H_