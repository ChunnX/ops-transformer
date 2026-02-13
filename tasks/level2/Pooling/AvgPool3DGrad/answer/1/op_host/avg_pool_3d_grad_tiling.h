#ifndef AVG_POOL_3D_GRAD_TILING_H
#define AVG_POOL_3D_GRAD_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AvgPool3dGradTilingAttrParam)
TILING_DATA_FIELD_DEF(uint64_t, N)
TILING_DATA_FIELD_DEF(uint64_t, C)
TILING_DATA_FIELD_DEF(uint64_t, outN)
TILING_DATA_FIELD_DEF(uint64_t, outD)
TILING_DATA_FIELD_DEF(uint64_t, outH)
TILING_DATA_FIELD_DEF(uint64_t, outW)
TILING_DATA_FIELD_DEF(uint64_t, inN)
TILING_DATA_FIELD_DEF(uint64_t, inD)
TILING_DATA_FIELD_DEF(uint64_t, inH)
TILING_DATA_FIELD_DEF(uint64_t, inW)
TILING_DATA_FIELD_DEF(uint64_t, kD)
TILING_DATA_FIELD_DEF(uint64_t, kH)
TILING_DATA_FIELD_DEF(uint64_t, kW)
TILING_DATA_FIELD_DEF(uint64_t, dD)
TILING_DATA_FIELD_DEF(uint64_t, dH)
TILING_DATA_FIELD_DEF(uint64_t, dW)
TILING_DATA_FIELD_DEF(uint64_t, padD)
TILING_DATA_FIELD_DEF(uint64_t, padH)
TILING_DATA_FIELD_DEF(uint64_t, padW)
TILING_DATA_FIELD_DEF(uint64_t, countIncludePad)
TILING_DATA_FIELD_DEF(int64_t, divisorOverride)
TILING_DATA_FIELD_DEF(uint64_t, isOverLap)
TILING_DATA_FIELD_DEF(uint64_t, isDetermine)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(AvgPool3dGradTilingAttrParamOp, AvgPool3dGradTilingAttrParam)

BEGIN_TILING_DATA_DEF(AvgPool3dGradTilingCParam)
TILING_DATA_FIELD_DEF(uint64_t, normalCoreCNum)
TILING_DATA_FIELD_DEF(uint64_t, lastCoreCNum)
TILING_DATA_FIELD_DEF(uint64_t, cAlign)
TILING_DATA_FIELD_DEF(uint64_t, cTotal)
TILING_DATA_FIELD_DEF(uint64_t, cCount)
TILING_DATA_FIELD_DEF(uint64_t, cNum)
TILING_DATA_FIELD_DEF(uint64_t, cLine)
TILING_DATA_FIELD_DEF(uint64_t, cTail)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(AvgPool3dGradTilingCParamOp, AvgPool3dGradTilingCParam)

BEGIN_TILING_DATA_DEF(AvgPool3dGradTilingCastCopy)
TILING_DATA_FIELD_DEF(uint64_t, maxDataNumInUb)
TILING_DATA_FIELD_DEF(uint64_t, normalCoreNum)
TILING_DATA_FIELD_DEF(uint64_t, tailCoreNum)
TILING_DATA_FIELD_DEF(uint64_t, normalCoreDataNum)
TILING_DATA_FIELD_DEF(uint64_t, tailCoreDataNum)
TILING_DATA_FIELD_DEF(uint64_t, normalCoreFormerCopyTime)
TILING_DATA_FIELD_DEF(uint64_t, normalCoreTailCopyTime)
TILING_DATA_FIELD_DEF(uint64_t, normalCoreFormerDataNum)
TILING_DATA_FIELD_DEF(uint64_t, normalCoreTailDataNum)
TILING_DATA_FIELD_DEF(uint64_t, tailCoreFormerCopyTime)
TILING_DATA_FIELD_DEF(uint64_t, tailCoreTailCopyTime)
TILING_DATA_FIELD_DEF(uint64_t, tailCoreFormerDataNum)
TILING_DATA_FIELD_DEF(uint64_t, tailCoreTailDataNum)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(AvgPool3dGradTilingCastCopyOp, AvgPool3dGradTilingCastCopy)

BEGIN_TILING_DATA_DEF(AvgPool3dGradTilingParam)
TILING_DATA_FIELD_DEF_STRUCT(AvgPool3dGradTilingAttrParam, attrParam)
TILING_DATA_FIELD_DEF_STRUCT(AvgPool3dGradTilingCParam, cParam)
TILING_DATA_FIELD_DEF_STRUCT(AvgPool3dGradTilingCastCopy, castCopyParam)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(AvgPool3DGrad, AvgPool3dGradTilingParam)

struct AvgPool3dGradCompileInfo {
};
} // namespace optiling
#endif // AVG_POOL_3D_GRAD_TILING_H