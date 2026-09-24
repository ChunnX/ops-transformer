#ifndef GMM_KDIM_TILING_H_
#define GMM_KDIM_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GmmKDimBaseParams)
  TILING_DATA_FIELD_DEF(uint32_t, groupNum);
  TILING_DATA_FIELD_DEF(uint32_t, M);
  TILING_DATA_FIELD_DEF(uint32_t, N);
  TILING_DATA_FIELD_DEF(uint32_t, TotalK);
  TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, trans_a);
  TILING_DATA_FIELD_DEF(uint32_t, singleM);
  TILING_DATA_FIELD_DEF(uint32_t, singleN);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GmmKDimBaseParamsOp, GmmKDimBaseParams)

BEGIN_TILING_DATA_DEF(GmmKDimTilingData)
  TILING_DATA_FIELD_DEF_STRUCT(GmmKDimBaseParams, gmmBaseParams);
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GmmKDim, GmmKDimTilingData)
}

#endif