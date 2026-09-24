#ifndef GMM_ADD_TILING_H_
#define GMM_ADD_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GmmAddBaseParams)
  TILING_DATA_FIELD_DEF(uint32_t, groupNum);
  TILING_DATA_FIELD_DEF(uint32_t, M);
  TILING_DATA_FIELD_DEF(uint32_t, N);
  TILING_DATA_FIELD_DEF(uint32_t, TotalK);
  TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, trans_a);
  TILING_DATA_FIELD_DEF(uint32_t, singleM);
  TILING_DATA_FIELD_DEF(uint32_t, singleN);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GmmAddBaseParamsOp, GmmAddBaseParams)

BEGIN_TILING_DATA_DEF(GmmAddTilingData)
  TILING_DATA_FIELD_DEF_STRUCT(GmmAddBaseParams, gmmBaseParams);
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GmmAdd, GmmAddTilingData)
}

#endif