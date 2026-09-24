#ifndef GMM_LOCAL_EXP_TILING_H_
#define GMM_LOCAL_EXP_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GmmLocalExpWithZeroBaseParams)
  TILING_DATA_FIELD_DEF(uint32_t, groupNum);
  TILING_DATA_FIELD_DEF(uint32_t, N);
  TILING_DATA_FIELD_DEF(uint32_t, K);
  TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, trans_b);
  TILING_DATA_FIELD_DEF(uint32_t, singleM);
  TILING_DATA_FIELD_DEF(uint32_t, singleN);
  TILING_DATA_FIELD_DEF(uint32_t, expStartIdx);
  TILING_DATA_FIELD_DEF(uint32_t, expEndIdx);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GmmLocalExpWithZeroBaseParamsOp, GmmLocalExpWithZeroBaseParams)

BEGIN_TILING_DATA_DEF(GmmLocalExpWithZeroTilingData)
TILING_DATA_FIELD_DEF_STRUCT(GmmLocalExpWithZeroBaseParams, gmmBaseParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
TILING_DATA_FIELD_DEF(uint32_t, totalEleNum);
TILING_DATA_FIELD_DEF(uint32_t, eleNumPerCore);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GmmLocalExpWithZero, GmmLocalExpWithZeroTilingData)
}
#endif
