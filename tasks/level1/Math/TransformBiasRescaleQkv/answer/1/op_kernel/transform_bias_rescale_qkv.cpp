/*!
 * \file transform_bias_rescale_qkv.cpp
 * \brief transform_bias_rescale_qkv kernel
 */

#include "transform_bias_rescale_qkv.h"

using namespace AscendC;

using namespace TransformBiasRescaleQkv;

extern "C" __global__ __aicore__ void transform_bias_rescale_qkv(
    GM_ADDR qkv, GM_ADDR qkv_bias, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    GM_ADDR userWs = nullptr;

#if __CCE_AICORE__ >= 220
    if (TILING_KEY_IS(1)) {
        TransformBiasRescaleQkvND<DTYPE_QKV> op;
        op.Init(qkv, qkv_bias, q, k, v, userWs, &tilingData);
        op.Process();
    }
#else
#endif
}