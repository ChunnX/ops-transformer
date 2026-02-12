/*!
 * \file foreach_pow_scalar_list.cpp
 * \brief
 */

#include "foreach_pow_scalar_list.h"

using namespace ForeachPowScalarList;

extern "C" __global__ __aicore__ void foreach_pow_scalar_list(
    GM_ADDR inputs, GM_ADDR scalar, GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    // foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachPowScalarListND<half> op;
        op.Init(inputs, scalar, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachPowScalarListND<float> op;
        op.Init(inputs, scalar, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        ForeachPowScalarListND<int> op;
        op.Init(inputs, scalar, outputs, userWS, &tilingData);
        op.Process();
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    } else if (TILING_KEY_IS(4)) {
        ForeachPowScalarListND<bfloat16_t> op;
        op.Init(inputs, scalar, outputs, userWS, &tilingData);
        op.Process();
#endif
    }
}