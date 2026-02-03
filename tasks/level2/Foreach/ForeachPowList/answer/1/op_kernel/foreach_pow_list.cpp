/*!
 * \file foreach_pow_list.cpp
 * \brief
 */

#include "foreach_pow_list.h"

using namespace ForeachPow;

extern "C" __global__ __aicore__ void foreach_pow_list(
    GM_ADDR self, GM_ADDR exponent, GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    // foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachPowList<half> op;
        op.Init(self, exponent, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachPowList<float> op;
        op.Init(self, exponent, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        ForeachPowList<int> op;
        op.Init(self, exponent, outputs, userWS, &tilingData);
        op.Process();
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    } else if (TILING_KEY_IS(4)) {
        ForeachPowList<bfloat16_t> op;
        op.Init(self, exponent, outputs, userWS, &tilingData);
        op.Process();
#endif
    }
}