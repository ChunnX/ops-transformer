/*!
 * \file foreach_lerp_list.cpp
 * \brief
 */

#include "foreach_lerp_list.h"

using namespace ForeachLerpList;

extern "C" __global__ __aicore__ void foreach_lerp_list(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    // foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachLerpListND<half> op;
        op.Init(x1, x2, weight, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachLerpListND<float> op;
        op.Init(x1, x2, weight, y, userWS, &tilingData);
        op.Process();
    }
#if __CCE_AICORE__ >= 220 && !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    else if (TILING_KEY_IS(4)) {
        ForeachLerpListND<bfloat16_t> op;
        op.Init(x1, x2, weight, y, userWS, &tilingData);
        op.Process();
    }
#endif
}