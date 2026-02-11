/*!
 * \file lin_space.cpp
 * \brief
 */

#include "lin_space_small_shape.h"

using namespace LinSpace;

extern "C" __global__ __aicore__ void lin_space(
    GM_ADDR start, GM_ADDR stop, GM_ADDR num, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(103)) {
        LinSpace::LinSpaceSmallShape<float> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(203)) {
        LinSpace::LinSpaceSmallShape<half> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    }
}