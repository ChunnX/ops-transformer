/*!
 * \file lin_space.cpp
 * \brief
 */

#include "lin_space_small_shape_and_cast.h"

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
    if (TILING_KEY_IS(303)) {
        LinSpace::LinSpaceSmallShapeAndCast<int8_t, half> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(403)) {
        LinSpace::LinSpaceSmallShapeAndCast<uint8_t, half> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(503)) {
        LinSpace::LinSpaceSmallShapeAndCast<int16_t, half> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(603)) {
        LinSpace::LinSpaceSmallShapeAndCast<int32_t, float> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    }
#if ORIG_DTYPE_START == DT_BF16
    else if (TILING_KEY_IS(703)) {
        LinSpace::LinSpaceSmallShapeAndCast<bfloat16_t, float> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    }
#endif
}