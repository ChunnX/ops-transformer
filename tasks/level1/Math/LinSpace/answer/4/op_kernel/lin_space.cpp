/*!
 * \file lin_space.cpp
 * \brief
 */

#include "lin_space_need_cast.h"

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
    if (TILING_KEY_IS(301)) {
        LinSpace::LinSpaceNeedCast<int8_t, half> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(401)) {
        LinSpace::LinSpaceNeedCast<uint8_t, half> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(501)) {
        LinSpace::LinSpaceNeedCast<int16_t, half> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(601)) {
        LinSpace::LinSpaceNeedCast<int32_t, float> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    }
#if ORIG_DTYPE_START == DT_BF16
    else if (TILING_KEY_IS(701)) {
        LinSpace::LinSpaceNeedCast<bfloat16_t, float> op;
        op.Init(start, stop, num, output, userWS, &tilingData);
        op.Process();
    }
#endif
}