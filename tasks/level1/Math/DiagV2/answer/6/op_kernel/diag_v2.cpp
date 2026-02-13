#include "diag_v2_t4.h"

using namespace DiagV2;

extern "C" __global__ __aicore__ void diag_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(2401)) {
        DiagV2::DiagV2T4<int8_t> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2402)) {
        DiagV2::DiagV2T4<int16_t> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2403)) {
        DiagV2::DiagV2T4<int32_t> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2404)) {
        DiagV2::DiagV2T4<int64_t> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    }
}