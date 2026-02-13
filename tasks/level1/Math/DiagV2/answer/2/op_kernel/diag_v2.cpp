#include "diag_v2_b16.h"

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
    if (TILING_KEY_IS(2102)) {
        DiagV2::DiagV2B16<int16_t> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    }
}