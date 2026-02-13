#include "diag_flat_nd_to_2d_b16_more64.h"

using namespace DiagFlat;

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
    if (TILING_KEY_IS(104)) {
        DiagFlat::DiagFlatND2To2DB16More64<int64_t> op;
        op.Init(x, y, workspace, &tilingData);
        op.Process();
    }
}