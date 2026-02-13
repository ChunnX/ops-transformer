#include "diag_flat_nd_to_2d_with_few.h"

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
    if (TILING_KEY_IS(101)) {
        // dtype is complex32/uint32/int32/float32
        if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
            DiagFlat::DiagFlatNDTo2DWithFew<int32_t> op;
            op.Init(x, y, workspace, &tilingData);
            op.Process();
            // dtype is complex64/uint64/int64/float64
        } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
            DiagFlat::DiagFlatNDTo2DWithFew<int64_t> op;
            op.Init(x, y, workspace, &tilingData);
            op.Process();
            // dtype is int16/uin16/float16
        } else {
            DiagFlat::DiagFlatNDTo2DWithFew<DTYPE_X> op;
            op.Init(x, y, workspace, &tilingData);
            op.Process();
        }
    }
}