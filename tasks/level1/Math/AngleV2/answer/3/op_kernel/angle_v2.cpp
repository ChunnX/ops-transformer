#include "angle_v2_u8.h"

using namespace AngleV2N;
#define KEY_DTYPE_BOOL 4
#define KEY_DTYPE_UINT8 5

extern "C" __global__ __aicore__ void angle_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(KEY_DTYPE_BOOL)) {
        AngleV2N::AngleV2U8<float> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(KEY_DTYPE_UINT8)) {
        AngleV2N::AngleV2U8<float> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    }
}