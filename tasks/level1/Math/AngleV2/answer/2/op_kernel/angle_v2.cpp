#include "angle_v2.h"

using namespace AngleV2N;
#define KEY_DTYPE_FP32 2
#define KEY_DTYPE_FP16 3

extern "C" __global__ __aicore__ void angle_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(KEY_DTYPE_FP32)) {
        AngleV2N::AngleV2<float> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(KEY_DTYPE_FP16)) {
        AngleV2N::AngleV2<half> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    }
}