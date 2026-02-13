#include "angle_v2_complex.h"

using namespace AngleV2N;
#define KEY_DTYPE_COMPLEX64 1

extern "C" __global__ __aicore__ void angle_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(KEY_DTYPE_COMPLEX64)) {
        AngleV2N::AngleV2Complex<float> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    }
}