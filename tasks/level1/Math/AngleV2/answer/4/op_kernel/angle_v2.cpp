#include "angle_v2_int.h"

using namespace AngleV2N;
#define KEY_DTYPE_INT8 6
#define KEY_DTYPE_INT16 7
#define KEY_DTYPE_INT32 8
#define KEY_DTYPE_INT64 9

extern "C" __global__ __aicore__ void angle_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(KEY_DTYPE_INT8)) {
        AngleV2N::AngleV2Int<int8_t, float> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(KEY_DTYPE_INT16)) {
        AngleV2N::AngleV2Int<int16_t, float> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(KEY_DTYPE_INT32)) {
        AngleV2N::AngleV2Int<int32_t, float> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(KEY_DTYPE_INT64)) {
        AngleV2N::AngleV2Int<int64_t, float> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    }
}