/*!
 * \file hans_decode.cpp
 * \brief
 */
#include "hans_decode_base.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void hans_decode(
    GM_ADDR mantissa, GM_ADDR fixed, GM_ADDR var, GM_ADDR pdf, GM_ADDR recover, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    SetSysWorkspace(workspace);
    TPipe pipein;
    HansDecodeNS::HansDecodeInitConfig config = {mantissa, fixed, var, pdf, recover, &tilingData};
#if ORIG_DTYPE_MANTISSA != DT_FLOAT
    if (TILING_KEY_IS(2)) {
#ifdef __DAV_C220_VEC__
        HansDecodeNS::HansDecode<true> op;
        op.Init(&pipein, config);
        op.Process();
#endif
    }
#else
    if (TILING_KEY_IS(4)) {
#ifdef __DAV_C220_VEC__
        HansDecodeNS::HansDecode<false> op;
        op.Init(&pipein, config);
        op.Process();
#endif
    }
#endif
}