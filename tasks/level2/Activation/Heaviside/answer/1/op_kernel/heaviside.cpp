/*!
 * \file heaviside.cpp
 * \brief heaviside kernel
 */

#include "heaviside.h"

using namespace AscendC;

using namespace Heaviside;

extern "C" __global__ __aicore__ void heaviside(
    GM_ADDR input, GM_ADDR values, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    GM_ADDR userWs = nullptr;

    if (TILING_KEY_IS(1)) {
        HeavisideND<DTYPE_INPUT> op;
        op.Init(input, values, output, userWs, &tilingData);
        op.Process();
    }
}