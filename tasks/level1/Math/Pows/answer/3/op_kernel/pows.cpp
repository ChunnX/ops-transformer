/*!
 * \file pows.cpp
 * \brief pows kernel
 */

#include "pows_fp32.h"

using namespace Pows;

extern "C" __global__ __aicore__ void pows(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
#endif

    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(301)) {
        Pows::PowsFp32<float> op;
        op.Init(x1, x2, y, workspace, &tilingData);
        op.Process();
    }
}