/*!
 * \file pows.cpp
 * \brief pows kernel
 */

#include "pows_bf16.h"

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

    if (TILING_KEY_IS(201)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        Pows::PowsBfp16<bfloat16_t> op;
        op.Init(x1, x2, y, workspace, &tilingData);
        op.Process();
#endif
    }
}