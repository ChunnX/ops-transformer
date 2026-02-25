#include "group_norm_silu_g16.h"

using namespace GroupNormSilu;

extern "C" __global__ __aicore__ void group_norm_silu(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR silu, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_VECTOR_CORE);
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(1051)) {
        GroupNormSilu::GroupNormSiluG16<DTYPE_X, DTYPE_X> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1052)) {
        GroupNormSilu::GroupNormSiluG16<DTYPE_X, float> op;
        op.Init(x, gamma, beta, silu, mean, rstd, userWS, &tilingData);
        op.Process();
    }
}