/*!
 * \file group_norm_grad.cpp
 * \brief
 */
#include "group_norm_grad.h"

extern "C" __global__ __aicore__ void group_norm_grad(
    GM_ADDR dy, GM_ADDR mean, GM_ADDR rstd, GM_ADDR x, GM_ADDR gamma, GM_ADDR dx, GM_ADDR dgamma, GM_ADDR dbeta,
    GM_ADDR workspace, GM_ADDR tilingdata)
{
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    GET_TILING_DATA(tiling_data, tilingdata);
    if (TILING_KEY_IS(0)) {
        GroupNormGrad<float, false> opFP32(dy, mean, rstd, x, gamma, dx, dgamma, dbeta, usrWorkspace, &tiling_data);
        opFP32.Process();
    } else if (TILING_KEY_IS(1)) {
        GroupNormGrad<half, false> opFP16(dy, mean, rstd, x, gamma, dx, dgamma, dbeta, usrWorkspace, &tiling_data);
        opFP16.Process();
    } else if (TILING_KEY_IS(2)) {
        GroupNormGrad<bfloat16_t, false> opBF16(
            dy, mean, rstd, x, gamma, dx, dgamma, dbeta, usrWorkspace, &tiling_data);
        opBF16.Process();
    } else if (TILING_KEY_IS(10)) {
        GroupNormGrad<float, true> opFP32Det(dy, mean, rstd, x, gamma, dx, dgamma, dbeta, usrWorkspace, &tiling_data);
        opFP32Det.Process();
    } else if (TILING_KEY_IS(11)) {
        GroupNormGrad<half, true> opFP16Det(dy, mean, rstd, x, gamma, dx, dgamma, dbeta, usrWorkspace, &tiling_data);
        opFP16Det.Process();
    } else if (TILING_KEY_IS(12)) {
        GroupNormGrad<bfloat16_t, true> opBF16Det(
            dy, mean, rstd, x, gamma, dx, dgamma, dbeta, usrWorkspace, &tiling_data);
        opBF16Det.Process();
    }
}