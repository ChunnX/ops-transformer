
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
// #include "mhc_pre_grad.h"
#include "mhc_pre_sinkhorn_backward_one_stage.h"

using namespace AscendC;
using namespace MicroAPI;

extern "C" __global__ __aicore__ void mhc_pre_sinkhorn_backward(GM_ADDR x, GM_ADDR phi, GM_ADDR h_pre, GM_ADDR grad_hin,
    GM_ADDR grad_h_post, GM_ADDR grad_h_res, GM_ADDR alpha, GM_ADDR bias, GM_ADDR hc_before_norm, GM_ADDR inv_rms, GM_ADDR sum_out,
    GM_ADDR norm_out, GM_ADDR grad_x, GM_ADDR grad_phi, GM_ADDR grad_alpha, GM_ADDR grad_bias, GM_ADDR workspace, GM_ADDR tiling)
{
    // GET_TILING_DATA(tilingData, tiling);
    // GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    // if (usrWorkspace == nullptr) {
    //     return;
    // }

    // TPipe pipe;

    // // MhcPreGradKernel<DTYPE_X> op;
    // MhcPreGradKernel<DTYPE_X, DTYPE_PHI> op;
    
    // op.mm1_.SetSubBlockIdx(0);
    // op.mm1_.Init(&tilingData.mm1TilingData, &pipe);
    // op.mm2_.SetSubBlockIdx(0);
    // op.mm2_.Init(&tilingData.mm2TilingData, &pipe);
    
// op.Init(x, phi, h_pre, grad_hin, grad_h_post, grad_h_res, alpha, bias, hc_before_norm, inv_rms, sum_out, norm_out,
//         grad_x, grad_phi, grad_alpha, grad_bias, usrWorkspace, &tilingData, &pipe);
    
    // KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
    
    // op.Process();
}