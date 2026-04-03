
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
// #include "mhc_pre_grad.h"
#include "mhc_pre_grad_one_stage.h"

using namespace AscendC;
using namespace MicroAPI;

extern "C" __global__ __aicore__ void mhc_pre_grad(GM_ADDR x, GM_ADDR hc_fn, GM_ADDR pre, GM_ADDR grad_y,
    GM_ADDR grad_post, GM_ADDR grad_res, GM_ADDR hc_scale, GM_ADDR hc_base, GM_ADDR hc_before_norm, GM_ADDR rsqrt, GM_ADDR sk_colSum,
    GM_ADDR sk_colNorm, GM_ADDR sk_rowSum, GM_ADDR sk_rowNorm, GM_ADDR grad_x, GM_ADDR grad_hc_fn, GM_ADDR grad_hc_scale, GM_ADDR grad_hc_base, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    if (usrWorkspace == nullptr) {
        return;
    }

    TPipe pipe;

    // MhcPreGradKernel<DTYPE_X> op;
    MhcPreGradOneStageKernel<DTYPE_X> op;
    
    op.mm1_.SetSubBlockIdx(0);
    op.mm1_.Init(&tilingData.mm1TilingData, &pipe);
    op.mm2_.SetSubBlockIdx(0);
    op.mm2_.Init(&tilingData.mm2TilingData, &pipe);
    
    op.Init(x, hc_fn, pre, grad_y, grad_post, grad_res, hc_scale, hc_base, hc_before_norm, rsqrt, sk_colSum, sk_colNorm,
        sk_rowSum, sk_rowNorm, grad_x, grad_hc_fn, grad_hc_scale, grad_hc_base, usrWorkspace, &tilingData, &pipe);
    
    KERNEL_TASK_TYPE(0, KERNEL_TYPE_MIX_AIC_1_2);
    
    op.Process();
}