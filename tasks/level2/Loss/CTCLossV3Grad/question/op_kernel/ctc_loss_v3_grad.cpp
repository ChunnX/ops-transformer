#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void ctc_loss_v3_grad(
    GM_ADDR grad_out, GM_ADDR log_probs, GM_ADDR targets, GM_ADDR input_lengths, GM_ADDR target_lengths,
    GM_ADDR neg_log_likelihood, GM_ADDR log_alpha, GM_ADDR grad, GM_ADDR workspace, GM_ADDR tiling)
{
}