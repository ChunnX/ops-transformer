#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void moe_token_unpermute_with_ep_grad(
    GM_ADDR unpermuted_tokens_grad, GM_ADDR sorted_indices, GM_ADDR permuted_tokens, GM_ADDR probs,
    GM_ADDR permuted_tokens_grad, GM_ADDR probs_grad, GM_ADDR workspace, GM_ADDR tiling)
{
}
