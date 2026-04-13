#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void moe_token_unpermute(
    GM_ADDR permuted_tokens, GM_ADDR sorted_indices, GM_ADDR probs, GM_ADDR unpermuted_tokens, GM_ADDR workspace,
    GM_ADDR tiling)
{
}
