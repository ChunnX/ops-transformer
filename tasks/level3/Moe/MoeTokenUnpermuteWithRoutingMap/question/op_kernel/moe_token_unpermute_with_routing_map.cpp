#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void moe_token_unpermute_with_routing_map(GM_ADDR permuted_tokens, GM_ADDR sorted_indices, GM_ADDR routing_map,
                                                          GM_ADDR probs, GM_ADDR unpermuted_tokens, GM_ADDR out_index, GM_ADDR permute_token_id,
                                                          GM_ADDR permute_probs, GM_ADDR workspace, GM_ADDR tiling)
{
}
