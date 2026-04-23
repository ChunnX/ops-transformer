#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void moe_token_permute(
    GM_ADDR tokens, GM_ADDR indices, GM_ADDR permuteTokens, GM_ADDR sortedIndices, GM_ADDR workspace, GM_ADDR tiling)
{
}