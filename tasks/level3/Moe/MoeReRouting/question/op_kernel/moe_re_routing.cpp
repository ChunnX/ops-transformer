#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void moe_re_routing(GM_ADDR tokens, GM_ADDR expertTokensNumPerRank,
    GM_ADDR perTokenScales, GM_ADDR permuteTokens, GM_ADDR permutePerTokenScales, GM_ADDR permuteTokenIdx,
    GM_ADDR expertTokenNum, GM_ADDR workspace, GM_ADDR tiling)
{
}