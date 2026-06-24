#include "./arch35/causal_conv1d_decode_cut_bh.h"

#define TILING_KEY_CCDC_BH_BF16 20000

extern "C" __global__ __aicore__ void causal_conv1d_decode(GM_ADDR x, GM_ADDR weight, GM_ADDR convStates,
                                                            GM_ADDR queryStartLoc, GM_ADDR cacheIndices,
                                                            GM_ADDR bias, GM_ADDR numAcceptedTokens, GM_ADDR y,
                                                            GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(CausalConv1dDecodeCutBHTilingData);
    TPipe pipe;
    if (TILING_KEY_IS(TILING_KEY_CCDC_BH_BF16)) {
        GET_TILING_DATA_WITH_STRUCT(CausalConv1dDecodeCutBHTilingData, tilingData, tiling);
        CausalConv1dDecodeCutBH op(&pipe);
        op.Init(x, weight, convStates, queryStartLoc, cacheIndices, bias, numAcceptedTokens, y, &tilingData);
        op.Process();
    }
}
