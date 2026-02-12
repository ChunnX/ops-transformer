#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#include "adaptive_max_pool3d_small_pool.h"

extern "C" __global__ __aicore__ void adaptive_max_pool3d(
    GM_ADDR x, GM_ADDR y, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling)
{
    __gm__ uint8_t* user = GetUserWorkspace(workspace);

    TPipe pipeBase;
    if (TILING_KEY_IS(320000UL)) {
        GET_TILING_DATA_WITH_STRUCT(AdaptiveMaxPool3dSmallPoolTilingData, tilingDataIn, tiling);
        const AdaptiveMaxPool3dSmallPoolTilingData* __restrict__ tilingData = &tilingDataIn;
        AdaptiveMaxPool3dSmallPool<float> op;
        op.Init(x, y, indices, user, &pipeBase, tilingData);
        op.Process();
        return;
    } else if (TILING_KEY_IS(321000UL)) {
        GET_TILING_DATA_WITH_STRUCT(AdaptiveMaxPool3dSmallPoolTilingData, tilingDataIn, tiling);
        const AdaptiveMaxPool3dSmallPoolTilingData* __restrict__ tilingData = &tilingDataIn;
        AdaptiveMaxPool3dSmallPool<half> op;
        op.Init(x, y, indices, user, &pipeBase, tilingData);
        op.Process();
        return;
    } else if (TILING_KEY_IS(322000UL)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        GET_TILING_DATA_WITH_STRUCT(AdaptiveMaxPool3dSmallPoolTilingData, tilingDataIn, tiling);
        const AdaptiveMaxPool3dSmallPoolTilingData* __restrict__ tilingData = &tilingDataIn;
        AdaptiveMaxPool3dSmallPool<bfloat16_t> op;
        op.Init(x, y, indices, user, &pipeBase, tilingData);
        op.Process();
        return;
#endif
    }

    return;
}