#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "max_pool3d_with_argmax_v2_splitd.h"

constexpr uint32_t PAD_DISABLE = 0;
constexpr uint32_t PAD_ENABLE = 1;

extern "C" __global__ __aicore__ void max_pool3d_with_argmax_v2(
    GM_ADDR x, GM_ADDR y, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    if (TILING_KEY_IS(110000)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2SplitDTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2SplitDTilingData* __restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2SplitD<float, float> op(tilingData);
        op.Init(x, y, indices, nullptr, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(110001)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2SplitDTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2SplitDTilingData* __restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2SplitD<half, half> op(tilingData);
        op.Init(x, y, indices, nullptr, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(110002)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2SplitDTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2SplitDTilingData* __restrict tilingData = &tilingDataIn;
        KernelMaxPool3DWithArgmaxV2SplitD<float, bfloat16_t> op(tilingData);
        op.Init(x, y, indices, nullptr, &pipe);
        op.Process();
#endif
    }
}