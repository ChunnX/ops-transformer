#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "max_pool3d_with_argmax_big_kernel.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    using mask_type = uint8_t;
#else
    using mask_type = uint16_t;
#endif

constexpr uint32_t PAD_DISABLE = 0;
constexpr uint32_t PAD_ENABLE = 1;

extern "C" __global__ __aicore__ void max_pool3d_with_argmax_v2(
    GM_ADDR x, GM_ADDR y, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    if (TILING_KEY_IS(311110)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2BigKernelTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2BigKernelTilingData* __restrict tilingData = &tilingDataIn;
        MaxPool3DWithArgmaxBigKernel<float, float, false, mask_type> op;
        op.Init(x, y, indices, GetUserWorkspace(workspace), &pipe, tilingData, 0);
        op.Process();
    } else if (TILING_KEY_IS(311111)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2BigKernelTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2BigKernelTilingData* __restrict tilingData = &tilingDataIn;
        MaxPool3DWithArgmaxBigKernel<half, half, false, mask_type> op;
        op.Init(x, y, indices, GetUserWorkspace(workspace), &pipe, tilingData, 1);
        op.Process();
    } else if (TILING_KEY_IS(311112)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2BigKernelTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2BigKernelTilingData* __restrict tilingData = &tilingDataIn;
        MaxPool3DWithArgmaxBigKernel<bfloat16_t, float, false, mask_type> op;
        op.Init(x, y, indices, GetUserWorkspace(workspace), &pipe, tilingData, 2);
        op.Process();
#endif
    }
}