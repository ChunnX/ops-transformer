#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "max_pool3d_with_argmax_v2_no_expand_indices.h"

constexpr uint32_t PAD_DISABLE = 0;
constexpr uint32_t PAD_ENABLE = 1;

extern "C" __global__ __aicore__ void max_pool3d_with_argmax_v2(
    GM_ADDR x, GM_ADDR y, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    if (TILING_KEY_IS(300001UL)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2NoExpandIndicesTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2NoExpandIndicesTilingData* __restrict tilingData = &tilingDataIn;
#if ORIG_DTYPE_X == DT_BF16
        MaxPool3DWithArgmaxV2NoExpandIndices<DTYPE_X, float, PAD_DISABLE> op;
#else
        MaxPool3DWithArgmaxV2NoExpandIndices<DTYPE_X, DTYPE_X, PAD_DISABLE> op;
#endif
        op.Init(x, y, indices, &pipe, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(300002UL)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPool3DWithArgmaxV2NoExpandIndicesTilingData, tilingDataIn, tiling);
        const MaxPool3DWithArgmaxV2NoExpandIndicesTilingData* __restrict tilingData = &tilingDataIn;
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003) && ORIG_DTYPE_X == DT_BF16
        MaxPool3DWithArgmaxV2NoExpandIndices<DTYPE_X, float, PAD_ENABLE> op;
#else
        MaxPool3DWithArgmaxV2NoExpandIndices<DTYPE_X, DTYPE_X, PAD_ENABLE> op;
#endif
        op.Init(x, y, indices, &pipe, tilingData);
        op.Process();
    }
}