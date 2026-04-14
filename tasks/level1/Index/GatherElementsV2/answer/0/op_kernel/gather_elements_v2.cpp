#include "gather_elements_v2_scalar.h"
#include "gather_elements_v2_transpose.h"
#include "gather_elements_v2_last_dim.h"

extern "C" __global__ __aicore__ void gather_elements_v2(
    GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR user = AscendC::GetUserWorkspace(workspace);
    if (user == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe tpipe;
    if (TILING_KEY_IS(0)) {
        AscendC::GatherElementsV2ScalarKernel<DTYPE_X, int32_t> op(x, index, y, workspace, tilingData, tpipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        AscendC::GatherElementsV2TransposeKernel<DTYPE_X, int32_t> op(x, index, y, workspace, tilingData, tpipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        if constexpr (
            std::is_same_v<DTYPE_X, bfloat16_t> || std::is_same_v<DTYPE_X, half> || std::is_same_v<DTYPE_X, int16_t>) {
            AscendC::GatherElementsV2LastDim<half, int32_t> op(x, index, y, tilingData, &tpipe);
            op.Process();
        } else if constexpr (std::is_same_v<DTYPE_X, float> || std::is_same_v<DTYPE_X, int32_t>) {
            AscendC::GatherElementsV2LastDim<float, int32_t> op(x, index, y, tilingData, &tpipe);
            op.Process();
        }
    }
}
