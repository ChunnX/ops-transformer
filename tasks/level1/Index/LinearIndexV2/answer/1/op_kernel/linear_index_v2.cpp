/*!
 * \file linear_index_v2.cpp
 * \brief
 */
#include "linear_index_v2.h"

namespace AscendC {
extern "C" __global__ __aicore__ void linear_index_v2(
    GM_ADDR indexList, GM_ADDR stride, GM_ADDR valueSize, GM_ADDR output, GM_ADDR workSpace, GM_ADDR tiling)
{
    if (workSpace == nullptr) {
        return;
    }
    GM_ADDR user = AscendC::GetUserWorkspace(workSpace);
    if (user == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe pipe;
    if (TILING_KEY_IS(0)) {
        LinearIndexKernelV2<int64_t> op(indexList, stride, valueSize, output, workSpace, tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        LinearIndexKernelV2<int32_t> op(indexList, stride, valueSize, output, workSpace, tilingData, pipe);
        op.Process();
    }
}
} // namespace AscendC