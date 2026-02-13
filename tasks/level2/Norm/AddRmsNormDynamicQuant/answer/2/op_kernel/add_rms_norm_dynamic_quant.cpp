#include "add_rms_norm_dynamic_quant_single_row_kernel.h"

extern "C" __global__ __aicore__ void add_rms_norm_dynamic_quant(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smooth1, GM_ADDR smooth2, GM_ADDR beta, GM_ADDR y1, GM_ADDR y2,
    GM_ADDR x, GM_ADDR outScale1, GM_ADDR outScale2, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

#define INIT_AND_PROCESS                                                                                        \
    op.Init(x1, x2, gamma, smooth1, smooth2, beta, y1, y2, x, outScale1, outScale2, usrWorkspace, &tilingData); \
    op.Process()
    if (TILING_KEY_IS(2)) {
        KernelAddRmsNormDynamicQuantSingleRow<DTYPE_X1, DTYPE_Y1, 2> op(&pipe);
        INIT_AND_PROCESS;
    }
}