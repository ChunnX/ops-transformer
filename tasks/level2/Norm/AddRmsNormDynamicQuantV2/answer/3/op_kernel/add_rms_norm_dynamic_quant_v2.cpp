/*!
 * \file add_rms_norm_dynamic_quant_v2.cpp
 * \brief
 */
#include "add_rms_norm_dynamic_quant_v2_cut_d_kernel.h"

#define INIT_AND_PROCESS                                                                                              \
    do {                                                                                                              \
        op.Init(x1, x2, gamma, smooth1, smooth2, y1, y2, y3, y4, x, outScale1, outScale2, usrWorkspace, &tilingData); \
        op.Process();                                                                                                 \
    } while (0)

extern "C" __global__ __aicore__ void add_rms_norm_dynamic_quant_v2(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smooth1, GM_ADDR smooth2, GM_ADDR y1, GM_ADDR y2, GM_ADDR y3,
    GM_ADDR y4, GM_ADDR x, GM_ADDR outScale1, GM_ADDR outScale2, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

    KernelAddRmsNormDynamicQuantV2SliceD<DTYPE_X1, 3> op(&pipe);
    INIT_AND_PROCESS;
}