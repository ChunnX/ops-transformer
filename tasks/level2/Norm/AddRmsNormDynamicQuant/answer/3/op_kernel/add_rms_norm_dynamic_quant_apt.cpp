#include "add_rms_norm_dynamic_quant_cut_d_kernel.h"

using namespace AscendC;

#define INIT_AND_PROCESS_WORKSPACE                                                                                 \
    do {                                                                                                           \
        op.Init(x1, x2, gamma, smooathScale1, smooathScale2, y1, y2, x, scale1, scale2, usrWorkspace, tilingData); \
        op.Process();                                                                                              \
    } while (0)

extern "C" __global__ __aicore__ void add_rms_norm_dynamic_quant(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smooathScale1, GM_ADDR smooathScale2, GM_ADDR beta, GM_ADDR y1,
    GM_ADDR y2, GM_ADDR x, GM_ADDR scale1, GM_ADDR scale2, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(AddRmsNormDynamicQuantTilingData, tilingDataIn, tiling);
    const AddRmsNormDynamicQuantTilingData* __restrict tilingData = &tilingDataIn;
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    if (TILING_KEY_IS(3)) {
        KernelAddRmsNormDynamicQuantSliceD<DTYPE_X1, DTYPE_Y1, 3> op(&pipe);
        INIT_AND_PROCESS_WORKSPACE;
    }
}