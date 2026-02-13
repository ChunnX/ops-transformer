/**
 * AddRmsNormDynamicQuantV2 - question template for code gen.
 * Kernel stub; actual impl is in answer/0/op_kernel.
 */
#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void add_rms_norm_dynamic_quant_v2(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smooth1, GM_ADDR smooth2,
    GM_ADDR y1, GM_ADDR y2, GM_ADDR y3, GM_ADDR y4, GM_ADDR x,
    GM_ADDR outScale1, GM_ADDR outScale2, GM_ADDR workspace, GM_ADDR tiling)
{
}
