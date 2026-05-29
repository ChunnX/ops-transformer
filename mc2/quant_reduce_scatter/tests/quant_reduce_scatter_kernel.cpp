/**
 * Standalone kernel wrapper for quant_reduce_scatter.
 * Uses TBuf+DataCopy to copy tiling data from GM to local memory.
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "quant_reduce_scatter_tiling_data.h"

#if defined(__NPU_ARCH__)
#include "quant_reduce_scatter_mte.h"
#include "profiler.h"

using namespace AscendC;
using namespace QuantReduceScatterImpl;

__aicore__ inline void InitTilingData(const __gm__ uint8_t *tiling, QuantReduceScatterTilingData *constData)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)constData;
    for (size_t i = 0; i < sizeof(QuantReduceScatterTilingData) / 4; i++) {
        *(dst + i) = *(src + i);
    }
}

extern "C" __global__ __aicore__ void QuantReduceScatter_Standalone(
    GM_ADDR x, GM_ADDR scales, GM_ADDR output,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    AscendC::SetHcclContext<0>(workspaceGM);

    QuantReduceScatterTilingData tilingData;
    InitTilingData(tilingGM, &tilingData);

    GM_ADDR profBufGm = (GM_ADDR)tilingData.quantReduceScatterTilingInfo.profBufAddr;

    PROF_INIT(profBufGm);
    PROF_MARK(profBufGm, EVT_KERNEL_BEGIN, 0);

    TPipe pipe;
    QuantReduceScatterMte<int8_t, float, bfloat16_t> op;
    op.Init(x, scales, output, &pipe, &tilingData);
    op.Process(profBufGm);

    PROF_MARK(profBufGm, EVT_KERNEL_END, 0);
}
#endif

extern "C" void quant_reduce_scatter_demo(uint32_t blockDim, void* stream,
    uint8_t* x, uint8_t* scales, uint8_t* output,
    uint8_t* workspaceGM, uint8_t* tilingGM)
{
    QuantReduceScatter_Standalone<<<blockDim, nullptr, stream>>>(
        x, scales, output, workspaceGM, tilingGM);
}
