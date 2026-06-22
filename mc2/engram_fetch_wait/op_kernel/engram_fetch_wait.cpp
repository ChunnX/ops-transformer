#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "engram_fetch_wait_tiling_data.h"
#include "arch35/engram_fetch_wait_arch35.h"

using namespace AscendC;

__global__ __aicore__ void engram_fetch_wait(GM_ADDR bufferHandle, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(EngramFetchWaitTilingData);
    GET_TILING_DATA_WITH_STRUCT(EngramFetchWaitTilingData, tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(0)) {
        TPipe pipe;
        EngramFetchWaitImpl::EngramFetchWaitArch35 op;
        op.Init(bufferHandle, workspaceGM, &pipe, &tilingData);
        op.Process();
    }
}
