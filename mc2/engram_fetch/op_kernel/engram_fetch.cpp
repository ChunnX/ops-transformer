#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "engram_fetch_tiling_data.h"
#include "engram_fetch_tiling_key.h"
#include "arch35/engram_fetch_arch35.h"

using namespace AscendC;

template<uint32_t engramFetchMode>
__global__ __aicore__ void engram_fetch(GM_ADDR bufferHandle, GM_ADDR indices, GM_ADDR fetched, GM_ADDR workspaceGM,
                                        GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(EngramFetchTilingData);
    GET_TILING_DATA_WITH_STRUCT(EngramFetchTilingData, tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr (engramFetchMode == ENGRAM_FETCH_DEFAULT_MODE) {
        TPipe pipe;
        EngramFetchImpl::EngramFetchArch35 op;
        op.Init(bufferHandle, indices, fetched, workspaceGM, &pipe, &tilingData);
        op.Process();
    }
}