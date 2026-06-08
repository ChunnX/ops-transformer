#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "engram_fetch_tiling_data.h"

using namespace AscendC;

__global__ __aicore__ void engram_fetch(GM_ADDR bufferHandle, GM_ADDR indices, GM_ADDR fetched, GM_ADDR workspaceGM,
                                        GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(EngramFetchTilingData);
    GET_TILING_DATA_WITH_STRUCT(EngramFetchTilingData, tilingData, tilingGM);
    printf("enter kernel\n");
    if (TILING_KEY_IS(0)) {
        printf("EngramFetch Tiling A5.\n");
    }
}