#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "engram_fetch_tiling_key.h"
#include "engram_fetch.h"
#include "arch35/engram_fetch_arch35.h"

using namespace EngramFetchImpl;
using namespace AscendC;
using namespace Mc2Tiling;

template <uint8_t ArchTag>
__global__ __aicore__ void engram_fetch(GM_ADDR bufferhandle, GM_ADDR indices, GM_ADDR fetched, GM_ADDR workspaceGM,
                                        GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(EngramFetchTilingData);

    if constexpr (ArchTag == TILINGKEY_TPL_A5) {
        GET_TILING_DATA_WITH_STRUCT(EngramFetchTilingData, tilingData, tilingGM);
        TPipe pipe;
        EngramFetchArch35<bfloat16_t> op;
        op.Init(bufferhandle, indices, fetched, workspaceGM, &pipe, &tilingData);
        op.Process();
    }
}