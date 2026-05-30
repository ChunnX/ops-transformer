#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "moe_ep_dispatch_tiling_key.h"
#include "moe_ep_dispatch_tiling_data.h"
#include "moe_ep_dispatch.h"

#if __has_include("arch35/moe_ep_dispatch_arch35.h")
#include "arch35/moe_ep_dispatch_arch35.h"
#endif

using namespace MoeEpDispatchImpl;
using namespace Mc2Tiling;
using namespace AscendC;

extern "C" {

template<uint32_t DO_CPU_SYNC, uint32_t IS_CACHED, uint32_t NETWORK_MODE>
__global__ __aicore__ void moe_ep_dispatch(
    GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights,
    GM_ADDR cachedNumPerExpert, GM_ADDR cachedSlotIdx, GM_ADDR cachedSrcMeta,
    GM_ADDR numRecvPerRankHp, GM_ADDR numRecvPerExpertHp,
    GM_ADDR dstBufferSlotIdx, GM_ADDR recvSrcMetadataInternal,
    GM_ADDR workspace, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeEpDispatchTilingData);
    REGISTER_TILING_FOR_TILINGKEY("ArchTag == TILINGKEY_TPL_A5", MoeEpDispatchTilingData);

    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(MoeEpDispatchTilingData, tilingData, tilingGM);
    KernelMoeEpDispatch<DO_CPU_SYNC, IS_CACHED, NETWORK_MODE> op;
    op.Init(x, topkIdx, topkWeights,
            cachedNumPerExpert, cachedSlotIdx, cachedSrcMeta,
            numRecvPerRankHp, numRecvPerExpertHp,
            dstBufferSlotIdx, recvSrcMetadataInternal,
            workspace, tilingGM, &pipe, &tilingData.moeEpDispatchInfo);
    op.Process();
}

}  // extern "C"