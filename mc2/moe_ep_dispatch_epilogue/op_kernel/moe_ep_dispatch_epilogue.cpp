#include "kernel_operator.h"

#include "moe_ep_dispatch_epilogue_tiling_key.h"
#include "moe_ep_dispatch_epilogue_tiling.h"
#include "moe_ep_dispatch_epilogue.h"

#if __has_include("arch35/moe_ep_dispatch_epilogue_arch35.h")
#include "arch35/moe_ep_dispatch_epilogue_arch35.h"
#endif

using namespace MoeEpDispatchEpilogueImpl;
using namespace Mc2Tiling;
using namespace AscendC;

template<uint32_t ArchTag, uint32_t IsCached>
__global__ __aicore__ void moe_ep_dispatch_epilogue(
    GM_ADDR context, GM_ADDR dstBufferSlotIdx,
    GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert, GM_ADDR cachedRecvSrcMetadata,
    GM_ADDR recvX, GM_ADDR recvTopkWeights,
    GM_ADDR recvSrcMetadata, GM_ADDR recvScales,
    GM_ADDR workspace, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeEpDispatchEpilogueTilingData);
    REGISTER_TILING_FOR_TILINGKEY("ArchTag == TILINGKEY_TPL_A5", MoeEpDispatchEpilogueTilingData);
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(MoeEpDispatchEpilogueTilingData, tilingData, tilingGM);

    KernelDispatchEpilogue<DTYPE_RECV_X, IsCached> op;
    op.Init(context, dstBufferSlotIdx,
            numRecvPerRank, numRecvPerExpert, cachedRecvSrcMetadata,
            recvX, recvTopkWeights,
            recvSrcMetadata, recvScales,
            workspace, tilingGM, &pipe, &tilingData.moeEpDispatchEpilogueInfo);
    op.Process();
}
