#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "moe_ep_dispatch_epilogue_tiling_key.h"
#include "moe_ep_dispatch_epilogue_tiling_data.h"
#include "moe_ep_dispatch_epilogue.h"

#if __has_include("arch22/moe_ep_dispatch_epilogue_arch22.h")
#include "arch22/moe_ep_dispatch_epilogue_arch22.h"
#endif
#if __has_include("arch35/moe_ep_dispatch_epilogue_arch35.h")
#include "arch35/moe_ep_dispatch_epilogue_arch35.h"
#endif

using namespace MoeEpDispatchEpilogueImpl;
using namespace Mc2Tiling;
using namespace AscendC;

extern "C" {

__global__ __aicore__ void moe_ep_dispatch_epilogue(
    GM_ADDR dstBufferSlotIdx, GM_ADDR recvSrcMetadataInternal,
    GM_ADDR numRecvPerRankHp, GM_ADDR numRecvPerExpertHp,
    GM_ADDR recvX, GM_ADDR recvTopkIdx, GM_ADDR recvTopkWeights,
    GM_ADDR recvSrcMetadata, GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert,
    GM_ADDR workspace, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeEpDispatchEpilogueA2TilingData);
    REGISTER_TILING_FOR_TILINGKEY("ArchTag == TILINGKEY_TPL_A5", MoeEpDispatchEpilogueTilingData);
    REGISTER_TILING_FOR_TILINGKEY("ArchTag == TILINGKEY_TPL_A3", MoeEpDispatchEpilogueTilingData);
    REGISTER_TILING_FOR_TILINGKEY("ArchTag == TILINGKEY_TPL_A2", MoeEpDispatchEpilogueA2TilingData);

    TPipe pipe;

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    GET_TILING_DATA_WITH_STRUCT(MoeEpDispatchEpilogueTilingData, tilingData, tilingGM);
    KernelDispatchEpilogue op;
    op.Init(dstBufferSlotIdx, recvSrcMetadataInternal,
            numRecvPerRankHp, numRecvPerExpertHp,
            recvX, recvTopkIdx, recvTopkWeights,
            recvSrcMetadata, numRecvPerRank, numRecvPerExpert,
            workspace, tilingGM, &pipe, &tilingData.moeEpDispatchEpilogueInfo);
    op.Process();
#else
    GET_TILING_DATA_WITH_STRUCT(MoeEpDispatchEpilogueA2TilingData, tilingData, tilingGM);
    KernelDispatchEpilogue op;
    op.Init(dstBufferSlotIdx, recvSrcMetadataInternal,
            numRecvPerRankHp, numRecvPerExpertHp,
            recvX, recvTopkIdx, recvTopkWeights,
            recvSrcMetadata, numRecvPerRank, numRecvPerExpert,
            workspace, tilingGM, &pipe, &tilingData.moeEpDispatchEpilogueInfo);
    op.Process();
#endif
}

}  // extern "C"