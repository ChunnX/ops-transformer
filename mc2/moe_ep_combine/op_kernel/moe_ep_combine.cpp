#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "moe_ep_combine_tiling_key.h"
#include "moe_ep_combine_tiling_data.h"
#include "moe_ep_combine.h"

#if __has_include("arch35/moe_ep_combine_arch35.h")
#include "arch35/moe_ep_combine_arch35.h"
#endif

using namespace MoeEpCombineImpl;
using namespace Mc2Tiling;
using namespace AscendC;

extern "C" {

template<uint32_t HAS_TOPK_WEIGHTS>
__global__ __aicore__ void moe_ep_combine(
    GM_ADDR x, GM_ADDR dstBufferSlotIdx, GM_ADDR recvSrcMetadata,
    GM_ADDR numRecvPerRank, GM_ADDR topkWeights, GM_ADDR bias0, GM_ADDR bias1,
    GM_ADDR combinedX, GM_ADDR combinedTopkWeightsOptional,
    GM_ADDR workspace, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeEpCombineTilingData);
    REGISTER_TILING_FOR_TILINGKEY("ArchTag == TILINGKEY_TPL_A5", MoeEpCombineTilingData);

    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(MoeEpCombineTilingData, tilingData, tilingGM);
    KernelMoeEpCombine<HAS_TOPK_WEIGHTS> op;
    op.Init(x, dstBufferSlotIdx, recvSrcMetadata, numRecvPerRank,
            topkWeights, bias0, bias1,
            combinedX, combinedTopkWeightsOptional,
            workspace, tilingGM, &pipe, &tilingData.moeEpCombineInfo);
    op.Process();
}

}  // extern "C"