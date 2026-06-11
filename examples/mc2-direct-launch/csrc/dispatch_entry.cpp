#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "op_kernel/moe_distribute_dispatch_v2.h"
#include "op_kernel/moe_distribute_dispatch_v2_tiling.h"
#include "op_kernel/moe_distribute_dispatch_v2_full_mesh.h"


using namespace MoeDistributeDispatchV2Impl;
using namespace MoeDistributeDispatchV2FullMeshImpl;
using namespace Mc2Kernel;
using namespace AscendC;

#define FLOAT_OVERFLOW_MODE_CTRL 60

template<typename XType, typename ExpandXType, uint8_t QuantMode, bool ScaleMode, bool IsFullMesh>
__attribute__((always_inline)) __aicore__ __inline__ void mc2_dispatch_impl(
    GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIds,
    GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales,
    GM_ADDR elasticInfo, GM_ADDR performanceInfo,
    GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut,
    GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut,
    GM_ADDR expandScalesOut, GM_ADDR workspaceGM,
    MoeDistributeDispatchV2TilingData tilingData)
{
    TPipe pipe;
    int64_t ori = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    if constexpr (IsFullMesh) {
        MoeDistributeDispatchV2FullMesh<XType, ExpandXType, QuantMode, ScaleMode, false> op;
        op.Init(mc2Context, x, expertIds, scales, xActiveMask, elasticInfo, performanceInfo,
                expandXOut, dynamicScalesOut, assistInfoOut,
                expertTokenNumsOut, epSendCountsOut, tpSendCountsOut,
                workspaceGM, &pipe, &tilingData);
        op.Process();
    } else {
        MoeDistributeDispatchV2<XType, ExpandXType, QuantMode, ScaleMode, false> op;
        op.Init(mc2Context, x, expertIds, scales, xActiveMask, elasticInfo, performanceInfo,
                expandXOut, dynamicScalesOut, assistInfoOut,
                expertTokenNumsOut, epSendCountsOut, tpSendCountsOut,
                workspaceGM, &pipe, &tilingData);
        op.Process();
    }
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(ori);
}

extern "C" __global__ __aicore__ void mc2_dispatch_kernel(
    int32_t tilingKey,
    GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIds,
    GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales,
    GM_ADDR elasticInfo, GM_ADDR performanceInfo,
    GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut,
    GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut,
    GM_ADDR expandScalesOut, GM_ADDR workspaceGM,
    MoeDistributeDispatchV2TilingData tilingData)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    switch (tilingKey) {
    case 0:
        mc2_dispatch_impl<half, half, 0, false, false>(
            mc2Context, x, expertIds, scales, xActiveMask, expertScales,
            elasticInfo, performanceInfo, expandXOut, dynamicScalesOut,
            assistInfoOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut,
            expandScalesOut, workspaceGM, tilingData);
        break;
    case 1:
        mc2_dispatch_impl<bfloat16_t, bfloat16_t, 0, false, false>(
            mc2Context, x, expertIds, scales, xActiveMask, expertScales,
            elasticInfo, performanceInfo, expandXOut, dynamicScalesOut,
            assistInfoOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut,
            expandScalesOut, workspaceGM, tilingData);
        break;
    case 10:
        mc2_dispatch_impl<half, half, 0, false, true>(
            mc2Context, x, expertIds, scales, xActiveMask, expertScales,
            elasticInfo, performanceInfo, expandXOut, dynamicScalesOut,
            assistInfoOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut,
            expandScalesOut, workspaceGM, tilingData);
        break;
    case 11:
        mc2_dispatch_impl<bfloat16_t, bfloat16_t, 0, false, true>(
            mc2Context, x, expertIds, scales, xActiveMask, expertScales,
            elasticInfo, performanceInfo, expandXOut, dynamicScalesOut,
            assistInfoOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut,
            expandScalesOut, workspaceGM, tilingData);
        break;
    default:
        break;
    }
#endif
}

void mc2_dispatch_launch(int32_t tilingKey, uint32_t blockDim, void *stream,
    void *mc2Context, void *x, void *expertIds,
    void *scales, void *xActiveMask, void *expertScales,
    void *elasticInfo, void *performanceInfo,
    void *expandXOut, void *dynamicScalesOut,
    void *assistInfoOut, void *expertTokenNumsOut,
    void *epSendCountsOut, void *tpSendCountsOut,
    void *expandScalesOut, void *workspaceGM,
    MoeDistributeDispatchV2Info tilingInfo)
{
    MoeDistributeDispatchV2TilingData tilingData = {};
    tilingData.moeDistributeDispatchV2Info = tilingInfo;
    mc2_dispatch_kernel<<<blockDim, nullptr, stream>>>(
        tilingKey,
        (GM_ADDR)mc2Context, (GM_ADDR)x, (GM_ADDR)expertIds,
        (GM_ADDR)scales, (GM_ADDR)xActiveMask, (GM_ADDR)expertScales,
        (GM_ADDR)elasticInfo, (GM_ADDR)performanceInfo,
        (GM_ADDR)expandXOut, (GM_ADDR)dynamicScalesOut,
        (GM_ADDR)assistInfoOut, (GM_ADDR)expertTokenNumsOut,
        (GM_ADDR)epSendCountsOut, (GM_ADDR)tpSendCountsOut,
        (GM_ADDR)expandScalesOut, (GM_ADDR)workspaceGM,
        tilingData);
}
