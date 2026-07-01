/**
 * @file dispatch_entry.cpp
 * @brief MC2 dispatch v2 <<<>>> direct launch entry (A5 arch35 only)
 *
 * 与原始 moe_distribute_dispatch_v2.cpp 中 A5 MTE 路径逐分支对齐：
 *   - A5 MTE noFullMesh → MoeDistributeDispatchV2<Mc2MoeContextHolder, ...>
 *   - A5 MTE FullMesh   → MoeDistributeDispatchV2FullMesh<Mc2MoeContextHolder, ...>
 * CCU/A3/A2 路径已裁剪（不移到直调）。
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "op_kernel/mc2_moe_context_holder.h"
#include "op_kernel/moe_distribute_dispatch_v2.h"
#include "op_kernel/moe_distribute_dispatch_v2_full_mesh.h"
#include "op_kernel/moe_distribute_dispatch_v2_tiling.h"
#include <cstring>

using namespace Mc2Kernel;
using namespace AscendC;

constexpr uint32_t DISPATCH_FLOAT_OVERFLOW_MODE_CTRL = 60;

template<typename XType, typename ExpandXType, uint8_t QuantMode, bool ScaleMode, bool IsFullMesh>
__attribute__((always_inline)) __aicore__ __inline__ void mc2_dispatch_impl(
    GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIds,
    GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR elasticInfo,
    GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut,
    GM_ADDR workspaceGM, MoeDistributeDispatchV2TilingData tilingData)
{
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<DISPATCH_FLOAT_OVERFLOW_MODE_CTRL,
                                                  DISPATCH_FLOAT_OVERFLOW_MODE_CTRL>();
    TPipe pipe;
    if constexpr (IsFullMesh) {
        MoeDistributeDispatchV2FullMesh<Mc2Kernel::Mc2MoeContextHolder,
            XType, ExpandXType, QuantMode, ScaleMode> op;
        op.Init(mc2Context, x, expertIds, scales, xActiveMask, elasticInfo,
                performanceInfo, expandXOut, dynamicScalesOut, expandIdxOut,
                expertTokenNumsOut, sendCountsOut,
                workspaceGM, &pipe, &tilingData);
        op.Process();
    } else {
        MoeDistributeDispatchV2<Mc2Kernel::Mc2MoeContextHolder,
            XType, ExpandXType, QuantMode, ScaleMode> op;
        op.Init(mc2Context, x, expertIds, scales, xActiveMask, elasticInfo,
                performanceInfo, expandXOut, dynamicScalesOut, expandIdxOut,
                expertTokenNumsOut, sendCountsOut,
                workspaceGM, &pipe, &tilingData);
        op.Process();
    }
    AscendC::SetCtrlSpr<DISPATCH_FLOAT_OVERFLOW_MODE_CTRL,
                        DISPATCH_FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

extern "C" __global__ __aicore__ void mc2_dispatch_kernel(
    int32_t tilingKey,
    GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIds,
    GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR elasticInfo,
    GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut,
    GM_ADDR tpSendCountsOut, GM_ADDR expandScalesOut, GM_ADDR workspaceGM,
    MoeDistributeDispatchV2TilingData tilingData)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    switch (tilingKey) {
    case 0:
        mc2_dispatch_impl<half, half, 0, false, false>(
            mc2Context, x, expertIds, scales, xActiveMask, elasticInfo,
            performanceInfo, expandXOut, dynamicScalesOut, expandIdxOut,
            expertTokenNumsOut, epSendCountsOut, workspaceGM, tilingData);
        break;
    case 1:
        mc2_dispatch_impl<bfloat16_t, bfloat16_t, 0, false, false>(
            mc2Context, x, expertIds, scales, xActiveMask, elasticInfo,
            performanceInfo, expandXOut, dynamicScalesOut, expandIdxOut,
            expertTokenNumsOut, epSendCountsOut, workspaceGM, tilingData);
        break;
    case 10:
        mc2_dispatch_impl<half, half, 0, false, true>(
            mc2Context, x, expertIds, scales, xActiveMask, elasticInfo,
            performanceInfo, expandXOut, dynamicScalesOut, expandIdxOut,
            expertTokenNumsOut, epSendCountsOut, workspaceGM, tilingData);
        break;
    case 11:
        mc2_dispatch_impl<bfloat16_t, bfloat16_t, 0, false, true>(
            mc2Context, x, expertIds, scales, xActiveMask, elasticInfo,
            performanceInfo, expandXOut, dynamicScalesOut, expandIdxOut,
            expertTokenNumsOut, epSendCountsOut, workspaceGM, tilingData);
        break;
    }
#endif
}

extern "C" void mc2_dispatch_launch(int32_t tilingKey, uint32_t blockDim, void *stream,
    void *mc2ContextPtr, void *xPtr, void *expertIdsPtr,
    void *scalesPtr, void *xActiveMaskPtr, void *elasticInfoPtr,
    void *performanceInfoPtr, void *expandXOutPtr, void *dynamicScalesOutPtr,
    void *expandIdxOutPtr, void *expertTokenNumsOutPtr, void *epSendCountsOutPtr,
    void *tpSendCountsOutPtr, void *expandScalesOutPtr, void *workspaceGMPtr,
    void *tilingInfoBytesPtr)
{
    MoeDistributeDispatchV2TilingData tilingData = {};
    std::memcpy(&tilingData.moeDistributeDispatchV2Info, tilingInfoBytesPtr,
                sizeof(tilingData.moeDistributeDispatchV2Info));
    mc2_dispatch_kernel<<<blockDim, nullptr, stream>>>(
        tilingKey,
        (GM_ADDR)mc2ContextPtr, (GM_ADDR)xPtr, (GM_ADDR)expertIdsPtr,
        (GM_ADDR)scalesPtr, (GM_ADDR)xActiveMaskPtr, (GM_ADDR)elasticInfoPtr,
        (GM_ADDR)performanceInfoPtr, (GM_ADDR)expandXOutPtr,
        (GM_ADDR)dynamicScalesOutPtr, (GM_ADDR)expandIdxOutPtr,
        (GM_ADDR)expertTokenNumsOutPtr, (GM_ADDR)epSendCountsOutPtr,
        (GM_ADDR)tpSendCountsOutPtr, (GM_ADDR)expandScalesOutPtr,
        (GM_ADDR)workspaceGMPtr, tilingData);
}
