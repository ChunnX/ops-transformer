/**
 * @file combine_entry.cpp
 * @brief MC2 combine v2 <<<>>> direct launch entry (A5 arch35 MTE only)
 *
 * 与原始 moe_distribute_combine_v2.cpp 中 A5 MTE 路径逐分支对齐：
 *   - A5 MTE → MoeDistributeCombineV2A5Mte<ExpandXType, XType, int32_t, QuantMode, false>
 *     （注意：无 ContextHolder 模板参数，与 dispatch 不同）
 * CCU/A3/A2 路径已裁剪。
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "op_kernel/moe_distribute_combine_v2_a5_mte.h"
#include "op_kernel/moe_distribute_combine_v2_tiling.h"
#include <cstring>

using namespace MoeDistributeCombineV2A5MteImpl;
using namespace Mc2Kernel;
using namespace AscendC;

constexpr uint32_t COMBINE_FLOAT_OVERFLOW_MODE_CTRL = 60;

template<typename ExpandXType, typename XType, uint8_t QuantMode>
__attribute__((always_inline)) __aicore__ __inline__ void mc2_combine_impl(
    GM_ADDR mc2Context, GM_ADDR expandX, GM_ADDR expertIds,
    GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR tpSendCount,
    GM_ADDR residualX, GM_ADDR gammaGrad, GM_ADDR expertScales,
    GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR elasticInfo,
    GM_ADDR oriX, GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2,
    GM_ADDR constExpertV, GM_ADDR performanceInfo, GM_ADDR yOut, GM_ADDR rstdOut,
    GM_ADDR XOut, GM_ADDR workspaceGM,
    MoeDistributeCombineV2TilingData tilingData)
{
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<COMBINE_FLOAT_OVERFLOW_MODE_CTRL,
                                                  COMBINE_FLOAT_OVERFLOW_MODE_CTRL>();
    TPipe pipe;
    MoeDistributeCombineV2A5Mte<ExpandXType, XType, int32_t, QuantMode, false> op;
    op.Init(mc2Context, expandX, expertIds, expandIdx, epSendCount,
            residualX, gammaGrad, expertScales, xActiveMask, sharedExpertX,
            elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2,
            constExpertV, performanceInfo, yOut, rstdOut,
            XOut, workspaceGM, &pipe, &tilingData);
    op.Process();
    AscendC::SetCtrlSpr<COMBINE_FLOAT_OVERFLOW_MODE_CTRL,
                        COMBINE_FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

extern "C" __global__ __aicore__ void mc2_combine_kernel(
    int32_t tilingKey,
    GM_ADDR mc2Context, GM_ADDR expandX, GM_ADDR expertIds,
    GM_ADDR expandIdx, GM_ADDR epSendCount,
    GM_ADDR tpSendCount, GM_ADDR residualX, GM_ADDR gammaGrad,
    GM_ADDR expertScales, GM_ADDR xActiveMask, GM_ADDR sharedExpertX,
    GM_ADDR elasticInfo, GM_ADDR oriX, GM_ADDR constExpertAlpha1,
    GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR performanceInfo,
    GM_ADDR yOut, GM_ADDR rstdOut, GM_ADDR XOut, GM_ADDR workspaceGM,
    MoeDistributeCombineV2TilingData tilingData)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    switch (tilingKey) {
    case 0:
        mc2_combine_impl<half, half, 0>(
            mc2Context, expandX, expertIds, expandIdx, epSendCount,
            tpSendCount, residualX, gammaGrad, expertScales,
            xActiveMask, sharedExpertX, elasticInfo, oriX,
            constExpertAlpha1, constExpertAlpha2, constExpertV,
            performanceInfo, yOut, rstdOut, XOut, workspaceGM, tilingData);
        break;
    case 1:
        mc2_combine_impl<bfloat16_t, bfloat16_t, 0>(
            mc2Context, expandX, expertIds, expandIdx, epSendCount,
            tpSendCount, residualX, gammaGrad, expertScales,
            xActiveMask, sharedExpertX, elasticInfo, oriX,
            constExpertAlpha1, constExpertAlpha2, constExpertV,
            performanceInfo, yOut, rstdOut, XOut, workspaceGM, tilingData);
        break;
    case 20:
        mc2_combine_impl<half, half, 2>(
            mc2Context, expandX, expertIds, expandIdx, epSendCount,
            tpSendCount, residualX, gammaGrad, expertScales,
            xActiveMask, sharedExpertX, elasticInfo, oriX,
            constExpertAlpha1, constExpertAlpha2, constExpertV,
            performanceInfo, yOut, rstdOut, XOut, workspaceGM, tilingData);
        break;
    case 21:
        mc2_combine_impl<bfloat16_t, bfloat16_t, 2>(
            mc2Context, expandX, expertIds, expandIdx, epSendCount,
            tpSendCount, residualX, gammaGrad, expertScales,
            xActiveMask, sharedExpertX, elasticInfo, oriX,
            constExpertAlpha1, constExpertAlpha2, constExpertV,
            performanceInfo, yOut, rstdOut, XOut, workspaceGM, tilingData);
        break;
    }
#endif
}

extern "C" void mc2_combine_launch(int32_t tilingKey, uint32_t blockDim, void *stream,
    void *mc2ContextPtr, void *expandXPtr, void *expertIdsPtr,
    void *expandIdxPtr, void *epSendCountPtr,
    void *tpSendCountPtr, void *residualXPtr, void *gammaGradPtr,
    void *expertScalesPtr, void *xActiveMaskPtr, void *sharedExpertXPtr,
    void *elasticInfoPtr, void *oriXPtr, void *constExpertAlpha1Ptr,
    void *constExpertAlpha2Ptr, void *constExpertVPtr, void *performanceInfoPtr,
    void *yOutPtr, void *rstdOutPtr, void *XOutPtr, void *workspaceGMPtr,
    void *tilingInfoBytesPtr)
{
    MoeDistributeCombineV2TilingData tilingData = {};
    std::memcpy(&tilingData.moeDistributeCombineV2Info, tilingInfoBytesPtr,
                sizeof(tilingData.moeDistributeCombineV2Info));
    mc2_combine_kernel<<<blockDim, nullptr, stream>>>(
        tilingKey,
        (GM_ADDR)mc2ContextPtr, (GM_ADDR)expandXPtr, (GM_ADDR)expertIdsPtr,
        (GM_ADDR)expandIdxPtr, (GM_ADDR)epSendCountPtr,
        (GM_ADDR)tpSendCountPtr, (GM_ADDR)residualXPtr, (GM_ADDR)gammaGradPtr,
        (GM_ADDR)expertScalesPtr, (GM_ADDR)xActiveMaskPtr,
        (GM_ADDR)sharedExpertXPtr, (GM_ADDR)elasticInfoPtr, (GM_ADDR)oriXPtr,
        (GM_ADDR)constExpertAlpha1Ptr, (GM_ADDR)constExpertAlpha2Ptr,
        (GM_ADDR)constExpertVPtr, (GM_ADDR)performanceInfoPtr,
        (GM_ADDR)yOutPtr, (GM_ADDR)rstdOutPtr, (GM_ADDR)XOutPtr,
        (GM_ADDR)workspaceGMPtr, tilingData);
}
