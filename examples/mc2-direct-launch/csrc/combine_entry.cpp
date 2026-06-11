#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "op_kernel/moe_distribute_combine_v2.h"
#include "op_kernel/moe_distribute_combine_v2_tiling.h"


using namespace MoeDistributeCombineV2Impl;
using namespace Mc2Kernel;
using namespace AscendC;

#define FLOAT_OVERFLOW_MODE_CTRL 60

template<typename ExpandXType, typename XType, uint8_t QuantMode>
__attribute__((always_inline)) __aicore__ __inline__ void mc2_combine_impl(
    GM_ADDR mc2Context, GM_ADDR expandX, GM_ADDR expertIds,
    GM_ADDR assistInfo, GM_ADDR epSendCount, GM_ADDR tpSendCount,
    GM_ADDR residualX, GM_ADDR gamma, GM_ADDR expertScales,
    GM_ADDR xActiveMask, GM_ADDR sharedExpertX,
    GM_ADDR elasticInfo, GM_ADDR oriX,
    GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV,
    GM_ADDR performanceInfo, GM_ADDR yOut, GM_ADDR rstdOut, GM_ADDR XOut,
    GM_ADDR workspaceGM, MoeDistributeCombineV2TilingData tilingData)
{
    TPipe pipe;
    int64_t ori = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    MoeDistributeCombineV2<ExpandXType, XType, int32_t, false, QuantMode, false> op;
    op.Init(mc2Context, expandX, expertIds, assistInfo, epSendCount, tpSendCount,
            residualX, gamma, expertScales, xActiveMask, sharedExpertX,
            elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV,
            performanceInfo, yOut, rstdOut, XOut, workspaceGM, &pipe, &tilingData);
    op.Process();
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(ori);
}

extern "C" __global__ __aicore__ void mc2_combine_kernel(
    int32_t tilingKey,
    GM_ADDR mc2Context, GM_ADDR expandX, GM_ADDR expertIds,
    GM_ADDR assistInfo, GM_ADDR epSendCount, GM_ADDR tpSendCount,
    GM_ADDR residualX, GM_ADDR gamma, GM_ADDR expertScales,
    GM_ADDR xActiveMask, GM_ADDR sharedExpertX,
    GM_ADDR elasticInfo, GM_ADDR oriX,
    GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV,
    GM_ADDR performanceInfo, GM_ADDR yOut, GM_ADDR rstdOut, GM_ADDR XOut,
    GM_ADDR workspaceGM, MoeDistributeCombineV2TilingData tilingData)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    switch (tilingKey) {
    case 0:
        mc2_combine_impl<half, half, 0>(
            mc2Context, expandX, expertIds, assistInfo, epSendCount, tpSendCount,
            residualX, gamma, expertScales, xActiveMask, sharedExpertX,
            elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV,
            performanceInfo, yOut, rstdOut, XOut, workspaceGM, tilingData);
        break;
    case 1:
        mc2_combine_impl<bfloat16_t, bfloat16_t, 0>(
            mc2Context, expandX, expertIds, assistInfo, epSendCount, tpSendCount,
            residualX, gamma, expertScales, xActiveMask, sharedExpertX,
            elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV,
            performanceInfo, yOut, rstdOut, XOut, workspaceGM, tilingData);
        break;
    case 20:
        mc2_combine_impl<half, half, 2>(
            mc2Context, expandX, expertIds, assistInfo, epSendCount, tpSendCount,
            residualX, gamma, expertScales, xActiveMask, sharedExpertX,
            elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV,
            performanceInfo, yOut, rstdOut, XOut, workspaceGM, tilingData);
        break;
    case 21:
        mc2_combine_impl<bfloat16_t, bfloat16_t, 2>(
            mc2Context, expandX, expertIds, assistInfo, epSendCount, tpSendCount,
            residualX, gamma, expertScales, xActiveMask, sharedExpertX,
            elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV,
            performanceInfo, yOut, rstdOut, XOut, workspaceGM, tilingData);
        break;
    default:
        break;
    }
#endif
}

void mc2_combine_launch(int32_t tilingKey, uint32_t blockDim, void *stream,
    void *mc2Context, void *expandX, void *expertIds,
    void *assistInfo, void *epSendCount, void *tpSendCount,
    void *residualX, void *gamma, void *expertScales,
    void *xActiveMask, void *sharedExpertX,
    void *elasticInfo, void *oriX,
    void *constExpertAlpha1, void *constExpertAlpha2, void *constExpertV,
    void *performanceInfo, void *yOut, void *rstdOut, void *XOut,
    void *workspaceGM, MoeDistributeCombineV2Info tilingInfo)
{
    MoeDistributeCombineV2TilingData tilingData = {};
    tilingData.moeDistributeCombineV2Info = tilingInfo;
    mc2_combine_kernel<<<blockDim, nullptr, stream>>>(
        tilingKey,
        (GM_ADDR)mc2Context, (GM_ADDR)expandX, (GM_ADDR)expertIds,
        (GM_ADDR)assistInfo, (GM_ADDR)epSendCount, (GM_ADDR)tpSendCount,
        (GM_ADDR)residualX, (GM_ADDR)gamma, (GM_ADDR)expertScales,
        (GM_ADDR)xActiveMask, (GM_ADDR)sharedExpertX,
        (GM_ADDR)elasticInfo, (GM_ADDR)oriX,
        (GM_ADDR)constExpertAlpha1, (GM_ADDR)constExpertAlpha2, (GM_ADDR)constExpertV,
        (GM_ADDR)performanceInfo, (GM_ADDR)yOut, (GM_ADDR)rstdOut, (GM_ADDR)XOut,
        (GM_ADDR)workspaceGM, tilingData);
}
