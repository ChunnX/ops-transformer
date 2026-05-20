#include "kernel_operator.h"
#include "dispatch_normal_tiling.h"
#include "dispatch_normal.h"
#include "dispatch_normal_a5.h"

using namespace AscendC;
using namespace DispatchNormalImpl;
using namespace DispatchNormalA5Impl;

#define TILINGKEY_A3_NO_QUANT 13000
#define TILINGKEY_A3_QUANT 13002
#define TILINGKEY_A5_NO_QUANT 15000
#define TILINGKEY_A5_QUANT 15002

extern "C" __global__ __aicore__ void dispatch_normal(GM_ADDR context, GM_ADDR x, GM_ADDR expertIds,
                                                      GM_ADDR send_offset, GM_ADDR send_token_idx, GM_ADDR recv_offset,
                                                      GM_ADDR expert_output_offset, GM_ADDR expandXOut,
                                                      GM_ADDR dynamicScalesOut, GM_ADDR assist_info_for_combine,
                                                      GM_ADDR waitRecvCostStatsOut, GM_ADDR extHandleData,
                                                      GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(DispatchNormalTilingData);
    TPipe pipe;
#if (ORIG_DTYPE_RECV_X == DT_BF16 || ORIG_DTYPE_RECV_X == DT_FLOAT16)
    if (TILING_KEY_IS(TILINGKEY_A3_NO_QUANT)) {
        GET_TILING_DATA_WITH_STRUCT(DispatchNormalTilingData, tilingData, tilingGM);
        DispatchNormal<DTYPE_X, DTYPE_RECV_X, false, false, false> op;
        op.Init(context, x, expertIds, send_offset, send_token_idx, recv_offset, expert_output_offset, expandXOut,
                dynamicScalesOut, assist_info_for_combine, waitRecvCostStatsOut, extHandleData, workspaceGM, &pipe,
                &tilingData);
        op.Process();
        return;
    } else if (TILING_KEY_IS(TILINGKEY_A5_NO_QUANT)) {
        GET_TILING_DATA_WITH_STRUCT(DispatchNormalTilingData, tilingData, tilingGM);
        DispatchNormalA5<DTYPE_X, DTYPE_RECV_X, false, false, false> op;
        op.Init(context, x, expertIds, send_offset, send_token_idx, recv_offset, expert_output_offset, expandXOut,
                dynamicScalesOut, assist_info_for_combine, waitRecvCostStatsOut, extHandleData, workspaceGM, &pipe,
                &tilingData);
        op.Process();
        return;
    }
#elif (ORIG_DTYPE_RECV_X == DT_INT8)
    if (TILING_KEY_IS(TILINGKEY_A3_QUANT)) {
        GET_TILING_DATA_WITH_STRUCT(DispatchNormalTilingData, tilingData, tilingGM);
        DispatchNormal<DTYPE_X, DTYPE_RECV_X, true, false, false> op;
        op.Init(context, x, expertIds, send_offset, send_token_idx, recv_offset, expert_output_offset, expandXOut,
                dynamicScalesOut, assist_info_for_combine, waitRecvCostStatsOut, extHandleData, workspaceGM, &pipe,
                &tilingData);
        op.Process();
        return;
    } else if (TILING_KEY_IS(TILINGKEY_A5_QUANT)) {
        GET_TILING_DATA_WITH_STRUCT(DispatchNormalTilingData, tilingData, tilingGM);
        DispatchNormalA5<DTYPE_X, DTYPE_RECV_X, true, false, false> op;
        op.Init(context, x, expertIds, send_offset, send_token_idx, recv_offset, expert_output_offset, expandXOut,
                dynamicScalesOut, assist_info_for_combine, waitRecvCostStatsOut, extHandleData, workspaceGM, &pipe,
                &tilingData);
        op.Process();
    }
#endif
}
