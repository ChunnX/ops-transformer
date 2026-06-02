#include "kernel_operator.h"
#include "dispatch_normal_tiling.h"
#include "dispatch_normal_tiling_key.h"
#include "dispatch_normal.h"
#include "dispatch_normal_a5.h"

using namespace AscendC;
using namespace DispatchNormalImpl;
using namespace DispatchNormalA5Impl;
using namespace Mc2Tiling;

template <uint8_t QuantMode, uint8_t ArchTag>
__global__ __aicore__ void dispatch_normal(GM_ADDR context, GM_ADDR x, GM_ADDR expertIds, GM_ADDR send_offset,
                                           GM_ADDR send_token_idx, GM_ADDR recv_offset, GM_ADDR expert_output_offset,
                                           GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
                                           GM_ADDR assist_info_for_combine, GM_ADDR waitRecvCostStatsOut,
                                           GM_ADDR extHandleData, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(DispatchNormalTilingData);
    TPipe pipe;
#if (ORIG_DTYPE_RECV_X == DT_BF16 || ORIG_DTYPE_RECV_X == DT_FLOAT16)
    if constexpr ((QuantMode == TILINGKEY_NO_QUANT) && (ArchTag == TILINGKEY_TPL_A3)) {
        GET_TILING_DATA_WITH_STRUCT(DispatchNormalTilingData, tilingData, tilingGM);
        DispatchNormal<DTYPE_X, DTYPE_RECV_X, false, false, false> op;
        op.Init(context, x, expertIds, send_offset, send_token_idx, recv_offset, expert_output_offset, expandXOut,
                dynamicScalesOut, assist_info_for_combine, waitRecvCostStatsOut, extHandleData, workspaceGM, &pipe,
                &tilingData);
        op.Process();
    } else if constexpr ((QuantMode == TILINGKEY_NO_QUANT) && (ArchTag == TILINGKEY_TPL_A5)) {
        GET_TILING_DATA_WITH_STRUCT(DispatchNormalTilingData, tilingData, tilingGM);
        DispatchNormalA5<DTYPE_X, DTYPE_RECV_X, false, false, false> op;
        op.Init(context, x, expertIds, send_offset, send_token_idx, recv_offset, expert_output_offset, expandXOut,
                dynamicScalesOut, assist_info_for_combine, waitRecvCostStatsOut, extHandleData, workspaceGM, &pipe,
                &tilingData);
        op.Process();
    }
#elif (ORIG_DTYPE_RECV_X == DT_INT8)
    if constexpr ((QuantMode == TILINGKEY_PERTOKEN_QUANT) && (ArchTag == TILINGKEY_TPL_A3)) {
        GET_TILING_DATA_WITH_STRUCT(DispatchNormalTilingData, tilingData, tilingGM);
        DispatchNormal<DTYPE_X, DTYPE_RECV_X, true, false, false> op;
        op.Init(context, x, expertIds, send_offset, send_token_idx, recv_offset, expert_output_offset, expandXOut,
                dynamicScalesOut, assist_info_for_combine, waitRecvCostStatsOut, extHandleData, workspaceGM, &pipe,
                &tilingData);
        op.Process();
    } else if constexpr ((QuantMode == TILINGKEY_PERTOKEN_QUANT) && (ArchTag == TILINGKEY_TPL_A5)) {
        GET_TILING_DATA_WITH_STRUCT(DispatchNormalTilingData, tilingData, tilingGM);
        DispatchNormalA5<DTYPE_X, DTYPE_RECV_X, true, false, false> op;
        op.Init(context, x, expertIds, send_offset, send_token_idx, recv_offset, expert_output_offset, expandXOut,
                dynamicScalesOut, assist_info_for_combine, waitRecvCostStatsOut, extHandleData, workspaceGM, &pipe,
                &tilingData);
        op.Process();
    }
#endif
}
