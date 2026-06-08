#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "arch35/lightning_indexer_grad_vector.h"

using namespace AscendC;
using namespace MicroAPI;

extern "C" __global__ __aicore__ void lightning_indexer_grad(GM_ADDR query, GM_ADDR key, GM_ADDR dy,
    GM_ADDR sparse_indices, GM_ADDR weights, GM_ADDR actual_seq_lengths_query, GM_ADDR actual_seq_lengths_key,
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dweights, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    if (usrWorkspace == nullptr) {
        return;
    }

    TPipe pipe;

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    LIG::LightningIndexerGradKernel<DTYPE_QUERY> op;

    op.Init(query, key, dy, sparse_indices, weights, actual_seq_lengths_query, actual_seq_lengths_key,
        dq, dk, dweights, usrWorkspace, &tilingData, &pipe);
    op.Process();
}