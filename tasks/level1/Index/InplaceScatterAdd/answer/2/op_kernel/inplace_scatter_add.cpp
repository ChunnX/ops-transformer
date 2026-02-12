#include "kernel_operator.h"
#include "inplace_scatter_add_big_tail.h"

extern "C" __global__ __aicore__ void inplace_scatter_add(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR var_ref, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    AscendC::TPipe tpipe;

#if (defined(DTYPE_VAR) && defined(DTYPE_INDICES))
    InplaceScatterAddNS::BigTailOp<DTYPE_VAR, DTYPE_INDICES> op;
    op.Init(var, indices, updates, &tiling_data, &tpipe);
    op.Process();
#endif
}