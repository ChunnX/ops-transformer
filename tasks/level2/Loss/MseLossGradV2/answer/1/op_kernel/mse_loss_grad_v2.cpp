#include <cstdint>
#include "kernel_operator.h"
#include "mse_loss_grad_v2_310p.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void mse_loss_grad_v2(GM_ADDR predict, GM_ADDR label, GM_ADDR dout,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
#define INIT_AND_PROCESS \
    op.Init(predict, label, dout, y, tiling_data.cof, \
            tiling_data.totalLength, tiling_data.tileNum, \
            tiling_data.blockLength, tiling_data.padLength, tiling_data.usedDb); \
    op.Process()

    if (TILING_KEY_IS(1)) {
        KernelMseLossGrad310P<float> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2)) {
        KernelMseLossGrad310P<half> op;
        INIT_AND_PROCESS;
    }
}