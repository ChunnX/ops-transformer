#include "logit_grad.h"

using namespace AscendC;

using namespace LogitGrad;

extern "C" __global__ __aicore__ void logit_grad(GM_ADDR x, GM_ADDR dy, GM_ADDR dx, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    GM_ADDR userWs = nullptr;

#if __CCE_AICORE__ == 220
    if (TILING_KEY_IS(1)) {
        LogitGradND<half> op;
        op.Init(x, dy, dx, userWs, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        LogitGradND<float> op;
        op.Init(x, dy, dx, userWs, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        LogitGradND<bfloat16_t> op;
        op.Init(x, dy, dx, userWs, &tilingData);
        op.Process();
    }
#endif
}