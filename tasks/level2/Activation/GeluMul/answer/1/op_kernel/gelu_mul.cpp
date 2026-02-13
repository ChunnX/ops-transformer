#include "gelu_mul.h"

using namespace AscendC;

using namespace GeluMul;

extern "C" __global__ __aicore__ void gelu_mul(GM_ADDR input, GM_ADDR output,
                                               GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    GM_ADDR userWs = nullptr;

#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    if (TILING_KEY_IS(1)) {
        GeluMulND<half> op;
        op.Init(input, output, userWs, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        GeluMulND<float> op;
        op.Init(input, output, userWs, &tilingData);
        op.Process();
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    } else if (TILING_KEY_IS(3)) {
        GeluMulND<bfloat16_t> op;
        op.Init(input, output, userWs, &tilingData);
        op.Process();
#endif
    }
#else
#endif
}