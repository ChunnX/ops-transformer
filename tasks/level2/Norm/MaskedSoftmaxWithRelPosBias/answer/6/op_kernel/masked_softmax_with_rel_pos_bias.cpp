#include "kernel_operator.h"
#include "masked_softmax_with_rel_pos_bias_ONES2.h"

using namespace MaskedSoftmaxWithRelPosBias;

extern "C" __global__ __aicore__ void masked_softmax_with_rel_pos_bias(GM_ADDR x, GM_ADDR atten_mask, GM_ADDR bias,
                                                                       GM_ADDR y, GM_ADDR work_space, GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  if (TILING_KEY_IS(100)) {
    MaskedSoftmaxWithRelPosBiasONES2<float> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(110)) {
    MaskedSoftmaxWithRelPosBiasONES2<half> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(120)) {
#if (__CCE_AICORE__ > 200) && !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    MaskedSoftmaxWithRelPosBiasONES2<bfloat16_t> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
#endif
  }
}