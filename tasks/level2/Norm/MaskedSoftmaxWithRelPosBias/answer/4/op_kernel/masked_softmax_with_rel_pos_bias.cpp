#include "kernel_operator.h"
#include "masked_softmax_with_rel_pos_bias_S1_bias.h"

using namespace MaskedSoftmaxWithRelPosBias;

extern "C" __global__ __aicore__ void masked_softmax_with_rel_pos_bias(GM_ADDR x, GM_ADDR atten_mask, GM_ADDR bias,
                                                                       GM_ADDR y, GM_ADDR work_space, GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  if (TILING_KEY_IS(203)) {
    MaskedSoftmaxWithRelPosBiasBS1Bias<float, true, true> op(&tiling_data);
    op.Init(x, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(204)) {
    MaskedSoftmaxWithRelPosBiasBS1Bias<float, false, true> op(&tiling_data);
    op.Init(x, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(213)) {
    MaskedSoftmaxWithRelPosBiasBS1Bias<half, true, true> op(&tiling_data);
    op.Init(x, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(214)) {
    MaskedSoftmaxWithRelPosBiasBS1Bias<half, false, true> op(&tiling_data);
    op.Init(x, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(223)) {
#if (__CCE_AICORE__ > 200) && !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    MaskedSoftmaxWithRelPosBiasBS1Bias<bfloat16_t, true, true> op(&tiling_data);
    op.Init(x, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(224)) {
    MaskedSoftmaxWithRelPosBiasBS1Bias<bfloat16_t, false, true> op(&tiling_data);
    op.Init(x, bias, y);
    op.Process();
#endif
  } else if (TILING_KEY_IS(2103)) {
    MaskedSoftmaxWithRelPosBiasBS1Bias<float, true, false> op(&tiling_data);
    op.Init(x, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(2104)) {
    MaskedSoftmaxWithRelPosBiasBS1Bias<float, false, false> op(&tiling_data);
    op.Init(x, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(2113)) {
    MaskedSoftmaxWithRelPosBiasBS1Bias<half, true, false> op(&tiling_data);
    op.Init(x, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(2114)) {
    MaskedSoftmaxWithRelPosBiasBS1Bias<half, false, false> op(&tiling_data);
    op.Init(x, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(2123)) {
#if (__CCE_AICORE__ > 200) && !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    MaskedSoftmaxWithRelPosBiasBS1Bias<bfloat16_t, true, false> op(&tiling_data);
    op.Init(x, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(2124)) {
    MaskedSoftmaxWithRelPosBiasBS1Bias<bfloat16_t, false, false> op(&tiling_data);
    op.Init(x, bias, y);
    op.Process();
#else
  }
#endif
  }
}