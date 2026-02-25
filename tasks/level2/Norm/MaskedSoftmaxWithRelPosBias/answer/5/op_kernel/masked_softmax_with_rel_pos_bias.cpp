#include "kernel_operator.h"
#include "masked_softmax_with_rel_pos_bias_BWN.h"

using namespace MaskedSoftmaxWithRelPosBias;

extern "C" __global__ __aicore__ void masked_softmax_with_rel_pos_bias(GM_ADDR x, GM_ADDR atten_mask, GM_ADDR bias,
                                                                       GM_ADDR y, GM_ADDR work_space, GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  if (TILING_KEY_IS(301)) {
    MaskedSoftmaxWithRelPosBiasBWN<float, true, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(302)) {
    MaskedSoftmaxWithRelPosBiasBWN<float, true, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(303)) {
    MaskedSoftmaxWithRelPosBiasBWN<float, false, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(304)) {
    MaskedSoftmaxWithRelPosBiasBWN<float, false, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(311)) {
    MaskedSoftmaxWithRelPosBiasBWN<half, true, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(312)) {
    MaskedSoftmaxWithRelPosBiasBWN<half, true, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(313)) {
    MaskedSoftmaxWithRelPosBiasBWN<half, false, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(314)) {
    MaskedSoftmaxWithRelPosBiasBWN<half, false, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(321)) {
#if (__CCE_AICORE__ > 200) && !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    MaskedSoftmaxWithRelPosBiasBWN<bfloat16_t, true, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(322)) {
    MaskedSoftmaxWithRelPosBiasBWN<bfloat16_t, true, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(323)) {
    MaskedSoftmaxWithRelPosBiasBWN<bfloat16_t, false, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(324)) {
    MaskedSoftmaxWithRelPosBiasBWN<bfloat16_t, false, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  }
#else
  }
#endif
}