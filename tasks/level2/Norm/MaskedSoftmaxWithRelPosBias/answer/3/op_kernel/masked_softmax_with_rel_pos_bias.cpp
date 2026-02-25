#include "kernel_operator.h"
#include "masked_softmax_with_rel_pos_bias_BW.h"

using namespace MaskedSoftmaxWithRelPosBias;

extern "C" __global__ __aicore__ void masked_softmax_with_rel_pos_bias(GM_ADDR x, GM_ADDR atten_mask, GM_ADDR bias,
                                                                       GM_ADDR y, GM_ADDR work_space, GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  if (TILING_KEY_IS(401)) {
    MaskedSoftmaxWithRelPosBiasBW<float, true, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(402)) {
    MaskedSoftmaxWithRelPosBiasBW<float, true, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(403)) {
    MaskedSoftmaxWithRelPosBiasBW<float, false, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(404)) {
    MaskedSoftmaxWithRelPosBiasBW<float, false, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(411)) {
    MaskedSoftmaxWithRelPosBiasBW<half, true, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(412)) {
    MaskedSoftmaxWithRelPosBiasBW<half, true, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(413)) {
    MaskedSoftmaxWithRelPosBiasBW<half, false, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(414)) {
    MaskedSoftmaxWithRelPosBiasBW<half, false, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(421)) {
#if (__CCE_AICORE__ > 200) && !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    MaskedSoftmaxWithRelPosBiasBW<bfloat16_t, true, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(422)) {
    MaskedSoftmaxWithRelPosBiasBW<bfloat16_t, true, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(423)) {
    MaskedSoftmaxWithRelPosBiasBW<bfloat16_t, false, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(424)) {
    MaskedSoftmaxWithRelPosBiasBW<bfloat16_t, false, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  }
#else
  }
#endif
}