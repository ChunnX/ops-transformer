#include "kernel_operator.h"
#include "masked_softmax_with_rel_pos_bias_B.h"

using namespace MaskedSoftmaxWithRelPosBias;

extern "C" __global__ __aicore__ void masked_softmax_with_rel_pos_bias(GM_ADDR x, GM_ADDR atten_mask, GM_ADDR bias,
                                                                       GM_ADDR y, GM_ADDR work_space, GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  if (TILING_KEY_IS(5011)) {
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<half, true, true, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5012)) {
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<half, true, false, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5013)) {
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<half, false, true, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5014)) {
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<half, false, false, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5021)) {
#if (__CCE_AICORE__ > 200) && !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<bfloat16_t, true, true, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5022)) {
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<bfloat16_t, true, false, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5023)) {
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<bfloat16_t, false, true, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5024)) {
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<bfloat16_t, false, false, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
#endif
  } else if (TILING_KEY_IS(5111)) {
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<half, true, true, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5112)) {
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<half, true, false, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5113)) {
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<half, false, true, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5114)) {
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<half, false, false, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5121)) {
#if (__CCE_AICORE__ > 200) && !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<bfloat16_t, true, true, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5122)) {
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<bfloat16_t, true, false, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5123)) {
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<bfloat16_t, false, true, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5124)) {
    MaskedSoftmaxWithRelPosBiasBBf16AndHalf<bfloat16_t, false, false, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
#endif
  }
}