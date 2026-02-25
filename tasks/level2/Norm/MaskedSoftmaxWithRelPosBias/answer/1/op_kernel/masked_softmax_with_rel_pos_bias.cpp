#include "kernel_operator.h"
#include "masked_softmax_with_rel_pos_bias_B.h"

using namespace MaskedSoftmaxWithRelPosBias;

extern "C" __global__ __aicore__ void masked_softmax_with_rel_pos_bias(GM_ADDR x, GM_ADDR atten_mask, GM_ADDR bias,
                                                                       GM_ADDR y, GM_ADDR work_space, GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  if (TILING_KEY_IS(5001)) {
    MaskedSoftmaxWithRelPosBiasB<float, true, true, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5002)) {
    MaskedSoftmaxWithRelPosBiasB<float, true, false, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5003)) {
    MaskedSoftmaxWithRelPosBiasB<float, false, true, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5004)) {
    MaskedSoftmaxWithRelPosBiasB<float, false, false, true> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5101)) {
    MaskedSoftmaxWithRelPosBiasB<float, true, true, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5102)) {
    MaskedSoftmaxWithRelPosBiasB<float, true, false, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5103)) {
    MaskedSoftmaxWithRelPosBiasB<float, false, true, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  } else if (TILING_KEY_IS(5104)) {
    MaskedSoftmaxWithRelPosBiasB<float, false, false, false> op(&tiling_data);
    op.Init(x, atten_mask, bias, y);
    op.Process();
  }
}