#include "kernel_operator.h"

template <int TRANS, int QUANT_TYPE, int OPTION_ATTRS, int WEIGHTNZ, int KERNEL_TEMPLATE_TYPE>
__global__ __aicore__ void quant_batch_matmul_v4(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale,
                                                            GM_ADDR x2_scale, GM_ADDR y_scale, GM_ADDR x1_offset,
                                                            GM_ADDR x2_offset, GM_ADDR y_offset, GM_ADDR x2_table,
                                                            GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    // TODO: Implement kernel logic
}
