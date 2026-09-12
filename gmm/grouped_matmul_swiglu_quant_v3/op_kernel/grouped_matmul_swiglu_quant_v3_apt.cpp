/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Ascend 950 MXFP8 weight-NZ kernel entry reserved for V3.
 * The concrete Ascend C implementation is intentionally deferred until a
 * 950 build environment is available.
 */

#include "kernel_tiling/kernel_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "arch35/grouped_matmul_swiglu_quant_v3_tiling_key.h"

template <int8_t QUANT_B_TRANS, int8_t QUANT_A_TRANS, int8_t KERNEL_TYPE>
__global__ __aicore__ void grouped_matmul_swiglu_quant_v3(
    GM_ADDR x, GM_ADDR xScale, GM_ADDR groupList, GM_ADDR weight, GM_ADDR weightScale,
    GM_ADDR weightAssistanceMatrix, GM_ADDR bias, GM_ADDR smoothScale, GM_ADDR y, GM_ADDR yScale,
    GM_ADDR workspace, GM_ADDR tiling)
{
    // TODO: implement the Ascend 950 MXFP8 weight-NZ tiling/kernel path.
    (void)QUANT_B_TRANS;
    (void)QUANT_A_TRANS;
    (void)KERNEL_TYPE;
    (void)x;
    (void)xScale;
    (void)groupList;
    (void)weight;
    (void)weightScale;
    (void)weightAssistanceMatrix;
    (void)bias;
    (void)smoothScale;
    (void)y;
    (void)yScale;
    (void)workspace;
    (void)tiling;
}
