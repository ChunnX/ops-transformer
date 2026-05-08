/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * LLM question template — 函数签名与 answer 一致，函数体为空。
 */
#define K_MAX_SHAPE_DIM 0

#include "basic_api/kernel_basic_intf.h"
#include "grouped_mat_mul_all_reduce_utils.h"
#include "grouped_mat_mul_all_reduce.h"

using namespace AscendC;
using namespace matmul;
using namespace GROUPED_MAT_MUL_ALL_REDUCE;

extern "C" __global__ __aicore__ void grouped_mat_mul_all_reduce(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR group_list, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
}
