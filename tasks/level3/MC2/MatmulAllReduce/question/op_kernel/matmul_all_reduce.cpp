/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_all_reduce.cpp
 * \brief
 */

#include "basic_api/kernel_basic_intf.h"
#include "lib/matmul_intf.h"
#include "common.h"
#include "matmul_all_reduce_tiling_key.h"

namespace MatmulAllReduceImpl {}

using namespace AscendC;
using namespace Mc2Tiling;
using namespace MatmulAllReduceImpl;

template<int MM_TYPE, TPL_PARAMS_COMM, TPL_PARAMS_SHARE_MM, TPL_PARAMS_FP_MM>
__global__ __aicore__ void fp_matmul_all_reduce(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM, GM_ADDR antiquantScaleGM, GM_ADDR antiquantOffsetGM,
    GM_ADDR dequantGM, GM_ADDR pertokenGM, GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM, GM_ADDR cGM,
    GM_ADDR workspaceGM, GM_ADDR tilingGM, TPipe &tPipe, GM_ADDR userWS)
{

}

template<int MM_TYPE, TPL_PARAMS_COMM, TPL_PARAMS_SHARE_MM, TPL_PARAMS_QUANT_MM>
__global__ __aicore__ void quant_matmul_allreduce(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM, GM_ADDR antiquantScaleGM, GM_ADDR antiquantOffsetGM,
    GM_ADDR dequantGM, GM_ADDR pertokenGM, GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM, GM_ADDR cGM,
    GM_ADDR workspaceGM, GM_ADDR tilingGM, TPipe &tPipe, GM_ADDR userWS)
{
}

template<int MM_TYPE, TPL_PARAMS_COMM, TPL_PARAMS_SHARE_MM, TPL_PARAMS_WEIGHT_QUANT_MM>
__global__ __aicore__ void weight_quant_matmul_allreduce(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM, GM_ADDR antiquantScaleGM, GM_ADDR antiquantOffsetGM,
    GM_ADDR dequantGM, GM_ADDR pertokenGM, GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM, GM_ADDR cGM,
    GM_ADDR workspaceGM, GM_ADDR tilingGM, TPipe &tPipe, GM_ADDR userWS)
{
}

template<int AICORE_TYPE, int MM_TYPE, TPL_PARAMS_COMM, TPL_PARAMS_SHARE_MM, TPL_PARAMS_FP_MM, TPL_PARAMS_QUANT_MM, TPL_PARAMS_WEIGHT_QUANT_MM>
__global__ __aicore__ void matmul_all_reduce(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM, GM_ADDR antiquantScaleGM, GM_ADDR antiquantOffsetGM,
    GM_ADDR dequantGM, GM_ADDR pertokenGM, GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM, GM_ADDR cGM,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
}