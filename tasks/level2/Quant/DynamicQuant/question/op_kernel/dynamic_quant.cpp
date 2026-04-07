/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
/*!
 * \file dynamic_quant.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "dynamic_quant.h"
#if __CCE_AICORE__ < 220
#include "dynamic_quant_align_310p.h"
#include "dynamic_quant_unalign_310p.h"
#include "dynamic_quant_unalign_large_310p.h"
#else
#include "dynamic_quant_single_row.h"
#endif
#include "dynamic_quant_db.h"
#include "dynamic_quant_large_shape_opt.h"
#include "dynamic_quant_moe.h"
#include "dynamic_quant_moe_large_shape.h"

#ifdef __CCE_KT_TEST__
#define KERNEL_LINKAGE static
#else
#define KERNEL_LINKAGE extern "C"
#endif

using namespace AscendC;
using namespace DynamicQuantNDOpt;

KERNEL_LINKAGE __global__ __aicore__ void dynamic_quant(
    GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR group_index, GM_ADDR y, GM_ADDR scale, GM_ADDR workSpace, GM_ADDR tiling)
{
}