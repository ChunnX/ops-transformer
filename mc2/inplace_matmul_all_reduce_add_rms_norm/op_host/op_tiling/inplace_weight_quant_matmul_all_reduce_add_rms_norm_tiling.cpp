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
 * \file inplace_weight_quant_matmul_all_reduce_add_rms_norm_tiling.cc
 * \brief
 */
#ifndef _INPLACE_WEIGHT_QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_CC_
#define _INPLACE_WEIGHT_QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_CC_

#include "inplace_weight_quant_matmul_all_reduce_add_rms_norm_tiling.h"
namespace optiling {
namespace {
constexpr char MRN[] = "MatmulAllReduceAddRmsNorm";
constexpr char IMRN[] = "InplaceMatmulAllReduceAddRmsNorm";
} // namespace
using InplaceWeightQuantMatmulAllReduceAddRmsNormTiling = WeightQuantMatmulAllReduceAddRmsNormTiling;
REGISTER_TILING_TEMPLATE(IMRN, InplaceWeightQuantMatmulAllReduceAddRmsNormTiling, 1);
} // namespace optiling
#endif // _INPLACE_WEIGHT_QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_CC_