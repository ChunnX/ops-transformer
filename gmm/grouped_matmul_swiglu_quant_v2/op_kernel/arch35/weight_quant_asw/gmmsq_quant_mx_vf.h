/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gmmsq_quant_mx_vf.h
 * \brief SIMD-VF functions for GMM SwiGLU + MX Quantization.
 */

#pragma once

#define GMMSQ_VF_USE_MICRO_API
#include "../weight_quant_basic_block/gmmsq_quant_mx_vf.h"
#undef GMMSQ_VF_USE_MICRO_API

namespace WeightQuantGroupedMatmul::Arch35 {
using AscendC::BLOCK_CUBE;
using GMMSQWeightQuant::GmmSwigluVf;
using GMMSQWeightQuant::ComputeMaxExpVf;
using GMMSQWeightQuant::ComputeScaleVf;
using GMMSQWeightQuant::TransMxScaleLayoutVf;
using GMMSQWeightQuant::ComputeDataForQuantTargetFp8Vf;

} // namespace WeightQuantGroupedMatmul::Arch35
