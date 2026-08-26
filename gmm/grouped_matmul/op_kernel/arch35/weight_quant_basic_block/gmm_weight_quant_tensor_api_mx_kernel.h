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
 * \file gmm_weight_quant_tensor_api_mx_kernel.h
 * \brief Selects the MX A8W4 cube implementation supported by the toolkit.
 */
#ifndef GROUPED_MATMUL_WEIGHT_QUANT_TENSOR_API_MX_KERNEL_H
#define GROUPED_MATMUL_WEIGHT_QUANT_TENSOR_API_MX_KERNEL_H

#if defined(ASC_DEVKIT_MAJOR) && defined(ASC_DEVKIT_MINOR) &&                                                  \
    ((ASC_DEVKIT_MAJOR > 9) || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0))
#define GMM_WEIGHT_QUANT_USE_TENSOR_API_MX 1
#include "blaze/gemm/kernel/kernel_wqgmm_mix_weight_prologue.h"
#else
#define GMM_WEIGHT_QUANT_USE_TENSOR_API_MX 0
#include "basic_api/weight_quant_basic_api_v1.h"
#endif

#endif // GROUPED_MATMUL_WEIGHT_QUANT_TENSOR_API_MX_KERNEL_H
