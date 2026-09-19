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
 * \file quant_flash_attn_block_vector_mxfp8_softmax_fp16.h
 * \brief MxFP8 Softmax FP16 Vector Block 最小桩（阶段二占位）
 */

#ifndef QUANT_FLASH_ATTN_BLOCK_VECTOR_MXFP8_SOFTMAX_FP16_H_
#define QUANT_FLASH_ATTN_BLOCK_VECTOR_MXFP8_SOFTMAX_FP16_H_

// ===== vf 头文件（编译冒烟：让编译器解析已实现的 vf 代码，提前暴露问题）=====
#include "vf/vf_common_def.h"
#include "vf/vf_calc_rescale_factor.h"
#include "vf/vf_calc_pscale.h"
#include "vf/vf_flashupdate_acc.h"
#include "vf/vf_div_cast_bf16.h"

namespace QFA_KERNEL {

template <typename QUANT_T, typename SCALE_T, typename OUT_T>
class QuantFlashAttnBlockVectorMxfp8SoftmaxFp16 {};

} // namespace QFA_KERNEL

#endif // QUANT_FLASH_ATTN_BLOCK_VECTOR_MXFP8_SOFTMAX_FP16_H_
