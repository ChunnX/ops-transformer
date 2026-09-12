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
 * \file quant_flash_attn_kernel_mxfp8.h
 * \brief QuantFlashAttn MxFP8 Softmax FP16 kernel 空壳（后续填 C1/V1/C2/V2 核间流水）
 */

#ifndef QUANT_FLASH_ATTN_KERNEL_MXFP8_H_
#define QUANT_FLASH_ATTN_KERNEL_MXFP8_H_

#include "kernel_operator.h"
#include "quant_flash_attn_tiling_data.h"

using namespace AscendC;
using namespace optiling;

namespace BaseApi {

class QuantFlashAttnKernelMxfp8 {
public:
    __aicore__ inline void Init(const __gm__ QuantFlashAttnTilingData *tiling)
    {
        (void)tiling;
    }

    __aicore__ inline void Process()
    {
        // 调试：确认 kernel 是否被调度（打印每个 block 的索引）
        AscendC::printf("[QFA_MXFP8_SOFTMAX_FP16] kernel Process called, blockIdx=%d\n",
                        static_cast<int32_t>(GetBlockIdx()));
    }
};

} // namespace BaseApi

#endif // QUANT_FLASH_ATTN_KERNEL_MXFP8_H_