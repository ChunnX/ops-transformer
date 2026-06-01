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
 * \file norm_flash_attention.cpp
 * \brief 算子Kernel实现文件
 */

#include "norm_flash_attention.h"
#include "kernel_operator.h"

using namespace AscendC;
using namespace NormFlashAttentionNs;

extern "C" __global__ __aicore__ void norm_flash_attention(GM_ADDR query, GM_ADDR key, GM_ADDR value,
                                                           GM_ADDR attentionOut, GM_ADDR workspace, GM_ADDR tiling) {
    InitParams initParams;
    initParams.query = query;
    initParams.key = key;
    initParams.value = value;
    initParams.attentionOut = attentionOut;
    initParams.workspace = workspace;
    initParams.tilingData = reinterpret_cast<NormFlashAttentionTilingData *>(tiling);
    initParams.tPipeIn = &GetTilingPipe();

    // TODO: 根据数据类型创建Kernel实例
    // NormFlashAttentionKernel<half_t> kernel;
    // kernel.Init(initParams);
    // kernel.Process();
}
