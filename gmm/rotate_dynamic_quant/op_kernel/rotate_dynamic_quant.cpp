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
 * \file rotate_dynamic_quant.cpp
 * \brief 算子Kernel实现文件
 */

#include "rotate_dynamic_quant.h"
#include "kernel_operator.h"

using namespace AscendC;
using namespace RotateDynamicQuant;

// Kernel 入口函数
extern "C" __global__ __aicore__ void rotate_dynamic_quant(
    GM_ADDR x, GM_ADDR rot, GM_ADDR yOut, GM_ADDR scaleOut,
    GM_ADDR workspace, GM_ADDR tiling) {

    // 解析 tiling 数据
    RotateDynamicQuantTilingData *tilingData = reinterpret_cast<RotateDynamicQuantTilingData *>(tiling);

    // 构建 InitParams
    InitParams initParams;
    initParams.x = x;
    initParams.rot = rot;
    initParams.yOut = yOut;
    initParams.scaleOut = scaleOut;
    initParams.workspace = workspace;
    initParams.tilingData = tilingData;
    initParams.tPipeIn = &GetTilingPipe();

    // 根据 dstType 创建不同类型的 Kernel 实例
    int64_t dstType = tilingData->get_dstType();

    // 获取输入数据类型（通过 tilingKey 判断）
    // 这里需要根据实际的 dtype 分发到不同的模板实例

    // TODO: 根据 dtype 和 dstType 创建对应的 Kernel 实例
    // 示例：
    // if (dstType == 0) {
    //     // INT8 输出
    //     RotateDynamicQuantKernel<half_t, int8_t> kernel;
    //     kernel.Init(initParams);
    //     kernel.Process();
    // } else {
    //     // INT32 输出
    //     RotateDynamicQuantKernel<half_t, int32_t> kernel;
    //     kernel.Init(initParams);
    //     kernel.Process();
    // }
}
