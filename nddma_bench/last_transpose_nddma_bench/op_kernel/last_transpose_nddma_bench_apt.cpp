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
 * \file last_transpose_nddma_bench_apt.cpp
 * \brief LastTransposeNddmaBench registry-invoke kernel 入口（arch35 / ascend950）。
 *
 * 由 Kernel 直调入口 LastTransposeNDDMABenchKernel(x,y,attr) 迁移：
 *   - 原 attr GM 结构读取 -> GET_TILING_DATA_WITH_STRUCT 读 TilingData
 *   - 原 dtype 分支 -> TILING_KEY_IS（fp32=0 / fp16=1 / int8=2）
 *   - 计算逻辑（KernelLastTransposeNDDMA）逐字保留在 arch35/last_transpose_nddma_bench.h
 */

#include "arch35/last_transpose_nddma_bench.h"

using namespace LastTransposeNddmaBenchNs;

#define TILING_KEY_FP32 0
#define TILING_KEY_FP16 1
#define TILING_KEY_INT8 2

extern "C" __global__ __aicore__ void last_transpose_nddma_bench(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    InitSocState();  // 静态 Tensor 编程：kernel 入口必须调用
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);  // 纯向量 NDDMA，AIV-only：opc 跳过 cube 通道推断

    GET_TILING_DATA_WITH_STRUCT(LastTransposeNddmaBenchTilingData, tilingData, tiling);
    uint32_t dim0 = tilingData.dim0;
    uint32_t dim1 = tilingData.dim1;
    uint32_t dim2 = tilingData.dim2;
    uint32_t dim3 = tilingData.dim3;
    uint32_t isMte3 = tilingData.isMte3;

    if (TILING_KEY_IS(TILING_KEY_FP16)) {  // 1 -> half
        KernelLastTransposeNDDMA<half, UB_CAP_ELEMS_B16> op;
        op.Init(x, y, dim0, dim1, dim2, dim3, isMte3);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP32)) {  // 0 -> float
        KernelLastTransposeNDDMA<float, UB_CAP_ELEMS_B32> op;
        op.Init(x, y, dim0, dim1, dim2, dim3, isMte3);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_INT8)) {  // 2 -> int8
        KernelLastTransposeNDDMA<int8_t, UB_CAP_ELEMS_B8> op;
        op.Init(x, y, dim0, dim1, dim2, dim3, isMte3);
        op.Process();
    }

    PipeBarrier<PIPE_ALL>();  // 静态 Tensor 编程：kernel 结束前必须调用
}
