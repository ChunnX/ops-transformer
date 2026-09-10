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
 * \file kernel_launcher.h
 * \brief AllToAllQuantMatmul UDMA kernel launcher 入口（从 udma_impl.h 抽取）
 */

#pragma once

#include "apace/kernel/fusions/all_to_all_quant_matmul/all_to_all_mx_quant_matmul_urma_impl.h"
#include "apace/kernel/fusions/all_to_all_quant_matmul/all_to_all_mx_quant_matmul_hcomm_impl.h"

__global__ __aicore__ void AllToAllQuantMatmulKernelE4M3E4M3_Udma(__gm__ CommContext *hcommCtx, GM_ADDR aGM,
                                                                  GM_ADDR scaleAGM, GM_ADDR bGM, GM_ADDR scaleBGM,
                                                                  GM_ADDR cGM, allToAllMatmulTilingData tilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    Apace::AllToAllMxQuantMatmulUrmaImpl<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t, false, true> impl;
    impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, cGM, nullptr, &tilingData);
    impl.Run();
}

__global__ __aicore__ void AllToAllQuantMatmulKernelE5M2E5M2_Udma(__gm__ CommContext *hcommCtx, GM_ADDR aGM,
                                                                  GM_ADDR scaleAGM, GM_ADDR bGM, GM_ADDR scaleBGM,
                                                                  GM_ADDR cGM, allToAllMatmulTilingData tilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    Apace::AllToAllMxQuantMatmulUrmaImpl<fp8_e5m2_t, fp8_e5m2_t, bfloat16_t, false, true> impl;
    impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, cGM, nullptr, &tilingData);
    impl.Run();
}

__global__ __aicore__ void AllToAllQuantMatmulKernelE4M3E5M2_Udma(__gm__ CommContext *hcommCtx, GM_ADDR aGM,
                                                                  GM_ADDR scaleAGM, GM_ADDR bGM, GM_ADDR scaleBGM,
                                                                  GM_ADDR cGM, allToAllMatmulTilingData tilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    Apace::AllToAllMxQuantMatmulUrmaImpl<fp8_e4m3fn_t, fp8_e5m2_t, bfloat16_t, false, true> impl;
    impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, cGM, nullptr, &tilingData);
    impl.Run();
}

__global__ __aicore__ void AllToAllQuantMatmulKernelE5M2E4M3_Udma(__gm__ CommContext *hcommCtx, GM_ADDR aGM,
                                                                  GM_ADDR scaleAGM, GM_ADDR bGM, GM_ADDR scaleBGM,
                                                                  GM_ADDR cGM, allToAllMatmulTilingData tilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    Apace::AllToAllMxQuantMatmulUrmaImpl<fp8_e5m2_t, fp8_e4m3fn_t, bfloat16_t, false, true> impl;
    impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, cGM, nullptr, &tilingData);
    impl.Run();
}

// HiF4 全量化 Hcomm 通路 ST 入口(transB 运行时入参, CCU 纯C通信):
// 与算子主线入口 allto_all_matmul_apt.cpp 使用相同的 AllToAllMxQuantMatmulHcommImpl。
// 依赖: blaze 侧 BlockMmad HiF4 分发与 hif4_scale 类型(devkit 提供)。
// [NOTE] Hcomm 通路 HCCL 上下文由 GetHcclContext<0> 获取, 独立 ST 需由测试框架
// 按主线方式注册通信上下文, 否则仅可经 aclnn 主线触发。
__global__ __aicore__ void AllToAllQuantMatmulKernelHiF4HiF4_Hcomm(GM_ADDR aGM, GM_ADDR scaleAGM, GM_ADDR bGM,
                                                                   GM_ADDR scaleBGM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                                                   bool transB,
                                                                   Apace::hcommAllToAllMatmulTilingData tilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    // IsMxFp4=false; HiF4 的 2:1 打包由 impl 内 isHiF4Scale_ 分支处理
    if (transB) {
        Apace::AllToAllMxQuantMatmulHcommImpl<hifloat4x2_t, hifloat4x2_t, bfloat16_t, hifloat4x2_t, hif4_scale,
                                              Apace::hcommAllToAllMatmulTilingData,
                                              AscendC::HcclServerType::HCCL_SERVER_TYPE_CCU, false, true>
            impl(&tilingData);
        impl.Init(aGM, bGM, nullptr, cGM, nullptr, scaleAGM, scaleBGM, workspaceGM);
        impl.Run();
    } else {
        Apace::AllToAllMxQuantMatmulHcommImpl<hifloat4x2_t, hifloat4x2_t, bfloat16_t, hifloat4x2_t, hif4_scale,
                                              Apace::hcommAllToAllMatmulTilingData,
                                              AscendC::HcclServerType::HCCL_SERVER_TYPE_CCU, false, false>
            impl(&tilingData);
        impl.Init(aGM, bGM, nullptr, cGM, nullptr, scaleAGM, scaleBGM, workspaceGM);
        impl.Run();
    }
}
