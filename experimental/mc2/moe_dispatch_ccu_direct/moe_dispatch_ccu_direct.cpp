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
 * \file moe_dispatch_ccu_direct.cpp
 * \brief CCU dispatch setup / teardown 两个核函数的 <<<>>> 直调 device 实现。
 *
 * 不走算子注册框架 / aclnn —— 直接以 <<<blockDim, nullptr, stream>>> 拉起
 * moe_distribute_dispatch_ccu_setup / moe_distribute_dispatch_ccu_teardown。
 * 核函数体当前为空壳（与 op_kernel/ 下注册算子的空壳实现一致，
 * 实际 CCU 引擎实现待后续阶段），<<<>>> 直调不消费 tiling 注册，故省去
 * REGISTER_TILING_DEFAULT —— 与 quant_all_reduce 直调样例保持同一形态。
 */

#include "kernel_operator.h"

using namespace AscendC;

// ===== CCU dispatch setup 核函数入口（空壳，签名与注册算子完全一致）=====
extern "C" __global__ __aicore__ void moe_distribute_dispatch_ccu_setup(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask,
    GM_ADDR YOut, GM_ADDR expandIdxOut, GM_ADDR commCmdInfoOut,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    (void)x; (void)expertIds; (void)scales; (void)xActiveMask;
    (void)YOut; (void)expandIdxOut; (void)commCmdInfoOut;
    (void)workspaceGM; (void)tilingGM;
}

// ===== CCU dispatch teardown 核函数入口（空壳，签名与注册算子完全一致）=====
extern "C" __global__ __aicore__ void moe_distribute_dispatch_ccu_teardown(
    GM_ADDR x, GM_ADDR y, GM_ADDR expertIds, GM_ADDR commCmdInfo,
    GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR assistInfoForCombineOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    (void)x; (void)y; (void)expertIds; (void)commCmdInfo;
    (void)expandXOut; (void)dynamicScalesOut; (void)assistInfoForCombineOut;
    (void)expertTokenNumsOut; (void)workspaceGM; (void)tilingGM;
}

// ===== <<<>>> 直调封装：setup（host 侧调用入口）=====
void moe_distribute_dispatch_ccu_setup_do(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *expertIds, uint8_t *scales, uint8_t *xActiveMask,
    uint8_t *YOut, uint8_t *expandIdxOut, uint8_t *commCmdInfoOut,
    uint8_t *workspaceGM, uint8_t *tilingGM)
{
    moe_distribute_dispatch_ccu_setup<<<blockDim, nullptr, stream>>>(
        x, expertIds, scales, xActiveMask,
        YOut, expandIdxOut, commCmdInfoOut, workspaceGM, tilingGM);
}

// ===== <<<>>> 直调封装：teardown（host 侧调用入口）=====
void moe_distribute_dispatch_ccu_teardown_do(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *y, uint8_t *expertIds, uint8_t *commCmdInfo,
    uint8_t *expandXOut, uint8_t *dynamicScalesOut, uint8_t *assistInfoForCombineOut,
    uint8_t *expertTokenNumsOut, uint8_t *workspaceGM, uint8_t *tilingGM)
{
    moe_distribute_dispatch_ccu_teardown<<<blockDim, nullptr, stream>>>(
        x, y, expertIds, commCmdInfo,
        expandXOut, dynamicScalesOut, assistInfoForCombineOut,
        expertTokenNumsOut, workspaceGM, tilingGM);
}
