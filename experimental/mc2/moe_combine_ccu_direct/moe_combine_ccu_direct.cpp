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
 * \file moe_combine_ccu_direct.cpp
 * \brief CCU combine setup / teardown 两个核函数的 <<<>>> 直调 device 实现。
 *
 * 不走算子注册框架 / aclnn —— 直接以 <<<blockDim, nullptr, stream>>> 拉起
 * moe_distribute_combine_ccu_setup / moe_distribute_combine_ccu_teardown。
 * 核函数体当前为空壳（实际 CCU 引擎实现待后续阶段），与 moe_dispatch_ccu_direct 同形态。
 * 签名分别与注册算子 moe_distribute_combine_setup / moe_distribute_combine_teardown 一致。
 */

#include "kernel_operator.h"

using namespace AscendC;

// ===== CCU combine setup 核函数入口（空壳，签名与 moe_distribute_combine_setup 一致）=====
extern "C" __global__ __aicore__ void moe_distribute_combine_ccu_setup(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine, GM_ADDR quantExpandX,
    GM_ADDR commCmdInfoOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    (void)expandX; (void)expertIds; (void)assistInfoForCombine; (void)quantExpandX;
    (void)commCmdInfoOut; (void)workspaceGM; (void)tilingGM;
}

// ===== CCU combine teardown 核函数入口（空壳，签名与 moe_distribute_combine_teardown 一致）=====
extern "C" __global__ __aicore__ void moe_distribute_combine_ccu_teardown(
    GM_ADDR expandX, GM_ADDR quantExpandX, GM_ADDR expertIds, GM_ADDR expandIdx,
    GM_ADDR expertScales, GM_ADDR commCmdInfo, GM_ADDR xActiveMask, GM_ADDR sharedExpertX,
    GM_ADDR XOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    (void)expandX; (void)quantExpandX; (void)expertIds; (void)expandIdx;
    (void)expertScales; (void)commCmdInfo; (void)xActiveMask; (void)sharedExpertX;
    (void)XOut; (void)workspaceGM; (void)tilingGM;
}

// ===== <<<>>> 直调封装：setup（host 侧调用入口）=====
void moe_distribute_combine_ccu_setup_do(
    uint32_t blockDim, void *stream,
    uint8_t *expandX, uint8_t *expertIds, uint8_t *assistInfoForCombine, uint8_t *quantExpandX,
    uint8_t *commCmdInfoOut, uint8_t *workspaceGM, uint8_t *tilingGM)
{
    moe_distribute_combine_ccu_setup<<<blockDim, nullptr, stream>>>(
        expandX, expertIds, assistInfoForCombine, quantExpandX,
        commCmdInfoOut, workspaceGM, tilingGM);
}

// ===== <<<>>> 直调封装：teardown（host 侧调用入口）=====
void moe_distribute_combine_ccu_teardown_do(
    uint32_t blockDim, void *stream,
    uint8_t *expandX, uint8_t *quantExpandX, uint8_t *expertIds, uint8_t *expandIdx,
    uint8_t *expertScales, uint8_t *commCmdInfo, uint8_t *xActiveMask, uint8_t *sharedExpertX,
    uint8_t *XOut, uint8_t *workspaceGM, uint8_t *tilingGM)
{
    moe_distribute_combine_ccu_teardown<<<blockDim, nullptr, stream>>>(
        expandX, quantExpandX, expertIds, expandIdx,
        expertScales, commCmdInfo, xActiveMask, sharedExpertX,
        XOut, workspaceGM, tilingGM);
}
