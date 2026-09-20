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
 * \file moe_distribute_dispatch_wait_debug.h
 * \brief dispatchv2 等待超时诊断（GetSystemCycle 看门狗 + 远端写入前对端打印）
 *
 * 排障用途：
 * 1. 算子内各等待循环带超时看门狗：用 GetSystemCycle 计时，等待超过阈值
 *    （默认 10s，可编译期 -DDISPATCH_WAIT_TIMEOUT_SEC=n 覆盖）时，经 AscendC::Printf
 *    打印当前等待阶段与缺失的对端信息（哪些 rank 的状态/数据未到），随后继续原等待，
 *    不改变任何通信协议行为；同一等待点每超过一个超时周期重复打印一次。
 * 2. 发送侧在向对端 window 写入之前打印对端信息（dstRank/expert/token 槽位），
 *    每核（aiv）对每个 dstRank 仅打印一次，用于与接收侧超时信息互相对照。
 *
 * 使用：
 * - 默认编译即生效；正式性能测试编译时加 -DDISPATCH_WAIT_DEBUG_DISABLE=1 整体剔除。
 * - 输出走 device 日志（plog），grep 关键字 "DispatchV2"。
 */
#ifndef MOE_DISTRIBUTE_DISPATCH_WAIT_DEBUG_H
#define MOE_DISTRIBUTE_DISPATCH_WAIT_DEBUG_H

#include "moe_distribute_v2_constant.h" // CYCLES_PER_US

#ifndef DISPATCH_WAIT_TIMEOUT_SEC
#define DISPATCH_WAIT_TIMEOUT_SEC 10U // 等待超时阈值，单位：秒
#endif

#ifdef DISPATCH_WAIT_DEBUG_DISABLE
#define DISPATCH_WAIT_DEBUG_ON 0
#else
#define DISPATCH_WAIT_DEBUG_ON 1
#endif

#if DISPATCH_WAIT_DEBUG_ON
// 打印统一出口：若部署环境 AscendC::Printf 接口有差异，仅需调整此处
#define DISPATCH_WAIT_PRINT(...) AscendC::Printf(__VA_ARGS__)
// 看门狗节流：等待已超阈值，且距上次打印已超过一个超时周期
#define DISPATCH_WAIT_TIMEOUT_HIT(waitStartCycle, lastPrintCycle, nowCycle) \
    (((nowCycle) - (waitStartCycle) >= DISPATCH_WAIT_TIMEOUT_CYCLE) &&      \
     ((nowCycle) - (lastPrintCycle) >= DISPATCH_WAIT_TIMEOUT_CYCLE))
#else
#define DISPATCH_WAIT_PRINT(...) static_cast<void>(0)
#define DISPATCH_WAIT_TIMEOUT_HIT(waitStartCycle, lastPrintCycle, nowCycle) (false)
#endif

namespace Mc2Kernel {
// cycle 换算：1us = CYCLES_PER_US(50) cycle；10s = 10 * 10^6 * 50 = 5e8 cycle
constexpr uint64_t DISPATCH_WAIT_TIMEOUT_CYCLE =
    static_cast<uint64_t>(DISPATCH_WAIT_TIMEOUT_SEC) * 1000000ULL * CYCLES_PER_US;
} // namespace Mc2Kernel

#endif // MOE_DISTRIBUTE_DISPATCH_WAIT_DEBUG_H
