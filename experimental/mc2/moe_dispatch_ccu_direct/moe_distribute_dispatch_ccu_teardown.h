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
 * \file moe_distribute_dispatch_ccu_teardown.h
 * \brief A6 MoeDistributeDispatchCcuTeardown ascendc kernel 接口声明（v10 架构纠正版）。
 *
 * v10 重大架构纠正：
 *   之前 v1-v9 把 teardown 实现为 hcomm CcuKernel（跑在 CCU 引擎）——这是错的。
 *   teardown 的工作（读 commCmdInfo + DataCopy y→expand_x + 写 assist_info）是
 *   **纯 vector 计算**，应该跑在 **AIV (AI Vector Core)** 上，用标准 ascendc kernel
 *   语法实现，通过 `<<<blockDim, stream>>>` 直接 launch（即"AIV 直驱"）。
 *
 *   配套调整：
 *     - 0.5RTT 等齐 + 清零（WaitallpeersarriveNotify / clearallpeersarriveNotify）
 *       从 v9 teardown 上移到 v10 setup CCU 末尾（setup CCU 的段 9）。
 *     - setup CCU 完成时所有对端 0.5RTT 数据已就位；stream 顺序保证 AIV teardown
 *       跑时 y / commCmdInfo 内容稳定，无需再做等齐。
 *
 *   端到端 IO 契约（字段名 / dtype / shape）严格保留：
 *     Inputs:  x, y, expertIds, commCmdInfo
 *     Outputs: expandX, dynamicScales, assistInfoForCombine, expertTokenNums
 *
 *   简化范围（v10，精简签名后）：
 *     - device 端 teardown 只实际读 commCmdInfo + commWindow，只实际写 expandX + assistInfoForCombine
 *     - 精简后 kernel 入参已不含 x / y / expertIds / dynamicScales / expertTokenNums（上方 op_def 契约
 *       仍保留以记录库上接口）；生产接库上 op 时由 host 侧对 dynamicScales / expertTokenNums 归零，
 *       本 demo driver 不产出这两路（既不 alloc 也不归零）
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_CCU_TEARDOWN_H
#define MOE_DISTRIBUTE_DISPATCH_CCU_TEARDOWN_H

#include <cstdint>

// ----- v10 Tiling 结构（host 端填充、device 端读取）-----
//   通过 tilingGM 传给 ascendc kernel；host driver alloc + memcpy 一份到 device。
//   host 和 device 端都 include 本头，结构对齐必须一致（POD struct + 显式字段类型）。
// 彻底精简版：只保留 Process 实际读取的 tiling 字段。
//   已删 device 不读的字段：epRankId / moeExpertNum / moeExpertPerRankNum /
//     expertTokenNumsType / hiddenSize（Process 未用）/ segBytes（Process 用 segElemsCapacity 自算）。
struct MoeDistributeDispatchCcuTeardownTilingData {
    uint32_t epWorldSize;            // EP 通信域大小（WaitForAllSrcRanks + 段循环）
    uint32_t bs;                      // batch size
    uint32_t topK;                    // topK
    uint64_t tokenMsgLen;            // = Align256(H)*2 [BF16] / Align32(H+H/32) [MXFP8]（干净 data 拷贝 stride = tokenMsgLen/sizeof(YType)）
    uint64_t tokenWithMetaLen;       // = tokenMsgLen + 32（commWindow 每槽 data + metadata）
};

// ----- v13 Kernel 函数前向声明 -----
//   host driver 通过 `<<<blockDim, l2_ctrl, stream>>>(...)` 直调时需见此声明。
//   注意：本声明依赖 ascendc 工具链的 GM_ADDR / __global__ / __aicore__ 关键字——
//   纯 host gcc 编译环境（如 A5 demo）下不能直接 include；本工程目标为 A6 ascendc 仿真/真机。
//
// v13 ★ 加 GM_ADDR commWindow（位于 expertTokenNums 与 workspace 之间）——op_def yAddr
//   字段语义恢复库上（src 端发送缓冲，device 端不读不写），对端接收 window 由独立的
//   commWindow 参数承载（工程内部 scratch，库上 op_def 无此字段）。
#ifdef __ASCENDC_KERNEL_HEADER_AVAILABLE__   // ascendc 编译路径下定义此宏；host gcc 不定义
#include "kernel_operator.h"

// 彻底精简版签名：只保留 device 端实际读写的 GM。
//   IN : commCmdInfo（token 数 + sentinel）/ commWindow（接收数据）
//   OUT: expandX（紧密区，干净 data）/ assistInfoForCombine（单字段 dstByteOff int64）/ commCmdInfo（sendBytes+negCnt）
extern "C" __global__ __aicore__ void moe_distribute_dispatch_ccu_teardown(
    GM_ADDR commCmdInfo, GM_ADDR commWindow, GM_ADDR expandX,
    GM_ADDR assistInfoForCombine, GM_ADDR tilingGM);
#endif

#endif // MOE_DISTRIBUTE_DISPATCH_CCU_TEARDOWN_H
