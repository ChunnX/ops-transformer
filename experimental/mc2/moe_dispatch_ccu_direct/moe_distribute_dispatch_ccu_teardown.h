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
 *     - 0.5RTT 等齐 + 清零（WaitAllPeersArriveNotify / ClearAllPeersArriveNotify）
 *       从 v9 teardown 上移到 v10 setup CCU 末尾（setup CCU 的段 9）。
 *     - setup CCU 完成时所有对端 0.5RTT 数据已就位；stream 顺序保证 AIV teardown
 *       跑时 y / commCmdInfo 内容稳定，无需再做等齐。
 *
 *   端到端 IO 契约（字段名 / dtype / shape）严格保留：
 *     Inputs:  x, y, expertIds, commCmdInfo
 *     Outputs: expandX, dynamicScales, assistInfoForCombine, expertTokenNums
 *
 *   简化范围（v10）：
 *     - device 端 teardown 只实际读 y + commCmdInfo，只实际写 expandX + assistInfoForCombine
 *     - x / expertIds / dynamicScales / expertTokenNums 端到端 IO 契约保留但 device
 *       端不使用——host driver sync 后对 dynamicScales / expertTokenNums 显式归零
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_CCU_TEARDOWN_H
#define MOE_DISTRIBUTE_DISPATCH_CCU_TEARDOWN_H

#include <cstdint>

// ----- v10 Tiling 结构（host 端填充、device 端读取）-----
//   通过 tilingGM 传给 ascendc kernel；host driver alloc + memcpy 一份到 device。
//   host 和 device 端都 include 本头，结构对齐必须一致（POD struct + 显式字段类型）。
struct MoeDistributeDispatchCcuTeardownTilingData {
    uint32_t epWorldSize;            // EP 通信域大小（对应 op_def epWorldSize）
    uint32_t epRankId;                // 本卡 EP rank id
    uint32_t moeExpertNum;            // MoE 总 expert 数
    uint32_t moeExpertPerRankNum;     // 本卡 expert 数 = op_def 的 localExpertNum
    uint32_t bs;                       // batch size
    uint32_t topK;                     // topK
    uint32_t hiddenSize;               // H
    uint32_t expertTokenNumsType;     // 0=前缀和 / 1=数量（v10 简化版 device 不写，仅占位）
    uint64_t tokenMsgLen;              // = Align256(H) * sizeof(YOutType)（非量化）
    uint64_t tokenWithMetaLen;         // v15 ★ = tokenMsgLen + 32（含三元组 metadata）
    uint64_t segBytes;                 // = bs * topK * tokenWithMetaLen，每段固定字节数（含 padding）
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

extern "C" __global__ __aicore__ void moe_distribute_dispatch_ccu_teardown(
    GM_ADDR x, GM_ADDR y, GM_ADDR expertIds, GM_ADDR commCmdInfo,
    GM_ADDR expandX, GM_ADDR dynamicScales,
    GM_ADDR assistInfoForCombine, GM_ADDR expertTokenNums,
    GM_ADDR commWindow,                       // v13 ★ device 端实际读的接收 window
    GM_ADDR workspace, GM_ADDR tilingGM);
#endif

#endif // MOE_DISTRIBUTE_DISPATCH_CCU_TEARDOWN_H
