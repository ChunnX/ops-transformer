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
 * \file moe_combine_ccu_api.h
 * \brief A6 CCU MoeDistributeCombineCcu 算子的 0.5RTT 资源 tag + 复用 dispatch 的占位基类。
 *
 * v11 架构（与 dispatch_ccu_direct 完全对称）：
 *   - combine_setup 走 CCU 路径（hcomm CcuKernel），反向发送 expand_x 到对端
 *   - combine_teardown 走 AIV ascendc 路径（直调），内部轮询等齐 + 加权求和
 *
 * Mc2CcuKernelBase 占位基类 + 4 个 0.5RTT API 声明（RecordExpectCount /
 *   WriteNbWithArriveNotify / WaitAllPeersArriveNotify / ClearAllPeersArriveNotify）
 *   直接复用 moe_dispatch_ccu_api.h——dispatch / combine 共享同一基类，只是 tag 不同。
 *
 * CCU 0.5RTT 资源组：CCU 最多 4 组 wishCntXn/totalCntXn 资源；本工程占用 2 组：
 *   tag = MOE_DISPATCH_CCU_TAG（"moe_dispatch_05rtt"）→ dispatch 路径用
 *   tag = MOE_COMBINE_CCU_TAG （"moe_combine_05rtt" ）→ combine  路径用
 *   两 tag 独立累加，dispatch/combine 流水互不干扰。
 */

#ifndef MOE_COMBINE_CCU_API_H
#define MOE_COMBINE_CCU_API_H

#include "moe_dispatch_ccu_api.h"   // 复用 Mc2CcuKernelBase + CHK_RET 宏

// 本算子使用的 0.5RTT 资源组 tag（与 dispatch 不同，独立累加）
#define MOE_COMBINE_CCU_TAG  "moe_combine_05rtt"

#endif // MOE_COMBINE_CCU_API_H
