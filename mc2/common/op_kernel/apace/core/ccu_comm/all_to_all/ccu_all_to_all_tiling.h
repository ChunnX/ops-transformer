/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ccu_all_to_all_tiling.h
 * \brief CCU AllToAll 通信 tiling——通信参数契约（kernel 侧代码）
 *
 * 本结构是通信参数的 host 侧唯一事实源：形状/轮次由消费方 host 的 Validate 计算，
 * 资源字段由 Prepare 填入；每次调用经 RefreshCcuCommStage 把 params 整体抄入
 * kernelArg->stage（随 CCU kernel launch 按值直达，AIC 不经手）；matmul 部分仍
 * 随 AIC kernel 的 <<<>>> 按值入参。
 *
 * 事件编号契约（AIC↔CCU 同步，消费方按此对接）：
 *   ev[0] = wake（AIC commit → CCU process，循环前唯一一次）
 *   ev[t+1] = data 轮 t done（CCU notify → AIC wait，轮 0 含其搭载的 scale 整块；
 *   等待节奏由消费方 kernel 自定——如经 CommPolicy 逐 tile 等待）
 */

#pragma once

#include "../ccu_comm.h"

// CCU AllToAll 通信侧 tiling（通信标量散参全部收纳于此；同 shape 下除 params 内
// 四个地址外恒定）
#pragma pack(push, 8)
struct alignas(8) CcuCommTilingData {
    // ---- 形状/轮次（消费方 host 填；host/AIC 消费，CCU 走 kernelArg）----
    uint32_t rankId{0};
    uint32_t rankSize{0};
    uint32_t dataCommTurn{0};         // data 轮数（CCU 总轮数 = dataCommTurn，scale 整块随轮 0 搭载）
    uint32_t tileM{0};                // 每 data 轮传输行数
    // ---- 通信参数（host 填；RefreshCcuCommStage 抄入 kernelArg->stage 直达 CCU）----
    CcuCommStageParams params;
    uint64_t workspaceScaleOffset{0}; // workspace 内 recvScale 区偏移（host 侧派生地址用）
    // ---- CCU 资源（host 填；AIC 经 launch 传参消费）----
    CcuMappedAddrs mappedAddrs{};     // CKE 事件槽 MMIO 地址映射（AIC 清扫/commit/轮询用）
    // 轮参数闭式派生（CCU 侧 LoadRoundArgs）：data 轮 t 的 srcOffset = t*subBytes、
    // dstOffset = rankId*blockSize + t*subBytes、sliceSize = subBytes；scale 整块仅随
    // 轮 0 搭载：srcOffset = 0、dstOffset = scaleDstOffset（已含 region 偏移
    // recvScale = recvBuf + inputSize + rankId 块内偏移，host 烘焙）、
    // sliceSize = scaleSliceSize。契约强制均分轮次（Validate 整除校验）——若未来
    // 消费方需非均匀轮次，须在 stage 参数块中恢复 per-round 表（闭式与表二选一）
};
#pragma pack(pop)
